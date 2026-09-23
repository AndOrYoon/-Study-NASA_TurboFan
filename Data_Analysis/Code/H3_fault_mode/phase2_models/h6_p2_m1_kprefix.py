# -*- coding: utf-8 -*-
"""
H6 Phase 2 -- Fair M1 Baseline (M1_kprefix)

Reviewer concern (GPT Major Comment #6): original M1 trains GMM on full-trajectory
features (late_mean, slope) but routes test engines via the last 30-cycle window --
a train/test feature-construction mismatch that may partly cause FD004 branch collapse.

This script implements a fair baseline where BOTH training cluster assignment AND
test routing use the same K-cycle prefix (first K=10 cycles), matching the
information horizon of M3's GatingNet.

If M1_kprefix still collapses on FD004, the collapse is attributable to the dataset's
structural properties (near-uniform fault mode distribution in early cycles) rather
than the feature mismatch, strengthening the M3 deployment argument.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from h6_p2_model_utils import (
    set_seed, SEEDS, RESULTS_DIR, MODELS_DIR,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    apply_op_residual_fd004, fit_op_residual_fd004,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    make_loader, LSTMBranch,
    train_epoch, eval_epoch, predict_sequences,
    compute_metrics, save_predictions,
)

DATASETS   = ["FD003", "FD004"]
K_PREFIX   = 10            # same information horizon as M3
MAX_EPOCHS = 100
PATIENCE   = 15
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4

FAULT_SENSORS = ["s15", "s20", "s21", "s7", "s12", "s2", "s4"]


# ---------------------------------------------------------------------------
# K-prefix GMM helpers
# ---------------------------------------------------------------------------

def extract_prefix_features(df: pd.DataFrame, sensor_cols: list, K: int) -> pd.DataFrame:
    """Per-engine mean of fault-discriminant sensors over the first K cycles."""
    active = [s for s in FAULT_SENSORS if s in sensor_cols]
    records = []
    for unit, grp in df.groupby("unit_number"):
        prefix = grp.head(K)[active]
        row = {"unit_number": unit}
        for s in active:
            row[f"{s}_pmean"] = float(prefix[s].mean())
        records.append(row)
    return pd.DataFrame(records).set_index("unit_number")


def fit_kprefix_gmm(feat_df: pd.DataFrame):
    """Fit GMM(k=2) on K-prefix features. Returns (scaler, gmm, feature_cols)."""
    feature_cols = list(feat_df.columns)
    X_sc = StandardScaler().fit_transform(feat_df[feature_cols].values)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(feat_df[feature_cols].values)
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          random_state=42, n_init=10)
    gmm.fit(X_sc)
    return scaler, gmm, feature_cols


def assign_clusters(feat_df: pd.DataFrame, scaler, gmm, feature_cols) -> dict:
    """Return {unit_number: cluster_id} from K-prefix features."""
    X_sc = scaler.transform(feat_df[feature_cols].values)
    labels = gmm.predict(X_sc)
    return dict(zip(feat_df.index, labels.tolist()))


# ---------------------------------------------------------------------------
# Branch training (identical to original M1)
# ---------------------------------------------------------------------------

def train_one_branch(X_tr, y_tr, X_va, y_va, n_feat, device, seed):
    set_seed(seed)
    model     = LSTMBranch(n_feat).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()
    tr_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X_va, y_va, BATCH_SIZE, shuffle=False)

    best_val, pc, best_sd = float("inf"), 0, None
    for _ in range(1, MAX_EPOCHS + 1):
        train_epoch(model, tr_loader, optimizer, criterion, device)
        va_l = eval_epoch(model, va_loader, criterion, device)
        if va_l < best_val:
            best_val, pc = va_l, 0
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pc += 1
            if pc >= PATIENCE:
                break
    return best_sd, best_val


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(dataset: str, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load & op-residualize training data --------------------------------
    train_df    = load_cmapss(dataset, "train")
    sensor_cols = get_sensor_cols(dataset)

    sc_op = km_op = cm_op = None
    if dataset == "FD004":
        sc_op, km_op, cm_op = fit_op_residual_fd004(train_df, sensor_cols)
        train_df = apply_op_residual_fd004(train_df, sc_op, km_op, cm_op, sensor_cols)

    # ---- Fit K-prefix GMM on training engines (before normalization) --------
    feat_train = extract_prefix_features(train_df, sensor_cols, K_PREFIX)
    scaler_gmm, gmm, feature_cols = fit_kprefix_gmm(feat_train)
    cluster_map = assign_clusters(feat_train, scaler_gmm, gmm, feature_cols)

    c0_units = {u for u, c in cluster_map.items() if c == 0}
    c1_units = {u for u, c in cluster_map.items() if c == 1}
    print(f"  Train cluster sizes: C0={len(c0_units)}  C1={len(c1_units)} engines")

    # ---- Build training sequences -------------------------------------------
    train_df  = add_rul(train_df)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n   = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)

    m0 = np.array([u in c0_units for u in units])
    m1 = np.array([u in c1_units for u in units])
    X0, y0 = X[m0], y[m0]
    X1, y1 = X[m1], y[m1]

    tr0, va0 = split_engines(units[m0], 0.2, seed=42)
    tr1, va1 = split_engines(units[m1], 0.2, seed=42)
    n_feat = len(sensor_cols)

    # ---- Train two branches -------------------------------------------------
    sd0, bv0 = train_one_branch(X0[tr0], y0[tr0], X0[va0], y0[va0], n_feat, device, seed)
    sd1, bv1 = train_one_branch(X1[tr1], y1[tr1], X1[va1], y1[va1], n_feat, device, seed + 100)

    branch0 = LSTMBranch(n_feat).to(device); branch0.load_state_dict(sd0)
    branch1 = LSTMBranch(n_feat).to(device); branch1.load_state_dict(sd1)

    # ---- Test: K-prefix routing (first K cycles of each test engine) --------
    test_raw = load_cmapss(dataset, "test")
    if dataset == "FD004" and sc_op is not None:
        test_resid = apply_op_residual_fd004(test_raw, sc_op, km_op, cm_op, sensor_cols)
    else:
        test_resid = test_raw.copy()

    # Extract K-prefix features from test engines using the same GMM
    feat_test   = extract_prefix_features(test_resid, sensor_cols, K_PREFIX)
    X_te_sc     = scaler_gmm.transform(feat_test[feature_cols].values)
    cluster_ids = gmm.predict(X_te_sc)          # shape (n_test_engines,)

    dist = np.bincount(cluster_ids, minlength=2)
    print(f"  Test cluster dist: C0={dist[0]}  C1={dist[1]}")

    # LSTM predictions: last 30-cycle window (standard)
    test_n      = apply_normalization(test_resid, sensor_cols, min_v, max_v)
    X_te, y_te  = make_test_sequences(test_n, sensor_cols, load_test_rul(dataset))

    y0_all = predict_sequences(branch0, X_te, device)
    y1_all = predict_sequences(branch1, X_te, device)

    preds = np.where(cluster_ids == 0, y0_all, y1_all)
    rmse, ns = compute_metrics(preds, y_te)
    save_predictions(preds, y_te, "M1_kprefix", dataset, seed)
    print(f"  -> RMSE={rmse:.4f}  NASA={ns:.2f}")
    return rmse, ns


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[M1_kprefix] K={K_PREFIX} prefix fair baseline")
    summary = {}
    for dataset in DATASETS:
        print(f"\n{'='*55}\nDataset: {dataset}\n{'='*55}")
        rmse_l, ns_l = [], []
        for seed in SEEDS:
            print(f"\n  seed={seed}")
            r, n = run(dataset, seed)
            rmse_l.append(r)
            ns_l.append(n)
        mean_r = float(np.mean(rmse_l))
        std_r  = float(np.std(rmse_l))
        mean_n = float(np.mean(ns_l))
        std_n  = float(np.std(ns_l))
        summary[dataset] = dict(rmse_mean=mean_r, rmse_std=std_r,
                                 nasa_mean=mean_n, nasa_std=std_n)
        print(f"\n[M1_kprefix {dataset}] "
              f"RMSE={mean_r:.4f}+/-{std_r:.4f}  "
              f"NASA={mean_n:.2f}+/-{std_n:.2f}")

    print("\n\n=== SUMMARY ===")
    for ds, v in summary.items():
        print(f"{ds}: RMSE={v['rmse_mean']:.4f}+/-{v['rmse_std']:.4f}  "
              f"NASA={v['nasa_mean']:.2f}+/-{v['nasa_std']:.2f}")
    print("[DONE]")
