# -*- coding: utf-8 -*-
"""
H6 Phase 2 ??M1: Hard-Routing Multi-Branch LSTM
GMM argmax cluster -> dedicated LSTM branch per cluster.
Branch 0 trained on cluster-0 engines, branch 1 on cluster-1 engines.
At test time: GMM assigns test engine to a branch.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim

from h6_p2_model_utils import (
    set_seed, SEEDS, RESULTS_DIR, MODELS_DIR,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    apply_op_residual_fd004, fit_op_residual_fd004,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    make_loader, LSTMBranch,
    train_epoch, eval_epoch, predict_sequences,
    compute_metrics, save_predictions, save_checkpoint,
)
from h6_p2_inference import get_test_cluster_assignments

DATASETS   = ["FD003", "FD004"]
MAX_EPOCHS = 100
PATIENCE   = 15
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4


def train_one_branch(X_tr, y_tr, X_va, y_va, n_feat, device, seed):
    """Train one LSTMBranch with early stopping. Returns best state_dict."""
    set_seed(seed)
    model     = LSTMBranch(n_feat).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()
    tr_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X_va, y_va, BATCH_SIZE, shuffle=False)

    best_val, pc, best_sd = float("inf"), 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_l = train_epoch(model, tr_loader, optimizer, criterion, device)
        va_l = eval_epoch(model, va_loader, criterion, device)
        if va_l < best_val:
            best_val = va_l
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= PATIENCE:
                break
    return best_sd, best_val


def run(dataset: str, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Preprocessing ----
    train_df    = load_cmapss(dataset, "train")
    sensor_cols = get_sensor_cols(dataset)

    sc_op = km_op = cm_op = None
    if dataset == "FD004":
        sc_op, km_op, cm_op = fit_op_residual_fd004(train_df, sensor_cols)
        train_df = apply_op_residual_fd004(train_df, sc_op, km_op, cm_op, sensor_cols)

    train_df = add_rul(train_df)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n  = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)

    # ---- Cluster assignments ----
    tag  = dataset.lower()
    ca   = pd.read_csv(os.path.join(RESULTS_DIR, f"cluster_assignments_{tag}.csv"), index_col=0)
    c0_u = set(ca[ca["cluster"] == 0].index.tolist())
    c1_u = set(ca[ca["cluster"] == 1].index.tolist())

    # Filter sequences by cluster
    m0 = np.array([u in c0_u for u in units])
    m1 = np.array([u in c1_u for u in units])

    # Per-cluster train/val split (seed=42 fixed, not training seed)
    tr0, va0 = split_engines(units[m0], 0.2, seed=42)
    tr1, va1 = split_engines(units[m1], 0.2, seed=42)

    X0, y0 = X[m0], y[m0]
    X1, y1 = X[m1], y[m1]

    n_feat = len(sensor_cols)
    print(f"  Cluster 0: {m0.sum()} seqs  Cluster 1: {m1.sum()} seqs")

    # ---- Train branch 0 ----
    print(f"  Training branch-0 (cluster 0) ...")
    set_seed(seed)
    sd0, bv0 = train_one_branch(X0[tr0], y0[tr0], X0[va0], y0[va0], n_feat, device, seed)
    print(f"  Branch-0 best val: {bv0:.4f}")

    # ---- Train branch 1 ----
    print(f"  Training branch-1 (cluster 1) ...")
    set_seed(seed + 100)
    sd1, bv1 = train_one_branch(X1[tr1], y1[tr1], X1[va1], y1[va1], n_feat, device, seed + 100)
    print(f"  Branch-1 best val: {bv1:.4f}")

    # Save checkpoints
    base_ckpt = {"n_features": n_feat, "sensor_cols": sensor_cols,
                 "sensor_min": min_v.to_dict(), "sensor_max": max_v.to_dict()}
    save_checkpoint({**base_ckpt, "model_state_dict": sd0}, "M1_branch0", dataset, seed)
    save_checkpoint({**base_ckpt, "model_state_dict": sd1}, "M1_branch1", dataset, seed)

    # ---- Test inference ----
    branch0 = LSTMBranch(n_feat).to(device); branch0.load_state_dict(sd0)
    branch1 = LSTMBranch(n_feat).to(device); branch1.load_state_dict(sd1)

    test_raw = load_cmapss(dataset, "test")
    if dataset == "FD004" and sc_op is not None:
        test_df_resid = apply_op_residual_fd004(test_raw, sc_op, km_op, cm_op, sensor_cols)
    else:
        test_df_resid = test_raw.copy()

    test_n = apply_normalization(test_df_resid, sensor_cols, min_v, max_v)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, load_test_rul(dataset))

    # Get predictions from both branches
    y0_all = predict_sequences(branch0, X_te, device)
    y1_all = predict_sequences(branch1, X_te, device)

    # Hard cluster assignment for each test engine (uses raw test_df for FD003, residualized for FD004)
    assign_df = test_raw if dataset == "FD003" else test_df_resid
    cluster_ids = get_test_cluster_assignments(assign_df, dataset, hard=True)

    preds = np.where(cluster_ids == 0, y0_all, y1_all)
    rmse, ns = compute_metrics(preds, y_te)
    save_predictions(preds, y_te, "M1", dataset, seed)
    print(f"  -> RMSE={rmse:.4f}  NASA={ns:.2f}  cluster_dist={np.bincount(cluster_ids)}")
    return rmse, ns


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M1] Device: {device}")
    for dataset in DATASETS:
        print(f"\n{'='*50}\nDataset: {dataset}\n{'='*50}")
        rmse_l, ns_l = [], []
        for seed in SEEDS:
            print(f"\n  seed={seed}")
            r, n = run(dataset, seed)
            rmse_l.append(r); ns_l.append(n)
        print(f"\n[M1 {dataset}] RMSE={np.mean(rmse_l):.4f}+/-{np.std(rmse_l):.4f}  "
              f"NASA={np.mean(ns_l):.2f}+/-{np.std(ns_l):.2f}")
    print("\n[DONE] M1 complete.")
