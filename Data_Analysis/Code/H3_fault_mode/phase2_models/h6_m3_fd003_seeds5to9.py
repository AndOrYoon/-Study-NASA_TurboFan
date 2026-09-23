# -*- coding: utf-8 -*-
"""
H6 M3 FD003 — Extended seeds (5–9) runner + 10-seed summary.

Runs the identical M3 Attention-Gate pipeline used in h6_p2_attention_gate.py
but for seeds {5, 6, 7, 8, 9} on FD003 only.
After completion it reads existing seed 0-4 predictions, combines all 10 seeds,
and writes:
  - raw_predictions_M3_FD003_seed{5..9}.csv  (per-seed, same format as existing)
  - H6_M3_FD003_10seeds_summary.csv           (combined stats)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from h6_p2_model_utils import (
    set_seed, WINDOW_SIZE, RESULTS_DIR,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    SeqDatasetM3, AttentionGateModel,
    train_m3_epoch, eval_m3_epoch, predict_m3,
    compute_metrics, save_predictions, save_checkpoint,
)

DATASET    = "FD003"
NEW_SEEDS  = [5, 6, 7, 8, 9]
OLD_SEEDS  = [0, 1, 2, 3, 4]
MAX_EPOCHS = 100
PATIENCE   = 15
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4
K_INIT     = 10   # GatingNet uses first K cycles


def run(seed: int, device):
    """Train and evaluate M3 on FD003 for one random seed."""
    set_seed(seed)

    # ---- Preprocessing (FD003: no op-condition residualization needed) ----
    train_df    = load_cmapss(DATASET, "train")
    sensor_cols = get_sensor_cols(DATASET)

    train_df = add_rul(train_df)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n  = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)

    # ---- Engine-level train/val split (seed=42 fixed, same as original) ----
    tr_m, va_m = split_engines(units, 0.2, seed=42)
    ds_tr = SeqDatasetM3(X[tr_m], y[tr_m], K=K_INIT)
    ds_va = SeqDatasetM3(X[va_m], y[va_m], K=K_INIT)
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model ----
    n_feat    = len(sensor_cols)
    model     = AttentionGateModel(n_feat, K=K_INIT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()

    best_val, pc, best_sd = float("inf"), 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_l = train_m3_epoch(model, tr_loader, optimizer, criterion, device)
        va_l = eval_m3_epoch(model, va_loader, criterion, device)
        if va_l < best_val:
            best_val = va_l
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= PATIENCE:
                print(f"    EarlyStop @epoch{epoch}  bestVal={best_val:.4f}")
                break
        if epoch % 20 == 0 or epoch == 1:
            print(f"    ep{epoch:3d}: tr={tr_l:.4f}  va={va_l:.4f}  best={best_val:.4f}")

    save_checkpoint({"model_state_dict": best_sd,
                     "n_features": n_feat,
                     "K_init": K_INIT,
                     "sensor_cols": sensor_cols,
                     "sensor_min": min_v.to_dict(),
                     "sensor_max": max_v.to_dict()},
                    "M3", DATASET, seed)

    # ---- Test inference ----
    model.load_state_dict(best_sd)
    test_raw = load_cmapss(DATASET, "test")
    test_n   = apply_normalization(test_raw, sensor_cols, min_v, max_v)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, load_test_rul(DATASET))

    preds  = predict_m3(model, X_te, device, K=K_INIT)
    rmse, ns = compute_metrics(preds, y_te)
    save_predictions(preds, y_te, "M3", DATASET, seed)
    print(f"  -> RMSE={rmse:.4f}  NASA={ns:.2f}")
    return rmse, ns


def compute_seed_metrics_from_csv(seed: int):
    """Recompute RMSE and NASA score from saved prediction CSV."""
    path = os.path.join(RESULTS_DIR,
                        f"raw_predictions_M3_{DATASET}_seed{seed}.csv")
    df   = pd.read_csv(path)
    rmse = float(np.sqrt(np.mean((df["pred"].values - df["true"].values) ** 2)))
    d    = df["pred"].values - df["true"].values
    pen  = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    ns   = float(pen.sum())   # SUM (matches nasa_score() in h6_p2_model_utils.py)
    return rmse, ns


def build_combined_summary(all_results: list):
    """all_results: list of (seed, rmse, nasa) tuples."""
    rows = []
    for seed, rmse, ns in all_results:
        rows.append({"seed": seed, "rmse": rmse, "nasa_score": ns})
    df = pd.DataFrame(rows)

    summary_rows = []
    # Per-seed rows
    for _, row in df.iterrows():
        summary_rows.append({
            "type": "per_seed",
            "seed": int(row["seed"]),
            "rmse": round(row["rmse"], 4),
            "nasa_score": round(row["nasa_score"], 4),
        })
    # 5-seed summary (seeds 0-4)
    old5 = df[df["seed"].isin(OLD_SEEDS)]
    summary_rows.append({
        "type": "5seed_mean",
        "seed": -1,
        "rmse": round(old5["rmse"].mean(), 4),
        "nasa_score": round(old5["nasa_score"].mean(), 4),
    })
    summary_rows.append({
        "type": "5seed_std",
        "seed": -1,
        "rmse": round(old5["rmse"].std(ddof=1), 4),
        "nasa_score": round(old5["nasa_score"].std(ddof=1), 4),
    })
    # 10-seed summary (seeds 0-9)
    summary_rows.append({
        "type": "10seed_mean",
        "seed": -1,
        "rmse": round(df["rmse"].mean(), 4),
        "nasa_score": round(df["nasa_score"].mean(), 4),
    })
    summary_rows.append({
        "type": "10seed_std",
        "seed": -1,
        "rmse": round(df["rmse"].std(ddof=1), 4),
        "nasa_score": round(df["nasa_score"].std(ddof=1), 4),
    })
    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M3-extended] Device: {device}")
    print(f"Dataset: {DATASET}  |  New seeds: {NEW_SEEDS}\n")

    # ---- Run new seeds ----
    new_results = []
    for seed in NEW_SEEDS:
        print(f"\n  seed={seed}")
        r, n = run(seed, device)
        new_results.append((seed, r, n))

    # ---- Load old seed metrics from existing CSVs ----
    print("\n---- Loading existing seed 0-4 results ----")
    old_results = []
    for seed in OLD_SEEDS:
        r, n = compute_seed_metrics_from_csv(seed)
        print(f"  seed={seed}: RMSE={r:.4f}  NASA={n:.4f}")
        old_results.append((seed, r, n))

    # ---- Combine and report ----
    all_results = sorted(old_results + new_results, key=lambda x: x[0])

    print("\n---- Combined 10-seed results ----")
    for seed, r, n in all_results:
        print(f"  seed={seed}: RMSE={r:.4f}  NASA={n:.4f}")

    rmse_all = [r for _, r, _ in all_results]
    ns_all   = [n for _, _, n in all_results]
    print(f"\n[M3 FD003 10-seed]")
    print(f"  RMSE = {np.mean(rmse_all):.4f} ± {np.std(rmse_all, ddof=1):.4f}")
    print(f"  NASA = {np.mean(ns_all):.4f} ± {np.std(ns_all, ddof=1):.4f}")

    # ---- Save summary CSV ----
    summary_df = build_combined_summary(all_results)
    out_path   = os.path.join(RESULTS_DIR, "H6_M3_FD003_10seeds_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\n[DONE]")
