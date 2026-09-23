# -*- coding: utf-8 -*-
"""
H6 K Sensitivity Analysis
Sweeps K ∈ {5, 10, 15, 20, 30} for M3 GatingNet on FD003.
Results saved to Data_Analysis/Results/H3_fault_mode/k_sensitivity/
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from h6_p2_model_utils import (
    set_seed, SEEDS, WINDOW_SIZE,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    SeqDatasetM3, AttentionGateModel,
    train_m3_epoch, eval_m3_epoch, predict_m3,
    compute_metrics,
)

DATASET    = "FD003"
K_VALUES   = [5, 10, 15, 20, 30]
MAX_EPOCHS = 100
PATIENCE   = 15
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "Results" / "H3_fault_mode" / "k_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_one(dataset: str, seed: int, k: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    train_df    = load_cmapss(dataset, "train")
    sensor_cols = get_sensor_cols(dataset)
    train_df    = add_rul(train_df)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n  = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)

    tr_m, va_m = split_engines(units, 0.2, seed=42)
    ds_tr = SeqDatasetM3(X[tr_m], y[tr_m], K=k)
    ds_va = SeqDatasetM3(X[va_m], y[va_m], K=k)
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    n_feat = len(sensor_cols)
    model  = AttentionGateModel(n_feat, K=k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()

    best_val, pc, best_sd = float("inf"), 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_l = train_m3_epoch(model, tr_loader, optimizer, criterion, device)
        va_l = eval_m3_epoch(model, va_loader, criterion, device)
        if va_l < best_val:
            best_val = va_l
            best_sd  = {k2: v.clone() for k2, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= PATIENCE:
                break

    model.load_state_dict(best_sd)
    test_raw = load_cmapss(dataset, "test")
    test_n   = apply_normalization(test_raw, sensor_cols, min_v, max_v)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, load_test_rul(dataset))

    preds      = predict_m3(model, X_te, device, K=k)
    rmse, ns   = compute_metrics(preds, y_te)
    return rmse, ns


def main():
    records = []
    for k in K_VALUES:
        print(f"\n{'='*50}\nK = {k}\n{'='*50}")
        rmse_l, ns_l = [], []
        for seed in SEEDS:
            print(f"  seed={seed}", end=" ", flush=True)
            r, n = run_one(DATASET, seed, k)
            rmse_l.append(r); ns_l.append(n)
            print(f"RMSE={r:.4f}  NASA={n:.2f}")
            records.append({"K": k, "seed": seed, "rmse": r, "nasa_score": n})
        print(f"[K={k}] RMSE={np.mean(rmse_l):.4f}±{np.std(rmse_l):.4f}  "
              f"NASA={np.mean(ns_l):.2f}±{np.std(ns_l):.2f}")

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "k_sensitivity_fd003.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    summary = (df.groupby("K")
                 .agg(rmse_mean=("rmse", "mean"),
                      rmse_std=("rmse", "std"),
                      nasa_mean=("nasa_score", "mean"),
                      nasa_std=("nasa_score", "std"))
                 .reset_index())
    summary_path = RESULTS_DIR / "k_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
    print("\n[DONE] K sensitivity complete.")
