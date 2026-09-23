# -*- coding: utf-8 -*-
"""
H6 Transformer Pilot — FD003 only
Validates that M3's early-cycle gating principle is backbone-agnostic.

Models
------
Transformer-M0 : Single Transformer Encoder baseline (no gating)
Transformer-M3 : GatingNet (first K=10 raw cycles) + two Transformer branches
                 Loss = MSE(final) + 0.05*MSE(branch0) + 0.05*MSE(branch1)

Hyperparameters: identical to LSTM experiments
  d_model=64, nhead=4, 2 encoder layers, dim_ff=128, dropout=0.1
  Adam lr=1e-3, wd=1e-4, batch=256, patience=15, max_epochs=100

Output: Data_Analysis/Results/H6_transformer_pilot_FD003.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ---- Shared utilities from H6 Phase 2 ----------------------------------------
from h6_p2_model_utils import (
    set_seed, SEEDS, WINDOW_SIZE,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    make_loader, SeqDatasetM3,
    train_epoch, eval_epoch, predict_sequences,
    train_m3_epoch, eval_m3_epoch, predict_m3,
    compute_metrics,
)

# ---- Transformer models -------------------------------------------------------
from h6_transformer_backbone import TransformerRUL, TransformerM3

# ---- Config ------------------------------------------------------------------
DATASET     = "FD003"
MAX_EPOCHS  = 100
PATIENCE    = 15
BATCH_SIZE  = 256
LR, WD      = 1e-3, 1e-4
K_INIT      = 10    # GatingNet reads first K cycles

RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# Data preparation (shared for both models within a seed)
# ==============================================================================

def prepare_data(dataset: str, seed: int):
    """Load, normalise, and split FD003 (no op-condition residualization needed)."""
    train_df    = load_cmapss(dataset, "train")
    sensor_cols = get_sensor_cols(dataset)

    train_df = add_rul(train_df)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)

    # Engine-level val split — seed fixed at 42 (data split, not model seed)
    tr_m, va_m = split_engines(units, val_frac=0.2, seed=42)

    # Test sequences
    test_df = load_cmapss(dataset, "test")
    test_n  = apply_normalization(test_df, sensor_cols, min_v, max_v)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, load_test_rul(dataset))

    return X, y, tr_m, va_m, X_te, y_te, sensor_cols


# ==============================================================================
# Transformer-M0 (baseline)
# ==============================================================================

def run_transformer_m0(dataset: str, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    X, y, tr_m, va_m, X_te, y_te, sensor_cols = prepare_data(dataset, seed)
    n_feat = len(sensor_cols)

    tr_loader = make_loader(X[tr_m], y[tr_m], BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X[va_m], y[va_m], BATCH_SIZE, shuffle=False)

    model     = TransformerRUL(n_feat).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()

    best_val, patience_cnt, best_sd = float("inf"), 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_epoch(model, tr_loader, optimizer, criterion, device)
        va_loss = eval_epoch(model, va_loader, criterion, device)
        if va_loss < best_val:
            best_val     = va_loss
            best_sd      = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    EarlyStop @epoch{epoch}  bestVal={best_val:.4f}")
                break
        if epoch % 20 == 0 or epoch == 1:
            print(f"    ep{epoch:3d}: tr={tr_loss:.4f}  va={va_loss:.4f}  best={best_val:.4f}")

    model.load_state_dict(best_sd)
    preds        = predict_sequences(model, X_te, device)
    rmse, ns     = compute_metrics(preds, y_te)
    print(f"  -> Transformer-M0 RMSE={rmse:.4f}  NASA={ns:.2f}")
    return rmse, ns


# ==============================================================================
# Transformer-M3 (gating)
# ==============================================================================

def run_transformer_m3(dataset: str, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    X, y, tr_m, va_m, X_te, y_te, sensor_cols = prepare_data(dataset, seed)
    n_feat = len(sensor_cols)

    ds_tr     = SeqDatasetM3(X[tr_m], y[tr_m], K=K_INIT)
    ds_va     = SeqDatasetM3(X[va_m], y[va_m], K=K_INIT)
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    model     = TransformerM3(n_feat, K=K_INIT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()

    best_val, patience_cnt, best_sd = float("inf"), 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_m3_epoch(model, tr_loader, optimizer, criterion, device)
        va_loss = eval_m3_epoch(model, va_loader, criterion, device)
        if va_loss < best_val:
            best_val     = va_loss
            best_sd      = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    EarlyStop @epoch{epoch}  bestVal={best_val:.4f}")
                break
        if epoch % 20 == 0 or epoch == 1:
            print(f"    ep{epoch:3d}: tr={tr_loss:.4f}  va={va_loss:.4f}  best={best_val:.4f}")

    model.load_state_dict(best_sd)
    preds    = predict_m3(model, X_te, device, K=K_INIT)
    rmse, ns = compute_metrics(preds, y_te)
    print(f"  -> Transformer-M3 RMSE={rmse:.4f}  NASA={ns:.2f}")
    return rmse, ns


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {DATASET}  |  Seeds: {SEEDS}  |  K_INIT={K_INIT}")
    print("=" * 60)

    records = []

    # ---------- Transformer-M0 ----------
    print(f"\n[Transformer-M0]  {DATASET}")
    tm0_rmse, tm0_ns = [], []
    for seed in SEEDS:
        print(f"\n  seed={seed}")
        r, n = run_transformer_m0(DATASET, seed)
        tm0_rmse.append(r)
        tm0_ns.append(n)
        records.append({"model": "Transformer-M0", "dataset": DATASET,
                         "seed": seed, "rmse": r, "nasa": n})

    print(f"\n[Transformer-M0 {DATASET}]  "
          f"RMSE={np.mean(tm0_rmse):.4f}+/-{np.std(tm0_rmse):.4f}  "
          f"NASA={np.mean(tm0_ns):.2f}+/-{np.std(tm0_ns):.2f}")

    # ---------- Transformer-M3 ----------
    print(f"\n[Transformer-M3]  {DATASET}")
    tm3_rmse, tm3_ns = [], []
    for seed in SEEDS:
        print(f"\n  seed={seed}")
        r, n = run_transformer_m3(DATASET, seed)
        tm3_rmse.append(r)
        tm3_ns.append(n)
        records.append({"model": "Transformer-M3", "dataset": DATASET,
                         "seed": seed, "rmse": r, "nasa": n})

    print(f"\n[Transformer-M3 {DATASET}]  "
          f"RMSE={np.mean(tm3_rmse):.4f}+/-{np.std(tm3_rmse):.4f}  "
          f"NASA={np.mean(tm3_ns):.2f}+/-{np.std(tm3_ns):.2f}")

    # ---------- Save results CSV ----------
    df_results = pd.DataFrame(records)
    out_path   = os.path.join(RESULTS_DIR, "H6_transformer_pilot_FD003.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    # ---------- Comparison table ----------
    tm0_mean = np.mean(tm0_rmse)
    tm3_mean = np.mean(tm3_rmse)
    pct_reduction = (tm0_mean - tm3_mean) / tm0_mean * 100.0

    print("\n" + "=" * 60)
    print("COMPARISON TABLE (FD003)")
    print("=" * 60)
    print(f"{'Model':<22} {'RMSE mean':>10} {'RMSE std':>10} {'NASA mean':>11}")
    print("-" * 60)
    # Reference LSTM results from H6 Phase 2
    print(f"{'LSTM-M0 (reference)':<22} {'43.23':>10} {'0.18':>10} {'--':>11}")
    print(f"{'LSTM-M3 (reference)':<22} {'14.78':>10} {'1.32':>10} {'--':>11}")
    print(f"{'Transformer-M0':<22} {np.mean(tm0_rmse):>10.4f} {np.std(tm0_rmse):>10.4f}"
          f" {np.mean(tm0_ns):>11.2f}")
    print(f"{'Transformer-M3':<22} {np.mean(tm3_rmse):>10.4f} {np.std(tm3_rmse):>10.4f}"
          f" {np.mean(tm3_ns):>11.2f}")
    print("-" * 60)
    print(f"\nTransformer-M3 vs Transformer-M0 RMSE reduction: {pct_reduction:.1f}%")
    print(f"Transformer-M3 RMSE mean: {tm3_mean:.4f}  |  LSTM-M3 RMSE mean: 14.78")
    print(f"Absolute RMSE difference (Transformer-M3 vs LSTM-M3): "
          f"{tm3_mean - 14.78:+.4f}")
    print("=" * 60)
    print("\n[DONE] Transformer pilot complete.")
