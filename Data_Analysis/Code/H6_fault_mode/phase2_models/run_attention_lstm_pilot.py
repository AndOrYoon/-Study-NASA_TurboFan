# -*- coding: utf-8 -*-
"""
H6 Attention-LSTM Pilot — FD003 only
Tests whether M3 gating still provides additional benefit when the backbone
already includes a self-attention layer between LSTM₁ and the FC head.

Models
------
AttnLSTM-M0 : LSTM₁(64) → Self-Attention(heads=4) → Dropout → FC  [no gating]
AttnLSTM-M3 : GatingNet (first K=10 raw cycles) + two AttnLSTM-M0 branches
              Loss = MSE(final) + 0.05*MSE(branch0) + 0.05*MSE(branch1)

Hyperparameters: identical to LSTM and Transformer experiments
  hidden=64, num_heads=4, dropout=0.2
  Adam lr=1e-3, wd=1e-4, batch=256, patience=15, max_epochs=100
  seeds {0,1,2,3,4}

Output: Data_Analysis/Results/H6_attention_lstm_pilot_FD003.csv
"""

import sys
import os

# Ensure UTF-8 output on Windows (avoids cp949 encoding errors in print)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

# ---- Attention-LSTM models ---------------------------------------------------
from h6_attention_lstm_backbone import AttnLSTMM0, AttnLSTMM3

# ---- Config ------------------------------------------------------------------
DATASET    = "FD003"
MAX_EPOCHS = 100
PATIENCE   = 15
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4
K_INIT     = 10   # GatingNet reads first K cycles

RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# Data preparation (identical to Transformer pilot — FD003 needs no op-residual)
# ==============================================================================

def prepare_data(dataset: str):
    """Load, normalise, and split FD003.  Val split always uses seed=42."""
    train_df    = load_cmapss(dataset, "train")
    sensor_cols = get_sensor_cols(dataset)

    train_df = add_rul(train_df)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)

    # Engine-level val split — seed fixed at 42 (same as all H6 experiments)
    tr_m, va_m = split_engines(units, val_frac=0.2, seed=42)

    # Test sequences
    test_df = load_cmapss(dataset, "test")
    test_n  = apply_normalization(test_df, sensor_cols, min_v, max_v)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, load_test_rul(dataset))

    return X, y, tr_m, va_m, X_te, y_te, sensor_cols


# ==============================================================================
# AttnLSTM-M0
# ==============================================================================

def run_attnlstm_m0(dataset: str, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    X, y, tr_m, va_m, X_te, y_te, sensor_cols = prepare_data(dataset)
    n_feat = len(sensor_cols)

    tr_loader = make_loader(X[tr_m], y[tr_m], BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X[va_m], y[va_m], BATCH_SIZE, shuffle=False)

    model     = AttnLSTMM0(n_feat).to(device)
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
    preds    = predict_sequences(model, X_te, device)
    rmse, ns = compute_metrics(preds, y_te)
    print(f"  -> AttnLSTM-M0  RMSE={rmse:.4f}  NASA={ns:.2f}")
    return rmse, ns


# ==============================================================================
# AttnLSTM-M3
# ==============================================================================

def run_attnlstm_m3(dataset: str, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    X, y, tr_m, va_m, X_te, y_te, sensor_cols = prepare_data(dataset)
    n_feat = len(sensor_cols)

    ds_tr     = SeqDatasetM3(X[tr_m], y[tr_m], K=K_INIT)
    ds_va     = SeqDatasetM3(X[va_m], y[va_m], K=K_INIT)
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    model     = AttnLSTMM3(n_feat, K=K_INIT).to(device)
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
    print(f"  -> AttnLSTM-M3  RMSE={rmse:.4f}  NASA={ns:.2f}")
    return rmse, ns


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {DATASET}  |  Seeds: {SEEDS}  |  K_INIT={K_INIT}")
    print("=" * 65)

    records = []

    # ---------- AttnLSTM-M0 ----------
    print(f"\n[AttnLSTM-M0]  {DATASET}")
    am0_rmse, am0_ns = [], []
    for seed in SEEDS:
        print(f"\n  seed={seed}")
        r, n = run_attnlstm_m0(DATASET, seed)
        am0_rmse.append(r)
        am0_ns.append(n)
        records.append({"model": "AttnLSTM-M0", "dataset": DATASET,
                         "seed": seed, "rmse": r, "nasa": n})

    print(f"\n[AttnLSTM-M0 {DATASET}]  "
          f"RMSE={np.mean(am0_rmse):.4f}+/-{np.std(am0_rmse):.4f}  "
          f"NASA={np.mean(am0_ns):.2f}+/-{np.std(am0_ns):.2f}")

    # ---------- AttnLSTM-M3 ----------
    print(f"\n[AttnLSTM-M3]  {DATASET}")
    am3_rmse, am3_ns = [], []
    for seed in SEEDS:
        print(f"\n  seed={seed}")
        r, n = run_attnlstm_m3(DATASET, seed)
        am3_rmse.append(r)
        am3_ns.append(n)
        records.append({"model": "AttnLSTM-M3", "dataset": DATASET,
                         "seed": seed, "rmse": r, "nasa": n})

    print(f"\n[AttnLSTM-M3 {DATASET}]  "
          f"RMSE={np.mean(am3_rmse):.4f}+/-{np.std(am3_rmse):.4f}  "
          f"NASA={np.mean(am3_ns):.2f}+/-{np.std(am3_ns):.2f}")

    # ---------- Save results CSV ----------
    df_results = pd.DataFrame(records)
    out_path   = os.path.join(RESULTS_DIR, "H6_attention_lstm_pilot_FD003.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nResults saved -> {out_path}")

    # ---------- Comparison table ----------
    am0_mean = np.mean(am0_rmse)
    am3_mean = np.mean(am3_rmse)
    am0_std  = np.std(am0_rmse)
    am3_std  = np.std(am3_rmse)

    # RMSE reductions relative to LSTM-M0 baseline (43.23)
    lstm_m0_rmse = 43.23
    lstm_m3_rmse = 14.78
    tr_m0_rmse   = 14.34   # Transformer-M0 from pilot
    tr_m3_rmse   = 14.50   # Transformer-M3 from pilot

    def pct_vs_lstmm0(r):
        return (lstm_m0_rmse - r) / lstm_m0_rmse * 100.0

    def pct_vs_am0(r):
        return (am0_mean - r) / am0_mean * 100.0

    print("\n" + "=" * 75)
    print("COMPARISON TABLE (FD003, 5 seeds)")
    print("=" * 75)
    print(f"{'Model':<26} {'RMSE mean':>10} {'RMSE std':>10} {'vs LSTM-M0':>12}")
    print("-" * 75)
    print(f"{'LSTM-M0 (reference)':<26} {lstm_m0_rmse:>10.2f} {'0.18':>10} {'baseline':>12}")
    print(f"{'LSTM-M3 (reference)':<26} {lstm_m3_rmse:>10.2f} {'1.32':>10}"
          f" {pct_vs_lstmm0(lstm_m3_rmse):>+11.1f}%")
    print(f"{'Transformer-M0':<26} {tr_m0_rmse:>10.2f} {'--':>10}"
          f" {pct_vs_lstmm0(tr_m0_rmse):>+11.1f}%")
    print(f"{'Transformer-M3':<26} {tr_m3_rmse:>10.2f} {'--':>10}"
          f" {pct_vs_lstmm0(tr_m3_rmse):>+11.1f}%")
    print(f"{'AttnLSTM-M0':<26} {am0_mean:>10.4f} {am0_std:>10.4f}"
          f" {pct_vs_lstmm0(am0_mean):>+11.1f}%")
    print(f"{'AttnLSTM-M3':<26} {am3_mean:>10.4f} {am3_std:>10.4f}"
          f" {pct_vs_lstmm0(am3_mean):>+11.1f}%")
    print("-" * 75)

    m3_vs_m0_abs = am0_mean - am3_mean
    m3_vs_m0_pct = pct_vs_am0(am3_mean)

    print(f"\nAttnLSTM-M3 vs AttnLSTM-M0:  "
          f"delta RMSE = {m3_vs_m0_abs:+.4f}  ({m3_vs_m0_pct:+.1f}%)")

    if m3_vs_m0_abs > 0.3:
        verdict = "M3 gating STILL HELPS on top of Attention-LSTM."
    elif abs(m3_vs_m0_abs) <= 0.3:
        verdict = "M3 gating provides NO additional benefit over Attention-LSTM."
    else:
        verdict = "M3 gating HURTS on top of Attention-LSTM (regression)."

    print(f"Interpretation: {verdict}")
    print("=" * 75)
    print("\n[DONE] Attention-LSTM pilot complete.")
