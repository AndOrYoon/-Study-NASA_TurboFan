# -*- coding: utf-8 -*-
"""
H6 Phase 2 — M3: Attention-Gate Multi-Branch LSTM
End-to-end: GatingNet(first K=10 cycles) + two LSTM branches.
Loss = MSE(y_final) + 0.05*MSE(y0) + 0.05*MSE(y1)
No GMM needed — the gate is learned from data.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

from h6_p2_model_utils import (
    set_seed, SEEDS, WINDOW_SIZE,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    apply_op_residual_fd004, fit_op_residual_fd004,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    SeqDatasetM3, AttentionGateModel,
    train_m3_epoch, eval_m3_epoch, predict_m3,
    compute_metrics, save_predictions, save_checkpoint,
)

DATASETS   = ["FD003", "FD004"]
MAX_EPOCHS = 100
PATIENCE   = 15
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4
K_INIT     = 10   # GatingNet uses first K cycles


def run(dataset: str, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

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

    # ---- Train / val split ----
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
                    "M3", dataset, seed)

    # ---- Test inference ----
    model.load_state_dict(best_sd)
    test_raw = load_cmapss(dataset, "test")
    if dataset == "FD004" and sc_op is not None:
        test_raw = apply_op_residual_fd004(test_raw, sc_op, km_op, cm_op, sensor_cols)

    test_n = apply_normalization(test_raw, sensor_cols, min_v, max_v)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, load_test_rul(dataset))

    preds  = predict_m3(model, X_te, device, K=K_INIT)
    rmse, ns = compute_metrics(preds, y_te)
    save_predictions(preds, y_te, "M3", dataset, seed)
    print(f"  -> RMSE={rmse:.4f}  NASA={ns:.2f}")
    return rmse, ns


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M3] Device: {device}")
    for dataset in DATASETS:
        print(f"\n{'='*50}\nDataset: {dataset}\n{'='*50}")
        rmse_l, ns_l = [], []
        for seed in SEEDS:
            print(f"\n  seed={seed}")
            r, n = run(dataset, seed)
            rmse_l.append(r); ns_l.append(n)
        print(f"\n[M3 {dataset}] RMSE={np.mean(rmse_l):.4f}+/-{np.std(rmse_l):.4f}  "
              f"NASA={np.mean(ns_l):.2f}+/-{np.std(ns_l):.2f}")
    print("\n[DONE] M3 complete.")
