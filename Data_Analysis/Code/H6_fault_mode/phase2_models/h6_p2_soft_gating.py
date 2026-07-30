# -*- coding: utf-8 -*-
"""
H6 Phase 2 — M2: Soft-Gating Multi-Branch LSTM
GMM soft probabilities weight-sum two LSTM branches.
Loss = MSE(y_final) + 0.1*MSE(y0) + 0.1*MSE(y1)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

from h6_p2_model_utils import (
    set_seed, SEEDS, RESULTS_DIR,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    apply_op_residual_fd004, fit_op_residual_fd004,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    SeqDatasetWithProbs, SoftGatingModel,
    train_m2_epoch, eval_m2_epoch, predict_m2,
    compute_metrics, save_predictions, save_checkpoint,
)
from h6_p2_inference import get_test_cluster_assignments

DATASETS   = ["FD003", "FD004"]
MAX_EPOCHS = 100
PATIENCE   = 15
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4


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

    # ---- Load cluster probabilities per unit ----
    tag = dataset.lower()
    ca  = pd.read_csv(os.path.join(RESULTS_DIR, f"cluster_assignments_{tag}.csv"), index_col=0)
    # Map unit -> [p0, p1]
    p0_map = ca["p0"].to_dict()
    p1_map = ca["p1"].to_dict()
    # Build per-sequence prob array (fallback 0.5 if unit not in training assignments)
    probs = np.array([[p0_map.get(u, 0.5), p1_map.get(u, 0.5)] for u in units],
                     dtype=np.float32)

    # ---- Train / val split ----
    tr_m, va_m = split_engines(units, 0.2, seed=42)
    ds_tr = SeqDatasetWithProbs(X[tr_m], y[tr_m], probs[tr_m])
    ds_va = SeqDatasetWithProbs(X[va_m], y[va_m], probs[va_m])
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model ----
    n_feat    = len(sensor_cols)
    model     = SoftGatingModel(n_feat).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()

    best_val, pc, best_sd = float("inf"), 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_l = train_m2_epoch(model, tr_loader, optimizer, criterion, device)
        va_l = eval_m2_epoch(model, va_loader, criterion, device)
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
                     "sensor_cols": sensor_cols,
                     "sensor_min": min_v.to_dict(),
                     "sensor_max": max_v.to_dict()},
                    "M2", dataset, seed)

    # ---- Test inference ----
    model.load_state_dict(best_sd)
    test_raw = load_cmapss(dataset, "test")
    if dataset == "FD004" and sc_op is not None:
        test_df_resid = apply_op_residual_fd004(test_raw, sc_op, km_op, cm_op, sensor_cols)
    else:
        test_df_resid = test_raw.copy()

    test_n = apply_normalization(test_df_resid, sensor_cols, min_v, max_v)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, load_test_rul(dataset))

    # Soft assignment for each test engine
    assign_df   = test_raw if dataset == "FD003" else test_df_resid
    soft_probs  = get_test_cluster_assignments(assign_df, dataset, hard=False)
    w0 = soft_probs[:, 0]
    w1 = soft_probs[:, 1]

    preds  = predict_m2(model, X_te, w0, w1, device)
    rmse, ns = compute_metrics(preds, y_te)
    save_predictions(preds, y_te, "M2", dataset, seed)
    print(f"  -> RMSE={rmse:.4f}  NASA={ns:.2f}")
    return rmse, ns


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M2] Device: {device}")
    for dataset in DATASETS:
        print(f"\n{'='*50}\nDataset: {dataset}\n{'='*50}")
        rmse_l, ns_l = [], []
        for seed in SEEDS:
            print(f"\n  seed={seed}")
            r, n = run(dataset, seed)
            rmse_l.append(r); ns_l.append(n)
        print(f"\n[M2 {dataset}] RMSE={np.mean(rmse_l):.4f}+/-{np.std(rmse_l):.4f}  "
              f"NASA={np.mean(ns_l):.2f}+/-{np.std(ns_l):.2f}")
    print("\n[DONE] M2 complete.")
