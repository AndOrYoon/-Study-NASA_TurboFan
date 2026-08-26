# -*- coding: utf-8 -*-
"""
02_run_unified.py
H2-H3 Unified-Control Ad-hoc Analysis — Experiment Runner

Conditions:
  A: N1 + M0  (FD001-FD004, 5 seeds)  →  20 runs
  B: N3 + M0  (FD001-FD004, 5 seeds)  →  20 runs
  C: N1 + M3  (FD003, FD004, 5 seeds) →  10 runs
  D: N3 + M3  (FD003, FD004, 5 seeds) →  10 runs  [optional, set RUN_D=True]

Total mandatory: 50 runs  |  with D: 60 runs
"""

import sys
import os
import time
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "Code" / "shared"))
sys.path.insert(0, str(_ROOT / "Code" / "H5_normalization"))
sys.path.insert(0, str(_ROOT / "Code" / "H6_fault_mode" / "phase2_models"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# H5 normalizers
import importlib
_norm_mod = importlib.import_module("02_normalizers")
FleetMinMax   = _norm_mod.FleetMinMax
PerUnitMinMax = _norm_mod.PerUnitMinMax

# H6 model utils
from h6_p2_model_utils import (
    LSTMBranch, AttentionGateModel,
    SeqDataset, SeqDatasetM3,
    train_epoch, eval_epoch,
    train_m3_epoch, eval_m3_epoch,
    predict_sequences, predict_m3,
    compute_metrics, set_seed,
)

# Unified data loader
from importlib import import_module as _im
_loader = _im("01_unified_data_loader")
prepare_data = _loader.prepare_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = _ROOT / "Results" / "Ad-hoc_Analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "unified_results.csv"

SEEDS    = [0, 1, 2, 3, 4]
RUL_CLIP = 125
K_GATE   = 10          # GatingNet prefix length
PATIENCE = 15
MAX_EPOCHS = 300
LR       = 1e-3
WD       = 1e-4
RUN_D    = False       # set True to also run condition D

CONDITIONS = [
    # (cond_id, norm_name, model_type, datasets)
    ("A", "N1", "M0", ["FD001", "FD002", "FD003", "FD004"]),
    ("B", "N3", "M0", ["FD001", "FD002", "FD003", "FD004"]),
    ("C", "N1", "M3", ["FD003", "FD004"]),
]
if RUN_D:
    CONDITIONS.append(("D", "N3", "M3", ["FD003", "FD004"]))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Normalizer factory
# ---------------------------------------------------------------------------

def _make_norm(name: str):
    if name == "N1":
        return FleetMinMax()
    if name == "N3":
        return PerUnitMinMax(n_init=5)
    raise ValueError(f"Unknown normalizer: {name}")


# ---------------------------------------------------------------------------
# Gate confidence (M3 only): mean max(w0, w1) over training set
# ---------------------------------------------------------------------------

@torch.no_grad()
def gate_confidence(model, train_loader_m3, device) -> float:
    model.eval()
    confs = []
    for xb_full, xb_init, _ in train_loader_m3:
        xb_init = xb_init.to(device)
        w = model.gating(xb_init).cpu().numpy()      # (B, 2)
        confs.append(np.max(w, axis=1))              # (B,)
    return float(np.concatenate(confs).mean())


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(cond_id, norm_name, model_type, dataset, seed) -> dict:
    set_seed(seed)
    t0 = time.time()

    # --- Data ---
    (tr_m0, va_m0,
     tr_m3, va_m3,
     X_test, true_rul, n_feat) = prepare_data(
        dataset, _make_norm(norm_name), seed=seed, K=K_GATE
    )

    # --- Model ---
    criterion = nn.MSELoss()

    if model_type == "M0":
        model = LSTMBranch(n_features=n_feat).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        tr_loader, va_loader = tr_m0, va_m0
        _train_fn = train_epoch
        _eval_fn  = eval_epoch

    else:  # M3
        model = AttentionGateModel(n_features=n_feat, K=K_GATE).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        tr_loader, va_loader = tr_m3, va_m3
        _train_fn = train_m3_epoch
        _eval_fn  = eval_m3_epoch

    # --- Training with early stopping ---
    best_val  = float("inf")
    best_sd   = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        _train_fn(model, tr_loader, optimizer, criterion, DEVICE)
        val_loss = _eval_fn(model, va_loader, criterion, DEVICE)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_sd  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    # --- Load best weights ---
    model.load_state_dict(best_sd)
    model.to(DEVICE)

    # --- Inference on test set ---
    if model_type == "M0":
        preds = predict_sequences(model, X_test, DEVICE)
    else:
        preds = predict_m3(model, X_test, DEVICE, K=K_GATE)

    # Eval clip [0, 125]
    preds = np.clip(preds, 0, RUL_CLIP)

    rmse, nasa = compute_metrics(preds, true_rul)

    # Gate confidence (M3 only)
    conf = gate_confidence(model, tr_m3, DEVICE) if model_type == "M3" else float("nan")

    elapsed = time.time() - t0
    result = {
        "condition":   cond_id,
        "normalizer":  norm_name,
        "model":       model_type,
        "dataset":     dataset,
        "seed":        seed,
        "rmse":        round(rmse, 4),
        "nasa_score":  round(nasa, 4),
        "gate_conf":   round(conf, 4),
        "best_val_loss": round(best_val, 6),
        "elapsed_s":   round(elapsed, 1),
    }
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # Build run list
    runs = []
    for cond_id, norm_name, model_type, datasets in CONDITIONS:
        for ds in datasets:
            for seed in SEEDS:
                runs.append((cond_id, norm_name, model_type, ds, seed))

    total = len(runs)
    print(f"Device: {DEVICE}")
    print(f"Total runs: {total}  (RUN_D={RUN_D})")
    print("-" * 70)

    # Load existing results to allow resuming
    if OUT_CSV.exists():
        done_df = pd.read_csv(OUT_CSV)
        done_keys = set(
            zip(done_df["condition"], done_df["normalizer"],
                done_df["model"], done_df["dataset"], done_df["seed"])
        )
        print(f"Resuming: {len(done_keys)} runs already complete.")
    else:
        done_df   = pd.DataFrame()
        done_keys = set()

    results = []
    for i, (cond_id, norm_name, model_type, ds, seed) in enumerate(runs, 1):
        key = (cond_id, norm_name, model_type, ds, seed)
        if key in done_keys:
            print(f"[{i:3d}/{total}] SKIP  {cond_id} {norm_name}+{model_type} {ds} seed={seed}")
            continue

        print(f"[{i:3d}/{total}] RUN   {cond_id} {norm_name}+{model_type} {ds} seed={seed} ...",
              end=" ", flush=True)
        try:
            row = run_one(cond_id, norm_name, model_type, ds, seed)
            print(f"RMSE={row['rmse']:.2f}  NASA={row['nasa_score']:.1f}"
                  f"  conf={row['gate_conf']:.3f}  [{row['elapsed_s']:.0f}s]")
            results.append(row)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "condition": cond_id, "normalizer": norm_name,
                "model": model_type, "dataset": ds, "seed": seed,
                "rmse": float("nan"), "nasa_score": float("nan"),
                "gate_conf": float("nan"), "best_val_loss": float("nan"),
                "elapsed_s": float("nan"),
            })

        # Save after every run (safe restart)
        new_df = pd.DataFrame(results)
        combined = pd.concat([done_df, new_df], ignore_index=True) if not done_df.empty else new_df
        combined.to_csv(OUT_CSV, index=False)

    print("\nDone. Results saved to:", OUT_CSV)
    _print_summary(OUT_CSV)


def _print_summary(csv_path):
    df = pd.read_csv(csv_path)
    print("\n=== RMSE Summary (mean ± std over 5 seeds) ===")
    summary = (
        df.groupby(["condition", "normalizer", "model", "dataset"])["rmse"]
        .agg(["mean", "std"])
        .round(2)
        .reset_index()
    )
    summary["rmse"] = (
        summary["mean"].map("{:.2f}".format) + " ± " +
        summary["std"].map("{:.2f}".format)
    )
    pivot = summary.pivot_table(
        index=["condition", "normalizer", "model"],
        columns="dataset",
        values="rmse",
        aggfunc="first",
    )
    print(pivot.to_string())


if __name__ == "__main__":
    main()
