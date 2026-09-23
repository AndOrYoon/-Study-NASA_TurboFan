# -*- coding: utf-8 -*-
"""
04_run_h6_corrected.py
H6 — Corrected-Protocol Re-Run (Option C)

Four protocol fixes vs. original H6:
  Fix 1: val-split seed = per-run training seed  (was fixed seed=42)
  Fix 2: MIN_EPOCHS = 30 warmup before early stopping counts  (was absent)
  Fix 3: MAX_EPOCHS = 300  (was 100)
  Fix 4: predictions clipped [0, 125]; test RUL clipped to 125  (both absent)

Original H6 scripts (phase2_models/) are NOT modified.
Phase 1 GMM artifacts are reused as-is (clustering is unchanged).

Results -> Data_Analysis/Results/H6_corrected/h6_corrected_results.csv
           Data_Analysis/Results/H6_corrected/raw_predictions_*.csv
           Data_Analysis/Results/H6_corrected/models/*.pt
"""

import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

_ROOT   = Path(__file__).resolve().parents[2]   # Data_Analysis/
_H6_P2  = _ROOT / "Code" / "H3_fault_mode" / "phase2_models"
_SHARED = _ROOT / "Code" / "shared"

sys.path.insert(0, str(_H6_P2))
sys.path.insert(0, str(_SHARED))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from h6_p2_model_utils import (
    set_seed, SEEDS, RUL_CLIP, WINDOW_SIZE,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_op_residual_fd004, apply_op_residual_fd004,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    make_loader,
    LSTMBranch, AttentionGateModel,
    SeqDatasetM3, SeqDatasetWithProbs, SoftGatingModel,
    train_epoch, eval_epoch, predict_sequences,
    train_m2_epoch, eval_m2_epoch, predict_m2,
    train_m3_epoch, eval_m3_epoch, predict_m3,
    compute_metrics,
)
from h6_p2_inference import get_test_cluster_assignments

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
H6_RESULTS    = _ROOT / "Results" / "H3_fault_mode"   # Phase1 artifacts (read-only)
CORRECTED_DIR = _ROOT / "Results" / "H6_corrected"
CKPT_DIR      = CORRECTED_DIR / "models"
CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = CORRECTED_DIR / "h6_corrected_results.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASETS   = ["FD003", "FD004"]
MODELS     = ["M0", "M1", "M2", "M3"]
MIN_EPOCHS = 30     # Fix 2
MAX_EPOCHS = 300    # Fix 3
PATIENCE   = 15
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4
K_INIT     = 10     # GatingNet prefix length (M3)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Corrected early-stopping loop
# ---------------------------------------------------------------------------

def _train_loop(model, tr_loader, va_loader, optimizer, criterion,
                train_fn, eval_fn, device):
    """
    Unified training loop with Fixes 2 & 3.
    Returns (best_state_dict, best_val_loss).
    """
    best_val, best_sd, no_improve = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_fn(model, tr_loader, optimizer, criterion, device)
        val_loss = eval_fn(model, va_loader, criterion, device)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_sd    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        elif epoch >= MIN_EPOCHS:          # Fix 2: no early stop before warmup
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    return best_sd, best_val


# ---------------------------------------------------------------------------
# Preprocessing (shared by all models, deterministic given dataset)
# ---------------------------------------------------------------------------

def _preprocess(dataset: str):
    """
    Load, op-residualize (FD004), add RUL, normalize.

    Returns
    -------
    X, y, units  : training sequences (all engines, before val split)
    X_te         : test sequences (n_test, WINDOW, n_feat)
    y_te         : test RUL, clipped to RUL_CLIP (Fix 4)
    n_feat       : number of sensor features
    test_df_gmm  : test DataFrame for M1/M2 GMM assignment
                   (op-residualized for FD004, raw for FD003)
    """
    train_df    = load_cmapss(dataset, "train")
    sensor_cols = get_sensor_cols(dataset)

    sc_op = km_op = cm_op = None
    if dataset == "FD004":
        sc_op, km_op, cm_op = fit_op_residual_fd004(train_df, sensor_cols)
        train_df = apply_op_residual_fd004(train_df, sc_op, km_op, cm_op, sensor_cols)

    train_df = add_rul(train_df)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n      = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units  = make_train_sequences(train_n, sensor_cols)

    test_raw = load_cmapss(dataset, "test")
    if dataset == "FD004":
        test_df_gmm = apply_op_residual_fd004(test_raw, sc_op, km_op, cm_op, sensor_cols)
    else:
        test_df_gmm = test_raw.copy()

    test_n = apply_normalization(test_df_gmm, sensor_cols, min_v, max_v)

    # Fix 4: clip test RUL before passing to make_test_sequences
    rul_raw = load_test_rul(dataset)                # raw pandas Series
    rul_clipped = rul_raw.clip(upper=RUL_CLIP)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, rul_clipped)
    y_te = np.minimum(y_te, RUL_CLIP).astype(np.float32)

    return X, y, units, X_te, y_te, len(sensor_cols), test_df_gmm


# ---------------------------------------------------------------------------
# Gate confidence (M3 only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _gate_conf(model, loader, device) -> float:
    model.eval()
    confs = []
    for xb_full, xb_init, _ in loader:
        w = model.gating(xb_init.to(device)).cpu().numpy()  # (B, 2)
        confs.append(np.max(w, axis=1))
    return float(np.concatenate(confs).mean()) if confs else float("nan")


# ---------------------------------------------------------------------------
# M0 — Single LSTM baseline
# ---------------------------------------------------------------------------

def run_m0(dataset: str, seed: int) -> dict:
    set_seed(seed)
    t0 = time.time()

    X, y, units, X_te, y_te, n_feat, _ = _preprocess(dataset)

    # Fix 1: val split uses training seed
    tr_m, va_m = split_engines(units, 0.2, seed=seed)
    tr_loader  = make_loader(X[tr_m], y[tr_m], BATCH_SIZE, shuffle=True)
    va_loader  = make_loader(X[va_m], y[va_m], BATCH_SIZE, shuffle=False)

    model     = LSTMBranch(n_feat).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()

    best_sd, best_val = _train_loop(
        model, tr_loader, va_loader, optimizer, criterion,
        train_epoch, eval_epoch, DEVICE,
    )

    model.load_state_dict(best_sd); model.to(DEVICE)
    preds = np.clip(predict_sequences(model, X_te, DEVICE), 0, RUL_CLIP)  # Fix 4
    rmse, nasa = compute_metrics(preds, y_te)

    _save_ckpt(best_sd, "M0", dataset, seed)
    _save_preds(preds, y_te, "M0", dataset, seed)
    return _row("M0", dataset, seed, rmse, nasa, float("nan"), best_val, time.time() - t0)


# ---------------------------------------------------------------------------
# M1 — Hard Routing (GMM argmax, two separate branches)
# ---------------------------------------------------------------------------

def run_m1(dataset: str, seed: int) -> dict:
    set_seed(seed)
    t0 = time.time()

    X, y, units, X_te, y_te, n_feat, test_df_gmm = _preprocess(dataset)

    ca_path = H6_RESULTS / f"cluster_assignments_{dataset.lower()}.csv"
    if not ca_path.exists():
        raise FileNotFoundError(f"Phase1 cluster assignments not found: {ca_path}")
    ca   = pd.read_csv(ca_path, index_col=0)
    c0_u = set(ca[ca["cluster"] == 0].index.tolist())
    c1_u = set(ca[ca["cluster"] == 1].index.tolist())

    m0_mask = np.array([u in c0_u for u in units])
    m1_mask = np.array([u in c1_u for u in units])
    print(f"    cluster sizes: C0={m0_mask.sum()} seqs  C1={m1_mask.sum()} seqs")

    def _train_branch(X_b, y_b, units_b, branch_seed):
        # Fix 1: per-branch val split uses branch seed (not fixed 42)
        tr_m, va_m = split_engines(units_b, 0.2, seed=branch_seed)
        tr_ld = make_loader(X_b[tr_m], y_b[tr_m], BATCH_SIZE, shuffle=True)
        va_ld = make_loader(X_b[va_m], y_b[va_m], BATCH_SIZE, shuffle=False)
        set_seed(branch_seed)
        mdl = LSTMBranch(n_feat).to(DEVICE)
        opt = torch.optim.Adam(mdl.parameters(), lr=LR, weight_decay=WD)
        sd, bv = _train_loop(mdl, tr_ld, va_ld, opt, nn.MSELoss(),
                              train_epoch, eval_epoch, DEVICE)
        return sd, bv

    sd0, bv0 = _train_branch(X[m0_mask], y[m0_mask], units[m0_mask], seed)
    sd1, bv1 = _train_branch(X[m1_mask], y[m1_mask], units[m1_mask], seed + 100)

    branch0 = LSTMBranch(n_feat).to(DEVICE); branch0.load_state_dict(sd0)
    branch1 = LSTMBranch(n_feat).to(DEVICE); branch1.load_state_dict(sd1)

    y0_all = predict_sequences(branch0, X_te, DEVICE)
    y1_all = predict_sequences(branch1, X_te, DEVICE)

    cluster_ids = get_test_cluster_assignments(test_df_gmm, dataset, hard=True)
    preds = np.clip(np.where(cluster_ids == 0, y0_all, y1_all), 0, RUL_CLIP)  # Fix 4
    rmse, nasa = compute_metrics(preds, y_te)
    print(f"    test cluster_dist={np.bincount(cluster_ids).tolist()}")

    _save_ckpt(sd0, "M1_branch0", dataset, seed)
    _save_ckpt(sd1, "M1_branch1", dataset, seed)
    _save_preds(preds, y_te, "M1", dataset, seed)
    return _row("M1", dataset, seed, rmse, nasa, float("nan"),
                max(bv0, bv1), time.time() - t0)


# ---------------------------------------------------------------------------
# M2 — Soft Gating (GMM probabilities, joint branch training)
# ---------------------------------------------------------------------------

def run_m2(dataset: str, seed: int) -> dict:
    set_seed(seed)
    t0 = time.time()

    X, y, units, X_te, y_te, n_feat, test_df_gmm = _preprocess(dataset)

    ca_path = H6_RESULTS / f"cluster_assignments_{dataset.lower()}.csv"
    if not ca_path.exists():
        raise FileNotFoundError(f"Phase1 cluster assignments not found: {ca_path}")
    ca     = pd.read_csv(ca_path, index_col=0)
    p0_map = ca["p0"].to_dict()
    p1_map = ca["p1"].to_dict()
    probs  = np.array([[p0_map.get(u, 0.5), p1_map.get(u, 0.5)] for u in units],
                      dtype=np.float32)

    # Fix 1: val split uses training seed
    tr_m, va_m = split_engines(units, 0.2, seed=seed)
    ds_tr = SeqDatasetWithProbs(X[tr_m], y[tr_m], probs[tr_m])
    ds_va = SeqDatasetWithProbs(X[va_m], y[va_m], probs[va_m])
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = SoftGatingModel(n_feat).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()

    best_sd, best_val = _train_loop(
        model, tr_loader, va_loader, optimizer, criterion,
        train_m2_epoch, eval_m2_epoch, DEVICE,
    )

    model.load_state_dict(best_sd); model.to(DEVICE)

    soft_probs = get_test_cluster_assignments(test_df_gmm, dataset, hard=False)
    w0, w1 = soft_probs[:, 0], soft_probs[:, 1]
    preds = np.clip(predict_m2(model, X_te, w0, w1, DEVICE), 0, RUL_CLIP)  # Fix 4
    rmse, nasa = compute_metrics(preds, y_te)

    _save_ckpt(best_sd, "M2", dataset, seed)
    _save_preds(preds, y_te, "M2", dataset, seed)
    return _row("M2", dataset, seed, rmse, nasa, float("nan"), best_val, time.time() - t0)


# ---------------------------------------------------------------------------
# M3 — Attention Gate (end-to-end, no GMM)
# ---------------------------------------------------------------------------

def run_m3(dataset: str, seed: int) -> dict:
    set_seed(seed)
    t0 = time.time()

    X, y, units, X_te, y_te, n_feat, _ = _preprocess(dataset)

    # Fix 1: val split uses training seed
    tr_m, va_m = split_engines(units, 0.2, seed=seed)
    ds_tr = SeqDatasetM3(X[tr_m], y[tr_m], K=K_INIT)
    ds_va = SeqDatasetM3(X[va_m], y[va_m], K=K_INIT)
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = AttentionGateModel(n_feat, K=K_INIT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()

    best_sd, best_val = _train_loop(
        model, tr_loader, va_loader, optimizer, criterion,
        train_m3_epoch, eval_m3_epoch, DEVICE,
    )

    model.load_state_dict(best_sd); model.to(DEVICE)
    preds = np.clip(predict_m3(model, X_te, DEVICE, K=K_INIT), 0, RUL_CLIP)  # Fix 4
    rmse, nasa = compute_metrics(preds, y_te)
    conf = _gate_conf(model, tr_loader, DEVICE)

    _save_ckpt(best_sd, "M3", dataset, seed)
    _save_preds(preds, y_te, "M3", dataset, seed)
    return _row("M3", dataset, seed, rmse, nasa, conf, best_val, time.time() - t0)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_ckpt(state_dict, model_name: str, dataset: str, seed: int):
    path = CKPT_DIR / f"{model_name}_{dataset}_seed{seed}_best.pt"
    torch.save(state_dict, str(path))


def _save_preds(preds: np.ndarray, trues: np.ndarray,
                model_name: str, dataset: str, seed: int):
    path = CORRECTED_DIR / f"raw_predictions_{model_name}_{dataset}_seed{seed}.csv"
    pd.DataFrame({"pred": preds, "true": trues}).to_csv(path, index=False)


def _row(model, dataset, seed, rmse, nasa, gate_conf, best_val, elapsed) -> dict:
    return {
        "model":         model,
        "dataset":       dataset,
        "seed":          seed,
        "rmse":          round(float(rmse),      4),
        "nasa_score":    round(float(nasa),      4),
        "gate_conf":     round(float(gate_conf), 4),
        "best_val_loss": round(float(best_val),  6),
        "elapsed_s":     round(float(elapsed),   1),
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame):
    print("\n=== RMSE Summary (mean ± std over 5 seeds) ===")
    grp = (
        df.groupby(["model", "dataset"])["rmse"]
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
    )
    grp["rmse_str"] = (
        grp["mean"].map("{:.4f}".format) + " ± " +
        grp["std"].map("{:.4f}".format)
    )
    pivot = grp.pivot_table(
        index="model", columns="dataset",
        values="rmse_str", aggfunc="first",
    )
    # Enforce model order
    pivot = pivot.reindex([m for m in MODELS if m in pivot.index])
    print(pivot.to_string())

    print("\n=== Gate Confidence (M3 only) ===")
    m3_df = df[df["model"] == "M3"]
    if not m3_df.empty:
        for ds, grp_ds in m3_df.groupby("dataset"):
            c = grp_ds["gate_conf"]
            print(f"  {ds}: {c.mean():.4f} ± {c.std():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RUN_FN = {"M0": run_m0, "M1": run_m1, "M2": run_m2, "M3": run_m3}


def main():
    runs  = [(m, ds, s) for m in MODELS for ds in DATASETS for s in SEEDS]
    total = len(runs)   # 4 × 2 × 5 = 40

    print(f"Device : {DEVICE}")
    print(f"Runs   : {total}  ({len(MODELS)} models × {len(DATASETS)} datasets × {len(SEEDS)} seeds)")
    print(f"Config : MIN_EPOCHS={MIN_EPOCHS}  MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}")
    print(f"Output : {OUT_CSV}")
    print("-" * 72)

    # Resume support
    if OUT_CSV.exists():
        done_df   = pd.read_csv(OUT_CSV)
        done_keys = set(zip(done_df["model"], done_df["dataset"], done_df["seed"]))
        print(f"Resuming: {len(done_keys)} runs already complete.\n")
    else:
        done_df   = pd.DataFrame()
        done_keys = set()

    results = []
    for i, (model, ds, seed) in enumerate(runs, 1):
        key = (model, ds, seed)
        if key in done_keys:
            print(f"[{i:3d}/{total}] SKIP  {model} {ds} seed={seed}")
            continue

        print(f"[{i:3d}/{total}] RUN   {model} {ds} seed={seed} ...", end=" ", flush=True)
        try:
            row = RUN_FN[model](ds, seed)
            print(
                f"RMSE={row['rmse']:.4f}  NASA={row['nasa_score']:.1f}"
                f"  conf={row['gate_conf']:.3f}  [{row['elapsed_s']:.0f}s]"
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            row = _row(model, ds, seed,
                       float("nan"), float("nan"), float("nan"),
                       float("nan"), float("nan"))

        results.append(row)

        # Save after every run (safe restart)
        new_df   = pd.DataFrame(results)
        combined = (pd.concat([done_df, new_df], ignore_index=True)
                    if not done_df.empty else new_df)
        combined.to_csv(OUT_CSV, index=False)

    print(f"\nAll done.  Results -> {OUT_CSV}")
    _print_summary(pd.read_csv(OUT_CSV))


if __name__ == "__main__":
    main()
