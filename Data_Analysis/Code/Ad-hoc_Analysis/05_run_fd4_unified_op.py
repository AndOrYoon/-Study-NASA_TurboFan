# -*- coding: utf-8 -*-
"""
05_run_fd4_unified_op.py
FD004 — 통일 op-condition 처리 검증 실험

04_run_h6_corrected.py와 동일한 프로토콜 (Fix 1-4) 유지.
단, FD004 op-condition residualization을 h6_p2_model_utils의
fit_op_residual_fd004 대신 shared/op_condition_utils의
fit_op_condition_kmeans + compute_cluster_means + apply_op_residual
(canonical) 로 교체.

목적: 두 구현이 수학적으로 동일한지 실증 확인.
     결과 차이가 전처리가 아닌 LSTM 랜덤 시드 차이에서 기인함을 검증.

Results -> Data_Analysis/Results/H6_unified_op/h6_fd4_unified_op_results.csv
"""

import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

_ROOT   = Path(__file__).resolve().parents[2]
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

# op_condition_utils (canonical shared implementation)
from op_condition_utils import (
    fit_op_condition_kmeans,
    compute_cluster_means,
    apply_op_residual,
)

# h6_p2_model_utils column names for op conditions
H6_OP_COLS = ("op_setting_1", "op_setting_2", "op_setting_3")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
H6_RESULTS  = _ROOT / "Results" / "H3_fault_mode"   # Phase1 artifacts (read-only)
UNIFIED_DIR = _ROOT / "Results" / "H6_unified_op"
CKPT_DIR    = UNIFIED_DIR / "models"
UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = UNIFIED_DIR / "h6_fd4_unified_op_results.csv"

# ---------------------------------------------------------------------------
# Config (identical to 04_run_h6_corrected.py)
# ---------------------------------------------------------------------------
DATASETS   = ["FD004"]           # FD003 제외: op-condition 처리 없음
MODELS     = ["M0", "M1", "M2", "M3"]
MIN_EPOCHS = 30
MAX_EPOCHS = 300
PATIENCE   = 15
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4
K_INIT     = 10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Training loop (identical to 04_run_h6_corrected.py)
# ---------------------------------------------------------------------------

def _train_loop(model, tr_loader, va_loader, optimizer, criterion,
                train_fn, eval_fn, device):
    best_val, best_sd, no_improve = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_fn(model, tr_loader, optimizer, criterion, device)
        val_loss = eval_fn(model, va_loader, criterion, device)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_sd    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        elif epoch >= MIN_EPOCHS:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    return best_sd, best_val


# ---------------------------------------------------------------------------
# Preprocessing — canonical op_condition_utils for FD004
# ---------------------------------------------------------------------------

def _preprocess(dataset: str):
    """
    Load, op-residualize (FD004 via op_condition_utils), add RUL, normalize.

    Returns
    -------
    X, y, units  : training sequences
    X_te         : test sequences
    y_te         : test RUL, clipped to RUL_CLIP
    n_feat       : number of sensor features
    test_df_gmm  : test DataFrame for M1/M2 GMM assignment
                   (op-residualized for FD004, raw for FD003)
    """
    train_df    = load_cmapss(dataset, "train")
    sensor_cols = get_sensor_cols(dataset)

    if dataset == "FD004":
        # canonical op_condition_utils — same algorithm as fit_op_residual_fd004
        # but uses shared/op_condition_utils.py interface
        scaler, km = fit_op_condition_kmeans(
            train_df, k=6, op_cols=H6_OP_COLS
        )
        cluster_means = compute_cluster_means(
            train_df, scaler, km, sensor_cols, op_cols=H6_OP_COLS
        )
        train_df = apply_op_residual(
            train_df, scaler, km, cluster_means, sensor_cols, op_cols=H6_OP_COLS
        )

    train_df = add_rul(train_df)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n      = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units  = make_train_sequences(train_n, sensor_cols)

    test_raw = load_cmapss(dataset, "test")
    if dataset == "FD004":
        test_df_gmm = apply_op_residual(
            test_raw, scaler, km, cluster_means, sensor_cols, op_cols=H6_OP_COLS
        )
    else:
        test_df_gmm = test_raw.copy()

    test_n = apply_normalization(test_df_gmm, sensor_cols, min_v, max_v)

    rul_raw     = load_test_rul(dataset)
    rul_clipped = rul_raw.clip(upper=RUL_CLIP)
    X_te, y_te  = make_test_sequences(test_n, sensor_cols, rul_clipped)
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
        w = model.gating(xb_init.to(device)).cpu().numpy()
        confs.append(np.max(w, axis=1))
    return float(np.concatenate(confs).mean()) if confs else float("nan")


# ---------------------------------------------------------------------------
# M0 — Single LSTM baseline
# ---------------------------------------------------------------------------

def run_m0(dataset: str, seed: int) -> dict:
    set_seed(seed)
    t0 = time.time()

    X, y, units, X_te, y_te, n_feat, _ = _preprocess(dataset)

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
    preds = np.clip(predict_sequences(model, X_te, DEVICE), 0, RUL_CLIP)
    rmse, nasa = compute_metrics(preds, y_te)

    _save_ckpt(best_sd, "M0", dataset, seed)
    _save_preds(preds, y_te, "M0", dataset, seed)
    return _row("M0", dataset, seed, rmse, nasa, float("nan"), best_val, time.time() - t0)


# ---------------------------------------------------------------------------
# M1 — Hard Routing (GMM argmax)
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
    preds = np.clip(np.where(cluster_ids == 0, y0_all, y1_all), 0, RUL_CLIP)
    rmse, nasa = compute_metrics(preds, y_te)
    print(f"    test cluster_dist={np.bincount(cluster_ids).tolist()}")

    _save_ckpt(sd0, "M1_branch0", dataset, seed)
    _save_ckpt(sd1, "M1_branch1", dataset, seed)
    _save_preds(preds, y_te, "M1", dataset, seed)
    return _row("M1", dataset, seed, rmse, nasa, float("nan"),
                max(bv0, bv1), time.time() - t0)


# ---------------------------------------------------------------------------
# M2 — Soft Gating (GMM probabilities)
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
    preds = np.clip(predict_m2(model, X_te, w0, w1, DEVICE), 0, RUL_CLIP)
    rmse, nasa = compute_metrics(preds, y_te)

    _save_ckpt(best_sd, "M2", dataset, seed)
    _save_preds(preds, y_te, "M2", dataset, seed)
    return _row("M2", dataset, seed, rmse, nasa, float("nan"), best_val, time.time() - t0)


# ---------------------------------------------------------------------------
# M3 — Attention Gate (end-to-end)
# ---------------------------------------------------------------------------

def run_m3(dataset: str, seed: int) -> dict:
    set_seed(seed)
    t0 = time.time()

    X, y, units, X_te, y_te, n_feat, _ = _preprocess(dataset)

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
    preds = np.clip(predict_m3(model, X_te, DEVICE, K=K_INIT), 0, RUL_CLIP)
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
    path = UNIFIED_DIR / f"raw_predictions_{model_name}_{dataset}_seed{seed}.csv"
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
    print("\n=== FD004 RMSE Summary — op_condition_utils (canonical) ===")
    print("=== 비교: 04_run_h6_corrected (fit_op_residual_fd004) ===")
    ref = {
        "M0": "18.9560 ± 3.9699",
        "M1": "33.2823 ± 2.0527",
        "M2": "18.8165 ± 1.5577",
        "M3": "17.2976 ± 1.0360",
    }
    grp = (
        df.groupby("model")["rmse"]
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
    )
    grp["new"] = (
        grp["mean"].map("{:.4f}".format) + " ± " +
        grp["std"].map("{:.4f}".format)
    )
    grp["old"] = grp["model"].map(ref)
    grp = grp.reindex([grp.index[grp["model"] == m].tolist()[0]
                        for m in MODELS if m in grp["model"].values])
    print(f"{'Model':<6} {'New (op_cond_utils)':<24} {'Old (h6_p2_model_utils)':<24}")
    print("-" * 55)
    for _, r in grp.iterrows():
        print(f"{r['model']:<6} {r['new']:<24} {r['old']:<24}")

    print("\n=== Gate Confidence (M3 only) ===")
    m3_df = df[df["model"] == "M3"]
    if not m3_df.empty:
        c = m3_df["gate_conf"]
        print(f"  FD004: {c.mean():.4f} ± {c.std():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RUN_FN = {"M0": run_m0, "M1": run_m1, "M2": run_m2, "M3": run_m3}


def main():
    runs  = [(m, ds, s) for m in MODELS for ds in DATASETS for s in SEEDS]
    total = len(runs)   # 4 × 1 × 5 = 20

    print(f"Device : {DEVICE}")
    print(f"Runs   : {total}  ({len(MODELS)} models × FD004 × {len(SEEDS)} seeds)")
    print(f"Config : MIN_EPOCHS={MIN_EPOCHS}  MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}")
    print(f"Op-cond: op_condition_utils (canonical, H6_OP_COLS={H6_OP_COLS})")
    print(f"Output : {OUT_CSV}")
    print("-" * 72)

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
            print(f"[{i:2d}/{total}] SKIP  {model} {ds} seed={seed}")
            continue

        print(f"[{i:2d}/{total}] RUN   {model} {ds} seed={seed} ...", end=" ", flush=True)
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

        new_df   = pd.DataFrame(results)
        combined = (pd.concat([done_df, new_df], ignore_index=True)
                    if not done_df.empty else new_df)
        combined.to_csv(OUT_CSV, index=False)

    print(f"\nAll done.  Results -> {OUT_CSV}")
    _print_summary(pd.read_csv(OUT_CSV))


if __name__ == "__main__":
    main()
