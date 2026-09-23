# -*- coding: utf-8 -*-
"""
H6 Phase 3 — Uncertainty Quantification for M3 on FD003
========================================================
Two complementary UQ methods at 90% target coverage:
  1. MC Dropout      : 50 stochastic forward passes per engine
  2. Conformal Prediction : calibrated on the 20% validation-engine holdout

Metrics reported per seed and as mean±std over 5 seeds:
  RMSE  — point-prediction accuracy
  PICP  — Prediction Interval Coverage Probability  (target ≥ 0.90)
  MPIW  — Mean Prediction Interval Width            (lower is better)

Outputs:
  C:\BMAD_PY313\Data_Analysis\Results\H6_M3_FD003_UQ_results.csv
  C:\BMAD_PY313\Data_Analysis\Results\H6_M3_FD003_UQ_figure.png
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase2_models'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from h6_p2_model_utils import (
    set_seed, SEEDS, WINDOW_SIZE,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    SeqDatasetM3, AttentionGateModel,
    predict_m3, compute_metrics, load_checkpoint,
    RESULTS_DIR,
)

# ── Constants ────────────────────────────────────────────────────────────────
DATASET   = "FD003"
K_INIT    = 10       # GatingNet uses first K cycles
N_PASSES  = 50       # MC Dropout forward passes
ALPHA     = 0.10     # 1 – ALPHA = 90% target coverage

# Output directory: Data_Analysis/Results/  (parent of H3_fault_mode)
UQ_OUT_DIR = os.path.dirname(RESULTS_DIR)
os.makedirs(UQ_OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(UQ_OUT_DIR, "H6_M3_FD003_UQ_results.csv")
OUT_FIG = os.path.join(UQ_OUT_DIR, "H6_M3_FD003_UQ_figure.png")


# ── MC Dropout helpers ───────────────────────────────────────────────────────

def enable_mc_dropout(model: nn.Module) -> None:
    """Set model to eval (disables batchnorm, etc.) but keep Dropout in train mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _stochastic_pass_m3(model: nn.Module, X_np: np.ndarray,
                         device, K: int = K_INIT,
                         batch_size: int = 512) -> np.ndarray:
    """Single forward pass WITHOUT changing model mode (dropout stays as-is)."""
    ds     = SeqDatasetM3(X_np, np.zeros(len(X_np), dtype=np.float32), K)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds  = []
    with torch.no_grad():
        for xb_full, xb_init, _ in loader:
            y, _, _ = model(xb_full.to(device), xb_init.to(device))
            preds.append(y.cpu().numpy())
    return np.concatenate(preds).flatten()


def mc_dropout_inference(model: nn.Module, X_np: np.ndarray,
                          device, K: int = K_INIT,
                          n_passes: int = N_PASSES) -> np.ndarray:
    """
    Run n_passes stochastic forward passes with dropout enabled.
    Returns array of shape (n_passes, n_engines).
    """
    enable_mc_dropout(model)
    samples = [_stochastic_pass_m3(model, X_np, device, K) for _ in range(n_passes)]
    return np.array(samples)  # (n_passes, n_engines)


# ── Conformal Prediction helpers ─────────────────────────────────────────────

def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """
    Compute the marginal conformal quantile q̂ for (1-alpha) coverage.
    Formula: q̂ = ⌈(1-α)(n+1)⌉/n-th quantile of the calibration scores.
    """
    n     = len(scores)
    level = min(1.0, np.ceil((1.0 - alpha) * (n + 1)) / n)
    return float(np.quantile(scores, level))


# ── UQ metrics ───────────────────────────────────────────────────────────────

def uq_metrics(y_true: np.ndarray, mean_pred: np.ndarray,
               lower: np.ndarray, upper: np.ndarray):
    """Return (rmse, picp, mpiw)."""
    rmse = float(np.sqrt(np.mean((mean_pred - y_true) ** 2)))
    picp = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mpiw = float(np.mean(upper - lower))
    return rmse, picp, mpiw


# ── Model loading ─────────────────────────────────────────────────────────────

def load_m3_model(seed: int, n_features: int, device):
    """Load M3 checkpoint (read-only — does not modify saved files)."""
    ckpt  = load_checkpoint("M3", DATASET, seed)
    model = AttentionGateModel(n_features, K=K_INIT).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data_fd003():
    """
    Load and preprocess FD003 exactly as in training (no op-residualization needed).
    Returns:
        train_n      – normalised training DataFrame (with RUL column)
        X_te         – test sequences (n_te, 30, 15)
        y_te         – test RUL labels (n_te,)
        sensor_cols  – list of 15 feature column names
        min_v, max_v – normalisation parameters (fitted on train only)
    """
    train_df    = load_cmapss(DATASET, "train")
    sensor_cols = get_sensor_cols(DATASET)          # 15 features for FD003
    train_df    = add_rul(train_df)                 # clip=125
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n      = apply_normalization(train_df, sensor_cols, min_v, max_v)

    test_raw  = load_cmapss(DATASET, "test")
    test_n    = apply_normalization(test_raw, sensor_cols, min_v, max_v)
    rul_test  = load_test_rul(DATASET)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, rul_test)

    return train_n, X_te, y_te, sensor_cols, min_v, max_v


def get_calib_sequences(train_n: pd.DataFrame, sensor_cols: list):
    """
    Build last-window sequences from the 20% engine-level validation holdout.
    These mirror test-set conditions (last-30-cycle window per engine).
    Calibration RUL is the last cycle's clipped RUL in the training data.
    """
    # Reproduce the same engine split used during M3 training
    X_all, _, units_all = make_train_sequences(train_n, sensor_cols)
    _, va_m = split_engines(units_all, val_frac=0.2, seed=42)
    val_units = np.unique(units_all[va_m])

    n_feat  = len(sensor_cols)
    X_cal, y_cal = [], []
    for unit in val_units:
        grp  = train_n[train_n["unit_number"] == unit].sort_values("cycle")
        vals = grp[sensor_cols].values.astype(np.float32)
        rul  = grp["RUL"].values.astype(np.float32)
        if len(vals) >= WINDOW_SIZE:
            X_cal.append(vals[-WINDOW_SIZE:])
        else:
            pad = np.zeros((WINDOW_SIZE - len(vals), n_feat), dtype=np.float32)
            X_cal.append(np.vstack([pad, vals]))
        y_cal.append(rul[-1])   # RUL at the end of the engine's life

    return np.array(X_cal), np.array(y_cal)


# ── Per-seed UQ ───────────────────────────────────────────────────────────────

def run_uq_seed(seed: int,
                X_te: np.ndarray, y_te: np.ndarray,
                X_cal: np.ndarray, y_cal: np.ndarray,
                sensor_cols: list, device,
                store_arrays: bool = False) -> dict:
    """
    Run MC Dropout and Conformal Prediction for one trained M3 checkpoint.

    MC Dropout
    ----------
    - Enable dropout at inference time (keep Dropout layers in train mode).
    - 50 stochastic passes → 5th/95th percentile interval.
    - Mean prediction used for RMSE.

    Conformal Prediction
    --------------------
    - Deterministic predictions on calibration engines.
    - Non-conformity score: α_i = |ŷ_i – y_i|
    - q̂ = ⌈(1-α)(n+1)⌉/n-th quantile of calibration scores.
    - Symmetric test intervals: [ŷ – q̂, ŷ + q̂].
    """
    n_feat = len(sensor_cols)
    model  = load_m3_model(seed, n_feat, device)

    # ------------------------------------------------------------------ MC Dropout
    mc_samples = mc_dropout_inference(model, X_te, device)   # (50, n_te)
    mc_mean    = mc_samples.mean(axis=0)
    mc_std     = mc_samples.std(axis=0)
    mc_lower   = np.percentile(mc_samples,  5, axis=0)
    mc_upper   = np.percentile(mc_samples, 95, axis=0)

    mc_rmse, mc_picp, mc_mpiw = uq_metrics(y_te, mc_mean, mc_lower, mc_upper)

    # ------------------------------------------------------------------ Conformal
    # Deterministic calibration predictions (model.eval() is called inside predict_m3)
    cal_preds_det  = predict_m3(model, X_cal, device, K=K_INIT)
    nonconf_scores = np.abs(cal_preds_det - y_cal)
    q_hat          = conformal_quantile(nonconf_scores)

    test_preds_det = predict_m3(model, X_te, device, K=K_INIT)
    cp_lower       = test_preds_det - q_hat
    cp_upper       = test_preds_det + q_hat

    cp_rmse, cp_picp, cp_mpiw = uq_metrics(y_te, test_preds_det, cp_lower, cp_upper)

    result = dict(
        seed=seed,
        mc_rmse=mc_rmse, mc_picp=mc_picp, mc_mpiw=mc_mpiw,
        cp_rmse=cp_rmse, cp_picp=cp_picp, cp_mpiw=cp_mpiw,
        q_hat=q_hat,
    )

    if store_arrays:
        result.update(
            mc_mean=mc_mean, mc_std=mc_std, mc_lower=mc_lower, mc_upper=mc_upper,
            cp_mean=test_preds_det, cp_lower=cp_lower, cp_upper=cp_upper,
            nonconf_scores=nonconf_scores,
        )

    return result


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_uq(y_te: np.ndarray, seed0: dict, out_path: str) -> None:
    """
    Two-panel figure (seed=0):
      Left  – MC Dropout 90% PI (5th–95th percentile of 50 passes)
      Right – Conformal Prediction 90% PI
    Both panels: engines sorted by true RUL, green=covered, red=missed.
    """
    sort_idx = np.argsort(y_te)
    y_s      = y_te[sort_idx]
    idx      = np.arange(len(y_s))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    panels = [
        ("MC Dropout (90% PI, 50 passes)",
         seed0["mc_mean"][sort_idx],
         seed0["mc_lower"][sort_idx],
         seed0["mc_upper"][sort_idx]),
        ("Conformal Prediction (90% PI)",
         seed0["cp_mean"][sort_idx],
         seed0["cp_lower"][sort_idx],
         seed0["cp_upper"][sort_idx]),
    ]

    for ax, (title, mean_p, lo, hi) in zip(axes, panels):
        covered  = (y_s >= lo) & (y_s <= hi)
        picp_val = np.mean(covered)
        mpiw_val = np.mean(hi - lo)

        ax.fill_between(idx, lo, hi, alpha=0.25, color="steelblue", label="90% PI")
        ax.plot(idx, mean_p, color="steelblue", lw=1.5, label="Mean Prediction")
        ax.scatter(idx[covered],  y_s[covered],
                   s=18, c="green",  alpha=0.65, zorder=3, label=f"Covered ({covered.sum()})")
        ax.scatter(idx[~covered], y_s[~covered],
                   s=22, c="red",    alpha=0.85, zorder=4, label=f"Missed ({(~covered).sum()})")

        ax.set_title(f"{title}\nPICP={picp_val:.3f}  MPIW={mpiw_val:.1f}  (seed=0)",
                     fontsize=11)
        ax.set_xlabel("Test Engine Index (sorted by true RUL)", fontsize=10)
        ax.set_ylabel("RUL (cycles)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"M3 (Attention Gate) — Uncertainty Quantification on FD003 Test Set",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {out_path}")


def plot_calibration_scores(nonconf_scores: np.ndarray, q_hat: float,
                             out_path: str) -> None:
    """Histogram of non-conformity scores with q̂ marked (supplemental diagnostic)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(nonconf_scores, bins=15, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(q_hat, color="red", lw=2, linestyle="--",
               label=f"q̂ = {q_hat:.1f} cycles (90% coverage)")
    ax.set_xlabel("Non-conformity Score |ŷ – y| (cycles)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Conformal Calibration Scores — Validation Engines (seed=0)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Calibration histogram saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UQ] Device:   {device}")
    print(f"[UQ] Dataset:  {DATASET}")
    print(f"[UQ] n_passes: {N_PASSES}  |  target coverage: {int((1-ALPHA)*100)}%")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_n, X_te, y_te, sensor_cols, min_v, max_v = prepare_data_fd003()
    print(f"[UQ] Test set:   {X_te.shape[0]} engines, {X_te.shape[2]} features, window={X_te.shape[1]}")

    X_cal, y_cal = get_calib_sequences(train_n, sensor_cols)
    print(f"[UQ] Calib set: {len(X_cal)} validation engines (last-window, same regime as test)")

    # ── Per-seed loop ─────────────────────────────────────────────────────────
    all_results = []
    seed0_data  = None

    for seed in SEEDS:
        print(f"\n[UQ] ── Seed {seed} ──────────────────────────────")
        res = run_uq_seed(seed, X_te, y_te, X_cal, y_cal, sensor_cols, device,
                          store_arrays=(seed == 0))
        all_results.append(res)
        if seed == 0:
            seed0_data = res

        print(f"  MC Dropout  — RMSE={res['mc_rmse']:.4f}  "
              f"PICP={res['mc_picp']:.4f}  MPIW={res['mc_mpiw']:.2f}")
        print(f"  Conformal   — RMSE={res['cp_rmse']:.4f}  "
              f"PICP={res['cp_picp']:.4f}  MPIW={res['cp_mpiw']:.2f}  "
              f"q̂={res['q_hat']:.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    metric_keys = ["mc_rmse", "mc_picp", "mc_mpiw", "cp_rmse", "cp_picp", "cp_mpiw", "q_hat"]
    summary = {k: {"mean": float(np.mean([r[k] for r in all_results])),
                   "std":  float(np.std( [r[k] for r in all_results]))}
               for k in metric_keys}

    print("\n" + "=" * 65)
    print("[UQ] Summary — mean ± std over 5 seeds")
    print("-" * 65)
    print(f"  MC Dropout  : RMSE={summary['mc_rmse']['mean']:.4f}±{summary['mc_rmse']['std']:.4f}"
          f"  PICP={summary['mc_picp']['mean']:.4f}±{summary['mc_picp']['std']:.4f}"
          f"  MPIW={summary['mc_mpiw']['mean']:.2f}±{summary['mc_mpiw']['std']:.2f}")
    print(f"  Conformal   : RMSE={summary['cp_rmse']['mean']:.4f}±{summary['cp_rmse']['std']:.4f}"
          f"  PICP={summary['cp_picp']['mean']:.4f}±{summary['cp_picp']['std']:.4f}"
          f"  MPIW={summary['cp_mpiw']['mean']:.2f}±{summary['cp_mpiw']['std']:.2f}"
          f"  q̂={summary['q_hat']['mean']:.2f}±{summary['q_hat']['std']:.2f}")
    print("=" * 65)

    best_mc = "MC Dropout" if abs(summary['mc_picp']['mean'] - 0.90) < abs(summary['cp_picp']['mean'] - 0.90) \
              else "Conformal"
    print(f"  Closer to 90% PICP target: {best_mc}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    rows = []
    for res in all_results:
        for method, r_pfx, n_p in [("MC_Dropout", "mc_", N_PASSES),
                                    ("Conformal",  "cp_", 1)]:
            rows.append({
                "dataset":   DATASET,
                "model":     "M3_AttentionGate",
                "seed":      res["seed"],
                "method":    method,
                "n_passes":  n_p,
                "rmse":      round(res[r_pfx + "rmse"], 6),
                "picp":      round(res[r_pfx + "picp"], 6),
                "mpiw":      round(res[r_pfx + "mpiw"], 4),
                "q_hat":     round(res["q_hat"], 4) if method == "Conformal" else float("nan"),
                "alpha":     ALPHA,
                "n_calib":   len(X_cal),
            })

    # Append summary rows
    for method, r_pfx, n_p in [("MC_Dropout", "mc_", N_PASSES),
                                ("Conformal",  "cp_", 1)]:
        for stat in ["mean", "std"]:
            rows.append({
                "dataset":  DATASET,
                "model":    "M3_AttentionGate",
                "seed":     stat,
                "method":   method,
                "n_passes": n_p,
                "rmse":     round(summary[r_pfx + "rmse"][stat], 6),
                "picp":     round(summary[r_pfx + "picp"][stat], 6),
                "mpiw":     round(summary[r_pfx + "mpiw"][stat], 4),
                "q_hat":    round(summary["q_hat"][stat], 4) if method == "Conformal" else float("nan"),
                "alpha":    ALPHA,
                "n_calib":  len(X_cal),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[UQ] Results CSV saved: {OUT_CSV}")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_uq(y_te, seed0_data, OUT_FIG)

    calib_hist_path = os.path.join(UQ_OUT_DIR, "H6_M3_FD003_UQ_calib_hist.png")
    plot_calibration_scores(seed0_data["nonconf_scores"], seed0_data["q_hat"],
                            calib_hist_path)

    print("\n[UQ] Complete.")
    return summary


if __name__ == "__main__":
    main()
