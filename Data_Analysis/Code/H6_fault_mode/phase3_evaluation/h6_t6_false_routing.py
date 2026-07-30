# -*- coding: utf-8 -*-
"""
T6 — False Routing Sensitivity Analysis for M3 Attention-Gate
Loads saved M3 checkpoints and evaluates RMSE under four routing conditions:
  (1) Normal   : GatingNet produces gate weights w0, w1
  (2) Flipped  : gate weights swapped (w0 <-> w1) — 100% mis-routing
  (3) Branch-0 : all engines forced to branch 0 (w0=1, w1=0)
  (4) Branch-1 : all engines forced to branch 1 (w0=0, w1=1)
Also records per-engine gate confidence = max(w0, w1).
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase2_models'))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from h6_p2_model_utils import (
    RESULTS_DIR, MODELS_DIR, SEEDS,
    set_seed, load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_op_residual_fd004, apply_op_residual_fd004,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences,
    AttentionGateModel, SeqDatasetM3,
    compute_metrics,
)

K_INIT   = 10
DATASETS = ["FD003", "FD004"]
OUT_DIR  = os.path.join(RESULTS_DIR, "T6_false_routing")
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def predict_m3_variants(model, X_np: np.ndarray, device, K: int = 10, batch_size: int = 512):
    """
    Run M3 inference and return four prediction arrays + gate weights.
    Returns dict with keys: normal, flipped, branch0, branch1, w0, w1
    """
    model.eval()
    ds     = SeqDatasetM3(X_np, np.zeros(len(X_np)), K)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    p_normal, p_flipped, p_b0, p_b1 = [], [], [], []
    g_w0, g_w1 = [], []

    for xb_full, xb_init, _ in loader:
        xb_full = xb_full.to(device)
        xb_init = xb_init.to(device)

        w = model.gating(xb_init)          # (B,2)
        w0 = w[:, 0:1]
        w1 = w[:, 1:2]
        y0 = model.branch0(xb_full)        # (B,1)
        y1 = model.branch1(xb_full)        # (B,1)

        p_normal.append((w0 * y0 + w1 * y1).cpu().numpy())
        p_flipped.append((w1 * y0 + w0 * y1).cpu().numpy())
        p_b0.append(y0.cpu().numpy())
        p_b1.append(y1.cpu().numpy())
        g_w0.append(w0.cpu().numpy())
        g_w1.append(w1.cpu().numpy())

    return {
        "normal":  np.concatenate(p_normal).flatten(),
        "flipped": np.concatenate(p_flipped).flatten(),
        "branch0": np.concatenate(p_b0).flatten(),
        "branch1": np.concatenate(p_b1).flatten(),
        "w0":      np.concatenate(g_w0).flatten(),
        "w1":      np.concatenate(g_w1).flatten(),
    }


def run_dataset(dataset: str):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds   = SEEDS  # [0,1,2,3,4]

    rows = []
    all_engine_rows = []

    for seed in seeds:
        set_seed(seed)

        # ── Preprocessing (must match training exactly) ──────────────────────
        train_df    = load_cmapss(dataset, "train")
        sensor_cols = get_sensor_cols(dataset)

        sc_op = km_op = cm_op = None
        if dataset == "FD004":
            sc_op, km_op, cm_op = fit_op_residual_fd004(train_df, sensor_cols)
            train_df = apply_op_residual_fd004(train_df, sc_op, km_op, cm_op, sensor_cols)

        train_df = add_rul(train_df)
        min_v, max_v = fit_normalization(train_df, sensor_cols)

        test_raw = load_cmapss(dataset, "test")
        if dataset == "FD004" and sc_op is not None:
            test_raw = apply_op_residual_fd004(test_raw, sc_op, km_op, cm_op, sensor_cols)
        test_n = apply_normalization(test_raw, sensor_cols, min_v, max_v)
        X_te, y_te = make_test_sequences(test_n, sensor_cols, load_test_rul(dataset))

        # ── Load checkpoint ──────────────────────────────────────────────────
        ckpt_path = os.path.join(MODELS_DIR, f"M3_{dataset}_seed{seed}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            continue
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        model = AttentionGateModel(len(sensor_cols), K=K_INIT).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        # ── Four-condition inference ─────────────────────────────────────────
        out = predict_m3_variants(model, X_te, device, K=K_INIT)

        rmse_normal,  ns_normal  = compute_metrics(out["normal"],  y_te)
        rmse_flipped, ns_flipped = compute_metrics(out["flipped"], y_te)
        rmse_b0,      ns_b0      = compute_metrics(out["branch0"], y_te)
        rmse_b1,      ns_b1      = compute_metrics(out["branch1"], y_te)

        gate_conf  = np.maximum(out["w0"], out["w1"])  # max(w0,w1) per engine
        conf_mean  = float(gate_conf.mean())
        conf_std   = float(gate_conf.std())
        # fraction with high confidence (> 0.8)
        frac_hi    = float((gate_conf > 0.8).mean())

        delta_rmse = rmse_flipped - rmse_normal
        worst_rmse = max(rmse_b0, rmse_b1)
        best_branch_rmse = min(rmse_b0, rmse_b1)

        print(f"  [{dataset} seed={seed}]"
              f"  normal={rmse_normal:.2f}"
              f"  flipped={rmse_flipped:.2f} (Δ={delta_rmse:+.2f})"
              f"  b0={rmse_b0:.2f}  b1={rmse_b1:.2f}"
              f"  conf={conf_mean:.3f}±{conf_std:.3f}"
              f"  hi_conf={frac_hi:.1%}")

        rows.append({
            "dataset":          dataset,
            "seed":             seed,
            "rmse_normal":      round(rmse_normal,  4),
            "rmse_flipped":     round(rmse_flipped, 4),
            "rmse_branch0":     round(rmse_b0,      4),
            "rmse_branch1":     round(rmse_b1,      4),
            "delta_rmse_flip":  round(delta_rmse,   4),
            "worst_branch_rmse":round(worst_rmse,   4),
            "best_branch_rmse": round(best_branch_rmse, 4),
            "nasa_normal":      round(ns_normal,  2),
            "nasa_flipped":     round(ns_flipped, 2),
            "gate_conf_mean":   round(conf_mean,  4),
            "gate_conf_std":    round(conf_std,   4),
            "gate_conf_frac_hi":round(frac_hi,   4),
        })

        # per-engine breakdown for detailed analysis
        for i in range(len(y_te)):
            all_engine_rows.append({
                "dataset":   dataset,
                "seed":      seed,
                "engine":    i + 1,
                "true_rul":  y_te[i],
                "pred_normal":  out["normal"][i],
                "pred_flipped": out["flipped"][i],
                "pred_branch0": out["branch0"][i],
                "pred_branch1": out["branch1"][i],
                "w0":           out["w0"][i],
                "w1":           out["w1"][i],
                "gate_conf":    gate_conf[i],
            })

    return rows, all_engine_rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("dataset")
    summary_rows = []
    for dataset, g in grp:
        summary_rows.append({
            "dataset":                    dataset,
            "rmse_normal_mean":           round(g["rmse_normal"].mean(),      4),
            "rmse_normal_std":            round(g["rmse_normal"].std(),       4),
            "rmse_flipped_mean":          round(g["rmse_flipped"].mean(),     4),
            "rmse_flipped_std":           round(g["rmse_flipped"].std(),      4),
            "delta_rmse_flip_mean":       round(g["delta_rmse_flip"].mean(),  4),
            "delta_rmse_flip_std":        round(g["delta_rmse_flip"].std(),   4),
            "rmse_branch0_mean":          round(g["rmse_branch0"].mean(),     4),
            "rmse_branch1_mean":          round(g["rmse_branch1"].mean(),     4),
            "worst_branch_rmse_mean":     round(g["worst_branch_rmse"].mean(),4),
            "best_branch_rmse_mean":      round(g["best_branch_rmse"].mean(), 4),
            "gate_conf_mean":             round(g["gate_conf_mean"].mean(),   4),
            "gate_conf_frac_hi_mean":     round(g["gate_conf_frac_hi"].mean(),4),
            "nasa_normal_mean":           round(g["nasa_normal"].mean(),      2),
            "nasa_flipped_mean":          round(g["nasa_flipped"].mean(),     2),
        })
    return pd.DataFrame(summary_rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[T6] Device: {device}\n")

    all_rows   = []
    all_engine = []

    for dataset in DATASETS:
        print(f"\n{'='*55}\nDataset: {dataset}\n{'='*55}")
        rows, eng_rows = run_dataset(dataset)
        all_rows.extend(rows)
        all_engine.extend(eng_rows)

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(os.path.join(OUT_DIR, "T6_raw_results.csv"), index=False)
    print(f"\n[T6] Raw results saved.")

    df_eng = pd.DataFrame(all_engine)
    df_eng.to_csv(os.path.join(OUT_DIR, "T6_per_engine_results.csv"), index=False)
    print(f"[T6] Per-engine results saved.")

    df_sum = summarise(df_raw)
    df_sum.to_csv(os.path.join(OUT_DIR, "T6_summary.csv"), index=False)
    print(f"[T6] Summary saved.\n")

    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    for _, r in df_sum.iterrows():
        ds = r["dataset"]
        print(f"\n{ds}:")
        print(f"  Normal   RMSE : {r['rmse_normal_mean']:.2f} ± {r['rmse_normal_std']:.2f}")
        print(f"  Flipped  RMSE : {r['rmse_flipped_mean']:.2f} ± {r['rmse_flipped_std']:.2f}"
              f"  (Δ = {r['delta_rmse_flip_mean']:+.2f} ± {r['delta_rmse_flip_std']:.2f})")
        print(f"  Branch-0 RMSE : {r['rmse_branch0_mean']:.2f}")
        print(f"  Branch-1 RMSE : {r['rmse_branch1_mean']:.2f}")
        print(f"  Worst branch  : {r['worst_branch_rmse_mean']:.2f}")
        print(f"  Gate conf     : {r['gate_conf_mean']:.3f}  "
              f"high-conf(>0.8): {r['gate_conf_frac_hi_mean']:.1%}")
        print(f"  NASA normal   : {r['nasa_normal_mean']:.1f}  "
              f"flipped: {r['nasa_flipped_mean']:.1f}")

    print(f"\n[T6] Complete. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
