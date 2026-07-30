# -*- coding: utf-8 -*-
"""
H6 Phase 3 — Evaluate All Models
Loads per-seed prediction CSVs and computes RMSE / NASA Score mean +/- std
for M0, M1, M2, M3 on FD003 and FD004.
Saves model_comparison.csv.
"""

import numpy as np
import pandas as pd
import os

RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H6_fault_mode"
DATASETS    = ["FD003", "FD004"]
MODELS      = ["M0", "M1", "M2", "M3"]
SEEDS       = [0, 1, 2, 3, 4]


def nasa_score(pred, true):
    d   = np.asarray(pred) - np.asarray(true)
    pen = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(pen.sum())


def load_preds(model, dataset, seed):
    path = os.path.join(RESULTS_DIR,
                        f"raw_predictions_{model}_{dataset}_seed{seed}.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    return df["pred"].values, df["true"].values


def evaluate():
    records = []
    for dataset in DATASETS:
        for model in MODELS:
            rmse_l, ns_l = [], []
            for seed in SEEDS:
                pred, true = load_preds(model, dataset, seed)
                if pred is None:
                    print(f"  [MISSING] {model} {dataset} seed{seed}")
                    continue
                rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
                ns   = nasa_score(pred, true)
                rmse_l.append(rmse)
                ns_l.append(ns)

            if not rmse_l:
                continue
            rec = {
                "model":      model,
                "dataset":    dataset,
                "rmse_mean":  round(np.mean(rmse_l), 4),
                "rmse_std":   round(np.std(rmse_l), 4),
                "nasa_mean":  round(np.mean(ns_l), 2),
                "nasa_std":   round(np.std(ns_l), 2),
                "n_seeds":    len(rmse_l),
            }
            records.append(rec)
            print(f"  {model} {dataset}: RMSE={rec['rmse_mean']:.4f}+/-{rec['rmse_std']:.4f}  "
                  f"NASA={rec['nasa_mean']:.2f}+/-{rec['nasa_std']:.2f}")

    df_out = pd.DataFrame(records)
    out_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df_out


def print_improvement(df):
    """Check whether best model improves >= 10% over M0."""
    print("\n--- Improvement vs M0 (RMSE) ---")
    for dataset in DATASETS:
        sub  = df[df["dataset"] == dataset]
        m0   = sub[sub["model"] == "M0"]
        if m0.empty: continue
        m0_rmse = float(m0["rmse_mean"].values[0])
        for model in ["M1", "M2", "M3"]:
            row = sub[sub["model"] == model]
            if row.empty: continue
            rmse = float(row["rmse_mean"].values[0])
            imp  = (m0_rmse - rmse) / m0_rmse * 100.0
            verdict = "SUCCESS (>=10%)" if imp >= 10 else f"MISS ({imp:.1f}%)"
            print(f"  {dataset} {model} vs M0: {imp:+.1f}% -> {verdict}")


if __name__ == "__main__":
    print("=" * 60)
    print("H6 Phase 3 — Model Evaluation")
    print("=" * 60)
    df = evaluate()
    if not df.empty:
        print("\n--- Full Table ---")
        print(df.to_string(index=False))
        print_improvement(df)
    print("\n[DONE]")
