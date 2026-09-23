# -*- coding: utf-8 -*-
"""
H6 Phase 3 — Ablation Study
Compares AB_full vs AB_slope clustering quality (Silhouette),
and M1_full (default) vs M1_slope RMSE (if M1_slope predictions exist).
Saves ablation_results.csv.
"""

import numpy as np
import pandas as pd
import os

RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H3_fault_mode"
DATASETS    = ["FD003", "FD004"]
SEEDS       = [0, 1, 2, 3, 4]


def load_preds(model, dataset, seed):
    path = os.path.join(RESULTS_DIR,
                        f"raw_predictions_{model}_{dataset}_seed{seed}.csv")
    if not os.path.exists(path): return None, None
    df = pd.read_csv(path)
    return df["pred"].values, df["true"].values


def nasa_score(pred, true):
    d = np.asarray(pred) - np.asarray(true)
    return float(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1).sum())


def main():
    records = []

    # ---- 1. Silhouette ablation ----------------------------------------
    sil_path = os.path.join(RESULTS_DIR, "ablation_silhouette.csv")
    if os.path.exists(sil_path):
        sil_df = pd.read_csv(sil_path)
        print("\n--- Silhouette Ablation ---")
        print(sil_df.to_string(index=False))

        for dataset in DATASETS:
            sub = sil_df[sil_df["dataset"] == dataset]
            for _, row in sub.iterrows():
                records.append({
                    "type":    "silhouette",
                    "dataset": dataset,
                    "variant": row["variant"],
                    "metric":  "silhouette",
                    "value":   row["silhouette"],
                })
                records.append({
                    "type":    "silhouette",
                    "dataset": dataset,
                    "variant": row["variant"],
                    "metric":  "bic_k2",
                    "value":   row["bic_k2"],
                })

    # ---- 2. M1 AB_full vs AB_slope RMSE (if slope predictions exist) ---
    print("\n--- M1 RMSE: AB_full vs AB_slope ---")
    for dataset in DATASETS:
        for variant in ["M1", "M1_slope"]:
            rmse_l = []
            for seed in SEEDS:
                pred, true = load_preds(variant, dataset, seed)
                if pred is None: continue
                rmse_l.append(float(np.sqrt(np.mean((pred - true)**2))))
            if rmse_l:
                mean_rmse = np.mean(rmse_l)
                std_rmse  = np.std(rmse_l)
                print(f"  {dataset} {variant}: RMSE={mean_rmse:.4f}+/-{std_rmse:.4f}")
                records.append({
                    "type":    "rmse_ablation",
                    "dataset": dataset,
                    "variant": variant,
                    "metric":  "rmse_mean",
                    "value":   round(mean_rmse, 4),
                })
                records.append({
                    "type":    "rmse_ablation",
                    "dataset": dataset,
                    "variant": variant,
                    "metric":  "rmse_std",
                    "value":   round(std_rmse, 4),
                })

    # ---- 3. Robustness check -------------------------------------------
    if os.path.exists(sil_path):
        sil_df = pd.read_csv(sil_path)
        print("\n--- Robustness: AB_slope Sil >= AB_full * 0.7 ---")
        for dataset in DATASETS:
            sub = sil_df[sil_df["dataset"] == dataset]
            full_row  = sub[sub["variant"] == "AB_full"]
            slope_row = sub[sub["variant"] == "AB_slope"]
            if full_row.empty or slope_row.empty: continue
            sf = float(full_row["silhouette"].values[0])
            ss = float(slope_row["silhouette"].values[0])
            thr = sf * 0.7
            print(f"  {dataset}: {ss:.4f} >= {thr:.4f} [{sf:.4f}*0.7] "
                  f"-> {'PASS' if ss >= thr else 'FAIL'}")

    # Save
    out_df   = pd.DataFrame(records)
    out_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("H6 Phase 3 — Ablation Study")
    print("=" * 60)
    main()
    print("\n[DONE]")
