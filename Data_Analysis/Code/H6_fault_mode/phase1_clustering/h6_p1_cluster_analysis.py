# -*- coding: utf-8 -*-
"""
H6 Phase 1 — Cluster Analysis
Prints a Silhouette/BIC comparison table for AB_full, AB_slope, AB_late
variants on FD003 and FD004, and reports robustness checks.
"""

import pandas as pd
import os

RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H6_fault_mode"


def main():
    abl_path = os.path.join(RESULTS_DIR, "ablation_silhouette.csv")
    if not os.path.exists(abl_path):
        raise FileNotFoundError(f"Run h6_p1_gmm_fitting.py first: {abl_path}")

    df = pd.read_csv(abl_path)

    print("\n" + "=" * 68)
    print("H6 P1 — Silhouette & BIC: AB-full / AB-slope / AB-late Comparison")
    print("=" * 68)

    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset].copy()
        print(f"\nDataset: {dataset.upper()}")
        print(f"  {'Variant':<12}  {'Silhouette':>12}  {'BIC(k=2)':>14}")
        print("  " + "-" * 44)
        for _, row in sub.iterrows():
            print(f"  {row['variant']:<12}  {row['silhouette']:>12.4f}  {row['bic_k2']:>14.2f}")

    print("\n--- Robustness Check: AB_slope Silhouette >= AB_full × 0.7 ---")
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        full_row  = sub[sub["variant"] == "AB_full"]
        slope_row = sub[sub["variant"] == "AB_slope"]
        if full_row.empty or slope_row.empty:
            continue
        sil_full  = float(full_row["silhouette"].values[0])
        sil_slope = float(slope_row["silhouette"].values[0])
        threshold = sil_full * 0.7
        verdict   = "PASS" if sil_slope >= threshold else "FAIL"
        print(f"  {dataset.upper()}: AB_slope({sil_slope:.4f}) >= "
              f"AB_full×0.7({threshold:.4f}) → {verdict}")

    print("\n--- FD004 Clustering Quality Check (Silhouette > 0.5?) ---")
    for dataset in ["fd004"]:
        sub = df[(df["dataset"] == dataset) & (df["variant"] == "AB_full")]
        if sub.empty:
            continue
        sil = float(sub["silhouette"].values[0])
        verdict = "PASS" if sil > 0.5 else "FAIL (weak clustering)"
        print(f"  {dataset.upper()} AB_full Silhouette={sil:.4f} → {verdict}")


if __name__ == "__main__":
    main()
