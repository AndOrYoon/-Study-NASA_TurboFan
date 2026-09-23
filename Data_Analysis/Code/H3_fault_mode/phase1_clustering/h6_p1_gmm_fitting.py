# -*- coding: utf-8 -*-
"""
H6 Phase 1 - GMM Fitting
Fits Gaussian Mixture Model (k=2) on AB_full, AB_slope, AB_late feature vectors
for FD003 and FD004. Saves GMM pkl files and ablation_silhouette.csv.
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H3_fault_mode"
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

VARIANTS = ["AB_full", "AB_slope", "AB_late"]
DATASETS = ["fd003", "fd004"]


def fit_gmm(X_scaled, n_components=2, n_init=20, random_state=42):
    """Fit GMM and return (model, labels, silhouette, BIC)."""
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        n_init=n_init,
        max_iter=300,
    )
    labels = gmm.fit_predict(X_scaled)
    n_unique = len(set(labels))
    sil = silhouette_score(X_scaled, labels) if n_unique > 1 else 0.0
    bic = gmm.bic(X_scaled)
    return gmm, labels, sil, bic


def main():
    records = []

    for dataset in DATASETS:
        for variant in VARIANTS:
            feat_path = os.path.join(RESULTS_DIR, f"features_{dataset}_{variant}.csv")
            if not os.path.exists(feat_path):
                print(f"  [SKIP] Not found: {feat_path}")
                continue

            df = pd.read_csv(feat_path, index_col=0)
            print(f"\n[{dataset.upper()} / {variant}] shape={df.shape}")

            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(df.values)

            gmm, labels, sil, bic = fit_gmm(X_scaled)

            sizes = pd.Series(labels).value_counts().sort_index().to_dict()
            print(f"  Silhouette={sil:.4f}  BIC={bic:.2f}  cluster_sizes={sizes}")

            # Save GMM pkl only for AB_full and AB_slope (used by Phase 2)
            if variant in ("AB_full", "AB_slope"):
                gmm_key  = "full" if variant == "AB_full" else "slope"
                pkl_name = f"gmm_{dataset}_{gmm_key}.pkl"
                pkl_path = os.path.join(MODELS_DIR, pkl_name)
                bundle = {
                    "gmm":          gmm,
                    "scaler":       scaler,
                    "feature_cols": list(df.columns),
                }
                with open(pkl_path, "wb") as f:
                    pickle.dump(bundle, f)
                print(f"  Saved: {pkl_path}")

            records.append({
                "dataset":   dataset,
                "variant":   variant,
                "silhouette": round(float(sil), 4),
                "bic_k2":    round(float(bic), 2),
            })

    abl_df  = pd.DataFrame(records)
    out_path = os.path.join(RESULTS_DIR, "ablation_silhouette.csv")
    abl_df.to_csv(out_path, index=False)
    print(f"\nSaved ablation_silhouette.csv")
    print(abl_df.to_string(index=False))


if __name__ == "__main__":
    print("=" * 60)
    print("H6 Phase 1 — GMM Fitting")
    print("=" * 60)
    main()
    print("\n[DONE]")
