# -*- coding: utf-8 -*-
"""
H6 Phase 1 — Cluster Visualization
PCA 2D scatter plot of AB_full features coloured by GMM cluster for
FD003 and FD004.  Also plots sensor s15 and s7 mean profiles per cluster.
Saves fig_H6_P1_clusters.png to Results/H3_fault_mode/figures/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H3_fault_mode"
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e"]
MARKERS = ["o", "s"]


def load_bundle(dataset, variant="full"):
    pkl_path = os.path.join(MODELS_DIR, f"gmm_{dataset}_{variant}.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def plot_pca_clusters(ax, dataset):
    feat_path = os.path.join(RESULTS_DIR, f"features_{dataset}_AB_full.csv")
    df     = pd.read_csv(feat_path, index_col=0)
    bundle = load_bundle(dataset)
    gmm    = bundle["gmm"]
    scaler = bundle["scaler"]

    X_sc  = scaler.transform(df.values)
    labels = gmm.predict(X_sc)
    probs  = gmm.predict_proba(X_sc)

    pca   = PCA(n_components=2, random_state=42)
    X2    = pca.fit_transform(X_sc)
    var   = pca.explained_variance_ratio_

    for c in [0, 1]:
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=COLORS[c], marker=MARKERS[c],
                   label=f"Cluster {c} (n={mask.sum()})",
                   alpha=0.75, edgecolors="k", linewidths=0.3, s=65)

    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.set_title(f"{dataset.upper()} — AB-full GMM k=2 (PCA)")
    ax.legend(framealpha=0.9, fontsize=9)


def plot_sensor_profiles(ax, dataset, sensor):
    """Box-plot of per-engine slope for a sensor, grouped by cluster."""
    feat_path  = os.path.join(RESULTS_DIR, f"features_{dataset}_AB_full.csv")
    assign_path = os.path.join(RESULTS_DIR, f"cluster_assignments_{dataset}.csv")

    feat_df  = pd.read_csv(feat_path, index_col=0)
    assign   = pd.read_csv(assign_path, index_col=0)

    col_slope = f"{sensor}_slope"
    col_late  = f"{sensor}_late_mean"

    # Use late_mean if available, else slope
    col = col_late if col_late in feat_df.columns else col_slope
    if col not in feat_df.columns:
        ax.set_visible(False)
        return

    joined = feat_df[[col]].join(assign[["cluster"]])
    groups = [joined.loc[joined["cluster"] == c, col].values for c in [0, 1]]

    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color="k", linewidth=2))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(["Cluster 0", "Cluster 1"])
    ax.set_ylabel(col.replace("_", " "))
    ax.set_title(f"{dataset.upper()} — {col}")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("H6 Phase 1: Fault Mode Cluster Visualisation", fontsize=14, y=1.01)

    # Row 0: FD003
    plot_pca_clusters(axes[0, 0], "fd003")
    plot_sensor_profiles(axes[0, 1], "fd003", "s15")
    plot_sensor_profiles(axes[0, 2], "fd003", "s7")

    # Row 1: FD004
    plot_pca_clusters(axes[1, 0], "fd004")
    plot_sensor_profiles(axes[1, 1], "fd004", "s15")
    plot_sensor_profiles(axes[1, 2], "fd004", "s7")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_H6_P1_clusters.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
