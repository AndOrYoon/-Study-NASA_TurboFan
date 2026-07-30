# -*- coding: utf-8 -*-
"""
H6 Phase 3 — Generate Figures
fig_H6_01_model_comparison.png  — RMSE bar chart M0/M1/M2/M3 for FD003 & FD004
fig_H6_02_ablation_silhouette.png — Silhouette comparison by variant
fig_H6_03_cluster_profiles.png  — Sensor s15 & s7 mean trend by cluster
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H6_fault_mode"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DATASET_DIR = r"C:\BMAD_PY313\Dataset"
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS   = ["M0", "M1", "M2", "M3"]
DATASETS = ["FD003", "FD004"]
COLORS   = {"M0": "#7f7f7f", "M1": "#1f77b4", "M2": "#ff7f0e", "M3": "#2ca02c"}
COL_NAMES = (["unit_number","cycle","op_setting_1","op_setting_2","op_setting_3"]
             + [f"s{i}" for i in range(1,22)])


# ---- Figure 1: Model comparison RMSE bar chart ---------------------------

def fig_model_comparison():
    cmp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if not os.path.exists(cmp_path):
        print(f"  [SKIP] {cmp_path} not found")
        return

    df  = pd.read_csv(cmp_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle("H6: Multi-Branch LSTM — RMSE by Model", fontsize=14)

    for ax, dataset in zip(axes, DATASETS):
        sub  = df[df["dataset"] == dataset]
        x    = np.arange(len(MODELS))
        bars = []
        errs = []
        for m in MODELS:
            row = sub[sub["model"] == m]
            if row.empty:
                bars.append(0.0); errs.append(0.0)
            else:
                bars.append(float(row["rmse_mean"].values[0]))
                errs.append(float(row["rmse_std"].values[0]))

        bp = ax.bar(x, bars, yerr=errs, capsize=5, width=0.55,
                    color=[COLORS[m] for m in MODELS], alpha=0.85, edgecolor="k")
        ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=11)
        ax.set_ylabel("RMSE (cycles)", fontsize=11)
        ax.set_title(f"{dataset}", fontsize=12)
        for bar, val in zip(bp, bars):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_H6_01_model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


# ---- Figure 2: Silhouette ablation ----------------------------------------

def fig_ablation_silhouette():
    sil_path = os.path.join(RESULTS_DIR, "ablation_silhouette.csv")
    if not os.path.exists(sil_path):
        print(f"  [SKIP] {sil_path} not found")
        return

    df  = pd.read_csv(sil_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    fig.suptitle("H6: Clustering Silhouette by Feature Variant", fontsize=13)
    variants = df["variant"].unique()
    v_colors = {"AB_full": "#1f77b4", "AB_slope": "#ff7f0e", "AB_late": "#2ca02c"}

    for ax, dataset in zip(axes, DATASETS):
        sub = df[df["dataset"] == dataset]
        x   = np.arange(len(variants))
        vals = [float(sub[sub["variant"]==v]["silhouette"].values[0])
                if not sub[sub["variant"]==v].empty else 0.0 for v in variants]
        colors = [v_colors.get(v, "#888888") for v in variants]
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="k", width=0.55)
        ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=9)
        ax.set_ylabel("Silhouette")
        ax.set_title(dataset)
        ax.axhline(0.5, ls="--", color="red", alpha=0.6, label="Sil=0.5 threshold")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_H6_02_ablation_silhouette.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


# ---- Figure 3: Sensor s15 & s7 profiles by cluster -----------------------

def fig_cluster_profiles():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("H6: Sensor Mean Trajectories by Cluster", fontsize=13)

    for row_idx, dataset in enumerate(DATASETS):
        assign_path = os.path.join(RESULTS_DIR, f"cluster_assignments_{dataset.lower()}.csv")
        train_path  = os.path.join(DATASET_DIR, f"train_{dataset}.txt")

        if not os.path.exists(assign_path) or not os.path.exists(train_path):
            print(f"  [SKIP] Missing files for {dataset}")
            continue

        assign = pd.read_csv(assign_path, index_col=0)
        train  = pd.read_csv(train_path, sep=r"\s+", header=None, names=COL_NAMES)
        train  = train.merge(assign[["cluster"]], left_on="unit_number", right_index=True,
                             how="left")

        for col_idx, sensor in enumerate(["s15", "s7"]):
            ax = axes[row_idx, col_idx]
            if sensor not in train.columns:
                ax.set_visible(False); continue

            for c, color in enumerate(["#1f77b4", "#ff7f0e"]):
                eng_ids = assign[assign["cluster"] == c].index.tolist()
                grp_data = train[train["unit_number"].isin(eng_ids)]
                # Normalize cycle to [0,1] per engine and compute mean at 20 buckets
                for unit, g in list(grp_data.groupby("unit_number"))[:30]:
                    max_cyc = g["cycle"].max()
                    pct     = g["cycle"] / max_cyc
                    ax.plot(pct, g[sensor], color=color, alpha=0.12, linewidth=0.5)
                # Mean trend
                grp_data = grp_data.copy()
                grp_data["_pct"] = grp_data.groupby("unit_number")["cycle"].transform(
                    lambda x: x / x.max())
                grp_data["_bin"] = (grp_data["_pct"] * 20).astype(int).clip(0, 19)
                mean_tr = grp_data.groupby("_bin")[sensor].mean()
                ax.plot(mean_tr.index / 20, mean_tr.values,
                        color=color, linewidth=2.2, label=f"Cluster {c} mean")

            ax.set_xlabel("Normalized cycle (0=start, 1=end)")
            ax.set_ylabel(f"{sensor} value")
            ax.set_title(f"{dataset} — {sensor}")
            ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_H6_03_cluster_profiles.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("H6 Phase 3 — Generating Figures")
    print("=" * 60)
    fig_model_comparison()
    fig_ablation_silhouette()
    fig_cluster_profiles()
    print("\n[DONE]")
