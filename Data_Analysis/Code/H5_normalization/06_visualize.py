# -*- coding: utf-8 -*-
"""
06_visualize.py
---------------
Generate five figures for H2 normalisation comparison.

Figures saved to:
  Results/H5_normalization/figures/
    fig_H5_01_rmse_heatmap.png
    fig_H5_02_nasa_heatmap.png
    fig_H5_03_subgroup_rmse.png
    fig_H5_04_normalization_effect.png
    fig_H5_05_statistical_test.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR    = Path(__file__).parent
ROOT        = CODE_DIR.parent.parent
RESULTS_DIR = ROOT / "Results" / "H5_normalization"
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
NORM_IDS = ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]
NORM_LABELS = {
    "N1": "N1\nFleet MinMax",
    "N2": "N2\nFleet Std",
    "N3": "N3\nPU MinMax 5",
    "N4": "N4\nPU MinMax 10",
    "N5": "N5\nPU Std 5",
    "N6": "N6\nPU Std 10",
    "N7": "N7\nRevIN",
}
GROUPS = ["boundary", "medium", "long"]

PALETTE = sns.color_palette("tab10", n_colors=7)
NORM_COLORS = dict(zip(NORM_IDS, PALETTE))


# ---------------------------------------------------------------------------
# Load CSVs
# ---------------------------------------------------------------------------

def load_metrics():
    path = RESULTS_DIR / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run 05_evaluate.py first: {path}")
    return pd.read_csv(path)


def load_subgroup():
    path = RESULTS_DIR / "subgroup_rmse.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_stat():
    path = RESULTS_DIR / "statistical_test.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Fig 1 — RMSE heatmap
# ---------------------------------------------------------------------------

def fig_rmse_heatmap(metrics: pd.DataFrame):
    pivot = metrics.pivot(index="norm_id", columns="dataset",
                          values="rmse_mean").reindex(index=NORM_IDS,
                                                      columns=DATASETS)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "RMSE (cycles)"},
    )
    ax.set_title("H2 — RMSE by Normalizer × Dataset (mean over 5 seeds)",
                 fontsize=12, pad=10)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("Normalizer", fontsize=10)
    ax.set_yticklabels(
        [NORM_LABELS.get(nid, nid) for nid in NORM_IDS],
        rotation=0, fontsize=8,
    )
    _save(fig, "fig_H5_01_rmse_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 2 — NASA Score heatmap
# ---------------------------------------------------------------------------

def fig_nasa_heatmap(metrics: pd.DataFrame):
    pivot = metrics.pivot(index="norm_id", columns="dataset",
                          values="score_mean").reindex(index=NORM_IDS,
                                                       columns=DATASETS)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "NASA Score (lower = better)"},
    )
    ax.set_title("H2 — NASA Prognostic Score by Normalizer × Dataset",
                 fontsize=12, pad=10)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("Normalizer", fontsize=10)
    ax.set_yticklabels(
        [NORM_LABELS.get(nid, nid) for nid in NORM_IDS],
        rotation=0, fontsize=8,
    )
    _save(fig, "fig_H5_02_nasa_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 3 — Subgroup RMSE by lifetime group
# ---------------------------------------------------------------------------

def fig_subgroup_rmse(subgroup: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for ax, grp in zip(axes, GROUPS):
        sub = subgroup[subgroup["group"] == grp]
        if sub.empty:
            ax.set_title(f"{grp.capitalize()} (no data)")
            continue

        pivot = sub.pivot_table(
            index="norm_id", columns="dataset",
            values="rmse_mean", aggfunc="mean"
        ).reindex(index=NORM_IDS, columns=DATASETS)

        pivot.plot(kind="bar", ax=ax, colormap="tab10", width=0.7)
        ax.set_title(f"Lifetime group: {grp.capitalize()}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("RMSE (cycles)" if grp == GROUPS[0] else "")
        ax.set_xticklabels(NORM_IDS, rotation=0, fontsize=8)
        ax.legend(title="Dataset", fontsize=7, title_fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("H2 — Subgroup RMSE by Lifetime Group", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, "fig_H5_03_subgroup_rmse.png")


# ---------------------------------------------------------------------------
# Fig 4 — N1 vs N3 vs N7 RMSE comparison
# ---------------------------------------------------------------------------

def fig_normalization_effect(metrics: pd.DataFrame):
    compare_ids = ["N1", "N3", "N7"]
    labels      = ["N1 (Fleet MinMax)", "N3 (PU MinMax 5)", "N7 (RevIN)"]
    sub = metrics[metrics["norm_id"].isin(compare_ids)].copy()

    x       = np.arange(len(DATASETS))
    width   = 0.22
    offsets = np.linspace(-(len(compare_ids) - 1) * width / 2,
                           (len(compare_ids) - 1) * width / 2,
                           len(compare_ids))

    fig, ax = plt.subplots(figsize=(9, 5))

    for nid, label, offset in zip(compare_ids, labels, offsets):
        row = sub[sub["norm_id"] == nid].set_index("dataset")
        means = [row.loc[ds, "rmse_mean"] if ds in row.index else np.nan
                 for ds in DATASETS]
        stds  = [row.loc[ds, "rmse_std"]  if ds in row.index else np.nan
                 for ds in DATASETS]
        ax.bar(x + offset, means, width, yerr=stds, label=label,
               capsize=4, alpha=0.85, color=NORM_COLORS[nid])

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=10)
    ax.set_ylabel("RMSE (cycles)", fontsize=10)
    ax.set_title("H2 — Normalization Effect: N1 vs N3 vs N7", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_H5_04_normalization_effect.png")


# ---------------------------------------------------------------------------
# Fig 5 — p-values after BH-FDR
# ---------------------------------------------------------------------------

def fig_statistical_test(stat: pd.DataFrame):
    comparators = ["N2", "N3", "N4", "N5", "N6", "N7"]
    labels      = [NORM_LABELS.get(n, n).replace("\n", " ") for n in comparators]

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 4), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        sub = stat[stat["dataset"] == ds].set_index("norm_id")
        pvals = [sub.loc[nid, "p_bh"] if nid in sub.index else np.nan
                 for nid in comparators]
        colors = []
        for p in pvals:
            if np.isnan(p):
                colors.append("lightgray")
            elif p < 0.05:
                colors.append("#2ca02c")   # significant: green
            else:
                colors.append("#d62728")   # not significant: red

        bars = ax.bar(range(len(comparators)), pvals, color=colors, alpha=0.85)
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1,
                   label="α=0.05")
        ax.set_title(ds, fontsize=10)
        ax.set_xticks(range(len(comparators)))
        ax.set_xticklabels(comparators, rotation=0, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if ds == DATASETS[0]:
            ax.set_ylabel("p-value (BH-FDR adjusted)", fontsize=9)

    # Legend patches — placed inside last subplot (FD004 bars are near 0, top area free)
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#2ca02c", label="p_BH < 0.05 (significant)"),
        Patch(facecolor="#d62728", label="p_BH ≥ 0.05"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="α=0.05"),
    ]
    axes[-1].legend(handles=legend_elems, loc="upper right", fontsize=8, framealpha=0.9)
    # Title embedded via fig.text for precise placement above tight_layout area
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.text(0.5, 0.97, "Fig. 5 — BH-FDR Adjusted p-values vs N1 Baseline (H2)",
             ha="center", va="top", fontsize=12, fontweight="bold",
             transform=fig.transFigure)
    _save(fig, "fig_H5_05_statistical_test.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    metrics  = load_metrics()
    subgroup = load_subgroup()
    stat     = load_stat()

    fig_rmse_heatmap(metrics)
    fig_nasa_heatmap(metrics)

    if subgroup is not None and not subgroup.empty:
        fig_subgroup_rmse(subgroup)
    else:
        print("Subgroup data not available; skipping fig 3.")

    fig_normalization_effect(metrics)

    if stat is not None and not stat.empty:
        fig_statistical_test(stat)
    else:
        print("Statistical test data not available; skipping fig 5.")

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
