"""
H4 Phase 8 — Visualization
Generates 6 figures from Phase 2b results.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from config_h7 import RESULT_DIR, FIGURES_DIR, LOSS_NAMES, DATASETS

MATRIX_CSV = os.path.join(RESULT_DIR, "phase2b_results_matrix.csv")
RESS_DPI = 300
WONG_7 = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442']

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'grid.color': '#cccccc',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.framealpha': 0.85,
})

print("=" * 60)
print("H4 Visualization")
print("=" * 60)

if not os.path.exists(MATRIX_CSV):
    raise FileNotFoundError(f"Phase 2b matrix not found: {MATRIX_CSV}")

matrix = pd.read_csv(MATRIX_CSV)

CLIP_ORDER = ["clip_100", "clip_125", "clip_130", "clip_none"]
LOSS_ORDER = LOSS_NAMES  # L1..L7

# ─────────────────────────────────────────────────────────────────────────────
# Helper: NASA Score heatmap for one dataset
# ─────────────────────────────────────────────────────────────────────────────
def plot_nasa_heatmap(dataset: str, ax=None, title_prefix=""):
    sub = matrix[matrix["dataset"] == dataset]
    pivot = (sub.groupby(["clip", "loss_fn"])["nasa_mean"]
                .mean()
                .unstack("loss_fn")
                .reindex(index=CLIP_ORDER, columns=LOSS_ORDER))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    vmin, vmax = pivot.values[~np.isnan(pivot.values)].min(), \
                 pivot.values[~np.isnan(pivot.values)].max()

    im = ax.imshow(pivot.values, aspect="auto",
                   cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(LOSS_ORDER)))
    ax.set_xticklabels(LOSS_ORDER, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(CLIP_ORDER)))
    ax.set_yticklabels(CLIP_ORDER, fontsize=9)
    ax.set_title(f"{title_prefix}{dataset} — NASA Score (lower=better)", fontsize=11)

    # Annotate cells
    for i in range(len(CLIP_ORDER)):
        for j in range(len(LOSS_ORDER)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7,
                        color="black" if abs(val - vmin) / max(vmax - vmin, 1) > 0.3
                        else "white")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# fig_H7_01 .. 04: NASA Score heatmaps per dataset
# ─────────────────────────────────────────────────────────────────────────────
for i, ds in enumerate(DATASETS, 1):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_nasa_heatmap(ds, ax=ax)
    fig.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"fig_H7_0{i}_nasa_heatmap_{ds}.png")
    fig.savefig(fname, dpi=RESS_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# fig_H7_05: Clip × Loss interaction effect
# ─────────────────────────────────────────────────────────────────────────────
CLIP_LABELS = {"clip_100": "100", "clip_125": "125", "clip_130": "130", "clip_none": "None"}
clip_x = [CLIP_LABELS[c] for c in CLIP_ORDER]

fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=False)
for ax, ds in zip(axes, DATASETS):
    sub   = matrix[matrix["dataset"] == ds]
    pivot = (sub.groupby(["clip", "loss_fn"])["nasa_mean"]
                .mean()
                .unstack("loss_fn")
                .reindex(index=CLIP_ORDER, columns=LOSS_ORDER))
    for li, loss in enumerate(LOSS_ORDER):
        if loss in pivot.columns:
            ax.plot(clip_x, pivot[loss].values, marker="o",
                    label=loss, linewidth=1.5, color=WONG_7[li % len(WONG_7)])
    ax.set_yscale("log")
    ax.set_title(ds, fontsize=10)
    ax.set_xlabel("Clip threshold")
    ax.set_ylabel("NASA Score (log)" if ax is axes[0] else "")
    ax.tick_params(axis="x", labelsize=8)

handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="upper right", fontsize=8, ncol=1,
           bbox_to_anchor=(1.0, 0.95), title="Loss fn", title_fontsize=8)
fig.tight_layout(rect=[0, 0, 0.88, 1])
fname = os.path.join(FIGURES_DIR, "fig_H7_05_clip_loss_interaction.png")
fig.savefig(fname, dpi=RESS_DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# fig_H7_06: Pareto frontier (RMSE vs NASA Score)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, ds in zip(axes, DATASETS):
    sub = matrix[matrix["dataset"] == ds]
    agg = (sub.groupby(["clip", "loss_fn"])
              .agg(nasa=("nasa_mean", "mean"), rmse=("rmse_mean", "mean"))
              .reset_index())

    colors  = plt.cm.tab10(np.linspace(0, 1, len(LOSS_ORDER)))
    markers = ["o", "s", "^", "D", "v", "P", "*"]
    clip_colors = {c: plt.cm.Set2(i / 4) for i, c in enumerate(CLIP_ORDER)}

    for _, row in agg.iterrows():
        li = LOSS_ORDER.index(row["loss_fn"]) if row["loss_fn"] in LOSS_ORDER else 0
        ax.scatter(row["nasa"], row["rmse"],
                   color=colors[li],
                   marker=markers[li % len(markers)],
                   s=60, alpha=0.8,
                   label=row["loss_fn"])

    # Pareto front (lower RMSE and lower NASA is better)
    pts = agg[["nasa", "rmse"]].values
    pareto_mask = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        for j, q in enumerate(pts):
            if i == j: continue
            if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                pareto_mask[i] = False
                break
    pareto_pts = pts[pareto_mask]
    if len(pareto_pts) > 1:
        idx_sort = np.argsort(pareto_pts[:, 0])
        pareto_pts = pareto_pts[idx_sort]
        ax.plot(pareto_pts[:, 0], pareto_pts[:, 1], "k--", lw=1.2, alpha=0.5,
                label="Pareto front")

    ax.set_title(ds, fontsize=10)
    ax.set_xlabel("NASA Score")
    ax.set_ylabel("RMSE" if ax is axes[0] else "")
    ax.grid(True, alpha=0.3)

# Shared legend
handles_all, labels_all = [], []
seen = set()
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
            handles_all.append(h)
            labels_all.append(l)
            seen.add(l)
fig.legend(handles_all, labels_all, loc="upper right", fontsize=7, ncol=1,
           bbox_to_anchor=(1.10, 0.95))
fig.suptitle("H4 Pareto Frontier: RMSE vs NASA Score", fontsize=12)
fig.tight_layout()
fname = os.path.join(FIGURES_DIR, "fig_H7_06_pareto_frontier.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fname}")

print("\nAll figures generated.")
