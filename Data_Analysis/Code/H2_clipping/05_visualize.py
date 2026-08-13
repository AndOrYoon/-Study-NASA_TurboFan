# -*- coding: utf-8 -*-
"""
H1 Hypothesis - Visualisation
Creates 4 publication-quality figures:
  fig_H2_01_rmse_heatmap.png      -- RMSE heatmap (datasets x clips)
  fig_H2_02_nasa_heatmap.png      -- NASA Score heatmap
  fig_H2_03_subgroup_rmse.png     -- Subgroup RMSE bar chart (by lifetime group)
  fig_H2_04_pvalue_plot.png       -- Wilcoxon p-values vs clip_125

Reads:
  metrics_summary.csv
  statistical_test.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(r"C:\BMAD_PY313\Data_Analysis\Results\H2_clipping")
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = RESULTS_DIR / "metrics_summary.csv"
STATS_CSV   = RESULTS_DIR / "statistical_test.csv"

# Ordered clip labels for consistent x-axis
CLIP_ORDER  = ['75', '100', '125', '130', 'None']
DATASETS    = ['FD001', 'FD002', 'FD003', 'FD004']

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

WONG = ['#0072B2', '#E69F00', '#009E73', '#D55E00']   # colorblind-friendly
RESS_DPI = 300


def set_style():
    """Apply RESS/Elsevier publication style."""
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
        'figure.dpi': RESS_DPI,
    })


# ---------------------------------------------------------------------------
# Figure 1: RMSE heatmap
# ---------------------------------------------------------------------------

def fig_rmse_heatmap(df_metrics):
    """Heatmap of mean RMSE across datasets and clip values."""
    pivot = df_metrics.pivot(index='dataset', columns='clip_value', values='rmse')
    pivot = pivot.reindex(index=DATASETS, columns=[c for c in CLIP_ORDER if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'clip={c}' for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9,
                        color='white' if val > pivot.values.max() * 0.7 else 'black')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('RMSE (cycles)', rotation=270, labelpad=14)

    ax.set_title('RMSE by Dataset and Clipping Threshold')
    ax.set_xlabel('Clipping Threshold')
    ax.set_ylabel('Dataset')
    fig.tight_layout()

    out = FIG_DIR / 'fig_H2_01_rmse_heatmap.png'
    fig.savefig(out, dpi=RESS_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: NASA Score heatmap
# ---------------------------------------------------------------------------

def fig_nasa_heatmap(df_metrics):
    """Heatmap of NASA Score (lower is better)."""
    pivot = df_metrics.pivot(index='dataset', columns='clip_value', values='nasa_score')
    pivot = pivot.reindex(index=DATASETS, columns=[c for c in CLIP_ORDER if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'clip={c}' for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9,
                        color='white' if val > pivot.values.max() * 0.7 else 'black')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Mean NASA Score (per engine, lower=better)', rotation=270, labelpad=18)

    ax.set_title('Mean NASA Score by Dataset and Clipping Threshold')
    ax.set_xlabel('Clipping Threshold')
    ax.set_ylabel('Dataset')
    fig.tight_layout()

    out = FIG_DIR / 'fig_H2_02_nasa_heatmap.png'
    fig.savefig(out, dpi=RESS_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Subgroup RMSE bar chart
# ---------------------------------------------------------------------------

def fig_subgroup_rmse(df_metrics):
    """
    Grouped bar chart: x=clip threshold, bars=lifetime_group (boundary/medium/long),
    one subplot per dataset.
    """
    groups     = ['rmse_boundary', 'rmse_medium', 'rmse_long']
    group_lbls = ['Boundary (<150)', 'Medium (150-250)', 'Long (>250)']
    colors     = WONG[:3]

    clips_present = [c for c in CLIP_ORDER if c in df_metrics['clip_value'].unique()]
    x = np.arange(len(clips_present))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    axes = axes.flatten()

    # sort key: position in CLIP_ORDER, unknown labels go last
    order_map = {c: i for i, c in enumerate(CLIP_ORDER)}

    for idx, fd_key in enumerate(DATASETS):
        ax = axes[idx]
        sub = df_metrics[df_metrics['dataset'] == fd_key].copy()
        sub = sub.sort_values('clip_value',
                              key=lambda s: s.map(lambda v: order_map.get(str(v), 999)))

        for g_idx, (col, lbl, color) in enumerate(zip(groups, group_lbls, colors)):
            vals = sub[col].values.astype(float)
            offset = (g_idx - 1) * width
            bars = ax.bar(x + offset, vals, width=width, label=lbl,
                          color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
            # Add value labels on bars
            for bar, v in zip(bars, vals):
                if not np.isnan(v) and v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{v:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_title(fd_key, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'clip={c}' for c in clips_present], fontsize=9, rotation=20)
        ax.set_ylabel('RMSE (cycles)')
        ax.set_xlabel('Clipping Threshold')
        if idx == 0:
            ax.legend(title='Lifetime Group', fontsize=8, title_fontsize=9)

    fig.suptitle('RMSE by Lifetime Group and Clipping Threshold',
                 fontsize=11, y=1.01)
    fig.tight_layout()

    out = FIG_DIR / 'fig_H2_03_subgroup_rmse.png'
    fig.savefig(out, dpi=RESS_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Wilcoxon p-value bar chart
# ---------------------------------------------------------------------------

def fig_pvalue_plot(df_stats):
    """
    Bar chart of Wilcoxon p-values (clip_X vs clip_125).
    Each dataset is a group; bars = clip values (excl. 125).
    Dashed red line at alpha=0.05.
    """
    df_non_baseline = df_stats[df_stats['clip_value'] != '125'].copy()
    clips_non_base  = [c for c in CLIP_ORDER if c != '125' and c in df_non_baseline['clip_value'].unique()]

    x     = np.arange(len(DATASETS))
    width = 0.18
    colors = WONG

    fig, ax = plt.subplots(figsize=(10, 5))

    for g_idx, cv in enumerate(clips_non_base):
        sub  = df_non_baseline[df_non_baseline['clip_value'] == cv]
        vals = []
        for ds in DATASETS:
            row = sub[sub['dataset'] == ds]
            vals.append(float(row['p_value'].values[0]) if len(row) > 0 else np.nan)
        offset = (g_idx - len(clips_non_base) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, label=f'clip={cv}',
                      color=colors[g_idx % len(colors)], alpha=0.85,
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

    # Alpha threshold line
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1.5,
               label='α = 0.05 significance')

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=11)
    ax.set_ylabel('Wilcoxon p-value (two-sided)')
    ax.set_xlabel('Dataset')
    ax.set_title('Wilcoxon Rank-Sum p-values (clip_X vs clip_125 baseline)')
    ax.set_ylim(0, max(ax.get_ylim()[1], 0.15))
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()

    out = FIG_DIR / 'fig_H2_04_pvalue_plot.png'
    fig.savefig(out, dpi=RESS_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_style()

    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Not found: {METRICS_CSV}. Run 03_evaluate.py first.")
    if not STATS_CSV.exists():
        raise FileNotFoundError(f"Not found: {STATS_CSV}. Run 04_statistical_test.py first.")

    # keep_default_na=False prevents pandas from converting the string 'None'
    # into NaN, which would cause it to be dropped from unique() lookups
    df_metrics = pd.read_csv(METRICS_CSV, keep_default_na=False)
    df_stats   = pd.read_csv(STATS_CSV,   keep_default_na=False)

    # Make sure clip_value is string (handles any int coercion by csv reader)
    df_metrics['clip_value'] = df_metrics['clip_value'].astype(str)
    df_stats['clip_value']   = df_stats['clip_value'].astype(str)

    print("Creating figures ...")
    fig_rmse_heatmap(df_metrics)
    fig_nasa_heatmap(df_metrics)
    fig_subgroup_rmse(df_metrics)
    fig_pvalue_plot(df_stats)
    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == '__main__':
    print("=" * 70)
    print("H1 Visualisation")
    print("=" * 70)
    main()
