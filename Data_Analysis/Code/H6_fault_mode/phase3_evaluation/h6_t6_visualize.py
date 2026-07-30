# -*- coding: utf-8 -*-
"""T6 — False Routing Sensitivity Visualization"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR     = r"C:\BMAD_PY313\Data_Analysis\Results\H6_fault_mode\T6_false_routing"
FIG_DIR     = r"C:\BMAD_PY313\Data_Analysis\Results\H6_fault_mode\T6_false_routing\figures"
os.makedirs(FIG_DIR, exist_ok=True)

df_raw = pd.read_csv(os.path.join(OUT_DIR, "T6_raw_results.csv"))
df_eng = pd.read_csv(os.path.join(OUT_DIR, "T6_per_engine_results.csv"))

COLORS = {"FD003": "#2196F3", "FD004": "#FF5722"}

# ── Fig A: RMSE under four routing conditions (bar + scatter) ────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

conditions = ["rmse_normal", "rmse_flipped", "rmse_branch0", "rmse_branch1"]
labels     = ["Normal\n(GatingNet)", "Flipped\n(100% mis-route)", "Branch-0\nonly", "Branch-1\nonly"]
x          = np.arange(len(conditions))
width      = 0.35

for ax_idx, (dataset, color) in enumerate(COLORS.items()):
    ax   = axes[ax_idx]
    sub  = df_raw[df_raw["dataset"] == dataset]
    means = [sub[c].mean() for c in conditions]
    stds  = [sub[c].std()  for c in conditions]

    bars = ax.bar(x, means, width=0.55, color=color, alpha=0.75, zorder=2,
                  yerr=stds, capsize=4, error_kw={"elinewidth": 1.5})

    # individual seed dots
    for ci, cond in enumerate(conditions):
        vals = sub[cond].values
        ax.scatter([ci] * len(vals), vals, color="black", s=25, zorder=5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE (cycles)", fontsize=11)
    ax.set_title(f"{dataset} — Routing Condition Comparison", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    # annotate delta
    delta_mean = sub["delta_rmse_flip"].mean()
    delta_pct  = delta_mean / sub["rmse_normal"].mean() * 100
    ax.annotate(f"Δ={delta_mean:+.1f}\n({delta_pct:+.0f}%)",
                xy=(1, means[1]), xytext=(1.35, means[1] + stds[1] + 0.5),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "T6_routing_rmse_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[Fig A] Routing RMSE comparison saved.")

# ── Fig B: Gate confidence vs ΔRMSE (scatter, per seed) ─────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for dataset, color in COLORS.items():
    sub = df_raw[df_raw["dataset"] == dataset]
    ax.scatter(sub["gate_conf_mean"], sub["delta_rmse_flip"],
               color=color, s=80, zorder=5, label=dataset, alpha=0.9)
    for _, row in sub.iterrows():
        ax.annotate(f"s{int(row['seed'])}",
                    (row["gate_conf_mean"], row["delta_rmse_flip"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)

ax.axhline(0, linestyle="--", color="gray", lw=1)
ax.set_xlabel("Gate Confidence  mean(max(w₀, w₁))", fontsize=11)
ax.set_ylabel("ΔRMSE  (Flipped − Normal)", fontsize=11)
ax.set_title("Gate Confidence vs. False-Routing Sensitivity", fontsize=12)
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "T6_conf_vs_delta_rmse.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[Fig B] Confidence vs. delta RMSE saved.")

# ── Fig C: Per-engine gate confidence distribution (FD003) ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax_idx, (dataset, color) in enumerate(COLORS.items()):
    ax  = axes[ax_idx]
    sub = df_eng[df_eng["dataset"] == dataset]
    for seed in sub["seed"].unique():
        vals = sub[sub["seed"] == seed]["gate_conf"].values
        ax.hist(vals, bins=20, alpha=0.4, color=color, density=True)
    ax.axvline(0.8, linestyle="--", color="red", lw=1.5, label="conf=0.8 threshold")
    ax.set_xlabel("Gate confidence  max(w₀, w₁)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{dataset} — Per-engine gate confidence (all seeds)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "T6_gate_conf_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[Fig C] Gate confidence distribution saved.")

print(f"\n[T6 Viz] All figures saved to {FIG_DIR}")
