"""
NASA CMAPSS TurboFan EDA Script
Generates plots -> Dataset/Figure/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = r"C:\BMAD_PY313\Dataset"
FIG_DIR    = r"C:\BMAD_PY313\Dataset\Figure"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})

COLS = (["unit", "cycle"] +
        [f"op{i}" for i in range(1, 4)] +
        [f"s{i}"  for i in range(1, 22)])

DATASETS = ["FD001", "FD002", "FD003", "FD004"]

# ── Load data ──────────────────────────────────────────────────────────────
def load_dataset(name):
    train = pd.read_csv(os.path.join(DATA_DIR, f"train_{name}.txt"),
                        sep=r"\s+", header=None, names=COLS)
    test  = pd.read_csv(os.path.join(DATA_DIR, f"test_{name}.txt"),
                        sep=r"\s+", header=None, names=COLS)
    rul   = pd.read_csv(os.path.join(DATA_DIR, f"RUL_{name}.txt"),
                        header=None, names=["RUL"])
    # Train RUL: linear piece-wise (clip at 125 for CMAPSS standard)
    max_cycle = train.groupby("unit")["cycle"].max().reset_index()
    max_cycle.columns = ["unit", "max_cycle"]
    train = train.merge(max_cycle, on="unit")
    train["RUL"] = train["max_cycle"] - train["cycle"]
    train.drop("max_cycle", axis=1, inplace=True)
    return train, test, rul

print("Loading datasets...")
trains, tests, ruls = {}, {}, {}
for ds in DATASETS:
    trains[ds], tests[ds], ruls[ds] = load_dataset(ds)

all_train = pd.concat(trains.values(), keys=DATASETS, names=["dataset"])
all_train = all_train.reset_index(level=0).reset_index(drop=True)

sensor_cols = [f"s{i}" for i in range(1, 22)]
op_cols     = ["op1", "op2", "op3"]

CLIP_VALUE = 125  # piece-wise linear RUL clipping (CMAPSS standard)

# Clipped version: used for correlation analysis and model-relevant statistics.
# Raw trains[] is kept for Fig 2 (distribution) and Fig 10 (clipping comparison).
trains_clipped = {}
for ds in DATASETS:
    trains_clipped[ds] = trains[ds].copy()
    trains_clipped[ds]["RUL"] = trains_clipped[ds]["RUL"].clip(upper=CLIP_VALUE)

# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Dataset Overview (per-dataset stats)
# ══════════════════════════════════════════════════════════════════════════
print("[1/12] Dataset overview bar chart...")
stats_rows = []
for ds in DATASETS:
    tr = trains[ds]
    stats_rows.append({
        "dataset":       ds,
        "train_engines": tr["unit"].nunique(),
        "test_engines":  tests[ds]["unit"].nunique(),
        "train_rows":    len(tr),
        "max_cycle_mean": tr.groupby("unit")["cycle"].max().mean(),
        "max_cycle_std":  tr.groupby("unit")["cycle"].max().std(),
    })
stats_df = pd.DataFrame(stats_rows)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
stats_df.plot.bar(x="dataset", y=["train_engines", "test_engines"],
                  ax=axes[0], color=["#2196F3", "#FF9800"], rot=0)
axes[0].set_title("Number of Engines")
axes[0].set_ylabel("Count"); axes[0].legend(["Train", "Test"])

stats_df.plot.bar(x="dataset", y="train_rows", ax=axes[1],
                  color="#4CAF50", rot=0, legend=False)
axes[1].set_title("Training Rows per Dataset")
axes[1].set_ylabel("Rows")

axes[2].bar(stats_df["dataset"], stats_df["max_cycle_mean"],
            yerr=stats_df["max_cycle_std"], color="#9C27B0",
            capsize=5)
axes[2].set_title("Avg. Max Cycle (Train) ± Std")
axes[2].set_ylabel("Cycles")
plt.suptitle("Fig 1 — Dataset Overview", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig01_dataset_overview.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — RUL Distribution (train, per dataset)
# ══════════════════════════════════════════════════════════════════════════
print("[2/12] RUL distribution...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, ds in zip(axes.flat, DATASETS):
    rul_vals = trains[ds]["RUL"]
    ax.hist(rul_vals, bins=60, color="#2196F3", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(rul_vals.mean(), color="red",    linestyle="--", lw=1.5, label=f"Mean={rul_vals.mean():.0f}")
    ax.axvline(rul_vals.median(), color="orange", linestyle=":",  lw=1.5, label=f"Median={rul_vals.median():.0f}")
    ax.set_title(f"{ds}  (skew={rul_vals.skew():.2f})")
    ax.set_xlabel("RUL (cycles)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)
plt.suptitle("Fig 2 — RUL Distribution per Dataset (Training)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig02_rul_distribution.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Operating Conditions Scatter (FD002/FD004 multi-condition)
# ══════════════════════════════════════════════════════════════════════════
print("[3/12] Operating conditions...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ds in zip(axes, ["FD002", "FD004"]):
    tr = trains[ds].sample(n=min(5000, len(trains[ds])), random_state=42)
    sc = ax.scatter(tr["op1"], tr["op2"], c=tr["op3"],
                    cmap="plasma", alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label="op3")
    ax.set_xlabel("op1"); ax.set_ylabel("op2")
    ax.set_title(f"{ds} — Operational Settings (op1 vs op2, colored by op3)")
plt.suptitle("Fig 3 — Operating Condition Clusters (multi-condition datasets)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig03_op_conditions.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Sensor Variance (useful vs near-zero variance)
# ══════════════════════════════════════════════════════════════════════════
print("[4/12] Sensor variance...")
var_df = pd.DataFrame({ds: trains[ds][sensor_cols].std() for ds in DATASETS})
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(sensor_cols))
width = 0.2
colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
for i, ds in enumerate(DATASETS):
    ax.bar(x + i*width, var_df[ds], width, label=ds, color=colors[i], alpha=0.85)
ax.axhline(1.0, color="gray", linestyle="--", lw=1, label="σ=1 threshold")
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(sensor_cols, rotation=45)
ax.set_ylabel("Std Dev"); ax.set_yscale("log")
ax.set_title("Sensor Standard Deviation per Dataset (log scale)")
ax.legend()
plt.suptitle("Fig 4 — Sensor Variance Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig04_sensor_variance.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Sensor–RUL Correlation (FD001)
# ══════════════════════════════════════════════════════════════════════════
print("[5/12] Sensor-RUL correlation (using clipped RUL=125)...")
corr_data = {}
for ds in DATASETS:
    # Use clipped RUL: model training will use clip=125, so correlation
    # should reflect the same label distribution.
    corr_data[ds] = trains_clipped[ds][sensor_cols + ["RUL"]].corr()["RUL"].drop("RUL")

corr_df = pd.DataFrame(corr_data)
fig, ax = plt.subplots(figsize=(13, 5))
corr_df.plot.bar(ax=ax, color=colors, alpha=0.85, width=0.7, rot=45)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(0.3,  color="green",  lw=1, linestyle="--", label="|r|=0.3")
ax.axhline(-0.3, color="green",  lw=1, linestyle="--")
ax.set_ylabel("Pearson r with RUL")
ax.set_title(f"Sensor Correlation with RUL (Pearson r, RUL clipped at {CLIP_VALUE})")
ax.legend()
plt.suptitle("Fig 5 — Sensor–RUL Correlation (clipped RUL)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig05_sensor_rul_correlation.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 6 — Sensor Degradation Trends (FD001, high-corr sensors)
# ══════════════════════════════════════════════════════════════════════════
print("[6/12] Degradation trends...")
# Pick top-6 sensors by |corr| with RUL in FD001
top6 = corr_data["FD001"].abs().nlargest(6).index.tolist()
tr1  = trains["FD001"]
sample_units = np.random.RandomState(42).choice(tr1["unit"].unique(), size=10, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, s in zip(axes.flat, top6):
    for u in sample_units:
        unit_df = tr1[tr1["unit"] == u].sort_values("cycle")
        ax.plot(unit_df["cycle"], unit_df[s], alpha=0.45, lw=0.8, color="#2196F3")
    # mean trend across all units (binned by cycle)
    bins = pd.cut(tr1["cycle"], bins=50)
    mean_trend = tr1.groupby(bins, observed=True)[s].mean()
    bin_mids   = [interval.mid for interval in mean_trend.index]
    ax.plot(bin_mids, mean_trend.values, color="red", lw=2, label="Fleet mean")
    r_val = corr_data["FD001"][s]
    ax.set_title(f"{s}  (r={r_val:.2f})")
    ax.set_xlabel("Cycle"); ax.set_ylabel(s)
    ax.legend(fontsize=7)

plt.suptitle("Fig 6 — Top-6 Sensor Degradation Trends (FD001, 10 engines + fleet mean)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig06_degradation_trends.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 7 — Sensor Correlation Heatmap (FD001)
# ══════════════════════════════════════════════════════════════════════════
print("[7/12] Sensor correlation heatmap...")
# Drop near-zero variance sensors (std < 0.01) for FD001
valid_sensors = trains["FD001"][sensor_cols].std()
valid_sensors = valid_sensors[valid_sensors >= 0.01].index.tolist()
corr_matrix = trains["FD001"][valid_sensors].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", center=0,
            annot=True, fmt=".2f", linewidths=0.4, ax=ax,
            annot_kws={"size": 8}, vmin=-1, vmax=1)
ax.set_title("Inter-Sensor Pearson Correlation (FD001, active sensors only)")
plt.suptitle("Fig 7 — Sensor Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig07_sensor_heatmap.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 8 — Engine Lifetime Distribution
# ══════════════════════════════════════════════════════════════════════════
print("[8/12] Engine lifetime distributions...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, ds in zip(axes.flat, DATASETS):
    lifetimes = trains[ds].groupby("unit")["cycle"].max()
    ax.hist(lifetimes, bins=30, color="#FF5722", alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(lifetimes.mean(), color="blue",  lw=1.5, linestyle="--",
               label=f"μ={lifetimes.mean():.0f}")
    ax.axvline(lifetimes.std() + lifetimes.mean(), color="gray", lw=1, linestyle=":",
               label=f"σ={lifetimes.std():.0f}")
    ax.set_title(ds); ax.set_xlabel("Max Cycle (Lifetime)"); ax.set_ylabel("Engines")
    ax.legend(fontsize=8)
plt.suptitle("Fig 8 — Engine Lifetime Distribution (Training Set)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig08_engine_lifetime.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 9 — Early vs Late Cycle Sensor Distributions (FD001)
# ══════════════════════════════════════════════════════════════════════════
print("[9/12] Early vs Late sensor distributions...")
tr1 = trains["FD001"].copy()
tr1["life_pct"] = tr1["cycle"] / tr1.groupby("unit")["cycle"].transform("max")
early = tr1[tr1["life_pct"] < 0.2]
late  = tr1[tr1["life_pct"] > 0.8]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, s in zip(axes.flat, top6):
    ax.hist(early[s], bins=40, alpha=0.6, color="#2196F3", label="Early (<20%)", density=True)
    ax.hist(late[s],  bins=40, alpha=0.6, color="#F44336", label="Late (>80%)",  density=True)
    stat, p = stats.ks_2samp(early[s].dropna(), late[s].dropna())
    ax.set_title(f"{s}  KS p={p:.1e}")
    ax.set_xlabel(s); ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.suptitle("Fig 9 — Early vs Late Cycle Sensor Distributions (FD001, KS test)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig09_early_vs_late.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 10 — Clipped RUL (piece-wise linear) effect
# ══════════════════════════════════════════════════════════════════════════
print("[10/12] Clipped RUL comparison...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, clip in zip(axes, [None, 125]):
    rul_vals = trains["FD001"]["RUL"].copy()
    if clip:
        rul_vals = rul_vals.clip(upper=clip)
        title = f"RUL clipped at {clip} (piece-wise linear)"
    else:
        title = "Raw RUL (linear decay)"
    ax.hist(rul_vals, bins=60, color="#673AB7", alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(rul_vals.mean(), color="red", lw=1.5, linestyle="--",
               label=f"Mean={rul_vals.mean():.0f}")
    ax.set_title(title); ax.set_xlabel("RUL"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

plt.suptitle("Fig 10 — Effect of RUL Clipping Strategy (FD001)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig10_rul_clipping.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 11 — Rolling Mean Smoothing on top sensors (FD001 Unit 1)
# ══════════════════════════════════════════════════════════════════════════
print("[11/12] Rolling mean smoothing...")
unit1 = trains["FD001"][trains["FD001"]["unit"] == 1].sort_values("cycle")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, s in zip(axes.flat, top6):
    raw    = unit1[s].values
    smooth = pd.Series(raw).rolling(window=10, center=True).mean().values
    ax.plot(unit1["cycle"], raw,    alpha=0.4, color="#90CAF9", lw=0.8, label="Raw")
    ax.plot(unit1["cycle"], smooth, color="#1565C0", lw=1.8,           label="Rolling(10)")
    ax.set_title(s); ax.set_xlabel("Cycle"); ax.set_ylabel("Value")
    ax.legend(fontsize=8)

plt.suptitle("Fig 11 — Sensor Signal Smoothing (FD001 Engine #1, window=10)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig11_signal_smoothing.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIG 12 — Test RUL label distribution (ground truth)
# ══════════════════════════════════════════════════════════════════════════
print("[12/12] Test RUL label distributions...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, ds in zip(axes.flat, DATASETS):
    rul_vals = ruls[ds]["RUL"]
    ax.hist(rul_vals, bins=30, color="#009688", alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(rul_vals.mean(),   color="red",    lw=1.5, linestyle="--", label=f"Mean={rul_vals.mean():.0f}")
    ax.axvline(rul_vals.median(), color="orange", lw=1.5, linestyle=":",  label=f"Median={rul_vals.median():.0f}")
    ax.set_title(ds); ax.set_xlabel("True RUL at Test End"); ax.set_ylabel("Engines")
    ax.legend(fontsize=8)
plt.suptitle("Fig 12 — Ground-Truth RUL Distribution (Test Set)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig12_test_rul_labels.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY STATS for report
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Summary Stats ===")
for ds in DATASETS:
    tr = trains[ds]
    lifetimes = tr.groupby("unit")["cycle"].max()
    print(f"\n{ds}:")
    print(f"  Train rows: {len(tr):,}  |  Engines: {tr['unit'].nunique()}")
    print(f"  Lifetime: mean={lifetimes.mean():.1f}, std={lifetimes.std():.1f}, "
          f"min={lifetimes.min()}, max={lifetimes.max()}")
    print(f"  RUL(train, raw):     mean={tr['RUL'].mean():.1f}, std={tr['RUL'].std():.1f}")
    tr_c = trains_clipped[ds]
    print(f"  RUL(train, clip125): mean={tr_c['RUL'].mean():.1f}, std={tr_c['RUL'].std():.1f}")
    print(f"  RUL(test):  mean={ruls[ds]['RUL'].mean():.1f}")

# Top correlated sensors
print("\n=== Top sensors (|r| with RUL, FD001) ===")
print(corr_data["FD001"].abs().sort_values(ascending=False).head(10).to_string())

# Near-zero variance sensors
for ds in DATASETS:
    low_var = trains[ds][sensor_cols].std()
    dead = low_var[low_var < 0.01].index.tolist()
    print(f"\n{ds} near-zero variance sensors: {dead}")

print("\nAll figures saved to", FIG_DIR)
