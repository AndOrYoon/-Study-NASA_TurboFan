"""
H6 Additional EDA: FD003 Fault Mode Cluster Analysis
Checks whether the two fault modes (HPC Degradation vs Fan Degradation)
in FD003 produce distinguishable sensor signatures.

Output: Dataset/Figure/h6_*.png  +  terminal report
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\BMAD_PY313\Dataset"
FIG_DIR  = r"C:\BMAD_PY313\Dataset\Figure"

# FD003 non-informative sensors (near-zero std, from EDA)
CONST_SENSORS = {"s1", "s5", "s10", "s16", "s18", "s19"}
ALL_SENSORS   = [f"s{i}" for i in range(1, 22)]
ACTIVE        = [s for s in ALL_SENSORS if s not in CONST_SENSORS]  # 15 sensors

# Physical meaning of active sensors (Saxena 2008 column mapping)
SENSOR_LABEL = {
    "s2":  "T24 (LPC outlet temp)",
    "s3":  "T30 (HPC outlet temp)",
    "s4":  "T50 (LPT outlet temp)",
    "s6":  "P15 (bypass-duct pressure)",
    "s7":  "P30 (HPC outlet pressure)",
    "s8":  "Nf (fan speed)",
    "s9":  "Nc (core speed)",
    "s11": "Ps30 (static HPC pressure)",
    "s12": "phi (fuel/Ps30)",
    "s13": "NRf (corr. fan speed)",
    "s14": "NRc (corr. core speed)",
    "s15": "BPR (bypass ratio)",
    "s17": "htBleed",
    "s20": "W31 (HPT bleed)",
    "s21": "W32 (LPT bleed)",
}

# Sensors theoretically linked to each fault mode (Saxena 2008 + domain knowledge)
HPC_SENSORS = ["s3", "s7", "s11", "s9", "s14", "s12"]
FAN_SENSORS = ["s2", "s8", "s13", "s15", "s4"]

COLS = (["unit", "cycle"] +
        [f"op{i}" for i in range(1, 4)] +
        ALL_SENSORS)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.facecolor": "white",
})

# ── Load FD003 ──────────────────────────────────────────────────────────────
print("Loading FD003...")
train = pd.read_csv(os.path.join(DATA_DIR, "train_FD003.txt"),
                    sep=r"\s+", header=None, names=COLS)
max_cycle = train.groupby("unit")["cycle"].max().rename("max_cycle")
train = train.join(max_cycle, on="unit")
train["RUL"]       = train["max_cycle"] - train["cycle"]
train["life_norm"] = train["cycle"] / train["max_cycle"]   # 0 = start, 1 = failure
n_engines = train["unit"].nunique()
print(f"  Engines: {n_engines}, Cycles: {len(train)}")

# ── Feature Extraction: late-life mean (last 20%) per engine ────────────────
print("\n[1/5] Extracting late-life features (last 20% of cycles per engine)...")
records = []
for unit, grp in train.groupby("unit"):
    n = len(grp)
    late = grp.tail(max(5, int(n * 0.20)))
    early = grp.head(max(5, int(n * 0.20)))
    rec = {"unit": unit, "life": n}
    for s in ACTIVE:
        rec[f"{s}_late_mean"] = late[s].mean()
        rec[f"{s}_late_std"]  = late[s].std()
        rec[f"{s}_delta"]     = late[s].mean() - early[s].mean()  # degradation delta
    records.append(rec)

feat_df = pd.DataFrame(records).set_index("unit")
life_vec = feat_df.pop("life")

# Use late-mean features for clustering
feat_cols = [f"{s}_late_mean" for s in ACTIVE]
X_raw = feat_df[feat_cols].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── Cluster quality sweep (k = 1..5) ───────────────────────────────────────
print("\n[2/5] Cluster quality sweep (K-means & GMM BIC)...")
k_range = range(1, 6)
sil_scores = []
gmm_bic    = []

for k in k_range:
    # K-means silhouette
    if k >= 2:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        lbl = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, lbl))
    else:
        sil_scores.append(np.nan)

    # GMM BIC
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=42, n_init=5)
    gmm.fit(X)
    gmm_bic.append(gmm.bic(X))

print("  Silhouette scores:", {k: f"{s:.3f}" for k, s in zip(k_range, sil_scores) if not np.isnan(s)})
print("  GMM BIC:          ", {k: f"{b:.1f}" for k, b in zip(k_range, gmm_bic)})
best_k_bic = list(k_range)[np.argmin(gmm_bic)]
best_k_sil = list(k_range)[1 + np.nanargmax(sil_scores[1:])]
print(f"  Best k (BIC): {best_k_bic} | Best k (Silhouette): {best_k_sil}")

# ── Final clustering with k=2 ───────────────────────────────────────────────
print("\n[3/5] K-means & GMM with k=2...")
km2  = KMeans(n_clusters=2, random_state=42, n_init=20)
labels_km = km2.fit_predict(X)

gmm2 = GaussianMixture(n_components=2, covariance_type="full",
                        random_state=42, n_init=10)
gmm2.fit(X)
labels_gmm = gmm2.predict(X)
proba_gmm  = gmm2.predict_proba(X)

sil_km  = silhouette_score(X, labels_km)
sil_gmm = silhouette_score(X, labels_gmm)

# Align cluster 0/1 labels across methods: cluster 0 = longer life
for lbl in [labels_km, labels_gmm]:
    if life_vec.values[lbl == 0].mean() < life_vec.values[lbl == 1].mean():
        lbl[:] = 1 - lbl  # flip so cluster 0 = shorter life

feat_df["cluster_km"]  = labels_km
feat_df["cluster_gmm"] = labels_gmm
feat_df["life"]        = life_vec

for c in [0, 1]:
    n = (labels_km == c).sum()
    ml = life_vec[labels_km == c].mean()
    print(f"  KM Cluster {c}: n={n}, mean_life={ml:.1f}")

# Sensor-level discrimination: |z-score diff| between clusters
z_diff = {}
for s in ACTIVE:
    col = f"{s}_late_mean"
    v0 = feat_df.loc[labels_km == 0, col].values
    v1 = feat_df.loc[labels_km == 1, col].values
    pooled_std = np.sqrt((v0.var() + v1.var()) / 2 + 1e-9)
    z_diff[s] = abs(v0.mean() - v1.mean()) / pooled_std

ranked = sorted(z_diff.items(), key=lambda x: -x[1])
print("\n  Sensor discrimination (|z-diff|, descending):")
for s, d in ranked:
    print(f"    {s:4s} {SENSOR_LABEL.get(s,''):30s}  |Δz|={d:.3f}")

# ── Figure A: Cluster quality + PCA ────────────────────────────────────────
print("\n[4/5] Generating figures...")
fig_a, axes = plt.subplots(1, 3, figsize=(15, 4))

# A1: Silhouette vs k
ax = axes[0]
ks = [k for k in k_range if k >= 2]
ax.plot(ks, [s for s in sil_scores if not np.isnan(s)], "o-", color="steelblue")
ax.axvline(2, color="red", linestyle="--", alpha=0.6, label="k=2 (H6 hypothesis)")
ax.set_xlabel("k (number of clusters)")
ax.set_ylabel("Silhouette Score")
ax.set_title("K-means Silhouette vs k")
ax.legend(fontsize=8)
ax.set_xticks(list(k_range))
ax.set_ylim(0, 1)

# A2: GMM BIC vs k
ax = axes[1]
ax.plot(list(k_range), gmm_bic, "s-", color="darkorange")
ax.axvline(best_k_bic, color="red", linestyle="--", alpha=0.6,
           label=f"Best k={best_k_bic}")
ax.set_xlabel("k (number of components)")
ax.set_ylabel("BIC")
ax.set_title("GMM BIC vs k")
ax.legend(fontsize=8)
ax.set_xticks(list(k_range))

# A3: PCA 2D scatter colored by K-means cluster
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
ax = axes[2]
colors = ["steelblue", "tomato"]
for c in [0, 1]:
    mask = labels_km == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[c], alpha=0.7, s=40, label=f"Cluster {c} (n={mask.sum()})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title(f"PCA 2D — K-means k=2 (Sil={sil_km:.3f})")
ax.legend(fontsize=8)

fig_a.suptitle("H6 EDA — FD003 Fault Mode Cluster Quality", fontsize=11, fontweight="bold")
fig_a.tight_layout()
fig_a.savefig(os.path.join(FIG_DIR, "h6_fig01_cluster_quality.png"), bbox_inches="tight")
plt.close(fig_a)
print("  Saved: h6_fig01_cluster_quality.png")

# ── Figure B: Mean sensor profile per cluster (bar chart) ──────────────────
fig_b, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax_idx, cluster_col in enumerate(["cluster_km"]):
    ax = axes[0]
    sensors_sorted = [s for s, _ in ranked]
    z_diffs_sorted = [z_diff[s] for s in sensors_sorted]
    bar_colors = []
    for s in sensors_sorted:
        if s in HPC_SENSORS:
            bar_colors.append("steelblue")
        elif s in FAN_SENSORS:
            bar_colors.append("tomato")
        else:
            bar_colors.append("gray")
    bars = ax.barh(sensors_sorted, z_diffs_sorted, color=bar_colors, alpha=0.8)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="|Δz|=0.5 threshold")
    ax.set_xlabel("|Δz| (normalized mean difference between clusters)")
    ax.set_title("Sensor Discrimination Between Clusters\n(blue=HPC-related, red=Fan-related, gray=other)")
    ax.legend(fontsize=8)

# Cluster mean sensor heatmap (standardized values)
ax = axes[1]
cluster_means = {}
for c in [0, 1]:
    mask = labels_km == c
    row = []
    for s in ACTIVE:
        col = f"{s}_late_mean"
        row.append(feat_df.loc[mask, col].mean())
    cluster_means[c] = row

cm_arr = np.array([cluster_means[0], cluster_means[1]])  # (2, n_sensors)
cm_z   = (cm_arr - cm_arr.mean(axis=0)) / (cm_arr.std(axis=0) + 1e-9)

im = ax.imshow(cm_z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
ax.set_xticks(range(len(ACTIVE)))
ax.set_xticklabels(ACTIVE, rotation=45, ha="right", fontsize=8)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Cluster 0", "Cluster 1"])
ax.set_title("Cluster Mean Sensor Profile (standardized)\nRed=higher, Blue=lower vs other cluster")
plt.colorbar(im, ax=ax, shrink=0.7, label="z-score")

# Mark HPC / Fan sensor columns
for i, s in enumerate(ACTIVE):
    if s in HPC_SENSORS:
        ax.axvline(i - 0.5, color="steelblue", linewidth=0.5, alpha=0.4)
        ax.axvline(i + 0.5, color="steelblue", linewidth=0.5, alpha=0.4)
    elif s in FAN_SENSORS:
        ax.axvline(i - 0.5, color="tomato", linewidth=0.5, alpha=0.4)
        ax.axvline(i + 0.5, color="tomato", linewidth=0.5, alpha=0.4)

fig_b.suptitle("H6 EDA — FD003 Sensor Profile Differences Between Clusters", fontsize=11, fontweight="bold")
fig_b.tight_layout()
fig_b.savefig(os.path.join(FIG_DIR, "h6_fig02_sensor_profiles.png"), bbox_inches="tight")
plt.close(fig_b)
print("  Saved: h6_fig02_sensor_profiles.png")

# ── Figure C: Degradation trajectories for top discriminating sensors ───────
top_sensors = [s for s, d in ranked if d > 0.3][:6]
if len(top_sensors) < 4:
    top_sensors = [s for s, _ in ranked[:6]]

fig_c, axes = plt.subplots(2, 3, figsize=(15, 8))
axes_flat = axes.flatten()

bins = np.linspace(0, 1, 11)  # time-normalized bins

for ax_i, s in enumerate(top_sensors[:6]):
    ax = axes_flat[ax_i]
    for c, color, label in [(0, "steelblue", "Cluster 0"), (1, "tomato", "Cluster 1")]:
        mask = labels_km == c
        grp_engines = feat_df.index[mask]
        sub = train[train["unit"].isin(grp_engines)].copy()
        sub["bin"] = pd.cut(sub["life_norm"], bins=bins, labels=False)
        agg = sub.groupby("bin")[s].agg(["mean", "std"])
        x = agg.index.astype(float)
        ax.plot(x / 10, agg["mean"], color=color, label=label, linewidth=1.8)
        ax.fill_between(x / 10,
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        color=color, alpha=0.15)
    is_hpc = s in HPC_SENSORS
    is_fan = s in FAN_SENSORS
    tag = " [HPC]" if is_hpc else (" [FAN]" if is_fan else "")
    ax.set_title(f"{s}: {SENSOR_LABEL.get(s,'')}{tag}", fontsize=8)
    ax.set_xlabel("Normalized cycle (0=start, 1=failure)")
    ax.set_ylabel("Sensor value")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig_c.suptitle(
    "H6 EDA — FD003 Degradation Trajectories by Cluster\n"
    "(Top discriminating sensors, mean ± 1σ, time-normalized)",
    fontsize=11, fontweight="bold"
)
fig_c.tight_layout()
fig_c.savefig(os.path.join(FIG_DIR, "h6_fig03_degradation_trajectories.png"), bbox_inches="tight")
plt.close(fig_c)
print("  Saved: h6_fig03_degradation_trajectories.png")

# ── Figure D: Life distribution per cluster + GMM soft assignment ───────────
fig_d, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
for c, color in [(0, "steelblue"), (1, "tomato")]:
    mask = labels_km == c
    life_c = life_vec[mask]
    ax.hist(life_c, bins=20, alpha=0.6, color=color, label=f"Cluster {c} (n={mask.sum()})")
ax.set_xlabel("Engine lifetime (cycles)")
ax.set_ylabel("Count")
ax.set_title("Engine Lifetime Distribution by Cluster")
ax.legend()

ax = axes[1]
max_proba = proba_gmm.max(axis=1)
ax.hist(max_proba, bins=25, color="purple", alpha=0.7, edgecolor="white")
ax.axvline(0.9, color="red", linestyle="--", label="p=0.90 threshold")
high_conf = (max_proba > 0.9).sum()
ax.set_xlabel("GMM Max Assignment Probability")
ax.set_ylabel("Count")
ax.set_title(f"GMM Soft Assignment Confidence\n(>{0.9:.0%} threshold: {high_conf}/{n_engines} engines, {high_conf/n_engines*100:.0f}%)")
ax.legend()

fig_d.suptitle("H6 EDA — FD003 Cluster Characteristics", fontsize=11, fontweight="bold")
fig_d.tight_layout()
fig_d.savefig(os.path.join(FIG_DIR, "h6_fig04_cluster_characteristics.png"), bbox_inches="tight")
plt.close(fig_d)
print("  Saved: h6_fig04_cluster_characteristics.png")

# ── Terminal report ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("H6 EDA REPORT - FD003 FAULT MODE CLUSTER ANALYSIS")
print("=" * 65)

print(f"\n▶ Cluster validity")
print(f"  K-means  silhouette (k=2):  {sil_km:.3f}  (>0.5=strong, 0.25-0.5=moderate)")
print(f"  GMM      silhouette (k=2):  {sil_gmm:.3f}")
print(f"  GMM BIC best k:             {best_k_bic}")
print(f"  GMM engines with p>0.9:     {high_conf}/{n_engines} ({high_conf/n_engines*100:.0f}%)")

print(f"\n▶ Cluster composition (K-means)")
for c in [0, 1]:
    mask = labels_km == c
    print(f"  Cluster {c}: n={mask.sum():3d}, "
          f"life={life_vec[mask].mean():.1f}±{life_vec[mask].std():.1f}")

print(f"\n▶ Top 5 discriminating sensors (|Δz|)")
for s, d in ranked[:5]:
    tag = "[HPC]" if s in HPC_SENSORS else ("[FAN]" if s in FAN_SENSORS else "[---]")
    print(f"  {s:4s} {tag} {SENSOR_LABEL.get(s,''):30s}  |Δz|={d:.3f}")

print(f"\n▶ Pattern consistency check")
hpc_disc = [z_diff[s] for s in HPC_SENSORS if s in z_diff]
fan_disc  = [z_diff[s] for s in FAN_SENSORS if s in z_diff]
print(f"  HPC-related sensors avg |Δz|: {np.mean(hpc_disc):.3f}")
print(f"  FAN-related sensors avg |Δz|: {np.mean(fan_disc):.3f}")
print(f"  Interpretation:")
if np.mean(hpc_disc) > 0.3 and np.mean(fan_disc) > 0.3:
    print("  → Both HPC and FAN sensors show notable between-cluster differences.")
    print("    Clusters may reflect the two fault modes. H6 premise PARTIALLY supported.")
elif max(np.mean(hpc_disc), np.mean(fan_disc)) > 0.5:
    dominant = "HPC" if np.mean(hpc_disc) > np.mean(fan_disc) else "FAN"
    print(f"  → {dominant} sensors dominate discrimination.")
    print("    Cluster split may not cleanly separate fault modes.")
else:
    print("  → Neither HPC nor FAN sensors clearly separate the clusters.")
    print("    Cluster split likely reflects engine lifetime differences, NOT fault modes.")
    print("    ⚠️  H6 premise at risk — further analysis needed.")

print("\n" + "=" * 65)
print(f"Figures saved to: {FIG_DIR}")
print("  h6_fig01_cluster_quality.png")
print("  h6_fig02_sensor_profiles.png")
print("  h6_fig03_degradation_trajectories.png")
print("  h6_fig04_cluster_characteristics.png")
print("=" * 65)
