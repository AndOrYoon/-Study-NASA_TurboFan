"""
H7 Phase 7 — Evaluation & Statistical Tests
Wilcoxon signed-rank tests vs L1 MSE baseline, BH-FDR correction, Cohen's d.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

from config_h7 import RESULT_DIR, LOSS_NAMES, DATASETS

PHASE2B_CSV  = os.path.join(RESULT_DIR, "phase2b_all_seeds.csv")
MATRIX_CSV   = os.path.join(RESULT_DIR, "phase2b_results_matrix.csv")
OUTPUT_CSV   = os.path.join(RESULT_DIR, "evaluation_statistical_tests.csv")

print("=" * 60)
print("H7 Evaluation — Statistical Tests")
print("=" * 60)

if not os.path.exists(PHASE2B_CSV):
    raise FileNotFoundError(f"Phase 2b results not found: {PHASE2B_CSV}")

df = pd.read_csv(PHASE2B_CSV).dropna(subset=["nasa_score", "rmse"])

# ── BH-FDR correction ─────────────────────────────────────────────────────────
def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n  = len(p_values)
    idx = np.argsort(p_values)
    p_sorted = p_values[idx]
    bh = np.minimum.accumulate(p_sorted * n / (np.arange(n) + 1))[::-1]
    bh_adj = np.empty(n)
    bh_adj[idx] = bh[::-1]
    # ensure non-decreasing
    bh_adj = np.clip(bh_adj, 0, 1)
    return bh_adj


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (a vs b). Positive → a > b."""
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


# ── Wilcoxon tests: each loss vs L1 MSE ──────────────────────────────────────
test_rows = []
baseline  = "L1_MSE"
clip_vals = df["clip"].unique()

for clip in clip_vals:
    for ds in DATASETS:
        # baseline
        bl = df[(df["clip"] == clip) & (df["loss_fn"] == baseline) &
                (df["dataset"] == ds)]["nasa_score"].values
        if len(bl) == 0:
            continue

        for loss in LOSS_NAMES:
            if loss == baseline:
                continue
            alt = df[(df["clip"] == clip) & (df["loss_fn"] == loss) &
                     (df["dataset"] == ds)]["nasa_score"].values
            if len(alt) == 0 or len(alt) != len(bl):
                continue

            try:
                stat, p = stats.wilcoxon(alt, bl, alternative="less")
                # "less" → test if loss < baseline (i.e., improves NASA Score)
            except Exception:
                p = 1.0
            d = cohens_d(alt, bl)   # negative d → loss beats baseline

            test_rows.append({
                "clip":     clip,
                "dataset":  ds,
                "loss_fn":  loss,
                "mean_diff": float(alt.mean() - bl.mean()),   # <0 = better
                "cohens_d": round(d, 4),
                "p_raw":    round(p, 6),
            })

test_df = pd.DataFrame(test_rows)
if len(test_df) > 0:
    test_df["p_BH"] = bh_fdr(test_df["p_raw"].values)
    test_df["p_BH"] = test_df["p_BH"].round(6)
    test_df["significant"] = (
        (test_df["p_BH"] < 0.05) &
        (test_df["cohens_d"].abs() >= 0.3) &
        (test_df["mean_diff"] < 0)          # actually better
    )
    test_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Statistical test results saved → {OUTPUT_CSV}")

    print("\nSignificant improvements over L1 MSE (p_BH<0.05, |d|≥0.3):")
    sig = test_df[test_df["significant"]].sort_values("mean_diff")
    if len(sig) > 0:
        print(sig[["clip","dataset","loss_fn","mean_diff","cohens_d","p_BH"]]
                .to_string(index=False))
    else:
        print("  None found.")

# ── Best per dataset ──────────────────────────────────────────────────────────
print("\nBest (clip, loss_fn) per dataset by NASA Score:")
matrix = pd.read_csv(MATRIX_CSV)
for ds in DATASETS:
    sub = matrix[matrix["dataset"] == ds].sort_values("nasa_mean")
    b   = sub.iloc[0]
    print(f"  {ds}: {b['clip']:10s} {b['loss_fn']:12s} "
          f"NASA={b['nasa_mean']:.2f}±{b['nasa_std']:.2f}  "
          f"RMSE={b['rmse_mean']:.2f}±{b['rmse_std']:.2f}")

# ── Clip × Loss interaction: does optimal loss change by dataset? ─────────────
print("\nClip × Loss interaction (NASA, all datasets):")
pivot = (matrix.groupby(["clip", "loss_fn"])["nasa_mean"]
               .mean()
               .unstack(level="loss_fn")
               .round(2))
print(pivot.to_string())

# ── L5 vs L2 vs L6 ranking ────────────────────────────────────────────────────
print("\nL5 TWA vs L2 NASA vs L6 Pinball (mean NASA over all clips & datasets):")
compare = (matrix[matrix["loss_fn"].isin(["L1_MSE","L2_NASA","L5_TWA","L6_Pinball"])]
           .groupby("loss_fn")["nasa_mean"].mean()
           .sort_values()
           .reset_index())
print(compare.to_string(index=False))
