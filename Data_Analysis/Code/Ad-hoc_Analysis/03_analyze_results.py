# -*- coding: utf-8 -*-
"""
03_analyze_results.py
H2-H3 Unified-Control Ad-hoc Analysis — Statistical Analysis

Computes:
  - Tier 2 effect : RMSE(N3+M0) - RMSE(N1+M0) per dataset/seed
  - Tier 3 effect : RMSE(N1+M0) - RMSE(N1+M3) on FD003/FD004 per seed
  - Ratio Tier3/Tier2 for FD003 (key validation of ordering)
  - Wilcoxon rank-sum (one-sided) + Benjamini-Hochberg FDR
  - Cohen's d (pooled SD)
  - Comparison with original H5/H6 results
  - Bar chart of tier effects
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

RESULTS_DIR = _ROOT / "Results" / "Ad-hoc_Analysis"
IN_CSV  = RESULTS_DIR / "unified_results.csv"
OUT_DIR = RESULTS_DIR

DATASETS    = ["FD001", "FD002", "FD003", "FD004"]
SEEDS       = [0, 1, 2, 3, 4]
ALPHA       = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled SD), a > b implies positive d."""
    n1, n2 = len(a), len(b)
    pooled_var = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    return float((a.mean() - b.mean()) / (np.sqrt(pooled_var) + 1e-12))


def wilcoxon_onesided(a: np.ndarray, b: np.ndarray):
    """H1: mean(a) > mean(b)  (a is worse, b is better)."""
    stat, p_two = ranksums(a, b)
    p_one = p_two / 2 if stat > 0 else 1 - p_two / 2
    return float(stat), float(p_one)


# ---------------------------------------------------------------------------
# Load and pivot
# ---------------------------------------------------------------------------

def load_pivot(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only completed (non-NaN) rows
    df = df.dropna(subset=["rmse"])
    return df


def get_rmse(df, cond, norm, model, dataset) -> np.ndarray:
    mask = (
        (df["condition"] == cond) &
        (df["normalizer"] == norm) &
        (df["model"] == model) &
        (df["dataset"] == dataset)
    )
    vals = df.loc[mask].sort_values("seed")["rmse"].values
    return vals.astype(np.float64)


# ---------------------------------------------------------------------------
# Effect-size computation
# ---------------------------------------------------------------------------

def compute_tier_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each dataset compute:
      tier2_delta  = mean RMSE(N3+M0) - mean RMSE(N1+M0)   [positive = N1 better]
      tier3_delta  = mean RMSE(N1+M0) - mean RMSE(N1+M3)   [positive = M3 better, FD003/FD004 only]
    """
    rows = []
    for ds in DATASETS:
        n1_m0 = get_rmse(df, "A", "N1", "M0", ds)
        n3_m0 = get_rmse(df, "B", "N3", "M0", ds)

        if len(n1_m0) == 0 or len(n3_m0) == 0:
            continue

        tier2_per_seed = n3_m0 - n1_m0   # positive = N1 is better
        row = {
            "dataset": ds,
            "N1_M0_mean": n1_m0.mean(),
            "N1_M0_std":  n1_m0.std(ddof=1),
            "N3_M0_mean": n3_m0.mean(),
            "N3_M0_std":  n3_m0.std(ddof=1),
            "tier2_delta_mean": tier2_per_seed.mean(),
            "tier2_delta_std":  tier2_per_seed.std(ddof=1),
        }

        if ds in ("FD003", "FD004"):
            n1_m3 = get_rmse(df, "C", "N1", "M3", ds)
            if len(n1_m3) > 0:
                tier3_per_seed = n1_m0 - n1_m3   # positive = M3 is better
                row["N1_M3_mean"] = n1_m3.mean()
                row["N1_M3_std"]  = n1_m3.std(ddof=1)
                row["tier3_delta_mean"] = tier3_per_seed.mean()
                row["tier3_delta_std"]  = tier3_per_seed.std(ddof=1)
                # Ratio: how much larger is architecture effect vs. normalization effect?
                if abs(row["tier2_delta_mean"]) > 1e-3:
                    row["tier3_over_tier2"] = row["tier3_delta_mean"] / row["tier2_delta_mean"]
                else:
                    row["tier3_over_tier2"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows).set_index("dataset")


# ---------------------------------------------------------------------------
# Statistical tests (BH-FDR corrected)
# ---------------------------------------------------------------------------

def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comparisons:
      For each dataset: H1 → RMSE(N3+M0) > RMSE(N1+M0)    [Tier 2 effect]
      For FD003/FD004:  H1 → RMSE(N1+M0) > RMSE(N1+M3)    [Tier 3 effect]
    """
    tests = []

    for ds in DATASETS:
        n1_m0 = get_rmse(df, "A", "N1", "M0", ds)
        n3_m0 = get_rmse(df, "B", "N3", "M0", ds)
        if len(n1_m0) == 0 or len(n3_m0) == 0:
            continue
        stat, p = wilcoxon_onesided(n3_m0, n1_m0)  # H1: N3 > N1 (N1 is better)
        d = cohen_d(n3_m0, n1_m0)
        tests.append({
            "comparison": f"Tier2_N3vsN1_{ds}",
            "dataset": ds, "tier": "T2",
            "worse_mean": n3_m0.mean(), "better_mean": n1_m0.mean(),
            "delta": n3_m0.mean() - n1_m0.mean(),
            "stat": stat, "p_raw": p, "cohen_d": d,
        })

    for ds in ("FD003", "FD004"):
        n1_m0 = get_rmse(df, "A", "N1", "M0", ds)
        n1_m3 = get_rmse(df, "C", "N1", "M3", ds)
        if len(n1_m0) == 0 or len(n1_m3) == 0:
            continue
        stat, p = wilcoxon_onesided(n1_m0, n1_m3)  # H1: M0 > M3 (M3 is better)
        d = cohen_d(n1_m0, n1_m3)
        tests.append({
            "comparison": f"Tier3_M0vsM3_{ds}",
            "dataset": ds, "tier": "T3",
            "worse_mean": n1_m0.mean(), "better_mean": n1_m3.mean(),
            "delta": n1_m0.mean() - n1_m3.mean(),
            "stat": stat, "p_raw": p, "cohen_d": d,
        })

    test_df = pd.DataFrame(tests)
    if len(test_df) == 0:
        return test_df

    # BH-FDR correction
    _, p_adj, _, _ = multipletests(test_df["p_raw"].values, method="fdr_bh")
    test_df["p_bh"]        = p_adj
    test_df["significant"] = (test_df["p_bh"] < ALPHA) & (test_df["cohen_d"].abs() >= 0.3)

    return test_df.round(4)


# ---------------------------------------------------------------------------
# Gate confidence summary (M3 only)
# ---------------------------------------------------------------------------

def gate_confidence_summary(df: pd.DataFrame) -> pd.DataFrame:
    m3_df = df[df["model"] == "M3"].copy()
    if m3_df.empty:
        return pd.DataFrame()
    return (
        m3_df.groupby("dataset")["gate_conf"]
        .agg(["mean", "std", "min", "max"])
        .round(3)
    )


# ---------------------------------------------------------------------------
# Original results for comparison
# ---------------------------------------------------------------------------

# From manuscript Tables V and VI (mean ± std over 5 seeds)
ORIG_H5_N1_M0 = {           # H5 FD003 baseline (compact backbone, last-20% split)
    "FD001": (13.18, 0.26),
    "FD002": (17.28, 0.38),
    "FD003": (19.05, 12.86),
    "FD004": (21.91, 0.82),
}
ORIG_H6_M3 = {              # H6 M3 (full-capacity backbone, random split, no eval clip)
    "FD003": (14.78, 1.32),
    "FD004": (22.84, 0.84),
}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_tier_effects(effects_df: pd.DataFrame, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Tier 2 effect per dataset
        ds_list = list(effects_df.index)
        t2_means = [effects_df.loc[ds, "tier2_delta_mean"] for ds in ds_list]
        t2_stds  = [effects_df.loc[ds, "tier2_delta_std"]  for ds in ds_list]
        axes[0].bar(ds_list, t2_means, yerr=t2_stds, capsize=5,
                    color=["#4472C4"] * 4, alpha=0.8)
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title("Tier 2 Effect\nRMSE(N3+M0) − RMSE(N1+M0)", fontsize=11)
        axes[0].set_ylabel("ΔRMSE [cycles]")
        axes[0].set_xlabel("Dataset")

        # Right: Tier 3 effect (FD003/FD004 only)
        t3_ds    = [ds for ds in ["FD003", "FD004"] if "tier3_delta_mean" in effects_df.columns
                    and not np.isnan(effects_df.loc[ds, "tier3_delta_mean"])]
        t3_means = [effects_df.loc[ds, "tier3_delta_mean"] for ds in t3_ds]
        t3_stds  = [effects_df.loc[ds, "tier3_delta_std"]  for ds in t3_ds]
        axes[1].bar(t3_ds, t3_means, yerr=t3_stds, capsize=5,
                    color=["#ED7D31"] * len(t3_ds), alpha=0.8)
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title("Tier 3 Effect\nRMSE(N1+M0) − RMSE(N1+M3)", fontsize=11)
        axes[1].set_ylabel("ΔRMSE [cycles]")
        axes[1].set_xlabel("Dataset")

        plt.suptitle(
            "Unified-Protocol Tier Effect Comparison\n"
            "(N=5 seeds; error bars = ±1 SD)",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved: {out_path}")
    except Exception as e:
        print(f"[Warning] Plot failed: {e}")


def plot_rmse_comparison(df: pd.DataFrame, out_path: Path):
    """Bar chart comparing N1+M0, N3+M0, N1+M3 RMSE per dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
        colors = {"N1+M0": "#4472C4", "N3+M0": "#A9C4E8", "N1+M3": "#ED7D31"}

        for ax, ds in zip(axes, DATASETS):
            bars_data = {}
            n1_m0 = get_rmse(df, "A", "N1", "M0", ds)
            n3_m0 = get_rmse(df, "B", "N3", "M0", ds)
            if len(n1_m0): bars_data["N1+M0"] = (n1_m0.mean(), n1_m0.std(ddof=1))
            if len(n3_m0): bars_data["N3+M0"] = (n3_m0.mean(), n3_m0.std(ddof=1))
            if ds in ("FD003", "FD004"):
                n1_m3 = get_rmse(df, "C", "N1", "M3", ds)
                if len(n1_m3): bars_data["N1+M3"] = (n1_m3.mean(), n1_m3.std(ddof=1))

            labels = list(bars_data.keys())
            means  = [bars_data[k][0] for k in labels]
            stds   = [bars_data[k][1] for k in labels]
            bar_colors = [colors.get(k, "gray") for k in labels]
            ax.bar(labels, means, yerr=stds, capsize=5,
                   color=bar_colors, alpha=0.85)
            ax.set_title(ds, fontsize=11, fontweight="bold")
            ax.set_ylabel("RMSE [cycles]" if ds == "FD001" else "")
            ax.tick_params(axis="x", rotation=20)

        plt.suptitle(
            "RMSE by Condition — Unified Protocol\n"
            "(N1=Fleet MinMax, N3=Per-unit MinMax, M0=Single LSTM, M3=Attention Gate)",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved: {out_path}")
    except Exception as e:
        print(f"[Warning] Plot failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not IN_CSV.exists():
        print(f"Results file not found: {IN_CSV}")
        print("Run 02_run_unified.py first.")
        return

    df = load_pivot(IN_CSV)
    n_complete = len(df)
    print(f"Loaded {n_complete} completed runs from {IN_CSV}")

    # ---- Effect sizes ----
    effects = compute_tier_effects(df)
    print("\n=== Tier Effect Summary ===")
    cols_show = [c for c in [
        "N1_M0_mean", "N1_M0_std", "N3_M0_mean", "N3_M0_std",
        "N1_M3_mean", "N1_M3_std",
        "tier2_delta_mean", "tier3_delta_mean", "tier3_over_tier2",
    ] if c in effects.columns]
    print(effects[cols_show].round(2).to_string())
    effects.to_csv(OUT_DIR / "tier_comparison.csv")

    # ---- Statistical tests ----
    tests = run_statistical_tests(df)
    if not tests.empty:
        print("\n=== Statistical Tests (Wilcoxon one-sided + BH-FDR) ===")
        show = ["comparison", "worse_mean", "better_mean", "delta",
                "p_raw", "p_bh", "cohen_d", "significant"]
        print(tests[show].round(4).to_string(index=False))
        tests.to_csv(OUT_DIR / "statistical_tests.csv", index=False)

    # ---- Gate confidence ----
    conf_summary = gate_confidence_summary(df)
    if not conf_summary.empty:
        print("\n=== Gate Confidence (M3, mean max(w0,w1)) ===")
        print(conf_summary.to_string())

    # ---- Comparison with original results ----
    print("\n=== Protocol Change Impact (Unified vs Original) ===")
    print(f"{'Dataset':<8} {'H5 N1+M0 (orig)':>18} {'Unified N1+M0':>16} {'H6 M3 (orig)':>15} {'Unified N1+M3':>15}")
    for ds in DATASETS:
        n1_m0 = get_rmse(df, "A", "N1", "M0", ds)
        n1_m3 = get_rmse(df, "C", "N1", "M3", ds) if ds in ("FD003", "FD004") else np.array([])

        orig_base = f"{ORIG_H5_N1_M0[ds][0]:.2f}±{ORIG_H5_N1_M0[ds][1]:.2f}" if ds in ORIG_H5_N1_M0 else "—"
        new_base  = f"{n1_m0.mean():.2f}±{n1_m0.std(ddof=1):.2f}" if len(n1_m0) else "—"
        orig_m3   = f"{ORIG_H6_M3[ds][0]:.2f}±{ORIG_H6_M3[ds][1]:.2f}" if ds in ORIG_H6_M3 else "—"
        new_m3    = f"{n1_m3.mean():.2f}±{n1_m3.std(ddof=1):.2f}" if len(n1_m3) else "—"
        print(f"{ds:<8} {orig_base:>18} {new_base:>16} {orig_m3:>15} {new_m3:>15}")

    # ---- Key finding ----
    try:
        t2_fd3 = effects.loc["FD003", "tier2_delta_mean"]
        t3_fd3 = effects.loc["FD003", "tier3_delta_mean"]
        ratio  = effects.loc["FD003", "tier3_over_tier2"]
        print(f"\n=== Key Finding: FD003 ===")
        print(f"  Tier 2 effect (N3→N1, M0): ΔRMSE = {t2_fd3:+.2f} cycles")
        print(f"  Tier 3 effect (M0→M3, N1): ΔRMSE = {t3_fd3:+.2f} cycles")
        print(f"  Ratio (Tier3 / Tier2):     {ratio:.1f}×")
        if ratio > 1.0:
            print("  → Tier ordering SUPPORTED: architecture effect > normalization effect on FD003")
        else:
            print("  → Tier ordering NOT supported under unified protocol.")
    except Exception:
        pass

    # ---- Plots ----
    plot_tier_effects(effects, OUT_DIR / "figures" / "tier_effect_comparison.png")
    plot_rmse_comparison(df, OUT_DIR / "figures" / "rmse_by_condition.png")

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
