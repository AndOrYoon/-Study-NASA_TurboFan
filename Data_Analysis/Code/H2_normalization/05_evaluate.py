# -*- coding: utf-8 -*-
"""
05_evaluate.py
--------------
Aggregate raw per-engine predictions into metrics.

Outputs
-------
metrics_summary.csv    : RMSE & NASA Score per (dataset, norm_id), mean ± std over seeds
subgroup_rmse.csv      : per-lifetime-group RMSE per (dataset, norm_id)
statistical_test.csv   : Wilcoxon rank-sum + BH-FDR + Cohen's d  (N2-N7 vs N1)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR    = Path(__file__).parent
ROOT        = CODE_DIR.parent.parent
RESULTS_DIR = ROOT / "Results" / "H2_normalization"
RAW_DIR     = RESULTS_DIR / "raw_predictions"

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
NORM_IDS = ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]
SEEDS    = [0, 1, 2, 3, 4]

GROUPS   = ["boundary", "medium", "long"]

RUL_CLIP = 125


# ---------------------------------------------------------------------------
# NASA score (per engine, then summed)
# ---------------------------------------------------------------------------

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d     = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))


# ---------------------------------------------------------------------------
# BH-FDR correction (manual, works with any scipy version)
# ---------------------------------------------------------------------------

def bh_fdr(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n      = len(pvalues)
    order  = np.argsort(pvalues)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adj    = pvalues * n / ranked
    # enforce monotonicity (cumulative minimum from the right)
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return np.minimum(adj, 1.0)


# ---------------------------------------------------------------------------
# Load raw predictions for one (dataset, norm_id, seed)
# ---------------------------------------------------------------------------

def load_pred(dataset: str, norm_id: str, seed: int) -> pd.DataFrame:
    path = RAW_DIR / f"{dataset}_{norm_id}_seed{seed}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # 1. Aggregate RMSE / NASA Score per run
    # -----------------------------------------------------------------------
    records = []

    for ds in DATASETS:
        for nid in NORM_IDS:
            seed_rmse  = []
            seed_score = []
            for s in SEEDS:
                df = load_pred(ds, nid, s)
                if df is None:
                    continue
                rmse  = float(np.sqrt(np.mean((df["pred_rul"] - df["true_rul"]) ** 2)))
                score = nasa_score(df["true_rul"].values, df["pred_rul"].values)
                seed_rmse.append(rmse)
                seed_score.append(score)
                records.append({
                    "dataset": ds, "norm_id": nid, "seed": s,
                    "rmse": rmse, "nasa_score": score,
                })

    run_df = pd.DataFrame(records)

    # Summary: mean ± std over seeds
    summary = (
        run_df.groupby(["dataset", "norm_id"])
        .agg(
            rmse_mean  =("rmse",       "mean"),
            rmse_std   =("rmse",       "std"),
            score_mean =("nasa_score", "mean"),
            score_std  =("nasa_score", "std"),
            n_seeds    =("rmse",       "count"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print("Saved metrics_summary.csv")
    print(summary.to_string(index=False))

    # -----------------------------------------------------------------------
    # 2. Per-lifetime-group RMSE
    # -----------------------------------------------------------------------
    group_records = []

    for ds in DATASETS:
        for nid in NORM_IDS:
            for s in SEEDS:
                df = load_pred(ds, nid, s)
                if df is None:
                    continue
                for grp in GROUPS:
                    sub = df[df["group"] == grp]
                    if len(sub) == 0:
                        continue
                    rmse = float(np.sqrt(
                        np.mean((sub["pred_rul"] - sub["true_rul"]) ** 2)
                    ))
                    group_records.append({
                        "dataset": ds, "norm_id": nid, "seed": s,
                        "group": grp, "rmse": rmse, "n": len(sub),
                    })

    grp_df = pd.DataFrame(group_records)
    subgroup_summary = (
        grp_df.groupby(["dataset", "norm_id", "group"])
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), n_engines=("n", "first"))
        .reset_index()
    )
    subgroup_summary.to_csv(RESULTS_DIR / "subgroup_rmse.csv", index=False)
    print("\nSaved subgroup_rmse.csv")

    # -----------------------------------------------------------------------
    # 3. Statistical tests: N2-N7 vs N1 (Wilcoxon rank-sum) + BH-FDR + Cohen's d
    # -----------------------------------------------------------------------
    stat_records = []

    for ds in DATASETS:
        # Gather per-seed RMSE for N1 (baseline)
        baseline_rmse = []
        for s in SEEDS:
            df = load_pred(ds, "N1", s)
            if df is None:
                continue
            baseline_rmse.append(
                float(np.sqrt(np.mean((df["pred_rul"] - df["true_rul"]) ** 2)))
            )

        if len(baseline_rmse) < 2:
            continue

        comparators = ["N2", "N3", "N4", "N5", "N6", "N7"]
        pvals = []
        rows  = []

        for nid in comparators:
            cand_rmse = []
            for s in SEEDS:
                df = load_pred(ds, nid, s)
                if df is None:
                    continue
                cand_rmse.append(
                    float(np.sqrt(np.mean((df["pred_rul"] - df["true_rul"]) ** 2)))
                )
            if len(cand_rmse) < 2:
                pvals.append(np.nan)
                rows.append({
                    "dataset": ds, "norm_id": nid,
                    "mean_rmse_N1": np.mean(baseline_rmse),
                    "mean_rmse_cand": np.nan,
                    "delta_rmse": np.nan,
                    "p_raw": np.nan, "p_bh": np.nan,
                    "cohens_d": np.nan, "significant": False,
                })
                continue

            # Wilcoxon rank-sum (two-sided)
            try:
                _, p = stats.ranksums(baseline_rmse, cand_rmse)
            except Exception:
                p = np.nan
            pvals.append(p)

            # Cohen's d  (paired differences: baseline - cand; positive = cand better)
            diffs = np.array(baseline_rmse) - np.array(cand_rmse)
            pooled_std = np.std(diffs, ddof=1) + 1e-12
            cohens_d   = float(np.mean(diffs) / pooled_std)

            rows.append({
                "dataset":       ds,
                "norm_id":       nid,
                "mean_rmse_N1":  float(np.mean(baseline_rmse)),
                "mean_rmse_cand": float(np.mean(cand_rmse)),
                "delta_rmse":    float(np.mean(baseline_rmse) - np.mean(cand_rmse)),
                "p_raw":         p,
                "p_bh":          np.nan,   # filled after BH correction
                "cohens_d":      cohens_d,
                "significant":   False,    # filled after BH correction
            })

        # BH-FDR correction across comparators for this dataset
        valid_mask = [not np.isnan(p) for p in pvals]
        if any(valid_mask):
            valid_p   = np.array([p for p, v in zip(pvals, valid_mask) if v])
            adj_p     = bh_fdr(valid_p)
            adj_iter  = iter(adj_p)
            for row, is_valid in zip(rows, valid_mask):
                if is_valid:
                    row["p_bh"]       = next(adj_iter)
                    row["significant"] = (
                        row["p_bh"] < 0.05 and abs(row["cohens_d"]) >= 0.3
                    )

        stat_records.extend(rows)

    stat_df = pd.DataFrame(stat_records)
    stat_df.to_csv(RESULTS_DIR / "statistical_test.csv", index=False)
    print("\nSaved statistical_test.csv")
    print(stat_df.to_string(index=False))

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
