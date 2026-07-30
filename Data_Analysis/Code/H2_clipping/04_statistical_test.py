# -*- coding: utf-8 -*-
"""
H2 Hypothesis - Statistical Tests
Wilcoxon rank-sum test: each clip threshold vs clip_125 baseline,
on per-engine RMSE (squared-error -> sqrt per engine).

Reads: C:\BMAD_PY313\Data_Analysis\Results\H2_clipping\raw_predictions\*.csv
Writes: C:\BMAD_PY313\Data_Analysis\Results\H2_clipping\statistical_test.csv

Columns: dataset, clip_value, p_value, significant (alpha=0.05),
         better_than_125 (mean RMSE of clip < mean RMSE of clip_125)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(r"C:\BMAD_PY313\Data_Analysis\Results\H2_clipping")
RAW_DIR     = RESULTS_DIR / "raw_predictions"
OUT_PATH    = RESULTS_DIR / "statistical_test.csv"

CLIP_VALUES = ['75', '100', '125', '130', 'None']
FD_IDS      = [1, 2, 3, 4]
ALPHA       = 0.05
BASELINE    = '125'

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not RAW_DIR.exists() or not any(RAW_DIR.glob('*.csv')):
        raise FileNotFoundError(
            f"No raw prediction files found in {RAW_DIR}. "
            "Please run 03_evaluate.py first."
        )

    records = []

    for fd_id in FD_IDS:
        fd_key = f'FD00{fd_id}'

        # Load baseline (clip_125)
        base_path = RAW_DIR / f'{fd_key}_clip{BASELINE}.csv'
        if not base_path.exists():
            print(f"  [WARN] Baseline file not found: {base_path}")
            continue
        base_df   = pd.read_csv(base_path)
        base_rmse = np.sqrt(base_df['sq_error'].values)   # per-engine RMSE

        for cv in CLIP_VALUES:
            if cv == BASELINE:
                # baseline vs itself: skip Wilcoxon, fill sentinel
                records.append({
                    'dataset':         fd_key,
                    'clip_value':      cv,
                    'mean_rmse':       round(float(base_rmse.mean()), 4),
                    'mean_rmse_125':   round(float(base_rmse.mean()), 4),
                    'p_value':         np.nan,
                    'significant':     False,
                    'better_than_125': False,
                })
                continue

            pred_path = RAW_DIR / f'{fd_key}_clip{cv}.csv'
            if not pred_path.exists():
                print(f"  [WARN] Missing: {pred_path}")
                continue

            pred_df   = pd.read_csv(pred_path)
            pred_rmse = np.sqrt(pred_df['sq_error'].values)

            # Wilcoxon rank-sum (Mann-Whitney U) - two-sided
            stat, p_val = stats.ranksums(pred_rmse, base_rmse)

            mean_pred = float(pred_rmse.mean())
            mean_base = float(base_rmse.mean())
            significant    = bool(p_val < ALPHA)
            better_than_125 = bool(mean_pred < mean_base)

            print(f"  {fd_key} clip={cv:>4s} vs clip=125: "
                  f"mean_rmse={mean_pred:.4f} vs {mean_base:.4f}  "
                  f"p={p_val:.4f}  sig={significant}  better={better_than_125}")

            records.append({
                'dataset':         fd_key,
                'clip_value':      cv,
                'mean_rmse':       round(mean_pred, 4),
                'mean_rmse_125':   round(mean_base, 4),
                'p_value':         round(float(p_val), 6),
                'significant':     significant,
                'better_than_125': better_than_125,
            })

    df = pd.DataFrame(records)
    # Reorder columns
    col_order = ['dataset', 'clip_value', 'mean_rmse', 'mean_rmse_125',
                 'p_value', 'significant', 'better_than_125']
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    # Summary pivot
    print("\n" + "=" * 70)
    print("Wilcoxon p-values (clip_X vs clip_125 baseline):")
    pivot = df[df['clip_value'] != BASELINE].pivot(
        index='dataset', columns='clip_value', values='p_value'
    )
    col_order_p = ['75', '100', '130', 'None']
    col_order_p = [c for c in col_order_p if c in pivot.columns]
    print(pivot[col_order_p].to_string())

    print("\nbetter_than_125 (True = lower mean per-engine RMSE):")
    pivot_b = df[df['clip_value'] != BASELINE].pivot(
        index='dataset', columns='clip_value', values='better_than_125'
    )
    print(pivot_b[[c for c in col_order_p if c in pivot_b.columns]].to_string())

    return df


if __name__ == '__main__':
    print("=" * 70)
    print("H2 Statistical Tests: Wilcoxon rank-sum vs clip_125 baseline")
    print("=" * 70)
    main()
