# -*- coding: utf-8 -*-
"""
H2 Hypothesis - RUL Clipping Threshold Experiment
Compares 5 clip thresholds x 4 datasets using Linear Regression.
Metrics: RMSE, NASA Score, pct_at_ceiling

Experimental matrix: 5 clips x 4 datasets = 20 configurations (deterministic)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"C:\BMAD_PY313\Dataset")
RESULTS_DIR = Path(r"C:\BMAD_PY313\Data_Analysis\Results\H1_clipping")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COL_NAMES = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

CONST_SENSORS = {
    'FD001': ['s1', 's5', 's6', 's10', 's16', 's18', 's19'],
    'FD002': ['s16'],
    'FD003': ['s1', 's5', 's10', 's16', 's18', 's19'],
    'FD004': ['s16'],
}

CLIP_VALUES = [75, 100, 125, 130, None]

FD_IDS = [1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(fd_id):
    train = pd.read_csv(
        DATA_DIR / f'train_FD00{fd_id}.txt',
        sep=r'\s+', header=None, names=COL_NAMES, engine='python'
    )
    test = pd.read_csv(
        DATA_DIR / f'test_FD00{fd_id}.txt',
        sep=r'\s+', header=None, names=COL_NAMES, engine='python'
    )
    rul = pd.read_csv(
        DATA_DIR / f'RUL_FD00{fd_id}.txt',
        sep=r'\s+', header=None, names=['rul'], engine='python'
    )
    return train, test, rul


def add_rul(train, clip_value=None):
    max_cycles = train.groupby('unit')['cycle'].max()
    train = train.copy()
    train['rul_linear'] = train['unit'].map(max_cycles) - train['cycle']
    if clip_value is not None:
        train['rul'] = train['rul_linear'].clip(upper=clip_value)
    else:
        train['rul'] = train['rul_linear']
    return train


def get_feature_cols(fd_id):
    drop = ['unit', 'cycle', 'op1', 'op2', 'op3'] + CONST_SENSORS[f'FD00{fd_id}']
    all_sensor_cols = [f's{i}' for i in range(1, 22)]
    op_cols = ['op1', 'op2', 'op3']
    # Keep only sensor cols not in drop list
    keep = [c for c in (all_sensor_cols) if c not in drop]
    return keep


def get_pct_at_ceiling(train_with_rul, clip_value):
    if clip_value is None:
        return 0.0
    return float((train_with_rul['rul'] == clip_value).mean())


# ---------------------------------------------------------------------------
# NASA Score
# ---------------------------------------------------------------------------

def nasa_score(y_true, y_pred):
    """
    NASA asymmetric scoring function.
    Penalises late predictions more heavily than early predictions.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(s))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(fd_id, clip_value):
    """
    Train LinearRegression on all rows of training data (X=sensor values, y=clipped RUL).
    Predict on last row of each test engine.
    Returns dict with metrics.
    """
    fd_key = f'FD00{fd_id}'
    clip_label = str(clip_value) if clip_value is not None else 'None'

    # Load data
    train_raw, test_raw, rul_df = load_dataset(fd_id)

    # Compute RUL labels for training set
    train = add_rul(train_raw, clip_value=clip_value)

    # Feature columns (active sensors only)
    feat_cols = get_feature_cols(fd_id)

    # --- Training set ---
    X_train = train[feat_cols].values
    y_train = train['rul'].values

    pct_ceil = get_pct_at_ceiling(train, clip_value)

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    # Train linear model
    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    # --- Test set: use LAST row per engine ---
    test_last = test_raw.groupby('unit').last().reset_index()
    # Sort by unit to align with RUL file (engines are numbered 1..N)
    test_last = test_last.sort_values('unit').reset_index(drop=True)

    X_test = test_last[feat_cols].values
    X_test_sc = scaler.transform(X_test)

    y_pred = model.predict(X_test_sc)
    y_pred = np.clip(y_pred, 0, None)   # RUL cannot be negative

    y_true = rul_df['rul'].values

    # Align lengths (should match, but guard against edge cases)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ns   = nasa_score(y_true, y_pred)

    # RUL distribution stats on training labels
    rul_mean = float(train['rul'].mean())
    rul_std  = float(train['rul'].std())
    rul_max  = float(train['rul'].max())

    return {
        'dataset':        fd_key,
        'clip_value':     clip_label,
        'n_train_rows':   len(X_train),
        'n_test_engines': n,
        'n_features':     len(feat_cols),
        'pct_at_ceiling': round(pct_ceil, 4),
        'rul_mean':       round(rul_mean, 2),
        'rul_std':        round(rul_std, 2),
        'rul_max':        round(rul_max, 2),
        'rmse':           round(rmse, 4),
        'nasa_score':     round(ns, 2),
    }


def main():
    print("=" * 65)
    print("H2 Hypothesis: RUL Clipping Threshold Optimization")
    print("Model: Linear Regression  |  Datasets: FD001-FD004")
    print("=" * 65)

    records = []

    for fd_id in FD_IDS:
        print(f"\n--- Dataset FD00{fd_id} ---")
        for clip_val in CLIP_VALUES:
            clip_label = str(clip_val) if clip_val is not None else 'None'
            result = run_experiment(fd_id, clip_val)
            records.append(result)
            print(
                f"  clip={clip_label:>4s}  "
                f"pct_ceil={result['pct_at_ceiling']:.3f}  "
                f"RMSE={result['rmse']:.2f}  "
                f"NASA={result['nasa_score']:.1f}"
            )

    # Save results
    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / 'h2_results.csv'
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # Print summary table
    print("\n" + "=" * 65)
    print("SUMMARY TABLE: RMSE by (Dataset, Clip Value)")
    print("=" * 65)
    pivot_rmse = df.pivot(index='dataset', columns='clip_value', values='rmse')
    # Reorder columns: 75, 100, 125, 130, None
    col_order = ['75', '100', '125', '130', 'None']
    col_order = [c for c in col_order if c in pivot_rmse.columns]
    print(pivot_rmse[col_order].to_string())

    print("\n" + "=" * 65)
    print("SUMMARY TABLE: NASA Score by (Dataset, Clip Value)")
    print("=" * 65)
    pivot_nasa = df.pivot(index='dataset', columns='clip_value', values='nasa_score')
    print(pivot_nasa[col_order].to_string())

    print("\n" + "=" * 65)
    print("SUMMARY TABLE: pct_at_ceiling by (Dataset, Clip Value)")
    print("=" * 65)
    pivot_pct = df.pivot(index='dataset', columns='clip_value', values='pct_at_ceiling')
    col_order_pct = [c for c in col_order if c in pivot_pct.columns]
    print(pivot_pct[col_order_pct].to_string())

    print("\nDone.")
    return df


if __name__ == '__main__':
    main()
