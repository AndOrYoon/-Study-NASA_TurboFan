# -*- coding: utf-8 -*-
"""
H2 Hypothesis - Evaluation Module
Reruns the 20 experiments (5 clips x 4 datasets) using LinearRegression,
saves per-engine predictions to raw_predictions/, computes aggregate metrics
with lifetime-group RMSE breakdown, and writes metrics_summary.csv.

Lifetime groups (based on total engine lifetime = last_test_cycle + true_RUL):
  boundary : total_lifetime < 150
  medium   : 150 <= total_lifetime <= 250
  long     : total_lifetime > 250
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = Path(r"C:\BMAD_PY313\Dataset")
RESULTS_DIR = Path(r"C:\BMAD_PY313\Data_Analysis\Results\H1_clipping")
RAW_DIR     = RESULTS_DIR / "raw_predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants  (mirror 02_run_experiments.py exactly)
# ---------------------------------------------------------------------------
COL_NAMES = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

CONST_SENSORS = {
    'FD001': ['s1', 's5', 's6', 's10', 's16', 's18', 's19'],
    'FD002': ['s16'],
    'FD003': ['s1', 's5', 's10', 's16', 's18', 's19'],
    'FD004': ['s16'],
}

CLIP_VALUES = [75, 100, 125, 130, None]
FD_IDS      = [1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(fd_id):
    train = pd.read_csv(DATA_DIR / f'train_FD00{fd_id}.txt',
                        sep=r'\s+', header=None, names=COL_NAMES, engine='python')
    test  = pd.read_csv(DATA_DIR / f'test_FD00{fd_id}.txt',
                        sep=r'\s+', header=None, names=COL_NAMES, engine='python')
    rul   = pd.read_csv(DATA_DIR / f'RUL_FD00{fd_id}.txt',
                        sep=r'\s+', header=None, names=['rul'], engine='python')
    return train, test, rul


def add_rul(train, clip_value=None):
    max_cycles = train.groupby('unit')['cycle'].max()
    train = train.copy()
    train['rul_linear'] = train['unit'].map(max_cycles) - train['cycle']
    train['rul'] = (train['rul_linear'].clip(upper=clip_value)
                    if clip_value is not None else train['rul_linear'])
    return train


def get_feature_cols(fd_id):
    """Active sensor columns (identical to 02_run_experiments.py)."""
    drop = set(['unit', 'cycle', 'op1', 'op2', 'op3'] + CONST_SENSORS[f'FD00{fd_id}'])
    all_sensors = [f's{i}' for i in range(1, 22)]
    return [c for c in all_sensors if c not in drop]


def nasa_score_vec(y_true, y_pred):
    """Per-engine NASA asymmetric score values (not yet summed)."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)


def lifetime_group(total_lifetime):
    if total_lifetime < 150:
        return 'boundary'
    elif total_lifetime <= 250:
        return 'medium'
    else:
        return 'long'


def rmse_group(df_engine, group):
    sub = df_engine[df_engine['lifetime_group'] == group]
    if len(sub) == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(sub['y_true'], sub['y_pred'])))


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(fd_id, clip_value):
    fd_key     = f'FD00{fd_id}'
    clip_label = str(clip_value) if clip_value is not None else 'None'

    train_raw, test_raw, rul_df = load_dataset(fd_id)
    train      = add_rul(train_raw, clip_value=clip_value)
    feat_cols  = get_feature_cols(fd_id)

    X_train = train[feat_cols].values
    y_train = train['rul'].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    # Test: last row per engine, sorted by unit to align with RUL file
    test_last = test_raw.groupby('unit').last().reset_index()
    test_last = test_last.sort_values('unit').reset_index(drop=True)

    # Total lifetime = last observed cycle + true RUL
    last_cycle   = test_last['cycle'].values
    y_true       = rul_df['rul'].values
    n            = min(len(y_true), len(test_last))
    y_true       = y_true[:n]
    last_cycle   = last_cycle[:n]
    total_life   = last_cycle + y_true

    X_test    = test_last[feat_cols].values[:n]
    X_test_sc = scaler.transform(X_test)
    y_pred    = np.clip(model.predict(X_test_sc), 0, None)

    # Per-engine dataframe
    engine_df = pd.DataFrame({
        'dataset':        fd_key,
        'clip_value':     clip_label,
        'engine_id':      np.arange(1, n + 1),
        'y_true':         y_true,
        'y_pred':         y_pred,
        'total_lifetime': total_life,
        'lifetime_group': [lifetime_group(t) for t in total_life],
        'nasa_val':       nasa_score_vec(y_true, y_pred),
        'sq_error':       (y_pred - y_true) ** 2,
    })

    # Save raw predictions
    raw_fname = RAW_DIR / f'{fd_key}_clip{clip_label}.csv'
    engine_df.to_csv(raw_fname, index=False)

    # Aggregate metrics
    rmse_all      = float(np.sqrt(engine_df['sq_error'].mean()))
    nasa_total    = float(engine_df['nasa_val'].sum())
    nasa_mean     = nasa_total / n
    rmse_boundary = rmse_group(engine_df, 'boundary')
    rmse_medium   = rmse_group(engine_df, 'medium')
    rmse_long     = rmse_group(engine_df, 'long')

    n_boundary = (engine_df['lifetime_group'] == 'boundary').sum()
    n_medium   = (engine_df['lifetime_group'] == 'medium').sum()
    n_long     = (engine_df['lifetime_group'] == 'long').sum()

    return {
        'dataset':        fd_key,
        'clip_value':     clip_label,
        'n_engines':      n,
        'rmse':           round(rmse_all, 4),
        'nasa_score':     round(nasa_mean, 4),
        'rmse_boundary':  round(rmse_boundary, 4) if not np.isnan(rmse_boundary) else np.nan,
        'rmse_medium':    round(rmse_medium, 4)   if not np.isnan(rmse_medium)   else np.nan,
        'rmse_long':      round(rmse_long, 4)     if not np.isnan(rmse_long)     else np.nan,
        'n_boundary':     int(n_boundary),
        'n_medium':       int(n_medium),
        'n_long':         int(n_long),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("H2 Evaluate: 5 clips x 4 datasets = 20 experiments")
    print("=" * 70)

    records = []
    for fd_id in FD_IDS:
        print(f"\n--- FD00{fd_id} ---")
        for cv in CLIP_VALUES:
            result = run_experiment(fd_id, cv)
            records.append(result)
            print(f"  clip={result['clip_value']:>4s}  "
                  f"RMSE={result['rmse']:.4f}  "
                  f"NASA={result['nasa_score']:.4f}  "
                  f"[bnd={result['rmse_boundary']}, "
                  f"med={result['rmse_medium']}, "
                  f"lng={result['rmse_long']}]")

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / 'metrics_summary.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Raw per-engine predictions saved to: {RAW_DIR}")

    # Print pivot tables
    print("\n" + "=" * 70)
    print("RMSE pivot (dataset x clip_value):")
    col_order = ['75', '100', '125', '130', 'None']
    pivot = df.pivot(index='dataset', columns='clip_value', values='rmse')
    col_order_f = [c for c in col_order if c in pivot.columns]
    print(pivot[col_order_f].to_string())

    print("\nNASA Score pivot:")
    pivot_n = df.pivot(index='dataset', columns='clip_value', values='nasa_score')
    print(pivot_n[col_order_f].to_string())

    return df


if __name__ == '__main__':
    main()
