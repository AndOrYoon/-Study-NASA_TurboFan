# -*- coding: utf-8 -*-
"""
H2 Hypothesis - Data Loader Module
RUL Clipping Threshold Optimization Study
NASA CMAPSS TurboFan Dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\BMAD_PY313\Dataset")
RESULTS_DIR = Path(r"C:\BMAD_PY313\Data_Analysis\Results\H1_clipping")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COL_NAMES = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

# Constant sensors to drop per dataset (identified during EDA)
CONST_SENSORS = {
    'FD001': ['s1', 's5', 's6', 's10', 's16', 's18', 's19'],
    'FD002': ['s16'],
    'FD003': ['s1', 's5', 's10', 's16', 's18', 's19'],
    'FD004': ['s16'],
}

# H2 experiment: compare these clip thresholds
CLIP_VALUES = [75, 100, 125, 130, None]


def load_dataset(fd_id):
    """Load train, test, and RUL files for a given FD dataset ID (1-4)."""
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
    """
    Add RUL labels to training data.
    RUL_linear = max_cycle_per_unit - current_cycle
    RUL = min(RUL_linear, clip_value) if clip_value is not None else RUL_linear
    """
    max_cycles = train.groupby('unit')['cycle'].max()
    train = train.copy()
    train['rul_linear'] = train['unit'].map(max_cycles) - train['cycle']
    if clip_value is not None:
        train['rul'] = train['rul_linear'].clip(upper=clip_value)
    else:
        train['rul'] = train['rul_linear']
    return train


def get_features(df, fd_id):
    """Drop non-sensor columns and constant sensors; return feature DataFrame."""
    drop_cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + CONST_SENSORS[f'FD00{fd_id}']
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def get_pct_at_ceiling(train_with_rul, clip_value):
    """Fraction of training rows where RUL == clip_value (stuck at ceiling)."""
    if clip_value is None:
        return 0.0
    return float((train_with_rul['rul'] == clip_value).mean())


def get_last_row_per_unit(df):
    """Return the last observed row for each engine unit."""
    return df.groupby('unit').last().reset_index()


if __name__ == '__main__':
    # Quick sanity check
    for fd_id in [1, 2, 3, 4]:
        train, test, rul = load_dataset(fd_id)
        print(f"FD00{fd_id}: train={train.shape}, test={test.shape}, rul={rul.shape}")
        train_rul = add_rul(train, clip_value=125)
        pct = get_pct_at_ceiling(train_rul, 125)
        feats = get_features(train_rul, fd_id)
        print(f"  Features: {feats.shape[1]} sensors, pct_at_ceiling(125)={pct:.3f}")
