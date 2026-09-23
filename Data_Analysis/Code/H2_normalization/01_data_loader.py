# -*- coding: utf-8 -*-
"""
01_data_loader.py
-----------------
Load NASA CMAPSS TurboFan data, assign RUL labels (clip=125),
drop constant sensors, and generate sliding-window sequences.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"C:\BMAD_PY313\Dataset")

COLUMNS = (
    ["unit", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

CONST_SENSORS = {
    "FD001": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
    "FD002": ["s16"],
    "FD003": ["s1", "s5", "s10", "s16", "s18", "s19"],
    "FD004": ["s16"],
}

RUL_CLIP = 125
WINDOW   = 30


# ---------------------------------------------------------------------------
# Load raw text
# ---------------------------------------------------------------------------

def load_raw(dataset: str, split: str) -> pd.DataFrame:
    """Load train or test file for a given dataset (e.g. 'FD001')."""
    path = DATA_DIR / f"{split}_{dataset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df


def drop_const_sensors(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Drop known constant-sensor columns for a given dataset."""
    return df.drop(columns=CONST_SENSORS[dataset], errors="ignore")


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return sensor + op-setting column names (everything except unit/cycle/rul)."""
    exclude = {"unit", "cycle", "rul"}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# RUL labeling
# ---------------------------------------------------------------------------

def add_rul_labels(train: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    """Add piecewise-linear capped RUL column to the training dataframe."""
    max_cycle = train.groupby("unit")["cycle"].max()
    train = train.copy()
    train["rul"] = train.apply(
        lambda r: min(max_cycle[r["unit"]] - r["cycle"], clip), axis=1
    )
    return train


# ---------------------------------------------------------------------------
# Sliding-window sequence generation
# ---------------------------------------------------------------------------

def make_sequences(df: pd.DataFrame, feature_cols: list, window: int = WINDOW):
    """
    Generate (X, y) sliding-window sequences.

    Returns
    -------
    X : np.ndarray, shape (N, window, n_features)
    y : np.ndarray, shape (N,)
    """
    X_list, y_list = [], []

    for _, group in df.groupby("unit", sort=False):
        feats = group[feature_cols].values.astype(np.float32)
        ruls  = group["rul"].values.astype(np.float32)
        n     = len(group)

        for i in range(n - window + 1):
            X_list.append(feats[i : i + window])
            y_list.append(ruls[i + window - 1])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def make_test_sequences(df: pd.DataFrame, feature_cols: list, window: int = WINDOW):
    """
    For each test engine, take the LAST `window` cycles.
    Pad with the first cycle if the engine has fewer than `window` cycles.

    Returns
    -------
    X : np.ndarray, shape (n_engines, window, n_features)
    unit_ids : list of unit numbers in order
    """
    X_list, unit_ids = [], []

    for uid, group in df.groupby("unit", sort=True):
        feats = group[feature_cols].values.astype(np.float32)
        n     = len(feats)

        if n >= window:
            seq = feats[-window:]
        else:
            # pad from the front with the first cycle repeated
            pad_len = window - n
            pad     = np.tile(feats[0:1], (pad_len, 1))
            seq     = np.concatenate([pad, feats], axis=0)

        X_list.append(seq)
        unit_ids.append(uid)

    X = np.stack(X_list, axis=0)
    return X, unit_ids


# ---------------------------------------------------------------------------
# Load true RUL for test set
# ---------------------------------------------------------------------------

def load_true_rul(dataset: str, clip: int = RUL_CLIP) -> np.ndarray:
    """Load RUL_FDxxx.txt and clip to `clip`."""
    path = DATA_DIR / f"RUL_{dataset}.txt"
    rul  = np.loadtxt(path, dtype=np.float32)
    return np.minimum(rul, clip)


# ---------------------------------------------------------------------------
# Lifetime distribution helpers  (used by 06_boundary_analysis.py)
# ---------------------------------------------------------------------------

def get_lifetime_groups(train: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-engine lifetime with group label.

    Groups:
        boundary : < 50  cycles
        short    : 50-149 cycles
        medium   : 150-249 cycles
        long     : >= 250 cycles
    """
    lifetimes = train.groupby("unit")["cycle"].max().reset_index()
    lifetimes.columns = ["unit", "lifetime"]

    def _label(lt):
        if lt < 150:
            return "boundary"   # EDA min=128; <50 is always empty across all datasets
        elif lt < 250:
            return "medium"
        else:
            return "long"

    lifetimes["group"] = lifetimes["lifetime"].apply(_label)
    return lifetimes


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        train = load_raw(ds, "train")
        train = drop_const_sensors(train, ds)
        train = add_rul_labels(train)

        test = load_raw(ds, "test")
        test = drop_const_sensors(test, ds)

        feat_cols = get_feature_cols(train)
        n_feat    = len(feat_cols)
        n_engines = train["unit"].nunique()

        print(f"{ds}: {n_engines} engines, {n_feat} features, "
              f"train rows={len(train)}, test rows={len(test)}")

        X_tr, y_tr = make_sequences(train, feat_cols)
        X_te, _    = make_test_sequences(test, feat_cols)
        rul_te     = load_true_rul(ds)

        print(f"  X_train={X_tr.shape}, y_train={y_tr.shape}")
        print(f"  X_test={X_te.shape},  true_rul={rul_te.shape}")
