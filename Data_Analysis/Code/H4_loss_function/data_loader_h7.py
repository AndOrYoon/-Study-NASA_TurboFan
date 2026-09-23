"""
H7 Loss Function Optimization - Data Loader
Loads CMAPSS train/test data, adds RUL labels, applies clipping,
computes life_ratio for train/val, returns normalised feature arrays.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config_h7 import DATA_DIR, COL_NAMES, CONST_SENSORS, WINDOW, VAL_FRAC


# ─────────────────────────────────────────────────────────────────────────────
def _read_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)
    return df


def _add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'rul' column (raw, un-clipped)."""
    max_cyc = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cyc, on="unit")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def _feature_cols(dataset: str) -> list:
    op_cols     = ["op1", "op2", "op3"]
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    drop_s      = CONST_SENSORS.get(dataset, [])
    keep_s      = [c for c in sensor_cols if c not in drop_s]
    return op_cols + keep_s


# ─────────────────────────────────────────────────────────────────────────────
def load_train_val(dataset: str, clip_value, scaler=None):
    """
    Returns
    -------
    train_engines, val_engines, scaler, feat_cols, clip_used
    Each engine: {'unit', 'X', 'rul', 'life_ratio'}
    """
    path = os.path.join(DATA_DIR, f"train_{dataset}.txt")
    df   = _read_raw(path)
    df   = _add_rul(df)

    feat_cols = _feature_cols(dataset)

    max_rul   = int(df["rul"].max())
    clip_used = clip_value if clip_value is not None else max_rul
    df["rul_clipped"] = df["rul"].clip(upper=clip_used)
    df["life_ratio"]  = 1.0 - df["rul_clipped"] / clip_used

    # ── Engine-level train / val split (prevents RUL distribution mismatch) ────
    # Per-cycle split causes train(high RUL) vs val(low RUL) mismatch → early stop
    units   = np.array(sorted(df["unit"].unique()))
    n_val   = max(1, int(len(units) * VAL_FRAC))
    val_set = set(units[-n_val:].tolist())
    train_df = df[~df["unit"].isin(val_set)].reset_index(drop=True)
    val_df   = df[ df["unit"].isin(val_set)].reset_index(drop=True)

    # ── Fit scaler on train feature columns ──────────────────────────────────
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(train_df[feat_cols].values)

    def _build_engines(sub_df):
        engines = []
        for uid, g in sub_df.groupby("unit", sort=True):
            g      = g.sort_values("cycle")
            X_raw  = g[feat_cols].values.astype(np.float32)
            X_norm = scaler.transform(X_raw).astype(np.float32)
            rul    = g["rul_clipped"].values.astype(np.float32)
            lr     = g["life_ratio"].values.astype(np.float32)
            engines.append({"unit": uid, "X": X_norm,
                            "rul": rul, "life_ratio": lr})
        return engines

    train_engines = _build_engines(train_df)
    val_engines   = _build_engines(val_df)

    return train_engines, val_engines, scaler, feat_cols, clip_used


# ─────────────────────────────────────────────────────────────────────────────
def load_test(dataset: str, scaler, feat_cols: list):
    """
    Returns
    -------
    X_windows : (N_engines, WINDOW, n_features) float32
    rul_true  : (N_engines,) float32
    """
    test_path = os.path.join(DATA_DIR, f"test_{dataset}.txt")
    rul_path  = os.path.join(DATA_DIR, f"RUL_{dataset}.txt")

    df      = _read_raw(test_path)
    rul_arr = pd.read_csv(rul_path, header=None).squeeze().values.astype(np.float32)

    windows = []
    for uid, g in df.groupby("unit", sort=True):
        g      = g.sort_values("cycle")
        X_raw  = g[feat_cols].values.astype(np.float32)
        X_norm = scaler.transform(X_raw).astype(np.float32)
        if len(X_norm) >= WINDOW:
            win = X_norm[-WINDOW:]
        else:
            pad = np.repeat(X_norm[:1], WINDOW - len(X_norm), axis=0)
            win = np.vstack([pad, X_norm])
        windows.append(win)

    X_windows = np.stack(windows, axis=0)   # (N, W, F)
    return X_windows, rul_arr


# ─────────────────────────────────────────────────────────────────────────────
def build_windows(engines: list, window: int = WINDOW):
    """
    Sliding-window expansion of engine dicts.

    Returns
    -------
    X          : (N_samples, window, n_features) float32
    y          : (N_samples,) float32
    life_ratio : (N_samples,) float32
    """
    X_list, y_list, lr_list = [], [], []
    for eng in engines:
        T = len(eng["rul"])
        for t in range(T):
            start = max(0, t - window + 1)
            seg   = eng["X"][start: t + 1]
            if len(seg) < window:
                pad = np.repeat(seg[:1], window - len(seg), axis=0)
                seg = np.vstack([pad, seg])
            X_list.append(seg)
            y_list.append(eng["rul"][t])
            lr_list.append(eng["life_ratio"][t])

    X          = np.stack(X_list,  axis=0).astype(np.float32)
    y          = np.array(y_list,  dtype=np.float32)
    life_ratio = np.array(lr_list, dtype=np.float32)
    return X, y, life_ratio
