# -*- coding: utf-8 -*-
"""
01_unified_data_loader.py
H2-H3 Unified-Control Ad-hoc Analysis

Unified protocol (single source of truth for all conditions A/B/C/D):
  - Backbone    : LSTM(64)×2 → FC(64→32→1)  [H6 full-capacity]
  - Features    : sensor-only (op cols residualized then dropped)
  - Val split   : random engine-level 20%, RandomState(seed)
  - Eval clip   : predictions clamped to [0, 125]
  - RUL clip    : 125 cycles
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]   # C:\BMAD_PY313\Data_Analysis
sys.path.insert(0, str(_ROOT / "Code" / "shared"))
sys.path.insert(0, str(_ROOT / "Code" / "H2_normalization"))
sys.path.insert(0, str(_ROOT / "Code" / "H3_fault_mode" / "phase2_models"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from op_condition_utils import (
    fit_op_condition_kmeans, compute_cluster_means, apply_op_residual,
)
from h6_p2_model_utils import (
    SeqDataset, SeqDatasetM3, split_engines,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DIR = Path(r"C:\BMAD_PY313\Dataset")

COLUMNS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Zero-variance sensors dropped per dataset (matches H5 data loader)
CONST_SENSORS = {
    "FD001": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
    "FD002": ["s16"],
    "FD003": ["s1", "s5", "s10", "s16", "s18", "s19"],
    "FD004": ["s16"],
}

OP_COLS  = ("op1", "op2", "op3")
RUL_CLIP = 125
WINDOW   = 30
BATCH    = 256
MULTI_OP_DATASETS = {"FD002", "FD004"}


# ---------------------------------------------------------------------------
# Raw data helpers
# ---------------------------------------------------------------------------

def _load_raw(dataset: str, split: str) -> pd.DataFrame:
    path = DATASET_DIR / f"{split}_{dataset}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def _sensor_cols(dataset: str) -> list:
    drop = set(CONST_SENSORS.get(dataset, []))
    return [f"s{i}" for i in range(1, 22) if f"s{i}" not in drop]


def _add_rul(df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    df = df.copy()
    max_cyc = df.groupby("unit")["cycle"].max()
    df["rul"] = (df["unit"].map(max_cyc) - df["cycle"]).clip(upper=clip)
    return df


def _load_test_rul(dataset: str) -> np.ndarray:
    path = DATASET_DIR / f"RUL_{dataset}.txt"
    rul = np.loadtxt(path, dtype=np.float32)
    return np.minimum(rul, RUL_CLIP)


# ---------------------------------------------------------------------------
# Sequence builders (sensor_cols already filtered)
# ---------------------------------------------------------------------------

def _make_train_seqs(df: pd.DataFrame, sensor_cols: list):
    X_list, y_list, unit_list = [], [], []
    for uid, grp in df.groupby("unit", sort=True):
        vals = grp[sensor_cols].values.astype(np.float32)
        ruls = grp["rul"].values.astype(np.float32)
        for i in range(len(vals) - WINDOW + 1):
            X_list.append(vals[i: i + WINDOW])
            y_list.append(ruls[i + WINDOW - 1])
            unit_list.append(uid)
    return np.array(X_list), np.array(y_list, np.float32), np.array(unit_list)


def _make_test_seqs(df: pd.DataFrame, sensor_cols: list):
    n_feat = len(sensor_cols)
    X_list = []
    for _, grp in df.groupby("unit", sort=True):
        vals = grp[sensor_cols].values.astype(np.float32)
        if len(vals) >= WINDOW:
            X_list.append(vals[-WINDOW:])
        else:
            pad = np.zeros((WINDOW - len(vals), n_feat), np.float32)
            X_list.append(np.vstack([pad, vals]))
    return np.array(X_list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_data(dataset: str, normalizer, seed: int, K: int = 10):
    """
    Full data preparation pipeline under unified protocol.

    Parameters
    ----------
    dataset    : 'FD001' | 'FD002' | 'FD003' | 'FD004'
    normalizer : FleetMinMax() or PerUnitMinMax() instance (unfitted)
    seed       : integer seed for validation split AND normalizer randomness
    K          : GatingNet prefix length (only used for M3 DatasetM3)

    Returns
    -------
    train_loader_m0  : DataLoader  (SeqDataset)
    val_loader_m0    : DataLoader  (SeqDataset)
    train_loader_m3  : DataLoader  (SeqDatasetM3, K)
    val_loader_m3    : DataLoader  (SeqDatasetM3, K)
    X_test           : np.ndarray  (n_test, WINDOW, n_feat)
    true_rul         : np.ndarray  (n_test,)
    n_features       : int
    """
    sensor_cols = _sensor_cols(dataset)

    # 1. Load raw data
    train_df = _load_raw(dataset, "train")
    test_df  = _load_raw(dataset, "test")

    # 2. Op-condition residualization (FD002/FD004 only) — train-only fit
    if dataset in MULTI_OP_DATASETS:
        scaler, km = fit_op_condition_kmeans(train_df, k=6, op_cols=OP_COLS)
        cluster_means = compute_cluster_means(
            train_df, scaler, km, sensor_cols, op_cols=OP_COLS
        )
        train_df = apply_op_residual(
            train_df, scaler, km, cluster_means, sensor_cols, op_cols=OP_COLS
        )
        test_df = apply_op_residual(
            test_df, scaler, km, cluster_means, sensor_cols, op_cols=OP_COLS
        )

    # 3. RUL labels (train only)
    train_df = _add_rul(train_df)

    # 4. Normalization — fit on train sensor values only
    # H5 normalizers return a DataFrame of ONLY the feature cols; re-attach meta cols
    normalizer.fit(train_df, sensor_cols)
    train_norm = train_df.copy()
    train_norm[sensor_cols] = normalizer.transform_train(train_df, sensor_cols).values
    test_norm = test_df.copy()
    test_norm[sensor_cols]  = normalizer.transform_test(test_df, sensor_cols).values

    # 5. Build sequences
    X_all, y_all, units_all = _make_train_seqs(train_norm, sensor_cols)
    X_test = _make_test_seqs(test_norm, sensor_cols)
    true_rul = _load_test_rul(dataset)

    # 6. Random engine-level 20% validation split
    tr_mask, va_mask = split_engines(units_all, val_frac=0.2, seed=seed)
    X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
    X_va, y_va = X_all[va_mask], y_all[va_mask]

    # 7. DataLoaders — M0 (SeqDataset) and M3 (SeqDatasetM3)
    train_loader_m0 = DataLoader(
        SeqDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader_m0 = DataLoader(
        SeqDataset(X_va, y_va), batch_size=BATCH, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    train_loader_m3 = DataLoader(
        SeqDatasetM3(X_tr, y_tr, K=K), batch_size=BATCH, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader_m3 = DataLoader(
        SeqDatasetM3(X_va, y_va, K=K), batch_size=BATCH, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    n_features = len(sensor_cols)
    return (
        train_loader_m0, val_loader_m0,
        train_loader_m3, val_loader_m3,
        X_test, true_rul, n_features,
    )


# ---------------------------------------------------------------------------
# Normalizer shim: PerUnitMinMax needs unit column for transform_test
# We patch transform_test to accept a plain array path via a wrapper.
# ---------------------------------------------------------------------------

class _FleetMinMaxAdapter:
    """Thin wrapper so FleetMinMax works with the prepare_data API."""
    name = "N1"

    def __init__(self):
        from h5_normalizer_import import FleetMinMax as _FM
        self._inner = _FM()

    def fit(self, df, cols):
        self._inner.fit(df, cols)

    def transform_train(self, df, cols):
        return self._inner.transform_train(df, cols)

    def transform_test(self, df, cols):
        return self._inner.transform_test(df, cols)


# Rather than wrapper classes, callers import directly from 02_normalizers.py.
# The prepare_data() function expects normalizer objects with the H5 API:
#   .fit(train_df, sensor_cols)
#   .transform_train(train_df, sensor_cols) -> pd.DataFrame with sensor_cols
#   .transform_test(test_df, sensor_cols)   -> pd.DataFrame with sensor_cols


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import importlib
    norms_mod     = importlib.import_module("02_normalizers")
    FleetMinMax   = norms_mod.FleetMinMax
    PerUnitMinMax = norms_mod.PerUnitMinMax

    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        for norm_cls, norm_name in [(FleetMinMax, "N1"), (PerUnitMinMax, "N3")]:
            tr0, va0, tr3, va3, X_te, rul, n_feat = prepare_data(
                ds, norm_cls(), seed=0
            )
            print(
                f"{ds} {norm_name} | n_feat={n_feat} "
                f"train_batches={len(tr0)} val_batches={len(va0)} "
                f"test={X_te.shape[0]} engines"
            )
