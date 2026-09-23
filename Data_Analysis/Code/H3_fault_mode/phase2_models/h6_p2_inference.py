# -*- coding: utf-8 -*-
"""
H6 Phase 2 — Inference Utilities
Provides GMM-based test-engine cluster assignment for M1 (hard) and M2 (soft).
Used by the training scripts after model training to generate test predictions.
"""

import numpy as np
import pandas as pd
import os
import pickle
import sys

sys.path.insert(0, r"C:\BMAD_PY313\Data_Analysis\Code\shared")
import op_condition_utils as ocu

MODELS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H3_fault_mode\models"
OP_COLS = ("op_setting_1", "op_setting_2", "op_setting_3")
FAULT_DISCRIMINANT_SENSORS = ["s15", "s20", "s21", "s7", "s12", "s2", "s4"]


def load_gmm_bundle(dataset: str, variant: str = "full") -> dict:
    pkl_path = os.path.join(MODELS_DIR, f"gmm_{dataset.lower()}_{variant}.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_op_artifacts():
    with open(os.path.join(MODELS_DIR, "op_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "op_kmeans.pkl"), "rb") as f:
        km = pickle.load(f)
    cluster_means = pd.read_csv(
        os.path.join(MODELS_DIR, "op_cluster_means.csv"), index_col=0
    )
    return scaler, km, cluster_means


def _compute_late_mean(arr: np.ndarray, pct: float = 0.20) -> float:
    n = max(1, int(len(arr) * pct))
    return float(arr[-n:].mean())


def _compute_slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def extract_engine_features(engine_df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """
    Extract GMM feature vector from a single engine's time series.
    Feature columns must match bundle['feature_cols'].
    Returns scaled 2D array shape (1, n_features).
    """
    feat_cols = bundle["feature_cols"]
    row = {}
    for fc in feat_cols:
        if fc.endswith("_late_mean"):
            s = fc[: -len("_late_mean")]
            row[fc] = _compute_late_mean(engine_df[s].values)
        elif fc.endswith("_slope"):
            s = fc[: -len("_slope")]
            row[fc] = _compute_slope(engine_df[s].values)
        else:
            row[fc] = 0.0

    feat_vec = np.array([[row[fc] for fc in feat_cols]], dtype=np.float32)
    return bundle["scaler"].transform(feat_vec)


def assign_test_engine_hard(engine_df: pd.DataFrame, bundle: dict) -> int:
    """Return hard cluster assignment (0 or 1)."""
    X_sc = extract_engine_features(engine_df, bundle)
    return int(bundle["gmm"].predict(X_sc)[0])


def assign_test_engine_soft(engine_df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Return soft cluster probabilities [p0, p1]."""
    X_sc = extract_engine_features(engine_df, bundle)
    return bundle["gmm"].predict_proba(X_sc)[0]   # shape (2,)


def get_test_cluster_assignments(test_df: pd.DataFrame, dataset: str, hard: bool = True):
    """
    Batch test-engine cluster assignment.
    For FD004: applies op-condition residualization before feature extraction.

    Returns
    -------
    If hard=True : np.ndarray of int, shape (n_test_engines,)
    If hard=False: np.ndarray of float, shape (n_test_engines, 2)
    """
    bundle = load_gmm_bundle(dataset)

    # Apply op residualization for FD004
    active_sensors = [s for s in FAULT_DISCRIMINANT_SENSORS
                      if s in test_df.columns]
    if dataset.upper() == "FD004":
        scaler_op, km_op, cm_op = load_op_artifacts()
        test_df = test_df.copy()
        op_sc   = scaler_op.transform(test_df[list(OP_COLS)])
        test_df["_opc"] = km_op.predict(op_sc)
        for s in active_sensors:
            test_df[s] = test_df[s] - test_df["_opc"].map(cm_op[s])
        test_df = test_df.drop(columns=["_opc"])

    results = []
    for _, grp in test_df.groupby("unit_number"):
        if hard:
            results.append(assign_test_engine_hard(grp, bundle))
        else:
            results.append(assign_test_engine_soft(grp, bundle))

    return np.array(results)
