# -*- coding: utf-8 -*-
"""
02_normalizers.py
-----------------
Six normalization strategies for H5 hypothesis:

  Fleet-level  : FleetMinMax, FleetStd
  Per-unit     : PerUnitMinMax(n_init), PerUnitStd(n_init)

API convention
--------------
All normalizers expose:
    fit(train_df, feature_cols)          -> self
    transform_train(train_df, feat_cols) -> pd.DataFrame  (same index)
    transform_test(test_df, feat_cols)   -> pd.DataFrame  (same index)

The returned DataFrames contain ONLY the feature columns (normalized).
The caller is responsible for re-attaching unit/cycle/rul.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: safe division (avoid div-by-zero)
# ---------------------------------------------------------------------------

def _safe_scale(arr: np.ndarray, center: np.ndarray, scale: np.ndarray,
                eps: float = 1e-8) -> np.ndarray:
    """(arr - center) / max(scale, eps)."""
    scale = np.where(np.abs(scale) < eps, eps, scale)
    return (arr - center) / scale


# ---------------------------------------------------------------------------
# Fleet-level MinMax
# ---------------------------------------------------------------------------

class FleetMinMax:
    """Min-max scale each feature to [0, 1] using fleet-wide train statistics."""

    name = "fleet_minmax"

    def __init__(self):
        self.min_ = None
        self.range_ = None

    def fit(self, train_df: pd.DataFrame, feature_cols: list) -> "FleetMinMax":
        vals = train_df[feature_cols].values.astype(np.float64)
        self.min_   = vals.min(axis=0)
        self.range_ = vals.max(axis=0) - self.min_
        return self

    def _apply(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        vals = df[feature_cols].values.astype(np.float64)
        scaled = _safe_scale(vals, self.min_, self.range_)
        result = df[feature_cols].copy()
        result[feature_cols] = scaled
        return result

    def transform_train(self, train_df: pd.DataFrame,
                        feature_cols: list) -> pd.DataFrame:
        return self._apply(train_df, feature_cols)

    def transform_test(self, test_df: pd.DataFrame,
                       feature_cols: list) -> pd.DataFrame:
        return self._apply(test_df, feature_cols)


# ---------------------------------------------------------------------------
# Fleet-level Standard Scaling
# ---------------------------------------------------------------------------

class FleetStd:
    """Standard (z-score) scaling using fleet-wide train mean/std."""

    name = "fleet_std"

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, train_df: pd.DataFrame, feature_cols: list) -> "FleetStd":
        vals = train_df[feature_cols].values.astype(np.float64)
        self.mean_ = vals.mean(axis=0)
        self.std_  = vals.std(axis=0, ddof=0)
        return self

    def _apply(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        vals   = df[feature_cols].values.astype(np.float64)
        scaled = _safe_scale(vals, self.mean_, self.std_)
        result = df[feature_cols].copy()
        result[feature_cols] = scaled
        return result

    def transform_train(self, train_df: pd.DataFrame,
                        feature_cols: list) -> pd.DataFrame:
        return self._apply(train_df, feature_cols)

    def transform_test(self, test_df: pd.DataFrame,
                       feature_cols: list) -> pd.DataFrame:
        return self._apply(test_df, feature_cols)


# ---------------------------------------------------------------------------
# Per-unit MinMax
# ---------------------------------------------------------------------------

class PerUnitMinMax:
    """
    Min-max scale each feature using each engine's OWN first n_init cycles.

    This eliminates initial condition uncertainty: sensor offsets due to
    manufacturing variation or installation differences are removed by
    referencing each engine's own early-life baseline.

    Training:  stats computed from engine's first n_init cycles in train set.
    Testing:   stats computed from engine's first n_init cycles in test set.
               (Test files contain the full run up to near-EOL, so early
                cycles are available for reference.)
    """

    def __init__(self, n_init: int = 5):
        self.n_init = n_init
        self.name   = f"perunit_minmax_{n_init}"
        # dict: unit_id -> (min_arr, range_arr)
        self._train_stats: dict = {}

    def fit(self, train_df: pd.DataFrame, feature_cols: list) -> "PerUnitMinMax":
        self._train_stats = {}
        for uid, grp in train_df.groupby("unit", sort=True):
            early = grp[feature_cols].head(self.n_init).values.astype(np.float64)
            mn    = early.min(axis=0)
            rng   = early.max(axis=0) - mn
            self._train_stats[uid] = (mn, rng)
        return self

    def _apply_perunit(self, df: pd.DataFrame, feature_cols: list,
                       stats: dict) -> pd.DataFrame:
        result = df[feature_cols].copy().astype(np.float64)
        for uid, grp in df.groupby("unit", sort=True):
            if uid in stats:
                mn, rng = stats[uid]
            else:
                # fallback: compute from this engine's available data
                vals = grp[feature_cols].values.astype(np.float64)
                early = vals[: self.n_init]
                mn    = early.min(axis=0)
                rng   = early.max(axis=0) - mn
            vals   = grp[feature_cols].values.astype(np.float64)
            scaled = _safe_scale(vals, mn, rng)
            result.loc[grp.index, feature_cols] = scaled
        return result

    def transform_train(self, train_df: pd.DataFrame,
                        feature_cols: list) -> pd.DataFrame:
        return self._apply_perunit(train_df, feature_cols, self._train_stats)

    def transform_test(self, test_df: pd.DataFrame,
                       feature_cols: list) -> pd.DataFrame:
        # For test engines: compute stats fresh from test data (no train leakage)
        test_stats = {}
        for uid, grp in test_df.groupby("unit", sort=True):
            early = grp[feature_cols].head(self.n_init).values.astype(np.float64)
            mn    = early.min(axis=0)
            rng   = early.max(axis=0) - mn
            test_stats[uid] = (mn, rng)
        return self._apply_perunit(test_df, feature_cols, test_stats)


# ---------------------------------------------------------------------------
# Per-unit Standard Scaling
# ---------------------------------------------------------------------------

class PerUnitStd:
    """
    Z-score scale each feature using each engine's OWN first n_init cycles.

    Same rationale as PerUnitMinMax but uses mean/std instead of min/range.
    When n_init is small (e.g. 5), std may be near zero for slowly-varying
    sensors; the _safe_scale eps guard handles this gracefully.
    """

    def __init__(self, n_init: int = 5):
        self.n_init = n_init
        self.name   = f"perunit_std_{n_init}"
        self._train_stats: dict = {}

    def fit(self, train_df: pd.DataFrame, feature_cols: list) -> "PerUnitStd":
        self._train_stats = {}
        for uid, grp in train_df.groupby("unit", sort=True):
            early = grp[feature_cols].head(self.n_init).values.astype(np.float64)
            mu    = early.mean(axis=0)
            sigma = early.std(axis=0, ddof=0)
            self._train_stats[uid] = (mu, sigma)
        return self

    def _apply_perunit(self, df: pd.DataFrame, feature_cols: list,
                       stats: dict) -> pd.DataFrame:
        result = df[feature_cols].copy().astype(np.float64)
        for uid, grp in df.groupby("unit", sort=True):
            if uid in stats:
                mu, sigma = stats[uid]
            else:
                vals  = grp[feature_cols].values.astype(np.float64)
                early = vals[: self.n_init]
                mu    = early.mean(axis=0)
                sigma = early.std(axis=0, ddof=0)
            vals   = grp[feature_cols].values.astype(np.float64)
            scaled = _safe_scale(vals, mu, sigma)
            result.loc[grp.index, feature_cols] = scaled
        return result

    def transform_train(self, train_df: pd.DataFrame,
                        feature_cols: list) -> pd.DataFrame:
        return self._apply_perunit(train_df, feature_cols, self._train_stats)

    def transform_test(self, test_df: pd.DataFrame,
                       feature_cols: list) -> pd.DataFrame:
        test_stats = {}
        for uid, grp in test_df.groupby("unit", sort=True):
            early = grp[feature_cols].head(self.n_init).values.astype(np.float64)
            mu    = early.mean(axis=0)
            sigma = early.std(axis=0, ddof=0)
            test_stats[uid] = (mu, sigma)
        return self._apply_perunit(test_df, feature_cols, test_stats)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_NORMALIZERS = [
    FleetMinMax(),
    FleetStd(),
    PerUnitMinMax(n_init=5),
    PerUnitMinMax(n_init=10),
    PerUnitStd(n_init=5),
    PerUnitStd(n_init=10),
]

NORMALIZER_NAMES = [n.name for n in ALL_NORMALIZERS]


def get_normalizer(name: str):
    """Retrieve a fresh normalizer instance by name."""
    mapping = {
        "fleet_minmax":      FleetMinMax,
        "fleet_std":         FleetStd,
        "perunit_minmax_5":  lambda: PerUnitMinMax(n_init=5),
        "perunit_minmax_10": lambda: PerUnitMinMax(n_init=10),
        "perunit_std_5":     lambda: PerUnitStd(n_init=5),
        "perunit_std_10":    lambda: PerUnitStd(n_init=10),
    }
    if name not in mapping:
        raise ValueError(f"Unknown normalizer '{name}'. "
                         f"Choose from: {list(mapping)}")
    return mapping[name]()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pathlib import Path
    from data_loader_import_helper import _demo_load  # handled below

    # Minimal inline demo without importing 01_data_loader to keep self-test simple
    import pandas as pd, numpy as np
    np.random.seed(0)
    n = 200
    demo = pd.DataFrame({
        "unit":  np.repeat([1, 2, 3, 4], 50),
        "cycle": np.tile(np.arange(1, 51), 4),
        "s2":    np.random.randn(n),
        "s3":    np.random.randn(n) + 5,
    })
    feat_cols = ["s2", "s3"]

    for norm in ALL_NORMALIZERS:
        norm.fit(demo, feat_cols)
        out = norm.transform_train(demo, feat_cols)
        print(f"{norm.name:25s}  mean={out['s2'].mean():.4f}  "
              f"std={out['s2'].std():.4f}")
