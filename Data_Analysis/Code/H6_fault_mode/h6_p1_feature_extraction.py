# -*- coding: utf-8 -*-
"""
H6 Phase 1 - Feature Extraction
Extracts late_mean and slope features for FAULT_DISCRIMINANT_SENSORS
from FD003 and FD004 training data.

For FD004: computes operating condition residuals before feature extraction.
"""

import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---- Paths ----------------------------------------------------------------
DATASET_DIR = r"C:\BMAD_PY313\Dataset"
RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H6_fault_mode"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Constants ------------------------------------------------------------
COL_NAMES = (
    ["unit_number", "cycle",
     "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)

FAULT_DISCRIMINANT_SENSORS = ["s15", "s20", "s21", "s7", "s12", "s2", "s4"]

# Constant sensors to drop per dataset
CONST_FD003 = {"s1", "s5", "s10", "s16", "s18", "s19"}
CONST_FD004 = {"s16"}


# ---- Helper functions -----------------------------------------------------

def load_data(filepath):
    """Load a CMAPSS txt file."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COL_NAMES)
    return df


def compute_slope(values):
    """Linear regression slope over cycle index."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr))
    coeffs = np.polyfit(x, arr, 1)
    return float(coeffs[0])


def compute_late_mean(values, pct=0.20):
    """Mean of last pct fraction of values."""
    arr = np.asarray(values, dtype=float)
    n = max(1, int(len(arr) * pct))
    return float(arr[-n:].mean())


def extract_features(df, sensors, dataset_label):
    """
    For each engine (unit_number) compute late_mean and slope
    for each sensor in sensors.

    Returns a DataFrame indexed by unit_number with columns:
      {sensor}_late_mean, {sensor}_slope  for each sensor.
    """
    print(f"  Extracting features for {dataset_label} ...")
    records = []
    for unit, grp in df.groupby("unit_number"):
        row = {"unit_number": unit}
        for s in sensors:
            vals = grp[s].values
            row[f"{s}_late_mean"] = compute_late_mean(vals)
            row[f"{s}_slope"]     = compute_slope(vals)
        records.append(row)
    feat_df = pd.DataFrame(records).set_index("unit_number")
    print(f"    -> {len(feat_df)} engines, {feat_df.shape[1]} raw feature columns")
    return feat_df


def add_life_cycles(df, feat_df):
    """Append max cycle (life length) per engine."""
    life = df.groupby("unit_number")["cycle"].max().rename("life_cycles")
    return feat_df.join(life)


# ---- FD003 ----------------------------------------------------------------

def process_fd003():
    print("\n[FD003] Loading training data ...")
    train = load_data(os.path.join(DATASET_DIR, "train_FD003.txt"))

    # FD003 uses only 1 operating condition -> no residual needed
    # Drop constant sensors (they are not in FAULT_DISCRIMINANT_SENSORS anyway)
    active = [s for s in FAULT_DISCRIMINANT_SENSORS
              if s not in CONST_FD003]
    print(f"  Active fault sensors: {active}")

    feat_df = extract_features(train, active, "FD003")
    feat_df = add_life_cycles(train, feat_df)

    # Build the three variant matrices
    late_cols  = [c for c in feat_df.columns if c.endswith("_late_mean")]
    slope_cols = [c for c in feat_df.columns if c.endswith("_slope")]

    save_variants(feat_df, late_cols, slope_cols, "fd003")


# ---- FD004 ----------------------------------------------------------------

def process_fd004():
    print("\n[FD004] Loading training data ...")
    train = load_data(os.path.join(DATASET_DIR, "train_FD004.txt"))

    # Step 1: identify 6 operating conditions via KMeans on op settings
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    op_data = train[op_cols].values
    scaler_op = StandardScaler()
    op_scaled = scaler_op.fit_transform(op_data)

    print("  Fitting KMeans(k=6) on operating conditions ...")
    km = KMeans(n_clusters=6, random_state=42, n_init=20)
    train = train.copy()
    train["op_setting_cat"] = km.fit_predict(op_scaled)

    # Step 2: compute per-op-condition sensor means
    active = [s for s in FAULT_DISCRIMINANT_SENSORS
              if s not in CONST_FD004]
    print(f"  Active fault sensors: {active}")

    op_means = train.groupby("op_setting_cat")[active].mean()

    # Step 3: subtract op-condition mean -> residuals
    print("  Computing operating-condition residuals ...")
    residual_df = train.copy()
    for s in active:
        means_mapped = train["op_setting_cat"].map(op_means[s])
        residual_df[s] = train[s] - means_mapped

    feat_df = extract_features(residual_df, active, "FD004")
    feat_df = add_life_cycles(train, feat_df)

    late_cols  = [c for c in feat_df.columns if c.endswith("_late_mean")]
    slope_cols = [c for c in feat_df.columns if c.endswith("_slope")]

    save_variants(feat_df, late_cols, slope_cols, "fd004")


# ---- Save variants --------------------------------------------------------

def save_variants(feat_df, late_cols, slope_cols, tag):
    """Save AB_full, AB_slope, AB_late CSVs."""
    # AB_full: late_mean + slope (exclude life_cycles column)
    full_cols = late_cols + slope_cols
    ab_full  = feat_df[full_cols].copy()
    ab_slope = feat_df[slope_cols].copy()
    ab_late  = feat_df[late_cols].copy()

    for variant, df_v in [("AB_full", ab_full),
                           ("AB_slope", ab_slope),
                           ("AB_late", ab_late)]:
        path = os.path.join(RESULTS_DIR, f"features_{tag}_{variant}.csv")
        df_v.to_csv(path)
        print(f"  Saved: {path}  shape={df_v.shape}")


# ---- Main -----------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("H6 Phase 1 - Feature Extraction")
    print("=" * 60)

    process_fd003()
    process_fd004()

    print("\n[DONE] All feature CSVs written to:")
    print(f"  {RESULTS_DIR}")
