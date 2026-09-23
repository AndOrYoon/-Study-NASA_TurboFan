# -*- coding: utf-8 -*-
"""
H6 Phase 1 — Cluster Saver
Saves:
  - FD004 operating-condition K-means artifacts (op_scaler, op_kmeans, op_cluster_means)
  - cluster_assignments_fd003.csv and cluster_assignments_fd004.csv
"""

import sys, os
sys.path.insert(0, r"C:\BMAD_PY313\Data_Analysis\Code\shared")

import numpy as np
import pandas as pd
import pickle

import op_condition_utils as ocu

DATASET_DIR = r"C:\BMAD_PY313\Dataset"
RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H3_fault_mode"
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

COL_NAMES = (
    ["unit_number", "cycle",
     "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)

FAULT_DISCRIMINANT_SENSORS = ["s15", "s20", "s21", "s7", "s12", "s2", "s4"]
CONST_FD004 = {"s16"}

# Use actual op column names (avoids renaming the df)
OP_COLS = ("op_setting_1", "op_setting_2", "op_setting_3")


def load_data(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)


def load_gmm_bundle(dataset, variant="full"):
    pkl_path = os.path.join(MODELS_DIR, f"gmm_{dataset}_{variant}.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_cluster_assignments(feat_path, bundle):
    """Predict cluster assignments and probabilities using a saved GMM bundle."""
    df      = pd.read_csv(feat_path, index_col=0)
    gmm     = bundle["gmm"]
    scaler  = bundle["scaler"]
    X_sc    = scaler.transform(df.values)
    labels  = gmm.predict(X_sc)
    probs   = gmm.predict_proba(X_sc)
    out = pd.DataFrame({
        "unit":    df.index,
        "cluster": labels,
        "p0":      probs[:, 0],
        "p1":      probs[:, 1],
    }).set_index("unit")
    return out


# ---- FD003 ----------------------------------------------------------------

def save_fd003_assignments():
    print("\n[FD003] Computing cluster assignments ...")
    train = load_data(os.path.join(DATASET_DIR, "train_FD003.txt"))
    life  = train.groupby("unit_number")["cycle"].max().rename("lifetime")

    bundle    = load_gmm_bundle("fd003", "full")
    feat_path = os.path.join(RESULTS_DIR, "features_fd003_AB_full.csv")
    assign    = get_cluster_assignments(feat_path, bundle)
    assign    = assign.join(life.rename_axis("unit"))

    out_path = os.path.join(RESULTS_DIR, "cluster_assignments_fd003.csv")
    assign.to_csv(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Cluster distribution:\n{assign['cluster'].value_counts().to_string()}")
    return assign


# ---- FD004 ----------------------------------------------------------------

def save_fd004_op_artifacts():
    """Refit FD004 K-means on op conditions and save artifacts."""
    print("\n[FD004] Fitting op-condition K-means (k=6) ...")
    train = load_data(os.path.join(DATASET_DIR, "train_FD004.txt"))

    active_sensors = [s for s in FAULT_DISCRIMINANT_SENSORS if s not in CONST_FD004]

    # fit_op_condition_kmeans supports custom op_cols
    scaler, km = ocu.fit_op_condition_kmeans(train, k=6, op_cols=OP_COLS)
    cluster_means = ocu.compute_cluster_means(train, scaler, km, active_sensors,
                                              op_cols=OP_COLS)

    # Save to MODELS_DIR
    p = MODELS_DIR
    with open(os.path.join(p, "op_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(p, "op_kmeans.pkl"), "wb") as f:
        pickle.dump(km, f)
    cluster_means.to_csv(os.path.join(p, "op_cluster_means.csv"))
    print(f"  Saved: op_scaler.pkl, op_kmeans.pkl, op_cluster_means.csv → {p}")
    return scaler, km, cluster_means


def save_fd004_assignments():
    print("\n[FD004] Computing cluster assignments ...")
    train = load_data(os.path.join(DATASET_DIR, "train_FD004.txt"))
    life  = train.groupby("unit_number")["cycle"].max().rename("lifetime")

    bundle    = load_gmm_bundle("fd004", "full")
    feat_path = os.path.join(RESULTS_DIR, "features_fd004_AB_full.csv")
    assign    = get_cluster_assignments(feat_path, bundle)
    assign    = assign.join(life.rename_axis("unit"))

    out_path = os.path.join(RESULTS_DIR, "cluster_assignments_fd004.csv")
    assign.to_csv(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Cluster distribution:\n{assign['cluster'].value_counts().to_string()}")
    return assign


# ---- Main -----------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("H6 Phase 1 — Cluster Saver")
    print("=" * 60)

    assign3 = save_fd003_assignments()
    save_fd004_op_artifacts()
    assign4 = save_fd004_assignments()

    print("\n[DONE] Cluster saver complete.")
