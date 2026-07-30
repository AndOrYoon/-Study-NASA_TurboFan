# -*- coding: utf-8 -*-
"""
shared/op_condition_utils.py
-----------------------------
K-means operating condition residualization utility.
Used by H5 (normalization comparison) and H6 (fault mode clustering).

For FD002/FD004: raw sensor values are contaminated by 6 distinct
operating conditions. Subtracting cluster-mean residuals isolates
the degradation signal from operating-point offsets.

Test-time safety: fit on train data only; pass scaler+km+cluster_means
to apply_op_residual() for test data to avoid data leakage.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

OP_COLS = ('op1', 'op2', 'op3')
K_DEFAULT = 6


def fit_op_condition_kmeans(train_df: pd.DataFrame,
                             k: int = K_DEFAULT,
                             op_cols: tuple = OP_COLS,
                             random_state: int = 42):
    """
    Fit K-means on operating conditions of train data.

    Returns
    -------
    scaler : StandardScaler  (fitted on train op columns)
    km     : KMeans          (fitted on scaled op columns)
    """
    scaler = StandardScaler()
    op_scaled = scaler.fit_transform(train_df[list(op_cols)])
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    km.fit(op_scaled)
    return scaler, km


def compute_cluster_means(train_df: pd.DataFrame,
                           scaler: StandardScaler,
                           km: KMeans,
                           feature_cols: list,
                           op_cols: tuple = OP_COLS) -> pd.DataFrame:
    """
    Compute per-cluster mean for each feature column (train data only).

    Returns
    -------
    cluster_means : DataFrame, index=cluster_id, columns=feature_cols
    """
    df = train_df.copy()
    op_scaled = scaler.transform(df[list(op_cols)])
    df['_op_cluster'] = km.predict(op_scaled)
    cluster_means = df.groupby('_op_cluster')[feature_cols].mean()
    return cluster_means


def apply_op_residual(df: pd.DataFrame,
                      scaler: StandardScaler,
                      km: KMeans,
                      cluster_means: pd.DataFrame,
                      feature_cols: list,
                      op_cols: tuple = OP_COLS) -> pd.DataFrame:
    """
    Subtract cluster-mean from each feature to produce residuals.

    Parameters
    ----------
    df            : DataFrame to transform (train or test)
    scaler        : fitted StandardScaler from fit_op_condition_kmeans()
    km            : fitted KMeans from fit_op_condition_kmeans()
    cluster_means : DataFrame from compute_cluster_means() — train-derived
    feature_cols  : sensor/feature columns to residualize

    Returns
    -------
    df_residual : copy of df with feature_cols replaced by residuals
    """
    df = df.copy()
    op_scaled = scaler.transform(df[list(op_cols)])
    df['_op_cluster'] = km.predict(op_scaled)

    for col in feature_cols:
        means_map = cluster_means[col]
        df[col] = df[col] - df['_op_cluster'].map(means_map)

    return df.drop(columns=['_op_cluster'])


def save_op_artifacts(path: str,
                       scaler: StandardScaler,
                       km: KMeans,
                       cluster_means: pd.DataFrame):
    """Serialize K-means artifacts for test-time reuse."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / 'op_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(p / 'op_kmeans.pkl', 'wb') as f:
        pickle.dump(km, f)
    cluster_means.to_csv(p / 'op_cluster_means.csv')


def load_op_artifacts(path: str):
    """Load serialized K-means artifacts."""
    p = Path(path)
    with open(p / 'op_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(p / 'op_kmeans.pkl', 'rb') as f:
        km = pickle.load(f)
    cluster_means = pd.read_csv(p / 'op_cluster_means.csv', index_col=0)
    return scaler, km, cluster_means
