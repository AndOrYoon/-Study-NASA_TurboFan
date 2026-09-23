# -*- coding: utf-8 -*-
"""
H6 Phase 2 — Shared Model Utilities
Provides: constants, data loading, preprocessing, LSTM building blocks,
Dataset classes, train/eval loops, and metric computation.
"""

import os, sys, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- Paths ----------------------------------------------------------------
DATASET_DIR = r"C:\BMAD_PY313\Dataset"
RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H3_fault_mode"
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---- Constants ------------------------------------------------------------
COL_NAMES = (
    ["unit_number", "cycle",
     "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)

CONST_SENSORS = {
    "FD003": {"s1", "s5", "s10", "s16", "s18", "s19"},
    "FD004": {"s16"},
}

FAULT_DISCRIMINANT_SENSORS = ["s15", "s20", "s21", "s7", "s12", "s2", "s4"]

RUL_CLIP    = 125
WINDOW_SIZE = 30
SEEDS       = [0, 1, 2, 3, 4]
OP_COLS     = ("op_setting_1", "op_setting_2", "op_setting_3")

# ---- Reproducibility -------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- Metrics ---------------------------------------------------------------

def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    d = pred - true
    pen = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(pen.sum())


def compute_metrics(pred: np.ndarray, true: np.ndarray):
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ns   = nasa_score(pred, true)
    return rmse, ns


# ---- Data loading ----------------------------------------------------------

def load_cmapss(dataset: str, split: str = "train") -> pd.DataFrame:
    path = os.path.join(DATASET_DIR, f"{split}_{dataset}.txt")
    df   = pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)
    return df


def load_test_rul(dataset: str) -> pd.Series:
    """Return RUL series indexed 1..N for the test engines."""
    path = os.path.join(DATASET_DIR, f"RUL_{dataset}.txt")
    rul  = pd.read_csv(path, header=None, names=["RUL"])
    rul.index = rul.index + 1  # 1-based engine index
    return rul["RUL"]


def add_rul(df: pd.DataFrame, rul_clip: int = RUL_CLIP) -> pd.DataFrame:
    """Add clipped RUL column to training DataFrame."""
    max_cyc = df.groupby("unit_number")["cycle"].max()
    df = df.copy()
    df["RUL"] = df["unit_number"].map(max_cyc) - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=rul_clip)
    return df


def get_sensor_cols(dataset: str) -> list:
    const = CONST_SENSORS.get(dataset, set())
    return [f"s{i}" for i in range(1, 22) if f"s{i}" not in const]


# ---- Operating condition residualization (FD004) ---------------------------

def fit_op_residual_fd004(train_df: pd.DataFrame, sensor_cols: list):
    """Fit K-means(k=6) on op conditions and compute cluster means for ALL sensor_cols.
    Returns (scaler, km, cluster_means_df) — call apply_op_residual_fd004 for transform."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler as _SS
    sc = _SS()
    op_sc = sc.fit_transform(train_df[list(OP_COLS)])
    km = KMeans(n_clusters=6, random_state=42, n_init=20)
    km.fit(op_sc)
    tmp = train_df.copy()
    tmp["_opc"] = km.predict(op_sc)
    cmeans = tmp.groupby("_opc")[sensor_cols].mean()
    return sc, km, cmeans


def apply_op_residual_fd004(df: pd.DataFrame,
                             scaler, km, cluster_means: pd.DataFrame,
                             sensor_cols: list) -> pd.DataFrame:
    """Subtract per-op-cluster mean from sensor values (only cols in cluster_means)."""
    df = df.copy()
    op_sc = scaler.transform(df[list(OP_COLS)])
    df["_opc"] = km.predict(op_sc)
    for col in sensor_cols:
        if col not in cluster_means.columns:
            continue  # skip sensors not in cluster_means
        means_map = cluster_means[col]
        df[col] = df[col] - df["_opc"].map(means_map)
    return df.drop(columns=["_opc"])


def load_op_artifacts_fd004():
    with open(os.path.join(MODELS_DIR, "op_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "op_kmeans.pkl"), "rb") as f:
        km = pickle.load(f)
    cluster_means = pd.read_csv(
        os.path.join(MODELS_DIR, "op_cluster_means.csv"), index_col=0
    )
    return scaler, km, cluster_means


# ---- Normalization ---------------------------------------------------------

def fit_normalization(train_df: pd.DataFrame, sensor_cols: list):
    """Return (min_series, max_series) fitted on train data."""
    return train_df[sensor_cols].min(), train_df[sensor_cols].max()


def apply_normalization(df: pd.DataFrame, sensor_cols: list,
                        min_v: pd.Series, max_v: pd.Series) -> pd.DataFrame:
    rng = (max_v - min_v).replace(0, 1.0)
    df  = df.copy()
    df[sensor_cols] = (df[sensor_cols] - min_v) / rng
    return df


# ---- Sequence creation -----------------------------------------------------

def make_train_sequences(df: pd.DataFrame, sensor_cols: list,
                          window: int = WINDOW_SIZE):
    """Sliding window sequences for training."""
    X_list, y_list, unit_list = [], [], []
    for unit, grp in df.groupby("unit_number"):
        vals = grp[sensor_cols].values.astype(np.float32)
        ruls = grp["RUL"].values.astype(np.float32)
        for i in range(len(vals) - window + 1):
            X_list.append(vals[i : i + window])
            y_list.append(ruls[i + window - 1])
            unit_list.append(unit)
    return np.array(X_list), np.array(y_list), np.array(unit_list)


def make_test_sequences(df: pd.DataFrame, sensor_cols: list,
                         rul_series: pd.Series,
                         window: int = WINDOW_SIZE):
    """Last-window sequence per test engine (+ padding if needed)."""
    n_feat = len(sensor_cols)
    X_list, y_list = [], []
    for unit, grp in df.groupby("unit_number"):
        vals = grp[sensor_cols].values.astype(np.float32)
        if len(vals) >= window:
            X_list.append(vals[-window:])
        else:
            pad = np.zeros((window - len(vals), n_feat), dtype=np.float32)
            X_list.append(np.vstack([pad, vals]))
        y_list.append(float(rul_series.get(unit, 0)))
    return np.array(X_list), np.array(y_list)


# ---- Train/val split by engine unit ----------------------------------------

def split_engines(units: np.ndarray, val_frac: float = 0.2, seed: int = 42):
    """Return (train_mask, val_mask) over sequence-level unit array."""
    unique_units = np.unique(units)
    rng  = np.random.RandomState(seed)
    n_val = max(1, int(len(unique_units) * val_frac))
    val_u = set(rng.choice(unique_units, n_val, replace=False).tolist())
    tr_m  = np.array([u not in val_u for u in units])
    va_m  = ~tr_m
    return tr_m, va_m


# ---- PyTorch Dataset -------------------------------------------------------

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SeqDatasetWithProbs(Dataset):
    """For M2: includes per-sequence GMM soft probabilities."""
    def __init__(self, X: np.ndarray, y: np.ndarray, probs: np.ndarray):
        self.X     = torch.from_numpy(X).float()
        self.y     = torch.from_numpy(y).float().unsqueeze(-1)
        self.probs = torch.from_numpy(probs.astype(np.float32))

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.probs[i]


class SeqDatasetM3(Dataset):
    """For M3: full sequence + initial K cycles for GatingNet."""
    def __init__(self, X: np.ndarray, y: np.ndarray, K: int = 10):
        self.X_full = torch.from_numpy(X).float()
        self.X_init = torch.from_numpy(X[:, :K, :]).float()
        self.y      = torch.from_numpy(y).float().unsqueeze(-1)

    def __len__(self): return len(self.X_full)
    def __getitem__(self, i): return self.X_full[i], self.X_init[i], self.y[i]


def make_loader(X, y, batch_size=256, shuffle=True, **kw):
    return DataLoader(SeqDataset(X, y), batch_size=batch_size, shuffle=shuffle, **kw)


# ---- LSTM building blocks --------------------------------------------------

class LSTMBranch(nn.Module):
    """LSTM(64) → Drop → LSTM(64) → Drop → FC(64→32→1)"""
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc    = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = self.drop2(out)
        return self.fc(out[:, -1, :])          # last time-step


class SoftGatingModel(nn.Module):
    """M2: Two branches weighted by external GMM soft probabilities."""
    def __init__(self, n_features: int):
        super().__init__()
        self.branch0 = LSTMBranch(n_features)
        self.branch1 = LSTMBranch(n_features)

    def forward(self, x, w0, w1):
        y0 = self.branch0(x)          # (B,1)
        y1 = self.branch1(x)          # (B,1)
        w0 = w0.view(-1, 1)
        w1 = w1.view(-1, 1)
        y_final = w0 * y0 + w1 * y1
        return y_final, y0, y1


class GatingNet(nn.Module):
    def __init__(self, K: int, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(K * n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x_init):
        return self.net(x_init)        # (B,2)


class AttentionGateModel(nn.Module):
    """M3: End-to-end GatingNet + two LSTM branches."""
    def __init__(self, n_features: int, K: int = 10):
        super().__init__()
        self.K       = K
        self.gating  = GatingNet(K, n_features)
        self.branch0 = LSTMBranch(n_features)
        self.branch1 = LSTMBranch(n_features)

    def forward(self, x_full, x_init):
        w    = self.gating(x_init)     # (B,2)
        w0   = w[:, 0:1]
        w1   = w[:, 1:2]
        y0   = self.branch0(x_full)
        y1   = self.branch1(x_full)
        return w0 * y0 + w1 * y1, y0, y1


# ---- Training loops --------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    n     = 0
    for batch in loader:
        xb, yb = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
        n     += len(xb)
    return total / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n     = 0
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0].to(device), batch[1].to(device)
            pred   = model(xb)
            total += criterion(pred, yb).item() * len(xb)
            n     += len(xb)
    return total / n


def train_branch_epoch(model, loader, optimizer, criterion, device):
    """For M1: single branch, standard training."""
    return train_epoch(model, loader, optimizer, criterion, device)


def train_m2_epoch(model, loader, optimizer, criterion, device):
    """M2 training epoch: MSE(final) + 0.1*MSE(branch0) + 0.1*MSE(branch1)."""
    model.train()
    total = 0.0; n = 0
    for xb, yb, probs in loader:
        xb, yb = xb.to(device), yb.to(device)
        w0 = probs[:, 0].to(device)
        w1 = probs[:, 1].to(device)
        optimizer.zero_grad()
        y_final, y0, y1 = model(xb, w0, w1)
        loss = criterion(y_final, yb) + 0.1 * criterion(y0, yb) + 0.1 * criterion(y1, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n


def eval_m2_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0; n = 0
    with torch.no_grad():
        for xb, yb, probs in loader:
            xb, yb = xb.to(device), yb.to(device)
            w0 = probs[:, 0].to(device)
            w1 = probs[:, 1].to(device)
            y_final, y0, y1 = model(xb, w0, w1)
            loss = criterion(y_final, yb) + 0.1 * criterion(y0, yb) + 0.1 * criterion(y1, yb)
            total += loss.item() * len(xb); n += len(xb)
    return total / n


def train_m3_epoch(model, loader, optimizer, criterion, device):
    """M3: Loss = MSE(final) + 0.05*MSE(b0) + 0.05*MSE(b1)."""
    model.train()
    total = 0.0; n = 0
    for xb_full, xb_init, yb in loader:
        xb_full = xb_full.to(device)
        xb_init = xb_init.to(device)
        yb      = yb.to(device)
        optimizer.zero_grad()
        y_final, y0, y1 = model(xb_full, xb_init)
        loss = criterion(y_final, yb) + 0.05 * criterion(y0, yb) + 0.05 * criterion(y1, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb_full); n += len(xb_full)
    return total / n


def eval_m3_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0; n = 0
    with torch.no_grad():
        for xb_full, xb_init, yb in loader:
            xb_full = xb_full.to(device); xb_init = xb_init.to(device); yb = yb.to(device)
            y_final, y0, y1 = model(xb_full, xb_init)
            loss = criterion(y_final, yb) + 0.05 * criterion(y0, yb) + 0.05 * criterion(y1, yb)
            total += loss.item() * len(xb_full); n += len(xb_full)
    return total / n


# ---- Inference helpers -----------------------------------------------------

@torch.no_grad()
def predict_sequences(model, X_np: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    """Run model on X_np (N, W, F) and return flat predictions (N,)."""
    model.eval()
    ds     = SeqDataset(X_np, np.zeros(len(X_np), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds  = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds).flatten()


@torch.no_grad()
def predict_m2(model, X_np, w0_arr, w1_arr, device, batch_size=512):
    """M2 inference with soft probabilities."""
    model.eval()
    probs  = np.stack([w0_arr, w1_arr], axis=1).astype(np.float32)
    ds     = SeqDatasetWithProbs(X_np, np.zeros(len(X_np)), probs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds  = []
    for xb, _, pb in loader:
        w0 = pb[:, 0].to(device); w1 = pb[:, 1].to(device)
        y, _, _ = model(xb.to(device), w0, w1)
        preds.append(y.cpu().numpy())
    return np.concatenate(preds).flatten()


@torch.no_grad()
def predict_m3(model, X_np, device, K=10, batch_size=512):
    """M3 inference."""
    model.eval()
    ds     = SeqDatasetM3(X_np, np.zeros(len(X_np)), K)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds  = []
    for xb_full, xb_init, _ in loader:
        y, _, _ = model(xb_full.to(device), xb_init.to(device))
        preds.append(y.cpu().numpy())
    return np.concatenate(preds).flatten()


# ---- Save predictions ------------------------------------------------------

def save_predictions(preds: np.ndarray, trues: np.ndarray,
                     model_name: str, dataset: str, seed: int):
    path = os.path.join(
        RESULTS_DIR, f"raw_predictions_{model_name}_{dataset}_seed{seed}.csv"
    )
    pd.DataFrame({"pred": preds, "true": trues}).to_csv(path, index=False)
    return path


# ---- Checkpoint helpers ----------------------------------------------------

def save_checkpoint(state_dict, model_name, dataset, seed, tag="best"):
    fname = f"{model_name}_{dataset}_seed{seed}_{tag}.pt"
    path  = os.path.join(MODELS_DIR, fname)
    torch.save(state_dict, path)
    return path


def load_checkpoint(model_name, dataset, seed, tag="best"):
    fname = f"{model_name}_{dataset}_seed{seed}_{tag}.pt"
    path  = os.path.join(MODELS_DIR, fname)
    return torch.load(path, map_location="cpu")
