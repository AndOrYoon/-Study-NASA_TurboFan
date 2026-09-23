# -*- coding: utf-8 -*-
"""
03_lstm_model.py
----------------
LSTM models for H5 normalisation comparison.

Classes
-------
LSTMBase        : standard LSTM backbone for N1-N6 (pre-processed inputs)
RevIN           : Reversible Instance Normalisation layer  (Kim et al. ICLR 2022)
LSTMWithRevIN   : LSTM backbone with built-in RevIN for N7

Utilities
---------
set_seed(seed)
train_epoch(model, loader, optimizer, criterion, device) -> float
eval_epoch(model, loader, criterion, device)             -> float
nasa_score(y_true, y_pred)                               -> float
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# NASA asymmetric score
# ---------------------------------------------------------------------------

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA prognostic score.
    Penalises late predictions (d > 0) more harshly than early ones.
        s = sum(exp( d/10) - 1)  for d < 0
        s = sum(exp(-d/13) - 1)  for d >= 0
    where d = y_pred - y_true.
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))


# ---------------------------------------------------------------------------
# Standard LSTM backbone  (N1–N6)
# ---------------------------------------------------------------------------

class LSTMBase(nn.Module):
    """
    Canonical §0.4 backbone.
    LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Linear(32→16) → ReLU → Linear(16→1)
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc    = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        out, _ = self.lstm1(x)               # (batch, window, 64)
        out     = self.drop1(out)             # dropout on full sequence
        out, _ = self.lstm2(out)             # (batch, window, 32)
        out     = self.drop2(out[:, -1, :])  # last timestep -> (batch, 32)
        return self.fc(out).squeeze(-1)       # (batch,)


# ---------------------------------------------------------------------------
# RevIN layer  (Kim et al. ICLR 2022)
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """
    Reversible Instance Normalisation.
    Normalises each sample along the time dimension at forward pass.
    Affine learnable parameters (gamma, beta) per feature.
    """

    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps    = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta  = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        mean = x.mean(dim=1, keepdim=True).detach()
        std  = x.std(dim=1, keepdim=True, unbiased=False).detach() + self.eps
        x    = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x


# ---------------------------------------------------------------------------
# LSTMWithRevIN  (N7)
# ---------------------------------------------------------------------------

class LSTMWithRevIN(nn.Module):
    """
    Same §0.4 backbone but with a RevIN layer prepended.
    No external pre-processing normalisation is applied for this variant.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.revin = RevIN(n_features)
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc    = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x       = self.revin(x)              # (batch, window, n_features)
        out, _ = self.lstm1(x)              # (batch, window, 64)
        out     = self.drop1(out)            # dropout on full sequence
        out, _ = self.lstm2(out)            # (batch, window, 32)
        out     = self.drop2(out[:, -1, :]) # last timestep -> (batch, 32)
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dev: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(dev)
        y_batch = y_batch.to(dev)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    dev: torch.device,
) -> float:
    """Run one evaluation epoch. Returns mean loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(dev)
            y_batch = y_batch.to(dev)
            preds   = model(X_batch)
            loss    = criterion(preds, y_batch)
            total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def predict_all(
    model: nn.Module,
    X: np.ndarray,
    dev: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Return predictions for all samples in X as a numpy array."""
    model.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds   = []
    with torch.no_grad():
        for (X_batch,) in loader:
            preds.append(model(X_batch.to(dev)).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    B, W, F = 32, 30, 14
    x = torch.randn(B, W, F).to(dev)

    base = LSTMBase(F).to(dev)
    revin_model = LSTMWithRevIN(F).to(dev)

    out_base  = base(x)
    out_revin = revin_model(x)
    print(f"LSTMBase output shape:       {out_base.shape}")
    print(f"LSTMWithRevIN output shape:  {out_revin.shape}")
    print("Self-test passed.")
