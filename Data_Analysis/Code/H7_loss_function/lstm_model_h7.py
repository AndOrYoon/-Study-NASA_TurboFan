"""
H7 Loss Function Optimization - LSTM Model + Training Loop
Architecture: LSTM(32) → Dropout(0.2) → FC(32→16) → ReLU → FC(16→1)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config_h7 import (
    LSTM_HIDDEN, LSTM_DROPOUT, LSTM_LR, LSTM_WD,
    LSTM_BATCH, LSTM_MAX_EPOCH, LSTM_PATIENCE, WINDOW
)
from data_loader_h7 import load_train_val, load_test, build_windows
from loss_functions_h7 import nasa_score, rmse, LOSS_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
class TurboLSTM(nn.Module):
    def __init__(self, n_features: int,
                 hidden: int = LSTM_HIDDEN,
                 dropout: float = LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden, 16)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, W, H)
        h      = self.drop(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(h))).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
def _make_loader(X, y, lr, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(lr),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def clip_name(clip_value) -> str:
    return f"clip_{int(clip_value)}" if clip_value is not None else "clip_none"


# ─────────────────────────────────────────────────────────────────────────────
def train_lstm(dataset: str, clip_value, loss_name: str,
               seed: int, device: torch.device,
               loss_fn_override=None) -> dict:
    """
    Full train + evaluate for one (dataset, clip, loss, seed) combination.

    Parameters
    ----------
    loss_fn_override : optional callable — use instead of LOSS_REGISTRY lookup
                       (for Phase 3 hyperparameter sweeps).

    Returns dict with keys:
        clip, loss_fn, dataset, seed, rmse, nasa_score, preds, rul_true
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_eng, val_eng, scaler, feat_cols, clip_used = \
        load_train_val(dataset, clip_value)

    X_tr, y_tr, lr_tr = build_windows(train_eng, WINDOW)
    X_vl, y_vl, lr_vl = build_windows(val_eng,   WINDOW)

    tr_loader = _make_loader(X_tr, y_tr, lr_tr, LSTM_BATCH, shuffle=True)
    vl_loader = _make_loader(X_vl, y_vl, lr_vl, LSTM_BATCH, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    n_feat  = X_tr.shape[2]
    model   = TurboLSTM(n_feat).to(device)
    opt     = torch.optim.Adam(model.parameters(),
                               lr=LSTM_LR, weight_decay=LSTM_WD)
    loss_fn = loss_fn_override or LOSS_REGISTRY[loss_name]

    best_val  = float("inf")
    best_state = None
    patience  = 0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(LSTM_MAX_EPOCH):
        model.train()
        for Xb, yb, lrb in tr_loader:
            Xb  = Xb.to(device, non_blocking=True)
            yb  = yb.to(device, non_blocking=True)
            lrb = lrb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb, life_ratio=lrb, clip_value=clip_used)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # Validation
        model.eval()
        vl_sum, vl_n = 0.0, 0
        with torch.no_grad():
            for Xb, yb, lrb in vl_loader:
                Xb  = Xb.to(device, non_blocking=True)
                yb  = yb.to(device, non_blocking=True)
                lrb = lrb.to(device, non_blocking=True)
                v   = loss_fn(model(Xb), yb, life_ratio=lrb, clip_value=clip_used)
                vl_sum += v.item() * len(yb)
                vl_n   += len(yb)
        avg_vl = vl_sum / vl_n if vl_n else float("inf")

        if avg_vl < best_val:
            best_val   = avg_vl
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= LSTM_PATIENCE:
                break

    # ── Test ─────────────────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    X_test, rul_true = load_test(dataset, scaler, feat_cols)
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    preds = np.clip(preds, 0, None)

    return {
        "clip":       clip_name(clip_value),
        "loss_fn":    loss_name,
        "dataset":    dataset,
        "seed":       seed,
        "rmse":       rmse(preds, rul_true),
        "nasa_score": nasa_score(preds, rul_true),
        "preds":      preds,
        "rul_true":   rul_true,
    }
