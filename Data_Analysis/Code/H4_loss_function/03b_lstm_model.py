"""
H7 Loss Function Optimization - LSTM Model + Train/Eval Loop
Architecture: LSTM(hidden=32) → Dropout(0.2) → Linear(32→16) → ReLU → Linear(16→1)
"""

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
# Model definition
# ─────────────────────────────────────────────────────────────────────────────
class TurboLSTM(nn.Module):
    def __init__(self, n_features: int,
                 hidden: int = LSTM_HIDDEN,
                 dropout: float = LSTM_DROPOUT):
        super().__init__()
        self.lstm    = nn.LSTM(n_features, hidden,
                               batch_first=True,
                               dropout=0.0)   # single layer → no internal dropout
        self.drop    = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden, 16)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(16, 1)

    def forward(self, x):
        # x : (B, W, F)
        out, _ = self.lstm(x)          # (B, W, H)
        h      = out[:, -1, :]        # last time-step
        h      = self.drop(h)
        h      = self.relu(self.fc1(h))
        return self.fc2(h).squeeze(-1) # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────────────────────
def _make_loader(X, y, lr, batch_size: int, shuffle: bool):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(lr),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def train_lstm(dataset: str, clip_value, loss_name: str,
               seed: int, device: torch.device) -> dict:
    """
    Train LSTM on (dataset, clip, loss_name) with given seed.
    Returns dict with keys: clip, loss_fn, dataset, seed, rmse, nasa_score
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Load data ────────────────────────────────────────────────────────────
    train_eng, val_eng, scaler, feat_cols, clip_used = \
        load_train_val(dataset, clip_value)

    X_tr, y_tr, lr_tr = build_windows(train_eng, WINDOW)
    X_vl, y_vl, lr_vl = build_windows(val_eng,   WINDOW)

    train_loader = _make_loader(X_tr, y_tr, lr_tr, LSTM_BATCH, shuffle=True)
    val_loader   = _make_loader(X_vl, y_vl, lr_vl, LSTM_BATCH, shuffle=False)

    n_features = X_tr.shape[2]

    # ── Model ────────────────────────────────────────────────────────────────
    model  = TurboLSTM(n_features).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WD)
    loss_fn = LOSS_REGISTRY[loss_name]

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(LSTM_MAX_EPOCH):
        # Train
        model.train()
        for Xb, yb, lrb in train_loader:
            Xb  = Xb.to(device)
            yb  = yb.to(device)
            lrb = lrb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb, life_ratio=lrb, clip_value=clip_used)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb, lrb in val_loader:
                Xb  = Xb.to(device)
                yb  = yb.to(device)
                lrb = lrb.to(device)
                pred = model(Xb)
                vl   = loss_fn(pred, yb, life_ratio=lrb, clip_value=clip_used)
                val_losses.append(vl.item() * len(yb))
        total_val = len(y_vl)
        avg_val   = sum(val_losses) / total_val if total_val > 0 else float("inf")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= LSTM_PATIENCE:
                break

    # ── Test evaluation ──────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    X_test, rul_true = load_test(dataset, scaler, feat_cols)
    X_test_t         = torch.from_numpy(X_test).to(device)

    with torch.no_grad():
        preds_raw = model(X_test_t).cpu().numpy()

    preds = np.clip(preds_raw, 0, None)

    clip_name = f"clip_{clip_value}" if clip_value is not None else "clip_none"

    return {
        "clip":       clip_name,
        "loss_fn":    loss_name,
        "dataset":    dataset,
        "seed":       seed,
        "rmse":       rmse(preds, rul_true),
        "nasa_score": nasa_score(preds, rul_true),
        "preds":      preds,
        "rul_true":   rul_true,
    }
