# -*- coding: utf-8 -*-
"""
H6 Attention-LSTM Backbone
Adds a self-attention layer (nn.MultiheadAttention) between LSTM₁ and LSTM₂
of the standard stacked-LSTM backbone to test whether M3 gating still
provides additional benefit on top of the attention mechanism.

Classes
-------
AttnLSTMBranch   : M0-equivalent — LSTM1 → Self-Attention → Dropout → FC
AttnLSTMM0       : Thin wrapper around AttnLSTMBranch (single backbone, no gating)
GatingNetAttn    : Identical to LSTM-M3/Transformer-M3 GatingNet
                   Reads first K raw cycles → soft routing weights (B,2)
AttnLSTMM3       : M3-equivalent — GatingNet + two AttnLSTMBranch instances
                   Training loss: MSE(final) + 0.05*MSE(b0) + 0.05*MSE(b1)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Attention-LSTM Branch (M0-equivalent building block)
# ---------------------------------------------------------------------------

class AttnLSTMBranch(nn.Module):
    """LSTM₁(64) → Self-Attention → Dropout → FC(64→32→ReLU→1)

    Architecture:
        Input (B, T, F)
        -> LSTM1(F, 64, batch_first=True)  [return full sequence]
        -> Dropout(0.2) on h1
        -> MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
             query=h1, key=h1, value=h1   [self-attention]
        -> take last timestep: out[:, -1, :]
        -> Dropout(0.2)
        -> Linear(64, 32) -> ReLU -> Linear(32, 1)
    """

    def __init__(self, n_features: int, hidden: int = 64,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.attn  = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, batch_first=True,
            dropout=dropout,
        )
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h1, _   = self.lstm1(x)              # (B, T, hidden)
        h1      = self.drop1(h1)
        out, _  = self.attn(h1, h1, h1)      # self-attention; (B, T, hidden)
        last    = out[:, -1, :]               # last timestep (B, hidden)
        last    = self.drop2(last)
        return self.fc(last)                  # (B, 1)


# ---------------------------------------------------------------------------
# AttnLSTM-M0: single backbone, no gating
# ---------------------------------------------------------------------------

class AttnLSTMM0(nn.Module):
    """M0-equivalent: single Attention-LSTM backbone, no fault-mode routing."""

    def __init__(self, n_features: int, hidden: int = 64,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.backbone = AttnLSTMBranch(n_features, hidden, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# GatingNet — identical design to LSTM-M3 and Transformer-M3 GatingNets
# ---------------------------------------------------------------------------

class GatingNetAttn(nn.Module):
    """Reads the first K raw sensor cycles and produces soft routing weights.

    Identical to GatingNet in h6_p2_model_utils.py and h6_transformer_backbone.py:
        Flatten(K * F) -> Linear(K*F -> 32) -> ReLU -> Linear(32 -> 2) -> Softmax
    """

    def __init__(self, K: int, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(K * n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x_init: torch.Tensor) -> torch.Tensor:
        # x_init: (B, K, F)
        return self.net(x_init)  # (B, 2)


# ---------------------------------------------------------------------------
# AttnLSTM-M3: GatingNet + two independent AttnLSTMBranch instances
# ---------------------------------------------------------------------------

class AttnLSTMM3(nn.Module):
    """M3-equivalent with Attention-LSTM branches.

    Architecture:
        w = GatingNetAttn(x_init)          # (B, 2) — from first K raw cycles
        y0 = branch0(x_full)               # (B, 1)
        y1 = branch1(x_full)               # (B, 1)
        y_final = w[:,0:1]*y0 + w[:,1:2]*y1

    Training loss (caller computes):
        MSE(y_final, y) + 0.05*MSE(y0, y) + 0.05*MSE(y1, y)
    """

    def __init__(self, n_features: int, K: int = 10, hidden: int = 64,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.K       = K
        self.gating  = GatingNetAttn(K, n_features)
        self.branch0 = AttnLSTMBranch(n_features, hidden, num_heads, dropout)
        self.branch1 = AttnLSTMBranch(n_features, hidden, num_heads, dropout)

    def forward(self, x_full: torch.Tensor, x_init: torch.Tensor):
        # x_full: (B, T, F)  — full 30-cycle window
        # x_init: (B, K, F)  — first K cycles (raw, same normalisation as x_full)
        w  = self.gating(x_init)       # (B, 2)
        w0 = w[:, 0:1]                 # (B, 1)
        w1 = w[:, 1:2]                 # (B, 1)
        y0 = self.branch0(x_full)      # (B, 1)
        y1 = self.branch1(x_full)      # (B, 1)
        y_final = w0 * y0 + w1 * y1   # (B, 1)
        return y_final, y0, y1
