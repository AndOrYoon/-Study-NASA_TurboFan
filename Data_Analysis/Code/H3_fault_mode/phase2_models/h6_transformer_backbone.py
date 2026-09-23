# -*- coding: utf-8 -*-
"""
H6 Transformer Backbone — Pilot Experiment (FD003)
Validates that M3's early-cycle gating principle is backbone-agnostic.

Classes
-------
PositionalEncoding    : Sinusoidal positional encoding (batch_first=True)
TransformerBranch     : Single Transformer Encoder branch (shared building block)
TransformerRUL        : M0-equivalent — single Transformer backbone, no gating
GatingNetTransformer  : Identical to LSTM-M3 GatingNet — reads first K raw cycles
TransformerM3         : M3-equivalent — GatingNet + two TransformerBranch instances
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, batch_first=True convention."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store as (1, max_len, d_model) so it broadcasts over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer branch (shared backbone building block)
# ---------------------------------------------------------------------------

class TransformerBranch(nn.Module):
    """Independent Transformer Encoder backbone for RUL regression.

    Pipeline:
        Input (B, T, F)
        -> Linear(F -> d_model)       [feature embedding]
        -> PositionalEncoding
        -> TransformerEncoder(num_layers, nhead, dim_feedforward, dropout)
        -> mean pooling over T
        -> Linear(d_model -> 32) -> ReLU -> Linear(32 -> 1)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, d_model) convention throughout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.embedding(x)      # (B, T, d_model)
        x = self.pos_enc(x)        # (B, T, d_model)
        x = self.transformer(x)    # (B, T, d_model)
        x = x.mean(dim=1)          # (B, d_model)  — mean pooling over time
        return self.fc(x)          # (B, 1)


# ---------------------------------------------------------------------------
# Transformer-M0: single backbone (no gating)
# ---------------------------------------------------------------------------

class TransformerRUL(nn.Module):
    """M0-equivalent: single Transformer Encoder, no fault-mode routing."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = TransformerBranch(
            n_features, d_model, nhead, num_layers, dim_feedforward, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# GatingNet — identical design to LSTM-M3's GatingNet
# ---------------------------------------------------------------------------

class GatingNetTransformer(nn.Module):
    """Reads the first K raw sensor cycles and produces soft routing weights.

    Identical to the GatingNet used in LSTM-M3:
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
        # x_init: (B, K, F)  raw (un-embedded) sensor values
        return self.net(x_init)   # (B, 2)


# ---------------------------------------------------------------------------
# Transformer-M3: GatingNet + two independent Transformer branches
# ---------------------------------------------------------------------------

class TransformerM3(nn.Module):
    """M3-equivalent with Transformer Encoder branches.

    Architecture:
        w = GatingNetTransformer(x_init)       # (B, 2) — from first K raw cycles
        y0 = branch0(x_full)                   # (B, 1)
        y1 = branch1(x_full)                   # (B, 1)
        y_final = w[:,0:1]*y0 + w[:,1:2]*y1   # (B, 1)

    Training loss (caller computes):
        MSE(y_final, y) + 0.05*MSE(y0, y) + 0.05*MSE(y1, y)
    """

    def __init__(
        self,
        n_features: int,
        K: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.gating = GatingNetTransformer(K, n_features)
        self.branch0 = TransformerBranch(
            n_features, d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.branch1 = TransformerBranch(
            n_features, d_model, nhead, num_layers, dim_feedforward, dropout
        )

    def forward(self, x_full: torch.Tensor, x_init: torch.Tensor):
        # x_full: (B, T, F)  — full 30-cycle window
        # x_init: (B, K, F)  — first K cycles (raw, same scale as x_full)
        w = self.gating(x_init)         # (B, 2)
        w0 = w[:, 0:1]                  # (B, 1)
        w1 = w[:, 1:2]                  # (B, 1)
        y0 = self.branch0(x_full)       # (B, 1)
        y1 = self.branch1(x_full)       # (B, 1)
        y_final = w0 * y0 + w1 * y1    # (B, 1)
        return y_final, y0, y1
