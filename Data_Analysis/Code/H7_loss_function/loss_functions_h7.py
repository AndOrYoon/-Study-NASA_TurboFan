"""
H7 Loss Function Optimization - Loss Functions (L1-L7)
All accept: (pred, true, life_ratio=None, clip_value=None) -> scalar tensor
"""

import torch
import numpy as np

from config_h7 import (
    LAMBDA_DYN, GAMMA_FOCAL, LAMBDA_T, LAMBDA_A,
    TAU_PINBALL, DELTA_HUBA, LAMBDA_A_HUBA
)


# ─────────────────────────────────────────────────────────────────────────────
# L1 – MSE
# ─────────────────────────────────────────────────────────────────────────────
def loss_L1_mse(pred, true, life_ratio=None, clip_value=None):
    return torch.mean((pred - true) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# L2 – NASA Score Loss
# ─────────────────────────────────────────────────────────────────────────────
def loss_L2_nasa(pred, true, life_ratio=None, clip_value=None):
    d     = pred - true
    s_neg = torch.exp(-d / 13.0) - 1.0
    s_pos = torch.exp( d / 10.0) - 1.0
    return torch.mean(torch.where(d < 0, s_neg, s_pos))


# ─────────────────────────────────────────────────────────────────────────────
# L3 – Dynamically Weighted MSE
# ─────────────────────────────────────────────────────────────────────────────
def make_loss_L3(lambda_dyn: float = LAMBDA_DYN):
    def fn(pred, true, life_ratio=None, clip_value=None):
        if life_ratio is None:
            return torch.mean((pred - true) ** 2)
        w = 1.0 + lambda_dyn * life_ratio
        return torch.mean(w * (pred - true) ** 2)
    fn.__name__ = f"L3_DynMSE_lam{lambda_dyn}"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# L4 – Focal-RUL
# ─────────────────────────────────────────────────────────────────────────────
def make_loss_L4(gamma: float = GAMMA_FOCAL):
    def fn(pred, true, life_ratio=None, clip_value=None):
        resid   = torch.abs(pred - true)
        w_focal = (resid / (resid + 1.0)) ** gamma
        return torch.mean(w_focal * (pred - true) ** 2)
    fn.__name__ = f"L4_Focal_gam{gamma}"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# L5 – Time-Weighted + Asymmetric (TWA) — novel
# ─────────────────────────────────────────────────────────────────────────────
def make_loss_L5(lambda_t: float = LAMBDA_T, lambda_a: float = LAMBDA_A):
    def fn(pred, true, life_ratio=None, clip_value=None):
        if life_ratio is None:
            w_time = torch.ones_like(true)
        else:
            w_time = 1.0 + lambda_t * life_ratio
        d      = pred - true
        w_asym = torch.where(d < 0,
                             torch.ones_like(d),
                             torch.full_like(d, lambda_a))
        return torch.mean(w_time * w_asym * d ** 2)
    fn.__name__ = f"L5_TWA_lt{lambda_t}_la{lambda_a}"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# L6 – Pinball (Quantile) Loss
# ─────────────────────────────────────────────────────────────────────────────
def make_loss_L6(tau: float = TAU_PINBALL):
    def fn(pred, true, life_ratio=None, clip_value=None):
        d        = pred - true
        loss_pos = (1.0 - tau) * d
        loss_neg = tau * (-d)
        return torch.mean(torch.where(d >= 0, loss_pos, loss_neg))
    fn.__name__ = f"L6_Pinball_tau{tau}"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# L7 – Huber-Asymmetric (HubA) — novel
# ─────────────────────────────────────────────────────────────────────────────
def make_loss_L7(delta: float = DELTA_HUBA, lambda_a: float = LAMBDA_A_HUBA):
    def fn(pred, true, life_ratio=None, clip_value=None):
        d     = pred - true
        abs_d = torch.abs(d)
        huber = torch.where(
            abs_d <= delta,
            0.5 * d ** 2,
            delta * (abs_d - 0.5 * delta)
        )
        w_asym = torch.where(d < 0,
                             torch.ones_like(d),
                             torch.full_like(d, lambda_a))
        return torch.mean(huber * w_asym)
    fn.__name__ = f"L7_HubA_delta{delta}_la{lambda_a}"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
LOSS_REGISTRY = {
    "L1_MSE":     loss_L1_mse,
    "L2_NASA":    loss_L2_nasa,
    "L3_DynMSE":  make_loss_L3(LAMBDA_DYN),
    "L4_Focal":   make_loss_L4(GAMMA_FOCAL),
    "L5_TWA":     make_loss_L5(LAMBDA_T, LAMBDA_A),
    "L6_Pinball": make_loss_L6(TAU_PINBALL),
    "L7_HubA":    make_loss_L7(DELTA_HUBA, LAMBDA_A_HUBA),
}


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers (numpy)
# ─────────────────────────────────────────────────────────────────────────────
def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Per-engine NASA score (lower is better; 0 = perfect)."""
    d = pred - true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0,
                          np.exp( d / 10.0) - 1.0)
    return float(np.mean(s))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))
