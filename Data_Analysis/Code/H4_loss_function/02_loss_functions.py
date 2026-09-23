"""
H7 Loss Function Optimization - Loss Functions
All seven loss functions as PyTorch callables.
Signature: loss_fn(pred, true, life_ratio=None, clip_value=None) -> scalar tensor
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# L1 – MSE (baseline)
# ─────────────────────────────────────────────────────────────────────────────
def loss_L1_mse(pred, true, life_ratio=None, clip_value=None):
    return torch.mean((pred - true) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# L2 – NASA Score Loss  (differentiable approximation)
# ─────────────────────────────────────────────────────────────────────────────
def loss_L2_nasa(pred, true, life_ratio=None, clip_value=None):
    d = pred - true
    # piecewise: d<0 → exp(-d/13)-1,  d>=0 → exp(d/10)-1
    s_neg = torch.exp(-d / 13.0) - 1.0
    s_pos = torch.exp( d / 10.0) - 1.0
    s = torch.where(d < 0, s_neg, s_pos)
    return torch.mean(s)


# ─────────────────────────────────────────────────────────────────────────────
# L3 – Dynamically Weighted MSE
# ─────────────────────────────────────────────────────────────────────────────
def loss_L3_dynmse(pred, true, life_ratio=None, clip_value=None,
                   lambda_dyn: float = 1.0):
    if life_ratio is None:
        # fallback to plain MSE
        return torch.mean((pred - true) ** 2)
    w = 1.0 + lambda_dyn * life_ratio
    return torch.mean(w * (pred - true) ** 2)


def make_loss_L3(lambda_dyn: float = 1.0):
    def fn(pred, true, life_ratio=None, clip_value=None):
        return loss_L3_dynmse(pred, true, life_ratio, clip_value, lambda_dyn)
    fn.__name__ = f"L3_DynMSE_λ{lambda_dyn}"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# L4 – Focal-RUL Loss
# ─────────────────────────────────────────────────────────────────────────────
def loss_L4_focal(pred, true, life_ratio=None, clip_value=None,
                  gamma: float = 2.0):
    resid   = torch.abs(pred - true)
    w_focal = (resid / (resid + 1.0)) ** gamma
    return torch.mean(w_focal * (pred - true) ** 2)


def make_loss_L4(gamma: float = 2.0):
    def fn(pred, true, life_ratio=None, clip_value=None):
        return loss_L4_focal(pred, true, life_ratio, clip_value, gamma)
    fn.__name__ = f"L4_Focal_γ{gamma}"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# L5 – Time-Weighted + Asymmetric (TWA) — novel proposed loss
# ─────────────────────────────────────────────────────────────────────────────
def loss_L5_twa(pred, true, life_ratio=None, clip_value=None,
                lambda_t: float = 1.0, lambda_a: float = 2.0):
    if life_ratio is None:
        w_time = torch.ones_like(true)
    else:
        w_time = 1.0 + lambda_t * life_ratio

    d = pred - true
    # asymmetry: late prediction (d>0) is penalised more
    w_asym = torch.where(d < 0,
                         torch.ones_like(d),
                         torch.full_like(d, lambda_a))
    return torch.mean(w_time * w_asym * d ** 2)


def make_loss_L5(lambda_t: float = 1.0, lambda_a: float = 2.0):
    def fn(pred, true, life_ratio=None, clip_value=None):
        return loss_L5_twa(pred, true, life_ratio, clip_value, lambda_t, lambda_a)
    fn.__name__ = f"L5_TWA_λt{lambda_t}_λa{lambda_a}"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# L6 – Pinball (Quantile) Loss
# ─────────────────────────────────────────────────────────────────────────────
def loss_L6_pinball(pred, true, life_ratio=None, clip_value=None,
                    tau: float = 0.35):
    d = pred - true
    loss_pos = (1.0 - tau) * d          # d > 0  (late prediction)
    loss_neg = tau * (-d)               # d < 0  (early prediction)
    return torch.mean(torch.where(d >= 0, loss_pos, loss_neg))


def make_loss_L6(tau: float = 0.35):
    def fn(pred, true, life_ratio=None, clip_value=None):
        return loss_L6_pinball(pred, true, life_ratio, clip_value, tau)
    fn.__name__ = f"L6_Pinball_τ{tau}"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# L7 – Huber-Asymmetric (HubA) — novel proposed loss
# ─────────────────────────────────────────────────────────────────────────────
def loss_L7_huba(pred, true, life_ratio=None, clip_value=None,
                 delta: float = 20.0, lambda_a: float = 2.0):
    d    = pred - true
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


def make_loss_L7(delta: float = 20.0, lambda_a: float = 2.0):
    def fn(pred, true, life_ratio=None, clip_value=None):
        return loss_L7_huba(pred, true, life_ratio, clip_value, delta, lambda_a)
    fn.__name__ = f"L7_HubA_δ{delta}_λa{lambda_a}"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Registry: name → callable
# ─────────────────────────────────────────────────────────────────────────────
from config_h7 import (
    LAMBDA_DYN, GAMMA_FOCAL, LAMBDA_T, LAMBDA_A,
    TAU_PINBALL, DELTA_HUBA, LAMBDA_A_HUBA
)

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
# NASA Score metric (numpy)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """NASA prognostics score: sum(s(d_i)) / N_engines.
    Penalises late predictions more than early ones.
    Lower is better (positive contribution for late, negative for early).
    """
    d = pred - true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    # s > 0 always → mean over engines
    return float(np.mean(s))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))
