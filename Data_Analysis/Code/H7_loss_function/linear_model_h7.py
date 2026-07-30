"""
H7 Loss Function Optimization - Linear Model Wrapper
Fleet min-max normalised → sklearn LinearRegression on flattened 30-cycle windows.
Used only for Phase 1 screening (L1-L5).
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from config_h7 import WINDOW
from data_loader_h7 import load_train_val, load_test, build_windows
from loss_functions_h7 import nasa_score, rmse


def _flatten(X_windows: np.ndarray) -> np.ndarray:
    N, W, F = X_windows.shape
    return X_windows.reshape(N, W * F)


def _sample_weights(life_ratio: np.ndarray, loss_name: str) -> np.ndarray:
    """
    For L3 and L5 we use life_ratio-based sample weights for WLS.
    Other losses → plain OLS (weights=None).
    """
    if loss_name in ("L3_DynMSE",):
        return (1.0 + 1.0 * life_ratio).astype(np.float64)
    elif loss_name in ("L5_TWA",):
        # TWA uses both time-weight and asymmetry.
        # For linear OLS we approximate with time-weight only.
        return (1.0 + 1.0 * life_ratio).astype(np.float64)
    return None


def clip_name(clip_value) -> str:
    if clip_value is None:
        return "clip_none"
    return f"clip_{int(clip_value)}"


def run_linear(dataset: str, clip_value, loss_name: str) -> dict:
    """
    Train linear model and evaluate on test set.

    Returns dict: {clip, loss_fn, dataset, rmse, nasa_score}
    """
    train_eng, val_eng, scaler, feat_cols, clip_used = \
        load_train_val(dataset, clip_value)

    all_engines = train_eng + val_eng
    X_win, y_win, lr_win = build_windows(all_engines, window=WINDOW)
    X_flat = _flatten(X_win)
    sw     = _sample_weights(lr_win, loss_name)

    model = LinearRegression()
    model.fit(X_flat, y_win, sample_weight=sw)

    X_test, rul_true = load_test(dataset, scaler, feat_cols)
    X_test_flat      = _flatten(X_test)
    pred             = np.clip(model.predict(X_test_flat), 0, None).astype(np.float32)

    return {
        "clip":       clip_name(clip_value),
        "loss_fn":    loss_name,
        "dataset":    dataset,
        "rmse":       rmse(pred, rul_true),
        "nasa_score": nasa_score(pred, rul_true),
    }
