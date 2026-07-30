"""
H7 Loss Function Optimization - Linear Model (Phase 1 screening)
Fleet min-max normalised → sklearn LinearRegression on last-30-cycles features.

For Phase 1 we only need L1-L5 (no GPU); L6/L7 need LSTM.
The "loss function" here affects how predictions are post-processed
(for L1/L2/L4 we use standard OLS; for L3/L5 we weight the samples).
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array

from config_h7 import WINDOW
from data_loader_h7 import load_train_val, load_test, build_windows
from loss_functions_h7 import nasa_score, rmse


# ─────────────────────────────────────────────────────────────────────────────
def _flatten_window(X_windows: np.ndarray) -> np.ndarray:
    """(N, W, F) → (N, W*F)"""
    N, W, F = X_windows.shape
    return X_windows.reshape(N, W * F)


def _get_sample_weights(life_ratio: np.ndarray, loss_name: str,
                        lambda_dyn: float = 1.0,
                        lambda_t: float = 1.0) -> np.ndarray | None:
    """Return sklearn sample_weight array for weighted regression,
    or None for unweighted."""
    if loss_name == "L3_DynMSE":
        return (1.0 + lambda_dyn * life_ratio).astype(np.float64)
    elif loss_name == "L5_TWA":
        return (1.0 + lambda_t * life_ratio).astype(np.float64)
    else:
        return None   # L1/L2/L4 – ordinary least squares


# ─────────────────────────────────────────────────────────────────────────────
def run_linear(dataset: str, clip_value, loss_name: str) -> dict:
    """
    Train a linear model on one (dataset, clip, loss) combo and
    evaluate on the test set.

    Returns
    -------
    dict with keys: clip, loss_fn, dataset, rmse, nasa_score
    """
    # Load data
    train_eng, val_eng, scaler, feat_cols, clip_used = \
        load_train_val(dataset, clip_value)

    # Build sliding-window arrays (train + val combined for linear model)
    all_engines = train_eng + val_eng
    X_win, y_win, lr_win = build_windows(all_engines, window=WINDOW)
    X_flat = _flatten_window(X_win)

    sw = _get_sample_weights(lr_win, loss_name)

    # Fit
    model = LinearRegression()
    model.fit(X_flat, y_win, sample_weight=sw)

    # Test
    X_test, rul_true = load_test(dataset, scaler, feat_cols)
    X_test_flat      = _flatten_window(X_test)
    pred             = model.predict(X_test_flat).astype(np.float32)
    pred             = np.clip(pred, 0, None)   # RUL cannot be negative

    return {
        "clip":      _clip_name(clip_value),
        "loss_fn":   loss_name,
        "dataset":   dataset,
        "rmse":      rmse(pred, rul_true),
        "nasa_score": nasa_score(pred, rul_true),
    }


def _clip_name(clip_value) -> str:
    if clip_value is None:
        return "clip_none"
    return f"clip_{clip_value}"
