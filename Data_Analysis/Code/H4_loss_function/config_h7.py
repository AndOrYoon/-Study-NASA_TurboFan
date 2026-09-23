"""
H7 Loss Function Optimization - Configuration
Constants and shared settings for all H7 scripts.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR     = r"C:\BMAD_PY313\Dataset"
RESULT_DIR   = r"C:\BMAD_PY313\Data_Analysis\Results\H4_loss_function"
RAW_PRED_DIR = os.path.join(RESULT_DIR, "raw_predictions")
FIGURES_DIR  = os.path.join(RESULT_DIR, "figures")

os.makedirs(RAW_PRED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR,  exist_ok=True)

# ── Experiment settings ───────────────────────────────────────────────────────
CLIPS = {
    "clip_100":  100,
    "clip_125":  125,   # baseline
    "clip_130":  130,
    "clip_none": None,  # no clipping → use max_train_rul
}

DATASETS   = ["FD001", "FD002", "FD003", "FD004"]
LOSS_NAMES = ["L1_MSE", "L2_NASA", "L3_DynMSE", "L4_Focal",
              "L5_TWA", "L6_Pinball", "L7_HubA"]
SEEDS      = [0, 1, 2, 3, 4]
WINDOW     = 30          # look-back window
VAL_FRAC   = 0.20        # last 20 % of each engine → validation

# ── LSTM hyper-parameters ─────────────────────────────────────────────────────
LSTM_HIDDEN    = 32
LSTM_DROPOUT   = 0.2
LSTM_LR        = 1e-3
LSTM_WD        = 1e-4
LSTM_BATCH     = 256
LSTM_MAX_EPOCH = 100
LSTM_PATIENCE  = 15

# ── Loss-function defaults ────────────────────────────────────────────────────
LAMBDA_DYN    = 1.0    # L3
GAMMA_FOCAL   = 2.0    # L4
LAMBDA_T      = 1.0    # L5 time weight
LAMBDA_A      = 2.0    # L5/L7 asymmetry
TAU_PINBALL   = 0.35   # L6
DELTA_HUBA    = 20.0   # L7 Huber threshold
LAMBDA_A_HUBA = 2.0    # L7 asymmetry

# ── Constant sensors to drop per dataset ─────────────────────────────────────
CONST_SENSORS = {
    "FD001": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
    "FD002": ["s16"],
    "FD003": ["s1", "s5", "s10", "s16", "s18", "s19"],
    "FD004": ["s16"],
}

# ── Column names (26 cols, 0-indexed) ────────────────────────────────────────
COL_NAMES = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)
