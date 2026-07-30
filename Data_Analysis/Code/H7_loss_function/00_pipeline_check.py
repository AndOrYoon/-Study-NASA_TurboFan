"""
H7 Phase 0 — Pipeline Check
Quick end-to-end: FD001, clip_125, L1 MSE, linear model, single run.
Verifies data loading, loss computation, metric calculation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

print("=" * 60)
print("H7 Pipeline Check")
print("=" * 60)

# ── Config ───────────────────────────────────────────────────────────────────
from config_h7 import DATA_DIR, RESULT_DIR, WINDOW
print(f"DATA_DIR   : {DATA_DIR}")
print(f"RESULT_DIR : {RESULT_DIR}")
print(f"WINDOW     : {WINDOW}")

# ── GPU ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device     : {device}")
if device.type == "cuda":
    print(f"GPU        : {torch.cuda.get_device_name(0)}")

# ── Data load ────────────────────────────────────────────────────────────────
from data_loader_h7 import load_train_val, load_test, build_windows

dataset    = "FD001"
clip_value = 125

print(f"\nLoading {dataset} with clip={clip_value} ...")
train_eng, val_eng, scaler, feat_cols, clip_used = \
    load_train_val(dataset, clip_value)
print(f"  Train engines : {len(train_eng)}")
print(f"  Val   engines : {len(val_eng)}")
print(f"  Feature cols  : {len(feat_cols)}  → {feat_cols[:5]} ...")
print(f"  clip_used     : {clip_used}")

# Sample engine stats
e0 = train_eng[0]
print(f"  Engine[0] shape: X={e0['X'].shape}  rul={e0['rul'].shape}")
print(f"  Engine[0] rul  : min={e0['rul'].min():.1f}  max={e0['rul'].max():.1f}")
print(f"  Engine[0] lr   : min={e0['life_ratio'].min():.3f}  max={e0['life_ratio'].max():.3f}")

# ── Build windows ────────────────────────────────────────────────────────────
X_tr, y_tr, lr_tr = build_windows(train_eng, WINDOW)
print(f"\nTrain windows : X={X_tr.shape}  y={y_tr.shape}  lr={lr_tr.shape}")
print(f"  y range: [{y_tr.min():.1f}, {y_tr.max():.1f}]")
print(f"  lr range: [{lr_tr.min():.3f}, {lr_tr.max():.3f}]")

# ── Loss functions ───────────────────────────────────────────────────────────
from loss_functions_h7 import LOSS_REGISTRY, nasa_score, rmse

print("\nLoss function smoke test ...")
pred_t  = torch.tensor([80.0, 60.0, 40.0, 20.0, 5.0])
true_t  = torch.tensor([90.0, 55.0, 45.0, 25.0, 0.0])
lr_t    = torch.tensor([0.1,  0.3,  0.5,  0.7,  0.9])

for name, fn in LOSS_REGISTRY.items():
    val = fn(pred_t, true_t, life_ratio=lr_t, clip_value=125)
    print(f"  {name:15s}: {val.item():.4f}")

# ── Linear model ─────────────────────────────────────────────────────────────
from linear_model_h7 import run_linear
print("\nRunning linear model (L1 MSE, FD001, clip_125) ...")
result = run_linear("FD001", 125, "L1_MSE")
print(f"  RMSE       : {result['rmse']:.4f}")
print(f"  NASA Score : {result['nasa_score']:.4f}")

# ── Test loader ──────────────────────────────────────────────────────────────
X_test, rul_true = load_test(dataset, scaler, feat_cols)
print(f"\nTest windows  : X={X_test.shape}  rul={rul_true.shape}")
print(f"  rul range   : [{rul_true.min():.1f}, {rul_true.max():.1f}]")

# ── Metric helpers ───────────────────────────────────────────────────────────
dummy_pred = rul_true + np.random.randn(len(rul_true)) * 10
print(f"\nDummy pred metrics (noise σ=10):")
print(f"  RMSE       : {rmse(dummy_pred, rul_true):.4f}")
print(f"  NASA Score : {nasa_score(dummy_pred, rul_true):.4f}")

print("\n" + "=" * 60)
print("Pipeline check PASSED")
print("=" * 60)
