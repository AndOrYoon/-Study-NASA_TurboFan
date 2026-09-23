"""
H7 Phase 1 — Linear Model Screening
4 clips × 5 losses (L1-L5) × 4 datasets = 80 runs
Identifies Top-5 (clip, loss_fn) combinations by NASA Score.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from tqdm import tqdm

from config_h7 import CLIPS, DATASETS, RESULT_DIR
from linear_model_h7 import run_linear

# Phase 1 only uses L1-L5 (L6/L7 are LSTM-only)
PHASE1_LOSSES = ["L1_MSE", "L2_NASA", "L3_DynMSE", "L4_Focal", "L5_TWA"]

SAVE_PATH = os.path.join(RESULT_DIR, "phase1_linear_screening.csv")

print("=" * 60)
print("H7 Phase 1 — Linear Screening")
print(f"Runs: {len(CLIPS)} clips × {len(PHASE1_LOSSES)} losses × {len(DATASETS)} datasets = "
      f"{len(CLIPS) * len(PHASE1_LOSSES) * len(DATASETS)}")
print("=" * 60)

results = []
total   = len(CLIPS) * len(PHASE1_LOSSES) * len(DATASETS)

with tqdm(total=total, desc="Phase1") as pbar:
    for clip_name, clip_val in CLIPS.items():
        for loss_name in PHASE1_LOSSES:
            for dataset in DATASETS:
                try:
                    row = run_linear(dataset, clip_val, loss_name)
                    results.append(row)
                    pbar.set_postfix(
                        clip=clip_name, loss=loss_name, ds=dataset,
                        nasa=f"{row['nasa_score']:.2f}"
                    )
                except Exception as e:
                    print(f"\nERROR: {clip_name} {loss_name} {dataset}: {e}")
                    results.append({
                        "clip": clip_name, "loss_fn": loss_name,
                        "dataset": dataset, "rmse": float("nan"),
                        "nasa_score": float("nan")
                    })
                pbar.update(1)

df = pd.DataFrame(results)
df.to_csv(SAVE_PATH, index=False)
print(f"\nSaved → {SAVE_PATH}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\nTop-10 (clip, loss_fn) by NASA Score (lower = better):")
top = (df.groupby(["clip", "loss_fn"])["nasa_score"]
         .mean()
         .reset_index()
         .sort_values("nasa_score"))
print(top.head(10).to_string(index=False))

print("\nTop-5 (clip, loss_fn) combos — these advance to Phase 2a:")
top5 = top.head(5)[["clip", "loss_fn"]].values.tolist()
for i, (c, l) in enumerate(top5, 1):
    print(f"  {i}. {c}  {l}")

# Save top5 for Phase 2a
import json
top5_path = os.path.join(RESULT_DIR, "phase1_top5.json")
with open(top5_path, "w") as f:
    json.dump(top5, f, indent=2)
print(f"\nTop-5 saved → {top5_path}")

print("\nPer-dataset summary:")
print(df.groupby(["dataset", "loss_fn"])["nasa_score"].mean()
        .unstack().round(2).to_string())
