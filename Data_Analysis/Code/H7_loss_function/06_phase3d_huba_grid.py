"""
H7 Phase 3d — L7 HubA δ × λ_a Grid
δ: [10,20,30] × λ_a: [1.5,2.0,3.0] × 5 seeds = 45 runs (FD001 screening)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from itertools import product
from tqdm import tqdm

from config_h7 import SEEDS, RESULT_DIR
from lstm_model_h7 import train_lstm
from loss_functions_h7 import make_loss_L7

SAVE_PATH = os.path.join(RESULT_DIR, "phase3d_huba_grid.csv")

DELTA_GRID  = [10.0, 20.0, 30.0]
LAMBDA_GRID = [1.5, 2.0, 3.0]

DATASET    = "FD001"
CLIP_VALUE = 125

print("=" * 60)
print("H7 Phase 3d — L7 HubA δ × λ_a Grid")
combos = list(product(DELTA_GRID, LAMBDA_GRID, SEEDS))
print(f"Runs: {len(DELTA_GRID)}δ × {len(LAMBDA_GRID)}λ_a × {len(SEEDS)} seeds = {len(combos)}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

results = []
done_keys = set()
if os.path.exists(SAVE_PATH):
    ex = pd.read_csv(SAVE_PATH)
    for _, row in ex.iterrows():
        done_keys.add((float(row["delta"]), float(row["lambda_a"]), int(row["seed"])))
    results = ex.to_dict("records")
    print(f"Resuming — {len(done_keys)} runs done.")

with tqdm(total=len(combos), desc="Phase3d") as pbar:
    for delta, lam_a, seed in combos:
        key = (delta, lam_a, seed)
        if key in done_keys:
            pbar.update(1)
            continue
        try:
            loss_fn = make_loss_L7(delta=delta, lambda_a=lam_a)
            row = train_lstm(DATASET, CLIP_VALUE, "L7_HubA",
                             seed=seed, device=device, loss_fn_override=loss_fn)
            row.pop("preds",    None)
            row.pop("rul_true", None)
            row["delta"]    = delta
            row["lambda_a"] = lam_a
            results.append(row)
            done_keys.add(key)
            pbar.set_postfix(d=delta, la=lam_a, seed=seed,
                             nasa=f"{row['nasa_score']:.2f}")
        except Exception as e:
            print(f"\nERROR δ={delta} λ_a={lam_a} seed={seed}: {e}")
            results.append({
                "clip": "clip_125", "loss_fn": "L7_HubA",
                "dataset": DATASET, "seed": seed,
                "delta": delta, "lambda_a": lam_a,
                "rmse": float("nan"), "nasa_score": float("nan"),
            })
            done_keys.add(key)
        pbar.update(1)

df = pd.DataFrame(results)
df.to_csv(SAVE_PATH, index=False)
print(f"\nSaved → {SAVE_PATH}")

print("\nBest (δ, λ_a) combinations (mean over seeds):")
agg = (df.groupby(["delta", "lambda_a"])
         .agg(nasa_mean=("nasa_score", "mean"), nasa_std=("nasa_score", "std"),
              rmse_mean=("rmse", "mean"))
         .reset_index()
         .sort_values("nasa_mean"))
print(agg.round(4).to_string(index=False))
best = agg.iloc[0]
print(f"\nBest: δ={best['delta']}  λ_a={best['lambda_a']}  "
      f"NASA={best['nasa_mean']:.2f}±{best['nasa_std']:.2f}")
