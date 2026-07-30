"""
H7 Phase 3a — L5 TWA Lambda Grid Search
FD001, clip_125, L5 TWA
λ_t: [0.5,1.0,2.0,3.0,5.0,10.0] × λ_a: [1.0,1.5,2.0,3.0,5.0] = 30 × 5 seeds = 150 runs
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from itertools import product
from tqdm import tqdm

from config_h7 import SEEDS, RESULT_DIR
from lstm_model_h7 import train_lstm
from loss_functions_h7 import make_loss_L5

SAVE_PATH = os.path.join(RESULT_DIR, "phase3a_lambda_grid.csv")

LAMBDA_T_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
LAMBDA_A_GRID = [1.0, 1.5, 2.0, 3.0, 5.0]

DATASET    = "FD001"
CLIP_VALUE = 125
CLIP_NAME  = "clip_125"

print("=" * 60)
print("H7 Phase 3a — L5 TWA Lambda Grid")
combos = list(product(LAMBDA_T_GRID, LAMBDA_A_GRID, SEEDS))
print(f"Runs: {len(LAMBDA_T_GRID)}×{len(LAMBDA_A_GRID)} params × {len(SEEDS)} seeds = {len(combos)}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

results = []

# Resume
done_keys = set()
if os.path.exists(SAVE_PATH):
    ex = pd.read_csv(SAVE_PATH)
    for _, row in ex.iterrows():
        done_keys.add((float(row["lambda_t"]), float(row["lambda_a"]), int(row["seed"])))
    results = ex.to_dict("records")
    print(f"Resuming — {len(done_keys)} runs done.")

with tqdm(total=len(combos), desc="Phase3a") as pbar:
    for lam_t, lam_a, seed in combos:
        key = (lam_t, lam_a, seed)
        if key in done_keys:
            pbar.update(1)
            continue
        try:
            loss_fn = make_loss_L5(lambda_t=lam_t, lambda_a=lam_a)
            row = train_lstm(DATASET, CLIP_VALUE, "L5_TWA", seed=seed,
                             device=device, loss_fn_override=loss_fn)
            row.pop("preds",    None)
            row.pop("rul_true", None)
            row["lambda_t"] = lam_t
            row["lambda_a"] = lam_a
            results.append(row)
            done_keys.add(key)
            pbar.set_postfix(lt=lam_t, la=lam_a, seed=seed,
                             nasa=f"{row['nasa_score']:.2f}")
        except Exception as e:
            print(f"\nERROR λ_t={lam_t} λ_a={lam_a} seed={seed}: {e}")
            results.append({
                "clip": CLIP_NAME, "loss_fn": "L5_TWA",
                "dataset": DATASET, "seed": seed,
                "lambda_t": lam_t, "lambda_a": lam_a,
                "rmse": float("nan"), "nasa_score": float("nan"),
            })
            done_keys.add(key)
        pbar.update(1)

df = pd.DataFrame(results)
df.to_csv(SAVE_PATH, index=False)
print(f"\nSaved → {SAVE_PATH}")

print("\nBest λ combinations by NASA Score (mean over seeds):")
agg = (df.groupby(["lambda_t", "lambda_a"])
         .agg(nasa_mean=("nasa_score", "mean"), nasa_std=("nasa_score", "std"),
              rmse_mean=("rmse", "mean"))
         .reset_index()
         .sort_values("nasa_mean"))
print(agg.head(10).round(4).to_string(index=False))
best = agg.iloc[0]
print(f"\nBest: λ_t={best['lambda_t']}  λ_a={best['lambda_a']}  "
      f"NASA={best['nasa_mean']:.2f}±{best['nasa_std']:.2f}")
