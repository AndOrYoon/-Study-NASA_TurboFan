"""
H7 Phase 3c — L6 Pinball τ Sweep
τ: [0.25,0.35,0.40,0.45,0.50] × best_clip × 4 datasets × 5 seeds = 100 runs
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from itertools import product
from tqdm import tqdm

from config_h7 import CLIPS, DATASETS, SEEDS, RESULT_DIR
from lstm_model_h7 import train_lstm
from loss_functions_h7 import make_loss_L6

SAVE_PATH = os.path.join(RESULT_DIR, "phase3c_pinball_tau.csv")

TAU_GRID = [0.25, 0.35, 0.40, 0.45, 0.50]

# Determine best_clip from Phase 2b results; fallback to clip_125
PHASE2B_MATRIX = os.path.join(RESULT_DIR, "phase2b_results_matrix.csv")
if os.path.exists(PHASE2B_MATRIX):
    m = pd.read_csv(PHASE2B_MATRIX)
    best_combo = (m.groupby("clip")["nasa_mean"].mean()
                   .idxmin())
    best_clip_name  = best_combo
    best_clip_value = CLIPS[best_clip_name]
    print(f"Using best clip from Phase 2b: {best_clip_name}")
else:
    best_clip_name  = "clip_125"
    best_clip_value = 125
    print(f"Phase 2b not found; using fallback: {best_clip_name}")

print("=" * 60)
print("H7 Phase 3c — Pinball τ Sweep")
combos = list(product(TAU_GRID, DATASETS, SEEDS))
print(f"Runs: {len(TAU_GRID)}τ × {len(DATASETS)} ds × {len(SEEDS)} seeds = {len(combos)}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

results = []
done_keys = set()
if os.path.exists(SAVE_PATH):
    ex = pd.read_csv(SAVE_PATH)
    for _, row in ex.iterrows():
        done_keys.add((float(row["tau"]), row["dataset"], int(row["seed"])))
    results = ex.to_dict("records")
    print(f"Resuming — {len(done_keys)} runs done.")

with tqdm(total=len(combos), desc="Phase3c") as pbar:
    for tau, dataset, seed in combos:
        key = (tau, dataset, seed)
        if key in done_keys:
            pbar.update(1)
            continue
        try:
            loss_fn = make_loss_L6(tau=tau)
            row = train_lstm(dataset, best_clip_value, "L6_Pinball",
                             seed=seed, device=device, loss_fn_override=loss_fn)
            row.pop("preds",    None)
            row.pop("rul_true", None)
            row["tau"] = tau
            results.append(row)
            done_keys.add(key)
            pbar.set_postfix(tau=tau, ds=dataset, seed=seed,
                             nasa=f"{row['nasa_score']:.2f}")
        except Exception as e:
            print(f"\nERROR τ={tau} {dataset} seed={seed}: {e}")
            results.append({
                "clip": best_clip_name, "loss_fn": "L6_Pinball",
                "dataset": dataset, "seed": seed, "tau": tau,
                "rmse": float("nan"), "nasa_score": float("nan"),
            })
            done_keys.add(key)
        pbar.update(1)

df = pd.DataFrame(results)
df.to_csv(SAVE_PATH, index=False)
print(f"\nSaved → {SAVE_PATH}")

print("\nBest τ per dataset (NASA Score):")
agg = (df.groupby(["tau", "dataset"])
         .agg(nasa_mean=("nasa_score", "mean"), nasa_std=("nasa_score", "std"))
         .reset_index())
for ds in DATASETS:
    sub  = agg[agg["dataset"] == ds].sort_values("nasa_mean")
    best = sub.iloc[0]
    print(f"  {ds}: τ={best['tau']}  NASA={best['nasa_mean']:.2f}±{best['nasa_std']:.2f}")

print("\nOverall best τ (averaged across datasets):")
overall = (agg.groupby("tau")["nasa_mean"].mean()
              .reset_index()
              .sort_values("nasa_mean"))
print(overall.to_string(index=False))
