"""
H7 Phase 2b — Full LSTM Experiment
4 clips × 7 losses × 4 datasets × 5 seeds = 560 runs
Saves raw predictions per run and aggregate results matrix.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

from config_h7 import CLIPS, DATASETS, LOSS_NAMES, SEEDS, RESULT_DIR, RAW_PRED_DIR
from lstm_model_h7 import train_lstm

SAVE_MATRIX = os.path.join(RESULT_DIR, "phase2b_results_matrix.csv")
SAVE_RAW    = os.path.join(RESULT_DIR, "phase2b_all_seeds.csv")

print("=" * 60)
print("H7 Phase 2b — Full LSTM (560 runs)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

combos = list(product(CLIPS.items(), LOSS_NAMES, DATASETS, SEEDS))
print(f"Total runs: {len(combos)}")

results = []

# ── Try to resume from partial run ───────────────────────────────────────────
done_keys = set()
if os.path.exists(SAVE_RAW):
    existing = pd.read_csv(SAVE_RAW)
    for _, row in existing.iterrows():
        done_keys.add((row["clip"], row["loss_fn"], row["dataset"], int(row["seed"])))
    results = existing.to_dict("records")
    print(f"Resuming — {len(done_keys)} runs already done.")

with tqdm(total=len(combos), desc="Phase2b") as pbar:
    for (clip_name, clip_val), loss_name, dataset, seed in combos:
        key = (clip_name, loss_name, dataset, seed)
        if key in done_keys:
            pbar.update(1)
            continue

        try:
            row = train_lstm(dataset, clip_val, loss_name,
                             seed=seed, device=device)

            # Save raw predictions
            preds_df = pd.DataFrame({
                "pred":     row["preds"],
                "rul_true": row["rul_true"],
            })
            fname = f"{clip_name}_{loss_name}_{dataset}_seed{seed}.csv"
            preds_df.to_csv(os.path.join(RAW_PRED_DIR, fname), index=False)

            row_clean = {k: v for k, v in row.items()
                         if k not in ("preds", "rul_true")}
            results.append(row_clean)
            done_keys.add(key)

            pbar.set_postfix(
                clip=clip_name, loss=loss_name, ds=dataset, seed=seed,
                nasa=f"{row['nasa_score']:.2f}", rmse=f"{row['rmse']:.2f}"
            )
        except Exception as e:
            print(f"\nERROR: {clip_name} {loss_name} {dataset} seed={seed}: {e}")
            import traceback; traceback.print_exc()
            results.append({
                "clip": clip_name, "loss_fn": loss_name,
                "dataset": dataset, "seed": seed,
                "rmse": float("nan"), "nasa_score": float("nan"),
            })
            done_keys.add(key)
        pbar.update(1)

        # Checkpoint every 50 runs
        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(SAVE_RAW, index=False)

df = pd.DataFrame(results)
df.to_csv(SAVE_RAW, index=False)
print(f"\nRaw results saved → {SAVE_RAW}")

# ── Aggregate matrix ──────────────────────────────────────────────────────────
matrix = (df.groupby(["clip", "loss_fn", "dataset"])
            .agg(
                rmse_mean=("rmse",       "mean"),
                rmse_std= ("rmse",       "std"),
                nasa_mean=("nasa_score", "mean"),
                nasa_std= ("nasa_score", "std"),
            )
            .reset_index()
            .round(4))

matrix.to_csv(SAVE_MATRIX, index=False)
print(f"Aggregated matrix saved → {SAVE_MATRIX}")

# ── Quick summary ─────────────────────────────────────────────────────────────
print("\nBest (clip, loss_fn) per dataset by NASA Score (mean over seeds):")
for ds in DATASETS:
    sub  = matrix[matrix["dataset"] == ds].sort_values("nasa_mean")
    best = sub.iloc[0]
    print(f"  {ds}: {best['clip']:10s} {best['loss_fn']:12s} "
          f"NASA={best['nasa_mean']:.2f}±{best['nasa_std']:.2f}  "
          f"RMSE={best['rmse_mean']:.2f}±{best['rmse_std']:.2f}")

print("\nL1 MSE baseline (clip_125) per dataset:")
bl = matrix[(matrix["clip"] == "clip_125") & (matrix["loss_fn"] == "L1_MSE")]
for _, row in bl.iterrows():
    print(f"  {row['dataset']}: NASA={row['nasa_mean']:.2f}  RMSE={row['rmse_mean']:.2f}")
