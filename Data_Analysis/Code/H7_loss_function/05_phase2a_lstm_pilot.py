"""
H7 Phase 2a — LSTM Pilot (FD001 × Top-5 combos × 5 seeds = 25 runs)
Checks Spearman ρ between linear ranking and LSTM ranking.
If ρ > 0.7 → proceed to Phase 2b.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import pandas as pd
from scipy import stats
from tqdm import tqdm

from config_h7 import CLIPS, SEEDS, RESULT_DIR
from lstm_model_h7 import train_lstm

SAVE_PATH  = os.path.join(RESULT_DIR, "phase2a_lstm_pilot.csv")
TOP5_PATH  = os.path.join(RESULT_DIR, "phase1_top5.json")

print("=" * 60)
print("H7 Phase 2a — LSTM Pilot")
print("=" * 60)

# ── Load top-5 from Phase 1 ───────────────────────────────────────────────────
if not os.path.exists(TOP5_PATH):
    raise FileNotFoundError(
        f"Top-5 file not found: {TOP5_PATH}\n"
        "Run 04_phase1_linear_screening.py first."
    )
with open(TOP5_PATH) as f:
    top5 = json.load(f)   # list of [clip_name, loss_name]

print(f"Top-5 combos from Phase 1:")
for i, (c, l) in enumerate(top5, 1):
    print(f"  {i}. {c}  {l}")

# ── Resolve clip name → clip value ───────────────────────────────────────────
def clip_val_from_name(name: str):
    return CLIPS[name]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Run pilot ────────────────────────────────────────────────────────────────
total   = len(top5) * len(SEEDS)
results = []

with tqdm(total=total, desc="Phase2a") as pbar:
    for clip_name, loss_name in top5:
        clip_value = clip_val_from_name(clip_name)
        for seed in SEEDS:
            try:
                row = train_lstm("FD001", clip_value, loss_name,
                                 seed=seed, device=device)
                row.pop("preds",    None)
                row.pop("rul_true", None)
                results.append(row)
                pbar.set_postfix(clip=clip_name, loss=loss_name,
                                 seed=seed, nasa=f"{row['nasa_score']:.2f}")
            except Exception as e:
                print(f"\nERROR: {clip_name} {loss_name} seed={seed}: {e}")
                results.append({
                    "clip": clip_name, "loss_fn": loss_name,
                    "dataset": "FD001", "seed": seed,
                    "rmse": float("nan"), "nasa_score": float("nan"),
                })
            pbar.update(1)

df = pd.DataFrame(results)
df.to_csv(SAVE_PATH, index=False)
print(f"\nSaved → {SAVE_PATH}")

# ── Spearman rank correlation with Phase 1 ───────────────────────────────────
# Phase 1 ranking (mean NASA over 4 datasets per (clip, loss))
phase1_csv = os.path.join(RESULT_DIR, "phase1_linear_screening.csv")
df_p1      = pd.read_csv(phase1_csv)
p1_rank    = (df_p1.groupby(["clip", "loss_fn"])["nasa_score"]
                .mean()
                .reset_index()
                .rename(columns={"nasa_score": "nasa_linear"}))

# Phase 2a ranking (mean NASA over 5 seeds, FD001 only)
p2a_rank   = (df.groupby(["clip", "loss_fn"])["nasa_score"]
                 .mean()
                 .reset_index()
                 .rename(columns={"nasa_score": "nasa_lstm"}))

merged = p2a_rank.merge(p1_rank, on=["clip", "loss_fn"], how="left")
merged["rank_linear"] = merged["nasa_linear"].rank()
merged["rank_lstm"]   = merged["nasa_lstm"].rank()

rho, p_val = stats.spearmanr(merged["rank_linear"], merged["rank_lstm"])
print(f"\nSpearman ρ (linear vs LSTM ranking): {rho:.3f}  p={p_val:.4f}")

if rho > 0.7:
    print("ρ > 0.7  → Proceed to Phase 2b (Full LSTM experiment)")
else:
    print("ρ ≤ 0.7  → Warning: linear screening may not predict LSTM ranking well.")
    print("           Proceeding to Phase 2b regardless (full evaluation).")

print("\nPhase 2a pilot results (mean ± std over seeds):")
summary = (df.groupby(["clip", "loss_fn"])[["rmse", "nasa_score"]]
             .agg(["mean", "std"])
             .round(4))
print(summary.to_string())
