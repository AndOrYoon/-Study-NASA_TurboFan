# -*- coding: utf-8 -*-
"""
04_run_experiments.py
---------------------
Main experiment loop: 7 normalizers × 4 datasets × 5 seeds = 140 runs.

For each run:
  1. Load data  (01_data_loader.py)
  2. Apply K-means residualization for FD002/FD004  (op_condition_utils.py)
  3. Apply normalizer N1-N6  (02_normalizers.py)  or skip for N7 (RevIN)
  4. Train LSTM with early stopping
  5. Evaluate on test set
  6. Save per-engine predictions to:
       Results/H5_normalization/raw_predictions/{dataset}_{norm_id}_seed{seed}.csv

Usage
-----
    python 04_run_experiments.py                   # all 140 runs
    python 04_run_experiments.py --pilot           # FD001 × N1 × seed=0 only
    python 04_run_experiments.py --dataset FD001   # one dataset, all norms/seeds
"""

import sys
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
CODE_DIR    = Path(__file__).parent
SHARED_DIR  = CODE_DIR.parent / "shared"
ROOT        = CODE_DIR.parent.parent        # …/Data_Analysis
RESULTS_DIR = ROOT / "Results" / "H5_normalization"
RAW_DIR     = RESULTS_DIR / "raw_predictions"
RAW_DIR.mkdir(parents=True, exist_ok=True)

for p in [str(CODE_DIR), str(SHARED_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Dynamic import of numbered modules  (01_…, 02_…, 03_…)
# ---------------------------------------------------------------------------

def _load(alias: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(alias, str(filepath))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

_dl = _load("dl",  CODE_DIR / "01_data_loader.py")
_nm = _load("nm",  CODE_DIR / "02_normalizers.py")
_lm = _load("lm",  CODE_DIR / "03_lstm_model.py")

# Bind symbols
load_raw            = _dl.load_raw
drop_const_sensors  = _dl.drop_const_sensors
add_rul_labels      = _dl.add_rul_labels
get_feature_cols    = _dl.get_feature_cols
make_sequences      = _dl.make_sequences
make_test_sequences = _dl.make_test_sequences
load_true_rul       = _dl.load_true_rul
get_lifetime_groups = _dl.get_lifetime_groups

get_normalizer   = _nm.get_normalizer
NORMALIZER_NAMES = _nm.NORMALIZER_NAMES

LSTMBase       = _lm.LSTMBase
LSTMWithRevIN  = _lm.LSTMWithRevIN
set_seed       = _lm.set_seed
device         = _lm.device
train_epoch    = _lm.train_epoch
eval_epoch     = _lm.eval_epoch
predict_all    = _lm.predict_all
nasa_score     = _lm.nasa_score

# Shared utility (plain filename, already on sys.path via SHARED_DIR)
import op_condition_utils as _ocu
fit_op_condition_kmeans = _ocu.fit_op_condition_kmeans
compute_cluster_means   = _ocu.compute_cluster_means
apply_op_residual       = _ocu.apply_op_residual


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DATASETS     = ["FD001", "FD002", "FD003", "FD004"]
SEEDS        = [0, 1, 2, 3, 4]
BATCH_SIZE   = 256
MAX_EPOCHS   = 100
PATIENCE     = 15
LR           = 1e-3
WEIGHT_DECAY = 1e-4
WINDOW       = 30
RUL_CLIP     = 125
VAL_FRAC     = 0.20   # last 20 % of cycles per engine → validation

NORM_IDS = ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]
NORM_NAMES = {
    "N1": "fleet_minmax",
    "N2": "fleet_std",
    "N3": "perunit_minmax_5",
    "N4": "perunit_minmax_10",
    "N5": "perunit_std_5",
    "N6": "perunit_std_10",
    "N7": "revin",   # handled by LSTMWithRevIN; no preprocessing
}

MULTI_COND = {"FD002", "FD004"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def val_split(train_df: pd.DataFrame, val_frac: float = VAL_FRAC):
    """Hold out val_frac of engines (not cycles) for validation.
    Engine-level split ensures val set contains the full RUL range,
    preventing train(high RUL) / val(low RUL) distribution mismatch."""
    units = np.array(sorted(train_df["unit"].unique()))
    n_val = max(1, int(len(units) * val_frac))
    val_units = set(units[-n_val:])
    mask = train_df["unit"].isin(val_units)
    return (train_df[~mask].reset_index(drop=True),
            train_df[mask].reset_index(drop=True))


def build_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=BATCH_SIZE):
    tr_ds  = TensorDataset(torch.tensor(X_tr,  dtype=torch.float32),
                           torch.tensor(y_tr,  dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    tr_ldr  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=True)
    val_ldr = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=True)
    return tr_ldr, val_ldr


def build_model(norm_id: str, n_features: int) -> nn.Module:
    if norm_id == "N7":
        return LSTMWithRevIN(n_features).to(device)
    return LSTMBase(n_features).to(device)


def train_model(model, tr_ldr, val_ldr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    best_val         = float("inf")
    patience_counter = 0
    best_state       = None

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss  = train_epoch(model, tr_ldr, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_ldr, criterion, device)

        if val_loss < best_val:
            best_val         = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def get_test_lifetime(train_df: pd.DataFrame) -> dict:
    """Proxy test-engine lifetime from train data (same unit numbering)."""
    lifetimes = train_df.groupby("unit")["cycle"].max()
    def _label(lt):
        if lt < 150:  return "boundary"
        elif lt < 250: return "medium"
        else:          return "long"
    return {uid: _label(lt) for uid, lt in lifetimes.items()}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(dataset: str, norm_id: str, seed: int, verbose: bool = True):
    t0 = time.time()
    set_seed(seed)

    # 1. Load raw data
    train_raw = load_raw(dataset, "train")
    test_raw  = load_raw(dataset, "test")
    train_raw = drop_const_sensors(train_raw, dataset)
    test_raw  = drop_const_sensors(test_raw,  dataset)
    train_raw = add_rul_labels(train_raw, clip=RUL_CLIP)

    feat_cols = get_feature_cols(train_raw)

    # 2. K-means residualization (FD002/FD004 only)
    if dataset in MULTI_COND:
        km_scaler, km_model = fit_op_condition_kmeans(train_raw, k=6)
        cluster_means       = compute_cluster_means(train_raw, km_scaler,
                                                    km_model, feat_cols)
        train_raw = apply_op_residual(train_raw, km_scaler, km_model,
                                      cluster_means, feat_cols)
        test_raw  = apply_op_residual(test_raw,  km_scaler, km_model,
                                      cluster_means, feat_cols)

    # 3. Normalisation
    if norm_id != "N7":
        norm = get_normalizer(NORM_NAMES[norm_id])
        norm.fit(train_raw, feat_cols)

        tr_normed = norm.transform_train(train_raw, feat_cols)
        te_normed = norm.transform_test(test_raw,   feat_cols)

        train_df = train_raw.copy()
        train_df[feat_cols] = tr_normed[feat_cols].values.astype(np.float32)

        test_df = test_raw.copy()
        test_df[feat_cols]  = te_normed[feat_cols].values.astype(np.float32)
    else:
        train_df = train_raw.copy()
        test_df  = test_raw.copy()

    # 4. Val split and sequence generation
    tr_df, val_df = val_split(train_df, val_frac=VAL_FRAC)

    X_tr,  y_tr  = make_sequences(tr_df,  feat_cols, window=WINDOW)
    X_val, y_val = make_sequences(val_df, feat_cols, window=WINDOW)

    # 5. Build model and train
    n_features = len(feat_cols)
    model      = build_model(norm_id, n_features)
    tr_ldr, val_ldr = build_dataloaders(X_tr, y_tr, X_val, y_val)
    model = train_model(model, tr_ldr, val_ldr)

    # 6. Test evaluation
    X_test, unit_ids = make_test_sequences(test_df, feat_cols, window=WINDOW)
    true_rul         = load_true_rul(dataset, clip=RUL_CLIP)
    pred_rul         = predict_all(model, X_test, device)
    pred_rul         = np.clip(pred_rul, 0, RUL_CLIP)

    rmse  = float(np.sqrt(np.mean((pred_rul - true_rul) ** 2)))
    score = nasa_score(true_rul, pred_rul)

    lt_groups = get_test_lifetime(train_raw)

    # 7. Save per-engine predictions
    out_df = pd.DataFrame({
        "unit":     unit_ids,
        "true_rul": true_rul,
        "pred_rul": pred_rul,
        "error":    pred_rul - true_rul,
        "group":    [lt_groups.get(uid, "unknown") for uid in unit_ids],
    })
    out_path = RAW_DIR / f"{dataset}_{norm_id}_seed{seed}.csv"
    out_df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    if verbose:
        print(f"  [{dataset} {norm_id} seed={seed}]  "
              f"RMSE={rmse:.4f}  Score={score:.1f}  t={elapsed:.1f}s")

    return {"dataset": dataset, "norm_id": norm_id, "seed": seed,
            "rmse": rmse, "nasa_score": score, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H5 normalisation experiment runner")
    parser.add_argument("--pilot",   action="store_true",
                        help="Run pilot: FD001 × N1 × seed=0")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Restrict to one dataset")
    parser.add_argument("--norm",    type=str, default=None,
                        help="Restrict to one normalizer (N1-N7)")
    args = parser.parse_args()

    if args.pilot:
        datasets = ["FD001"]
        norm_ids = ["N1"]
        seeds    = [0]
    else:
        datasets = [args.dataset] if args.dataset else DATASETS
        norm_ids = [args.norm]    if args.norm    else NORM_IDS
        seeds    = SEEDS

    total   = len(datasets) * len(norm_ids) * len(seeds)
    done    = 0
    results = []

    print(f"Device: {device}")
    print(f"Running {total} experiments  "
          f"(datasets={datasets}, norms={norm_ids}, seeds={seeds})\n")

    for ds in datasets:
        for nid in norm_ids:
            for s in seeds:
                try:
                    r = run_one(ds, nid, s, verbose=True)
                    results.append(r)
                except Exception as exc:
                    import traceback
                    print(f"  ERROR [{ds} {nid} seed={s}]: {exc}")
                    traceback.print_exc()
                done += 1
                print(f"  Progress: {done}/{total}")

    if results:
        summary = pd.DataFrame(results)
        summary_path = RESULTS_DIR / "run_log.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nRun log saved to {summary_path}")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
