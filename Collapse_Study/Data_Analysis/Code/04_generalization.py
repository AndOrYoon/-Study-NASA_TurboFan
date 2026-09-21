# -*- coding: utf-8 -*-
"""
04_generalization.py
Phase 3 — Dataset × Architecture Generalization

목적: LSTM·FD003 특이성 배제 — MPC가 다른 dataset·architecture에서도 재현되는지 확인

설계:
  - Dataset:      FD001 / FD002 / FD003 / FD004
  - Architecture: LSTM / GRU / MLP / CNN1D
  - Protocol:     A1(fixed split seed=42) + B2(no warmup) + C2(clip=125) = 붕괴 유발 조건
  - Seeds:        10 (0–9)
  - 총 runs:      4 × 4 × 10 = 160

FD002/FD004: K-means 운전조건 잔차화 적용 (op_condition_utils.py)
MPC 판정: PDR < 0.05 AND R² ≤ 0 (Phase 1A/1B 동일 기준)

결과 저장: Collapse_Study/Data_Analysis/Results/Phase3/
"""

import sys, time
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

_SCRIPT  = Path(__file__).resolve()
_CS_ROOT = _SCRIPT.parents[2]
_BMAD    = _CS_ROOT.parent
_H6_P2   = _BMAD / "Data_Analysis" / "Code" / "H6_fault_mode" / "phase2_models"
_SHARED  = _BMAD / "Data_Analysis" / "Code" / "shared"
sys.path.insert(0, str(_H6_P2))
sys.path.insert(0, str(_SHARED))

from h6_p2_model_utils import (
    set_seed, WINDOW_SIZE,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    make_loader, LSTMBranch,
    predict_sequences, compute_metrics,
)
from op_condition_utils import (
    fit_op_condition_kmeans, compute_cluster_means, apply_op_residual,
)

PHASE3_DIR = _CS_ROOT / "Data_Analysis" / "Results" / "Phase3"
LOG_DIR    = PHASE3_DIR / "epoch_logs"
FIG_DIR    = PHASE3_DIR / "figures"
for d in [PHASE3_DIR, LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLIP_VAL   = 125
VAL_SEED   = 42
MIN_EPOCHS = 0
MAX_EPOCHS = 200
PATIENCE   = 15
BATCH_SIZE = 256
LR         = 1e-3
SEEDS      = list(range(10))
EPS        = 1e-8

DATASETS      = ["FD001", "FD002", "FD003", "FD004"]
ARCH_NAMES    = ["LSTM", "GRU", "MLP", "CNN1D"]
OP_DATASETS   = {"FD002", "FD004"}   # K-means 잔차화 필요
OP_COLS       = ("op_setting_1", "op_setting_2", "op_setting_3")  # load_cmapss 컬럼명

# ---------------------------------------------------------------------------
# 아키텍처 정의
# ---------------------------------------------------------------------------

class GRUBranch(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.2):
        super().__init__()
        self.gru1 = nn.GRU(n_features, hidden, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(), nn.Linear(32,1))

    def forward(self, x):
        out, _ = self.gru1(x)
        out    = self.drop1(out)
        out, _ = self.gru2(out)
        out    = self.drop2(out)
        return self.fc(out[:, -1, :])


class MLPBranch(nn.Module):
    def __init__(self, n_features, window=WINDOW_SIZE, dropout=0.2):
        super().__init__()
        input_dim = window * n_features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 32),        nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.fc(x)


class CNN1DBranch(nn.Module):
    def __init__(self, n_features, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),          nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, window, features) → (batch, features, window)
        out = self.conv(x.permute(0, 2, 1))
        out = self.pool(out)
        return self.fc(out)


def build_model(arch: str, n_features: int) -> nn.Module:
    if arch == "LSTM":
        return LSTMBranch(n_features)
    elif arch == "GRU":
        return GRUBranch(n_features)
    elif arch == "MLP":
        return MLPBranch(n_features)
    elif arch == "CNN1D":
        return CNN1DBranch(n_features)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# MPC 지표
# ---------------------------------------------------------------------------

def mpc_metrics(pred, true, c_train):
    pdr   = float(np.std(pred) / (np.std(true) + EPS))
    rmse  = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmse_c = float(np.sqrt(np.mean((true - c_train) ** 2)))
    cbr   = rmse / (rmse_c + EPS)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2    = 1.0 - ss_res / (ss_tot + EPS)
    return {"PDR": round(pdr,6), "CBR": round(cbr,6), "R2": round(r2,6), "RMSE": round(rmse,4)}

def is_mpc(m):
    return m["PDR"] < 0.05 and m["R2"] <= 0.0

# ---------------------------------------------------------------------------
# 데이터 로드 (dataset별 1회)
# ---------------------------------------------------------------------------

def load_dataset(dataset: str):
    train_df    = load_cmapss(dataset, "train")
    sensor_cols = get_sensor_cols(dataset)

    if dataset in OP_DATASETS:
        scaler_km, km = fit_op_condition_kmeans(train_df, op_cols=OP_COLS)
        cluster_means = compute_cluster_means(train_df, scaler_km, km, sensor_cols, op_cols=OP_COLS)
        train_df = apply_op_residual(train_df, scaler_km, km, cluster_means, sensor_cols, op_cols=OP_COLS)

    train_df = add_rul(train_df, rul_clip=CLIP_VAL)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)
    c_train = float(train_df["RUL"].mean())

    test_df  = load_cmapss(dataset, "test")
    test_rul = load_test_rul(dataset)
    if dataset in OP_DATASETS:
        test_df = apply_op_residual(test_df, scaler_km, km, cluster_means, sensor_cols, op_cols=OP_COLS)
    test_n   = apply_normalization(test_df, sensor_cols, min_v, max_v)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, test_rul)
    y_te = np.clip(y_te, 0, CLIP_VAL)

    return X, y, units, sensor_cols, c_train, X_te, y_te

# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train_with_log(model, tr_loader, va_loader, X_val, y_val, c_train, device):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val, patience_cnt, best_sd = float("inf"), 0, None
    epoch_log = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_total, tr_n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_total += loss.item() * len(xb)
            tr_n     += len(xb)

        model.eval()
        va_total, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_total += criterion(model(xb), yb).item() * len(xb)
                va_n     += len(xb)
        va_loss = va_total / va_n

        val_preds = predict_sequences(model, X_val, device)
        val_m     = mpc_metrics(val_preds, y_val, c_train)

        stopped = False
        if va_loss < best_val:
            best_val = va_loss
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                stopped = True

        epoch_log.append({
            "epoch": epoch, "tr_loss": round(tr_total/tr_n, 4),
            "va_loss": round(va_loss, 4),
            "va_PDR": val_m["PDR"], "va_R2": val_m["R2"],
            "va_CBR": val_m["CBR"], "va_RMSE": val_m["RMSE"],
            "stopped": stopped,
        })
        if stopped:
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model, epoch_log

# ---------------------------------------------------------------------------
# 단일 run
# ---------------------------------------------------------------------------

def run_one(dataset, arch, seed, device, X, y, units, sensor_cols, c_train, X_te, y_te):
    set_seed(seed)

    tr_m, va_m = split_engines(units, val_frac=0.2, seed=VAL_SEED)
    X_tr, y_tr = X[tr_m], y[tr_m]
    X_va, y_va = X[va_m], y[va_m]

    tr_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X_va, y_va, BATCH_SIZE, shuffle=False)

    model = build_model(arch, len(sensor_cols)).to(device)

    t0 = time.time()
    model, epoch_log = train_with_log(
        model, tr_loader, va_loader, X_va, y_va, c_train, device
    )
    elapsed = time.time() - t0

    stop_ep = next((r["epoch"] for r in reversed(epoch_log) if r["stopped"]), len(epoch_log))

    preds = np.clip(predict_sequences(model, X_te, device), 0, CLIP_VAL)
    rmse, ns = compute_metrics(preds, y_te)
    m_test = mpc_metrics(preds, y_te, c_train)

    return {
        "dataset": dataset, "arch": arch, "seed": seed,
        "n_params": count_params(model),
        "stop_epoch": stop_ep,
        "RMSE": round(rmse, 4), "NASA": round(ns, 2),
        "test_PDR": m_test["PDR"], "test_R2": m_test["R2"], "test_CBR": m_test["CBR"],
        "elapsed_s": round(elapsed, 1),
        "is_collapsed": is_mpc(m_test),
    }, epoch_log

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Protocol: A1(fixed split={VAL_SEED}) + B2(no warmup) + C2(clip={CLIP_VAL})")
    print(f"총 조건: {len(DATASETS)} datasets × {len(ARCH_NAMES)} architectures = "
          f"{len(DATASETS)*len(ARCH_NAMES)} conditions × {len(SEEDS)} seeds = "
          f"{len(DATASETS)*len(ARCH_NAMES)*len(SEEDS)} runs\n")

    out_csv = PHASE3_DIR / "runs.csv"
    if out_csv.exists():
        df_done  = pd.read_csv(out_csv)
        done_keys = set(zip(df_done["dataset"], df_done["arch"], df_done["seed"]))
        all_rows  = df_done.to_dict("records")
        print(f"재시작: 기존 {len(df_done)}개 run 로드됨")
    else:
        done_keys, all_rows = set(), []

    total_runs = len(DATASETS) * len(ARCH_NAMES) * len(SEEDS)
    completed  = len(done_keys)
    print(f"실험 시작: {total_runs - completed}개 runs 남음 (총 {total_runs})\n")
    t_global = time.time()

    for dataset in DATASETS:
        print(f"{'='*60}")
        print(f"데이터 로딩: {dataset} ...")
        X, y, units, sensor_cols, c_train, X_te, y_te = load_dataset(dataset)
        print(f"  n_features={len(sensor_cols)}  n_seq={len(X)}  "
              f"n_test={len(X_te)}  c_train={c_train:.2f}")

        for arch in ARCH_NAMES:
            cond_rows = []
            for seed in SEEDS:
                key = (dataset, arch, seed)
                if key in done_keys:
                    continue

                completed += 1
                print(f"  [{completed:>3}/{total_runs}] {dataset} {arch:6s} seed={seed}",
                      end=" ... ", flush=True)

                try:
                    row, epoch_log = run_one(
                        dataset, arch, seed, device,
                        X, y, units, sensor_cols, c_train, X_te, y_te
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    completed -= 1
                    continue

                log_name = f"{dataset}_{arch}_seed{seed}.csv"
                pd.DataFrame(epoch_log).to_csv(LOG_DIR / log_name, index=False)

                all_rows.append(row)
                cond_rows.append(row)
                done_keys.add(key)

                tag = "COLLAPSED" if row["is_collapsed"] else "ok"
                print(f"ep={row['stop_epoch']:>3}  RMSE={row['RMSE']:.2f}"
                      f"  PDR={row['test_PDR']:.4f}  R²={row['test_R2']:.3f}"
                      f"  [{tag}]  ({row['elapsed_s']:.1f}s)")

            if cond_rows:
                pd.DataFrame(all_rows).to_csv(out_csv, index=False)

            all_cond = [r for r in all_rows if r["dataset"]==dataset and r["arch"]==arch]
            if all_cond:
                df_c = pd.DataFrame(all_cond)
                n_col = int(df_c["is_collapsed"].sum())
                print(f"  └─ {dataset}/{arch}: MPC {n_col}/{len(df_c)}"
                      f"  RMSE={df_c['RMSE'].mean():.2f}±{df_c['RMSE'].std():.2f}\n")

    # ---------------------------------------------------------------------------
    # 집계
    # ---------------------------------------------------------------------------
    df = pd.read_csv(out_csv)

    summary = (
        df.groupby(["dataset", "arch"])
        .agg(
            n_runs=("seed", "count"),
            mpc_rate=("is_collapsed", "mean"),
            rmse_mean=("RMSE", "mean"),
            rmse_std=("RMSE", "std"),
            pdr_mean=("test_PDR", "mean"),
            r2_mean=("test_R2", "mean"),
            stop_ep=("stop_epoch", "mean"),
            n_params=("n_params", "first"),
        )
        .reset_index()
    )
    summary.to_csv(PHASE3_DIR / "mpc_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("Phase 3 집계 결과 — MPC rate (dataset × architecture)")
    print("=" * 70)

    pivot = summary.pivot(index="arch", columns="dataset", values="mpc_rate")
    print(pivot.to_string())

    print("\n--- RMSE mean (붕괴 run 포함) ---")
    pivot_rmse = summary.pivot(index="arch", columns="dataset", values="rmse_mean")
    print(pivot_rmse.round(2).to_string())

    # go/no-go 판정
    print("\n--- go/no-go 판정 ---")
    n_ds_with_mpc  = (summary.groupby("dataset")["mpc_rate"].max() > 0).sum()
    n_arch_with_mpc = (summary.groupby("arch")["mpc_rate"].max() > 0).sum()
    print(f"MPC 발생 dataset 수: {n_ds_with_mpc} / {len(DATASETS)}")
    print(f"MPC 발생 arch 수:    {n_arch_with_mpc} / {len(ARCH_NAMES)}")
    if n_ds_with_mpc >= 2 and n_arch_with_mpc >= 2:
        print("→ GO: 2+ dataset, 2+ architecture → Full journal paper 범위")
    elif n_arch_with_mpc < 2:
        print("→ HOLD: LSTM 한정 → architecture-specific failure로 범위 축소 검토")
    elif n_ds_with_mpc < 2:
        print("→ HOLD: FD003 한정 → benchmark case study로 전환 검토")

    print(f"\n총 실험 시간: {(time.time()-t_global)/60:.1f}분")

    try:
        generate_figures(df, summary)
    except Exception as e:
        print(f"\n[경고] 시각화 실패: {e}")


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def generate_figures(df, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # ---- Fig 1: Heatmap — MPC rate (dataset × architecture) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, val_col, title, fmt, cmap in [
        (axes[0], "mpc_rate",  "MPC Rate",    ".2f", "Reds"),
        (axes[1], "rmse_mean", "RMSE (mean)", ".1f", "Blues"),
    ]:
        pivot = summary.pivot(index="arch", columns="dataset", values=val_col)
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                       vmin=0, vmax=1.0 if "rate" in val_col else None)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=9, color="white" if val > 0.6 else "black")
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=11)

    fig.suptitle(f"Phase 3: MPC Rate & RMSE\n"
                 f"(Protocol: A1+B2+C2, clip={CLIP_VAL}, 10 seeds)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_mpc_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[시각화] fig1_mpc_heatmap.png")

    # ---- Fig 2: Per-seed RMSE scatter (dataset별 패널) ----
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 5), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        for xi, arch in enumerate(ARCH_NAMES):
            rows = sub[sub["arch"] == arch]
            colors = ["#d62728" if c else "#2ca02c" for c in rows["is_collapsed"]]
            ax.scatter([xi] * len(rows), rows["RMSE"], c=colors, alpha=0.7, s=40, zorder=5)
        ax.set_xticks(range(len(ARCH_NAMES)))
        ax.set_xticklabels(ARCH_NAMES, fontsize=9, rotation=15, ha="right")
        ax.set_title(ds, fontsize=11)
        ax.set_ylabel("Test RMSE" if ax == axes[0] else "")
    fig.suptitle("Phase 3: Per-seed RMSE (red=collapsed, green=normal)", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_seed_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[시각화] fig2_seed_scatter.png")

    # ---- Fig 3: MPC rate bar (architecture별 dataset 비교) ----
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(ARCH_NAMES))
    width = 0.2
    colors_ds = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (ds, col) in enumerate(zip(DATASETS, colors_ds)):
        rates = [summary[(summary["dataset"]==ds) & (summary["arch"]==arch)]["mpc_rate"].values
                 for arch in ARCH_NAMES]
        rates = [r[0] if len(r) else 0 for r in rates]
        ax.bar(x + i * width - 1.5 * width, rates, width, label=ds, color=col, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ARCH_NAMES, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("MPC Rate")
    ax.set_title("Phase 3: MPC Rate by Architecture × Dataset", fontsize=11)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, label="Phase 1A/1B baseline (FD003/LSTM)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_mpc_by_arch.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[시각화] fig3_mpc_by_arch.png")


if __name__ == "__main__":
    main()
