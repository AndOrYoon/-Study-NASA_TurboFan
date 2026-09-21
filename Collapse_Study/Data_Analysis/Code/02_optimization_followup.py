# -*- coding: utf-8 -*-
"""
02_optimization_followup.py
Phase 1B — Optimization Follow-up (Sequential Design)

Phase 1A 결과 근거:
  - A1+B2+C2 (fixed split + no warmup + clip125) 조건 고정 → MPC rate=0.80
  - 이 환경에서 patience·LR·bias_init·loss의 기여를 분리

실험 구조 (sequential: 주효과 우선 → 유의한 것만 상호작용 확인):
  Stage 1 — 주효과 선별 (4 요인 × 각 3–4 수준, one-at-a-time 기준 고정값 포함)
    D — patience:       5 / 10 / 15(기준) / 30
    E — learning rate:  1e-4(0.1x) / 1e-3(기준) / 1e-2(10x)
    F — bias init:      zero(기준) / train_mean / random_calibrated
    G — loss function:  MSE(기준) / MAE / auxiliary(MSE+0.1×branch)

  Stage 2 — 상호작용 확인 (Stage 1 유의 요인 조합, Phase 1B 완료 후 결과 보고 판단)

고정 조건: A1(fixed split seed=42) + B2(no warmup) + C2(clip=125)
Seeds: 10 per condition (0–9)
MAX_EPOCHS: 200, PATIENCE: Stage 1에서 D 요인이 직접 변경

결과 저장: Collapse_Study/Data_Analysis/Results/Phase1B/
"""

import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

_SCRIPT  = Path(__file__).resolve()
_CS_ROOT = _SCRIPT.parents[2]
_BMAD    = _CS_ROOT.parent
_H6_P2   = _BMAD / "Data_Analysis" / "Code" / "H6_fault_mode" / "phase2_models"
sys.path.insert(0, str(_H6_P2))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from h6_p2_model_utils import (
    set_seed, RUL_CLIP, WINDOW_SIZE,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    make_loader, LSTMBranch,
    predict_sequences, compute_metrics,
)

PHASE1B_DIR = _CS_ROOT / "Data_Analysis" / "Results" / "Phase1B"
LOG_DIR     = PHASE1B_DIR / "epoch_logs"
FIG_DIR     = PHASE1B_DIR / "figures"
for d in [PHASE1B_DIR, LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 고정 조건 (Phase 1A A1+B2+C2)
# ---------------------------------------------------------------------------
DATASET    = "FD003"
CLIP_VAL   = 125
VAL_SEED   = 42       # A1: fixed split
MIN_EPOCHS = 0        # B2: no warmup
MAX_EPOCHS = 200
BATCH_SIZE = 256
SEEDS      = list(range(10))
EPS        = 1e-8

# ---------------------------------------------------------------------------
# Stage 1 실험 조건 정의
# ---------------------------------------------------------------------------
# 기준값: patience=15, lr=1e-3, bias=zero, loss=MSE
# 한 번에 하나씩 변경 (OAT: one-at-a-time screening)

STAGE1_CONDITIONS = []

# D — patience (LR=1e-3, bias=zero, loss=MSE 고정)
for p in [5, 10, 15, 30]:
    STAGE1_CONDITIONS.append({
        "stage": 1, "factor": "D_patience",
        "patience": p, "lr": 1e-3, "bias_init": "zero", "loss": "MSE",
        "label": f"D_patience{p}",
    })

# E — learning rate (patience=15, bias=zero, loss=MSE 고정)
for lr_key, lr_val in [("E_lr1e4", 1e-4), ("E_lr1e3", 1e-3), ("E_lr1e2", 1e-2)]:
    STAGE1_CONDITIONS.append({
        "stage": 1, "factor": "E_lr",
        "patience": 15, "lr": lr_val, "bias_init": "zero", "loss": "MSE",
        "label": lr_key,
    })

# F — bias init (patience=15, lr=1e-3, loss=MSE 고정)
for bias in ["zero", "train_mean", "random_calibrated"]:
    STAGE1_CONDITIONS.append({
        "stage": 1, "factor": "F_bias_init",
        "patience": 15, "lr": 1e-3, "bias_init": bias, "loss": "MSE",
        "label": f"F_bias_{bias}",
    })

# G — loss function (patience=15, lr=1e-3, bias=zero 고정)
for loss_name in ["MSE", "MAE", "auxiliary"]:
    STAGE1_CONDITIONS.append({
        "stage": 1, "factor": "G_loss",
        "patience": 15, "lr": 1e-3, "bias_init": "zero", "loss": loss_name,
        "label": f"G_loss_{loss_name}",
    })

# 중복 기준 조건 제거 (patience=15, lr=1e-3, bias=zero, loss=MSE)
seen = set()
unique_conds = []
for c in STAGE1_CONDITIONS:
    key = c["label"]
    if key not in seen:
        seen.add(key)
        unique_conds.append(c)
STAGE1_CONDITIONS = unique_conds

print(f"Phase 1B Stage 1 조건 수: {len(STAGE1_CONDITIONS)}")
print(f"Seeds per condition: {len(SEEDS)}")
print(f"총 runs: {len(STAGE1_CONDITIONS) * len(SEEDS)}")

# ---------------------------------------------------------------------------
# MPC 지표
# ---------------------------------------------------------------------------

def mpc_metrics(pred, true, c_train):
    pdr = float(np.std(pred) / (np.std(true) + EPS))
    rmse_m = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmse_c = float(np.sqrt(np.mean((true - c_train) ** 2)))
    cbr = rmse_m / (rmse_c + EPS)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + EPS)
    return {"PDR": round(pdr,6), "CBR": round(cbr,6),
            "R2": round(r2,6), "RMSE": round(rmse_m,4)}

def is_mpc(m):
    return m["PDR"] < 0.05 and m["R2"] <= 0.0

# ---------------------------------------------------------------------------
# 데이터 (한 번만 로드)
# ---------------------------------------------------------------------------

def load_data():
    train_df    = load_cmapss(DATASET, "train")
    sensor_cols = get_sensor_cols(DATASET)
    train_df    = add_rul(train_df, rul_clip=CLIP_VAL)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n     = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)
    c_train     = float(train_df["RUL"].mean())

    test_df  = load_cmapss(DATASET, "test")
    test_rul = load_test_rul(DATASET)
    test_n   = apply_normalization(test_df, sensor_cols, min_v, max_v)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, test_rul)
    y_te = np.clip(y_te, 0, CLIP_VAL)

    return X, y, units, sensor_cols, c_train, X_te, y_te

# ---------------------------------------------------------------------------
# 보조 loss (auxiliary branch proxy: 단일 LSTM에 보조 head 추가)
# ---------------------------------------------------------------------------

class LSTMWithAuxHead(nn.Module):
    """단일 LSTM backbone + main head + auxiliary head (이른 timestep 예측)."""
    def __init__(self, n_features, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head_main = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(), nn.Linear(32,1))
        self.head_aux  = nn.Sequential(nn.Linear(hidden,16), nn.ReLU(), nn.Linear(16,1))

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = self.drop2(out)
        last   = out[:, -1, :]        # last timestep → main
        mid    = out[:, out.size(1)//2, :]  # mid timestep → aux
        return self.head_main(last), self.head_aux(mid)

    def predict(self, x):
        main, _ = self.forward(x)
        return main

# ---------------------------------------------------------------------------
# 출력 bias 초기화
# ---------------------------------------------------------------------------

def init_output_bias(model, bias_mode: str, c_train: float):
    if bias_mode == "zero":
        pass  # PyTorch 기본값 0
    elif bias_mode == "train_mean":
        with torch.no_grad():
            if hasattr(model, "head_main"):
                model.head_main[-1].bias.fill_(c_train)
            else:
                model.fc[-1].bias.fill_(c_train)
    elif bias_mode == "random_calibrated":
        with torch.no_grad():
            if hasattr(model, "head_main"):
                dev = model.head_main[-1].bias.device
                model.head_main[-1].bias.data = (torch.tensor([c_train]) + torch.randn(1) * 5.0).to(dev)
            else:
                dev = model.fc[-1].bias.device
                model.fc[-1].bias.data = (torch.tensor([c_train]) + torch.randn(1) * 5.0).to(dev)

# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train_with_log(model, tr_loader, va_loader, X_val, y_val, c_train,
                   device, max_epochs, patience, lr, loss_name):
    if loss_name == "auxiliary":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
    elif loss_name == "MAE":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.L1Loss()
    else:  # MSE
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

    best_val = float("inf")
    patience_cnt = 0
    best_sd  = None
    epoch_log = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_total, tr_n = 0.0, 0
        for batch in tr_loader:
            xb, yb = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            if loss_name == "auxiliary":
                main_pred, aux_pred = model(xb)
                loss = criterion(main_pred, yb) + 0.1 * criterion(aux_pred, yb)
            else:
                loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_total += loss.item() * len(xb)
            tr_n     += len(xb)
        tr_loss = tr_total / tr_n

        model.eval()
        va_total, va_n = 0.0, 0
        with torch.no_grad():
            for batch in va_loader:
                xb, yb = batch[0].to(device), batch[1].to(device)
                if loss_name == "auxiliary":
                    main_pred, aux_pred = model(xb)
                    va_loss_b = criterion(main_pred, yb) + 0.1 * criterion(aux_pred, yb)
                else:
                    va_loss_b = criterion(model(xb), yb)
                va_total += va_loss_b.item() * len(xb)
                va_n     += len(xb)
        va_loss = va_total / va_n

        # val MPC 지표
        if loss_name == "auxiliary":
            val_preds = predict_sequences(model, X_val, device,
                                          use_main_head=True)
        else:
            val_preds = predict_sequences(model, X_val, device)
        val_m = mpc_metrics(val_preds, y_val, c_train)

        is_best = False
        stopped = False
        if va_loss < best_val:
            best_val = va_loss
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            is_best = True
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                stopped = True

        epoch_log.append({
            "epoch": epoch, "tr_loss": round(tr_loss,4),
            "va_loss": round(va_loss,4),
            "va_PDR": val_m["PDR"], "va_R2": val_m["R2"],
            "va_CBR": val_m["CBR"], "va_RMSE": val_m["RMSE"],
            "is_best": is_best, "stopped": stopped,
        })
        if stopped:
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model, epoch_log


# auxiliary head 모델용 predict 래퍼
@torch.no_grad()
def predict_sequences(model, X_np, device, batch_size=512, use_main_head=False):
    model.eval()
    from h6_p2_model_utils import SeqDataset
    from torch.utils.data import DataLoader
    ds = SeqDataset(X_np, np.zeros(len(X_np), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        if use_main_head:
            out, _ = model(xb)
        else:
            out = model(xb)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds).flatten()

# ---------------------------------------------------------------------------
# 단일 run
# ---------------------------------------------------------------------------

def run_one(cond: dict, seed: int, device,
            X, y, units, sensor_cols, c_train, X_te, y_te):
    set_seed(seed)

    # A1: fixed val split seed=42
    tr_m, va_m = split_engines(units, val_frac=0.2, seed=VAL_SEED)
    X_tr, y_tr = X[tr_m], y[tr_m]
    X_va, y_va = X[va_m], y[va_m]

    tr_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X_va, y_va, BATCH_SIZE, shuffle=False)

    loss_name = cond["loss"]
    if loss_name == "auxiliary":
        model = LSTMWithAuxHead(len(sensor_cols)).to(device)
    else:
        model = LSTMBranch(len(sensor_cols)).to(device)

    init_output_bias(model, cond["bias_init"], c_train)

    t0 = time.time()
    model, epoch_log = train_with_log(
        model, tr_loader, va_loader, X_va, y_va, c_train,
        device, MAX_EPOCHS, cond["patience"], cond["lr"], loss_name
    )
    elapsed = time.time() - t0

    stop_ep = next((r["epoch"] for r in reversed(epoch_log) if r["stopped"]), len(epoch_log))

    use_main = (loss_name == "auxiliary")
    preds = predict_sequences(model, X_te, device, use_main_head=use_main)
    preds = np.clip(preds, 0, CLIP_VAL)
    rmse, ns = compute_metrics(preds, y_te)
    m_test = mpc_metrics(preds, y_te, c_train)

    val_preds = predict_sequences(model, X_va, device, use_main_head=use_main)
    m_val = mpc_metrics(val_preds, y_va, c_train)

    return {
        "label": cond["label"], "stage": cond["stage"], "factor": cond["factor"],
        "patience": cond["patience"], "lr": cond["lr"],
        "bias_init": cond["bias_init"], "loss": cond["loss"],
        "seed": seed, "stop_epoch": stop_ep,
        "RMSE": round(rmse,4), "NASA": round(ns,2),
        "test_PDR": m_test["PDR"], "test_R2": m_test["R2"],
        "test_CBR": m_test["CBR"],
        "val_PDR": m_val["PDR"], "val_R2": m_val["R2"],
        "elapsed_s": round(elapsed,1),
        "is_collapsed": is_mpc(m_test),
    }, epoch_log

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"고정 조건: A1(fixed seed=42) + B2(no warmup) + C2(clip=125)\n")

    print("데이터 로딩 중...")
    X, y, units, sensor_cols, c_train, X_te, y_te = load_data()
    print(f"  train RUL mean={c_train:.2f}  n_seq={len(X)}  n_test={len(X_te)}\n")

    out_csv = PHASE1B_DIR / "runs.csv"
    if out_csv.exists():
        df_done = pd.read_csv(out_csv)
        done_keys = set(zip(df_done["label"], df_done["seed"]))
        all_rows  = df_done.to_dict("records")
        print(f"재시작: 기존 {len(df_done)}개 run 로드됨")
    else:
        done_keys = set()
        all_rows  = []

    total_runs = len(STAGE1_CONDITIONS) * len(SEEDS)
    completed  = len(done_keys)
    print(f"실험 시작: {total_runs - completed}개 runs 남음 (총 {total_runs})\n")

    t_global = time.time()

    for cond in STAGE1_CONDITIONS:
        label = cond["label"]
        cond_rows = []

        for seed in SEEDS:
            key = (label, seed)
            if key in done_keys:
                continue

            run_id = completed + 1
            print(f"  [{run_id:>3}/{total_runs}] {label:30s} seed={seed}", end=" ... ", flush=True)

            try:
                row, epoch_log = run_one(cond, seed, device,
                                         X, y, units, sensor_cols, c_train, X_te, y_te)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            log_name = f"{label}_seed{seed}.csv"
            pd.DataFrame(epoch_log).to_csv(LOG_DIR / log_name, index=False)

            all_rows.append(row)
            cond_rows.append(row)
            done_keys.add(key)
            completed += 1

            tag = "COLLAPSED" if row["is_collapsed"] else "ok"
            print(f"ep={row['stop_epoch']:>3}  RMSE={row['RMSE']:.2f}"
                  f"  PDR={row['test_PDR']:.4f}  R²={row['test_R2']:.3f}"
                  f"  [{tag}]  ({row['elapsed_s']:.1f}s)")

        if cond_rows:
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)

        cond_done = [r for r in all_rows if r["label"] == label]
        if cond_done:
            df_c = pd.DataFrame(cond_done)
            n_col = int(df_c["is_collapsed"].sum())
            print(f"  └─ {label}: MPC {n_col}/{len(df_c)}  "
                  f"RMSE={df_c['RMSE'].mean():.2f}±{df_c['RMSE'].std():.2f}\n")

    # ---------------------------------------------------------------------------
    # 집계
    # ---------------------------------------------------------------------------
    df = pd.read_csv(out_csv)

    print("\n" + "=" * 65)
    print("Phase 1B 집계 결과")
    print("=" * 65)

    summary = (
        df.groupby(["factor","label"])
        .agg(
            n_runs=("seed","count"),
            mpc_rate=("is_collapsed","mean"),
            rmse_mean=("RMSE","mean"),
            rmse_std=("RMSE","std"),
            pdr_mean=("test_PDR","mean"),
            r2_mean=("test_R2","mean"),
            stop_ep=("stop_epoch","mean"),
            patience=("patience","first"),
            lr=("lr","first"),
            bias_init=("bias_init","first"),
            loss=("loss","first"),
        )
        .reset_index()
    )
    summary_csv = PHASE1B_DIR / "mpc_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\n[저장] {summary_csv.relative_to(_CS_ROOT)}")

    # 요인별 출력
    for factor in ["D_patience","E_lr","F_bias_init","G_loss"]:
        sub = summary[summary["factor"]==factor].sort_values("label")
        print(f"\n--- {factor} ---")
        print(sub[["label","mpc_rate","rmse_mean","rmse_std","pdr_mean","stop_ep"]].to_string(index=False))

    # 기준 조건 (D_patience15 = B2 기준) 대비 MPC 감소 요인
    baseline = summary[summary["label"]=="D_patience15"]
    if len(baseline):
        base_mpc = float(baseline["mpc_rate"].values[0])
        print(f"\n기준 MPC rate (patience=15, lr=1e-3, bias=zero, loss=MSE): {base_mpc:.2f}")
        improved = summary[summary["mpc_rate"] < base_mpc].sort_values("mpc_rate")
        print("MPC rate 개선 조건:")
        if len(improved):
            print(improved[["label","mpc_rate","rmse_mean","rmse_std"]].to_string(index=False))
        else:
            print("  없음 (모든 조건이 기준 이상)")

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

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    factor_info = [
        ("D_patience", "patience", "D — Patience"),
        ("E_lr",       "lr",       "E — Learning Rate"),
        ("F_bias_init","bias_init","F — Bias Init"),
        ("G_loss",     "loss",     "G — Loss Function"),
    ]

    for ax, (factor, param_col, title) in zip(axes.flat, factor_info):
        sub = summary[summary["factor"]==factor].sort_values("label")
        x   = range(len(sub))
        mpc = sub["mpc_rate"].values
        rmse_m = sub["rmse_mean"].values
        rmse_s = sub["rmse_std"].values

        color = ["#d62728" if r > 0.5 else "#ff7f0e" if r > 0.1 else "#2ca02c" for r in mpc]
        ax2 = ax.twinx()
        ax2.bar(x, rmse_m, yerr=rmse_s, alpha=0.25, color="steelblue",
                capsize=4, label="RMSE (right)")
        ax.plot(x, mpc, "o-", color="crimson", linewidth=2, markersize=7, label="MPC rate")
        ax.set_xticks(x)
        labels = sub[param_col].astype(str).values
        ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("MPC rate", color="crimson")
        ax2.set_ylabel("RMSE", color="steelblue")
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, label="baseline MPC=0.80")
        ax.set_title(title, fontsize=11)

    fig.suptitle("Phase 1B: MPC Rate & RMSE by Optimization Factor\n"
                 "(Fixed: A1+B2+C2, 10 seeds/condition)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_phase1b_factors.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[시각화] fig1_phase1b_factors.png 저장")

    # per-seed scatter: 각 조건의 RMSE 분포
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ax, (factor, param_col, title) in zip(axes, factor_info):
        sub_df = df[df["factor"]==factor]
        labels_ordered = summary[summary["factor"]==factor].sort_values("label")["label"].tolist()
        for xi, lbl in enumerate(labels_ordered):
            rows = sub_df[sub_df["label"]==lbl]
            col_pts = ["#d62728" if c else "#2ca02c" for c in rows["is_collapsed"]]
            ax.scatter([xi]*len(rows), rows["RMSE"], c=col_pts, alpha=0.7, s=30, zorder=5)
        ax.set_xticks(range(len(labels_ordered)))
        ax.set_xticklabels([l.split("_")[-1] for l in labels_ordered], fontsize=8, rotation=20, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Test RMSE" if ax==axes[0] else "")
    fig.suptitle("Phase 1B: Per-seed RMSE (red=collapsed, green=normal)", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_phase1b_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[시각화] fig2_phase1b_scatter.png 저장")


if __name__ == "__main__":
    main()
