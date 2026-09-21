# -*- coding: utf-8 -*-
"""
04b_lstm_gru_mechanism.py
Phase 3B — LSTM vs GRU Mechanism: Why is LSTM uniquely vulnerable?

핵심 가설 (H_mech):
  LSTM의 forget gate가 trivial solution 근방에서 0에 수렴 →
  cell state를 통한 gradient 경로 차단 → early stopping과 결합해 MPC 고착.
  GRU는 별도 cell state가 없어 이 경로가 존재하지 않음.

실험 설계 (FD003 × A1+B2+C2, 10 seeds 각):
  Exp A: Gate Activation 분석
    - LSTM_gate: 커스텀 LSTM (forget gate 값 epoch별 기록)
    - GRU_gate:  커스텀 GRU  (update/reset gate 값 epoch별 기록)
    → 붕괴 run의 forget gate가 epoch 초기에 0으로 수렴하는지 확인

  Exp B: LSTM 변형 실험 (forget gate 조작)
    - V1_fb1: forget bias init = +1 (Jozefowicz et al. 2015)
    - V2_fg1: forget gate 상수 1 고정 (Constant Error Carousel)
    → forget gate 조작만으로 MPC rate가 0.80 → ? 로 변하는지 확인

  참조: V3_base (standard LSTM) = Phase 3 FD003/LSTM 결과 재사용 (MPC 8/10)

결과 저장: Collapse_Study/Data_Analysis/Results/Phase3B/
"""

import sys, time
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_SCRIPT  = Path(__file__).resolve()
_CS_ROOT = _SCRIPT.parents[2]
_BMAD    = _CS_ROOT.parent
_H6_P2   = _BMAD / "Data_Analysis" / "Code" / "H6_fault_mode" / "phase2_models"
sys.path.insert(0, str(_H6_P2))

from h6_p2_model_utils import (
    set_seed, WINDOW_SIZE,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    make_loader, SeqDataset,
    compute_metrics,
)

PHASE3B_DIR = _CS_ROOT / "Data_Analysis" / "Results" / "Phase3B"
LOG_DIR     = PHASE3B_DIR / "epoch_logs"
FIG_DIR     = PHASE3B_DIR / "figures"
for d in [PHASE3B_DIR, LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASET    = "FD003"
CLIP_VAL   = 125
VAL_SEED   = 42
MIN_EPOCHS = 0
MAX_EPOCHS = 200
PATIENCE   = 15
BATCH_SIZE = 256
LR         = 1e-3
SEEDS      = list(range(10))
EPS        = 1e-8
HIDDEN     = 64
DROPOUT    = 0.2

# ---------------------------------------------------------------------------
# 커스텀 셀 — gate activation 직접 접근 가능
# ---------------------------------------------------------------------------

class CustomLSTMCell(nn.Module):
    """LSTM cell that returns forget gate activations."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates    = self.linear(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new, f  # f: (batch, hidden) — forget gate values


class CustomGRUCell(nn.Module):
    """GRU cell that returns update/reset gate activations."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_rz  = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.linear_n_x = nn.Linear(input_size, hidden_size)
        self.linear_n_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        rz   = self.linear_rz(torch.cat([x, h], dim=1))
        r, z = rz.chunk(2, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(self.linear_n_x(x) + r * self.linear_n_h(h))
        h_new = (1 - z) * n + z * h
        return h_new, z, r  # z: update gate, r: reset gate


class LSTMFg1Cell(nn.Module):
    """LSTM with forget gate fixed to 1.0 (Constant Error Carousel)."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size)  # i, g, o

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates    = self.linear(combined)
        i, g, o  = gates.chunk(3, dim=1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = c + i * g          # f=1 → c_new = 1·c + i·g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

# ---------------------------------------------------------------------------
# 스택 모델 (2-layer + FC)
# ---------------------------------------------------------------------------

class LSTMGateModel(nn.Module):
    """Custom stacked LSTM — logs forget gate per timestep."""
    def __init__(self, n_features):
        super().__init__()
        self.cell1 = CustomLSTMCell(n_features, HIDDEN)
        self.drop1 = nn.Dropout(DROPOUT)
        self.cell2 = CustomLSTMCell(HIDDEN, HIDDEN)
        self.drop2 = nn.Dropout(DROPOUT)
        self.fc    = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, return_gates=False):
        B, T, F = x.shape
        h1 = torch.zeros(B, HIDDEN, device=x.device)
        c1 = torch.zeros(B, HIDDEN, device=x.device)
        h2 = torch.zeros(B, HIDDEN, device=x.device)
        c2 = torch.zeros(B, HIDDEN, device=x.device)

        fg1_list = []
        for t in range(T):
            h1, c1, fg1 = self.cell1(x[:, t, :], h1, c1)
            h1 = self.drop1(h1)
            h2, c2, _   = self.cell2(h1, h2, c2)

            if return_gates:
                fg1_list.append(fg1.mean().item())

        out = self.drop2(h2)
        pred = self.fc(out)

        if return_gates:
            return pred, np.mean(fg1_list)
        return pred


class GRUGateModel(nn.Module):
    """Custom stacked GRU — logs update/reset gate per timestep."""
    def __init__(self, n_features):
        super().__init__()
        self.cell1 = CustomGRUCell(n_features, HIDDEN)
        self.drop1 = nn.Dropout(DROPOUT)
        self.cell2 = CustomGRUCell(HIDDEN, HIDDEN)
        self.drop2 = nn.Dropout(DROPOUT)
        self.fc    = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, return_gates=False):
        B, T, F = x.shape
        h1 = torch.zeros(B, HIDDEN, device=x.device)
        h2 = torch.zeros(B, HIDDEN, device=x.device)

        ug1_list, rg1_list = [], []
        for t in range(T):
            h1, ug1, rg1 = self.cell1(x[:, t, :], h1)
            h1 = self.drop1(h1)
            h2, _, _     = self.cell2(h1, h2)

            if return_gates:
                ug1_list.append(ug1.mean().item())
                rg1_list.append(rg1.mean().item())

        out = self.drop2(h2)
        pred = self.fc(out)

        if return_gates:
            return pred, np.mean(ug1_list), np.mean(rg1_list)
        return pred


class LSTMFb1Model(nn.Module):
    """Standard LSTM with forget bias initialized to +1."""
    def __init__(self, n_features):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, HIDDEN, batch_first=True)
        self.drop1 = nn.Dropout(DROPOUT)
        self.lstm2 = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
        self.drop2 = nn.Dropout(DROPOUT)
        self.fc    = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))
        self._init_forget_bias()

    def _init_forget_bias(self):
        # PyTorch LSTM bias layout: [input, forget, cell, output] × hidden_size
        # bias_ih_l0 and bias_hh_l0 both have shape (4*hidden,)
        for lstm in [self.lstm1, self.lstm2]:
            with torch.no_grad():
                # forget gate bias = indices [HIDDEN:2*HIDDEN]
                lstm.bias_ih_l0[HIDDEN:2*HIDDEN].fill_(1.0)
                lstm.bias_hh_l0[HIDDEN:2*HIDDEN].fill_(1.0)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = self.drop2(out)
        return self.fc(out[:, -1, :])


class LSTMFg1Model(nn.Module):
    """LSTM with forget gate fixed to 1 (CEC)."""
    def __init__(self, n_features):
        super().__init__()
        self.cell1 = LSTMFg1Cell(n_features, HIDDEN)
        self.drop1 = nn.Dropout(DROPOUT)
        self.cell2 = LSTMFg1Cell(HIDDEN, HIDDEN)
        self.drop2 = nn.Dropout(DROPOUT)
        self.fc    = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        B, T, F = x.shape
        h1 = torch.zeros(B, HIDDEN, device=x.device)
        c1 = torch.zeros(B, HIDDEN, device=x.device)
        h2 = torch.zeros(B, HIDDEN, device=x.device)
        c2 = torch.zeros(B, HIDDEN, device=x.device)
        for t in range(T):
            h1, c1 = self.cell1(x[:, t, :], h1, c1)
            h1 = self.drop1(h1)
            h2, c2 = self.cell2(h1, h2, c2)
        return self.fc(self.drop2(h2))

# ---------------------------------------------------------------------------
# MPC 지표
# ---------------------------------------------------------------------------

def mpc_metrics(pred, true, c_train):
    pdr  = float(np.std(pred) / (np.std(true) + EPS))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2   = 1.0 - ss_res / (ss_tot + EPS)
    return {"PDR": round(pdr,6), "R2": round(r2,6), "RMSE": round(rmse,4)}

def is_mpc(m):
    return m["PDR"] < 0.05 and m["R2"] <= 0.0

# ---------------------------------------------------------------------------
# predict_sequences (모델 유형 분기)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_sequences(model, X_np, device, model_type="standard", batch_size=512):
    model.eval()
    ds     = SeqDataset(X_np, np.zeros(len(X_np), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds  = []
    for xb, _ in loader:
        xb = xb.to(device)
        if model_type == "lstm_gate":
            out, _ = model(xb, return_gates=True)
        elif model_type == "gru_gate":
            out, _, _ = model(xb, return_gates=True)
        else:
            out = model(xb)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds).flatten()

# ---------------------------------------------------------------------------
# 데이터 로드 (1회)
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
# 학습 루프 — gate logging 포함
# ---------------------------------------------------------------------------

def train_with_gate_log(model, tr_loader, va_loader,
                        X_val, y_val, c_train, device, model_type):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val, patience_cnt, best_sd = float("inf"), 0, None
    epoch_log = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # --- train ---
        model.train()
        tr_total, tr_n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if model_type == "lstm_gate":
                pred, _ = model(xb, return_gates=True)
            elif model_type == "gru_gate":
                pred, _, _ = model(xb, return_gates=True)
            else:
                pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_total += loss.item() * len(xb)
            tr_n     += len(xb)

        # --- val loss ---
        model.eval()
        va_total, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                if model_type == "lstm_gate":
                    pred, _ = model(xb, return_gates=True)
                elif model_type == "gru_gate":
                    pred, _, _ = model(xb, return_gates=True)
                else:
                    pred = model(xb)
                va_total += criterion(pred, yb).item() * len(xb)
                va_n     += len(xb)
        va_loss = va_total / va_n

        # --- gate activation (val set 대표 배치) ---
        gate_vals = {}
        model.eval()
        with torch.no_grad():
            xb_sample = torch.tensor(X_val[:min(128, len(X_val))],
                                     dtype=torch.float32, device=device)
            if model_type == "lstm_gate":
                _, fg = model(xb_sample, return_gates=True)
                gate_vals = {"fg_mean": round(fg, 6)}
            elif model_type == "gru_gate":
                _, ug, rg = model(xb_sample, return_gates=True)
                gate_vals = {"ug_mean": round(ug, 6), "rg_mean": round(rg, 6)}

        # --- val MPC 지표 ---
        val_preds = predict_sequences(model, X_val, device, model_type)
        val_m = mpc_metrics(val_preds, y_val, c_train)

        # --- early stopping ---
        stopped = False
        if va_loss < best_val:
            best_val = va_loss
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                stopped = True

        row = {
            "epoch": epoch,
            "tr_loss": round(tr_total / tr_n, 4),
            "va_loss": round(va_loss, 4),
            "va_PDR": val_m["PDR"], "va_R2": val_m["R2"], "va_RMSE": val_m["RMSE"],
            "stopped": stopped,
        }
        row.update(gate_vals)
        epoch_log.append(row)
        if stopped:
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model, epoch_log

# ---------------------------------------------------------------------------
# 단일 run
# ---------------------------------------------------------------------------

CONDITIONS = [
    {"label": "LSTM_gate", "model_type": "lstm_gate"},
    {"label": "GRU_gate",  "model_type": "gru_gate"},
    {"label": "V1_fb1",    "model_type": "standard"},   # LSTM forget bias +1
    {"label": "V2_fg1",    "model_type": "standard"},   # LSTM forget gate = 1
]

def build_model(label, n_features):
    if label == "LSTM_gate":
        return LSTMGateModel(n_features)
    elif label == "GRU_gate":
        return GRUGateModel(n_features)
    elif label == "V1_fb1":
        return LSTMFb1Model(n_features)
    elif label == "V2_fg1":
        return LSTMFg1Model(n_features)
    raise ValueError(label)


def run_one(cond, seed, device, X, y, units, sensor_cols, c_train, X_te, y_te):
    set_seed(seed)
    tr_m, va_m = split_engines(units, val_frac=0.2, seed=VAL_SEED)
    X_tr, y_tr = X[tr_m], y[tr_m]
    X_va, y_va = X[va_m], y[va_m]

    tr_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X_va, y_va, BATCH_SIZE, shuffle=False)

    model = build_model(cond["label"], len(sensor_cols)).to(device)

    t0 = time.time()
    model, epoch_log = train_with_gate_log(
        model, tr_loader, va_loader, X_va, y_va, c_train, device, cond["model_type"]
    )
    elapsed = time.time() - t0

    stop_ep = next((r["epoch"] for r in reversed(epoch_log) if r["stopped"]), len(epoch_log))
    preds   = np.clip(predict_sequences(model, X_te, device, cond["model_type"]), 0, CLIP_VAL)
    rmse, ns = compute_metrics(preds, y_te)
    m_test  = mpc_metrics(preds, y_te, c_train)

    return {
        "label": cond["label"], "seed": seed, "stop_epoch": stop_ep,
        "RMSE": round(rmse, 4), "NASA": round(ns, 2),
        "test_PDR": m_test["PDR"], "test_R2": m_test["R2"],
        "elapsed_s": round(elapsed, 1),
        "is_collapsed": is_mpc(m_test),
    }, epoch_log

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {DATASET}, Protocol: A1+B2+C2 (fixed split={VAL_SEED}, no warmup, clip={CLIP_VAL})")
    total_runs = len(CONDITIONS) * len(SEEDS)
    print(f"총 runs: {len(CONDITIONS)} conditions × {len(SEEDS)} seeds = {total_runs}\n")

    out_csv = PHASE3B_DIR / "runs.csv"
    if out_csv.exists():
        df_done  = pd.read_csv(out_csv)
        done_keys = set(zip(df_done["label"], df_done["seed"]))
        all_rows  = df_done.to_dict("records")
        print(f"재시작: 기존 {len(df_done)}개 run 로드됨")
    else:
        done_keys, all_rows = set(), []

    completed = len(done_keys)
    print("데이터 로딩 중...")
    X, y, units, sensor_cols, c_train, X_te, y_te = load_data()
    print(f"  n_features={len(sensor_cols)}  n_seq={len(X)}  c_train={c_train:.2f}\n")

    t_global = time.time()

    for cond in CONDITIONS:
        label = cond["label"]
        cond_rows = []

        for seed in SEEDS:
            key = (label, seed)
            if key in done_keys:
                continue

            completed += 1
            print(f"  [{completed:>2}/{total_runs}] {label:12s} seed={seed}",
                  end=" ... ", flush=True)

            try:
                row, epoch_log = run_one(cond, seed, device, X, y, units,
                                         sensor_cols, c_train, X_te, y_te)
            except Exception as e:
                print(f"ERROR: {e}")
                completed -= 1
                continue

            pd.DataFrame(epoch_log).to_csv(
                LOG_DIR / f"{label}_seed{seed}.csv", index=False)

            all_rows.append(row)
            cond_rows.append(row)
            done_keys.add(key)

            tag = "COLLAPSED" if row["is_collapsed"] else "ok"
            print(f"ep={row['stop_epoch']:>3}  RMSE={row['RMSE']:.2f}"
                  f"  PDR={row['test_PDR']:.4f}  R²={row['test_R2']:.3f}"
                  f"  [{tag}]  ({row['elapsed_s']:.1f}s)")

        if cond_rows:
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)

        all_cond = [r for r in all_rows if r["label"] == label]
        if all_cond:
            df_c = pd.DataFrame(all_cond)
            n_col = int(df_c["is_collapsed"].sum())
            print(f"  └─ {label}: MPC {n_col}/{len(df_c)}"
                  f"  RMSE={df_c['RMSE'].mean():.2f}±{df_c['RMSE'].std():.2f}\n")

    # -------------------------------------------------------------------------
    # 집계
    # -------------------------------------------------------------------------
    df = pd.read_csv(out_csv)

    print("\n" + "=" * 55)
    print("Phase 3B 집계 결과")
    print("=" * 55)

    summary = (
        df.groupby("label")
        .agg(n_runs=("seed","count"), mpc_rate=("is_collapsed","mean"),
             rmse_mean=("RMSE","mean"), rmse_std=("RMSE","std"),
             pdr_mean=("test_PDR","mean"), r2_mean=("test_R2","mean"),
             stop_ep=("stop_epoch","mean"))
        .reset_index()
    )
    summary.to_csv(PHASE3B_DIR / "mpc_summary.csv", index=False)
    print(summary[["label","mpc_rate","rmse_mean","rmse_std","pdr_mean","stop_ep"]].to_string(index=False))

    # V3_base 참조 추가 (Phase 3에서 FD003/LSTM 결과)
    p3_csv = _CS_ROOT / "Data_Analysis" / "Results" / "Phase3" / "runs.csv"
    if p3_csv.exists():
        p3 = pd.read_csv(p3_csv)
        p3_fd3_lstm = p3[(p3["dataset"]=="FD003") & (p3["arch"]=="LSTM")]
        if len(p3_fd3_lstm):
            v3_mpc = p3_fd3_lstm["is_collapsed"].mean()
            v3_rmse = p3_fd3_lstm["RMSE"].mean()
            print(f"\nV3_base 참조 (Phase 3 FD003/LSTM): MPC {v3_mpc:.2f}  RMSE={v3_rmse:.2f}")

    print(f"\n총 실험 시간: {(time.time()-t_global)/60:.1f}분")

    try:
        generate_figures(df)
    except Exception as e:
        print(f"\n[경고] 시각화 실패: {e}")


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def generate_figures(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Fig 1: MPC rate 비교 막대 + V3_base 참조선 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    labels_order = ["LSTM_gate", "GRU_gate", "V1_fb1", "V2_fg1"]
    label_display = {
        "LSTM_gate": "LSTM\n(gate log)",
        "GRU_gate":  "GRU\n(gate log)",
        "V1_fb1":    "LSTM\nfb=+1",
        "V2_fg1":    "LSTM\nfg=1 (CEC)",
    }

    summary = df.groupby("label").agg(
        mpc_rate=("is_collapsed","mean"),
        rmse_mean=("RMSE","mean"),
        rmse_std=("RMSE","std"),
    ).reindex(labels_order)

    ax = axes[0]
    colors = ["#d62728" if r > 0.5 else "#ff7f0e" if r > 0.1 else "#2ca02c"
              for r in summary["mpc_rate"].fillna(0)]
    bars = ax.bar(range(len(labels_order)),
                  summary["mpc_rate"].fillna(0), color=colors, alpha=0.8)
    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1,
               label="V3_base (standard LSTM) = 0.80")
    ax.set_xticks(range(len(labels_order)))
    ax.set_xticklabels([label_display[l] for l in labels_order], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("MPC Rate")
    ax.set_title("Phase 3B: MPC Rate by Model Variant", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(range(len(labels_order)),
           summary["rmse_mean"].fillna(0),
           yerr=summary["rmse_std"].fillna(0),
           color="steelblue", alpha=0.8, capsize=4)
    ax.set_xticks(range(len(labels_order)))
    ax.set_xticklabels([label_display[l] for l in labels_order], fontsize=9)
    ax.set_ylabel("Test RMSE (mean±std)")
    ax.set_title("Phase 3B: RMSE by Model Variant", fontsize=11)

    fig.suptitle("Phase 3B: LSTM Variants vs GRU — MPC Rate & RMSE\n"
                 "(FD003, A1+B2+C2, 10 seeds)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_mpc_variants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[시각화] fig1_mpc_variants.png")

    # ---- Fig 2: Forget gate / Update gate 궤적 (collapsed vs normal) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, model_label, gate_col, gate_name, color_c, color_n in [
        (axes[0], "LSTM_gate", "fg_mean", "Forget Gate (LSTM)", "#d62728", "#2ca02c"),
        (axes[1], "GRU_gate",  "ug_mean", "Update Gate (GRU)",  "#1f77b4", "#9467bd"),
    ]:
        sub_df = df[df["label"] == model_label]
        for _, row in sub_df.iterrows():
            log_path = LOG_DIR / f"{model_label}_seed{int(row['seed'])}.csv"
            if not log_path.exists():
                continue
            log = pd.read_csv(log_path)
            if gate_col not in log.columns:
                continue
            color = color_c if row["is_collapsed"] else color_n
            alpha = 0.6
            label_str = None
            ax.plot(log["epoch"], log[gate_col],
                    color=color, alpha=alpha, linewidth=0.8)

        # 대표 레전드
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=color_c, linewidth=2, label="Collapsed"),
            Line2D([0], [0], color=color_n, linewidth=2, label="Normal"),
        ]
        ax.legend(handles=handles, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Mean {gate_name} Activation")
        ax.set_title(f"{model_label}: {gate_name} Trajectory", fontsize=11)
        ax.set_xlim(1, 60)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Phase 3B: Gate Activation — Collapsed vs Normal\n"
                 "(first 60 epochs, FD003 A1+B2+C2)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_gate_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[시각화] fig2_gate_trajectory.png")

    # ---- Fig 3: Forget gate at stop epoch — collapsed vs normal ----
    lstm_logs = df[df["label"] == "LSTM_gate"]
    fg_at_stop_coll, fg_at_stop_norm = [], []
    for _, row in lstm_logs.iterrows():
        log_path = LOG_DIR / f"LSTM_gate_seed{int(row['seed'])}.csv"
        if not log_path.exists():
            continue
        log = pd.read_csv(log_path)
        if "fg_mean" not in log.columns:
            continue
        stop_row = log[log["stopped"] == True]
        if stop_row.empty:
            fg_val = log["fg_mean"].iloc[-1]
        else:
            fg_val = stop_row["fg_mean"].iloc[0]
        if row["is_collapsed"]:
            fg_at_stop_coll.append(fg_val)
        else:
            fg_at_stop_norm.append(fg_val)

    if fg_at_stop_coll or fg_at_stop_norm:
        fig, ax = plt.subplots(figsize=(7, 4))
        if fg_at_stop_coll:
            ax.scatter(range(len(fg_at_stop_coll)), fg_at_stop_coll,
                       c="#d62728", s=60, zorder=5, label=f"Collapsed (n={len(fg_at_stop_coll)})")
        if fg_at_stop_norm:
            ax.scatter([i + len(fg_at_stop_coll) + 1 for i in range(len(fg_at_stop_norm))],
                       fg_at_stop_norm,
                       c="#2ca02c", s=60, zorder=5, label=f"Normal (n={len(fg_at_stop_norm)})")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Forget Gate Mean at ES Stop")
        ax.set_title("Phase 3B: LSTM Forget Gate Value at Early Stopping", fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig3_fg_at_stop.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[시각화] fig3_fg_at_stop.png")


if __name__ == "__main__":
    main()
