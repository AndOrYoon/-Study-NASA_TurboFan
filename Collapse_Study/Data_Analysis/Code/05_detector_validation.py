# -*- coding: utf-8 -*-
"""
05_detector_validation.py
Phase 4 — 예방 분류체계 + 소급 진단 도구  (Decision_log.md D2)

Part A: 예방 분류체계 (Prevention Taxonomy)
  - 신규 실험: forget gate clamp [ε,1], ε ∈ {0.001, 0.01, 0.05} × 10 seeds = 30 runs
  - 기존 결과 통합: Phase1A(split/warmup) + Phase1B(loss/init/patience) + Phase3(GRU) + Phase3B(V2_fg1)
  - 비교 축: MPC rate × RMSE mean(정상) × 구현 비용

Part B: 소급 진단 도구 (Retrospective Audit Tool)
  - 보정 (Calibration): Phase1A 전체 runs (270) — 다양한 조건
  - 검증 (Held-out): Phase1B + Phase3 + Phase3B (330 runs, 다른 개입·아키텍처·구현)
  - ROC + PDR<0.05 고정 임계값 성능 + 단일/복합 지표 비교

Results: Collapse_Study/Data_Analysis/Results/Phase4/
"""

import sys, time
sys.stdout.reconfigure(encoding="utf-8")

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
    make_loader, SeqDataset, compute_metrics,
)

RES_DIR   = _CS_ROOT / "Data_Analysis" / "Results"
PHASE4    = RES_DIR / "Phase4"
FG_CLAMP  = PHASE4 / "fg_clamp_runs"
FIG_DIR   = PHASE4 / "figures"
for d in [PHASE4, FG_CLAMP, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────── 하이퍼파라미터 ───────────────
DATASET    = "FD003"
CLIP_VAL   = 125
VAL_SEED   = 42          # A1+B2+C2 붕괴 유발 조건 그대로
MIN_EPOCHS = 0
MAX_EPOCHS = 200
PATIENCE   = 15
BATCH_SIZE = 256
LR         = 1e-3
SEEDS      = list(range(10))
EPS        = 1e-8
HIDDEN     = 64
DROPOUT    = 0.2
FG_EPSILONS = [0.001, 0.01, 0.05]

# ─────────────── 모델 ───────────────

class CustomLSTMClampCell(nn.Module):
    """forget gate를 [fg_min, 1]로 clamp하는 LSTM cell."""
    def __init__(self, input_size, hidden_size, fg_min=0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.fg_min = fg_min
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates    = self.linear(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f).clamp(min=self.fg_min)   # ← clamp
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class LSTMFgClampModel(nn.Module):
    """2-layer stacked LSTM with forget gate clamp."""
    def __init__(self, n_features, fg_min=0.01):
        super().__init__()
        self.cell1 = CustomLSTMClampCell(n_features, HIDDEN, fg_min)
        self.drop1 = nn.Dropout(DROPOUT)
        self.cell2 = CustomLSTMClampCell(HIDDEN, HIDDEN, fg_min)
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

# ─────────────── 지표 ───────────────

def mpc_metrics(pred, true):
    pdr  = float(np.std(pred) / (np.std(true) + EPS))
    ss_r = float(np.sum((true - pred) ** 2))
    ss_t = float(np.sum((true - np.mean(true)) ** 2))
    r2   = 1.0 - ss_r / (ss_t + EPS)
    cbr  = float(np.sqrt(ss_r / len(pred)) / (np.sqrt(ss_t / len(true)) + EPS))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return {"PDR": round(pdr,6), "R2": round(r2,6), "CBR": round(cbr,6), "RMSE": round(rmse,4)}

def is_mpc(m):
    return m["PDR"] < 0.05 and m["R2"] <= 0.0

# ─────────────── 데이터 로드 ───────────────

def load_data():
    train_df    = load_cmapss(DATASET, "train")
    sensor_cols = get_sensor_cols(DATASET)
    train_df    = add_rul(train_df, rul_clip=CLIP_VAL)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n     = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)
    c_train     = float(train_df["RUL"].mean())
    test_df     = load_cmapss(DATASET, "test")
    test_rul    = load_test_rul(DATASET)
    test_n      = apply_normalization(test_df, sensor_cols, min_v, max_v)
    X_te, y_te  = make_test_sequences(test_n, sensor_cols, test_rul)
    y_te = np.clip(y_te, 0, CLIP_VAL)
    return X, y, units, sensor_cols, c_train, X_te, y_te

# ─────────────── 학습 루프 ───────────────

@torch.no_grad()
def predict(model, X_np, device, batch=512):
    model.eval()
    ds  = SeqDataset(X_np, np.zeros(len(X_np), dtype=np.float32))
    ld  = DataLoader(ds, batch_size=batch, shuffle=False)
    out = []
    for xb, _ in ld:
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out).flatten()


def train_one(model, X_tr, y_tr, X_va, y_va, device):
    tr_ld = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    va_ld = make_loader(X_va, y_va, BATCH_SIZE, shuffle=False)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit  = nn.MSELoss()
    best_val, patience_cnt, best_sd = float("inf"), 0, None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        va_loss = sum(
            crit(model(xb.to(device)), yb.to(device)).item() * len(xb)
            for xb, yb in va_ld
        ) / len(y_va)
        if va_loss < best_val:
            best_val = va_loss; patience_cnt = 0
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                if best_sd: model.load_state_dict(best_sd)
                return epoch
    if best_sd: model.load_state_dict(best_sd)
    return MAX_EPOCHS

# ─────────────── PART A: fg_clamp 신규 실험 ───────────────

def run_fg_clamp(device, X, y, units, sensor_cols, c_train, X_te, y_te):
    out_csv = FG_CLAMP / "runs.csv"
    if out_csv.exists():
        df_done  = pd.read_csv(out_csv)
        done_keys = set(zip(df_done["epsilon"].astype(str), df_done["seed"].astype(str)))
        all_rows  = df_done.to_dict("records")
    else:
        done_keys, all_rows = set(), []

    total = len(FG_EPSILONS) * len(SEEDS)
    done  = len(done_keys)
    print(f"\n[Part A] fg_clamp 신규 실험: {len(FG_EPSILONS)} ε × {len(SEEDS)} seeds = {total} runs")
    print(f"  기존 {done}개 완료, {total-done}개 남음")

    t0 = time.time()
    tr_m, va_m = split_engines(units, val_frac=0.2, seed=VAL_SEED)
    X_va, y_va = X[va_m], y[va_m]
    X_tr, y_tr = X[tr_m], y[tr_m]

    for eps in FG_EPSILONS:
        for seed in SEEDS:
            if (str(eps), str(seed)) in done_keys:
                continue
            done += 1
            set_seed(seed)
            model = LSTMFgClampModel(len(sensor_cols), fg_min=eps).to(device)
            t_s = time.time()
            ep  = train_one(model, X_tr, y_tr, X_va, y_va, device)
            pred = np.clip(predict(model, X_te, device), 0, CLIP_VAL)
            m    = mpc_metrics(pred, y_te)
            rmse, ns = compute_metrics(pred, y_te)
            row = {
                "epsilon": eps, "seed": seed, "stop_epoch": ep,
                "RMSE": round(rmse,4), "NASA": round(ns,2),
                "test_PDR": m["PDR"], "test_R2": m["R2"], "test_CBR": m["CBR"],
                "elapsed_s": round(time.time()-t_s, 1),
                "is_collapsed": is_mpc(m),
            }
            all_rows.append(row)
            done_keys.add((str(eps), str(seed)))
            tag = "COLLAPSED" if row["is_collapsed"] else "ok"
            print(f"  ε={eps}  seed={seed:>2}  ep={ep:>3}  RMSE={rmse:.2f}"
                  f"  PDR={m['PDR']:.4f}  [{tag}]  ({row['elapsed_s']:.1f}s)")

        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        sub = [r for r in all_rows if r["epsilon"] == eps]
        if sub:
            df_s = pd.DataFrame(sub)
            print(f"  └─ ε={eps}: MPC {int(df_s['is_collapsed'].sum())}/{len(df_s)}"
                  f"  RMSE={df_s['RMSE'].mean():.2f}±{df_s['RMSE'].std():.2f}")

    print(f"  총 소요: {(time.time()-t0)/60:.1f}분")
    return pd.read_csv(out_csv)

# ─────────────── PART A: 분류체계 통합 ───────────────

COST_MAP = {
    # (label, cost_score, cost_label)
    "Baseline":              (4, "없음"),
    "GRU 교체":              (3, "중간 — 아키텍처 변경"),
    "forget gate=1 (V2_fg1)":(2, "낮음 — 커스텀 셀"),
    "forget gate clamp 0.001":(1, "매우 낮음 — 1줄"),
    "forget gate clamp 0.01": (1, "매우 낮음 — 1줄"),
    "forget gate clamp 0.05": (1, "매우 낮음 — 1줄"),
    "MAE loss":              (1, "매우 낮음 — 1줄"),
    "bias_init=train_mean":  (1, "매우 낮음 — 1줄"),
    "per-seed split":        (1, "매우 낮음 — 프로토콜"),
    "ES warmup (B3)":        (1, "매우 낮음 — 프로토콜"),
    "patience=30":           (1, "매우 낮음 — 파라미터"),
    "V1_fb1 (forget bias+1)":(1, "매우 낮음 — init"),
}


def compile_taxonomy(fg_df):
    rows = []

    def add(label, mpc_rate, rmse_all_mean, rmse_all_std, rmse_norm_mean, n):
        cs, cl = COST_MAP.get(label, (2, "?"))
        rows.append({
            "method": label,
            "mpc_rate": round(mpc_rate, 2),
            "rmse_all": f"{rmse_all_mean:.2f}±{rmse_all_std:.2f}",
            "rmse_normal": f"{rmse_norm_mean:.2f}" if not np.isnan(rmse_norm_mean) else "—",
            "n_seeds": n,
            "cost_score": cs,
            "cost_label": cl,
        })

    # ── Baseline: Phase3B LSTM_gate
    p3b = pd.read_csv(RES_DIR / "Phase3B" / "runs.csv")
    base = p3b[p3b["label"] == "LSTM_gate"]
    norm = base[~base["is_collapsed"]]
    add("Baseline", base["is_collapsed"].mean(),
        base["RMSE"].mean(), base["RMSE"].std(),
        norm["RMSE"].mean() if len(norm) else np.nan, len(base))

    # ── GRU: Phase3 FD003/GRU
    p3 = pd.read_csv(RES_DIR / "Phase3" / "runs.csv")
    gru = p3[(p3["arch"] == "GRU") & (p3["dataset"] == "FD003")]
    norm = gru[~gru["is_collapsed"]]
    add("GRU 교체", gru["is_collapsed"].mean(),
        gru["RMSE"].mean(), gru["RMSE"].std(),
        norm["RMSE"].mean() if len(norm) else np.nan, len(gru))

    # ── V2_fg1: Phase3B
    v2 = p3b[p3b["label"] == "V2_fg1"]
    norm = v2[~v2["is_collapsed"]]
    add("forget gate=1 (V2_fg1)", v2["is_collapsed"].mean(),
        v2["RMSE"].mean(), v2["RMSE"].std(),
        norm["RMSE"].mean() if len(norm) else np.nan, len(v2))

    # ── V1_fb1: Phase3B
    v1 = p3b[p3b["label"] == "V1_fb1"]
    norm = v1[~v1["is_collapsed"]]
    add("V1_fb1 (forget bias+1)", v1["is_collapsed"].mean(),
        v1["RMSE"].mean(), v1["RMSE"].std(),
        norm["RMSE"].mean() if len(norm) else np.nan, len(v1))

    # ── fg_clamp (신규)
    for eps in FG_EPSILONS:
        sub = fg_df[fg_df["epsilon"] == eps]
        norm = sub[~sub["is_collapsed"]]
        lbl = f"forget gate clamp {eps}"
        add(lbl, sub["is_collapsed"].mean(),
            sub["RMSE"].mean(), sub["RMSE"].std(),
            norm["RMSE"].mean() if len(norm) else np.nan, len(sub))

    # ── Phase1B 조건들
    p1b = pd.read_csv(RES_DIR / "Phase1B" / "runs.csv")

    for factor, bias_init, lbl in [
        ("F_bias_init", "train_mean", "bias_init=train_mean"),
        ("F_bias_init", "random_calibrated", None),
    ]:
        if lbl is None: continue
        sub = p1b[(p1b["factor"] == factor) & (p1b["bias_init"] == bias_init)]
        if len(sub) == 0: continue
        norm = sub[~sub["is_collapsed"]]
        add(lbl, sub["is_collapsed"].mean(),
            sub["RMSE"].mean(), sub["RMSE"].std(),
            norm["RMSE"].mean() if len(norm) else np.nan, len(sub))

    mae = p1b[(p1b["factor"] == "G_loss") & (p1b["loss"] == "MAE")]
    if len(mae):
        norm = mae[~mae["is_collapsed"]]
        add("MAE loss", mae["is_collapsed"].mean(),
            mae["RMSE"].mean(), mae["RMSE"].std(),
            norm["RMSE"].mean() if len(norm) else np.nan, len(mae))

    pat30 = p1b[(p1b["factor"] == "D_patience") & (p1b["patience"] == 30)]
    if len(pat30):
        norm = pat30[~pat30["is_collapsed"]]
        add("patience=30", pat30["is_collapsed"].mean(),
            pat30["RMSE"].mean(), pat30["RMSE"].std(),
            norm["RMSE"].mean() if len(norm) else np.nan, len(pat30))

    # ── Phase1A 조건들
    p1a = pd.read_csv(RES_DIR / "Phase1A" / "runs.csv")
    BASE_B, BASE_C = "B2_from0", "C2_clip125"

    per_seed = p1a[(p1a["cond_a"].str.startswith("A2")) &
                   (p1a["cond_b"] == BASE_B) & (p1a["cond_c"] == BASE_C)]
    if len(per_seed):
        norm = per_seed[~per_seed["is_collapsed"]]
        add("per-seed split", per_seed["is_collapsed"].mean(),
            per_seed["RMSE"].mean(), per_seed["RMSE"].std(),
            norm["RMSE"].mean() if len(norm) else np.nan, len(per_seed))

    warmup = p1a[(p1a["cond_a"].str.startswith("A1")) &
                 (p1a["cond_b"].str.startswith("B3")) & (p1a["cond_c"] == BASE_C)]
    if len(warmup):
        norm = warmup[~warmup["is_collapsed"]]
        add("ES warmup (B3)", warmup["is_collapsed"].mean(),
            warmup["RMSE"].mean(), warmup["RMSE"].std(),
            norm["RMSE"].mean() if len(norm) else np.nan, len(warmup))

    df = pd.DataFrame(rows)
    df.to_csv(PHASE4 / "taxonomy.csv", index=False)
    return df


def print_taxonomy(df):
    print("\n" + "=" * 72)
    print("Phase 4 Part A — 예방 분류체계")
    print("=" * 72)
    print(f"{'방법':<28} {'MPC%':>5} {'RMSE(전체)':>12} {'RMSE(정상)':>10} {'n':>4} {'비용'}")
    print("-" * 72)
    for _, r in df.sort_values("mpc_rate").iterrows():
        print(f"{r['method']:<28} {r['mpc_rate']*100:>4.0f}%"
              f" {r['rmse_all']:>12} {r['rmse_normal']:>10}"
              f" {r['n_seeds']:>4}  {r['cost_label']}")
    print("=" * 72)

# ─────────────── PART B: 소급 진단 도구 ───────────────

def run_audit(fg_df):
    """Calibration: Phase1A / Held-out: Phase1B + Phase3 + Phase3B"""
    print("\n[Part B] 소급 진단 도구 (Retrospective Audit Tool)")

    # ── 보정 데이터 (Phase1A)
    p1a = pd.read_csv(RES_DIR / "Phase1A" / "runs.csv")
    cal = p1a[["test_PDR", "test_R2", "test_CBR", "RMSE", "is_collapsed"]].copy()
    cal["source"] = "Phase1A"

    # ── Held-out 1: Phase1B
    p1b = pd.read_csv(RES_DIR / "Phase1B" / "runs.csv")
    h1  = p1b[["test_PDR", "test_R2", "test_CBR", "RMSE", "is_collapsed"]].copy()
    h1["source"] = "Phase1B"

    # ── Held-out 2: Phase3 (all)
    p3 = pd.read_csv(RES_DIR / "Phase3" / "runs.csv")
    h2 = p3[["test_PDR", "test_R2", "RMSE", "is_collapsed"]].copy()
    h2["test_CBR"] = np.nan; h2["source"] = "Phase3"

    # ── Held-out 3: Phase3B
    p3b = pd.read_csv(RES_DIR / "Phase3B" / "runs.csv")
    h3  = p3b[["test_PDR", "test_R2", "RMSE", "is_collapsed"]].copy()
    h3["test_CBR"] = np.nan; h3["source"] = "Phase3B"

    held = pd.concat([h1, h2, h3], ignore_index=True)

    print(f"  보정 데이터: {len(cal)}개  (Phase1A, "
          f"collapsed={int(cal['is_collapsed'].sum())}, "
          f"normal={int((~cal['is_collapsed']).sum())})")
    print(f"  검증 데이터: {len(held)}개  (Phase1B+3+3B, "
          f"collapsed={int(held['is_collapsed'].sum())}, "
          f"normal={int((~held['is_collapsed']).sum())})")

    results = {}

    def roc_metrics(df, feature, threshold, direction="lt"):
        y_true = df["is_collapsed"].values.astype(int)
        y_score = df[feature].values
        if direction == "lt":
            y_pred = (y_score < threshold).astype(int)
        else:
            y_pred = (y_score > threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        ba   = (sens + spec) / 2
        return {"TP":tp,"FP":fp,"TN":tn,"FN":fn,
                "sensitivity":round(sens,4), "specificity":round(spec,4),
                "balanced_acc":round(ba,4)}

    def compute_auroc(df, feature, direction="lt"):
        y_true = df["is_collapsed"].values.astype(int)
        scores = df[feature].values
        if direction == "lt":
            scores = -scores   # invert so higher = more collapsed
        thresh_list = np.sort(np.unique(scores))[::-1]
        tprs, fprs = [0.0], [0.0]
        P = y_true.sum(); N = len(y_true) - P
        if P == 0 or N == 0:
            return float("nan")
        for t in thresh_list:
            pred = (scores >= t).astype(int)
            tprs.append(((pred==1)&(y_true==1)).sum() / P)
            fprs.append(((pred==1)&(y_true==0)).sum() / N)
        tprs.append(1.0); fprs.append(1.0)
        tprs, fprs = np.array(tprs), np.array(fprs)
        try:
            return float(np.trapezoid(tprs, fprs))   # NumPy 2.0+
        except AttributeError:
            return float(np.trapz(tprs, fprs))        # NumPy <2.0

    THRESHOLD_PDR = 0.05   # our definition
    THRESHOLD_R2  = 0.0
    THRESHOLD_RMSE_HIGH = 25.0   # domain-specific (FD003 scale)

    for label, ds in [("보정(Cal)", cal), ("검증(Held-out)", held)]:
        if ds["is_collapsed"].sum() == 0 or (~ds["is_collapsed"]).sum() == 0:
            continue
        m_pdr  = roc_metrics(ds, "test_PDR", THRESHOLD_PDR, "lt")
        auroc_pdr = compute_auroc(ds, "test_PDR", "lt")
        m_r2   = roc_metrics(ds, "test_R2",  THRESHOLD_R2,  "lt")
        auroc_r2  = compute_auroc(ds, "test_R2", "lt")
        m_rmse = roc_metrics(ds, "RMSE",     THRESHOLD_RMSE_HIGH, "gt")
        auroc_rmse = compute_auroc(ds, "RMSE", "gt")

        # Composite: PDR<0.05 OR R2<=0
        y_true = ds["is_collapsed"].values.astype(int)
        y_comb = ((ds["test_PDR"] < THRESHOLD_PDR) | (ds["test_R2"] <= THRESHOLD_R2)).astype(int)
        tp_c = int(((y_comb==1)&(y_true==1)).sum()); fp_c = int(((y_comb==1)&(y_true==0)).sum())
        tn_c = int(((y_comb==0)&(y_true==0)).sum()); fn_c = int(((y_comb==0)&(y_true==1)).sum())
        sens_c = tp_c/(tp_c+fn_c+1e-9); spec_c = tn_c/(tn_c+fp_c+1e-9)
        ba_c   = (sens_c+spec_c)/2

        results[label] = {
            "PDR<0.05":   m_pdr,   "AUROC(PDR)":  round(auroc_pdr,4),
            "R²≤0":       m_r2,    "AUROC(R²)":   round(auroc_r2,4),
            "RMSE>25":    m_rmse,  "AUROC(RMSE)": round(auroc_rmse,4),
            "PDR+R²(OR)": {"sensitivity":round(sens_c,4),"specificity":round(spec_c,4),
                           "balanced_acc":round(ba_c,4),"TP":tp_c,"FP":fp_c,"TN":tn_c,"FN":fn_c},
        }

    # ── 출력
    print("\n  [소급 진단 성능 비교]")
    header = f"{'지표':<20} {'데이터':>14} {'민감도':>8} {'특이도':>8} {'균형정확도':>10} {'AUROC':>8}"
    print("  " + header)
    print("  " + "-" * 72)
    for ds_label, res in results.items():
        for metric_name in ["PDR<0.05", "R²≤0", "RMSE>25", "PDR+R²(OR)"]:
            m = res[metric_name]
            auroc_key = {"PDR<0.05": "AUROC(PDR)", "R²≤0": "AUROC(R²)",
                         "RMSE>25": "AUROC(RMSE)", "PDR+R²(OR)": None}[metric_name]
            auroc_str = f"{res[auroc_key]:.4f}" if auroc_key else "  —  "
            print(f"  {metric_name:<20} {ds_label:>14}"
                  f" {m['sensitivity']:>8.4f} {m['specificity']:>8.4f}"
                  f" {m['balanced_acc']:>10.4f} {auroc_str:>8}")
        print()

    # ── CSV 저장
    rows_out = []
    for ds_label, res in results.items():
        for mname, m in res.items():
            if isinstance(m, dict) and "sensitivity" in m:
                rows_out.append({"dataset": ds_label, "metric": mname, **m})
    pd.DataFrame(rows_out).to_csv(PHASE4 / "audit_metrics.csv", index=False)

    return results

# ─────────────── 시각화 ───────────────

def generate_figures(tax_df, audit_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Fig 1: 예방 분류체계 — MPC rate + RMSE(정상) by cost
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MPC rate bar (sorted, colored by cost)
    df_sorted = tax_df.sort_values("mpc_rate", ascending=False)
    cost_colors = {1: "#2ca02c", 2: "#1f77b4", 3: "#ff7f0e", 4: "#d62728"}
    colors = [cost_colors.get(int(c), "gray") for c in df_sorted["cost_score"]]
    axes[0].barh(range(len(df_sorted)), df_sorted["mpc_rate"] * 100,
                 color=colors, alpha=0.85)
    axes[0].set_yticks(range(len(df_sorted)))
    axes[0].set_yticklabels(df_sorted["method"], fontsize=8)
    axes[0].set_xlabel("MPC Rate (%)")
    axes[0].set_title("예방 분류체계: MPC Rate\n(색상=구현 비용: 녹색=매우낮음, 파랑=낮음, 주황=중간, 빨강=없음)")
    axes[0].axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # RMSE(정상 run) bar
    def parse_rmse_normal(s):
        try: return float(s)
        except: return np.nan
    rmse_vals = [parse_rmse_normal(v) for v in df_sorted["rmse_normal"]]
    axes[1].barh(range(len(df_sorted)), rmse_vals,
                 color=colors, alpha=0.85)
    axes[1].set_yticks(range(len(df_sorted)))
    axes[1].set_yticklabels(df_sorted["method"], fontsize=8)
    axes[1].set_xlabel("RMSE (정상 수렴 run 평균)")
    axes[1].set_title("예방 분류체계: 정상 run RMSE\n(낮을수록 좋음)")
    axes[1].axvline(13.0, color="steelblue", linestyle="--", linewidth=1,
                    label="수정 M0 기준(~13)")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, 20)

    fig.suptitle("Phase 4 Part A: Prevention Taxonomy\n"
                 "(FD003, A1+B2+C2, 10 seeds)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_taxonomy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[시각화] fig1_taxonomy.png")

    # ── Fig 2: fg_clamp 상세 — MPC rate vs ε
    fg_csv = FG_CLAMP / "runs.csv"
    if fg_csv.exists():
        fg_df = pd.read_csv(fg_csv)
        agg   = fg_df.groupby("epsilon").agg(
            mpc_rate=("is_collapsed","mean"),
            rmse_mean=("RMSE","mean"), rmse_std=("RMSE","std"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax2 = ax.twinx()
        ax.bar(range(len(agg)), agg["mpc_rate"] * 100,
               alpha=0.7, color="#d62728", label="MPC Rate (%)")
        ax2.errorbar(range(len(agg)), agg["rmse_mean"], yerr=agg["rmse_std"],
                     fmt="o-", color="steelblue", linewidth=2, capsize=5,
                     label="RMSE mean±std")
        ax.axhline(80, color="gray", linestyle="--", linewidth=1, label="Baseline (ε=0, 0.80)")
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels([f"ε={v}" for v in agg["epsilon"]])
        ax.set_ylabel("MPC Rate (%)", color="#d62728")
        ax2.set_ylabel("RMSE", color="steelblue")
        ax.set_ylim(0, 100); ax2.set_ylim(0, 50)
        ax.set_title("Phase 4: forget gate clamp [ε, 1] — ε별 MPC rate & RMSE")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig2_fg_clamp.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[시각화] fig2_fg_clamp.png")

    # ── Fig 3: 소급 진단 ROC — 보정 vs 검증
    if audit_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        metric_styles = {
            "PDR<0.05":   ("#d62728", "PDR < 0.05"),
            "R²≤0":       ("#ff7f0e", "R² ≤ 0"),
            "RMSE>25":    ("#9467bd", "RMSE > 25"),
            "PDR+R²(OR)": ("#2ca02c", "PDR<0.05 OR R²≤0"),
        }
        for ax, (ds_label, res) in zip(axes, audit_results.items()):
            for metric_name, (color, display) in metric_styles.items():
                if metric_name not in res: continue
                m = res[metric_name]
                sens = m["sensitivity"]; spec = m["specificity"]
                ax.scatter([1-spec], [sens], color=color, s=120, zorder=5)
                ax.annotate(display, (1-spec, sens), textcoords="offset points",
                            xytext=(5, 3), fontsize=7, color=color)
            ax.plot([0,1],[0,1],"k--",linewidth=0.7)
            ax.set_xlabel("1 - Specificity (FPR)")
            ax.set_ylabel("Sensitivity (TPR)")
            ax.set_title(f"소급 진단 도구: {ds_label}\n(점 = 단일 임계값 성능)")
            ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
        fig.suptitle("Phase 4 Part B: Retrospective Audit Tool\n"
                     "(보정=Phase1A, 검증=Phase1B+3+3B)", fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig3_audit_roc.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[시각화] fig3_audit_roc.png")

    # ── Fig 4: PDR 분포 (보정 데이터 — collapsed vs normal)
    p1a = pd.read_csv(RES_DIR / "Phase1A" / "runs.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    coll = p1a[p1a["is_collapsed"]]["test_PDR"].values
    norm = p1a[~p1a["is_collapsed"]]["test_PDR"].values
    ax.hist(coll, bins=30, alpha=0.7, color="#d62728", label=f"Collapsed (n={len(coll)})", density=True)
    ax.hist(norm, bins=30, alpha=0.7, color="#2ca02c", label=f"Normal (n={len(norm)})", density=True)
    ax.axvline(0.05, color="black", linestyle="--", linewidth=1.5, label="Threshold = 0.05")
    ax.set_xlabel("test PDR (ES 종료 시점)")
    ax.set_ylabel("밀도")
    ax.set_title("Phase 4 Part B: PDR 분포 (보정 데이터 = Phase1A)\n"
                 "PDR < 0.05 → Collapsed로 진단")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 1.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_pdr_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[시각화] fig4_pdr_distribution.png")


# ─────────────── 메인 ───────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_all = time.time()
    print(f"Device: {device}")
    print(f"Phase 4: 예방 분류체계 + 소급 진단 도구\n")

    # ─ Part A: 신규 실험
    print("데이터 로딩 중...")
    X, y, units, sensor_cols, c_train, X_te, y_te = load_data()
    print(f"  n_features={len(sensor_cols)}  n_seq={len(X)}\n")
    fg_df = run_fg_clamp(device, X, y, units, sensor_cols, c_train, X_te, y_te)

    # ─ Part A: 분류체계 통합
    tax_df = compile_taxonomy(fg_df)
    print_taxonomy(tax_df)

    # ─ Part B: 소급 진단
    audit_res = run_audit(fg_df)

    # ─ 시각화
    try:
        generate_figures(tax_df, audit_res)
    except Exception as e:
        print(f"\n[경고] 시각화 실패: {e}")

    print(f"\n총 실험 시간: {(time.time()-t_all)/60:.1f}분")
    print(f"결과 저장: {PHASE4}")


if __name__ == "__main__":
    main()
