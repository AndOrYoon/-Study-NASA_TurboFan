# -*- coding: utf-8 -*-
"""
01_mechanism_screen.py
Phase 1A — Core Mechanism Factorial Screen

3×3×3 full factorial:
  Factor A — Validation Composition (3 levels)
    A1: fixed split (seed=42, 원본 버그 재현)
    A2: per-seed split (seed=training_seed)
    A3: stratified split (lifetime_quartile × fault_mode)

  Factor B — Early Stopping (3 levels)
    B1: off (run to MAX_EPOCHS)
    B2: on from epoch 0 (no warmup, 원본 버그 재현)
    B3: on after MIN_EPOCHS warmup

  Factor C — RUL Labeling (3 levels)
    C1: unclipped
    C2: clip=125 (기존 표준)
    C3: clip=100 (대안 threshold)

데이터셋: FD003 (단일 운전조건, 2 결함모드 — 붕괴 취약 환경)
아키텍처: LSTM backbone (Phase 0와 동일)
Seeds: 10 per condition (0–9)
MAX_EPOCHS: 200 (ES off 조건도 200까지)
MIN_EPOCHS: 30 (B3 조건)
PATIENCE: 15

결과 저장:
  Collapse_Study/Data_Analysis/Results/Phase1A/
    runs.csv           ← 조건×seed별 전체 결과
    epoch_logs/        ← 조건×seed별 epoch 궤적
    mpc_summary.csv    ← 조건별 MPC 발생률 집계
    figures/           ← MPC occurrence heatmap, trajectory plots
"""

import sys, os, time, itertools
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
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
from sklearn.preprocessing import LabelEncoder

from h6_p2_model_utils import (
    set_seed, RUL_CLIP, WINDOW_SIZE,
    load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    make_loader, LSTMBranch,
    predict_sequences, compute_metrics,
)

PHASE1A_DIR = _CS_ROOT / "Data_Analysis" / "Results" / "Phase1A"
LOG_DIR     = PHASE1A_DIR / "epoch_logs"
FIG_DIR     = PHASE1A_DIR / "figures"
for d in [PHASE1A_DIR, LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 실험 설정
# ---------------------------------------------------------------------------
DATASET    = "FD003"
SEEDS      = list(range(10))
MAX_EPOCHS = 200
MIN_EPOCHS = 30
PATIENCE   = 15
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4
EPS        = 1e-8

FACTOR_A = ["A1_fixed", "A2_per_seed", "A3_stratified"]
FACTOR_B = ["B1_off", "B2_from0", "B3_warmup"]
FACTOR_C = ["C1_unclipped", "C2_clip125", "C3_clip100"]

ALL_CONDITIONS = list(itertools.product(FACTOR_A, FACTOR_B, FACTOR_C))
print(f"총 실험 조건: {len(ALL_CONDITIONS)} (3×3×3 factorial)")
print(f"Seeds per condition: {len(SEEDS)}")
print(f"총 runs: {len(ALL_CONDITIONS) * len(SEEDS)}")

# ---------------------------------------------------------------------------
# MPC 지표
# ---------------------------------------------------------------------------

def mpc_metrics(pred: np.ndarray, true: np.ndarray, c_train: float) -> dict:
    pdr = float(np.std(pred) / (np.std(true) + EPS))
    rmse_model = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmse_const = float(np.sqrt(np.mean((true - c_train) ** 2)))
    cbr = rmse_model / (rmse_const + EPS)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + EPS)
    return {
        "PDR": round(pdr, 6), "CBR": round(cbr, 6),
        "R2": round(r2, 6), "RMSE": round(rmse_model, 4),
        "RMSE_const": round(rmse_const, 4),
    }


def is_mpc(m: dict) -> bool:
    """Phase 0 결과 기반 MPC 판정: PDR < 0.05 AND R2 <= 0."""
    return m["PDR"] < 0.05 and m["R2"] <= 0.0

# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------

def build_dataset(clip_value):
    """clip_value=None → unclipped."""
    train_df    = load_cmapss(DATASET, "train")
    sensor_cols = get_sensor_cols(DATASET)
    # RUL 계산 (clip 적용)
    max_cyc = train_df.groupby("unit_number")["cycle"].max()
    train_df = train_df.copy()
    train_df["RUL"] = train_df["unit_number"].map(max_cyc) - train_df["cycle"]
    if clip_value is not None:
        train_df["RUL"] = train_df["RUL"].clip(upper=clip_value)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)
    c_train = float(train_df["RUL"].mean())

    test_df  = load_cmapss(DATASET, "test")
    test_rul = load_test_rul(DATASET)
    test_n   = apply_normalization(test_df, sensor_cols, min_v, max_v)
    # test RUL도 clip 적용 (비교 일관성)
    if clip_value is not None:
        y_te_raw = np.array([float(test_rul.get(u, 0)) for u in
                              test_df.groupby("unit_number").groups.keys()])
    X_te, y_te = make_test_sequences(test_n, sensor_cols, test_rul)
    if clip_value is not None:
        y_te = np.clip(y_te, 0, clip_value)

    return X, y, units, sensor_cols, c_train, X_te, y_te


def stratified_split(units: np.ndarray, train_df_rul: pd.DataFrame,
                     val_frac: float = 0.2, seed: int = 42):
    """lifetime_quartile × fault_mode 층화 분할.
    fault_mode proxy: 각 엔진의 s20 센서 최종값 기준 2분위 (FD003 2결함모드).
    """
    raw_train = load_cmapss(DATASET, "train")
    unit_stats = raw_train.groupby("unit_number").agg(
        max_cycle=("cycle", "max"),
        s20_last=("s20", "last"),
    ).reset_index()
    unit_stats["lifetime_q"] = pd.qcut(unit_stats["max_cycle"], 4, labels=False, duplicates="drop")
    unit_stats["fault_q"]    = pd.qcut(unit_stats["s20_last"], 2, labels=False, duplicates="drop")
    unit_stats["stratum"]    = unit_stats["lifetime_q"].astype(str) + "_" + unit_stats["fault_q"].astype(str)

    rng = np.random.RandomState(seed)
    val_set = set()
    for _, grp in unit_stats.groupby("stratum"):
        n_val = max(1, int(len(grp) * val_frac))
        chosen = rng.choice(grp["unit_number"].values, n_val, replace=False)
        val_set.update(chosen.tolist())

    tr_m = np.array([u not in val_set for u in units])
    va_m = ~tr_m
    return tr_m, va_m

# ---------------------------------------------------------------------------
# 학습 루프 (epoch 로그 포함)
# ---------------------------------------------------------------------------

def train_with_log(model, tr_loader, va_loader, X_val, y_val, c_train,
                   device, max_epochs, patience, min_epochs=0):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()
    best_val  = float("inf")
    patience_cnt = 0
    best_sd   = None
    epoch_log = []

    for epoch in range(1, max_epochs + 1):
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
        tr_loss = tr_total / tr_n

        model.eval()
        va_total, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_total += criterion(model(xb), yb).item() * len(xb)
                va_n     += len(xb)
        va_loss = va_total / va_n

        # val MPC 지표
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
            if epoch > min_epochs:
                patience_cnt += 1
                if patience_cnt >= patience:
                    stopped = True

        epoch_log.append({
            "epoch": epoch,
            "tr_loss": round(tr_loss, 4),
            "va_loss": round(va_loss, 4),
            "va_PDR": val_m["PDR"],
            "va_R2":  val_m["R2"],
            "va_CBR": val_m["CBR"],
            "va_RMSE": val_m["RMSE"],
            "is_best": is_best,
            "stopped": stopped,
        })

        if stopped:
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model, epoch_log

# ---------------------------------------------------------------------------
# 단일 run 실행
# ---------------------------------------------------------------------------

def run_one(cond_a, cond_b, cond_c, seed, device, datasets):
    """하나의 (A, B, C, seed) 조합을 실행."""
    clip_map = {"C1_unclipped": None, "C2_clip125": 125, "C3_clip100": 100}
    clip_val = clip_map[cond_c]

    X, y, units, sensor_cols, c_train, X_te, y_te = datasets[cond_c]

    set_seed(seed)

    # Factor A: val split
    if cond_a == "A1_fixed":
        tr_m, va_m = split_engines(units, val_frac=0.2, seed=42)
    elif cond_a == "A2_per_seed":
        tr_m, va_m = split_engines(units, val_frac=0.2, seed=seed)
    else:  # A3_stratified
        raw_train = load_cmapss(DATASET, "train")
        max_cyc = raw_train.groupby("unit_number")["cycle"].max()
        raw_train = raw_train.copy()
        raw_train["RUL"] = raw_train["unit_number"].map(max_cyc) - raw_train["cycle"]
        tr_m, va_m = stratified_split(units, raw_train, val_frac=0.2, seed=seed)

    X_tr, y_tr = X[tr_m], y[tr_m]
    X_va, y_va = X[va_m], y[va_m]

    tr_loader = make_loader(X_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X_va, y_va, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMBranch(len(sensor_cols)).to(device)

    # Factor B: early stopping
    if cond_b == "B1_off":
        min_ep, patience_eff = MAX_EPOCHS, MAX_EPOCHS + 1  # never stop
    elif cond_b == "B2_from0":
        min_ep, patience_eff = 0, PATIENCE
    else:  # B3_warmup
        min_ep, patience_eff = MIN_EPOCHS, PATIENCE

    t0 = time.time()
    model, epoch_log = train_with_log(
        model, tr_loader, va_loader, X_va, y_va, c_train,
        device, MAX_EPOCHS, patience_eff, min_ep
    )
    elapsed = time.time() - t0

    stop_ep = next((r["epoch"] for r in reversed(epoch_log) if r["stopped"]), len(epoch_log))

    # Test 평가
    preds = predict_sequences(model, X_te, device)
    test_preds_clipped = np.clip(preds, 0, clip_val) if clip_val else preds
    rmse, ns = compute_metrics(test_preds_clipped, y_te)
    m_test = mpc_metrics(test_preds_clipped, y_te, c_train)

    # val MPC (최종 best 체크포인트 기준)
    val_preds_final = predict_sequences(model, X_va, device)
    m_val = mpc_metrics(val_preds_final, y_va, c_train)

    collapsed = is_mpc(m_test)

    return {
        "cond_a": cond_a, "cond_b": cond_b, "cond_c": cond_c,
        "seed": seed, "clip": str(clip_val),
        "stop_epoch": stop_ep,
        "RMSE": round(rmse, 4), "NASA": round(ns, 2),
        "test_PDR": m_test["PDR"], "test_R2": m_test["R2"],
        "test_CBR": m_test["CBR"], "test_RMSE_const": m_test["RMSE_const"],
        "val_PDR": m_val["PDR"], "val_R2": m_val["R2"],
        "val_CBR": m_val["CBR"], "val_RMSE": m_val["RMSE"],
        "elapsed_s": round(elapsed, 1),
        "is_collapsed": collapsed,
    }, epoch_log


# ---------------------------------------------------------------------------
# 메인 실험 루프
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 데이터셋 사전 로드 (clip level별)
    print("데이터 로딩 중...")
    datasets = {}
    for cond_c in FACTOR_C:
        clip_map = {"C1_unclipped": None, "C2_clip125": 125, "C3_clip100": 100}
        datasets[cond_c] = build_dataset(clip_map[cond_c])
        clip_val = clip_map[cond_c]
        _, y, _, _, c_train, _, y_te = datasets[cond_c]
        print(f"  {cond_c}: train RUL mean={c_train:.1f}  "
              f"train RUL max={float(np.max(y)):.0f}  "
              f"test RUL mean={float(np.mean(y_te)):.1f}")

    # 이미 완료된 runs 로드 (재시작 지원)
    out_csv = PHASE1A_DIR / "runs.csv"
    if out_csv.exists():
        df_done = pd.read_csv(out_csv)
        done_keys = set(
            zip(df_done["cond_a"], df_done["cond_b"],
                df_done["cond_c"], df_done["seed"])
        )
        all_rows = df_done.to_dict("records")
        print(f"\n재시작: 기존 {len(df_done)}개 run 로드됨")
    else:
        done_keys = set()
        all_rows  = []

    total_runs = len(ALL_CONDITIONS) * len(SEEDS)
    completed  = len(done_keys)
    print(f"\n실험 시작: {total_runs - completed}개 runs 남음 "
          f"(총 {total_runs}, 완료 {completed})\n")

    t_global = time.time()

    for i, (cond_a, cond_b, cond_c) in enumerate(ALL_CONDITIONS):
        cond_label = f"{cond_a}|{cond_b}|{cond_c}"
        cond_rows  = []

        for seed in SEEDS:
            key = (cond_a, cond_b, cond_c, seed)
            if key in done_keys:
                continue

            run_id = completed + 1
            print(f"  [{run_id:>3}/{total_runs}] {cond_label} seed={seed}", end=" ... ", flush=True)

            try:
                row, epoch_log = run_one(cond_a, cond_b, cond_c, seed, device, datasets)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            # epoch log 저장
            log_name = f"{cond_a}_{cond_b}_{cond_c}_seed{seed}.csv"
            pd.DataFrame(epoch_log).to_csv(LOG_DIR / log_name, index=False)

            all_rows.append(row)
            cond_rows.append(row)
            done_keys.add(key)
            completed += 1

            collapsed_str = "COLLAPSED" if row["is_collapsed"] else "ok"
            print(f"ep={row['stop_epoch']:>3}  RMSE={row['RMSE']:.2f}"
                  f"  PDR={row['test_PDR']:.4f}  R²={row['test_R2']:.3f}"
                  f"  [{collapsed_str}]  ({row['elapsed_s']:.1f}s)")

        # 조건 완료마다 중간 저장
        if cond_rows:
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)

        # 조건별 소결
        cond_done = [r for r in all_rows
                     if r["cond_a"] == cond_a
                     and r["cond_b"] == cond_b
                     and r["cond_c"] == cond_c]
        if cond_done:
            df_c = pd.DataFrame(cond_done)
            n_col = int(df_c["is_collapsed"].sum())
            print(f"  └─ {cond_label}: MPC {n_col}/{len(df_c)} seeds  "
                  f"RMSE={df_c['RMSE'].mean():.2f}±{df_c['RMSE'].std():.2f}\n")

    # ---------------------------------------------------------------------------
    # 집계 및 요약
    # ---------------------------------------------------------------------------
    df = pd.read_csv(out_csv)

    print("\n" + "=" * 70)
    print("Phase 1A 집계 결과")
    print("=" * 70)

    # 조건별 MPC 발생률
    summary = (
        df.groupby(["cond_a", "cond_b", "cond_c"])
        .agg(
            n_runs=("seed", "count"),
            n_collapsed=("is_collapsed", "sum"),
            mpc_rate=("is_collapsed", "mean"),
            rmse_mean=("RMSE", "mean"),
            rmse_std=("RMSE", "std"),
            pdr_mean=("test_PDR", "mean"),
            r2_mean=("test_R2", "mean"),
            stop_ep_mean=("stop_epoch", "mean"),
        )
        .reset_index()
    )
    summary["mpc_rate"] = summary["mpc_rate"].round(3)
    summary["rmse_mean"] = summary["rmse_mean"].round(3)
    summary["rmse_std"]  = summary["rmse_std"].round(3)

    summary_csv = PHASE1A_DIR / "mpc_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\n[저장] {summary_csv.relative_to(_CS_ROOT)}")

    # 상위/하위 조건 출력
    print("\n--- MPC 발생률 상위 5 조건 ---")
    top5 = summary.nlargest(5, "mpc_rate")[
        ["cond_a", "cond_b", "cond_c", "mpc_rate", "rmse_mean", "rmse_std"]
    ]
    print(top5.to_string(index=False))

    print("\n--- MPC 발생률 하위 5 조건 ---")
    bot5 = summary.nsmallest(5, "mpc_rate")[
        ["cond_a", "cond_b", "cond_c", "mpc_rate", "rmse_mean", "rmse_std"]
    ]
    print(bot5.to_string(index=False))

    # 요인별 MPC 발생률 (한계 평균)
    print("\n--- 요인별 한계 MPC 발생률 ---")
    for factor, col in [("A (val split)", "cond_a"),
                         ("B (early stop)", "cond_b"),
                         ("C (labeling)", "cond_c")]:
        mg = df.groupby(col)["is_collapsed"].mean().round(3)
        print(f"  {factor}:")
        for lv, rate in mg.items():
            print(f"    {lv}: {rate:.3f}")

    # Silence 측정 (val_loss 정상 발화이지만 MPC인 케이스)
    # val PDR < 0.05 AND model stopped (early stopping 발화) AND is_collapsed
    silence_cond = df[(df["is_collapsed"]) & (df["cond_b"] != "B1_off")]
    print(f"\n--- 'Silence' 케이스 (ES 발화 + MPC) ---")
    print(f"  {len(silence_cond)} / {len(df[df['cond_b'] != 'B1_off'])} runs")

    # 총 소요 시간
    total_s = time.time() - t_global
    print(f"\n총 실험 시간: {total_s/60:.1f}분")
    print(f"결과 위치: {PHASE1A_DIR.relative_to(_CS_ROOT)}")

    # 시각화
    try:
        generate_figures(df, summary)
    except Exception as e:
        print(f"\n[경고] 시각화 실패 (결과 CSV는 저장됨): {e}")


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def generate_figures(df: pd.DataFrame, summary: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Figure 1: MPC 발생률 heatmap (A×B, C별 패널)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    a_labels = ["A1\nfixed", "A2\nper-seed", "A3\nstratified"]
    b_labels = ["B1\nES off", "B2\nES ep0", "B3\nwarmup"]

    for ci, cond_c in enumerate(FACTOR_C):
        ax = axes[ci]
        matrix = np.full((3, 3), np.nan)
        for ai, ca in enumerate(FACTOR_A):
            for bi, cb in enumerate(FACTOR_B):
                sub = summary[
                    (summary["cond_a"] == ca) &
                    (summary["cond_b"] == cb) &
                    (summary["cond_c"] == cond_c)
                ]
                if len(sub) > 0:
                    matrix[bi, ai] = float(sub["mpc_rate"].values[0])

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(a_labels, fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(b_labels, fontsize=9)
        ax.set_title(f"{cond_c}", fontsize=10)
        ax.set_xlabel("Factor A (val split)", fontsize=8)
        if ci == 0:
            ax.set_ylabel("Factor B (early stop)", fontsize=8)
        for bi in range(3):
            for ai in range(3):
                v = matrix[bi, ai]
                if not np.isnan(v):
                    ax.text(ai, bi, f"{v:.2f}", ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if v > 0.5 else "black")

    fig.colorbar(im, ax=axes[-1], label="MPC rate", shrink=0.8)
    fig.suptitle("Phase 1A: MPC Occurrence Rate (FD003, 10 seeds/condition)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_mpc_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[시각화] fig1_mpc_heatmap.png 저장")

    # Figure 2: 요인별 한계 MPC 발생률 막대 그래프
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    factor_info = [
        ("Factor A — Validation Split", "cond_a",
         {"A1_fixed": "A1\nfixed", "A2_per_seed": "A2\nper-seed", "A3_stratified": "A3\nstratified"}),
        ("Factor B — Early Stopping", "cond_b",
         {"B1_off": "B1\nES off", "B2_from0": "B2\nES ep0", "B3_warmup": "B3\nwarmup"}),
        ("Factor C — RUL Labeling", "cond_c",
         {"C1_unclipped": "C1\nunclipped", "C2_clip125": "C2\nclip=125", "C3_clip100": "C3\nclip=100"}),
    ]
    for ax, (title, col, label_map) in zip(axes, factor_info):
        mg = df.groupby(col)["is_collapsed"].mean()
        levels = list(label_map.keys())
        rates  = [float(mg.get(lv, 0)) for lv in levels]
        colors = ["#d62728" if r > 0.5 else "#2ca02c" if r < 0.1 else "#ff7f0e"
                  for r in rates]
        ax.bar(range(len(levels)), rates, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([label_map[lv] for lv in levels], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("MPC rate" if ax == axes[0] else "")
        ax.set_title(title, fontsize=10)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        for xi, r in enumerate(rates):
            ax.text(xi, r + 0.03, f"{r:.2f}", ha="center", fontsize=10)
    fig.suptitle("Phase 1A: Marginal MPC Rate by Factor Level",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_marginal_mpc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[시각화] fig2_marginal_mpc.png 저장")

    # Figure 3: RMSE 분포 violin (A×B 상호작용, C2_clip125 고정)
    df_c2 = df[df["cond_c"] == "C2_clip125"].copy()
    df_c2["condition"] = df_c2["cond_a"].str.replace("_", "\n") + " × " + df_c2["cond_b"].str.replace("_", "\n")
    conds_sorted = sorted(df_c2["condition"].unique())

    fig, ax = plt.subplots(figsize=(14, 5))
    data_by_cond = [df_c2[df_c2["condition"] == c]["RMSE"].values for c in conds_sorted]
    parts = ax.violinplot(data_by_cond, positions=range(len(conds_sorted)),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)

    # 붕괴 run 점 표시
    for xi, c in enumerate(conds_sorted):
        sub = df_c2[df_c2["condition"] == c]
        collapsed = sub[sub["is_collapsed"]]["RMSE"].values
        normal    = sub[~sub["is_collapsed"]]["RMSE"].values
        if len(collapsed): ax.scatter([xi]*len(collapsed), collapsed, color="red",
                                       zorder=5, s=30, label="collapsed" if xi == 0 else "")
        if len(normal):    ax.scatter([xi]*len(normal),    normal,    color="steelblue",
                                       zorder=5, s=20, alpha=0.5, label="normal" if xi == 0 else "")

    ax.set_xticks(range(len(conds_sorted)))
    ax.set_xticklabels(conds_sorted, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Test RMSE"); ax.set_ylim(0, None)
    ax.set_title("Phase 1A: RMSE Distribution by Condition (C2_clip125)", fontsize=11)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_rmse_violin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[시각화] fig3_rmse_violin.png 저장")


if __name__ == "__main__":
    main()
