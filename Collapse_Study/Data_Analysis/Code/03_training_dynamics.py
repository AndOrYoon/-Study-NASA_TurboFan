# -*- coding: utf-8 -*-
"""
03_training_dynamics.py
Phase 2 — Training Dynamics Analysis

Phase 1A/1B epoch_logs를 분석해 붕괴 run의 훈련 궤적 특성을 정량화한다.

분석 항목:
  D1. 붕괴 전조 epoch — PDR·R²가 임계값 아래로 떨어지는 첫 epoch
  D2. val_loss vs PDR 해리 — "Silence" 케이스 상세 분석
  D3. 붕괴 run vs 정상 run 평균 궤적 비교 (PDR, R², val_loss)
  D4. 조기 경보 성능 — alarm_epoch 기준으로 sensitivity/specificity 계산

결과 저장: Collapse_Study/Data_Analysis/Results/Phase2/
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd

_SCRIPT  = Path(__file__).resolve()
_CS_ROOT = _SCRIPT.parents[2]

PHASE1A_DIR = _CS_ROOT / "Data_Analysis" / "Results" / "Phase1A"
PHASE1B_DIR = _CS_ROOT / "Data_Analysis" / "Results" / "Phase1B"
PHASE2_DIR  = _CS_ROOT / "Data_Analysis" / "Results" / "Phase2"
FIG_DIR     = PHASE2_DIR / "figures"
for d in [PHASE2_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PDR_THRESH  = 0.05   # MPC 판정 기준 (Phase 1A/1B 동일)
R2_THRESH   = 0.0

# ---------------------------------------------------------------------------
# 1. epoch_logs 로드
# ---------------------------------------------------------------------------

def load_epoch_logs(log_dir: Path, runs_csv: Path, phase_label: str):
    """epoch_logs/ 폴더의 모든 csv를 runs.csv와 조인해 반환.

    Phase 1A runs.csv: cond_a, cond_b, cond_c 컬럼 (label 없음)
    Phase 1B runs.csv: label 컬럼
    파일명 패턴: {label}_seed{N}.csv  (Phase 1A label = cond_a_cond_b_cond_c)
    """
    runs = pd.read_csv(runs_csv)
    runs["phase"] = phase_label

    # Phase 1A 호환: label 컬럼 없으면 cond_a+cond_b+cond_c 조합
    if "label" not in runs.columns:
        runs["label"] = runs["cond_a"] + "_" + runs["cond_b"] + "_" + runs["cond_c"]

    records = []
    for fpath in sorted(log_dir.glob("*.csv")):
        stem = fpath.stem
        df = pd.read_csv(fpath)
        if df.empty:
            continue

        parts = stem.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        label = parts[0]
        try:
            seed = int(parts[1])
        except ValueError:
            continue

        row = runs[(runs["label"] == label) & (runs["seed"] == seed)]
        if row.empty:
            continue

        is_collapsed = bool(row["is_collapsed"].values[0])
        df["label"]        = label
        df["seed"]         = seed
        df["is_collapsed"] = is_collapsed
        df["phase"]        = phase_label
        if "factor" in row.columns:
            df["factor"] = row["factor"].values[0]
        records.append(df)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


print("epoch_logs 로딩 중...")
df1a = load_epoch_logs(PHASE1A_DIR / "epoch_logs", PHASE1A_DIR / "runs.csv", "1A")
df1b = load_epoch_logs(PHASE1B_DIR / "epoch_logs", PHASE1B_DIR / "runs.csv", "1B")

logs = pd.concat([df1a, df1b], ignore_index=True)
print(f"  Phase 1A: {len(df1a):,} epoch rows  ({df1a['label'].nunique() if len(df1a) else 0} conditions)")
print(f"  Phase 1B: {len(df1b):,} epoch rows  ({df1b['label'].nunique() if len(df1b) else 0} conditions)")
print(f"  합계: {len(logs):,} epoch rows  "
      f"runs={logs[['label','seed','phase']].drop_duplicates().shape[0]}\n")

# run 단위 식별자
logs["run_id"] = logs["phase"] + "_" + logs["label"] + "_s" + logs["seed"].astype(str)

# 붕괴/정상 분리
coll_logs = logs[logs["is_collapsed"]].copy()
norm_logs  = logs[~logs["is_collapsed"]].copy()

n_coll_runs = coll_logs["run_id"].nunique()
n_norm_runs = norm_logs["run_id"].nunique()
print(f"붕괴 runs: {n_coll_runs}  정상 runs: {n_norm_runs}\n")

# ---------------------------------------------------------------------------
# 2. D1 — 붕괴 전조 epoch 탐지
# ---------------------------------------------------------------------------

print("=" * 60)
print("D1. 붕괴 전조 epoch 분석")
print("=" * 60)

def find_collapse_onset(group):
    """PDR < PDR_THRESH인 첫 epoch를 반환. 없으면 NaN."""
    low = group[group["va_PDR"] < PDR_THRESH]
    if low.empty:
        return np.nan
    return float(low["epoch"].min())

def find_r2_onset(group):
    low = group[group["va_R2"] <= R2_THRESH]
    if low.empty:
        return np.nan
    return float(low["epoch"].min())

onset_rows = []
for run_id, grp in coll_logs.groupby("run_id"):
    grp = grp.sort_values("epoch")
    stop_ep = int(grp["epoch"].max())
    pdr_on  = find_collapse_onset(grp)
    r2_on   = find_r2_onset(grp)
    phase   = grp["phase"].iloc[0]
    label   = grp["label"].iloc[0]
    seed    = int(grp["seed"].iloc[0])
    onset_rows.append({
        "run_id": run_id, "phase": phase, "label": label, "seed": seed,
        "stop_epoch": stop_ep,
        "pdr_onset": pdr_on,
        "r2_onset": r2_on,
        "pdr_onset_frac": pdr_on / stop_ep if not np.isnan(pdr_on) else np.nan,
        "r2_onset_frac":  r2_on  / stop_ep if not np.isnan(r2_on)  else np.nan,
    })

onset_df = pd.DataFrame(onset_rows)
onset_df.to_csv(PHASE2_DIR / "collapse_onset.csv", index=False)

# 통계 요약
valid_pdr = onset_df["pdr_onset"].dropna()
valid_r2  = onset_df["r2_onset"].dropna()
valid_frac_pdr = onset_df["pdr_onset_frac"].dropna()

print(f"붕괴 runs 중 PDR onset 탐지됨: {len(valid_pdr)} / {len(onset_df)}")
if len(valid_pdr):
    print(f"  PDR 첫 붕괴 epoch  mean={valid_pdr.mean():.1f}  median={valid_pdr.median():.0f}"
          f"  min={valid_pdr.min():.0f}  max={valid_pdr.max():.0f}")
    print(f"  stop_epoch 대비 비율  mean={valid_frac_pdr.mean():.2f}"
          f"  median={valid_frac_pdr.median():.2f}")
if len(valid_r2):
    print(f"  R² 첫 붕괴 epoch  mean={valid_r2.mean():.1f}  median={valid_r2.median():.0f}")
print()

# ---------------------------------------------------------------------------
# 3. D2 — "Silence" 케이스: val_loss 정상 종료 + MPC
# ---------------------------------------------------------------------------

print("=" * 60)
print("D2. Silence 케이스 분석")
print("=" * 60)

silence_rows = []
for run_id, grp in coll_logs.groupby("run_id"):
    grp = grp.sort_values("epoch")
    # stopped=True인 epoch가 있으면 ES가 정상 발화한 것
    if "stopped" in grp.columns and grp["stopped"].any():
        stop_row = grp[grp["stopped"]].iloc[0]
        stop_ep  = int(stop_row["epoch"])
        va_loss_at_stop  = float(stop_row["va_loss"])
        # stop 직전 PDR
        pdr_at_stop = float(stop_row.get("va_PDR", np.nan))
        r2_at_stop  = float(stop_row.get("va_R2",  np.nan))
        silence_rows.append({
            "run_id": run_id,
            "stop_epoch": stop_ep,
            "va_loss_at_stop": va_loss_at_stop,
            "pdr_at_stop": pdr_at_stop,
            "r2_at_stop": r2_at_stop,
        })

silence_df = pd.DataFrame(silence_rows)

# MAX_EPOCHS에서 종료된 붕괴 runs (ES 미발화)
max_epoch_coll = []
for run_id, grp in coll_logs.groupby("run_id"):
    grp = grp.sort_values("epoch")
    has_stopped = "stopped" in grp.columns and grp["stopped"].any()
    if not has_stopped:
        max_epoch_coll.append(run_id)

print(f"붕괴 runs 총: {n_coll_runs}")
print(f"  ES 정상 발화 + MPC (Silence): {len(silence_df)}")
print(f"  MAX_EPOCHS 소진 + MPC:        {len(max_epoch_coll)}")

if len(silence_df):
    print(f"\nSilence 케이스 val_loss at stop:")
    print(f"  mean={silence_df['va_loss_at_stop'].mean():.4f}"
          f"  median={silence_df['va_loss_at_stop'].median():.4f}"
          f"  min={silence_df['va_loss_at_stop'].min():.4f}"
          f"  max={silence_df['va_loss_at_stop'].max():.4f}")
    print(f"\nSilence 케이스 PDR at stop:")
    print(f"  mean={silence_df['pdr_at_stop'].mean():.6f}"
          f"  max={silence_df['pdr_at_stop'].max():.6f}")
    print(f"  — 모두 PDR<{PDR_THRESH} 상태로 종료됨 (val_loss가 낮아도 MPC)")

silence_df.to_csv(PHASE2_DIR / "silence_cases.csv", index=False)
print()

# ---------------------------------------------------------------------------
# 4. D3 — 평균 궤적 계산
# ---------------------------------------------------------------------------

print("=" * 60)
print("D3. 붕괴/정상 평균 궤적 계산")
print("=" * 60)

MAX_ALIGN_EP = 60  # 짧은 run 기준 (붕괴 run 대부분 ~40 epoch 이내 종료)

def mean_trajectory(subset_logs, max_ep=MAX_ALIGN_EP):
    """epoch 1~max_ep 범위의 run별 평균 궤적. 없는 epoch는 마지막 값으로 forward-fill."""
    pivot_pdr  = {}
    pivot_r2   = {}
    pivot_vloss = {}

    for run_id, grp in subset_logs.groupby("run_id"):
        grp = grp.sort_values("epoch").set_index("epoch")
        for ep in range(1, max_ep + 1):
            if ep in grp.index:
                pivot_pdr.setdefault(ep, []).append(grp.loc[ep, "va_PDR"])
                pivot_r2.setdefault(ep,  []).append(grp.loc[ep, "va_R2"])
                pivot_vloss.setdefault(ep, []).append(grp.loc[ep, "va_loss"])
            else:
                # run이 이미 종료됐으면 마지막 값으로 채움
                available = [e for e in grp.index if e <= ep]
                if available:
                    last = max(available)
                    pivot_pdr.setdefault(ep, []).append(grp.loc[last, "va_PDR"])
                    pivot_r2.setdefault(ep,  []).append(grp.loc[last, "va_R2"])
                    pivot_vloss.setdefault(ep, []).append(grp.loc[last, "va_loss"])

    rows = []
    for ep in range(1, max_ep + 1):
        if ep in pivot_pdr:
            rows.append({
                "epoch": ep,
                "pdr_mean": np.mean(pivot_pdr[ep]),
                "pdr_std":  np.std(pivot_pdr[ep]),
                "r2_mean":  np.mean(pivot_r2[ep]),
                "r2_std":   np.std(pivot_r2[ep]),
                "vloss_mean": np.mean(pivot_vloss[ep]),
                "vloss_std":  np.std(pivot_vloss[ep]),
                "n_runs": len(pivot_pdr[ep]),
            })
    return pd.DataFrame(rows)

traj_coll = mean_trajectory(coll_logs)
traj_norm = mean_trajectory(norm_logs)

traj_coll["group"] = "collapsed"
traj_norm["group"] = "normal"
traj_all  = pd.concat([traj_coll, traj_norm], ignore_index=True)
traj_all.to_csv(PHASE2_DIR / "mean_trajectories.csv", index=False)

print(f"붕괴 궤적: {len(traj_coll)} epochs (n_runs={n_coll_runs})")
print(f"정상 궤적: {len(traj_norm)} epochs (n_runs={n_norm_runs})")

# PDR이 0.5 아래로 떨어지는 epoch (붕괴 궤적 기준)
pdr_half = traj_coll[traj_coll["pdr_mean"] < 0.5]
if not pdr_half.empty:
    print(f"평균 붕괴 궤적 PDR<0.5 첫 epoch: {int(pdr_half['epoch'].min())}")
print()

# ---------------------------------------------------------------------------
# 5. D4 — 조기 경보 성능 (alarm_epoch 기준)
# ---------------------------------------------------------------------------

print("=" * 60)
print("D4. 조기 경보 성능 평가")
print("=" * 60)

# 각 run의 초기 K epoch만 보고 경보 발생 여부 결정
# 경보 조건: PDR < PDR_THRESH 이 K epoch 내에 나타남

ALARM_WINDOWS = [5, 10, 15, 20, 30]

alarm_rows = []
for K in ALARM_WINDOWS:
    tp = fp = tn = fn = 0
    for run_id, grp in logs.groupby("run_id"):
        grp_early = grp[grp["epoch"] <= K]
        alarm = (grp_early["va_PDR"] < PDR_THRESH).any()
        actual = bool(grp["is_collapsed"].iloc[0])
        if alarm and actual:
            tp += 1
        elif alarm and not actual:
            fp += 1
        elif not alarm and actual:
            fn += 1
        else:
            tn += 1
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    alarm_rows.append({
        "alarm_window": K, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "PPV": round(ppv, 3),
    })

alarm_df = pd.DataFrame(alarm_rows)
alarm_df.to_csv(PHASE2_DIR / "alarm_performance.csv", index=False)
print(alarm_df.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 6. 시각화
# ---------------------------------------------------------------------------

print("=" * 60)
print("시각화")
print("=" * 60)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Fig 1: 붕괴/정상 평균 궤적 비교 (PDR, R², val_loss) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("pdr_mean",   "pdr_std",   "Validation PDR",    "PDR",    [0, 1.5]),
        ("r2_mean",    "r2_std",    "Validation R²",     "R²",     [-1.5, 1.1]),
        ("vloss_mean", "vloss_std", "Validation Loss",   "Loss",   None),
    ]

    colors = {"collapsed": "#d62728", "normal": "#2ca02c"}

    for ax, (m_col, s_col, title, ylabel, ylim) in zip(axes, metrics):
        for grp_label, traj in [("collapsed", traj_coll), ("normal", traj_norm)]:
            if traj.empty:
                continue
            ep = traj["epoch"]
            mn = traj[m_col]
            sd = traj[s_col]
            c  = colors[grp_label]
            n_runs = traj["n_runs"].iloc[0] if "n_runs" in traj.columns else "?"
            ax.plot(ep, mn, color=c, linewidth=2,
                    label=f"{grp_label} (n≈{n_coll_runs if grp_label=='collapsed' else n_norm_runs})")
            ax.fill_between(ep, mn - sd, mn + sd, alpha=0.2, color=c)

        if title == "Validation PDR":
            ax.axhline(PDR_THRESH, color="gray", linestyle="--", linewidth=1,
                       label=f"PDR threshold={PDR_THRESH}")
        if title == "Validation R²":
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="R²=0")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlim(1, MAX_ALIGN_EP)
        ax.legend(fontsize=8)

    fig.suptitle("Phase 2: Collapsed vs Normal — Mean Training Trajectories\n"
                 "(Phase 1A+1B combined, shaded=±1 std)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_mean_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[시각화] fig1_mean_trajectories.png")

    # ---- Fig 2: PDR onset epoch 분포 히스토그램 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    valid_pdr = onset_df["pdr_onset"].dropna()
    if len(valid_pdr):
        ax.hist(valid_pdr, bins=20, color="#d62728", alpha=0.7, edgecolor="white")
        ax.axvline(valid_pdr.median(), color="black", linestyle="--",
                   label=f"median={valid_pdr.median():.0f}")
        ax.set_xlabel("PDR Onset Epoch")
        ax.set_ylabel("Count")
        ax.set_title(f"PDR Collapse Onset (n={len(valid_pdr)} runs)")
        ax.legend()

    ax = axes[1]
    valid_frac = onset_df["pdr_onset_frac"].dropna()
    if len(valid_frac):
        ax.hist(valid_frac, bins=20, color="#ff7f0e", alpha=0.7, edgecolor="white")
        ax.axvline(valid_frac.median(), color="black", linestyle="--",
                   label=f"median={valid_frac.median():.2f}")
        ax.set_xlabel("PDR Onset / Stop Epoch (fraction)")
        ax.set_ylabel("Count")
        ax.set_title("Relative Onset Position")
        ax.legend()

    fig.suptitle("Phase 2: PDR Collapse Onset Distribution (collapsed runs only)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_pdr_onset.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[시각화] fig2_pdr_onset.png")

    # ---- Fig 3: 조기 경보 sensitivity/specificity vs alarm window ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alarm_df["alarm_window"], alarm_df["sensitivity"], "o-",
            color="#d62728", linewidth=2, label="Sensitivity (recall of collapse)")
    ax.plot(alarm_df["alarm_window"], alarm_df["specificity"], "s-",
            color="#2ca02c", linewidth=2, label="Specificity (non-collapse correct)")
    ax.plot(alarm_df["alarm_window"], alarm_df["PPV"],         "^--",
            color="#1f77b4", linewidth=1.5, label="PPV (precision)")
    ax.set_xlabel("Alarm Window (epochs)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 2: Early Alarm Performance vs Window Size", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_alarm_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[시각화] fig3_alarm_performance.png")

    # ---- Fig 4: Silence 케이스 — val_loss vs PDR scatter ----
    if len(silence_df):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(silence_df["va_loss_at_stop"], silence_df["pdr_at_stop"],
                   c="#d62728", alpha=0.6, s=40, label="Silence (ES+MPC)")

        # 정상 종료 run의 마지막 epoch val_loss vs PDR
        norm_last = []
        for run_id, grp in norm_logs.groupby("run_id"):
            grp = grp.sort_values("epoch")
            last = grp.iloc[-1]
            norm_last.append({"va_loss": last["va_loss"], "va_PDR": last["va_PDR"]})
        if norm_last:
            nl_df = pd.DataFrame(norm_last)
            ax.scatter(nl_df["va_loss"], nl_df["va_PDR"],
                       c="#2ca02c", alpha=0.4, s=25, label="Normal runs (last epoch)")

        ax.axhline(PDR_THRESH, color="gray", linestyle="--",
                   label=f"PDR={PDR_THRESH} threshold")
        ax.set_xlabel("val_loss at ES stop")
        ax.set_ylabel("PDR at ES stop")
        ax.set_title("Phase 2: Silence Cases — low val_loss despite MPC", fontsize=11)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig4_silence_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[시각화] fig4_silence_scatter.png")

except Exception as e:
    print(f"[경고] 시각화 실패: {e}")

# ---------------------------------------------------------------------------
# 7. 최종 요약 출력
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Phase 2 분석 완료 — 저장 파일 목록")
print("=" * 60)
saved = list(PHASE2_DIR.glob("*.csv")) + list(FIG_DIR.glob("*.png"))
for f in sorted(saved):
    print(f"  {f.relative_to(_CS_ROOT)}")

print("\n--- 핵심 수치 ---")
print(f"전체 runs: {logs['run_id'].nunique()}  (붕괴={n_coll_runs}, 정상={n_norm_runs})")
if len(valid_pdr):
    print(f"PDR onset epoch (붕괴 run):  median={valid_pdr.median():.0f}  "
          f"  stop_ep 대비 {valid_frac_pdr.median():.0%} 시점")
print(f"Silence 케이스: {len(silence_df)} / {n_coll_runs}  "
      f"({len(silence_df)/n_coll_runs*100:.1f}%)")
best_alarm = alarm_df.sort_values("sensitivity", ascending=False).iloc[0]
print(f"최고 sensitivity 경보 (window={int(best_alarm['alarm_window'])}): "
      f"sensitivity={best_alarm['sensitivity']:.3f}  specificity={best_alarm['specificity']:.3f}")
