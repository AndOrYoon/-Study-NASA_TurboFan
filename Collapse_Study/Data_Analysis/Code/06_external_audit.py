# -*- coding: utf-8 -*-
"""
06_external_audit.py
Phase 5 — External Protocol Audit

5편의 published CMAPSS LSTM 논문 프로토콜을 재현하여 MPC 발생 여부를 실험적으로 확인한다.
논문별 고유 patience/max_epochs 값을 사용하는 것이 Phase 1A와의 차별점.

Paper Protocols:
  P1 (Expert Systems 2023 대표) : A1+B2+C2, patience=15  → Phase1A 결과 참조
  P2 (EAAI 2024 대표)           : A1+B2+C2, patience=20  → 신규 실험
  P3 (Zheng et al. 2017)        : A1+B2+C2, patience=10  → 신규 실험
  P4 (Pre-2020 unclipped 대표)  : A1+B2+C1, patience=15  → Phase1A 결과 참조
  P5 (Elsherif et al. 2025)     : A1+B1+C2, max_epochs=25 → 신규 실험

신규 실험만 실행 (P2, P3, P5). P1/P4는 Phase1A CSV에서 필터링.
Seeds: 10 per condition (0–9)
결과: Collapse_Study/Data_Analysis/Results/Phase5/runs_new.csv
"""

import sys, os, time
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
    set_seed, load_cmapss, load_test_rul, add_rul, get_sensor_cols,
    fit_normalization, apply_normalization,
    make_train_sequences, make_test_sequences, split_engines,
    make_loader, LSTMBranch,
    predict_sequences, compute_metrics,
)

PHASE5_DIR   = _CS_ROOT / "Data_Analysis" / "Results" / "Phase5"
PHASE1A_CSV  = _CS_ROOT / "Data_Analysis" / "Results" / "Phase1A" / "runs.csv"
PHASE5_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 논문 프로토콜 정의
# ---------------------------------------------------------------------------

AUDIT_PAPERS = [
    {
        "paper_id": "P1",
        "short_name": "ESWA_2023",
        "citation": (
            "Meng et al. (2023). Bayesian gated-transformer model for risk-aware prediction "
            "of aero-engine remaining useful life. Expert Systems with Applications, Vol. 238, "
            "Article 121859. https://doi.org/10.1016/j.eswa.2023.121859"
        ),
        "journal": "Expert Systems with Applications",
        "journal_confirmed": True,
        "reported_fd003_rmse": None,
        "val_split": "A1_fixed",
        "es_mode": "B2_from0",
        "clip_key": "C2_clip125",
        "clip_val": 125,
        "patience": 15,
        "max_epochs": 200,
        "protocol_evidence": "INFERRED — standard CMAPSS practice in ESWA 2023",
        "source": "phase1a_reference",
    },
    {
        "paper_id": "P2",
        "short_name": "EAAI_2024",
        "citation": (
            "Qin et al. (2024). Spatial and temporal attention-based and residual-driven long "
            "short-term memory networks with implicit features for remaining useful life "
            "prediction. Engineering Applications of Artificial Intelligence, Vol. 133, "
            "Article 108563. https://doi.org/10.1016/j.engappai.2024.108563"
        ),
        "journal": "Engineering Applications of Artificial Intelligence",
        "journal_confirmed": True,
        "reported_fd003_rmse": 12.14,
        "val_split": "A1_fixed",
        "es_mode": "B2_from0",
        "clip_key": "C2_clip125",
        "clip_val": 125,
        "patience": 20,
        "max_epochs": 200,
        "protocol_evidence": "INFERRED — patience=20 common in EAAI 2022-2024 papers",
        "source": "new_experiment",
    },
    {
        "paper_id": "P3",
        "short_name": "Zheng_2017",
        "citation": (
            "Zheng S., Ristovski K., Farahat A., Gupta C. (2017). Long short-term memory "
            "network for remaining useful life estimation. IEEE Int. Conf. on Prognostics "
            "and Health Management (ICPHM), 2017."
        ),
        "journal": "IEEE ICPHM (conference)",
        "journal_confirmed": True,
        "reported_fd003_rmse": 16.18,
        "val_split": "A1_fixed",
        "es_mode": "B2_from0",
        "clip_key": "C2_clip125",
        "clip_val": 125,
        "patience": 10,
        "max_epochs": 200,
        "protocol_evidence": "INFERRED — patience=10 is conservative for 2017-era papers; "
                             "original paper describes early stopping without warmup",
        "source": "new_experiment",
    },
    {
        "paper_id": "P4",
        "short_name": "Unclipped_EAAI_pre2020",
        "citation": (
            "Representative of EAAI / Expert Systems with Applications papers (2018–2021) "
            "that apply standard LSTM without piecewise-linear RUL clipping. "
            "Common in papers following the protocol of Babu et al. (2016) or Zheng et al. (2017) "
            "before clip=125 became the de facto standard around 2020."
        ),
        "journal": "EAAI / Expert Systems with Applications (representative)",
        "journal_confirmed": False,
        "reported_fd003_rmse": None,
        "val_split": "A1_fixed",
        "es_mode": "B2_from0",
        "clip_key": "C1_unclipped",
        "clip_val": None,
        "patience": 15,
        "max_epochs": 200,
        "protocol_evidence": "COMMON PATTERN — early CMAPSS papers frequently omit clipping; "
                             "this condition gives highest Phase1A MPC rate (0.90)",
        "source": "phase1a_reference",
    },
    {
        "paper_id": "P5",
        "short_name": "Elsherif_2025",
        "citation": (
            "Elsherif S.M., et al. (2025). A deep learning-based prognostic approach for "
            "predicting turbofan engine degradation and remaining useful life. "
            "Scientific Reports 15, 12959. https://doi.org/10.1038/s41598-025-09155-z"
        ),
        "journal": "Scientific Reports",
        "journal_confirmed": True,
        "reported_fd003_rmse": 13.40,
        "val_split": "A1_fixed",
        "es_mode": "B1_short",
        "clip_key": "C2_clip125",
        "clip_val": 125,
        "patience": 999,
        "max_epochs": 25,
        "protocol_evidence": "DIRECT — paper states 25 training rounds for FD003 (batch=32, LR=0.0001)",
        "source": "new_experiment",
    },
]

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
DATASET    = "FD003"
SEEDS      = list(range(10))
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4
EPS        = 1e-8


def mpc_metrics(pred: np.ndarray, true: np.ndarray, c_train: float) -> dict:
    pdr  = float(np.std(pred) / (np.std(true) + EPS))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmse_const = float(np.sqrt(np.mean((true - c_train) ** 2)))
    cbr  = rmse / (rmse_const + EPS)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + EPS)
    return {"PDR": round(pdr, 6), "CBR": round(cbr, 6),
            "R2": round(r2, 6), "RMSE": round(rmse, 4),
            "RMSE_const": round(rmse_const, 4)}


def is_mpc(m: dict) -> bool:
    return m["PDR"] < 0.05 and m["R2"] <= 0.0


def build_dataset(clip_val):
    train_df    = load_cmapss(DATASET, "train")
    sensor_cols = get_sensor_cols(DATASET)
    max_cyc     = train_df.groupby("unit_number")["cycle"].max()
    train_df    = train_df.copy()
    train_df["RUL"] = train_df["unit_number"].map(max_cyc) - train_df["cycle"]
    if clip_val is not None:
        train_df["RUL"] = train_df["RUL"].clip(upper=clip_val)
    min_v, max_v = fit_normalization(train_df, sensor_cols)
    train_n = apply_normalization(train_df, sensor_cols, min_v, max_v)
    X, y, units = make_train_sequences(train_n, sensor_cols)
    c_train = float(train_df["RUL"].mean())
    test_df  = load_cmapss(DATASET, "test")
    test_rul = load_test_rul(DATASET)
    test_n   = apply_normalization(test_df, sensor_cols, min_v, max_v)
    X_te, y_te = make_test_sequences(test_n, sensor_cols, test_rul)
    if clip_val is not None:
        y_te = np.clip(y_te, 0, clip_val)
    return X, y, units, sensor_cols, c_train, X_te, y_te


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
            tr_n += len(xb)
        tr_loss = tr_total / tr_n

        model.eval()
        va_total, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_total += criterion(model(xb), yb).item() * len(xb)
                va_n += len(xb)
        va_loss = va_total / va_n

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
            "epoch": epoch, "tr_loss": round(tr_loss, 4), "va_loss": round(va_loss, 4),
            "va_PDR": val_m["PDR"], "va_R2": val_m["R2"], "va_RMSE": val_m["RMSE"],
            "is_best": is_best, "stopped": stopped,
        })

        if stopped:
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model, epoch_log


def run_one_paper(paper: dict, seed: int, device, datasets: dict) -> dict:
    clip_val   = paper["clip_val"]
    clip_key   = paper["clip_key"]
    data       = datasets[clip_key]
    X, y, units, sensor_cols, c_train, X_te, y_te = data

    set_seed(seed)

    # Validation split: A1 (fixed seed=42) for all audit papers
    tr_m, va_m = split_engines(units, val_frac=0.2, seed=42)
    X_tr, y_tr = X[tr_m], y[tr_m]
    X_va, y_va = X[va_m], y[va_m]

    tr_loader = make_loader(X_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X_va, y_va, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMBranch(len(sensor_cols)).to(device)

    max_epochs = paper["max_epochs"]
    patience   = paper["patience"]
    min_epochs = 0  # B2: no warmup

    if paper["es_mode"] == "B1_short":
        # Fixed epochs, no early stopping: run exactly max_epochs regardless
        patience = max_epochs + 1  # never triggers patience
        min_epochs = max_epochs    # ensure full run

    t0 = time.time()
    model, epoch_log = train_with_log(
        model, tr_loader, va_loader, X_va, y_va, c_train,
        device, max_epochs, patience, min_epochs
    )
    elapsed = time.time() - t0

    stop_ep = next((r["epoch"] for r in reversed(epoch_log) if r["stopped"]), len(epoch_log))

    preds = predict_sequences(model, X_te, device)
    if clip_val is not None:
        preds_ev = np.clip(preds, 0, clip_val)
    else:
        preds_ev = preds
    rmse, ns = compute_metrics(preds_ev, y_te)
    m_test = mpc_metrics(preds_ev, y_te, c_train)
    m_val  = mpc_metrics(predict_sequences(model, X_va, device), y_va, c_train)

    return {
        "paper_id": paper["paper_id"],
        "short_name": paper["short_name"],
        "seed": seed,
        "clip": str(clip_val),
        "patience": paper["patience"],
        "max_epochs": max_epochs,
        "stop_epoch": stop_ep,
        "RMSE": round(rmse, 4),
        "NASA": round(ns, 2),
        "test_PDR": m_test["PDR"], "test_R2": m_test["R2"],
        "test_CBR": m_test["CBR"], "test_RMSE_const": m_test["RMSE_const"],
        "val_PDR": m_val["PDR"], "val_R2": m_val["R2"],
        "val_RMSE": m_val["RMSE"],
        "elapsed_s": round(elapsed, 1),
        "is_collapsed": is_mpc(m_test),
    }


def load_phase1a_results(paper_id: str, cond_a: str, cond_b: str, cond_c: str) -> pd.DataFrame:
    """Phase1A CSV에서 특정 조건의 결과를 불러와 paper_id를 붙인다."""
    if not PHASE1A_CSV.exists():
        print(f"  [경고] Phase1A CSV 없음: {PHASE1A_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(PHASE1A_CSV)
    mask = (df["cond_a"] == cond_a) & (df["cond_b"] == cond_b) & (df["cond_c"] == cond_c)
    sub  = df[mask].copy()
    if sub.empty:
        print(f"  [경고] Phase1A에 {cond_a}+{cond_b}+{cond_c} 결과 없음")
        return pd.DataFrame()
    sub["paper_id"]   = paper_id
    sub["short_name"] = paper_id
    sub["patience"]   = 15  # Phase1A standard
    # 필요한 컬럼만 추출 (Phase1A 컬럼 이름 맞춤)
    rename = {}
    for col in ["test_PDR", "test_R2", "test_CBR", "test_RMSE_const",
                "val_PDR", "val_R2", "val_RMSE", "NASA", "stop_epoch", "elapsed_s"]:
        if col not in sub.columns:
            rename[col] = col
    keep_cols = [c for c in [
        "paper_id", "short_name", "seed", "clip", "patience", "max_epochs",
        "stop_epoch", "RMSE", "NASA",
        "test_PDR", "test_R2", "test_CBR", "test_RMSE_const",
        "val_PDR", "val_R2", "val_RMSE", "elapsed_s", "is_collapsed",
    ] if c in sub.columns or c in ["paper_id", "short_name", "patience"]]
    # Phase1A uses 'RMSE' column
    if "RMSE" not in sub.columns and "rmse" in sub.columns:
        sub = sub.rename(columns={"rmse": "RMSE"})
    if "is_collapsed" not in sub.columns:
        sub["is_collapsed"] = (sub["test_PDR"] < 0.05) & (sub["test_R2"] <= 0.0)
    if "max_epochs" not in sub.columns:
        sub["max_epochs"] = 200
    return sub


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 5 External Protocol Audit — Device: {device}\n")

    # 신규 실험 대상
    new_papers   = [p for p in AUDIT_PAPERS if p["source"] == "new_experiment"]
    ref_papers   = [p for p in AUDIT_PAPERS if p["source"] == "phase1a_reference"]
    print(f"신규 실험 논문: {[p['paper_id'] for p in new_papers]}")
    print(f"Phase1A 참조 논문: {[p['paper_id'] for p in ref_papers]}\n")

    # 신규 실험용 데이터 사전 로드
    clip_keys_needed = list({p["clip_key"] for p in new_papers})
    print("데이터 로딩 중...")
    datasets = {}
    for ck in clip_keys_needed:
        clip_map = {"C1_unclipped": None, "C2_clip125": 125, "C3_clip100": 100}
        cv = clip_map.get(ck)
        datasets[ck] = build_dataset(cv)
        _, y, _, _, c_train, _, _ = datasets[ck]
        print(f"  {ck}: train_RUL_mean={c_train:.1f}, max={float(np.max(y)):.0f}")

    # 기존 결과 로드
    out_csv = PHASE5_DIR / "runs_new.csv"
    if out_csv.exists():
        df_done = pd.read_csv(out_csv)
        done_keys = set(zip(df_done["paper_id"], df_done["seed"]))
        all_rows  = df_done.to_dict("records")
        print(f"\n재시작: 기존 {len(df_done)}개 결과 로드됨")
    else:
        done_keys = set()
        all_rows  = []

    total_new = len(new_papers) * len(SEEDS)
    done_new  = sum(1 for k in done_keys if k[0] in {p["paper_id"] for p in new_papers})
    print(f"신규 실험: {total_new - done_new}개 남음 (총 {total_new})\n")

    t_global = time.time()

    # 신규 실험 실행
    for paper in new_papers:
        for seed in SEEDS:
            key = (paper["paper_id"], seed)
            if key in done_keys:
                continue
            print(f"  {paper['paper_id']}({paper['short_name']}) seed={seed} "
                  f"patience={paper['patience']} max_ep={paper['max_epochs']}", end=" ... ", flush=True)
            try:
                row = run_one_paper(paper, seed, device, datasets)
                status = "COLLAPSED" if row["is_collapsed"] else f"RMSE={row['RMSE']:.2f}"
                print(f"{status}  stop@ep{row['stop_epoch']}  [{row['elapsed_s']:.0f}s]")
                all_rows.append(row)
                done_keys.add(key)
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

            # 중간 저장
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    total_elapsed = time.time() - t_global
    print(f"\n신규 실험 완료: {total_elapsed/60:.1f}분")

    # Phase1A 참조 결과 불러오기
    ref_rows_list = []
    for paper in ref_papers:
        cond_b_map = {"B2_from0": "B2_from0"}
        cond_a = "A1_fixed"
        cond_b = "B2_from0"
        cond_c = paper["clip_key"]
        df_ref = load_phase1a_results(paper["paper_id"], cond_a, cond_b, cond_c)
        if not df_ref.empty:
            ref_rows_list.append(df_ref)
            print(f"Phase1A 참조 로드: {paper['paper_id']} ({len(df_ref)} rows)")

    # 전체 결과 통합
    df_new = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    df_ref = pd.concat(ref_rows_list, ignore_index=True) if ref_rows_list else pd.DataFrame()

    if not df_ref.empty:
        # 컬럼 통일
        for col in df_new.columns:
            if col not in df_ref.columns:
                df_ref[col] = None
        df_ref = df_ref[[c for c in df_new.columns if c in df_ref.columns]]

    df_all = pd.concat([df_new, df_ref], ignore_index=True) if not df_ref.empty else df_new

    # 요약 집계
    if not df_all.empty and "paper_id" in df_all.columns:
        agg = df_all.groupby("paper_id").agg(
            n_seeds=("seed", "count"),
            mpc_count=("is_collapsed", "sum"),
            mpc_rate=("is_collapsed", "mean"),
            rmse_mean=("RMSE", "mean"),
            rmse_std=("RMSE", "std"),
            pdr_mean=("test_PDR", "mean"),
            r2_mean=("test_R2", "mean"),
            stop_ep_mean=("stop_epoch", "mean"),
        ).round(4).reset_index()
        summary_csv = PHASE5_DIR / "summary.csv"
        agg.to_csv(summary_csv, index=False)
        print(f"\n집계 결과:\n{agg.to_string(index=False)}")
        print(f"\n요약 저장: {summary_csv}")

    # 전체 runs 저장
    all_out_csv = PHASE5_DIR / "runs_all.csv"
    df_all.to_csv(all_out_csv, index=False)
    print(f"전체 runs 저장: {all_out_csv}")


if __name__ == "__main__":
    main()
