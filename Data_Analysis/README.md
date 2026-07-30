# Data_Analysis — 실험 코드 및 결과

가설 검증을 위한 전처리, 모델링, 평가 코드 및 실험 결과 보관 폴더.

**현재 상태: 전 가설(H2/H5/H6/H7) 실험 완료**

---

## 실험 결과 요약

| 가설 | 모델 | 실험 횟수 | 판정 | 핵심 결과 |
|------|------|---------|------|---------|
| H2 (RUL 클리핑) | Ridge regression | 20 runs (5 clips × 4 datasets) | **부분 채택** | clip=125 전 데이터셋 최적; clip=None → FD003 NASA 4,014,724 |
| H5 (정규화 전략) | LSTM compact (hidden=32) | 140 runs (7 × 4 × 5) | **기각** | Fleet MinMax(N1) 최우수; per-unit · RevIN 모두 열세 |
| H6 (고장 모드 분리) | LSTM full (hidden=64) | 40 runs + K sensitivity 25 runs | **채택 (M3)** | M3 FD003 RMSE −65.8% vs M0 (14.78±1.32 vs 43.23±0.18) |
| H7 (손실 함수) | LSTM full (hidden=64) | 560 runs (7 × 4 × 4 × 5) | **기각** | BH-FDR 96개 비교 모두 p_BH > 0.05; clip이 손실보다 지배적 |

---

## 핵심 수치

### H2 — RUL 클리핑 (Ridge regression, 20 runs)

| clip | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD003 NASA |
|------|-----------|-----------|-----------|-----------|
| 75   | 33.65 | 47.37 | 32.48 | 27.36 |
| 100  | 24.51 | 37.67 | 23.00 | 9.11 |
| **125** | **21.90** | **32.39** | **21.62** | **13.09** |
| 130  | 22.06 | 31.83 | 22.24 | 16.17 |
| None | 31.90 | 33.05 | 56.09 | **4,014,724** |

### H5 — 정규화 전략 (LSTM, 5 seeds)

| Normalizer | FD001 RMSE | FD002 RMSE | FD003 RMSE±std | FD004 RMSE |
|-----------|-----------|-----------|--------------|-----------|
| **N1 Fleet MinMax** | **14.14±0.22** | **14.31±0.10** | 19.05±12.86 | **14.60±0.30** |
| N2 Fleet Std | 14.38±0.81 | 14.83±0.55 | **14.69±0.71** | 15.33±0.12 |
| N3–N6 Per-unit | 17.90~20.78 | 15.52~18.49 | 17.35~21.52 | 16.49~18.76 |
| N7 RevIN | 14.92±0.48 | 18.16±0.27 | 16.44±0.36 | 18.28±0.66 |

FD003 N1 std=12.86 → 고장 모드 이종성 신호 (정규화 실패 아님)

### H6 — 고장 모드 분리 (LSTM, FD003/FD004, 5 seeds)

| Model | FD003 RMSE±std | FD003 NASA | FD004 RMSE±std | FD004 NASA |
|-------|--------------|-----------|--------------|-----------|
| M0 baseline | 43.23±0.18 | 34,339 | 28.05±1.74 | 10,586 |
| M1 Hard routing | 32.45±11.37 | 27,098 | 49.20±7.17 | 132,003 |
| M2 Soft gating | 26.16±14.81 | 20,881 | 30.71±0.73 | 50,715 |
| **M3 Attention Gate** | **14.78±1.32** | **425** | **28.33±1.03** | 13,229 |

K-sensitivity (K ∈ {5,10,15,20,30}): RMSE 범위 14.13–14.78 (K에 둔감), K=5로 충분

### H7 — 손실 함수 (LSTM, 5 seeds, 560 runs)

BH-FDR 보정 후 MSE(L1) 대비 통계적으로 유의한 개선 없음 (96개 비교 전부 p_BH > 0.05)  
clip=None 조건에서만 Pinball(L6, τ=0.25)이 FD003 NASA: 9,614,000 → 3,124 (3,000× 개선)  
→ 그래도 clip=125 최악 결과보다 630× 나쁨 → **클리핑이 손실 선택을 압도**

---

## 폴더 구조

```
Data_Analysis/
├── Analysis_Plan.md              ← 전체 실험 설계 문서
├── 주요이슈_및_의사결정.md        ← 확정된 설계 결정 (변경 전 반드시 확인)
├── Code/
│   ├── shared/
│   │   └── op_condition_utils.py  ← K-means residualization (H5/H6 공용)
│   ├── H2_clipping/               ← 01~05 스크립트
│   ├── H5_normalization/          ← 01~06 스크립트
│   ├── H6_fault_mode/
│   │   ├── phase1_clustering/     ← GMM 클러스터링
│   │   ├── phase2_models/         ← h6_p2_model_utils.py (M0~M3 backbone)
│   │   └── phase3_evaluation/     ← 비교 평가
│   └── H7_loss_function/          ← 00~08 스크립트 + run_all_h7.py
└── Results/
    ├── H2_clipping/               ← metrics_summary.csv, figures/, raw_predictions/
    ├── H5_normalization/          ← raw_predictions/ (FD001~FD004 × N1~N7 × seed0~4)
    ├── H6_fault_mode/             ← models/ (.pt/.pkl), figures/, cluster CSVs, k_sensitivity/
    └── H7_loss_function/          ← phase1 screening, phase2b results CSV
```

---

## 공통 실험 설정

| 항목 | 값 |
|------|---|
| 공통 백본 | Stacked LSTM (LSTM₁ 64 → LSTM₂ → FC → 1) |
| 윈도우 크기 | 30 cycles (zero-pad 앞쪽) |
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| Batch size | 256 |
| Early stopping | patience=15 (val loss) |
| Val split | Engine-level 20% holdout |
| Seeds | {0, 1, 2, 3, 4} |

**H5 compact backbone:** LSTM₂ hidden=32, FC 32→16→1  
**H6/H7 full backbone:** LSTM₂ hidden=64, FC 64→32→1

---

## 평가 지표

| 지표 | 수식 | 비고 |
|------|------|------|
| RMSE | √(mean((ŷ−y)²)) | 주 성능 지표 |
| NASA Score | mean(exp(d/10)−1 if d≥0 else exp(−d/13)−1) | 공식 경쟁 지표, 낮을수록 좋음 |

통계 검정: Wilcoxon rank-sum (one-sided, H5/H6/H7) / Mann-Whitney U (two-sided, H2) + BH-FDR (α=0.05)

---

*상세 실험 설계 및 설계 결정 근거: `Analysis_Plan.md`, `주요이슈_및_의사결정.md` 참고.*
