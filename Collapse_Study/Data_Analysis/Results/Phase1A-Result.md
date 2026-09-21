# Phase 1A Result
## Core Mechanism Factorial Screen — FD003 × LSTM

**실험일:** 2026-09-14  
**소요 시간:** 121분 (GPU CUDA)  
**총 runs:** 270 (3×3×3 factorial × 10 seeds)

---

## 실험 설계

| 요인 | 수준 |
|------|------|
| **A — Validation Composition** | A1: fixed split (seed=42) / A2: per-seed split / A3: stratified split |
| **B — Early Stopping** | B1: off (MAX=200) / B2: from epoch 0 / B3: warmup 30 epoch 후 |
| **C — RUL Labeling** | C1: unclipped / C2: clip=125 / C3: clip=100 |

- Dataset: FD003 (단일 운전조건, 2 결함모드)
- Architecture: LSTM backbone (LSTM1(64)→Drop→LSTM2(64)→Drop→FC)
- MAX_EPOCHS=200, MIN_EPOCHS=30(B3), PATIENCE=15
- MPC 판정: PDR < 0.05 AND R² ≤ 0

---

## 핵심 발견

### 1. B2 (warmup 없는 early stopping)는 MPC의 필요조건이지 충분조건이 아니다

**요인별 한계 MPC 발생률:**

| 요인 | 수준 | MPC 발생률 |
|------|------|-----------|
| **B — Early Stopping** | B1_off | **0.000** |
| | B2_from0 | **0.444** |
| | B3_warmup | **0.000** |
| A — Val Split | A1_fixed | 0.244 |
| | A2_per_seed | 0.078 |
| | A3_stratified | 0.122 |
| C — RUL Labeling | C1_unclipped | 0.200 |
| | C2_clip125 | 0.156 |
| | C3_clip100 | 0.089 |

**인과 구조 요약:**
- B2는 MPC가 발생할 수 있는 **위험 창(risk window)을 여는 필요조건**
- B2 없이는 90 runs(B1) + 90 runs(B3) 전부 MPC 없음 — **예외 없음**
- B2가 있어도 A·C 조합에 따라 MPC 발생률이 0~90%로 변동
- B1·B3는 A·C가 어떻든 MPC를 **완전 차단** (충분조건적 예방책)

### 2. A×B×C 상호작용 — 조건별 MPC 발생률 전체 피벗

| A (val split) | B (early stop) | C (labeling) | MPC rate | RMSE mean±std | PDR mean | R² mean | stop ep |
|---|---|---|---:|---|---:|---:|---:|
| A1_fixed | B1_off | C1_unclipped | 0.00 | 31.86±2.84 | 1.399 | 0.403 | 200.0 |
| A1_fixed | B1_off | C2_clip125 | 0.00 | 12.37±0.49 | 0.997 | 0.900 | 200.0 |
| A1_fixed | B1_off | C3_clip100 | 0.00 | 7.48±0.29 | 0.989 | 0.946 | 200.0 |
| **A1_fixed** | **B2_from0** | **C1_unclipped** | **0.90** | **56.14±6.85** | 0.149 | −0.864 | 21.6 |
| A1_fixed | B2_from0 | C2_clip125 | 0.80 | 35.96±11.83 | 0.199 | 0.075 | 38.7 |
| A1_fixed | B2_from0 | C3_clip100 | 0.50 | 21.00±13.01 | 0.490 | 0.429 | 64.8 |
| A1_fixed | B3_warmup | C1_unclipped | 0.00 | 37.47±4.44 | 1.480 | 0.170 | 68.4 |
| A1_fixed | B3_warmup | C2_clip125 | 0.00 | 12.42±1.04 | 1.002 | 0.899 | 108.1 |
| A1_fixed | B3_warmup | C3_clip100 | 0.00 | 8.25±1.20 | 0.977 | 0.933 | 110.4 |
| A2_per_seed | B1_off | C1_unclipped | 0.00 | 36.26±6.99 | 1.466 | 0.207 | 200.0 |
| A2_per_seed | B1_off | C2_clip125 | 0.00 | 12.46±0.75 | 1.011 | 0.898 | 200.0 |
| A2_per_seed | B1_off | C3_clip100 | 0.00 | 7.86±0.66 | 0.988 | 0.940 | 200.0 |
| A2_per_seed | B2_from0 | C1_unclipped | 0.40 | 49.22±12.63 | 0.891 | −0.498 | 53.4 |
| A2_per_seed | B2_from0 | C2_clip125 | 0.30 | 21.50±14.02 | 0.716 | 0.583 | 102.7 |
| **A2_per_seed** | **B2_from0** | **C3_clip100** | **0.00** | **8.43±0.75** | 0.982 | 0.931 | 122.6 |
| A2_per_seed | B3_warmup | C1_unclipped | 0.00 | 38.79±7.78 | 1.467 | 0.090 | 81.5 |
| A2_per_seed | B3_warmup | C2_clip125 | 0.00 | 12.73±0.54 | 1.019 | 0.894 | 133.4 |
| A2_per_seed | B3_warmup | C3_clip100 | 0.00 | 8.43±0.75 | 0.982 | 0.931 | 122.6 |
| A3_stratified | B1_off | C1_unclipped | 0.00 | 33.93±6.17 | 1.412 | 0.308 | 200.0 |
| A3_stratified | B1_off | C2_clip125 | 0.00 | 12.31±0.53 | 1.018 | 0.901 | 200.0 |
| A3_stratified | B1_off | C3_clip100 | 0.00 | 8.10±0.84 | 0.983 | 0.936 | 200.0 |
| A3_stratified | B2_from0 | C1_unclipped | 0.50 | 51.15±13.52 | 0.745 | −0.623 | 55.2 |
| A3_stratified | B2_from0 | C2_clip125 | 0.30 | 21.99±13.84 | 0.705 | 0.572 | 82.6 |
| A3_stratified | B2_from0 | C3_clip100 | 0.30 | 16.49±11.77 | 0.685 | 0.618 | 89.6 |
| A3_stratified | B3_warmup | C1_unclipped | 0.00 | 37.78±7.56 | 1.446 | 0.137 | 84.1 |
| A3_stratified | B3_warmup | C2_clip125 | 0.00 | 13.20±1.35 | 1.012 | 0.885 | 114.9 |
| A3_stratified | B3_warmup | C3_clip100 | 0.00 | 8.70±1.17 | 0.980 | 0.926 | 127.8 |

### 3. B2 내에서의 A×C 효과

B2 조건 하에서만 분리해 보면:

| A | C | MPC rate |
|---|---|---:|
| A1_fixed | C1_unclipped | **0.90** |
| A1_fixed | C2_clip125 | 0.80 |
| A1_fixed | C3_clip100 | 0.50 |
| A2_per_seed | C1_unclipped | 0.40 |
| A2_per_seed | C2_clip125 | 0.30 |
| **A2_per_seed** | **C3_clip100** | **0.00** |
| A3_stratified | C1_unclipped | 0.50 |
| A3_stratified | C2_clip125 | 0.30 |
| A3_stratified | C3_clip100 | 0.30 |

- A1(fixed split): B2 내 한계 MPC rate = **0.733** — 가장 위험
- A2(per-seed): B2 내 한계 MPC rate = **0.233** — A1 대비 3분의 1 수준
- A3(stratified): B2 내 한계 MPC rate = **0.367**

**A가 MPC를 증폭시키는 메커니즘:** fixed split은 모든 훈련 seed에 동일한 20개 엔진이 검증셋으로 고정 → trivial solution의 val_loss가 seed 간 안정적으로 낮게 유지 → early stopping이 trivial solution을 best checkpoint로 채택할 확률 증가. per-seed split은 검증셋이 seed마다 달라 trivial solution의 val_loss가 진동 → patience counter가 리셋될 기회 증가.

**C가 MPC를 증폭시키는 메커니즘:** unclipped(C1)는 train RUL mean ≈ 138로 높아 trivial solution(상수 ≈ 138 예측)의 MSE가 clip125(mean ≈ 93)보다 상대적으로 낮게 형성 → trivial solution이 더 안정적인 local minimum. clip이 강할수록 trivial solution의 MSE 우위가 줄어 plateau 탈출이 용이.

### 4. "Silence" 케이스

- ES가 정상 발화했음에도 MPC 상태인 runs: **40 / 180 (22.2%)**
- val_loss 곡선만으로는 MPC를 감지할 수 없음을 실증
- PDR·R² 등 추가 지표의 필요성 확인 (Phase 4 Detector 개발 근거)

---

## 통계 분석 현황 및 미완 항목

> ⚠️ **Research_Plan §7.1에서 계획된 mixed-effects logistic regression이 미실행 상태.**  
> 현재 보고된 수치는 raw proportion (n=10 seeds per condition)이며, 공식 통계 검정이 없다.  
> - "0% vs 80%" 비교는 n=10으로도 Fisher's exact test p < 0.001 — 결론에 영향 없음  
> - **"20% vs 30%" (patience=30 vs per-seed split) 비교는 n=10으로 통계적 구분 불가** → 논문에서 구체적 비교 삼가  
> - 향후 필요 분석: (1) 각 MPC rate의 95% Clopper-Pearson CI, (2) B2 조건 내 A×C 교호작용 검정, (3) 조건별 RMSE의 Wilcoxon rank-sum  

## 결론 및 Phase 1B 방향

### 논문에서 쓸 표현
> "B (early stopping warmup 부재)는 MPC의 필요조건이지만 충분조건이 아니다. A (val split)와 C (RUL labeling)는 B2 조건 하에서 MPC 발생 확률을 0~90% 범위에서 조절한다. 세 조건이 최악으로 결합될 때(fixed split + no warmup + unclipped) MPC 발생률은 90%이며, A·C만 교정해도(per-seed + clip100) B2 환경에서도 0%로 낮아진다."

### Phase 1B 실험 설계 근거
Phase 1A에서 B2 조건(warmup 없는 early stopping)이 MPC의 필요조건임이 확인됐다. Phase 1B는 **B2 조건을 고정하고** (MPC가 발생 가능한 환경 유지) patience·LR·bias init·loss의 효과를 분리한다.

- 고정 조건: **A1 + B2 + C2** (fixed split + no warmup + clip125, MPC rate=0.80)
  - MPC가 충분히 빈번해 효과 측정 가능
  - clip=125는 표준 프로토콜 — 실용적 의의 있음
- 탐색 요인: patience / learning rate / output bias init / loss function

---

## 저장 파일 목록

| 파일 | 내용 |
|------|------|
| `Phase1A/runs.csv` | 270 runs 전체 원데이터 |
| `Phase1A/mpc_summary.csv` | 27 조건별 집계 (MPC rate, RMSE, PDR, R²) |
| `Phase1A/epoch_logs/*.csv` | 270개 run별 epoch 궤적 |
| `Phase1A/figures/fig1_mpc_heatmap.png` | A×B MPC 발생률 heatmap (C별 패널) |
| `Phase1A/figures/fig2_marginal_mpc.png` | 요인별 한계 MPC 발생률 막대 그래프 |
| `Phase1A/figures/fig3_rmse_violin.png` | 조건별 RMSE 분포 violin (C2 고정) |
