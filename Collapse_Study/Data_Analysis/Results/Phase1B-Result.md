# Phase 1B Result
## Optimization Follow-up — A1+B2+C2 고정, patience·LR·bias init·loss 탐색

**실험일:** 2026-09-14  
**소요 시간:** 2.6분 (GPU CUDA) — 이전 120 runs 재사용 + F_bias_random_calibrated 10 runs 추가  
**총 runs:** 130 (13 조건 × 10 seeds)

---

## 실험 설계

**고정 조건:** A1(fixed split seed=42) + B2(no warmup) + C2(clip=125) → 기준 MPC rate = **0.80**

Phase 1B는 B2 환경(MPC 발생 가능)에서 네 요인의 주효과를 OAT(one-at-a-time) 방식으로 분리한다.

| 요인 | 수준 | 기준값 |
|------|------|--------|
| **D — patience** | 5 / 10 / **15** / 30 | 15 |
| **E — learning rate** | 1e-4 / **1e-3** / 1e-2 | 1e-3 |
| **F — bias init** | **zero** / train_mean / random_calibrated | zero |
| **G — loss** | **MSE** / MAE / auxiliary(MSE+0.1×branch) | MSE |

- MAX_EPOCHS=200, BATCH=256, SEEDS=0–9
- MPC 판정: PDR < 0.05 AND R² ≤ 0

---

## 핵심 발견

### 1. 요인별 MPC 발생률 전체 표

| 요인 | 수준 | MPC rate | RMSE mean±std | PDR mean | stop_ep |
|------|------|---:|---|---:|---:|
| **D — patience** | patience=5 | **1.00** | 41.60±0.23 | 0.000002 | 9.9 |
| | patience=10 | 0.90 | 38.61±9.38 | 0.097 | 23.6 |
| | patience=15 *(기준)* | 0.80 | 35.96±11.83 | 0.199 | 38.7 |
| | patience=30 | 0.20 | 18.38±12.21 | 0.796 | 138.3 |
| **E — LR** | lr=1e-4 | **1.00** | 41.30±0.04 | 0.000002 | 52.1 |
| | lr=1e-3 *(기준)* | 0.80 | 35.96±11.83 | 0.199 | 38.7 |
| | lr=1e-2 | 0.90 | 38.44±8.91 | 0.096 | 37.8 |
| **F — bias init** | zero *(기준)* | 0.80 | 35.96±11.83 | 0.199 | 38.7 |
| | train_mean | **0.00** | 12.94±0.48 | 1.009 | 65.4 |
| | random_calibrated | **0.00** | 12.69±0.74 | 1.011 | 67.3 |
| **G — loss** | MSE *(기준)* | 0.80 | 35.96±11.83 | 0.199 | 38.7 |
| | MAE | **0.00** | 13.43±0.60 | 1.045 | 80.7 |
| | auxiliary | 0.40 | 24.24±15.11 | 0.605 | 77.9 |

---

### 2. MPC를 완전 제거하는 개입 (기준 0.80 대비)

| 개입 | MPC rate | RMSE mean±std | 메커니즘 |
|------|---:|---|---------|
| F_bias_train_mean | 0.00 | 12.94±0.48 | bias를 train RUL mean(≈93)으로 초기화 → trivial solution 근방을 바로 벗어남 |
| F_bias_random_calibrated | 0.00 | 12.69±0.74 | train_mean ± N(0,5) 노이즈 초기화 → 동일 효과 |
| G_loss_MAE | 0.00 | 13.43±0.60 | trivial solution 근방에서 constant gradient → 탈출 가능 |

---

### 3. 요인별 메커니즘 해석

#### D — patience: 단조 감소하나 제거 불가

patience를 늘리면 MPC rate가 줄지만 제거되지 않는다. patience=30에서도 0.20이 남는다.

- **메커니즘:** trivial solution은 val_loss가 장기간 안정적으로 낮아 patience counter가 느리게 증가한다. patience=30은 더 많은 탈출 기회를 주지만, 140 epoch에서 멈추는 run도 있어(stop_ep=138.3) 탈출 기회가 있음에도 불구하고 일부는 trivial solution 상태로 조기 종료.
- patience만으로 MPC 예방은 불완전하며, 큰 patience 값은 훈련 비용 증가를 수반.

#### E — LR: 효과 없음 (오히려 양 극단이 더 나쁨)

lr=1e-4은 MPC rate=1.00으로 기준보다 나쁘다. lr=1e-2도 0.90으로 기준보다 나쁘다.

- **lr=1e-4 메커니즘:** 학습 속도가 느려 초기 gradient step이 작아지고, trivial solution 근방에서 step이 더욱 작아 → patience 만료가 더 빠름 (stop_ep=52.1로 기준 38.7보다 더 오래 걸리나 MPC에서 빠져나오지 못함).
- **lr=1e-2 메커니즘:** 불안정한 훈련 → val_loss 진동 → patience가 일찍 소비되는 경우 많음. 실질적으로 patience를 줄이는 효과와 유사.
- **결론:** LR 조정은 MPC 예방에 효과 없음.

#### F — bias init: 가장 강력한 개입

zero init(0.80) vs train_mean/random_calibrated(0.00)의 격차가 극적이다.

- **메커니즘:** output bias=0으로 초기화하면 모델의 초기 예측이 0 근방 → train RUL mean(≈93)과 큰 괴리 → 초기 MSE loss가 크고 gradient가 활성화됨에도 불구하고, **val split이 고정(A1)**되어 있어 특정 random seed에서 trivial solution(≈93 상수 예측)이 val_loss를 조기에 안정시킴.
- bias=train_mean으로 시작하면 **이미 올바른 스케일**에서 시작 → trivial solution을 통과해 나가는 gradient가 존재. 모델이 초기부터 정보적 예측을 내놓기 때문에 patience counter가 trivial solution 구간에서 안정화되지 않음.
- **random_calibrated (train_mean ± 5):** train_mean과 MPC rate 동일(0.00). 스케일만 올바르면 미세한 노이즈는 무관함을 보임.
- **RMSE 주목:** train_mean(12.94) ≈ random_calibrated(12.69) — MPC 제거 후 성능도 동등하여 부작용 없음.

#### G — loss: MAE는 완전 제거, auxiliary는 부분 개선

- **MAE (0.00):** trivial solution 근방에서 MAE의 gradient는 sign(pred-true)으로 일정 → trivial solution이 local minimum이 아님. MSE는 gradient가 (pred-true)에 비례해 trivial solution 근방에서 매우 작아짐.
- **auxiliary (0.40):** 추가 gradient 경로로 부분 개선. 단, 보조 헤드도 MSE 기반이므로 trivial solution 탈출이 완전하지 않음. 4/10 runs에서 여전히 붕괴.
- **결론:** MSE gradient의 구조적 취약성이 MPC의 핵심. MAE(또는 다른 flat-gradient loss)가 근본 해결책.

---

### 4. 복합 해석: B2 환경에서의 예방 우선순위

Phase 1A에서 B2(warmup 없는 early stopping)가 MPC의 필요조건임이 확인됐다. Phase 1B는 **B2 환경 안에서** 무엇이 가장 효과적인 예방책인지 보여준다.

**예방 효과 순위:**
1. **F: bias init = train_mean 또는 calibrated** → MPC 0.00 (완전 제거, 저비용)
2. **G: MAE loss** → MPC 0.00 (완전 제거, 구조적 해결)
3. **D: patience=30** → MPC 0.20 (부분 개선, 훈련 비용 증가)
4. **G: auxiliary loss** → MPC 0.40 (부분 개선)
5. **E: LR 조정** → 효과 없음 (0.80~1.00)

**실용적 권고:** bias_init=train_mean은 단 한 줄의 코드 추가(`model.fc[-1].bias.fill_(c_train)`)로 MPC를 완전 제거하며, 정상 수렴 RMSE에도 영향 없다 (12.94 vs 12.97±0.67). 가장 낮은 비용의 예방책.

---

### 5. 논문에서 쓸 표현

> "MPC를 완전 제거하는 두 가지 독립적 개입이 확인됐다: (1) output layer bias의 train_mean 초기화 — trivial solution을 통과해 나가는 gradient 경로를 처음부터 확보하며 단 한 줄의 코드 수정; (2) MAE loss — trivial solution 근방에서 일정 크기의 gradient를 유지해 수렴 유도. 대조적으로 LR 조정은 효과가 없으며, patience 증가는 부분 개선에 그쳤다."

---

## Phase 2 방향 (Training Dynamics)

Phase 1B epoch_logs에서 수집된 궤적(130 runs × per-epoch PDR/R²/val_loss)을 분석한다:

- **붕괴 run vs. 정상 run의 epoch 궤적 비교** — PDR이 어느 epoch에서 0에 수렴하는가?
- **"Silence" 케이스 심층 분석** — val_loss 곡선은 정상인데 MPC인 40 cases (Phase 1A에서 확인)
- **PDR/R² 조기 경보 신호** — 붕괴 전 몇 epoch에서 신호가 나타나는가?

---

## 저장 파일 목록

| 파일 | 내용 |
|------|------|
| `Phase1B/runs.csv` | 130 runs 전체 원데이터 |
| `Phase1B/mpc_summary.csv` | 13 조건별 집계 (MPC rate, RMSE, PDR, R², stop_ep) |
| `Phase1B/epoch_logs/*.csv` | 130개 run별 epoch 궤적 (val_loss, PDR, R² per epoch) |
| `Phase1B/figures/fig1_phase1b_factors.png` | 4 요인 × MPC rate + RMSE 패널 그래프 |
| `Phase1B/figures/fig2_phase1b_scatter.png` | 조건별 per-seed RMSE scatter (red=collapsed) |
