# Essence_Research.md
## Mean-Prediction Collapse in Deep RUL Regression: Mechanisms, Detection, and Prevention

**최종 동기화:** 2026-09-15 (Gap 1~5 수정 반영)  
**동기화 출처:** Research_Plan.md (v0.4), Phase1A/1B/2/3/3B/4-Result.md, Decision_log.md (D1, D2, D3)  
**연구자:** Young Seog Yoon (ETRI)

---

## 요약 (Abstract 초안)

Stacked LSTM networks have a structural susceptibility — the forget gate can approach zero near a trivial constant-prediction solution, blocking cell state gradients via BPTT and locking the model in Mean-Prediction Collapse (MPC). Certain training protocols (absent early-stopping warmup, fixed validation split, homogeneous data) expose this susceptibility; other architectures (GRU, MLP, CNN1D) are immune under identical protocols. This study characterizes MPC, identifies its causal mechanism, and provides a prevention taxonomy and retrospective audit tool using C-MAPSS FD001–FD004.

**핵심 발견 (4줄 요약):**
1. MPC는 LSTM의 구조적 취약성이 세 조건의 교호작용으로 발현된다: (1) LSTM forget gate → 0 (아키텍처), (2) warmup 없는 early stopping + 고정 val split (프로토콜), (3) 높은 데이터 균질성 (데이터). GRU·MLP·CNN1D는 동일 조건에서 MPC 0% — LSTM 특이적 실패.
2. 인과 메커니즘은 BPTT: f^T → 0 (T=30)으로 cell state gradient가 지수 감쇠. f=1 고정(CEC)만이 이를 차단하며, fg_clamp [ε,1] (ε≤0.05)는 ε^30 ≈ 0이므로 효과 없음.
3. 다섯 가지 독립적 예방 개입이 MPC를 완전 제거한다: GRU 교체 / forget gate=1 고정 / bias_init=train_mean / MAE loss / ES warmup. 각각 정상 RMSE 비열등.
4. 학습 완료 모델의 검증셋 PDR < 0.05는 소급 진단 AUROC = 1.0000 (calibration 270 runs, held-out 330 runs) — 단일 지표로 완벽 판별.

---

## 1. 연구 배경 및 문제 정의

### 1.1 출발 관찰

BMAD 프로젝트 H6 실험에서 FD003의 단순 LSTM 기준선(M0)이 RMSE 43.23을 보였다. 이 수치는 후속 모델(M3, RMSE 14.78)과 비교해 65.8% 개선된 것처럼 보였다. 그러나 사후 분석 결과 M0는 **100개 테스트 엔진 전부에 대해 동일한 상수값 ~87을 출력**하고 있었다 (5개 시드 전부, 예측값 std < 0.0002). FD003 테스트 RUL 평균 ≈ 87과 일치 — 모델이 무조건 "평균"을 출력하는 trivial solution에 수렴한 것이다.

| 항목 | 수치 |
|------|------|
| H6 M0 FD003 RMSE (원본 프로토콜) | 43.23 ± 0.18 |
| 5개 시드 예측값 std | < 0.0002 |
| FD003 테스트 RUL 평균 | ~87 (예측값 일치) |
| 수정 프로토콜 적용 후 M0 RMSE | **12.97 ± 0.67** |

학습 프로토콜만 교정하면 동일한 LSTM으로 30+ RMSE 개선이 발생한다. **기준선이 정상이었다면 M3의 "개선"은 존재하지 않았다.**

### 1.2 개념 구분 — MPC ≠ CMC

| 현상 | 정의 | 판별 기준 |
|------|------|----------|
| **Conditional-Mean Compression (CMC)** | 정상 학습 완료 후 예측 분산이 축소되고 극단값이 평균 방향으로 편향 (Lee et al., 2025 [P7]) | PDR 감소하지만 양수 유지, R² > 0 |
| **Mean-Prediction Collapse (MPC)** | 입력 변화와 무관하게 예측이 거의 상수로 수렴 — 기능적 실패 상태 | PDR ≈ 0, R² ≤ 0 |

MPC는 단순한 "성능 부족"이 아니다. 입력 정보를 전혀 활용하지 않는 **기능적 퇴행**이다.

> **정식 정의:** Mean-Prediction Collapse is a functional regression failure in which a trained model produces an almost input-invariant prediction, exhibits negligible output dispersion relative to the target distribution, and performs no better in practical terms than an independently specified constant predictor.

### 1.3 연구 중요성

- PHM/RUL 맥락에서 **LSTM의 protocol-induced collapse**를 체계적으로 연구한 선행 연구 없음
- Undertuned-baseline 계보 (Musgrave et al. 2020 [P23], Lučić et al. 2018 [P24])의 **극단적 형태**: 기준선이 낮은 것이 아니라 기능적으로 실패한 상태
- **실무 파급:** RMSE=43 기준선 대비 "65% 개선" 모델이 실제로는 단순 기준선(RMSE=13)과 동등할 수 있다
- **본 연구의 의미는 배포 시스템이 아닌 연구 방법론 수준:** LSTM 기반 PHM 벤치마크 비교 신뢰성 문제
- **핵심 구별:** MPC는 "나쁜 프로토콜이 어떤 모델에서나 성능을 낮춘다"는 일반적 문제가 아니다. GRU·MLP·CNN1D는 동일 프로토콜에서도 영향 없으므로, MPC는 **LSTM의 구조적 취약성이 특정 조건에서 발현되는 아키텍처 특이 현상**이다.

---

## 2. 선행 연구 및 연구 갭

| 관련 현상 | 선행 연구 | 미탐구 영역 (본 연구) |
|----------|----------|-------------------|
| LSTM VGP | Al-Selwi et al. 2023 [P2] — "느린 수렴" | MPC는 즉각적 trivial solution 수렴 (다른 메커니즘) |
| 검증 편향 | Vabalas et al. 2019 [P6] — K-fold CV 편향 | RUL loss landscape의 trivial solution 왜곡 메커니즘 |
| 평균 회귀 편향 | Lee et al. 2025 [P7] — CMC 이론화 | 학습 포획(training capture), 기능적 붕괴 |
| 합성 데이터 한계 | Chao et al. 2021 [P4] — N-CMAPSS | 균질성이 collapse를 유발하는 역설 |
| Undertuned-baseline | Musgrave 2020 [P23], Lučić 2018 [P24] | PHM/RUL 도메인 적용; 기능적 붕괴 수준으로의 극단화 |
| 생성 모델 collapse | Dohmatob et al. 2024 [P1] | 단일 판별 훈련의 trivial solution 수렴 |

**결론:** PHM/RUL 맥락에서 protocol-induced baseline collapse를 체계적으로 연구한 선행 연구 없음.

---

## 3. 연구 질문 및 가설

### RQ1 — 존재와 메커니즘
LSTM 모델에서 어떤 프로토콜·데이터 조건의 교호작용이 MPC를 유발하는가? 그 구조적 원인은 무엇인가?
- **H1a:** validation composition × early stopping 교호작용이 핵심 → **확증** (Phase 1A)
- **H1b:** 붕괴 run은 epoch 1부터 PDR≈0, R²≤0 → **확증** (Phase 2)
- **H1c:** patience, bias init, loss가 MPC 발생률에 유의한 효과 → **확증** (Phase 1B)
- **H_mech (Phase 3B 추가):** LSTM forget gate → 0 시 BPTT 지수 감쇠로 cell state gradient 소멸이 근본 원인 → **확증** (V2_fg1 개입 실험)

### RQ2 — 취약성 및 일반화
어떤 아키텍처·데이터셋이 MPC에 취약한가? GRU·MLP·CNN1D도 동일 조건에서 취약한가?
- **H2a:** 취약성은 데이터 균질성(결함모드 수, op조건)으로 설명 → **부분 확증** (Phase 3)
- **H2b:** MLP·CNN1D·GRU에서 MPC가 재현되지 않음 → **확증** (H2b 원래 가설 방향과 반대 — MPC는 LSTM 특이적)
- **H2c:** 데이터셋별 취약성 다름 → **확증** (FD003>FD001, FD002/FD004=0)

### RQ3 — 소급 진단
test set 없이 학습 완료 모델을 진단할 수 있는가?
- **H3a:** PDR+R² 복합이 단일 pred_std보다 안정적 → **확증** (Phase 4 Part B)
- **H3b:** calibration 임계값이 외부 조건에서도 유지 → **확증** (AUROC=1.0, 330 held-out runs)

### RQ4 — 예방
MPC를 신뢰성 있게 방지하는 최소 비용 개입은?
- **H4a:** ES warmup이 per-seed split보다 안정적 → **확증** (warmup=0%, per-seed=0~30%)
- **H4b:** 예방 조치가 정상 run RMSE 비열등 → **확증** (모든 완전 예방 방법 RMSE ≈ 12.4~13.4)

---

## 4. MPC 조작적 정의 및 판정 지표

### 핵심 지표 4가지

| 지표 | 공식 | 의미 | 비고 |
|------|------|------|------|
| **PDR** (Prediction Dispersion Ratio) | `std(ŷ) / (std(y) + ε)` | 예측 분산 / 타깃 분산 | 본 연구 정의. PDR → 0이면 상수 예측 |
| **R²** (Explained Variance) | `1 - Σ(y-ŷ)² / Σ(y-ȳ)²` | 설명 분산 비율 | 표준 지표. R² ≤ 0이면 상수 기준선보다 나쁨 |
| **CBR** (Constant-Baseline Ratio) | `RMSE(y,ŷ) / RMSE(y, c_train)` | 상수 기준선과의 비율 | Murphy (1988) Skill Score와 수학적 동등. CBR ≈ 1이면 상수 예측기 수준 |
| **ISS** (Input Sensitivity Score) | 입력 교란 전후 출력 변화 비율 | 입력 민감성 | Zeiler & Fergus (2014) 일반화 |

### MPC 판정 규칙 (확정)

```
MPC = PDR < 0.05  AND  R² ≤ 0
```

- Phase 1A 270 runs에서 보정, Phase 1B+3+3B 330 runs에서 AUROC=1.0으로 검증
- 단일 절대 임계값(`pred_std < 1.0`)이 아닌 정규화 지표 → 데이터셋 스케일 독립

---

## 5. 방법론 (Phase 설계)

| Phase | 명칭 | 설계 | 데이터 | 아키텍처 | 총 runs |
|-------|------|------|--------|---------|---------|
| 0 | 재현·감사 | 구현 오류 배제, 단순 기준선 비교 | FD003 | LSTM (M0) | pilot |
| 1A | 핵심 메커니즘 | 3×3×3 full factorial (A×B×C) × 10 seeds | FD003 | LSTM | 270 |
| 1B | 최적화 후속 | OAT 4요인 (patience/LR/bias/loss) × 10 seeds, A1+B2+C2 고정 | FD003 | LSTM | 130 |
| 2 | Training Dynamics | Phase1A+1B epoch_logs 분석 (400 runs, 40,902 epoch rows) | FD003 | LSTM | — |
| 3 | 일반화 | 4 datasets × 4 architectures × 10 seeds | FD001–4 | LSTM/GRU/MLP/CNN1D | 160 |
| 3B | 메커니즘 확증 | 4 LSTM 변형 × 10 seeds, A1+B2+C2 | FD003 | LSTM variants + GRU | 40 |
| 4 | 예방·진단 | Part A: fg_clamp 30 runs; Part B: cal(270)+held-out(330) | FD003 → 전체 | 전체 | 30 + 분석 |

**총 실험 runs:** 630+ (Phase 0 제외)  
**Calibration / Held-out 분리:** Phase1A = calibration (270 runs); Phase1B+3+3B = held-out (330 runs)

---

## 6. 실험 결과

### Phase 1A — 핵심 조건 (3×3×3 factorial, 270 runs)

**요인별 한계 MPC 발생률:**

| 요인 | 수준 | MPC rate |
|------|------|:--------:|
| B — Early Stopping | B1_off (off) | **0.000** |
| | **B2_from0 (no warmup)** | **0.444** |
| | B3_warmup | **0.000** |
| A — Val Split | A1_fixed | 0.244 |
| | A2_per_seed | 0.078 |
| | A3_stratified | 0.122 |
| C — RUL Labeling | C1_unclipped | 0.200 |
| | C2_clip125 | 0.156 |
| | C3_clip100 | 0.089 |

**핵심 결론:**
- B2(warmup 없는 early stopping)는 **MPC의 필요조건** — B2 없이는 180 runs 전부 MPC 없음
- 최악 조합 A1+B2+C1: MPC **90%** / 최선 조합 A2+B2+C3: MPC **0%**
- "Silence": 붕괴 run 116개 중 116개(100%)가 val_loss 기준으로 "정상 수렴"처럼 보임

### Phase 1B — 최적화 요인 (A1+B2+C2 고정, 130 runs)

| 요인 | 최선 수준 | MPC rate | RMSE |
|------|---------|:--------:|------|
| **F — bias init** | **train_mean** | **0.00** | 12.94±0.48 |
| **F — bias init** | random_calibrated | **0.00** | 12.69±0.74 |
| **G — loss** | **MAE** | **0.00** | 13.43±0.60 |
| D — patience | 30 | 0.20 | 18.38±12.21 |
| G — loss | auxiliary | 0.40 | 24.24±15.11 |
| E — LR | 1e-4 / 1e-2 | **1.00 / 0.90** | 더 나쁨 |
| 기준 (patience=15, lr=1e-3, bias=0, MSE) | | 0.80 | 35.96±11.83 |

**실용 권고:** `model.fc[-1].bias.fill_(c_train)` — 1줄 코드로 MPC 완전 제거, RMSE 비영향

### Phase 2 — Training Dynamics (400 runs, 40,902 epoch rows)

| 발견 | 수치 | 의미 |
|------|------|------|
| PDR onset epoch | **median = 1** | MPC는 초기 탈출 실패 (중간 발산 아님) |
| Silence 비율 | **116/116 (100%)** | val_loss 곡선으로 MPC 감지 불가 |
| 실시간 경보 sensitivity | 1.000 (모든 창 크기) | |
| 실시간 경보 specificity | **0.063** | 정상 run도 초기 PDR≈0 → 구분 불가 |
| ES 발화 시점 PDR (정상) | ~0.980 | |
| ES 발화 시점 PDR (붕괴) | ~0.000002 | 완벽 분리 → **소급 진단 가능** |

**결론:** 실시간 경보는 불가. ES 발화 시점의 PDR 한 번 측정으로 완벽 사후 진단 가능.

### Phase 3 — Dataset × Architecture 일반화 (160 runs)

**MPC rate 매트릭스:**

| | FD001 | FD002 | FD003 | FD004 |
|---|:---:|:---:|:---:|:---:|
| **LSTM** | 0.10 | 0.00 | **0.80** | 0.00 |
| GRU | 0.00 | 0.00 | 0.00 | 0.00 |
| MLP | 0.00 | 0.00 | 0.00 | 0.00 |
| CNN1D | 0.00 | 0.00 | 0.00 | 0.00 |

**핵심 결론:**
- MPC는 **LSTM에 특이적** — GRU/MLP/CNN1D: 60/60 runs에서 전혀 없음
- FD003 취약성 최고 (2 결함모드, 단일 op조건) > FD001 > FD002/FD004=0
- FD002/FD004: op-condition 잔차화가 엔진 간 다양성 확보 → trivial solution 불안정

**범위 재정의 (go/no-go → HOLD):** 논문 범위 = "LSTM 특이적 MPC" — 일반 회귀 pathology 주장 철회

### Phase 3B — LSTM vs GRU 메커니즘 (40 runs, H_mech 가설)

| 조건 | MPC rate | RMSE | 해석 |
|------|:--------:|------|------|
| V3_base (표준 LSTM) | 0.80 | 35.96 | 기준 |
| LSTM_gate (커스텀, gate logging) | 0.60 | 30.18±14.98 | 구현 차이 |
| V1_fb1 (forget bias +1) | 0.40 | 24.29±14.80 | 부분 개선 |
| **V2_fg1 (forget gate = 1 고정)** | **0.00** | **12.96±0.95** | ✅ **완전 제거** |
| GRU_gate | 0.00 | 12.27±0.67 | ✅ Phase 3 재확인 |

**인과 사슬 (확증):**
```
LSTM forget gate → 0  (trivial solution 근방)
    ↓
BPTT T스텝 전개:  ∂L/∂c_{t-T} ≈ f^T · ∂L/∂c_t
  f → 0: 0^30 ≈ 0                    → gradient 전파 차단
  f ≥ ε (clamp): ε^30 ≈ 10⁻⁹⁰ ≈ 0   → 실질적으로 동일 (Gap 1 수정)
  f = 1 (CEC):   1^30 = 1             → decay 없음
    ↓
초기 타임스텝 파라미터 업데이트 불가 → trivial solution 고착
    ↓
early stopping이 trivial solution에서 종료
    ↓
MPC 고착 (PDR≈0, R²<0)

GRU: cell state 없음 → 이 BPTT 경로 자체 부재 → MPC 0/10
V2_fg1: f=1 고정(CEC) → 1^T = 1 → gradient decay 없음 → MPC 0/10
fg_clamp ε≤0.05: ε^30 ≈ 0 → 효과 없음 (Baseline과 동일)
```

**go/no-go: GO** — H_mech 인과 확증 완료 (V2_fg1 개입으로 직접 검증)

### Phase 4 — 예방 분류체계 + 소급 진단 (30 신규 runs + 통합 분석)

#### Part A: 예방 분류체계 (12개 방법, MPC rate 오름차순)

> ⚠️ **비교 조건 주의:** 방법마다 테스트된 baseline protocol이 다르다. "A1+B2+C2 고정" 방법들(bias_init, MAE, patience, fg_clamp, GRU, V2_fg1)은 동일 adversarial baseline(MPC=0.80) 위에서 비교된다. "프로토콜 자체 변경" 방법들(ES warmup: B2→B3, per-seed split: A1→A2)은 해당 프로토콜 factor가 변경된 결과다. Table 2에서 "개입 대상" 열로 구분 필요.

| 방법 | 개입 대상 | 테스트 조건 | MPC rate | RMSE(전체) | RMSE(정상) | 비용 |
|------|---------|-----------|:--------:|:----------:|:----------:|:----:|
| **GRU 교체** | 아키텍처 | A1+B2+C2 | **0%** | 12.61±0.47 | 12.61 | 중간 |
| **forget gate=1 (V2_fg1)** | 아키텍처 내부 | A1+B2+C2 | **0%** | 12.96±0.95 | 12.96 | 낮음 |
| **bias_init = train_mean** | 초기화 | A1+B2+C2 | **0%** | 12.94±0.48 | 12.94 | 매우 낮음 |
| **MAE loss** | Loss | A1+B2+C2 | **0%** | 13.43±0.60 | 13.43 | 매우 낮음 |
| **ES warmup (B3)** | 프로토콜 B | A1+**B3**+C2 | **0%** | 12.42±1.04 | 12.42 | 매우 낮음 |
| patience = 30 | 프로토콜 B | A1+B2+C2 | 20% | 18.38±12.21 | 12.59 | 매우 낮음 |
| per-seed random split | 프로토콜 A | **A2**+B2+C2 | 30% | 21.50±14.02 | 12.80 | 매우 낮음 |
| V1_fb1 (forget bias +1) | 초기화 | A1+B2+C2 | 40% | 24.29±14.80 | 12.84 | 매우 낮음 |
| fg_clamp ε=0.05 | 아키텍처 내부 | A1+B2+C2 | 60% | 30.11±14.95 | 12.74 | 매우 낮음 |
| fg_clamp ε=0.01 | 아키텍처 내부 | A1+B2+C2 | 60% | 30.08±14.99 | 12.67 | 매우 낮음 |
| fg_clamp ε=0.001 | 아키텍처 내부 | A1+B2+C2 | 60% | 30.08±14.99 | 12.67 | 매우 낮음 |
| **Baseline** | — | A1+B2+C2 | **60%** | 30.18±14.98 | 12.78 | — |

**fg_clamp 예상 밖 결과 — CEC 관점 해석:**  
ε=0.001~0.05 범위의 forget gate clamp는 효과 없음 (Baseline과 동일 60%). 이유는 단일 스텝 gradient의 크기가 아니라 **BPTT T스텝 전개의 지수 감쇠**: `ε^T`에서 ε=0.001, T=30이면 `10⁻⁹⁰`이 되어 실질적으로 0과 동일하다. V2_fg1(f=1)이 유일하게 작동하는 이유는 `1^30 = 1`로 감쇠 자체가 없기 때문(CEC 원리). **"f 하한 보장"은 지수 감쇠를 늦출 뿐이며, "f=1 고정(CEC)"만이 이를 원천 제거한다.**  
> 부가 통찰: 선택적 망각(f < 1)과 gradient decay 방지(f = 1)는 구조적으로 양립 불가. fg_clamp는 이 둘을 동시에 얻으려 했으나 불가능하다.

#### Part B: 소급 진단 도구

| 지표 | 데이터 | 민감도 | 특이도 | AUROC |
|------|--------|:------:|:------:|:-----:|
| **PDR < 0.05** | Calibration (270) | **1.000** | **1.000** | **1.0000** |
| R² ≤ 0 | Calibration (270) | 1.000 | 0.935 | 0.9796 |
| RMSE > 25 | Calibration (270) | 1.000 | 0.687 | 0.9370 |
| **PDR < 0.05** | Held-out (330) | **1.000** | **1.000** | **1.0000** |
| R² ≤ 0 | Held-out (330) | 1.000 | 1.000 | 1.0000 |
| RMSE > 25 | Held-out (330) | 1.000 | 0.957 | 1.0000 |

**권장 진단 루틴:**
```python
pdr = np.std(pred) / (np.std(true) + 1e-8)
if pdr < 0.05:  # 단 한 줄 — AUROC 1.0 보장
    print("WARNING: MPC detected. Re-train.")
```

---

## 7. 핵심 기여

### C1 — MPC 조작적 정의 (Characterization)
- PDR, CBR, R², ISS를 결합한 MPC 판정 체계
- CMC(정상 과소분산)와 MPC(기능적 붕괴)의 명확한 개념 구분
- 데이터셋 스케일 독립 지표(PDR) — FD001~FD004, 다중 아키텍처 동일 임계값 적용 가능

### C2 — 인과 메커니즘 (Mechanism)
- LSTM 특이적 취약성: forget gate → 0 시 BPTT T스텝 지수 감쇠(`f^T → 0`)로 cell state gradient 소멸
- V2_fg1(f=1, CEC) 개입 실험으로 인과적 필요조건 직접 확증; fg_clamp 실패로 "ε^T ≈ 0" 추가 확증
- GRU가 동일 조건에서 MPC를 전혀 보이지 않는 구조적 이유: cell state 자체 부재 → 이 BPTT 경로 없음

### C3 — 예방 분류체계 (Prevention Taxonomy)
- 12개 방법을 MPC rate / RMSE non-inferiority / 구현 비용 / **테스트 조건** 4축으로 체계적 비교
- 완전 예방 5개 방법 확정 (모두 정상 RMSE 비열등, 각각 독립적 개입 경로)
- fg_clamp 실패: "선택적 망각(f<1)"과 "gradient decay 방지(f=1)"는 구조적으로 양립 불가

### C4 — 소급 진단 도구 (Retrospective Audit Tool)
- 학습 완료 모델의 검증셋 출력만으로 MPC 완벽 진단 (AUROC=1.0000)
- 270-run calibration → 330-run held-out 완벽 일반화
- FD001~4, LSTM/GRU/MLP/CNN1D 동일 임계값 (PDR < 0.05) 적용 가능

### C5 — 벤치마크 신뢰성 (Benchmark Reliability) [Decision_log D3 — 외부 audit 포함 결정]
- BMAD H6 자기 사례(MPC 기준선 RMSE=43.23 → 정상 12.97) + **외부 published 논문 프로토콜 감사**
- Phase 1A 조건 충족(A1+B2 교호작용 MPC ≥ 60%): FD003 고-RMSE 보고 논문 3~5편의 methods 섹션을 체크리스트(val split 방식·ES 조건·loss 설정)로 분류 → "MPC 발생 개연성 있음/불확실/낮음"
- 주장 수위: "붕괴했다"가 아닌 **"프로토콜 서술에 근거한 발생 개연성"** — 코드 재실행 없이 텍스트 감사만으로 가능
- **이 기여가 추가되면 논문 임팩트가 "자기 사례 경고" → "문헌 신뢰성 체계 감사"로 격상**

---

## 8. 연구 한계

| 한계 | 수준 | 완화책 |
|------|------|--------|
| **LSTM 특이적** — GRU/MLP/CNN1D에서 MPC 없음 | 주요 | 논문 범위를 "LSTM 사용자를 위한 경고"로 명시. 프레이밍을 "LSTM의 구조적 취약성"으로 유지 |
| **C-MAPSS 한정** — N-CMAPSS, bearing 데이터 미검증 | 주요 | future work로 명시 (Decision_log D1) |
| FD001/FD002 상수 센서 미제거 | 방법론 | FD001 MPC rate 과소추정 가능성 표기. FD002 취약성 비교에 영향 가능 |
| **예방 taxonomy 비교 불공정** — ES warmup(B2→B3)·per-seed(A1→A2)는 다른 baseline | 방법론 | Table 2에 "개입 대상"·"테스트 조건" 열 추가. "protocol-level vs architecture-level" 분류 명시 |
| **통계 분석 미완** — 계획된 mixed-effects logistic regression 미실행 | 방법론 | MPC rate에 95% CI 또는 Fisher's exact test 추가 필요. "20% vs 30%" 비교는 n=10으로 통계적 구분 불가 |
| gradient 직접 측정 미완 (BPTT 이론의 실험적 확인) | 이론 | V2_fg1 개입 결과가 강력한 간접 증거. gradient norm logging은 future work |
| 실제 항공 엔진 배포 적용성 | 범위 | 합성 데이터 한계 서론에 명시 |
| 10 seeds — GRU 0/10이 정확히 0인지 단순히 낮은지 불명 | 통계 | Phase 1A 강한 조건(0.80)에서 0/10이므로 현저히 낮음은 명확 |

---

## 9. 논문 구조 (예상 섹션 매핑)

| 논문 섹션 | 내용 | 주요 수치 출처 |
|---------|------|-------------|
| §1 Introduction | MPC 발견 동기, 연구 중요성, 기여 요약 | BMAD H6, Research_Plan §1 |
| §2 Related Work | VGP, 검증 편향, CMC, undertuned-baseline 계보 | Research_Plan §2 |
| §3 Problem Formulation | MPC 조작적 정의, PDR/R²/CBR/ISS | Research_Plan §5 |
| §4 Methodology | Phase 설계, 공통 원칙, 통계 계획 | Research_Plan §6, §7 |
| §5 Results: Conditions | Phase 1A 3×3×3 결과, "Silence" | Phase1A-Result.md |
| §6 Results: Optimization | Phase 1B 4요인 결과 | Phase1B-Result.md |
| §7 Results: Dynamics | Phase 2 궤적 분석 | Phase2-Result.md |
| §8 Results: Generalization | Phase 3 MPC rate 매트릭스 | Phase3-Result.md |
| §9 Results: Mechanism | Phase 3B H_mech 확증 | Phase3-Result.md (3B) |
| §10 Results: Prevention | Phase 4 예방 분류체계 + 소급 진단 | Phase4-Result.md |
| §11 Discussion | 통합 해석, 벤치마크 신뢰성, 한계 | 이 문서 §7, §8 |
| §12 Conclusion | 4가지 핵심 기여, future work | 이 문서 §7 |

---

## 10. 핵심 그림 / 표 (논문 Figure/Table 후보)

| 번호 | 유형 | 내용 | 데이터 출처 |
|------|------|------|-----------|
| **Figure 1** | 궤적 | collapsed vs normal run의 epoch별 PDR·RMSE·val_loss + checkpoint 시점 | Phase2/figures/fig1_mean_trajectories.png |
| **Figure 2** | 산점도 | target-prediction 산점도: collapsed(상수선) vs normal(대각선) | Phase3B epoch_logs |
| **Figure 3** | 히트맵 | A×B×C MPC 발생률 (27 조건, C별 패널) | Phase1A/figures/fig1_mpc_heatmap.png |
| **Figure 4** | 매트릭스 | dataset × architecture MPC rate + RMSE | Phase3/figures/fig1_mpc_heatmap.png |
| **Figure 4B** | 메커니즘 | forget gate trajectory (collapsed vs normal) + 인과 다이어그램 | Phase3B/figures/fig2_gate_trajectory.png |
| **Figure 5** | ROC | 소급 진단 PDR/R²/RMSE ROC 곡선 (cal + held-out) | Phase4/figures/fig3_audit_roc.png |
| **Table 1** | 개념 표 | MPC vs CMC vs VGP 정의 비교 | Research_Plan §5.3 |
| **Table 2** | 분류체계 | 예방책 12개: MPC rate + RMSE + 비용 | Phase4/taxonomy.csv |
| **Table 3** | 수치 | Δ_inflation 예시: BMAD H6 65.8% 과장 사례 | H6_corrected_results.csv |

---

## 11. 한 문장 연구 요약

> **This study identifies Mean-Prediction Collapse as an LSTM-specific failure in which the forget gate approaching zero causes exponential BPTT gradient decay that, under certain protocol and data conditions, locks the model at a trivial constant-prediction solution; characterizes the three-way interaction triggering this failure; demonstrates five complete-elimination prevention interventions; and delivers a retrospective audit tool (PDR < 0.05, AUROC = 1.0000) that enables post-hoc diagnosis without accessing the test set.**

---

## 12. 동기화 체크리스트

마지막 업데이트 시 확인한 소스 파일 버전:

| 파일 | 버전/날짜 | 주요 수치 |
|------|---------|---------|
| `Research_Plan.md` | v0.4 (2026-09-15) | Phase 4 Option C 반영, Phase 5 흡수 |
| `Phase1A-Result.md` | 2026-09-14 | 270 runs, MPC rate 0~0.90 |
| `Phase1B-Result.md` | 2026-09-14 | 130 runs, bias_init/MAE → 0% MPC |
| `Phase2-Result.md` | 2026-09-14 | Silence=100%, PDR onset epoch=1 |
| `Phase3-Result.md` | 2026-09-14 | 160 runs, LSTM 특이적 확인 |
| `Phase3-Result.md` (3B section) | 2026-09-14 | 40 runs, V2_fg1 MPC 0/10 |
| `Phase4-Result.md` | 2026-09-15 (rev) | CEC 이론 수정, taxonomy 조건 명시, fg_clamp 설명 정정 |
| `Phase3-Result.md` (3B) | 2026-09-15 (rev) | BPTT 전개 추가, Phase 4 cross-reference |
| `Decision_log.md` | 2026-09-15 | D1 (bearing 제외), D2 (Option C), D3 (외부 audit 포함) |
