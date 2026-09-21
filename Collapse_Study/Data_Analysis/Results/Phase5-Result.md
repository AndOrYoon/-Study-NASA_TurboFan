# Phase 5 Result
## External Protocol Audit — Published CMAPSS LSTM Papers

**실험일:** 2026-09-15  
**목적:** 투고 우선순위 저널(EAAI, Expert Systems with Applications)에 게재된 CMAPSS LSTM 논문의 훈련 프로토콜을 재현하여 MPC 발생 여부를 실험적으로 확인한다.  
**방법:** 각 논문의 val split / early stopping / RUL labeling 조건을 추출 또는 추론하여 10 seeds 실험 실행. FD003, 표준 LSTM 백본(LSTM1(64)→LSTM2(64)→FC).  
**코드:** `Collapse_Study/Data_Analysis/Code/06_external_audit.py`  
**결과 파일:**
- `Phase5/runs_new.csv` — P2/P3/P5 신규 실험 raw data (30 runs)
- `Phase5/runs_all.csv` — P1~P5 전체 통합 (50 runs)
- `Phase5/summary.csv` — 논문별 집계 (MPC rate, RMSE mean±std)

---

## 최종 실험 결과 요약

| Paper | 저널 | 저널확인 | val_split | ES | clip | patience | **MPC rate** | RMSE mean±std | 판정 |
|-------|------|:---:|---|---|---|:---:|:---:|---|---|
| P1 ESWA_2023 | Expert Systems | ✅ | A1_fixed | B2_from0 | clip125 | 15 | **0.80 (8/10)** | 35.96±11.83 | 🔴 개연성 있음 |
| P2 EAAI_2024 | EAAI | ✅ | A1_fixed | B2_from0 | clip125 | 20 | **0.60 (6/10)** | 30.18±14.64 | 🟠 개연성 있음 |
| P3 Zheng_2017 | IEEE ICPHM | ✅ | A1_fixed | B2_from0 | clip125 | 10 | **0.90 (9/10)** | 38.61±9.38 | 🔴 개연성 높음 |
| P4 Pre2020_EAAI | EAAI/ESWA 대표 | 🟡 | A1_fixed | B2_from0 | unclipped | 15 | **0.90 (9/10)** | 56.14±6.85 | 🔴 개연성 가장 높음 |
| P5 Elsherif_2025 | Scientific Reports | ✅ | A1_fixed | B1_short | clip125 | ∞(25ep) | **0.90 (9/10)** | 39.33±7.14 | 🔴 개연성 높음 |

> **핵심 발견:** A1+B2 조합(고정 val split + warmup 없는 조기 종료)은 patience 값이 10~20 범위 어디에 있어도, RUL clipping을 써도 쓰지 않아도, 심지어 조기 종료 없이 고정 epoch 25번 훈련해도 MPC 발생률 60~90%를 기록한다.

---

## 집계 수치 (summary.csv 원본)

```
paper_id  n_seeds  mpc_count  mpc_rate  rmse_mean  rmse_std  pdr_mean  r2_mean  stop_ep_mean
      P1       10          8       0.80    35.9601   11.8252    0.1994   0.0750          38.7
      P2       10          6       0.60    30.1807   14.6408    0.3998   0.2804          60.3
      P3       10          9       0.90    38.6067    9.3825    0.0975  -0.0233          23.6
      P4       10          9       0.90    56.1402    6.8517    0.1493  -0.8639          21.6
      P5       10          9       0.90    39.3290    7.1360    0.0969  -0.0382          25.0
```

---

## 감사 논문 5편 상세

### P1 — Expert Systems with Applications 2023 (확인됨)

| 항목 | 내용 |
|------|------|
| **제목** | Bayesian gated-transformer model for risk-aware prediction of aero-engine remaining useful life |
| **저자** | Meng et al. (2023) |
| **저널** | Expert Systems with Applications, Vol. 238, Art. 121859 |
| **DOI / PII** | S0957417423023618 |
| **저널 확인** | ✅ Expert Systems with Applications |
| **보고 FD003 RMSE** | 미확인 (전문 접근 불가) |

**프로토콜 (추론):**

| 요인 | 조건 | 근거 |
|------|------|------|
| val split | A1_fixed (seed=42) | 2023년 ESWA CMAPSS 논문 지배적 관행 |
| early stopping | B2_from0 (warmup 없음) | warmup 미언급 = B2 간주 |
| RUL labeling | C2_clip125 | 2022년 이후 표준 |
| patience | 15 | Phase1A 기준값 |
| 증거 수준 | **INFERRED** | 전문 미접근 |

**실험 결과 (Phase1A 참조, A1+B2+C2 patience=15, 10 seeds):**
- MPC rate = **0.80** (8/10 seeds collapsed)
- RMSE: 35.96 ± 11.83  PDR: 0.199  R²: 0.075  mean stop_ep: 38.7
- 비붕괴 시드의 RMSE: 약 12~15 (정상 수렴 시 성능 수준 확인됨)

**판정:** 🔴 **MPC 발생 개연성 있음**
- 재현 시 10번 중 8번 LSTM mean prediction collapse 발생
- LSTM 비교 기준선 RMSE가 30–45대라면 붕괴 상태 보고 개연성 높음

---

### P2 — Engineering Applications of Artificial Intelligence 2024 (확인됨)

| 항목 | 내용 |
|------|------|
| **제목** | Spatial and temporal attention-based and residual-driven long short-term memory networks with implicit features for remaining useful life prediction |
| **저자** | Qin et al. (2024) |
| **저널** | Engineering Applications of Artificial Intelligence, Vol. 133, Art. 108563 |
| **DOI / PII** | S0952197624007073 |
| **저널 확인** | ✅ EAAI |
| **보고 FD003 RMSE** | **12.14** |

**프로토콜 (추론):**

| 요인 | 조건 | 근거 |
|------|------|------|
| val split | A1_fixed (seed=42) | EAAI 2022-2024 논문 지배적 관행 |
| early stopping | B2_from0 (warmup 없음) | warmup 미언급 |
| RUL labeling | C2_clip125 | 2022년 이후 표준 |
| patience | **20** | EAAI 2022-2024 논문에서 patience=20 자주 보고 |
| 증거 수준 | **INFERRED** | 전문 미접근; patience=20은 차별화 실험값 |

**실험 결과 (신규 실험, patience=20, 10 seeds):**
- MPC rate = **0.60** (6/10 seeds collapsed)
- RMSE: 30.18 ± 14.64  PDR: 0.400  R²: 0.280  mean stop_ep: 60.3

| seed | 결과 | stop_ep | RMSE |
|:---:|------|:---:|------|
| 0 | COLLAPSED | 23 | 41.24 |
| 1 | COLLAPSED | 24 | 41.61 |
| 2 | **정상** | 132 | 11.85 |
| 3 | **정상** | 100 | 12.57 |
| 4 | COLLAPSED | 23 | 41.57 |
| 5 | **정상** | 80 | 15.31 |
| 6 | COLLAPSED | 24 | 41.45 |
| 7 | **정상** | 143 | 13.07 |
| 8 | COLLAPSED | 31 | 41.84 |
| 9 | COLLAPSED | 23 | 41.31 |

**판정:** 🟠 **MPC 발생 개연성 있음**
- patience=20: MPC rate 80% → 60% 부분 감소; 완전 제거 불가
- 보고값 RMSE=12.14 ∈ 비붕괴 시드 범위(11.85–13.07) — 일치
- 60% 확률로 다른 실험자는 붕괴된 결과(RMSE≈41) 획득

> **주의:**
> - 제안 모델(Spatial-Temporal Attention LSTM): 보조 supervision·주의 메커니즘으로 기준선 LSTM 대비 붕괴 저항성 높을 수 있음
> - 이 audit 측정 대상: **LSTM 비교 기준선**이 붕괴 환경에서 훈련됐을 개연성

---

### P3 — IEEE ICPHM 2017 (Zheng et al. — 가장 많이 인용된 LSTM 기준선)

| 항목 | 내용 |
|------|------|
| **제목** | Long short-term memory network for remaining useful life estimation |
| **저자** | Zheng S., Ristovski K., Farahat A., Gupta C. (2017) |
| **저널** | IEEE Int. Conf. on Prognostics and Health Management (ICPHM) |
| **저널 확인** | ✅ (conference paper) |
| **보고 FD003 RMSE** | **16.18** |

**프로토콜 (추론):**

| 요인 | 조건 | 근거 |
|------|------|------|
| val split | A1_fixed (seed=42) | 2017년 논문; val seed 미언급 = 고정 추론 |
| early stopping | B2_from0 (warmup 없음) | 논문 설명 "조기 종료 적용" — warmup 미언급 |
| RUL labeling | C2_clip125 | 논문에서 clip=125 명시적 사용 |
| patience | **10** | 2017년 소규모 모델 관행; 논문 미명시 → 보수적 추정 |
| 증거 수준 | **INFERRED** | patience 값 원문 미확인 |

**실험 결과 (신규 실험, patience=10, 10 seeds):**
- MPC rate = **0.90** (9/10 seeds collapsed)
- RMSE: 38.61 ± 9.38  PDR: 0.098  R²: −0.023  mean stop_ep: 23.6

| seed | 결과 | stop_ep | RMSE |
|:---:|------|:---:|------|
| 0 | COLLAPSED | 13 | 41.24 |
| 1 | COLLAPSED | 14 | 41.61 |
| 2 | **정상** | 106 | 11.91 |
| 3 | COLLAPSED | 14 | 41.91 |
| 4 | COLLAPSED | 13 | 41.57 |
| 5 | COLLAPSED | 14 | 41.49 |
| 6 | COLLAPSED | 14 | 41.45 |
| 7 | COLLAPSED | 14 | 41.76 |
| 8 | COLLAPSED | 21 | 41.84 |
| 9 | COLLAPSED | 13 | 41.31 |

**판정:** 🔴 **MPC 발생 개연성 높음**
- patience=10: 대부분 시드를 ep13–14에서 조기 종료
- seed=2만 ep106 생존 → RMSE=11.91
- 보고값 16.18 ∈ 정상 수렴 범위(11–16) → 단일 비붕괴 시드 보고 가능성

> **역설:**
> - Zheng et al. 2017: CMAPSS 최다 인용 LSTM 기준선 → 재현 시 10번 중 9번 붕괴
> - 동일 프로토콜 따른 후속 논문: 기준선 비교에 심각한 오염 발생 가능성

---

### P4 — EAAI/Expert Systems 대표 Pre-2020 (Unclipped)

| 항목 | 내용 |
|------|------|
| **대표 유형** | 2018–2021 CMAPSS LSTM 논문, RUL clipping 미적용 패턴 |
| **저널** | EAAI / Expert Systems with Applications (대표 패턴) |
| **저널 확인** | 🟡 대표 패턴 (특정 논문 미확인) |

**프로토콜:**

| 요인 | 조건 |
|------|------|
| val split | A1_fixed |
| early stopping | B2_from0 |
| RUL labeling | **C1_unclipped** |
| patience | 15 |

**실험 결과 (Phase1A 참조, A1+B2+C1 patience=15, 10 seeds):**
- MPC rate = **0.90** (9/10 seeds collapsed)
- RMSE: 56.14 ± 6.85  PDR: 0.149  R²: −0.864  mean stop_ep: 21.6

**판정:** 🔴 **MPC 발생 개연성 가장 높음**
- clip 미적용 + warmup 없는 ES + 고정 split → Phase1A 최고 위험 조합(MPC 90%)
- 2018-2021년 EAAI/ESWA 논문이 이 패턴 사용 시: FD003 LSTM 기준선 붕괴 상태 거의 확실

---

### P5 — Scientific Reports 2025 (Elsherif et al.)

| 항목 | 내용 |
|------|------|
| **제목** | A deep learning-based prognostic approach for predicting turbofan engine degradation and remaining useful life |
| **저자** | Elsherif S.M. et al. (2025) |
| **저널** | Scientific Reports, Vol. 15, Art. 12959 |
| **DOI** | https://doi.org/10.1038/s41598-025-09155-z |
| **저널 확인** | ✅ Scientific Reports (open access) |
| **보고 FD003 RMSE** | **13.40** |

**프로토콜 (직접 확인):**

| 요인 | 조건 | 근거 |
|------|------|------|
| val split | A1_fixed (seed=42) | 논문 미명시 → 표준 추론 |
| early stopping | **B1_short (25 epochs 고정)** | **논문에 "25 training rounds for FD003" 명시** |
| RUL labeling | C2_clip125 | 2025년 표준 |
| max_epochs | **25** | 논문에서 직접 확인 |
| 주의 | 논문: batch=32, LR=0.0001 / 본 실험: batch=256, LR=1e-3 | 백본 공유; 완전 복제 아님 |

**실험 결과 (신규 실험, max_epochs=25 고정, no ES, 10 seeds):**
- MPC rate = **0.90** (9/10 seeds collapsed)
- RMSE: 39.33 ± 7.14  PDR: 0.097  R²: −0.038  mean stop_ep: 25.0 (전부 고정)

| seed | 결과 | stop_ep | RMSE |
|:---:|------|:---:|------|
| 0 | COLLAPSED | 25 | 41.24 |
| 1 | COLLAPSED | 25 | 41.61 |
| 2 | COLLAPSED | 25 | 41.24 |
| 3 | COLLAPSED | 25 | 41.91 |
| 4 | COLLAPSED | 25 | 41.57 |
| 5 | **정상** | 25 | **19.03** |
| 6 | COLLAPSED | 25 | 41.45 |
| 7 | COLLAPSED | 25 | 41.76 |
| 8 | COLLAPSED | 25 | 41.84 |
| 9 | COLLAPSED | 25 | 41.31 |

**판정:** 🔴 **MPC 발생 개연성 높음**
- 25 epoch 고정 훈련: MPC 예방 불가
- patience=20에서 정상 수렴(ep80–143) 시드도 ep25 강제 종료 시 붕괴
- 유일 비붕괴 시드(seed=5) RMSE=19.03 → 보고값 13.40 대비 현저히 열등

> **하이퍼파라미터 차이 주의:**
> - 논문: batch=32, LR=0.0001 / 본 실험: batch=256, LR=1e-3
> - 더 작은 배치·낮은 LR → 수렴 지연 → 25 epochs 내 학습 불완전 → 역설적으로 MPC 완전 수렴 억제 가능
> - P5 결과 해석 범위: "표준 LSTM 하이퍼파라미터 + 25-epoch 고정 훈련" 조건의 MPC 취약성 측정
> - 해당 논문의 정확한 복제 아님 → 보고값 13.40 달성 경위 불확실

---

## 핵심 발견 및 분석

### 1. Patience 단조 효과 (Monotone Effect)

| patience | MPC rate | mean stop_ep (collapsed) |
|:---:|:---:|:---:|
| 10 (P3) | 0.90 | ~14 |
| 15 (P1) | 0.80 | ~30 |
| 20 (P2) | 0.60 | ~25 |

- patience 10→15→20: MPC rate 90%→80%→60% 단조 감소
- patience=20에서도 60% — 여전히 높음
- **patience 증가: 부분 완화일 뿐 구조적 예방 아님**
- 메커니즘: 붕괴 시드는 더 늦게 종료될 뿐 결과 동일(RMSE≈41.3–41.9); patience가 클수록 더 많은 시드가 trivial solution 탈출 시간 확보

> **💡 부연 설명 — patience가 MPC rate를 낮추는 원리**
>
> - 모델이 초반에 trivial solution(평균값 ~87 예측) 진입 시: validation loss 정체 → ES가 "개선 없음" 판단 → patience 카운터 시작
> - **patience=10:** ep13–14에서 종료 → 탈출 시간 없음
> - **patience=15:** ep~30에서 종료 → 극소수 시드만 탈출 가능
> - **patience=20:** ep23–31에서 종료(붕괴 시드) — 일부 시드는 이 기간 내 탈출 → ep80–143 정상 학습
> - 핵심: patience는 결과를 바꾸는 게 아님 — **탈출 잠재력이 있는 시드에게 시간을 더 줄 뿐**
> - 붕괴 시드: patience 무관하게 동일한 RMSE≈41로 수렴
> - A1+B2 구조적 취약성(고정 val split + warmup 없는 ES): patience 조정으로 해소 불가

### 2. 단기 고정 훈련 (P5)은 예방이 아님

- P5(25 epochs fixed) MPC rate 0.90 > P2(patience=20) MPC rate 0.60 — 오히려 높은 이유:
  - patience=20 정상 수렴 시드(2,3,5,7): ep80–143 필요 → ep25 강제 종료 시 붕괴/underfitting
  - seed=5: ep25에서 RMSE=19.03 (patience=20 동일 시드 ep80에서 15.31)
- **짧은 고정 epoch 훈련: "우연한 MPC 예방" 아님 — 오히려 위험 증가 가능**

### 3. Clip vs Unclipped: P4 vs P1

| 조건 | MPC rate | RMSE mean |
|---|:---:|:---:|
| P1 (clip125, patience=15) | 0.80 | 35.96 |
| P4 (unclipped, patience=15) | 0.90 | 56.14 |

- clip=125 적용 효과: MPC rate 90% → 80% 감소; 붕괴 시 RMSE도 감소(56.14 → 35.96)
- clip은 MPC 자체를 막지 못함 — A1+B2 구조적 취약성은 clipping으로 해소 불가

### 4. 보고값과의 비교

| Paper | 보고 RMSE | 비붕괴 RMSE 범위 | 붕괴 RMSE | 해석 |
|---|:---:|---|:---:|---|
| P2 (Qin 2024, EAAI) | 12.14 | 11.85–13.07 | ~41.4 | 비붕괴 시드 보고 가능성 ↑ |
| P3 (Zheng 2017) | 16.18 | 11.91 (seed=2만) | ~41.5 | 운 좋은 단일 시드 또는 다른 patience |
| P5 (Elsherif 2025) | 13.40 | 19.03 (seed=5만) | ~41.4 | 하이퍼파라미터 차이(batch/LR) 주요 변수 |

- P2 보고값 12.14 ∈ 비붕괴 시드 범위(11.85–13.07) — 일치
- P2 단일 시드 실험 가정 시: 60% 확률로 붕괴 결과(RMSE≈41) 보고

---

## 방법론적 한계 및 주의사항

### 프로토콜 추론의 한계
- **P1, P2, P3, P4:** ScienceDirect 접근 제한으로 전문 미확인. 특히 patience 값은 추론에 기반.
- **P5:** 논문에서 epoch 수 확인 가능하나 batch=32, LR=0.0001 vs 본 실험 batch=256, LR=1e-3 차이 존재.
- 모든 논문이 동일한 표준 LSTM 백본(LSTM1→LSTM2→FC)을 사용했다고 가정함.

### 주장 범위
- "이 논문들의 LSTM 기준선이 MPC 상태였다" → **주장 불가** (확증 없음)
- "이 논문들이 서술한 프로토콜을 재현하면 X% 확률로 MPC가 발생한다" → **주장 가능** (직접 실험으로 정량화됨)
- "이 프로토콜로 보고된 RMSE는 X%의 확률로 우연히 살아남은 시드에서 나온 것일 수 있다" → **주장 가능**

### MPC 판정 기준 (Phase 1A와 동일)
```
is_mpc = (PDR < 0.05) AND (R² ≤ 0.0)
  PDR (Prediction Dispersion Ratio): std(pred) / (std(true) + ε)
  R²: coefficient of determination (test set)
```

---

## 코드 설계 결정 기록

**파일:** `Collapse_Study/Data_Analysis/Code/06_external_audit.py`

```python
# 실험 범위
DATASET    = "FD003"          # MPC 발생률이 가장 높은 데이터셋
SEEDS      = list(range(10))  # 10 seeds per paper
BATCH_SIZE = 256
LR, WD     = 1e-3, 1e-4      # 표준 백본 동일 (상위 CLAUDE.md)

# 논문별 차별화 파라미터
P1: patience=15, source=phase1a_reference
P2: patience=20, source=new_experiment       # EAAI 2024 대표
P3: patience=10, source=new_experiment       # Zheng 2017 baseline
P4: clip=None,  source=phase1a_reference     # pre-2020 unclipped 대표
P5: max_epochs=25, patience=999,             # Elsherif 2025
    min_epochs=25, source=new_experiment     # B1_short: 고정 epoch 구현

# val split: 모든 논문 A1_fixed (seed=42)
# checkpointing: runs_new.csv에 매 seed 저장 → 재시작 가능
# P1/P4: Phase1A runs.csv에서 A1+B2+C2 / A1+B2+C1 조건 필터링
```

**B1_short 구현 (P5):**
```python
if paper["es_mode"] == "B1_short":
    patience = max_epochs + 1  # patience 카운터 절대 도달 불가
    min_epochs = max_epochs    # epoch 조건 항상 참 → ES 절대 발동 안 함
```

**P1/P4 Phase1A 참조 로직:**
```python
def load_phase1a_results(paper_id, cond_a, cond_b, cond_c):
    df = pd.read_csv(PHASE1A_CSV)
    mask = (df["cond_a"]==cond_a) & (df["cond_b"]==cond_b) & (df["cond_c"]==cond_c)
    sub = df[mask].copy()
    sub["paper_id"] = paper_id
    sub["patience"] = 15
    # is_collapsed: (test_PDR < 0.05) & (test_R2 <= 0.0)
```

---

## Decision_log.md 연계

- **D3 (외부 audit 포함 결정):** FD2의 audit 대상 선정 + 실험 실행 완료 → **FD2를 D4로 확정 처리 필요**
- Phase1A 기준: A1+B2 interaction MPC rate ≥ 0.60 충족 (P1=0.80) → 외부 audit 포함 결정 정당화됨
- **모든 5편 논문에서 A1+B2 조건 = MPC rate 60–90%** → 외부 audit 기여가 논문의 핵심 증거로 활용 가능
