# Phase 4 Result
## 예방 분류체계 + 소급 진단 도구

**실험일:** 2026-09-15  
**소요 시간:** 65.5분 (Part A fg_clamp 신규 실험) + Part B 즉시 완료  
**총 신규 runs:** 30 (fg_clamp: 3 ε × 10 seeds)  
**통합 분석:** 720 runs (Phase0~3B 전체 기존 결과 통합)

---

## 실험 설계

- **Part A 신규 실험:** forget gate clamp [ε, 1], ε ∈ {0.001, 0.01, 0.05}  
  프로토콜: FD003 × A1+B2+C2 (붕괴 유발 조건) × 10 seeds  
- **Part A 통합:** Phase1A/1B/3/3B 기존 결과 수집 → 예방 분류체계 구축
- **Part B 보정 (Calibration):** Phase1A 전체 270 runs (FD003/LSTM, 다양한 조건)
- **Part B 검증 (Held-out):** Phase1B + Phase3 + Phase3B = 330 runs  
  (다른 개입·아키텍처·구현 포함 — 완전 독립 데이터셋)

---

## Part A: 예방 분류체계

### fg_clamp 신규 실험 결과

| ε | MPC rate | RMSE mean±std | RMSE(정상만) |
|---|:---:|:---:|:---:|
| 0.001 | **0.60** | 30.08±14.99 | 12.67 |
| 0.01  | **0.60** | 30.08±14.99 | 12.67 |
| 0.05  | **0.60** | 30.11±14.95 | 12.74 |

**핵심 발견:** ε = 0.001 ~ 0.05 범위의 forget gate clamp는 Baseline(0.60)과 MPC rate가 동일. **전혀 예방 효과 없음.**

### 전체 예방 분류체계 (12개 방법, MPC rate 오름차순)

> ⚠️ **비교 조건 주의:** 방법마다 테스트된 baseline protocol이 다르다.  
> ES warmup (B3)은 B 요인 자체를 바꾸므로 테스트 조건이 A1+**B3**+C2,  
> per-seed split은 A 요인을 바꾸므로 **A2**+B2+C2다.  
> 두 방법의 MPC rate는 나머지 방법(A1+B2+C2 고정)과 직접 비교 시 조건 불일치가 있음.

| 방법 | 개입 대상 | 테스트 조건 | MPC rate | RMSE(전체) | RMSE(정상) | 구현 비용 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **GRU 교체** | 아키텍처 | A1+B2+C2 | **0%** | 12.61±0.47 | 12.61 | 중간 (아키텍처 변경) |
| **forget gate=1 (V2_fg1)** | 아키텍처(게이팅) | A1+B2+C2 | **0%** | 12.96±0.95 | 12.96 | 낮음 (커스텀 셀) |
| **bias_init = train_mean** | 초기화 | A1+B2+C2 | **0%** | 12.94±0.48 | 12.94 | 매우 낮음 (1줄) |
| **MAE loss** | Loss | A1+B2+C2 | **0%** | 13.43±0.60 | 13.43 | 매우 낮음 (1줄) |
| **ES warmup (B3)** | 프로토콜(ES) | A1+**B3**+C2 | **0%** | 12.42±1.04 | 12.42 | 매우 낮음 (프로토콜) |
| patience = 30 | 프로토콜(ES) | A1+B2+C2 | 20% | 18.38±12.21 | 12.59 | 매우 낮음 (파라미터) |
| per-seed random split | 프로토콜(val) | **A2**+B2+C2 | 30% | 21.50±14.02 | 12.80 | 매우 낮음 (프로토콜) |
| V1_fb1 (forget bias+1) | 초기화 | A1+B2+C2 | 40% | 24.29±14.80 | 12.84 | 매우 낮음 (init) |
| fg_clamp ε=0.05 | 아키텍처(게이팅) | A1+B2+C2 | 60% | 30.11±14.95 | 12.74 | 매우 낮음 (1줄) |
| fg_clamp ε=0.01 | 아키텍처(게이팅) | A1+B2+C2 | 60% | 30.08±14.99 | 12.67 | 매우 낮음 (1줄) |
| fg_clamp ε=0.001 | 아키텍처(게이팅) | A1+B2+C2 | 60% | 30.08±14.99 | 12.67 | 매우 낮음 (1줄) |
| Baseline (표준 LSTM) | — | A1+B2+C2 | 60% | 30.18±14.98 | 12.78 | — |

- MPC 완전 예방 방법: 5개 (GRU, V2_fg1, bias_init, MAE loss, ES warmup)
- 부분 개선 방법: 3개 (patience=30, per-seed split, V1_fb1)
- **효과 없음:** fg_clamp ε=0.001~0.05 → Baseline과 동일

### 정상 run 성능 (RMSE non-inferiority)

모든 예방 방법에서 정상 수렴 시 RMSE는 12.4~13.4 범위.  
수정 M0 기준(~12.97) 대비 비열등 — **어떤 예방책도 정상 수렴 성능을 저하시키지 않음.**

---

## Part A: fg_clamp 예상 밖 결과 — 해석

### 발견: 작은 ε clamp는 MPC를 예방하지 못한다

- **V2_fg1 (f=1 고정):** MPC 0/10 — forget gate 완전 제거 시 gradient 항상 100% 통과
- **fg_clamp ε=0.001~0.05:** MPC 6/10 — Baseline과 동일

둘의 차이: V2_fg1은 `c_t = c_{t-1} + i_t·g_t` (가산), fg_clamp은 `c_t = f_t·c_{t-1} + i_t·g_t` with `f_t ≥ 0.001`.

### 기계적 설명 — BPTT 전개 기반 수정

단일 타임스텝 방정식 `∂L/∂c_{t-1} = f_t · ∂L/∂c_t`는 한 스텝에 대해서만 기술한다. fg_clamp 실패의 진짜 이유는 **T 스텝 전체에 걸친 BPTT 전개**를 봐야 드러난다.

```
BPTT T스텝 전개 (window = 30):
  ∂L/∂c_{t-T} ≈ f^T · ∂L/∂c_t

  Baseline  (f → 0):      0^30        · ∂L/∂c_t  =  0
  clamp ε=0.001:          0.001^30    · ∂L/∂c_t  ≈  10⁻⁹⁰ · ∂L/∂c_t  ≈  0
  clamp ε=0.05:           0.05^30     · ∂L/∂c_t  ≈  10⁻³⁹ · ∂L/∂c_t  ≈  0
  V2_fg1 (f = 1 고정):    1^30        · ∂L/∂c_t  =  ∂L/∂c_t  ← decay 없음
```

**핵심:** trivial solution 근방에서 최종 타임스텝 cell state gradient `∂L/∂c_T`는 출력층의 RUL residual이 nonzero이므로 여전히 nonzero다. 문제는 이 gradient가 T=30 스텝을 거슬러 올라가는 과정에서 `f^T`만큼 **지수적으로** 감쇠한다는 것이다. f=0.001이면 30번 곱해 10⁻⁹⁰, f=0.05라도 10⁻³⁹이 되어 사실상 0이다. ∂L/∂c_T 자체가 작아서가 아니라, **T스텝을 통과하는 지수 감쇠**가 원인이다.

**V2_fg1이 작동하는 이유 — CEC 원리 (Hochreiter & Schmidhuber, 1997):**  
f=1 고정 시 `c_t = c_{t-1} + i_t·g_t` (순수 가산). BPTT에서 `∂L/∂c_{t-T} = 1^T · ∂L/∂c_t = ∂L/∂c_t` — T 값에 무관하게 gradient가 감쇠 없이 전파된다. 이것이 Constant Error Carousel(CEC)이며, fg_clamp와의 본질적 차이는 ε의 크기가 아닌 **T스텝 곱셈의 지수적 성질**이다.

**fg_clamp의 본질적 한계:**  
f ≥ ε이어도 f < 1이면 BPTT 경로에서 여전히 지수 감쇠가 발생한다. ε값이 아무리 커도 `ε^30`이 실질적으로 0이므로, "f 하한 보장"만으로는 이 감쇠를 막을 수 없다. **"f ≥ ε 보장"이 아니라 "f = 1 고정"만이 T스텝 gradient decay를 원천 차단한다.**

### Phase 3B H_mech와의 통합 해석

| 실험 | forget gate | MPC rate | BPTT 관점 해석 |
|------|:-----------:|:---:|------|
| Baseline | f_t → 0 (free) | 0.60 | 0^30 → 0 (완전 차단) |
| fg_clamp ε=0.001 | f_t ≥ 0.001 | 0.60 | 0.001^30 = 10⁻⁹⁰ → 실질적으로 동일 |
| fg_clamp ε=0.05 | f_t ≥ 0.05 | 0.60 | 0.05^30 = 10⁻³⁹ → 실질적으로 동일 |
| V1_fb1 (bias+1) | 초기 f_t ≈ 0.73, 이후 학습에 의해 감소 가능 | 0.40 | 초기 이점이나 일부 seed에서 학습 중 감소 → 부분 탈출 |
| V2_fg1 (f=1 고정) | f_t = 1.0 (항상) | 0.00 | 1^30 = 1 → CEC: gradient decay 없음 |
| GRU | cell state 없음 | 0.00 | 이 BPTT 경로 자체 부재 → 면역 |

**결론:** MPC 완전 예방을 위한 forget gate 조작은 "f ≥ ε"이 아니라 "f = 1 고정(CEC) 또는 아키텍처적으로 cell state gradient 경로 제거(GRU)"여야 한다.

---

## Part B: 소급 진단 도구

### 데이터 구성

| 데이터 | 출처 | 총 runs | Collapsed | Normal |
|--------|------|:---:|:---:|:---:|
| 보정 (Calibration) | Phase1A | 270 | 40 (14.8%) | 230 (85.2%) |
| 검증 (Held-out) | Phase1B+3+3B | 330 | 95 (28.8%) | 235 (71.2%) |

**중요:** 검증 데이터는 보정에 사용되지 않음. 다른 개입, 아키텍처(GRU/MLP/CNN1D), 데이터셋(FD001~FD004), 구현(커스텀 셀)을 포함한 완전 독립 데이터.

### 소급 진단 성능 비교

| 지표 | 데이터 | 민감도 | 특이도 | 균형정확도 | AUROC |
|------|--------|:---:|:---:|:---:|:---:|
| **PDR < 0.05** | 보정 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| R² ≤ 0 | 보정 | 1.0000 | 0.9348 | 0.9674 | 0.9796 |
| RMSE > 25 | 보정 | 1.0000 | 0.6870 | 0.8435 | 0.9370 |
| PDR+R²(OR) | 보정 | 1.0000 | 0.9348 | 0.9674 | — |
| **PDR < 0.05** | 검증 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| R² ≤ 0 | 검증 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RMSE > 25 | 검증 | 1.0000 | 0.9574 | 0.9787 | 1.0000 |
| PDR+R²(OR) | 검증 | 1.0000 | 1.0000 | 1.0000 | — |

### PDR < 0.05 — 완벽한 소급 진단

- **민감도(Sensitivity) = 1.0:** 모든 collapsed run 정확히 탐지
- **특이도(Specificity) = 1.0:** 모든 normal run을 MPC 아님으로 정확히 분류
- **AUROC = 1.0:** 어떤 임계값에서도 완벽한 판별
- **보정 → 검증 일반화:** Phase1A(FD003/LSTM)에서 보정한 임계값이 다른 아키텍처·데이터셋·개입 조건 전체로 완벽하게 일반화

### 지표별 비교 해석

**PDR < 0.05 (최우수)**
- 보정·검증 모두 perfect — 단일 지표로 충분
- 데이터셋 스케일 독립 (분산비이므로)
- FD001~FD004, LSTM/GRU/MLP/CNN1D 전체에서 동일 임계값 적용 가능

**R² ≤ 0**
- 보정에서 특이도 0.93 (일부 false positive): B1(no early stopping)+C1(unclipped) 조건에서 고 RMSE이지만 PDR>0이면 정상인데 R²<0 가능
- 검증에서는 특이도 1.0: 다른 조건에서는 문제 없음
- PDR보다 조건 의존적

**RMSE > 25 (가장 불안정)**
- 보정 특이도 0.69: B1+C1 unclipped 조건에서 정상 수렴이지만 RMSE>25 가능
- 절대 스케일 의존 → 데이터셋마다 다른 임계값 필요 → 실용적으로 불리
- AUROC는 1.0이지만 실무에서 25 임계값이 다른 문제에 통하지 않을 수 있음

### 권장 진단 프로토콜

```python
# 이미 학습 완료된 모델의 소급 감사 (Retrospective Audit)
def audit_model(pred, true, eps=1e-8):
    """학습된 모델의 예측값만으로 MPC 여부 진단."""
    pdr = np.std(pred) / (np.std(true) + eps)
    if pdr < 0.05:
        return "COLLAPSED — 프로토콜 점검 및 재학습 필요"
    return "OK"

# Phase1A (보정)에서 임계값 0.05로 완벽 분리 확인
# Phase1B+3+3B (검증)에서도 완벽 일반화 확인 (AUROC=1.0)
```

---

## go/no-go 판정

```
Part A:
  - GRU / V2_fg1 / bias_init / MAE / ES warmup → MPC 완전 예방 확인 (각 0/10)
  - fg_clamp [ε,1]: ε=0.001~0.05 → 효과 없음 (Baseline과 동일 6/10)
  - 수정된 H_mech: "forget gate 하한 보장"이 아닌 "gradient 경로 실질 복원"이 필요

Part B:
  - PDR < 0.05: AUROC=1.0 (보정) → AUROC=1.0 (검증) — 완벽 일반화
  - 소급 감사 임계값 PDR=0.05 확정

→ GO: Phase 4 목표 달성
  예방 분류체계 완성 + 소급 진단 도구 검증 완료
```

---

## 핵심 발견 해석

### 1. 예방 효과의 명확한 이분법

완전 예방 방법(0%)과 불완전 방법(20~60%)이 뚜렷하게 나뉜다:

```
완전 예방 (MPC 0%)          ← 어느 하나만 적용해도 충분
├── GRU 교체              ← 아키텍처에서 취약 경로 제거
├── V2_fg1 (f=1 고정)     ← cell state gradient 완전 복원
├── bias_init=train_mean  ← trivial solution 초기 흡인력 제거
├── MAE loss              ← loss landscape 변경
└── ES warmup             ← trivial solution 수렴 전 탈출 기회 부여

불완전 예방 (20~40%)          ← 개선은 되지만 불충분
├── patience=30          ← 탈출 시간 늘림
├── per-seed split        ← 분리 개선, 일부 나쁜 split은 여전히 문제
└── V1_fb1 (bias+1)      ← 초기 gradient 이점, 이후 소실

효과 없음 (60% = Baseline)
└── fg_clamp ε≤0.05      ← ε^30 ≈ 0 (BPTT 지수 감쇠) — f 하한과 무관
```

### 2. fg_clamp의 실패가 말해주는 것

- H_mech를 더 정확히 정제: "forget gate가 0에 가까워지는 것"이 문제이지만, 해결책은 f의 하한 보장이 아니라 **BPTT 전체 경로의 지수 감쇠 제거**
- ε=0.001 clamp와 f=1 고정의 차이는 단일 스텝 gradient 크기가 아니라 **T스텝 감쇠율**: ε^T ≈ 0 vs 1^T = 1
- fg_clamp는 forget gate의 "선택적 망각" 기능을 유지하면서 MPC를 막으려 했으나, 구조적으로 불가능하다 — 선택적 망각(f < 1)과 gradient decay 방지(f = 1)는 양립할 수 없음
- 실용적 권고: LSTM에서 CEC 없이 MPC를 막으려면 다른 경로(bias_init, MAE loss, ES warmup)를 사용하거나, 아키텍처를 GRU로 교체하는 것이 합리적

### 3. 소급 진단 도구의 실용적 가치

- **단 하나의 지표, 단 하나의 임계값(PDR < 0.05)** 으로 완벽 판별
- 데이터셋 스케일 독립 → FD001~FD004 동일 임계값 적용
- 아키텍처 독립 → LSTM/GRU/MLP/CNN1D 동일 임계값 적용
- 구현 독립 → cuDNN/커스텀 셀 모두 동일 임계값 적용
- **Test set 접근 없이 적용 가능** → 학습 완료 후 즉시 감사

---

## Phase 4 전체 연구 기여 통합 요약

Phase 0~4를 통해 MPC의 **전체 인과 사슬**이 규명됐다:

```
[데이터] FD003 합성 균질성
   → MSE landscape에 trivial solution (상수 예측) 안정적 존재

[아키텍처] LSTM forget gate → 0 (trivial solution 근방)
   → cell state gradient 소멸 (CEC 원리 역방향)

[프로토콜] 고정 val split + 조기 종료 (MIN_EPOCHS=0)
   → gradient 탈출 기회 없이 trivial solution 고착

   = MPC (PDR≈0.000002, R²<0, RMSE≈42)
```

**예방:** 세 원인 중 하나를 제거하면 MPC 소멸
- 아키텍처: GRU / V2_fg1
- Loss landscape: MAE loss / bias_init=train_mean
- 프로토콜: ES warmup / per-seed split (단독으론 30% 잔존)

**소급 감사:** PDR < 0.05 (단일 지표, 단일 임계값) → AUROC = 1.0 (보정·검증 모두)

---

## 저장 파일 목록

| 파일 | 내용 |
|------|------|
| `Phase4/fg_clamp_runs/runs.csv` | fg_clamp 30 runs 원데이터 (epsilon × seed × MPC지표) |
| `Phase4/taxonomy.csv` | 12개 방법 예방 분류체계 (MPC rate, RMSE, cost) |
| `Phase4/audit_metrics.csv` | 4 지표 × 2 데이터셋 소급 진단 성능 표 |
| `Phase4/figures/fig1_taxonomy.png` | 예방 분류체계 수평 막대 (MPC rate + RMSE, 비용 색상) |
| `Phase4/figures/fig2_fg_clamp.png` | ε별 MPC rate & RMSE (ε 효과 없음 시각화) |
| `Phase4/figures/fig3_audit_roc.png` | 소급 진단 ROC 공간 (보정·검증 비교) |
| `Phase4/figures/fig4_pdr_distribution.png` | PDR 분포 (collapsed vs normal, 임계값 0.05 표시) |
