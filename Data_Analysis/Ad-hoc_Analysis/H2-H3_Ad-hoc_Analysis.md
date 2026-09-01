# H2–H3 Unified-Control Ad-hoc Analysis

> **작성일:** 2026-08-26
> **목적:** Tier 2(정규화)와 Tier 3(결함 모드 아키텍처) 비교를 단일 통제 프로토콜 하에서 재검증하여, 논문의 three-tier ordering 주장을 cross-hypothesis 프로토콜 차이 없이 지지할 수 있는지 확인
> **관련 섹션:** §V.A Limitations (5번째 제한사항 — H2/H3 프로토콜 불일치)
> **예상 실행 수:** 50 runs (선택적 확장 포함 시 60 runs)

---

## 1. 배경 및 동기

원 연구의 §V.A는 H2(정규화)와 H3(결함 모드 아키텍처) 간 직접 RMSE 비교가 네 가지 프로토콜 차이로 인해 제한된다고 명시한다:

| 항목 | H2 (H5_normalization) | H3 (H6_fault_mode) |
|------|----------------------|---------------------|
| Backbone | LSTM₂ hidden=32, FC(32→16→ReLU→1) | LSTM₂ hidden=64, FC(64→32→ReLU→1) |
| 입력 피처 | 17–23 (sensor + op1/op2/op3, dataset 의존적) | 14–20 (sensor only; op columns 제거) |
| Validation split | 결정론적: 마지막 20% 엔진 (unit ID 순) | 랜덤: `RandomState(seed)` |
| 예측값 클리핑 | 평가 시 [0, 125] 적용 | 미적용 |

이로 인해 H2 FD003 baseline RMSE(19.05 ± 12.86)와 H3 M0 baseline RMSE(43.23 ± 0.18)는 서로 다른 실험 조건을 반영하므로, 두 tier의 effect size를 직접 비교하는 것이 불가능하다.

**이 ad-hoc 분석의 질문:** 동일한 프로토콜 하에서 N1(Fleet MinMax) → N3(Per-unit MinMax) 교체로 발생하는 RMSE 변화(Tier 2 효과)와 M0 → M3 교체로 발생하는 RMSE 변화(Tier 3 효과)의 상대적 크기는 어떻게 되는가?

---

## 2. 통제된 단일 프로토콜 (Unified Protocol)

### 2.1 선택 기준

- Backbone: **H3 full-capacity** (LSTM₂ hidden=64) 채택 → 더 많이 검증되었고 논문 핵심 결과의 기반
- 피처: **sensor-only** (op columns 제외) → K-means residualization으로 operating condition 처리
- Validation split: **랜덤 엔진 레벨 20%** (`RandomState(seed)`) → H3 방식 채택
- 예측값 클리핑: **[0, 125] 적용** → H2의 평가 관행 채택 (더 보수적, 실용적)
- RUL 레이블 클리핑: **clip = 125 cycles** (고정)
- Seeds: `[0, 1, 2, 3, 4]` (5회 반복)

### 2.2 프로토콜 상세

```
공통 하이퍼파라미터 (CLAUDE.md 표준과 동일)
─────────────────────────────────────────────
Window     : 30 cycles
Batch size : 256
Optimizer  : Adam (lr=1e-3, weight_decay=1e-4)
Patience   : 15 epochs (early stopping)
Backbone   : StackedLSTM — LSTM1(64, return_seq=True) → Dropout(0.2)
                         → LSTM2(64, return_last) → Dropout(0.2)
                         → FC(64→32→ReLU→1)
Seeds      : [0, 1, 2, 3, 4]
RUL clip   : 125 cycles
Eval clip  : predictions clamped to [0, 125]

피처 세트 (dataset별)
─────────────────────
FD001/FD003 : 14 features (s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21)
FD002/FD004 : 20 features (상동 + s6 제외 불필요 센서 제거; op_condition_utils로 residualize)

Operating condition 처리
────────────────────────
FD001/FD003 : 해당 없음
FD002/FD004 : fit_op_condition_kmeans(train_df, k=6) → apply_op_residual()
              (shared/op_condition_utils.py 사용, train-only fit)

Validation split
────────────────
엔진 레벨 랜덤 20% hold-out; split seed = 실험 seed와 동일 (RandomState(seed))
```

---

## 3. 실험 조건

### 3.1 핵심 비교 조건 (필수, 50 runs)

| 조건 ID | 정규화 | 모델 | 데이터셋 | Runs | 목적 |
|---------|--------|------|---------|------|------|
| **A** | N1 (Fleet MinMax) | M0 (Single LSTM) | FD001–FD004 | 4×5 = **20** | Tier 3 비교 기준선; Tier 2 비교 기준선 |
| **B** | N3 (Per-unit MinMax) | M0 (Single LSTM) | FD001–FD004 | 4×5 = **20** | Tier 2 효과 정량화 (정규화 변화 효과) |
| **C** | N1 (Fleet MinMax) | M3 (Attention Gate) | FD003, FD004 | 2×5 = **10** | Tier 3 효과 정량화 (아키텍처 변화 효과) |

**총 필수 실험 수: 50 runs**

> **N3 선택 이유:** Per-unit MinMax는 원 H5에서 두 번째로 흔히 쓰이는 전략이며 FD003에서 가장 큰 RMSE 증가를 보인 대표 비교 대상. RevIN(N7)은 모델 서브클래스 구현이 필요하여 단일 프로토콜 적용이 복잡하므로 제외.

### 3.2 선택적 확장 (필요 시, +10 runs)

| 조건 ID | 정규화 | 모델 | 데이터셋 | Runs | 목적 |
|---------|--------|------|---------|------|------|
| **D** | N3 (Per-unit MinMax) | M3 (Attention Gate) | FD003, FD004 | 2×5 = **10** | 상호작용 확인 (normalization × architecture) |

> **D 조건 추가 시 가능한 분석:** N3+M3 vs N1+M3 비교로 "정규화 효과가 아키텍처 효과와 독립적인가?" 검증. 논문 §V.B의 "fault-mode heterogeneity가 normalization ranking에 영향" 주장 보강에 활용 가능.

---

## 4. 기대 결과 및 검증 가설

### 가설 1 (Tier 2 vs Tier 3 ordering 유지)
> 통일 프로토콜 하에서도, FD003에서 M0→M3 RMSE 감소폭이 N3→N1 RMSE 감소폭보다 유의미하게 크다.

- **기대:** |ΔRMSE(M0→M3, FD003)| >> |ΔRMSE(N3→N1, FD003)|
- **기각 조건:** 두 effect size가 통계적으로 구분 불가하거나 역전될 경우 → §V.A 제한사항이 실질적 문제임을 의미

### 가설 2 (FD003 결과 재현성)
> Unified protocol 하에서 N1+M3의 FD003 RMSE가 원 H3 결과(14.78 ± 1.32)와 통계적으로 동등하다.

- **기대:** RMSE 15.5 이하, 원 결과와 Wilcoxon p > 0.05 (비열등성)
- **주의:** backbone이 동일하므로 큰 차이는 없어야 함; 단 validation split 변경이 분산에 영향 가능

### 가설 3 (FD001/FD002/FD004 안정성)
> M3가 단일 결함 데이터셋(FD001, FD002)과 FD004에서 M0 대비 통계적 차이를 보이지 않는다.

- **기대:** FD001/FD002: ΔRMSE ≈ 0, p > 0.05; FD004: ΔRMSE < +5%

---

## 5. 통계 분석 계획

### 5.1 주요 비교

```
효과 크기 계산 (각 데이터셋별)
─────────────────────────────
Tier 2 효과: ΔRMSE_norm = RMSE(N3+M0) − RMSE(N1+M0)  [5 seeds]
Tier 3 효과: ΔRMSE_arch = RMSE(N1+M0) − RMSE(N1+M3)  [5 seeds, FD003/FD004만]

상대 효과: ratio = ΔRMSE_arch / ΔRMSE_norm  [FD003]
```

### 5.2 통계 검정

- **검정:** Wilcoxon rank-sum (one-sided, α=0.05)
- **비교 가족:** 8쌍 (4 datasets × {norm effect, arch effect})
- **보정:** Benjamini-Hochberg FDR
- **효과 크기:** Cohen's d (pooled SD)
- **유의 기준:** p_BH < 0.05 AND |d| ≥ 0.3

### 5.3 핵심 결과 표 형식

| Dataset | RMSE: N1+M0 | RMSE: N3+M0 | RMSE: N1+M3 | Tier2 Δ | Tier3 Δ | Tier3/Tier2 |
|---------|------------|-------------|-------------|---------|---------|-------------|
| FD001 | — | — | (N/A) | — | — | — |
| FD002 | — | — | (N/A) | — | — | — |
| FD003 | — | — | — | — | — | **핵심** |
| FD004 | — | — | — | — | — | — |

---

## 6. 코드 재사용 계획

### 6.1 재사용할 기존 모듈

| 모듈 | 위치 | 용도 |
|------|------|------|
| `op_condition_utils.py` | `Code/shared/` | FD002/FD004 K-means residualization |
| `h6_p2_model_utils.py` | `Code/H6_fault_mode/phase2_models/` | StackedLSTM (M0), M3AttentionGate, train_model(), nasa_score() |
| `01_data_loader.py` | `Code/H5_normalization/` | 데이터 로딩 + RUL 레이블 (수정 필요: op columns 제거) |
| `02_normalizers.py` | `Code/H5_normalization/` | Fleet MinMax (N1), Per-unit MinMax (N3) |

### 6.2 신규 작성 스크립트 (최소 구성)

```
Code/Ad-hoc_Analysis/
├── 01_unified_data_loader.py   ← H5 data loader 기반, sensor-only + op_residual 통합
├── 02_run_unified.py           ← 조건 A/B/C 일괄 실행 (config dict 방식)
└── 03_analyze_results.py       ← Tier2 vs Tier3 effect size 비교 + 시각화
```

### 6.3 `01_unified_data_loader.py` 핵심 설계

```python
# 원 H5와 다른 점:
# 1. op columns (op1, op2, op3) 피처에서 제외 (FD002/FD004는 residualize 후 제거)
# 2. validation split: RandomState(seed) 기반 랜덤 20% (H3 방식)
# 3. 예측 클리핑: evaluate() 단계에서 preds.clip(0, 125) 적용

def load_dataset_unified(fd_id, normalizer, seed, clip=125):
    # 1. 원시 데이터 로드
    # 2. FD002/FD004: fit_op_condition_kmeans → apply_op_residual (train-only fit)
    # 3. sensor-only 피처 선택 (op columns 제거)
    # 4. normalizer.fit(train) → transform(train, test)
    # 5. RandomState(seed) 기반 엔진 레벨 train/val split
    # 6. window=30 시퀀스 구성 (zero-pad if len < 30)
    return train_loader, val_loader, test_seqs, test_rul
```

### 6.4 `02_run_unified.py` 실험 매트릭스

```python
CONDITIONS = [
    # (condition_id, normalizer, model_type, datasets)
    ('A', 'N1', 'M0', ['FD001','FD002','FD003','FD004']),
    ('B', 'N3', 'M0', ['FD001','FD002','FD003','FD004']),
    ('C', 'N1', 'M3', ['FD003','FD004']),
    # ('D', 'N3', 'M3', ['FD003','FD004']),  # 선택적
]
SEEDS = [0, 1, 2, 3, 4]
# 총 50 runs (D 포함 시 60 runs)
```

---

## 7. 예상 소요 시간

| 조건 | Runs | 예상 시간/run | 합계 |
|------|------|------------|------|
| A: N1+M0 × 4 datasets | 20 | ~3분 | ~60분 |
| B: N3+M0 × 4 datasets | 20 | ~3분 | ~60분 |
| C: N1+M3 × FD003/FD004 | 10 | ~5분 | ~50분 |
| **합계** | **50** | | **~2.8시간** |

> M3는 GatingNet 추가 학습으로 M0 대비 약 30–50% 더 소요. GPU 가용 시 절반 이하.

---

## 8. 결과 저장 위치

```
Data_Analysis/Results/Ad-hoc_Analysis/
├── unified_results.csv         ← 전체 결과 (condition, dataset, seed, RMSE, NASA)
├── tier_comparison.csv         ← Tier2 vs Tier3 효과 크기 요약
├── statistical_tests.csv       ← Wilcoxon + BH-FDR 결과
└── figures/
    ├── tier_effect_comparison.png   ← 주요 bar chart
    └── rmse_by_condition.png
```

---

## 9. 논문 활용 방안

### 시나리오 A: 가설 1 지지 (tier ordering 유지)
→ §V.A Limitations에 보완 문장 추가:
> "A post-hoc 50-run unified-protocol replication (identical backbone, feature set, validation split, and prediction clipping across N1, N3, M0, and M3) confirmed that the architecture effect (M0→M3 on FD003: ΔRMSE = X cycles) substantially exceeded the normalization effect (N3→N1: ΔRMSE = Y cycles) under a controlled comparison, supporting the three-tier priority ordering independent of the original cross-hypothesis protocol differences."

### 시나리오 B: 가설 1 기각 (ordering 역전 또는 무효화)
→ §V.A에 솔직한 제한 사항으로 기술:
> "A unified-protocol replication suggests that the tier ordering may be sensitive to backbone capacity differences across hypotheses; the relative priority of normalization vs. fault-mode architecture requires further investigation under matched experimental conditions."
→ Contribution (i) 주장 약화 필요 여부 검토

---

## 10. 의사결정 체크리스트

- [ ] `01_unified_data_loader.py` 작성 및 단위 테스트 (FD002/FD004 residualization 확인)
- [ ] 조건 A 실행 (20 runs) → 원 H5 N1 결과와 비교하여 loader 검증
- [ ] 조건 B 실행 (20 runs)
- [ ] 조건 C 실행 (10 runs) → 원 H6 M3 결과와 비교 검증
- [ ] `03_analyze_results.py` 실행: Tier2/Tier3 effect size 계산
- [ ] 통계 검정 완료
- [ ] 결과에 따라 논문 §V.A 또는 §V.F 업데이트 여부 결정
- [ ] (선택) 조건 D 실행 (10 runs) — 상호작용 항 검토

---

## 11. Condition C 실증 분석 — mean-prediction collapse 탈출구 검증 (2026-08-27)

### 11.1 배경

`Ad-hoc_Analysis_Result.md` Section 13의 수정 프로토콜 실험 결과에서 **원본 H6 M3의 65.8% RMSE 감소 주장이 무효화**됐다. 수정된 비교 기준:

| | FD003 | FD004 |
|---|---|---|
| 원본 H6 M0 (버그 포함) | 43.23 ± 0.18 ← mean-prediction collapse | 28.05 |
| 수정 H6 M0 | 12.97 ± 0.67 | 18.96 ± 3.97 |
| 수정 H6 M3 | 13.24 ± 1.69 | 17.30 ± 1.04 |

이 시점에서 다음 가설이 제기됐다:

> **탈출구 가설:** 이 문서의 섹션 1이 인용한 "H2 FD003 baseline RMSE = 19.05 ± 12.86"은 고분산 상태다. 만약 통합 실험(01_unified_data_loader.py)의 동일 프로토콜 하에서 M3(Condition C)가 M0(Condition A, 19.05 수준) 대비 ~30% 개선을 보인다면, 65.8%가 아닌 더 낮지만 유효한 개선 수치로 논문을 방어할 수 있다.

이 가설을 검증하기 위해 이미 실행된 Condition C 데이터를 분석한다.

---

### 11.2 목적

1. 통합 프로토콜(01_unified_data_loader.py) 하에서 N1+M0(Condition A)와 N1+M3(Condition C)의 실제 FD003/FD004 RMSE를 비교한다.
2. "19.05 ± 12.86"이 Condition A의 실제 결과인지 확인하고, 탈출구 가설의 전제 자체가 성립하는지 검증한다.
3. 결과에 따라 논문 revision 방향을 결정한다.

---

### 11.3 실험 결과 (unified_results.csv)

#### 시드별 RMSE 원본 데이터

| 시드 | A: N1+M0 FD003 | C: N1+M3 FD003 |
|---|---|---|
| 0 | 13.13 | **16.18** |
| 1 | 12.17 | 12.08 |
| 2 | 12.72 | 12.71 |
| 3 | 12.86 | 12.17 |
| 4 | 13.99 | 13.06 |
| **mean ± std** | **12.97 ± 0.67** | **13.24 ± 1.69** |

| 시드 | A: N1+M0 FD004 | C: N1+M3 FD004 |
|---|---|---|
| 0 | 19.98 | 18.21 |
| 1 | 15.30 | 18.76 |
| 2 | 20.58 | 18.35 |
| 3 | 15.35 | 18.58 |
| 4 | 18.38 | 15.62 |
| **mean ± std** | **17.92 ± 2.50** | **17.90 ± 1.29** |

#### 핵심 발견: 탈출구 가설의 전제 붕괴

| 확인 사항 | 사실 |
|---|---|
| Condition A FD003가 "19.05 ± 12.86"인가? | **아니다. 12.97 ± 0.67** |
| Condition C(M3)가 Condition A(M0)보다 낮은 RMSE인가? | **아니다. FD003: 13.24 > 12.97 (M3가 소폭 악화)** |
| FD004에서 M3가 유의미하게 개선되는가? | **아니다. 17.90 vs 17.92 (사실상 동일)** |

---

### 11.4 "19.05 ± 12.86"의 실제 출처 분석

섹션 1에서 인용된 FD003 baseline 19.05 ± 12.86은 **원본 H5 정규화 실험 (H5_normalization 스크립트)** 의 결과였으며, 이 실험은 다음과 같이 통합 실험과 다른 설정을 사용했다:

| 항목 | 원본 H5 (19.05) | 통합 실험 (12.97) |
|---|---|---|
| Backbone | LSTM₂ hidden=**32**, FC(32→16→1) | LSTM₂ hidden=**64**, FC(64→32→1) |
| Validation split | 결정론적 마지막 20% (unit ID 순, seed 불변) | `RandomState(seed)` 랜덤 20% |
| 평가 클리핑 | 미적용 | [0, 125] 적용 |

따라서 19.05 ± 12.86의 **고분산은 fault-mode 이질성의 신호가 아니라 소규모 백본 + 결정론적 val split 조합의 프로토콜 아티팩트**다. 이 수치는 통합 프로토콜 하에서 M3와 비교하는 valid한 기준선으로 사용될 수 없다.

---

### 11.5 결과에 따른 방향

탈출구 가설은 **양쪽 전제 모두 성립하지 않아 완전히 닫혔다:**

```
전제 1: Condition A M0 FD003 ≈ 19.05 (고분산)  → 실제: 12.97 ± 0.67 (안정)
전제 2: Condition C M3 FD003 < M0              → 실제: 13.24 > 12.97 (소폭 악화)
∴ "통합 프로토콜 하에서도 M3가 M0를 개선한다"는 주장 불가
```

#### 시나리오별 논문 방향 결정표

| 시나리오 | 조건 | 방향 |
|---|---|---|
| (실제) M3 = M0, 고분산 기준선 없음 | Condition C ≈ A, FD003 std 이미 안정 | 65.8% 주장 완전 철회; M3 기여는 "M1 collapse 방지 + FD004 std 안정화"로 재정의 |
| (만약) M3 < M0, 유의하게 개선 | Wilcoxon p_BH < 0.05 | 개선 수치를 새 기여로 제시 (본 경우 해당 없음) |
| (만약) M3 FD003 고분산 유지 시 M3가 해소 | M0 std >> M3 std | std 감소가 실용적 기여로 제시 가능 (본 경우 M0 std=0.67로 이미 안정) |

#### 확정 방향 (2026-08-27)

수정 프로토콜 및 통합 프로토콜 양쪽에서 동일한 결론:

> **M3 FD003에서 M0 대비 통계적으로 유의한 개선 없음. 65.8% 수치 복원 불가.**

논문 revision 방향은 `Ad-hoc_Analysis_Result.md` Section 13.5(D3)에 서술된 재프레임으로 확정:
- Tier 2 아키텍처 = 성능 향상 레버 → **설계 경계 (M1 실패 방지, M2/M3 기준선 회복)**
- 핵심 기여점 = H2 정규화 ablation (Fleet MinMax 유의 우위)
- M3의 잔여 가치 = FD004 훈련 안정성 향상 (std 2.50→1.29, 통계 검정 미수행)
