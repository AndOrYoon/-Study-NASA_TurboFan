# NASA CMAPSS TurboFan — 가설 검증 분석 계획
## H2 · H5 · H6 · H7 Implementation Plan

> **최초 작성:** 2026-07-02 (H5·H6·H7)  
> **개정 1:** 2026-07-02 — H2 추가, 리뷰어 검토 반영 (`Hypothesis_critics.md`)  
> **개정 2:** 2026-07-02 — §0 공통 방법론 프레임워크, RevIN(H5·N7), Attention-Gate(H6·M3), Pinball+Composite Loss(H7·L6·L7) 추가  
> **기반 문서:** `Hypothesis/Hypothesis_PossibleValidation.MD`, `Hypothesis/Hypothesis_critics.md`  
> **작업 디렉토리:** `C:\BMAD_PY313\`  
> **Python 환경:** `C:\BMAD_PY313\NASA_TurboFan\` (Python 3.13.14)

---

## 개요 — 실험 우선순위 및 의존성

| 순위 | 가설 | Research Gap | 실험 복잡도 | 선행 조건 |
|------|------|------------|------------|---------|
| 1 | **H5** — Per-unit 정규화 | HIGH (peer-reviewed 비교 0편) | 중간 | 없음 (즉시 착수) |
| 2 | **H6** — 복합 고장 모드 멀티-브랜치 | HIGH (unsupervised DL 0편) | 높음 | Phase 0 EDA ✅ 완료 |
| 3 | **H7** — 비대칭/가중 손실 함수 | MEDIUM (cross-dataset 0편) | 중간 | numpy/scipy로 Phase 1 가능 |
| 4 | **H2** — 클리핑 임계값 최적화 | MEDIUM (H7 결합 시 HIGH) | 낮음 | H7과 통합 설계 |

**권장 실행 순서:** H5 + H6 Phase 1 병행 → H2 + H7 Phase 1 병행 → H6 Phase 2~3 → H7 Phase 2b

---

## 사전 준비 — 패키지 설치

현재 설치됨: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install tqdm
```

> **Python 3.13 호환성:** TensorFlow 제외. PyTorch 2.5+ 는 Python 3.13 지원.

---

## §0. 공통 실험 방법론 프레임워크

### 0.1 가설 검증 원칙

본 연구는 4개 가설(H2·H5·H6·H7)을 **사전 명세(pre-specified) 비교 실험**으로 검증한다. 비교군, 성공 기준, 통계 검정 방법을 결과 확인 전에 아래와 같이 정의하여 사후 p-hacking을 방지한다.

### 0.2 통계 검정 프레임워크

| 검정 단계 | 방법 | 적용 범위 |
|---------|------|---------|
| **1차 유의성** | Wilcoxon rank-sum test (단측, α = 0.05) | 처치군 vs baseline 엔진별 RMSE 비교 |
| **다중 비교 보정** | Benjamini-Hochberg FDR (α_FDR = 0.05) | 각 가설 내 복수 처치군 동시 비교 |
| **효과 크기** | Cohen's d (paired differences, pooled SD) | 통계적 유의성 + 실용적 유의미성 구분 |
| **신뢰구간** | Bootstrap 10,000회 percentile CI | 점추정값 불확실성 정량화 |

**해석 기준:**
- p_BH < 0.05 **AND** |d| ≥ 0.3 (small effect) → **유의미한 개선** (논문 주장 가능)
- p_BH < 0.05 이지만 |d| < 0.1 → "통계적 유의 / 실용적 기여 없음" 으로 논문에 명시
- 95% CI가 0을 포함하지 않으면 추가 근거로 활용

### 0.3 가설별 성공 기준 (Pre-registered)

| 가설 | Primary 지표 | 성공 임계값 | Secondary 확인 사항 |
|------|------------|------------|------------------|
| H2 | RMSE (FD003·FD004) | clip_125 대비 ≥ 5% 감소 | NASA Score ≥ clip_125 수준 유지 |
| H5 | RMSE (FD001, all groups) | fleet_minmax 대비 ≥ 5% 감소 | boundary group 개선 또는 동등 |
| H6 | RMSE (FD003·FD004) | M0(Baseline) 대비 ≥ 10% 감소 | NASA Score 개선, Silhouette > 0.5 |
| H7 | NASA Score per engine | L1(MSE) 대비 ≥ 10% 감소 | RMSE 악화 < 5% (트레이드오프 허용) |

### 0.4 공통 LSTM 백본 사양

모든 LSTM 실험에서 아래 아키텍처를 통일하여 전처리·손실 함수·클러스터링의 독립 효과를 분리 측정한다. 개별 가설 섹션에서 동일 사양이 반복되는 경우 이 절이 우선한다.

```
입력: (batch, window=30, n_features)
→ LSTM(hidden=64) → Dropout(0.2)
→ LSTM(hidden=32) → Dropout(0.2)
→ Linear(32→16) → ReLU → Linear(16→1)
```

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) | |
| Batch size | 256 | |
| Max epochs | 100 | |
| Early stopping | patience=15, monitor=val_loss | |
| Val split | 엔진별 후반 20% cycles | train 누수 없음 |
| Seeds | [0, 1, 2, 3, 4] | mean ± std 보고 |

```python
def set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### 0.5 문헌 비교 기준선

결과 논문 Table에 아래 문헌 수치를 포함한다. 재현 불가 시 원 논문 인용값 그대로 기재.

| 모델 | FD001 RMSE | FD003 RMSE | 출처 |
|------|-----------|-----------|------|
| LSTM (Zheng et al. 2017) | 16.14 | — | CMAPSS 표준 LSTM baseline |
| Bi-LSTM (Zhang et al. 2018) | 14.74 | — | 양방향 LSTM |
| DCNN (Li et al. 2018) | 12.61 | — | CNN baseline |
| CAELSTM (Meng et al. 2025) | 13.40 | — | 최신 비교 기준 |

### 0.6 재현성 프로토콜

- 각 실험 구성마다 5개 시드([0,1,2,3,4]) 실행 → **mean ± std** 보고
- 결과 CSV에 시드별 raw metric 전체 보존 (사후 재현 및 bootstrap용)
- 패키지 버전을 `requirements.txt`에 고정 (`pip freeze > requirements.txt`)

### 0.7 백본 민감도 분석 (선택적 확장)

주요 결론이 LSTM 아키텍처에 의존하지 않음을 확인하기 위해, H5 최적 정규화 조건에서 TinyTransformer를 추가 실험한다 (FD001만, 본문 Appendix 수준).

```
입력 임베딩: Linear(n_features → 32)
→ TransformerEncoder(d_model=32, nhead=4, num_layers=2, dim_ffn=128, dropout=0.1)
→ Global average pooling (window 차원) → Linear(32→1)
```

**실험 범위:** FD001 × {N1(fleet), N3(perunit_5), N7(RevIN)} × 5 seeds = **15회**  
**목적:** "정규화 효과는 백본 독립적" 한 줄 주장 근거 제공.

---

## H2 — RUL 클리핑 임계값 최적화 (Adaptive Clipping)

### 가설 요약

Piece-wise linear RUL 레이블의 클리핑 임계값을 데이터셋별 수명 분포에 맞게 최적화하면, 단일 임계값(125 cycles) 대비 RMSE·NASA Score가 향상된다.

**EDA 근거 (수명 통계):**

| Dataset | 평균 수명 | clip_125 비율 | 비고 |
|---------|---------|-------------|------|
| FD001 | 206.3 | 60.6% | FD001/FD002 그룹 |
| FD002 | 206.8 | 60.4% | — |
| FD003 | 247.2 | 50.6% | FD003/FD004 그룹 |
| FD004 | 246.0 | 50.8% | — |

두 그룹 간 클리핑 비율 차이 ~10%p → 동일한 125가 다른 비율의 RUL 구간을 평탄화함.

**Research Gap:**
- 단독: 낮음 (125는 학계 컨센서스)
- **H7 결합 시:** 높음 — 클리핑 × 손실 함수 상호작용 peer-reviewed **0편**
- 데이터셋별 최적 클리핑 정량화: **0편**

---

### 1. 실험 설계

**클리핑 임계값 5종:**

| 코드명 | 값 | 근거 |
|--------|-----|------|
| `clip_75` | 75 cycles | 탐색 하한 (FD001 평균의 ~36%) |
| `clip_100` | 100 cycles | 보수적 기준, H7과 공유 |
| `clip_125` | 125 cycles | 학계 표준 기준선 (baseline) |
| `clip_130` | 130 cycles | arXiv 2604.13459(2026) 사용 사례 |
| `clip_none` | 미적용 | 선형 RUL 레이블 |

**실험 매트릭스 (H2 단독):** 5 클리핑 × 4 데이터셋 = **20회** (선형 회귀는 결정론적 — seeds 불필요)

**통제 변수:** 손실 함수=MSE, 모델=선형 회귀(numpy), 정규화=Fleet min-max

---

### 2. Adaptive Clipping 구현

```python
def apply_rul_clipping(df, clip_value=None):
    df = df.copy()
    df['rul_linear'] = df['max_cycle'] - df['cycle']
    df['rul'] = df['rul_linear'].clip(upper=clip_value) if clip_value else df['rul_linear']
    return df

def compute_clipped_rul_stats(df, clip_values):
    results = []
    for clip in clip_values:
        clipped = df['rul_linear'].clip(upper=clip) if clip else df['rul_linear']
        results.append({
            'clip_value': clip,
            'pct_at_ceiling': (clipped == clip).mean() if clip else 0.0,
            'mean_rul': clipped.mean(),
            'std_rul': clipped.std(),
        })
    return pd.DataFrame(results)
```

> `pct_at_ceiling`: 천장에 달라붙은 데이터 비율. FD003+clip_125에서 ~50%로 높음.

---

### 3. H7과의 통합 실험 구조

H2와 H7은 클리핑 임계값 차원을 공유한다. H7 Phase 2b에 `clip_75`를 추가하여 H2+H7 통합 실험을 구성한다.

| 차원 | H7 단독 | H2+H7 통합 |
|------|---------|-----------|
| 클리핑 값 | 4종 (100/125/130/미적용) | **5종** (+clip_75) |
| 손실 함수 | 5종 | 5종 동일 |
| 총 조합 | 80회 | **100회** |

**핵심 질문:**
- 클리핑 값이 낮을수록 비대칭 손실 이점이 더 큰가?
- FD001/FD002 vs FD003/FD004 최적 클리핑이 다른가?
- arXiv 2604.13459의 clip_130은 어떤 조건에서 유효한가?

---

### 4. 평가 지표

- RMSE, NASA Score per engine (`NASA_Score / N_engines`)
- Wilcoxon rank-sum test — clip_125 대비 각 값, α=0.05
- 수명 구간별 RMSE (< 150 / 150~250 / > 250 cycles)

---

### 5. 실험 순서

| 파일 | 역할 |
|------|------|
| `Code/H2_clipping/01_data_loader.py` | 데이터 로드, 클리핑 적용, cycle_fraction |
| `Code/H2_clipping/02_clipping_analysis.py` | pct_at_ceiling, RUL 분포 통계 |
| `Code/H2_clipping/03_run_experiments.py` | 20회 자동 실행 (결정론적) |
| `Code/H2_clipping/04_evaluate.py` | RMSE/NASA Score 집계, Wilcoxon 검정 |
| `Code/H2_clipping/05_visualize.py` | 시각화 4종 |
| `Code/H2_clipping/06_h2h7_interaction.py` | H7 결과 병합 + 상호작용 히트맵 |

**예상 소요:** 선형 모델 기반, 전체 수 분 이내

---

### 6. 파일 구조

```
Data_Analysis/Code/H2_clipping/
├── 01_data_loader.py ~ 06_h2h7_interaction.py
├── results/
│   ├── clipping_rul_dist_stats.csv
│   ├── raw_predictions/        # FD00X_clipY_seedN.csv
│   ├── metrics_summary.csv
│   ├── evaluation_report.csv
│   └── subgroup_analysis.csv
└── figures/
    ├── fig_H2_01_rul_distribution.png
    ├── fig_H2_02_rmse_heatmap.png
    ├── fig_H2_03_nasa_heatmap.png
    ├── fig_H2_04_subgroup_rmse.png
    ├── fig_H2H7_01~04_interaction_FD00{1-4}.png
    └── fig_H2H7_05_optimal_combo.png
```

### 7. 예상 기여점

- 데이터셋별 최적 클리핑 임계값 최초 정량화 (FD001~FD004 교차 비교)
- 클리핑 × 손실 함수 상호작용 최초 분석 (H7 결합)
- arXiv 2604.13459의 130-cycle cap 조건부 유효성 검증
- 실용 지침: 새 데이터셋 적용 시 클리핑 초기값 경험적 규칙 제안

---

---

## H5 — Per-unit 정규화 vs Fleet 정규화

### 가설 요약

엔진 개별 초기 관측값 기준 정규화(per-unit normalization)는 fleet-level 정규화 대비 RUL 예측 성능을 향상시킨다.

**핵심 논리 — 초기 조건 불확실성 제거:**
Fleet 정규화는 훈련 전체 엔진의 평균 센서 범위를 기준으로 삼는다. 각 엔진은 새 상태에서도 초기 기준 센서값이 서로 달라 "초기 조건의 차이"와 "실제 열화 진행"이 혼재한다. Per-unit 정규화는 **엔진별 초기 관측값을 기준점으로 설정**하여 초기 조건 불확실성(initial condition uncertainty)을 전처리 단계에서 제거하고, 모델이 순수한 열화 신호(degradation signal)에 집중하도록 만든다.

이 논리는 CMAPSS가 합성 데이터임을 전제로도 타당하다: "제조 편차"라는 현실 해석 대신, "초기 기준값 편차의 전처리 제거"라는 일반적 신호 처리 목표(SNR 향상)로 프레이밍한다. 이 관점은 합성·실측 데이터 모두에서 방어 가능하며, 리뷰어의 "합성 데이터에서 제조 편차 근거 부재" 지적을 원천 차단한다.

**단, 한계:** 수명 < 50 cycles 엔진에서는 초기 5~10 cycles 통계가 대표성을 잃을 수 있다 → Section 2-B에서 명시적 분석.

**Research Gap:** Consensus MCP 200M+ 논문 스캔 — 명시적 비교 연구 **0편**

---

### 1. 실험 설계

**비교 정규화 방법 7종:**

| 실험 ID | 코드명 | 방법 | 기준 통계 | 참고 |
|---------|--------|------|---------|------|
| N1 | `fleet_minmax` | Fleet min-max [0,1] | train 전체 min·max | 문헌 표준 |
| N2 | `fleet_std` | Fleet standard scaling | train 전체 μ·σ | 문헌 표준 |
| N3 | `perunit_minmax_5` | Per-unit min-max | 엔진별 초기 **5 cycles** | **H5 핵심** |
| N4 | `perunit_minmax_10` | Per-unit min-max | 엔진별 초기 **10 cycles** | |
| N5 | `perunit_std_5` | Per-unit standard scaling | 엔진별 초기 5 cycles | |
| N6 | `perunit_std_10` | Per-unit standard scaling | 엔진별 초기 10 cycles | |
| N7 | `revin` | RevIN (학습 가능한 instance norm) | 추론 window별 μ·σ + 학습 affine | Kim et al. ICLR 2022 |

> **N7 RevIN 선정 근거:** 2022년 이후 시계열 예측 논문의 사실상 표준 정규화 기법. Fleet 정규화(N1/N2)와 per-unit 정규화(N3~N6)의 중간 지점으로, 학습 가능한 affine 파라미터(γ, β)로 정규화 효과를 데이터에서 최적화한다. CMAPSS 합성 데이터에서도 추론 시 window별 통계를 사용하므로 test-time distribution shift에 강인하다.

> ⚠️ **N7 구현 위치 — N1~N6과 다름:**  
> N1~N6은 `02_normalizers.py`의 numpy 클래스로 사전 전처리 후 데이터를 LSTM에 전달.  
> N7(RevIN)은 PyTorch `nn.Module`로 **`03_lstm_model.py`의 `LSTMWithRevIN` 클래스 내부**에 첫 번째 레이어로 삽입됨.  
> 실험 루프에서는 N7에 한해 `normalizer=None, model_class=LSTMWithRevIN` 경로를 사용하는 분기 1개로 처리.

**실험 매트릭스:** 7 정규화 × 4 데이터셋 × 5 seeds = **140회**  
(N1~N6: 전처리 후 `LSTMBase` / N7: `LSTMWithRevIN`, 모두 동일 학습 하이퍼파라미터)

**통제 변수:** RUL 클리핑=125, 슬라이딩 윈도우=30, 백본=LSTM(§0.4), 손실=MSE

> **⚙️ FD002·FD004 전처리 파이프라인 (H3 보조 요인 적용):**  
> FD002·FD004는 6가지 운전 조건이 센서 절대값을 왜곡하므로, H5 normalizer 비교 전에  
> **K-means 운전 조건 잔차화를 고정 선행 적용**한다. 이 pre-step은 모든 N1~N7에 공통 적용되어  
> "운전 조건 효과"와 "정규화 방법 효과"를 분리한다.
>
> ```
> FD001 · FD003 (단일 조건):  raw data → [H5 normalizer] → LSTM
> FD002 · FD004 (6가지 조건): raw data → [K-means 잔차화] → [H5 normalizer] → LSTM
> ```
>
> 논문 §3 파이프라인 절에 명시: "For FD002 and FD004, we first remove operational condition  
> effects via K-means clustering (k=6) to isolate the effect of the H5 normalization strategies."

---

### 2. Per-unit 정규화 구현 핵심

#### 2-A. FD002·FD004 운전 조건 잔차화 (pre-step, H3 보조 요인)

FD002·FD004에만 적용되는 고정 전처리. 학습 데이터의 K-means 중심점을 저장해 test 시에도 재사용한다.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def fit_op_condition_kmeans(train_df, k=6, op_cols=('op1', 'op2', 'op3')):
    """학습 데이터의 운전 조건 K-means 중심점 학습 및 반환."""
    scaler = StandardScaler()
    op_scaled = scaler.fit_transform(train_df[list(op_cols)])
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km.fit(op_scaled)
    return scaler, km

def apply_op_residual(df, scaler, km, feature_cols, op_cols=('op1', 'op2', 'op3')):
    """운전 조건별 센서 평균을 빼서 잔차로 변환."""
    df = df.copy()
    op_scaled = scaler.transform(df[list(op_cols)])
    df['_op_cluster'] = km.predict(op_scaled)

    # 클러스터별 센서 평균 (train에서 계산된 값을 이용해야 leakage 없음)
    # → fit 시 cluster_means를 함께 반환하고 여기서 그 값을 사용
    for col in feature_cols:
        cluster_means = df.groupby('_op_cluster')[col].transform('mean')
        df[col] = df[col] - cluster_means
    return df.drop(columns=['_op_cluster'])
```

> **Test 누수 방지:** `fit_op_condition_kmeans()`는 train에서만 호출. test 적용 시 학습된 `scaler`·`km`·`cluster_means`를 재사용.  
> **H6와 공유:** `Code/H6_fault_mode/phase1_clustering/`의 FD004 처리와 동일 로직 → 공통 유틸로 분리 권장 (`Data_Analysis/Code/shared/op_condition_utils.py`).

#### 2-C. 기본 구현

```python
def compute_perunit_stats(train_df, sensor_cols, n_init=5, method='minmax'):
    stats = {}
    for unit in train_df['unit'].unique():
        unit_data = (train_df[train_df['unit'] == unit]
                     .sort_values('cycle').iloc[:n_init][sensor_cols])
        if method == 'minmax':
            stats[unit] = {'min': unit_data.min(), 'max': unit_data.max()}
        elif method == 'std':
            stats[unit] = {'mean': unit_data.mean(),
                           'std': unit_data.std().clip(lower=1e-8)}
    return stats
```

**Test 셋 주의사항:**
- Per-unit: test 엔진의 초기 n cycles로 독립 통계 계산 (train 통계 재사용 불가)
- 범위 외 값(열화 진행 시)은 클리핑하지 않음 — 열화 방향 신호 보존

**수명 구간 분류:**
```python
# boundary : < 150 cycles  ← 경계 조건 분석 대상 (EDA 최소 수명 128, critics.md 권장 기준)
# medium   : 150~250 cycles
# long     : > 250 cycles
```

#### 2-D. RevIN 구현 (N7)

RevIN은 PyTorch 모듈로 LSTM 입력 직전에 삽입된다. 출력(RUL 스칼라)은 센서 공간이 아니므로 역정규화를 적용하지 않는다.

```python
import torch, torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization — Kim et al., ICLR 2022.
    Applied to input window (B, W, F); output (RUL scalar) is NOT denormalized
    because RUL is not in sensor space.
    """
    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))   # learnable scale
            self.beta  = nn.Parameter(torch.zeros(n_features))  # learnable shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        mean = x.mean(dim=1, keepdim=True).detach()
        std  = x.std(dim=1, keepdim=True, unbiased=False).detach() + self.eps
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x
```

**사용 방법:** LSTM 모델 `__init__`에서 `self.revin = RevIN(n_features)`, `forward`에서 `x = self.revin(x)` 후 LSTM에 전달.

**N7 vs N3/N5 핵심 차이:**
- N3/N5: 엔진별 초기 K cycles의 **고정 통계** (사전 계산, 그래디언트 없음)
- N7: 현재 추론 **window의 통계** (cycle-wise 적응) + **학습 가능 파라미터** (γ, β)

#### 2-E. 경계 조건 분석 — 수명 < 150 cycles 엔진

수명이 짧은 엔진(< 150 cycles)에서는 초기 5~10 cycles로 추정한 통계의 대표성이 저하될 수 있다.  
**EDA 근거:** 전 데이터셋 최소 수명 128 cycles — < 50 그룹은 실제로 항상 비어 있으므로 기준을 150으로 설정.  
**critics.md:** "수명 < 150 cycles 서브그룹 분석이 핵심 차별화 포인트" (H5 기여점).

**사전 분포 파악:**
```python
lifetimes = train_fd001.groupby('unit')['cycle'].max()
bins = [0, 150, 250, 9999]
labels = ['boundary(<150)', 'medium(150-250)', 'long(>250)']
print(pd.cut(lifetimes, bins=bins, labels=labels).value_counts().sort_index())
# FD001 예상: boundary ~10-20개 (128~150 구간), medium ~60개, long ~20개
```

| 시나리오 | 해석 | 논문 서술 |
|---------|------|---------|
| boundary 그룹도 per-unit ≥ fleet | 경계 조건 없음 | "수명 제약 없이 per-unit 적용 가능" |
| boundary 그룹에서 fleet > per-unit | 경계 조건 발견 | "수명 < 150 cycles에서는 fleet 권장; 실용 가이드라인 제시" |
| n_init=5 불안정, n_init=10 개선 | 초기 샘플 수 민감도 | "극단 단명 엔진에는 n_init 확대 권장" |

---

### 3. 베이스라인 LSTM 모델

```
입력: (batch, window=30, n_features)
→ LSTM(hidden=64) → Dropout(0.2)
→ LSTM(hidden=32) → Dropout(0.2)
→ Linear(32→16) → ReLU → Linear(16→1)
```

| 파라미터 | 값 |
|---------|-----|
| optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| batch_size | 256 |
| max_epochs | 100 |
| early_stopping | patience=15 |
| val_split | 각 엔진 후반 20% cycle |
| random_seeds | [0, 1, 2, 3, 4] |

---

### 4. 평가 지표

- **RMSE**, **NASA Score**
- **MAE_boundary / MAE_short / MAE_medium / MAE_long**
- **경계 조건 승패 집계** — fleet vs per-unit 엔진별 MAE 비교
- **Wilcoxon rank-sum test** — N1 대비 N3~N6, α=0.05

---

### 5. 실험 순서

| 파일 | 역할 |
|------|------|
| `Code/shared/op_condition_utils.py` | K-means 운전 조건 잔차화 공통 유틸 (H5·H6 공유) |
| `Code/H5_normalization/01_data_loader.py` | 데이터 로드, 수명 그룹(boundary<150 포함), 윈도우, FD002·FD004 잔차화 호출 |
| `Code/H5_normalization/02_normalizers.py` | 정규화 클래스 N1~N6 (numpy, 전처리용) |
| `Code/H5_normalization/03_lstm_model.py` | `LSTMBase` (N1~N6용) + `LSTMWithRevIN` (N7용), 학습 루프, NASA Score |
| `Code/H5_normalization/04_run_experiments.py` | 140회 자동 실행 (N1~N6: `LSTMBase`, N7: `LSTMWithRevIN` 분기) |
| `Code/H5_normalization/05_evaluate.py` | 통계 검정, 경계 조건 리포트 |
| `Code/H5_normalization/06_visualize.py` | 시각화 6종 |

**예상 소요:** CPU 기준 6~10시간. FD001 단독 파이프라인 검증 후 전체 실행 권장.

---

### 6. 파일 구조

```
Data_Analysis/Code/H5_normalization/
├── src/                       # 01~06.py
├── results/
│   ├── raw_predictions/       # FD00X_norm_id_seedN.csv (120개)
│   ├── metrics_summary.csv
│   ├── evaluation_report.csv
│   └── boundary_condition_report.csv
└── figures/
    ├── fig_H5_01_rmse_heatmap.png
    ├── fig_H5_02_score_heatmap.png
    ├── fig_H5_03_subgroup_mae.png
    ├── fig_H5_04_normalization_effect.png
    ├── fig_H5_05_statistical_test.png
    ├── fig_H5_06_short_life_error.png
    └── fig_H5_07_boundary_condition.png
```

### 7. 예상 기여점

- Per-unit vs fleet normalization **최초 체계적 ablation** (CMAPSS peer-reviewed 기준)
- **초기 조건 불확실성 제거** 관점의 이론적 프레이밍 — 합성·실측 데이터 공통 방어
- n_init=5 vs 10 sensitivity analysis
- **경계 조건 분석** (수명 < 50 cycles) — 실용 가이드라인 제시

---

---

## H6 — 복합 고장 모드 멀티-브랜치 모델 (FD003, FD004)

### 가설 요약

FD003/FD004의 HPC+Fan 이중 고장 모드를 레이블 없이(unsupervised) GMM으로 분리하고, 고장 모드별 전용 LSTM 브랜치를 구성하면 단일 모델 대비 RMSE·NASA Score가 향상된다.

**Research Gap:**
- Bayesian nonparametric (arXiv 2602.19263, 2025): unsupervised 근접, arXiv 단계
- Multi-task MoE (Expert Syst. Appl. 2025): **fault mode 레이블 사용** — supervised
- **딥러닝 기반 unsupervised multi-branch: peer-reviewed 미발표**

---

### Phase 0 — EDA 완료 결과 (h6_eda_fd003.py, 2026-07-02) ✅

#### 클러스터 적합성

| 지표 | 결과 | 해석 |
|------|------|------|
| K-means Silhouette (k=2) | **0.634** | > 0.5 → 강한 군집 구조 |
| GMM BIC 최적 k | **2** | 통계적으로 2-군집 최적 |
| GMM 배정 신뢰도 (p>0.90) | **100/100** | 모든 엔진 명확 배정 |

#### 클러스터 구성

| 클러스터 | 엔진 수 | 평균 수명 | σ |
|---------|-------|---------|---|
| Cluster 0 | 44 | **304.7** cycles | 94.3 |
| Cluster 1 | 56 | **202.1** cycles | 42.3 |

> **수명 교란 주의:** 50% 수명 차이 존재. Phase 1 slope ablation으로 교란 여부 검증.

#### 센서 판별력 순위 (|Δz|)

| 순위 | 센서 | 물리적 의미 | 고장 유형 | \|Δz\| |
|------|------|-----------|---------|-------|
| 1 | **s15** | BPR (bypass ratio) | Fan | **31.3** |
| 2 | **s20** | W31 (HPT bleed) | — | 16.0 |
| 3 | **s21** | W32 (LPT bleed) | — | 15.2 |
| 4 | **s7** | P30 (HPC pressure) | HPC | 13.0 |
| 5 | **s12** | phi (fuel/Ps30) | HPC | 12.8 |
| 6 | s2 | T24 (LPC temp) | Fan | 5.5 |
| 7 | s4 | T50 (LPT temp) | Fan | 4.7 |

FAN 관련 평균 |Δz|=9.43 > HPC 관련 |Δz|=5.34

---

### 1. 실험 설계 — 비교 모델 4종

```
[M0 Baseline LSTM] vs [M1 Hard-Routing] vs [M2 Soft-Gating] vs [M3 Attention-Gate]
```

| 모델 | 고장 모드 처리 | 핵심 특징 | 패러다임 |
|------|-------------|---------|--------|
| M0 Baseline | 없음 (단일 모델) | 문헌 기준선 | — |
| M1 Hard-Routing | GMM 레이블 → 2-LSTM 분기 | 분기 독립 학습 | Pre-trained clustering |
| M2 Soft-Gating | GMM 사후확률 → 가중 합산 | 경계 엔진 유연 처리 | Pre-trained clustering |
| M3 Attention-Gate | 학습 가능 MLP 게이팅 → 가중 합산 | GMM 전처리 불필요, end-to-end | MoE (2024-2025) |

> **M3 선정 근거:** Expert Syst. Appl. 2025의 Multi-task MoE는 supervised 레이블을 사용한다. M3는 같은 MoE 철학을 **unsupervised + end-to-end** 방식으로 구현하여, GMM 사전 단계(M1·M2)의 필요성을 학습 가능한 게이팅으로 대체할 수 있는지 검증한다.

---

### 2. 고장 모드 클러스터링 (Phase 1)

#### 2-1. 판별 센서 (EDA 결과 업데이트)

```python
# EDA |Δz| 상위 7개 (|Δz| > 4.0 기준)
FAULT_DISCRIMINANT_SENSORS = ['s15', 's20', 's21', 's7', 's12', 's2', 's4']
# 변경: s15 신규 추가 (EDA 1위, |Δz|=31.3); s3·s11 제거 (상위 미진입)

# 피처 A: 후반부 평균 (마지막 20%)
# 피처 B: 열화 기울기 (선형 회귀 β₁) — 수명 교란 완화
# 결합: 7×2 = 14차원 벡터 (엔진당)
```

#### 2-2. 수명 교란 대응 — Slope-only Ablation (신규)

| Ablation | 피처 | 의도 |
|---------|------|------|
| **AB-full** | late_mean + slope (14차원) | 기준 클러스터링 |
| **AB-slope** | slope만 (7차원) | 수명 교란 최소화 |
| **AB-late** | late_mean만 (7차원) | 대조군 |

**해석:**
- AB-slope Silhouette ≈ AB-full → 열화 패턴 차이 반영 → H6 가설 강화
- AB-slope Silhouette ≪ AB-full → 수명 교란 가능성 → 논문 limitation 명시

#### 2-3. FD004 운전 조건 보정

```python
# 6가지 운전 조건이 GMM 군집 오염 방지
op_cluster_means = df.groupby('op_condition')[FAULT_DISCRIMINANT_SENSORS].mean()
residual = sensor_value - op_cluster_means.loc[op_condition, sensor]
residual_slope = linreg_slope(unit_residuals[sensor])
```

#### 2-4. Test 엔진 클러스터 배정 (Inference-time Routing)

GMM 클러스터링은 train 데이터로 학습되므로, test 엔진을 어느 브랜치에 배정할지 명시적 절차가 필요하다.

**저장 아티팩트 (Phase 1 종료 시):**
```
models/
├── gmm_fd003_{full,slope}.pkl      ← 학습 GMM 모델
├── gmm_fd004_{full,slope}.pkl
├── op_kmeans_fd004.pkl             ← FD004 운전 조건 K-means (shared/op_condition_utils.py와 공유)
└── op_cluster_means_fd004.csv      ← 클러스터별 센서 평균 (잔차 계산용)
```

**Test 배정 절차 (M1·M2 공통):**

```python
def assign_test_engine(test_unit_df, gmm, scaler_feat, variant='full',
                       op_kmeans=None, op_means=None):
    """
    test_unit_df: 테스트 엔진의 전체 관측 이력 (CMAPSS test 파일, 마지막 관측까지)
    variant     : 'full' | 'slope'  (AB_full / AB_slope)
    """
    sensors = [s for s in FAULT_DISCRIMINANT_SENSORS if s in test_unit_df.columns]

    # FD004: 운전 조건 잔차화 (저장된 K-means 중심점 재사용)
    if op_kmeans is not None:
        test_unit_df = apply_op_residual(test_unit_df, op_kmeans, op_means, sensors)

    # 피처 추출 (학습과 동일 방식, 단 "last 20%"는 관측 이력 기준)
    features = {}
    for s in sensors:
        vals = test_unit_df[s].values
        if variant in ('full', 'late'):
            features[f'{s}_late_mean'] = compute_late_mean(vals, pct=0.20)
        if variant in ('full', 'slope'):
            features[f'{s}_slope'] = compute_slope(vals)

    feat_vec = scaler_feat.transform([list(features.values())])
    soft_prob = gmm.predict_proba(feat_vec)[0]   # [p_cluster0, p_cluster1]

    return soft_prob   # M1: argmax(soft_prob) / M2: soft_prob 그대로 사용
```

**M3 (Attention-Gate):** GatingNet이 초기 K=10 cycles를 직접 처리해 routing weights를 출력하므로, 별도 저장 아티팩트 및 배정 절차 불필요. End-to-end 학습 중 자동 처리.

**주의사항 (논문 limitation 절 명시):**
- Test "late 20%"는 EOL 기준이 아닌 관측 종료 기준 → train과 미세한 불일치 발생 가능
- CMAPSS test 파일은 EOL 직전까지 관측을 포함하므로 실용적 오차는 작음
- M3는 이 불일치에서 자유롭다는 점이 추가 기여점이 될 수 있음

---

### 3. 멀티-브랜치 아키텍처

```
입력: (batch, window=30, n_features=14)
→ LSTM(64) → Dropout(0.2)
→ LSTM(64) → Dropout(0.2)
→ FC(64→32→1)
```

**M2 Soft-Gating 손실:**
```python
total_loss = (MSE(y_final, y_true)
            + 0.1 * MSE(y_branch_0, y_true)
            + 0.1 * MSE(y_branch_1, y_true))
```

---

**M3 End-to-End Attention-Gate 아키텍처:**

GMM 사전 클러스터링 없이 게이팅 네트워크를 LSTM 브랜치와 함께 end-to-end 학습한다.

```
GatingNet (초기 K=10 cycles 읽음):
  입력: (batch, K, n_features) → Flatten → Linear(K×F → 32) → ReLU → Linear(32→2) → Softmax
  출력: w = [w₀, w₁]  (확률적 라우팅 가중치)

Branch 0: LSTM(64)→Dropout(0.2)→LSTM(32)→Dropout(0.2)→FC(32→16)→ReLU→FC(16→1)  →  y₀
Branch 1: (동일 구조, 독립 파라미터)  →  y₁

y_final = w₀ × y₀ + w₁ × y₁
```

```python
total_loss_M3 = (MSE(y_final, y_true)
               + 0.05 * MSE(y₀, y_true)
               + 0.05 * MSE(y₁, y_true))
```

**M3 설계 원칙:**
- GatingNet은 엔진의 **초기 센서 패턴**만 보고 고장 모드 확률을 추정한다 (추론 가능한 초기 정보 활용).
- Auxiliary loss(0.05 ×)는 두 브랜치가 모두 의미 있는 표현을 학습하도록 강제한다.
- M3가 M1·M2보다 우수하면 "pre-trained GMM 불필요, end-to-end MoE 충분" 결론; 열등하면 "사전 도메인 지식(GMM 군집 구조) 주입의 효용" 결론 → 양방향 모두 논문 기여.

---

### 4. 평가 지표

- **RMSE, NASA Score** — FD003/FD004 각각
- **Silhouette Score** — AB-full / AB-slope / AB-late 세 variant
- **사후 물리 해석** — s15/s7 프로파일 클러스터별 비교

| 모델 | FD003 RMSE | FD004 RMSE | 비고 |
|------|-----------|-----------|------|
| CAELSTM 2025 (문헌) | 13.40 | — | 비교 기준 |
| M0 Baseline | — | — | 단일 모델 |
| M1 Hard-Routing (AB-full) | — | — | GMM hard label |
| M2 Soft-Gating | — | — | GMM soft prob |
| M3 Attention-Gate | — | — | End-to-end MoE |

---

### 5. 실험 순서 (4단계)

**Phase 0 ✅ 완료:** `h6_eda_fd003.py` — FD003 군집 분석, 센서 판별력

**Phase 1 — 클러스터링 (scikit-learn):**

| 파일 | 역할 |
|------|------|
| `Code/H6_fault_mode/phase1_clustering/h6_p1_feature_extraction.py` | 열화 서명 추출 (s15 추가) |
| `Code/H6_fault_mode/phase1_clustering/h6_p1_gmm_fitting.py` | GMM 적합, AB-slope ablation |
| `Code/H6_fault_mode/phase1_clustering/h6_p1_cluster_saver.py` | GMM·K-means·cluster_means 직렬화 (`models/*.pkl`) |
| `Code/H6_fault_mode/phase1_clustering/h6_p1_cluster_visualization.py` | PCA 투영, 센서 프로필 |
| `Code/H6_fault_mode/phase1_clustering/h6_p1_cluster_analysis.py` | Silhouette 비교 표 |

**Phase 1 완료 기준:** FD003/FD004 BIC k=2, AB-slope Sil > AB-full × 0.7

**Phase 2 — 브랜치 모델 (PyTorch):**

| 파일 | 역할 |
|------|------|
| `Code/H6_fault_mode/phase2_models/h6_p2_model_utils.py` | 공유 유틸 |
| `Code/H6_fault_mode/phase2_models/h6_p2_baseline_lstm.py` | M0 |
| `Code/H6_fault_mode/phase2_models/h6_p2_hard_routing.py` | M1 (AB-full + AB-slope) |
| `Code/H6_fault_mode/phase2_models/h6_p2_soft_gating.py` | M2 |
| `Code/H6_fault_mode/phase2_models/h6_p2_attention_gate.py` | M3 (GatingNet + 2-branch, end-to-end) |
| `Code/H6_fault_mode/phase2_models/h6_p2_inference.py` | Test 배정 파이프라인 — M1·M2: `assign_test_engine()`, M3: 직접 forward |

**Phase 3 — 비교 평가:**

| 파일 | 역할 |
|------|------|
| `Code/H6_fault_mode/phase3_evaluation/h6_p3_evaluate_all.py` | RMSE/NASA Score |
| `Code/H6_fault_mode/phase3_evaluation/h6_p3_ablation.py` | AB-slope vs AB-full 비교 |
| `Code/H6_fault_mode/phase3_evaluation/h6_p3_figures.py` | 비교 플롯 |

---

### 6. 파일 구조

```
Data_Analysis/Code/H6_fault_mode/
├── phase0_eda/               ✅ 완료
│   ├── h6_eda_fd003.py
│   └── figures/              (h6_fig01~04.png)
├── phase1_clustering/
├── phase2_models/
├── phase3_evaluation/
├── figures/
│   └── h6_fig07_ablation_slope_vs_full.png  (신규)
├── results/
│   ├── unit_features_fd003_{full,slope,late}.csv
│   ├── cluster_assignments_fd003/fd004.csv
│   ├── ablation_silhouette_comparison.csv
│   └── model_comparison_fd003/fd004.csv
└── models/
    ├── gmm_fd003_{full,slope}.pkl
    └── *.pt 모델 파일들
```

### 7. 위험 요소 (EDA 반영 업데이트)

| 위험 | 상태 | 대응 |
|------|------|------|
| GMM n=3 선택 | ⬇️ 낮음 완화 (FD003 k=2 확인) | FD004 추가 확인 필요 |
| 클러스터가 수명 길이 반영 | ⬇️ 부분 완화 | AB-slope ablation으로 검증 |
| s15 누락 | ✅ 해결 | FAULT_DISCRIMINANT_SENSORS 업데이트 |
| FD004 운전 조건 오염 | 높음 유지 | 잔차 피처 사용 필수 |

### 8. 예상 기여점

- **Unsupervised multi-branch** CMAPSS RUL 최초 적용
- EDA 선행 확인: Silhouette 0.634, GMM 100% 신뢰 배정
- **수명 교란 통제 ablation** (slope vs full) — 리뷰어 우려 선제 대응
- Hard vs Soft gating 비교 — 소규모 데이터셋 전략 실증

---

---

## H7 — 비대칭/가중 손실 함수 최적화

### 가설 요약

NASA Score의 비대칭 패널티를 학습 손실 함수에 반영하면 NASA Score가 개선된다. 클리핑 임계값과 손실 함수는 상호작용 효과가 있으며, FD001~FD004별 최적 조합이 다를 수 있다.

**Research Gap (Consensus MCP):**
- 클리핑 × 손실 함수 상호작용: **0편**
- FD001~FD004 cross-dataset 비교: **0편**

**H2와의 관계:** 클리핑 × 손실 상호작용 실험 블록은 H7 Phase 2b에 통합. H2+H7 공동 기여.

---

### 0. life_ratio — 훈련/추론 역할 분리

| 단계 | life_ratio 사용 | 비고 |
|------|---------------|------|
| **훈련 (loss weighting)** | ✅ 사용 | `1 - RUL_true / RUL_max_clipped` |
| **검증 (val loss)** | ✅ 사용 | 동일 공식 |
| **추론 (test 예측)** | ❌ 불필요 | 손실 가중치 없이 순전파만 |

> 추론 시 max_cycle 미지 문제는 **발생하지 않는다.** 논문 §3에 명시하여 리뷰어 지적 선제 차단.

추론 시 cycle 비율이 필요한 경우(후처리 한정):
```python
cycle_fraction = min(current_cycle, clip_value) / clip_value
# 주의: 실제 수명 > clip_value 엔진에서 underestimate 발생 → limitation 절 명시
```

---

### 1. 손실 함수 정의 (5종)

**L1. MSE (Baseline)**
```
L_MSE = (1/N) · Σ (ŷ_i - y_i)²
```

**L2. NASA Score Loss**
```
d_i = ŷ_i - y_i
s(d) = exp(-d/13)-1 (d<0) / exp(d/10)-1 (d≥0)
```

**L3. Dynamically Weighted MSE**
```
life_ratio_i = 1 - RUL_i / RUL_max_clipped   ← 훈련 전용
w_dyn(i) = 1 + λ · life_ratio_i
L_DynW = (1/N) · Σ w_dyn · (ŷ - y)²
```

**L4. Focal-RUL Loss**
```
w_focal(i) = (|ŷ-y| / (|ŷ-y| + 1))^γ
```

**L5. Time-Weighted + Asymmetric (신규 제안)**
```
life_ratio_i = 1 - RUL_i / RUL_max_clipped   ← 훈련 전용
w_time(i) = 1 + λ_t · life_ratio_i
w_asym(i) = 1 (d<0) / λ_a (d≥0)
L_TWA = (1/N) · Σ w_time · w_asym · (ŷ - y)²
```
> 기존 연구(DynamicWeighted 2020, Asymmetric 2025)가 두 효과를 별도로 다룬 것과 달리 **곱 구조로 결합**

**L6. Pinball (Quantile) Loss (2022-2025 표준 비대칭 회귀)**
```
d_i = ŷ_i - y_i
L_pin(y, ŷ, τ) = (1/N) · Σ  { τ·(-d_i)    if d_i < 0  (early prediction)
                              { (1-τ)·d_i   if d_i ≥ 0  (late prediction)
```

**τ 탐색 범위:** [0.25, 0.35, 0.40, 0.45, 0.50]

> **물리적 해석:** τ < 0.5 → 지연 예측(d≥0)에 (1-τ)의 더 큰 패널티 → 보수적 예측 편향 → NASA Score 최적화 방향과 일치. τ = 0.35를 초기 기본값으로 설정하며 Phase 3c에서 최적 τ 탐색.
>
> **최신 근거:** Quantile regression 기반 RUL 예측은 2022년 이후 불확실성 정량화 맥락에서 활발히 연구됨. L6는 L2(NASA Score Loss)와 달리 미분 가능하여 역전파가 안정적이다.

**L7. Huber-Asymmetric Composite (이상치 강건 비대칭 손실)**
```
Huber(d, δ) = { 0.5·d²           if |d| ≤ δ
              { δ·(|d| - 0.5·δ)  if |d| > δ

w_asym(d) = { 1    if d < 0  (early prediction)
            { λ_a  if d ≥ 0  (late prediction)

L_HubA = (1/N) · Σ Huber(d_i, δ) × w_asym(d_i)
```

**하이퍼파라미터:** δ ∈ [10, 20, 30], λ_a ∈ [1.5, 2.0, 3.0] → Phase 3d Grid Search (9 조합, FD001 스크리닝)

> **설계 의도:** Huber의 이상치 강건성(수명이 극단적으로 긴/짧은 엔진에서 MSE 폭발 방지) + 비대칭 가중치(NASA Score 방향). L5(TWA)와 차별점: L5는 수명 단계별 동적 가중치, L7은 잔차 크기에 따른 강건 가중치.
> **2024-2025 동향:** 이상치에 강건한 손실 함수(Huber, pseudo-Huber)와 비대칭 패널티의 조합은 최근 PHM(Prognostics and Health Management) 분야에서 주목받고 있으나, CMAPSS 교차 검증 연구는 미발표.

---

### 2. 실험 설계 — H2×H7 교차 실험 매트릭스 (7종 손실 함수)

| | L1 MSE | L2 NASA | L3 DynW | L4 Focal | L5 TWA | L6 Pinball | L7 HubA |
|---|--------|---------|---------|---------|--------|------------|---------|
| **clip_100** | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| **clip_125** | ○(기준) | ○ | ○ | ○ | ○ | ○ | ○ |
| **clip_130** | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| **clip_none** | ○ | ○ | ○ | ○ | ○ | ○ | ○ |

> **L6 τ 설정:** Phase 2b에서는 τ = 0.35 (기본값) 사용. Phase 3c에서 5개 τ 탐색.  
> **L7 δ, λ_a 설정:** Phase 2b에서는 δ = 20, λ_a = 2.0 (기본값) 사용. Phase 3d에서 9조합 탐색.

**Phase 2b 총 실험:** 4 clips × 7 losses × 4 datasets × 5 seeds = **560회**

---

### 3. 백본 모델 전략 (2단계)

**Step 1 — 선형 모델 스크리닝 (numpy, 80회):** 빠른 조합 탐색, Top-5 후보 추출

**Step 2a — LSTM 파일럿 (FD001 × Top-5 × 5 seeds = 25회):**
- 판단 기준: 선형 랭킹 vs LSTM 랭킹 Spearman ρ > 0.7 → Phase 2b 진행

**Step 2b — 전체 LSTM 교차 실험 (400회):**

```
입력: (batch, window=30, n_features)
→ LSTM(hidden=32) → Dropout(0.2)
→ Linear(32→16) → ReLU → Linear(16→1)
```

---

### 4. λ 파라미터 탐색

**Phase 3a — Grid Search (L5 TWA):**
```
λ_t : [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
λ_a : [1.0, 1.5, 2.0, 3.0, 5.0]
총 30조합 → λ_t × λ_a 히트맵
```

**Phase 3b — λ 독립성 분석 (L5 TWA):**
```python
# λ_t × λ_a = 2 유지, 분배 비율만 변경
independence_pairs = [(2.0,1.0),(1.0,2.0),(1.41,1.41),(4.0,0.5),(0.5,4.0)]
```

| 결과 | 해석 | 논문 처리 |
|------|------|---------|
| NASA Score 차이 < 1% | 단일 λ로 축약 가능 | 파라미터 합리성 분석 1문단 |
| NASA Score 차이 ≥ 1% | 독립 기여 확인 | 히트맵으로 최적 분배 시각화 |

**Phase 3c — L6 Pinball τ 탐색:**
```
τ 탐색: [0.25, 0.35, 0.40, 0.45, 0.50]
고정: clip_best(Phase 2b 결과), 4 datasets, 5 seeds
총 실험: 5 × 4 × 5 = 100회
```

| τ 결과 | 해석 |
|-------|------|
| τ_best < 0.5 → NASA Score 개선 | "보수적 편향이 NASA Score 최적화에 유리" 주장 |
| τ_best ≈ 0.5 → RMSE 개선 | Pinball(0.5) = MAE로 귀결, 이상치 강건성 효과 |

**Phase 3d — L7 HubA δ × λ_a Grid (FD001 스크리닝):**
```
δ   : [10, 20, 30]
λ_a : [1.5, 2.0, 3.0]
고정: clip_125, FD001만, 5 seeds
총 실험: 3 × 3 × 5 = 45회 (스크리닝) → 최적 조합으로 4 datasets 확장 (20회)
```

---

### 5. 평가 지표

- **NASA Score per engine** = `NASA_Score / N_engines` (메인 비교 지표)
- RMSE, Pareto Frontier (RMSE vs NASA Score)

---

### 6. 실험 순서

| 파일 | 역할 |
|------|------|
| `Code/H7_loss_function/00_config.py` | 공통 상수 |
| `Code/H7_loss_function/01_data_loader.py` | 데이터 로드, RUL 클리핑, life_ratio |
| `Code/H7_loss_function/02_loss_functions.py` | L1~L5 (life_ratio 훈련 전용 명시) |
| `Code/H7_loss_function/03_linear_model.py` | numpy 선형 회귀 (스크리닝) |
| `Code/H7_loss_function/03b_lstm_model.py` | PyTorch LSTM(hidden=32) |
| `Code/H7_loss_function/00_pipeline_check.py` | Phase 0: end-to-end 검증 |
| `Code/H7_loss_function/04_phase1_linear_screening.py` | Phase 1: 80회 |
| `Code/H7_loss_function/05_phase2a_lstm_pilot.py` | Phase 2a: 25회 |
| `Code/H7_loss_function/05_phase2b_lstm_full.py` | Phase 2b: 400회 |
| `Code/H7_loss_function/06_phase3a_lambda_grid.py` | Phase 3a: L5 λ_t·λ_a 30조합 |
| `Code/H7_loss_function/06_phase3b_lambda_independence.py` | Phase 3b: L5 λ 독립성 5쌍 |
| `Code/H7_loss_function/06_phase3c_pinball_tau.py` | Phase 3c: L6 τ 탐색 100회 |
| `Code/H7_loss_function/06_phase3d_huba_grid.py` | Phase 3d: L7 δ×λ_a 45+20회 |
| `Code/H7_loss_function/07_evaluation.py` | RMSE, NASA Score |
| `Code/H7_loss_function/08_visualization.py` | Figure 일괄 생성 |

**예상 소요:** Phase 2b 560회 × ~5분 ≈ 47시간 (야간 실행 권장, FD001 파이럿 먼저 확인)

---

### 7. 파일 구조

```
Data_Analysis/Code/H7_loss_function/
├── *.py (파일 13개)
├── results/
│   ├── phase1_linear_screening.csv
│   ├── phase2a_lstm_pilot.csv
│   ├── phase2b_results_matrix.csv
│   ├── phase3a_lambda_grid_FD001.csv
│   ├── phase3a_cross_dataset_transfer.csv
│   ├── phase3a_life_stage_breakdown.csv
│   └── phase3b_lambda_independence.csv
└── figures/
    ├── fig_phase2_nasa_heatmap_FD00{1-4}.png
    ├── fig_phase2_clip_interaction.png       ← H2+H7 공동 기여 핵심
    ├── fig_pareto_frontier_FD001.png
    ├── fig_phase3a_lambda_heatmap_nasa.png
    ├── fig_phase3a_lambda_tradeoff_curve.png
    ├── fig_phase3b_lambda_independence.png
    ├── fig_phase3c_pinball_tau_sweep.png      ← L6 τ 탐색
    └── fig_phase3d_huba_delta_lambda_grid.png ← L7 δ×λ_a
```

### 8. 예상 기여점

- **L5(TWA) 신규 제안** — time-weighted × asymmetric 곱 구조 최초 결합
- **L6(Pinball) τ 최적화** — τ < 0.5의 보수적 편향이 NASA Score를 체계적으로 개선하는지 최초 정량화
- **L7(HubA) 신규 제안** — 이상치 강건성(Huber) + 비대칭 패널티의 곱 구조 (PHM 분야 최초 결합)
- **클리핑 × 손실 함수(7종) 상호작용 최초 정량화** (H2+H7 공동 기여)
- **FD001~FD004 cross-dataset** 비교 (기존 연구 대부분 FD001만)
- **λ 독립성 분석** — L5 TWA 구조 타당성 실험적 검증

**한계:** life_ratio는 훈련 손실 전용 — 추론 시 max_cycle 미지 문제 없음 (논문 명시 필요)

---

---

## 전체 실행 로드맵

```
[STAGE 0] 환경 준비 (즉시, 수 분)
├── pip install torch --index-url .../cpu
├── pip install tqdm scikit-learn scipy
└── pip freeze > requirements.txt

[STAGE 1] 선형 모델 스크리닝 (numpy/sklearn, torch 없이 가능)
├── H2: 5 clips × 4 datasets = 20회 (수 분)
├── H7 Phase 1: 7 losses × 4 datasets = 80회 (수 분)
└── → Top-5 손실 함수 후보 선별

[STAGE 2] H5 파이프라인 구축 + H6 Phase 1 (병행)
├── H5: FD001 × N1 단독 파이프라인 검증 (30분)
├── H5: 전체 7 normalizers × 4 datasets × 5 seeds = 140회 (8~12시간)
│         [RevIN N7 torch 모듈 포함]
└── H6 Phase 1: 피처 추출 → GMM → AB-slope ablation (scikit-learn, 30분)

[STAGE 3] LSTM 파일럿 + 검증 (H7, H6 병행)
├── H7 Phase 2a: FD001 × Top-5 losses × 5 seeds = 25회
│   └── Spearman ρ(선형 vs LSTM 랭킹) > 0.7 확인 → Phase 2b 진행
└── H6 Phase 2: M0·M1·M2·M3 구현 + FD003 소규모 테스트

[STAGE 4] 전체 LSTM 교차 실험 (야간 실행)
├── H7 Phase 2b: 4 clips × 7 losses × 4 datasets × 5 seeds = 560회 (~47시간)
└── H6 Phase 2 전체: M0~M3 × FD003/FD004 × 5 seeds = 40회

[STAGE 5] 하이퍼파라미터 탐색 (병행)
├── H7 Phase 3a: L5 λ_t·λ_a grid (30조합)
├── H7 Phase 3b: L5 λ 독립성 분석 (5쌍)
├── H7 Phase 3c: L6 τ 탐색 (100회)
├── H7 Phase 3d: L7 δ×λ_a grid (45+20회)
└── H6 Phase 3: 전체 평가 + ablation 비교

[STAGE 6] 백본 민감도 (선택적)
└── TinyTransformer: FD001 × {N1, N3, N7} × 5 seeds = 15회 (§0.7)

[STAGE 7] 최종 통계 분석 및 시각화
├── Wilcoxon + BH-FDR 다중 비교 보정 (§0.2)
├── Cohen's d + Bootstrap 95% CI
├── Pareto Frontier (RMSE vs NASA Score)
└── 문헌 비교 기준선 표 (§0.5)
```

---

*최초 작성: 2026-07-02 (H5·H6·H7)*  
*개정 1: 2026-07-02 — H2 추가, 리뷰어 검토 반영 (Hypothesis_critics.md)*  
*개정 2: 2026-07-02 — §0 공통방법론(통계검정·성공기준·재현성·백본민감도), N7 RevIN, M3 Attention-Gate, L6 Pinball, L7 HubA 추가*  
*기반: Hypothesis_PossibleValidation.MD · Hypothesis_critics.md · EDA_Report.MD · Consensus MCP 검색 결과*
