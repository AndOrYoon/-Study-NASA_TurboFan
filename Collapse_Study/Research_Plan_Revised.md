# Research Plan (Revised)
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression: Mechanisms, Detection, and Prevention

**작성일:** 2026-09-11  
**상태:** Revised Draft v0.2  
**연구자:** Young Seog Yoon (ETRI)  
**작업 디렉토리:** `C:\BMAD_PY313\Collapse_Study\`

---

## 1. 연구 배경과 문제 정의

### 1.1 출발 관찰

- 기존 BMAD H6 실험에서 FD003의 단순 LSTM 기준선(M0)은 RMSE `43.23 ± 0.18`을 보인 반면, 후속 모델(M3)은 `14.78 ± 1.32`를 기록하여 표면적으로 65.8% 개선된 것처럼 나타났다.
- 사후 분석 결과 M0는 서로 다른 엔진에 대해 거의 동일한 값(`약 87`)을 출력했으며, 5개 시드 모두 예측 표준편차가 `0.0002` 미만이었다.
- 학습 프로토콜을 수정한 동일 M0의 FD003 RMSE는 `12.97 ± 0.67`로 회복되었다. 이 값은 기존 FD003 연구에서 보고된 대략적인 성능 범위와도 정합적이다.
- 따라서 본 사례의 핵심은 정확도 저하가 아니라 **모델이 입력에 따른 예후 함수를 사실상 학습하지 못했음에도 통상적인 성능 비교에 기준선으로 사용될 수 있었다는 점**이다.

> 본 연구의 중심 질문은 “어떻게 RUL 정확도를 더 높일 것인가?”가 아니라, **“개선 폭을 주장하기 전에 기준선이 입력으로부터 유의미한 함수를 학습했는지를 어떻게 검증할 것인가?”**이다.

### 1.2 개념 구분

MSE 위험을 최소화하는 Bayes 예측자는 일반적으로 조건부 평균 `E[Y|X=x]`이다. 그러나 조건부 평균은 입력 `x`에 따라 변할 수 있으므로, 이것만으로 모든 입력에 거의 동일한 상수를 출력하는 현상을 설명할 수 없다. 본 연구는 다음 두 현상을 구분한다.

1. **Conditional-Mean Compression (CMC)**  
   입력 정보는 사용하지만 예측 분산이 축소되고 극단값이 평균 방향으로 편향되는 정상 회귀의 과소분산 현상.

2. **Functional Mean-Prediction Collapse (MPC)**  
   입력 변화와 무관하게 예측값이 거의 상수로 수렴하여 예후적 변별력을 상실하는 기능적 실패 상태.

본 연구에서 MPC는 다음과 같이 정의한다.

> **Mean-Prediction Collapse (MPC)** is a functional regression failure in which a trained model produces an almost input-invariant prediction, exhibits negligible output dispersion relative to the target distribution, and performs no better in practical terms than an independently specified constant predictor.

예측 상수 `c`가 학습 레이블 평균에 가까울 때 이를 **mean-centered MPC**라고 부른다. MPC라는 명칭의 새로움 자체가 기여점은 아니다. 본 연구의 기여는 기존 conditional-mean 또는 regression-to-the-mean 논의와 구분되는 **기능적 상수 예측 실패를 RUL 학습 프로토콜의 맥락에서 조작적으로 정의하고, 발생 동역학·감지·예방을 체계적으로 검증하는 것**이다.

### 1.3 선행 개념과의 경계

| 개념 | 핵심 의미 | 본 연구와의 관계 |
|---|---|---|
| Conditional-mean prediction | MSE 최적해가 `E[Y|X]` | 이론적 배경이지만 상수 붕괴와 동일하지 않음 |
| Regression-to-the-mean / central-tendency bias | 극단값을 평균 방향으로 과소·과대예측 | CMC에 가까움 |
| Mean collapse in multimodal regression | 여러 가능한 출력을 평균화 | 보조 이론; 본 연구의 프로토콜 실패와 구별 |
| Neural collapse | 분류 모델의 훈련 말기 표현 기하 현상 | 직접적 이론 근거로 사용하지 않음 |
| Generative model collapse | 생성 데이터를 반복 학습하며 분포가 퇴화 | 본 연구와 다른 현상 |
| Vanishing gradient | 장기 의존성 학습의 느림·실패 | 가능한 경쟁 설명 중 하나 |
| Protocol-induced baseline degradation | 절차 때문에 기준선이 비정상적으로 약화 | 본 연구의 직접적 문제 프레임 |

“Clever Hans effect”라는 표현은 사용하지 않고, 보다 정확한 **benchmark artifact**, **evaluation artifact**, **protocol-induced baseline degradation**을 사용한다.

### 1.4 연구 중요성

붕괴한 기준선과 후속 모델을 비교하면 모델 개선폭이 크게 과장될 수 있다. FD003의 `43.23 → 12.97` 변화는 특정 문헌의 결과가 잘못되었다는 증거가 아니라, **동일 구현에서도 학습 프로토콜만으로 기준선이 문헌과 부합하는 정상 영역과 기능적 붕괴 영역 사이를 이동할 수 있다는 사례 증거**다. 따라서 본 연구는 개별 모델의 성능 향상보다 PHM 벤치마크의 재현성과 비교 신뢰성을 다룬다.

---

## 2. 연구 목표와 핵심 기여

### 2.1 중심 주장

> Training and model-selection protocols can select an apparently valid but functionally non-informative, near-constant RUL predictor, thereby inflating claimed improvements over a degraded baseline.

### 2.2 목표

1. MPC를 일반적인 예측 과소분산과 구분되는 RUL 회귀의 기능적 실패로 정량화한다.
2. validation composition, early-stopping dynamics, RUL labeling 및 최적화 조건의 상호작용을 규명한다.
3. test set을 사용하지 않고 훈련·검증 출력만으로 MPC를 감지하는 저비용 sanity check를 개발한다.
4. 여러 데이터셋과 모델 계열에서 가장 낮은 비용으로 MPC를 예방하는 개입을 찾는다.

### 2.3 예상 기여

- **Characterization:** PDR, 입력 민감성, 독립 constant baseline 대비 성능을 결합한 MPC 조작적 정의.
- **Mechanism:** validation-set composition과 early stopping이 초기 mean-like solution을 선택하는 과정에 대한 training-dynamics 증거.
- **Benchmark reliability:** 붕괴 기준선이 후속 모델의 개선폭을 얼마나 왜곡하는지 정량화.
- **Safeguard:** test 접근 없이 실행 가능한 validation-time 경고 규칙과 최소비용 예방 프로토콜.
- **Scope evidence:** C-MAPSS 서브셋, 모델 계열 및 추가 PHM 데이터에서의 재현 범위와 한계 제시.

---

## 3. 연구 질문과 검증 가설

### RQ1 — 존재와 메커니즘

**RQ1:** Under what training, validation, labeling, and optimization conditions does an RUL regressor converge to an input-insensitive mean-like solution?

- **H1a:** fixed validation split 자체가 MPC를 직접 생성하지는 않으며, validation composition과 early stopping의 상호작용이 mean-like 초기 해를 선택·유지할 확률을 높인다.
- **H1b:** collapse run은 정상 run보다 훈련 초기 validation PDR이 빠르게 0에 접근하고, 낮은 입력 민감성과 비양(非正)의 validation `R²`를 보인다.
- **H1c:** patience, early-stopping 시작 epoch, learning rate, output bias initialization 및 RUL clipping이 MPC 발생 확률에 유의한 주효과 또는 상호작용 효과를 갖는다.

### RQ2 — 취약성 및 일반화

**RQ2:** Which dataset, label-distribution, and architecture characteristics increase susceptibility to MPC?

- **H2a:** MPC 취약성은 단순한 “합성 여부”보다 RUL clipping ratio, 레이블 분산, inter-unit variability, fault-mode heterogeneity 및 operating-condition heterogeneity로 더 잘 설명된다.
- **H2b:** LSTM에서만 나타나는 구현 특이 현상이 아니라면 MLP, 1D-CNN, GRU/LSTM 중 둘 이상의 모델 계열에서 재현된다.
- **H2c:** 발생률과 원인은 데이터셋별로 다르며, 사전에 특정 C-MAPSS 서브셋의 취약성 순위를 단정하지 않는다.

### RQ3 — 조기 감지

**RQ3:** Can MPC be detected using only training and validation outputs, without accessing the test set?

- **H3a:** validation PDR, validation `R²`, constant-baseline ratio 및 입력 민감성의 조합이 단일 raw prediction standard deviation보다 안정적으로 MPC를 탐지한다.
- **H3b:** calibration 데이터에서 고정한 경고 규칙은 미사용 데이터셋·시드·아키텍처에서도 사전 정의된 민감도와 특이도를 유지한다.

### RQ4 — 예방

**RQ4:** What is the lowest-cost intervention that reliably prevents MPC across datasets and architectures?

- **H4a:** early-stopping warm-up 또는 validation-time collapse monitor가 단순한 per-seed split 변경보다 데이터셋 전반에서 안정적으로 작동한다.
- **H4b:** 예방 조치는 정상 run의 RMSE를 유의하게 악화시키지 않으면서 MPC 발생률과 재시작 횟수를 감소시킨다.

Huang(2026)의 multimodal regression 논의는 squared loss와 평균화의 이론적 배경으로만 사용하며, 이를 반박하거나 “붕괴의 필연성”을 검증하는 별도 연구질문으로 두지 않는다.

---

## 4. MPC의 조작적 정의와 지표

### 4.1 핵심 지표

검증셋 `V`에서 다음을 계산한다.

**Prediction Dispersion Ratio (PDR)**

$$
\mathrm{PDR}_V=\frac{s(\hat{Y}_V)}{s(Y_V)+\epsilon}
$$

**Constant-Baseline Ratio (CBR)**

$$
\mathrm{CBR}_V=
\frac{\mathrm{RMSE}(Y_V,\hat{Y}_V)}
{\mathrm{RMSE}(Y_V,c_{train})+\epsilon},
\qquad c_{train}=\overline{Y}_{train}
$$

`CBR ≈ 1`이면 모델이 사전에 정의된 학습평균 상수 예측기와 실질적으로 비슷한 성능임을 뜻한다. `c_train`은 validation 또는 test 레이블을 사용해 정하지 않는다.

**Explained variance / coefficient of determination**

$$
R_V^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y_V)^2}
$$

**Input Sensitivity Score (ISS)**

$$
\mathrm{ISS}_V=
\frac{1}{|V|}\sum_{i\in V}
\frac{\|\hat f(x_i+\delta_i)-\hat f(x_i)\|}
{\|\delta_i\|+\epsilon}
$$

여기서 `δ`는 사전 정의된 작은 센서 교란 또는 시간축 permutation이다. 교란은 물리적으로 타당한 범위에서 설정한다.

### 4.2 판정 원칙

MPC 판정은 `pred_std < 1.0` 같은 단일 절대 임계값으로 정의하지 않는다. 초기 분석에서는 다음 세 축을 사용한다.

1. **저분산:** `PDR_V`가 calibration 분포의 사전 정의 하위 임계값 미만.
2. **비정보성:** `R²_V ≤ 0` 또는 낮은 ISS.
3. **상수 기준선 동등성:** CBR의 equivalence interval이 1 주변에 포함.

최종 binary rule과 임계값은 discovery/calibration set에서만 고정하고, held-out confirmatory conditions에 한 번 적용한다. MPC severity는 연속 지표로도 함께 보고하여 임의적 이분화의 영향을 줄인다.

### 4.3 독립적 참조 레이블

detector가 자기 자신을 정답으로 정의하는 순환성을 피하기 위해, calibration run의 참조 레이블은 아래 증거를 종합해 blind adjudication한다.

- prediction-versus-target 산점도 및 범위
- 입력 교란 전후 출력 변화
- constant predictor와의 paired prediction 차이
- 학습 epoch별 PDR·`R²` 궤적

판정자는 detector 후보의 최종 점수와 test 결과를 보지 않는다. 판정 일치도(Cohen's κ)와 불일치 해결 규칙을 사전 등록한다.

---

## 5. 실험 설계

### 5.1 공통 원칙

- test set은 최종 성능 및 외적 확인에만 한 번 사용한다.
- 모든 preprocessing 파라미터, RUL clipping, feature selection 및 normalization은 training data만으로 적합한다.
- engine/unit 단위로 분할하여 window leakage를 방지한다.
- 주 분석과 임계값, 제외 기준, seed 목록 및 통계모형을 가능한 범위에서 사전 등록한다.
- 모든 run에 대해 epoch별 metric, checkpoint, split manifest, 환경·코드 버전을 저장한다.
- 탐색 단계는 조건당 10 seeds, 확인 단계의 seed 수는 사전 pilot 기반 power 또는 precision analysis로 결정한다.

### Phase 0 — 관찰 재현과 구현 감사

**목적:** 원래 FD003 현상이 코드 오류, 데이터 누수 또는 평가 파이프라인 오류가 아님을 확인한다.

- 원본 M0 run과 수정 run을 동일 코드베이스에서 재현.
- label alignment, inverse scaling, test RUL 결합, padding/masking, hidden-state 처리, checkpoint 복원 및 seed 설정 감사.
- 학습평균 상수, validation평균 상수, 선형 회귀, persistence/last-observation 등 단순 기준선 추가.
- 5개 원본 seed의 예측 파일과 재실행 결과 비교.

**진행 기준:** 상수형 출력이 구현·평가 오류로 설명되지 않고 독립 재실행에서 재현될 때 Phase 1로 진행한다.

### Phase 1A — 핵심 메커니즘 선별 실험

**목적:** validation/model-selection 및 labeling 조건의 주효과와 상호작용을 추정한다.

| 요인 | 수준 |
|---|---|
| Validation composition | fixed split / per-seed split / stratified-by-lifetime-and-mode split |
| Early stopping | off / on from epoch 0 / on after warm-up |
| RUL labeling | unclipped 또는 기준 설정 / 기존 clipping threshold / 대안 threshold |

- 3×3×3 full factorial을 기본으로 하되 계산 예산이 부족하면 사전 정의 fractional factorial을 사용.
- FD003, 동일 LSTM backbone, 동일 optimizer를 우선 사용.
- 결과: MPC occurrence, PDR trajectory, validation/test RMSE, `R²`, CBR, stopping epoch.
- validation split별 `mean`, `std`, clipping mass, engine lifetime, fault-mode ratio를 기록.

**통계 분석:** mixed-effects logistic regression으로 MPC 발생 확률을 모델링하고, continuous severity에는 robust linear mixed model 또는 beta regression을 사용한다. seed와 split은 random effect 후보로 둔다.

### Phase 1B — 최적화 경쟁 설명 분석

**목적:** mean-like 초기 해에서 벗어나지 못하는 원인을 분리한다.

| 요인 | 후보 수준 |
|---|---|
| patience | 5 / 10 / 20 / 30 |
| learning rate | 기준의 0.1× / 1× / 10× |
| output bias initialization | 0 / train mean / random calibrated |
| optimizer | Adam / AdamW 또는 SGD 대조 |
| loss | MSE / MAE(Huber 포함) / 사전 정의 auxiliary loss |

Phase 1A에서 효과가 큰 조건을 고정하고 sequential design으로 조사한다. 모든 조합을 무차별적으로 실행하지 않고, 주효과 선별 후 필요한 상호작용만 확인한다.

### Phase 2 — Training dynamics 분석

**목적:** 최종 결과가 아니라 collapse가 형성·선택되는 과정을 입증한다.

각 epoch에서 다음을 저장한다.

- train/validation RMSE와 loss
- validation prediction mean, standard deviation, PDR, `R²`, CBR
- output-layer gradient norm과 weight/bias norm
- learning rate, best checkpoint epoch 및 early-stopping counter
- 가능하면 prediction-target correlation과 ISS의 저비용 근사치

대표 collapsed/normal run을 사후 선택하지 않고, 사전 정의 조건에서 모든 seed의 궤적을 제시한다. 핵심 그림은 validation RMSE와 PDR의 epoch별 변화 및 checkpoint 선택 시점을 중첩한 trajectory plot이다.

### Phase 3 — 데이터셋 및 아키텍처 일반화

**목적:** LSTM 구현 특이성과 FD003 특이성을 배제한다.

**데이터셋 축**

- C-MAPSS FD001–FD004
- N-CMAPSS: 실제 비행조건을 반영하지만 degradation trajectory는 합성인 보다 현실적인 synthetic benchmark
- 선택적 실제 run-to-failure 데이터: XJTU-SY, PRONOSTIA 또는 IMS bearing 중 데이터 구조와 RUL 정의가 가장 적합한 하나

**아키텍처 축**

- MLP
- 1D-CNN
- GRU 또는 LSTM

단순하고 재현 가능한 3개 family를 사용한다. 동일 parameter count를 완전히 강제하기보다 합리적인 capacity band와 공통 tuning budget을 적용하고 sensitivity analysis를 제공한다.

**데이터 속성:** label variance, clipping ratio, inter-unit variability, fault/operating-condition heterogeneity, sample size 및 sensor discriminability를 측정하고 MPC 발생률과의 연관성을 분석한다. “synthetic vs real”은 실제 데이터가 포함될 때만 탐색적으로 비교한다.

### Phase 4 — Detector 개발과 외부 확인

**목적:** test set 없는 validation-time MPC 경고 규칙을 개발한다.

1. Phase 1 일부를 calibration set으로 사용하여 PDR, `R²`, CBR, ISS 및 trajectory feature 후보를 비교한다.
2. repeated nested resampling으로 threshold와 모델을 선택한다.
3. Phase 3의 미사용 dataset×architecture 조건에서 rule을 고정한 채 확인한다.
4. sensitivity, specificity, balanced accuracy, AUROC/AUPRC 및 calibration을 95% confidence interval과 함께 보고한다.
5. 단일 지표 규칙과 복합 규칙의 성능·계산비용을 비교한다.

최종 실무 산출물은 “MPC를 확정 진단”하는 도구가 아니라 **추가 확인 또는 재학습을 촉발하는 저비용 warning check**로 표현한다.

### Phase 5 — 예방 전략과 비용-효과

**목적:** 데이터셋·아키텍처 전반에서 최소비용 예방책을 식별한다.

| 전략 | 기대 비용 | 검증 포인트 |
|---|---:|---|
| early-stopping warm-up | 낮음 | collapse 감소와 과훈련 위험 |
| patience 증가 | 낮음~중간 | 추가 epoch 대비 효과 |
| stratified/per-seed validation | 낮음 | split variance 감소 여부 |
| validation PDR monitor + restart | 낮음 | 재시작률과 총 계산량 |
| output-bias reinitialization | 매우 낮음 | 특정 초기화 의존성 |
| Huber/MAE 또는 auxiliary loss | 낮음~중간 | 정상 run 성능 손실 여부 |

**성공 기준**

- confirmatory conditions에서 MPC occurrence의 통계적으로 유의한 감소
- 정상 프로토콜 대비 RMSE non-inferiority margin 이내
- wall-clock time, epochs, restart를 포함한 총 계산 오버헤드 보고
- 한 데이터셋의 0% 발생률만으로 “완전 방지”를 주장하지 않음

---

## 6. 통계 분석 계획

### 6.1 주요 결과변수

- **Primary:** MPC occurrence 또는 continuous MPC severity
- **Secondary:** validation/test RMSE, NASA score, PDR, `R²`, CBR, ISS, stopping epoch, 계산시간
- **Benchmark distortion:**

$$
\Delta_{inflation}=
\left(\frac{E_{collapsed}-E_{new}}{E_{collapsed}}\right)
-\left(\frac{E_{valid}-E_{new}}{E_{valid}}\right)
$$

여기서 `E`는 동일 평가 지표의 error이며, `E_valid`는 정상성 검사를 통과한 기준선이다. 분모에 따라 개선율 해석이 불안정할 수 있으므로 absolute error difference도 함께 보고한다.

### 6.2 추론

- binary MPC: mixed-effects logistic regression, odds ratio와 95% CI.
- continuous outcomes: factorial ANOVA보다 mixed-effects model을 우선 사용하고 seed, split, dataset, architecture의 반복 구조를 반영.
- 다중 비교: 사전 정의한 주요 contrast에 Benjamini–Hochberg FDR 적용.
- 예방 효과: collapse risk difference/risk ratio와 RMSE non-inferiority를 함께 검정.
- detector: dataset 또는 architecture 단위 group split으로 leakage를 방지하고 bootstrap CI 제공.
- 효과크기와 불확실성을 중심으로 보고하며 p-value만으로 결론을 내리지 않는다.

### 6.3 강건성 분석

- MPC threshold 및 equivalence margin 변화에 대한 sensitivity analysis.
- RUL clipping threshold, normalization, window length 및 target scaling 변화.
- train-mean constant 외에 median constant와 validation-blind simple baseline 비교.
- collapse 판정에서 `R²`, ISS 또는 CBR 하나를 제외한 leave-one-metric-out 분석.

---

## 7. 재현성 및 품질관리

- split manifest와 engine ID를 저장하고 모든 결과 테이블에 run ID를 연결한다.
- Python, CUDA, PyTorch, GPU, 패키지 lockfile, git commit을 기록한다.
- deterministic option 사용 여부와 비결정적 연산을 명시한다.
- raw predictions, epoch log, checkpoint metadata 및 실패 run을 선택적으로 누락하지 않고 보존한다.
- 분석 코드는 결과 테이블 생성까지 자동화하며, confirmatory analysis 전 configuration을 동결한다.
- negative/null result와 예방 실패 조건도 보고한다.
- 기존 H6 논문과의 중복을 피하기 위해 본 연구의 목적, 데이터 재사용 범위 및 신규 분석을 명확히 기술한다.

---

## 8. 구현 자산과 제안 폴더 구조

```text
Collapse_Study\
├── Dataset\
├── Configs\
│   ├── discovery\
│   └── confirmatory\
├── Data_Analysis\
│   ├── Code\
│   ├── Logs\
│   ├── Predictions\
│   └── Results\
├── Protocols\
│   ├── split_manifests\
│   ├── preregistration.md
│   └── mpc_adjudication.md
└── Manuscript\
    ├── Figures\
    ├── Tables\
    └── Sections\
```

### 제안 스크립트

| 파일 | 역할 |
|---|---|
| `00_reproduce_and_audit.py` | 원 관찰 재현 및 구현 감사 |
| `01_mechanism_screen.py` | validation × early stopping × labeling |
| `02_optimization_followup.py` | patience, LR, initialization, loss |
| `03_training_dynamics.py` | epoch trajectory 집계·시각화 |
| `04_generalization.py` | dataset × architecture 확인 |
| `05_detector_validation.py` | calibration 및 held-out 검증 |
| `06_prevention_analysis.py` | 예방책 비용-효과 및 non-inferiority |

기존 LSTM backbone과 corrected protocol 코드는 비교 가능성을 위해 재사용하되, preprocessing·training·evaluation을 공통 모듈로 분리한다.

---

## 9. 예상 핵심 표와 그림

1. **Figure 1:** collapsed vs normal run의 epoch별 validation RMSE, PDR 및 checkpoint 선택 시점.
2. **Figure 2:** target–prediction 산점도와 constant baseline 비교.
3. **Figure 3:** validation composition × early stopping × labeling의 MPC 발생 확률과 interaction.
4. **Figure 4:** dataset × architecture별 MPC 발생률 및 PDR 분포.
5. **Figure 5:** detector ROC/PR curve와 외부 확인 성능.
6. **Table 1:** MPC, CMC 및 관련 collapse 개념의 정의 차이.
7. **Table 2:** 예방 전략의 risk reduction, RMSE 변화, 시간·재시작 비용.
8. **Table 3:** 붕괴 기준선과 정상 기준선 사용 시 apparent improvement의 변화.

---

## 10. 범위, 주장 수준 및 중단 기준

### 10.1 허용 가능한 주장

- 동일 구현에서 protocol-induced functional collapse가 발생하고 기준선 비교를 왜곡할 수 있다.
- 특정 조건에서 validation-time 지표가 해당 실패를 조기에 경고한다.
- 확인한 데이터셋과 아키텍처 범위 안에서 예방책의 효과가 재현된다.

### 10.2 피해야 할 주장

- MPC 또는 평균 붕괴 현상을 최초 발견했다.
- FD003 문헌의 40~50 RMSE가 다수이며 12~13만이 “올바른 값”이다.
- synthetic data가 MPC의 원인이다.
- N-CMAPSS가 실제 run-to-failure 엔진 데이터다.
- validation split이 training loss의 global minimum을 직접 만든다.
- `pred_std` 하나로 모든 도메인의 MPC를 99% 정확도로 진단할 수 있다.
- Huang(2026)을 반박하는 peer-reviewed 결론을 미리 전제한다.

### 10.3 의사결정 기준

- **구현 오류 확인:** 방법론 논문 대신 reproducibility/implementation note로 범위 축소.
- **LSTM 한정 재현:** 일반 회귀 pathology 주장을 철회하고 architecture-specific failure로 재정의.
- **FD003 한정 재현:** dataset-wide safeguard 주장을 철회하고 benchmark case study로 전환.
- **다수 family·dataset 재현:** mechanisms–detection–prevention의 전체 논문으로 진행.

---

## 11. 일정과 실행 우선순위

| 단계 | 내용 | 예상 기간 | 우선순위 |
|---|---|---:|---:|
| Phase 0 | 재현 및 구현 감사 | 1주 | 최상 |
| Phase 1A | 핵심 메커니즘 선별 | 2주 | 최상 |
| Phase 1B | 최적화 후속 분석 | 1~2주 | 높음 |
| Phase 2 | training dynamics 분석 | Phase 1과 병행 | 최상 |
| Phase 3 | dataset × architecture 일반화 | 2~3주 | 최상 |
| Phase 4 | detector calibration/confirmation | 1~2주 | 높음 |
| Phase 5 | 예방 전략 비용-효과 | 2주 | 높음 |
| 논문 작성 | 통계분석·그림·원고 | 3~4주 | 높음 |

총 run 수는 Phase 0 pilot의 평균 학습시간과 collapse prevalence를 확인한 뒤 확정한다. 기존 180 run 고정안보다 **discovery–confirmatory 순차 설계**를 사용해 계산량과 가설 탐색 편향을 동시에 관리한다.

---

## 12. 투고 전략

### 1순위: Reliability Engineering & System Safety (RESS)

신뢰성 예측의 benchmark validity, reproducibility, protocol-induced failure 및 실용 safeguard를 중심으로 구성한다. “새 모델”보다 **기준선이 실제 예후 정보를 학습했는지 검증하는 방법론**을 논문의 중심에 둔다.

### 대안

- IEEE Transactions on Reliability: 신뢰성 평가 프로토콜과 재현성 강조.
- Mechanical Systems and Signal Processing: 실제 bearing data와 강한 신호분석을 포함할 경우.
- PHM Society Conference: Phase 0–2의 mechanism 결과를 먼저 검증받는 경로.

RESS 직접 투고 여부는 최소한 두 architecture family와 C-MAPSS 두 개 이상에서 현상 또는 경계 조건이 확인된 뒤 결정한다.

---

## 13. 주요 위험과 완화책

| 위험 | 영향 | 완화책 |
|---|---|---|
| 단순 코드 버그로 판명 | 독립 연구 novelty 감소 | Phase 0 감사와 독립 구현 검증 |
| MPC 판정의 임의성 | detector 신뢰성 저하 | 다중 기준, blind adjudication, threshold sensitivity |
| test leakage | 성능 과대평가 | validation-only calibration과 held-out confirmation |
| LSTM/FD003 특이성 | 일반화 주장 제한 | MLP·CNN·RNN 및 FD001–FD004 확인 |
| 계산량 폭증 | 일정 지연 | sequential/fractional factorial 및 중간 의사결정 기준 |
| 최근 preprint 의존 | 이론적 기반 취약 | 고전적 squared-error·regression literature를 중심 인용 |
| 기존 H6 논문과 중복 | 출판 윤리·기여 모호 | 데이터 재사용 공개와 신규 질문·분석 명확화 |

---

## 14. 참고문헌 후보

1. Ellefsen, A. L. et al. (2019). Remaining useful life predictions for turbofan engine degradation using semi-supervised deep architecture. *Reliability Engineering & System Safety*.
2. Chao, M. A. et al. (2021). Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics. *Data*, 6(1), 5. N-CMAPSS는 실제 비행조건하의 합성 degradation trajectory임을 명확히 기술.
3. Vabalas, A. et al. (2019). Machine learning algorithm validation with a limited sample size. *PLoS ONE*.
4. Lee, M. & Chen, T. (2025). Systematic bias of machine learning regression models and correction. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
5. Bruna, J., Sprechmann, P., & LeCun, Y. (2015). Super-resolution with deep convolutional sufficient statistics. arXiv:1511.05666.
6. Mathieu, M., Couprie, C., & LeCun, Y. (2016). Deep multi-scale video prediction beyond mean square error. *ICLR*.
7. Prechelt, L. (1998). Early stopping—but when? In *Neural Networks: Tricks of the Trade*.
8. Huang (2026). Multimodal regression/mean-collapse 관련 arXiv preprint. 최종 원고에서는 정확한 서지정보와 버전을 확인하고 보조 이론으로만 인용.
9. CMAPSS 및 FD003 성능 비교 문헌은 최종 systematic search 후 포함하며, 개별 연구의 수치가 잘못되었다고 추정하지 않는다.

---

## 15. 한 문장 연구 요약

> **This study investigates when RUL training protocols select a functionally non-informative near-constant predictor, how that failure distorts benchmark comparisons, and how it can be detected and prevented without using the test set.**
