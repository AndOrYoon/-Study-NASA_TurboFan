# Research Plan
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression: Mechanisms, Detection, and Prevention

**작성일:** 2026-09-10  
**최종 수정:** 2026-09-15 (Draft v0.4 — Phase 4 Option C 재설계, Phase 5 흡수 통합; Decision_log D2)  
**연구자:** Young Seog Yoon (ETRI)  
**작업 디렉토리:** `C:\BMAD_PY313\Collapse_Study\`

---

## 1. 연구 배경과 문제 정의

### 1.1 출발 관찰

- 기존 BMAD H6 실험에서 FD003의 단순 LSTM 기준선(M0)은 RMSE `43.23 ± 0.18`을 보인 반면, 후속 모델(M3)은 `14.78 ± 1.32`를 기록하여 표면적으로 65.8% 개선된 것처럼 나타났다.
- 사후 분석 결과 M0는 서로 다른 엔진에 대해 거의 동일한 값(`약 87`)을 출력했으며, 5개 시드 모두 예측 표준편차가 `0.0002` 미만이었다.
- 학습 프로토콜을 수정한 동일 M0의 FD003 RMSE는 `12.97 ± 0.67`로 회복됐다.

**핵심 관찰 수치:**

| 항목 | 관찰값 |
|------|--------|
| H6 M0 FD003 RMSE (원본 프로토콜) | 43.23 ± 0.18 |
| 5개 시드 예측값 평균 | 86.65 ~ 88.63 |
| 5개 시드 예측값 std | < 0.0002 |
| FD003 테스트 RUL 평균 | ~87 (예측값과 일치) |
| 수정 프로토콜 후 M0 RMSE | **12.97 ± 0.67** |

수정 프로토콜: (1) per-seed random val split, (2) MIN_EPOCHS=30, (3) MAX_EPOCHS=300  
세 조건 중 핵심 원인은 미규명 → Phase 1A 실험 목표.

> 본 연구의 중심 질문은 "어떻게 RUL 정확도를 더 높일 것인가?"가 아니라, **"개선 폭을 주장하기 전에 기준선이 입력으로부터 유의미한 함수를 학습했는지를 어떻게 검증할 것인가?"**이다.

**Undertuned-Baseline 계보와의 관계 — 그리고 구별점:**

ML 재현성 문헌에서 Musgrave et al. (2020) [P23]과 Lučić et al. (2018) [P24]은 동일한 패턴을 독립적으로 확인했다 — "실험 조건을 공정하게 통제하면 주장된 성능 개선이 사라진다." 본 연구는 **같은 계보에 속하되 질적으로 다른 실패 상태**를 다룬다. 선행 연구가 기술한 현상은 *최적화 불충분* (더 충분히 최적화하면 해소)이지만, MPC는 그 극단 형태다 — 기준선이 단순히 낮은 것이 아니라, 입력과 무관하게 상수를 출력하는 **기능적 실패 상태**로 포획된다. 이 구별이 RUL 맥락에서 중요한 이유는, 기능적 붕괴는 적절한 하이퍼파라미터 탐색만으로 해소되지 않으며 특정 프로토콜 조건들의 복합 작용으로 발생하기 때문이다.

### 1.2 개념 구분

MSE 위험을 최소화하는 Bayes 예측자는 일반적으로 조건부 평균 `E[Y|X=x]`이다. 조건부 평균은 입력 `x`에 따라 변할 수 있으므로, 이것만으로 모든 입력에 거의 동일한 상수를 출력하는 현상을 설명할 수 없다. 본 연구는 다음 두 현상을 구분한다.

1. **Conditional-Mean Compression (CMC)**  
   입력 정보는 사용하지만 예측 분산이 축소되고 극단값이 평균 방향으로 편향되는 정상 회귀의 과소분산 현상. Lee et al. (2025) [P7]의 체계적 편향이 이에 해당한다.

2. **Functional Mean-Prediction Collapse (MPC)**  
   입력 변화와 무관하게 예측값이 거의 상수로 수렴하여 예후적 변별력을 상실하는 기능적 실패 상태.

> **Mean-Prediction Collapse (MPC)** is a functional regression failure in which a trained model produces an almost input-invariant prediction, exhibits negligible output dispersion relative to the target distribution, and performs no better in practical terms than an independently specified constant predictor.

MPC라는 명칭의 새로움 자체가 기여점은 아니다. 본 연구의 기여는 기능적 상수 예측 실패를 **RUL 학습 프로토콜의 맥락에서 조작적으로 정의하고, 발생 동역학·감지·예방을 체계적으로 검증하는 것**이다.

**이론적 배경:**
- Huang(2026) [P21]: "squared loss → conditional mean, K>1 모드 시 mean collapse" 이론화. 보조 이론으로만 사용.
- Bruna et al.(2015) [P19]: "regression-to-the-mean problem" 명명.
- Mathieu et al.(2016) [P20]: "inherently blurry predictions under MSE" 실증.

### 1.3 선행 개념과의 경계

| 개념 | 핵심 의미 | 본 연구와의 관계 |
|---|---|---|
| Conditional-mean prediction | MSE 최적해가 `E[Y\|X]` | 이론적 배경이지만 상수 붕괴와 동일하지 않음 |
| Regression-to-the-mean / CMC | 극단값을 평균 방향으로 과소·과대예측 | 정상 학습 완료 후 통계 현상 (≠ MPC) |
| Mean collapse in multimodal regression | 여러 가능한 출력을 평균화 | 보조 이론; 본 연구의 프로토콜 실패와 구별 |
| Neural collapse | 분류 모델 훈련 말기 표현 기하 현상 | 직접 이론 근거로 사용하지 않음 |
| Generative model collapse [P1] | 생성 데이터 반복 학습 시 분포 퇴화 | 단일 판별 훈련인 MPC와 다른 현상 |
| Vanishing gradient [P2] | 장기 의존성 학습의 느림·실패 | MPC는 느린 수렴이 아닌 잘못된 해로의 즉각 수렴 |
| Protocol-induced baseline degradation | 절차 때문에 기준선이 비정상적으로 약화 | 본 연구의 직접적 문제 프레임 |

"Clever Hans effect"라는 표현은 사용하지 않고, **benchmark artifact**, **evaluation artifact**, **protocol-induced baseline degradation**을 사용한다.

### 1.4 연구 중요성

붕괴한 기준선과 후속 모델을 비교하면 모델 개선폭이 크게 과장될 수 있다. FD003의 `43.23 → 12.97` 변화는 특정 문헌의 결과가 잘못되었다는 증거가 아니라, **동일 구현에서도 학습 프로토콜만으로 기준선이 정상 영역과 기능적 붕괴 영역 사이를 이동할 수 있다는 사례 증거**다.

**실무 파급 경로:**
```
붕괴 기준선(RMSE=43) 대비 "65% 개선" 모델 개발
    ↓ 실증 테스트
올바른 기준선이 이미 RMSE=13 수준임을 발견
    ↓
개발 모델 ≈ 단순 기준선 (실질 개선 없음)
    ↓
개발 투자 낭비, 비교 결과 신뢰도 손상
```

**대상별 시사점:**

| 대상 | 시사점 |
|------|--------|
| PHM 연구자 | CMAPSS 기준선 신뢰도 재검토. validation-time sanity check를 표준 프로토콜로 |
| 벤치마크 설계자 | 합성 데이터 균질성이 역설적으로 학습 취약점을 만든다는 설계 교훈 |
| 딥러닝 실무자 | 소규모 데이터 + 고정 val split + 단일 MSE loss 조합은 어느 도메인에서도 위험 |

본 연구는 개별 모델의 성능 향상보다 **PHM 벤치마크의 재현성과 비교 신뢰성**을 다룬다. 연구 대상은 엔진이 아닌 실험 프로토콜의 신뢰성이다.

---

## 2. 선행 연구

### 2.1 CMAPSS 기반 RUL 예측의 학습 불안정성

- **Al-Selwi et al. (2023) [P2]** — LSTM의 소실 기울기(VGP)를 CMAPSS에서 실증 분석. 한계: "느린 수렴" 문제 → MPC의 "즉각적 trivial solution 수렴"과 구별.
- **Ellefsen et al. (2019) [P3]** — 반지도학습으로 레이블 부족 환경의 수렴 불안정성 완화. 합성 데이터 균질성이 collapse 유발한다는 메커니즘 분석 없음.
- **Elsherif et al. (2025) [P5]** — CAELSTM, FD003 RMSE=13.40 달성. 수정 M0(12.97)와 동등 수준 → 올바른 프로토콜의 단순 LSTM이 최신 복잡 모델에 근접.

### 2.2 검증 분할과 편향

- **Vabalas et al. (2019) [P6]** — 소규모 데이터에서 고정 K-fold CV가 강한 성능 추정 편향 유발. RUL 손실 landscape의 trivial solution 왜곡은 미탐구.

### 2.3 회귀 모델의 평균 편향

- **Lee et al. (2025) [P7]** — 회귀 모델의 극값에서 평균 방향 체계적 편향(CMC) 이론화. 차이: Lee는 정상 학습 후 통계적 회귀 / MPC는 학습 자체가 trivial solution에 포획.

### 2.4 합성 데이터의 한계

- **Chao et al. (2021) [P4]** — N-CMAPSS 도입. N-CMAPSS는 실제 비행조건 하의 **합성** degradation trajectory임을 명확히 기술.
- **Das et al. (2024) [P8]** — 합성 데이터 훈련 DNN의 불확실성 정량화.

### 2.5 Model Collapse 용어 구별

- **Dohmatob et al. (2024) [P1]** — 생성 모델이 자신의 출력으로 반복 훈련될 때 성능 퇴화. **MPC와 완전히 다른 현상** — 단일 판별 훈련에서의 trivial solution 수렴.

### 2.6 Undertuned-Baseline Reproducibility 문헌

본 연구 논문의 계보에서 가장 직접적인 선행 흐름은 다음 두 논문이다:

- **Musgrave et al. (2020) [P23]** — *A Metric Learning Reality Check* (ECCV, 554 citations) — 거리 학습 방법들을 동일 조건으로 재평가했을 때, 기존 보고된 "4년간 두 배 이상 성능 향상"이 실험 조건 차이의 아티팩트임을 실증.
- **Lučić et al. (2018) [P24]** — *Are GANs Created Equal?* (NeurIPS, 1,120 citations) — 충분한 하이퍼파라미터 탐색 하에서 GAN 변형 간 유의 차이 없음을 대규모로 실증. "측정 프로토콜이 결론을 결정한다."

두 논문의 핵심 구조 — "공정한 조건에서 비교하면 주장된 개선이 사라진다" — 는 MPC 논문의 논리와 직접 대응한다.

> **배경 참고:** Ferrari Dacrema et al. (2019) [P22] *(RecSys, 694 citations)*는 신경망 추천 알고리즘들이 재현조차 불가능하다는 점을 지적한 논문이다. 주 관심사는 코드 공개·평가 표준화 등 **재현성 인프라**이며, "기준선이 제대로 구현되었는가?"라는 메커니즘 분석 측면에서는 P23/P24보다 우리 논문과의 직접 연결이 약하다.

> 이 계보 없이는 리뷰어가 "기존에 알려진 문제"라 처리할 수 있다. 상세 리뷰는 `Literature_Review.md §5.3` 참조.

### 2.7 연구 갭 요약

| 관련 현상 | 기존 연구 | 미탐구 영역 |
|----------|----------|------------|
| LSTM VGP | Al-Selwi 2023 | MPC는 빠른 수렴이 문제 (느린 수렴이 아님) |
| 검증 편향 | Vabalas 2019 | RUL 손실 landscape 왜곡 메커니즘 |
| 평균 회귀 편향 | Lee 2025 | 학습 포획(training capture) 현상 |
| 합성 데이터 한계 | Chao 2021, Das 2024 | 균질성이 collapse 유발하는 역설 |
| 생성 모델 collapse | Dohmatob 2024 | 단일 판별 모델의 trivial solution |
| Undertuned-baseline reproducibility | Musgrave 2020 [P23], Lucic 2018 [P24] (재현성 인프라: Ferrari Dacrema 2019 [P22]) | PHM/RUL 도메인 적용; 기능적 붕괴(MPC) 수준으로의 극단화 조건 |
| 다봉 회귀 붕괴 | Huang 2026 [P21] | PHM 적용 범위; 이론 배경으로만 사용 |

> **결론:** PHM/RUL 맥락에서 protocol-induced baseline collapse를 체계적으로 연구한 선행 연구 없음.

---

## 3. 연구 목표와 핵심 기여

### 3.1 중심 주장

> LSTM's forget gate approaching zero causes exponential BPTT gradient decay that, under adverse protocol conditions (absent ES warmup, fixed validation split) and homogeneous data, selects a functionally non-informative near-constant predictor — thereby inflating claimed improvements over a collapsed baseline. GRU and other architectures are structurally immune to this specific failure mode.

> *(수정 근거: Phase 3에서 MPC가 LSTM 특이적임이 확인됨 — "프로토콜이 어떤 모델에서나 MPC를 유발한다"는 framing은 실험 결과와 불일치. 핵심 claim을 LSTM 구조적 취약성으로 수정.)*

### 3.2 목표

1. MPC를 일반적인 예측 과소분산과 구분되는 RUL 회귀의 기능적 실패로 정량화한다.
2. validation composition, early-stopping dynamics, RUL labeling 및 최적화 조건의 상호작용을 규명하고, LSTM forget gate → 0 이 MPC의 인과적 메커니즘임을 확증한다.
3. 알려진 모든 예방 개입을 MPC 감소율·RMSE non-inferiority·구현 비용 3축으로 체계적으로 비교하고, 학습 완료 모델을 출력값만으로 소급 감사(retrospective audit)하는 저비용 진단 도구를 개발한다.
4. 여러 데이터셋과 모델 계열에서 최소비용으로 MPC를 예방하는 개입을 찾는다.

### 3.3 예상 기여

- **Characterization:** PDR, 입력 민감성, 독립 constant baseline 대비 성능을 결합한 MPC 조작적 정의.
- **Mechanism:** validation-set composition과 early stopping이 초기 mean-like solution을 선택하는 과정에 대한 training-dynamics 증거.
- **Benchmark reliability:** 붕괴 기준선이 후속 모델의 개선폭을 왜곡하는 정도 정량화 (Δ_inflation).
- **Safeguard:** (a) forget gate clamp [ε,1]·GRU 전환·MAE loss·bias_init 등 예방 개입의 비용-효과 분류체계; (b) 학습 완료 모델의 출력값만으로 MPC를 소급 진단하는 PDR/R² 감사 체크리스트. *(실시간 알람 불가 — Phase 2 확인. 소급 감사 도구로 재프레이밍.)*
- **Scope evidence:** C-MAPSS 서브셋 및 모델 계열에서의 재현 범위와 한계 제시.

---

## 4. 연구 질문과 검증 가설

### RQ1 — 존재와 메커니즘

**RQ1:** Under what training, validation, labeling, and optimization conditions does an RUL regressor converge to an input-insensitive mean-like solution?

- **H1a:** fixed validation split 자체가 MPC를 직접 생성하지는 않으며, validation composition과 early stopping의 상호작용이 mean-like 초기 해를 선택·유지할 확률을 높인다.
- **H1b:** collapse run은 정상 run보다 훈련 초기 validation PDR이 빠르게 0에 접근하고, 낮은 입력 민감성과 비양(非正)의 validation R²를 보인다.
- **H1c:** patience, early-stopping 시작 epoch, learning rate, output bias initialization 및 RUL clipping이 MPC 발생 확률에 유의한 주효과 또는 상호작용 효과를 갖는다.

### RQ2 — 취약성 및 일반화

**RQ2:** Which dataset, label-distribution, and architecture characteristics increase susceptibility to MPC?

- **H2a:** MPC 취약성은 단순한 "합성 여부"보다 RUL clipping ratio, 레이블 분산, inter-unit variability, fault-mode heterogeneity 및 operating-condition heterogeneity로 더 잘 설명된다.
- **H2b:** LSTM에서만 나타나는 구현 특이 현상이 아니라면 MLP, 1D-CNN, GRU/LSTM 중 둘 이상의 모델 계열에서 재현된다.
- **H2c:** 발생률과 원인은 데이터셋별로 다르며, 사전에 특정 C-MAPSS 서브셋의 취약성 순위를 단정하지 않는다.

### RQ3 — 조기 감지

**RQ3:** Can MPC be detected using only training and validation outputs, without accessing the test set?

- **H3a:** validation PDR, validation R², constant-baseline ratio 및 입력 민감성의 조합이 단일 raw prediction standard deviation보다 안정적으로 MPC를 탐지한다.
- **H3b:** calibration 데이터에서 고정한 경고 규칙은 미사용 데이터셋·시드·아키텍처에서도 사전 정의된 민감도와 특이도를 유지한다.

### RQ4 — 예방

**RQ4:** What is the lowest-cost intervention that reliably prevents MPC across datasets and architectures?

- **H4a:** early-stopping warm-up 또는 validation-time collapse monitor가 단순한 per-seed split 변경보다 데이터셋 전반에서 안정적으로 작동한다.
- **H4b:** 예방 조치는 정상 run의 RMSE를 유의하게 악화시키지 않으면서 MPC 발생률과 재시작 횟수를 감소시킨다.

Huang(2026)의 multimodal regression 논의는 squared loss와 평균화의 이론적 배경으로만 사용하며, 이를 반박하거나 "붕괴의 필연성"을 검증하는 별도 연구질문으로 두지 않는다.

---

## 5. MPC의 조작적 정의와 지표

### 5.1 핵심 지표

검증셋 `V`에서 다음을 계산한다.

**Prediction Dispersion Ratio (PDR)**  
*(본 연구에서 정의 — 기존 동일 지표 없음)*

$$\mathrm{PDR}_V = \frac{s(\hat{Y}_V)}{s(Y_V) + \epsilon}$$

예측 분산을 타깃 분산으로 정규화하여 절대 스케일 의존성을 제거한다. PDR → 0이면 예측이 입력과 무관한 상수에 가까워짐을 뜻한다.

**Constant-Baseline Ratio (CBR)**  
*(본 연구에서 명명 — 기저 개념은 Murphy (1988) [P27] MSE Skill Score와 수학적으로 동등: CBR² = 1 − SS)*

$$\mathrm{CBR}_V = \frac{\mathrm{RMSE}(Y_V, \hat{Y}_V)}{\mathrm{RMSE}(Y_V, c_{train}) + \epsilon}, \qquad c_{train} = \overline{Y}_{train}$$

`CBR ≈ 1`이면 모델이 학습평균 상수 예측기와 실질적으로 비슷한 성능임을 뜻한다. `c_train`은 validation 또는 test 레이블을 사용하지 않는다.

**Explained variance (R²)**  
*(표준 지표 — 별도 원출처 표기 불필요)*

$$R_V^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y}_V)^2}$$

**Input Sensitivity Score (ISS)**  
*(본 연구에서 정의 — 입력 교란 기반 민감도 개념은 Zeiler & Fergus (2014) [P28]의 occlusion sensitivity를 시계열·연속 회귀 맥락으로 일반화)*

$$\mathrm{ISS}_V = \frac{1}{|V|} \sum_{i \in V} \frac{\|\hat{f}(x_i + \delta_i) - \hat{f}(x_i)\|}{\|\delta_i\| + \epsilon}$$

여기서 `δ`는 사전 정의된 작은 센서 교란 또는 시간축 permutation이다. ISS → 0이면 모델 출력이 입력 변화에 무감각함을 뜻한다.

### 5.2 판정 원칙

MPC 판정은 `pred_std < 1.0` 같은 단일 절대 임계값으로 정의하지 않는다.

1. **저분산:** `PDR_V`가 calibration 분포의 사전 정의 하위 임계값 미만.
2. **비정보성:** `R²_V ≤ 0` 또는 낮은 ISS.
3. **상수 기준선 동등성:** CBR의 equivalence interval이 1 주변에 포함.

최종 binary rule과 임계값은 discovery/calibration set에서만 고정하고, held-out confirmatory conditions에 한 번 적용한다. MPC severity는 연속 지표로도 함께 보고하여 임의적 이분화의 영향을 줄인다.

### 5.3 독립적 참조 레이블

detector가 자기 자신을 정답으로 정의하는 순환성을 피하기 위해, calibration run의 참조 레이블은 아래 증거를 종합해 blind adjudication한다. Blind adjudication은 임상시험의 blinded endpoint assessment 관행 [P26]에서 적용 원리를 차용한 것이다 — 이 연구가 독창적으로 제안한 방법이 아니라, 순환 논리 문제를 식별하고 기존 임상 방법론을 MPC 감지 맥락에 적용한 것이다.

- prediction-versus-target 산점도 및 범위
- 입력 교란 전후 출력 변화
- constant predictor와의 paired prediction 차이
- 학습 epoch별 PDR·R² 궤적

판정자는 detector 후보의 최종 점수와 test 결과를 보지 않는다. 판정 일치도(Cohen's κ [P25])와 불일치 해결 규칙을 사전 등록한다.

---

## 6. 실험 설계

### 6.1 공통 원칙

- test set은 최종 성능 및 외적 확인에만 한 번 사용한다.
- 모든 preprocessing 파라미터, RUL clipping, feature selection 및 normalization은 training data만으로 적합한다.
- engine/unit 단위로 분할하여 window leakage를 방지한다.
- 주 분석과 임계값, 제외 기준, seed 목록 및 통계모형을 가능한 범위에서 사전 등록한다.
- 모든 run에 대해 epoch별 metric, checkpoint, split manifest, 환경·코드 버전을 저장한다.
- **탐색 단계는 조건당 10 seeds.** 확인 단계의 seed 수는 pilot 기반 power analysis로 결정한다. (기존 180 run 고정안 대신 discovery–confirmatory 순차 설계 사용.)

### Phase 0 — 관찰 재현과 구현 감사

**목적:** 원래 FD003 현상이 코드 오류, 데이터 누수 또는 평가 파이프라인 오류가 아님을 확인한다.

- 원본 M0 run과 수정 run을 동일 코드베이스에서 재현.
- label alignment, inverse scaling, test RUL 결합, padding/masking, hidden-state 처리, checkpoint 복원 및 seed 설정 감사.
- 학습평균 상수, validation평균 상수, 선형 회귀, persistence/last-observation 등 단순 기준선 추가.
- 5개 원본 seed의 예측 파일과 재실행 결과 비교.

**진행 기준:** 상수형 출력이 구현·평가 오류로 설명되지 않고 독립 재실행에서 재현될 때 Phase 1A로 진행한다. 구현 오류로 판명 시 §11.3 의사결정 기준 적용.

### Phase 1A — 핵심 메커니즘 선별 실험

**목적:** validation/model-selection 및 labeling 조건의 주효과와 상호작용을 추정한다. "Silence" 실증 — val-loss 곡선이 건강해 보이는 조건에서도 MPC가 발생하는지 측정한다.

| 요인 | 수준 |
|---|---|
| Validation composition | fixed split / per-seed split / stratified-by-lifetime-and-mode split |
| Early stopping | off / on from epoch 0 / on after warm-up |
| RUL labeling | unclipped / 기존 clipping threshold / 대안 threshold |

- 3×3×3 full factorial을 기본으로 하되 계산 예산이 부족하면 사전 정의 fractional factorial 사용.
- FD003, 동일 LSTM backbone, 동일 optimizer 우선 사용.
- 결과: MPC occurrence, PDR trajectory, validation/test RMSE, R², CBR, stopping epoch.
- validation split별 mean, std, clipping mass, engine lifetime, fault-mode ratio를 기록.
- **Silence 측정:** 표준 early-stopping이 정상 발화하면서 동시에 모델이 MPC 상태인 케이스 비율 보고.

**통계 분석:** mixed-effects logistic regression으로 MPC 발생 확률을 모델링. continuous severity에는 robust linear mixed model 또는 beta regression 사용.

### Phase 1B — 최적화 경쟁 설명 분석

**목적:** mean-like 초기 해에서 벗어나지 못하는 원인을 분리한다.

| 요인 | 후보 수준 |
|---|---|
| patience | 5 / 10 / 20 / 30 |
| learning rate | 기준의 0.1× / 1× / 10× |
| output bias initialization | 0 / train mean / random calibrated |
| optimizer | Adam / AdamW 또는 SGD 대조 |
| loss | MSE / MAE(Huber 포함) / 사전 정의 auxiliary loss |

Phase 1A에서 효과가 큰 조건을 고정하고 sequential design으로 조사한다. 주효과 선별 후 필요한 상호작용만 확인한다.

### Phase 2 — Training Dynamics 분석

**목적:** 최종 결과가 아니라 collapse가 형성·선택되는 과정을 입증한다. Phase 1과 병행 실행.

각 epoch에서 다음을 저장한다.

- train/validation RMSE와 loss
- validation prediction mean, std, PDR, R², CBR
- output-layer gradient norm과 weight/bias norm
- learning rate, best checkpoint epoch 및 early-stopping counter
- prediction-target correlation과 ISS의 저비용 근사치

대표 run을 사후 선택하지 않고, 사전 정의 조건에서 모든 seed의 궤적을 제시한다. 핵심 그림: validation RMSE와 PDR의 epoch별 변화에 checkpoint 선택 시점을 중첩한 trajectory plot.

### Phase 3 — 데이터셋 및 아키텍처 일반화

**목적:** LSTM 구현 특이성과 FD003 특이성을 배제한다.

**데이터셋:** C-MAPSS FD001–FD004  
**아키텍처:** MLP / 1D-CNN / GRU(또는 LSTM)

> N-CMAPSS 및 실제 bearing 데이터(XJTU-SY 등)는 Phase 3 범위에서 제외. Phase 3 결과에 따라 future work로 확장 검토. (Decision_log.md D1 참조)

합리적인 capacity band와 공통 tuning budget을 적용하고 sensitivity analysis를 제공한다. label variance, clipping ratio, inter-unit variability, fault/operating-condition heterogeneity를 측정하고 MPC 발생률과의 연관성을 분석한다.

### Phase 3B — LSTM vs GRU 메커니즘: 왜 LSTM만 취약한가?

**목적:** Phase 3 결과(LSTM 특이적 MPC, GRU 전혀 없음)의 구조적 원인을 규명한다.

**배경:** GRU는 LSTM과 capacity·설정이 거의 동일함에도 동일한 붕괴 유발 프로토콜(A1+B2+C2)에서 MPC가 전혀 발생하지 않았다. 두 아키텍처의 유일한 구조적 차이 — LSTM의 별도 cell state(c_t)와 forget gate — 가 trivial solution 고착 경로를 만드는지 직접 검증한다.

**핵심 가설 (H_mech):**
> LSTM의 forget gate가 trivial solution 근방에서 0에 수렴하면 cell state를 통한 gradient 경로가 차단된다. 이 gradient 차단이 early stopping과 결합하여 MPC를 고착시킨다. GRU는 별도 cell state가 없어 이 경로가 존재하지 않는다.

**실험 설계:**

1. **Gate activation 분석** — 기존 Phase 3 LSTM(collapsed/normal) run에 gate logging 추가 재실행
   - 매 epoch마다 forget gate 평균값 기록 (FD003, seeds 0–9)
   - GRU update gate / reset gate 비교 기록
   - 목표: collapsed LSTM run의 forget gate가 epoch 초기에 0으로 수렴하는지 확인

2. **LSTM 변형 실험** — FD003 × A1+B2+C2 조건 × 10 seeds (30 runs)
   | 변형 | 내용 | 검증 포인트 |
   |------|------|------------|
   | V1_fb1 | forget bias init = +1 (Jozefowicz et al. 2015) | init만 바꿔도 MPC 감소하는가? |
   | V2_fg1 | forget gate 상수 1 고정 (= CEC, gradient 항상 통과) | forget gate 제거 시 MPC 0인가? |
   | V3_base | 기준 LSTM (Phase 3 결과 재확인) | 비교 기준 |

3. **Gradient flow 분석** — collapsed vs normal LSTM/GRU run 비교
   - LSTM: cell state gradient norm vs hidden state gradient norm (epoch별)
   - GRU: hidden state gradient norm
   - 목표: collapsed run에서 cell state gradient가 먼저 소실되는지 확인

**go/no-go (Phase 3B 후):**
- forget gate 가설 확인 (V1/V2에서 MPC 감소 + gate activation 패턴 일치) → 논문에 메커니즘 섹션 추가
- 가설 기각 (변형에서도 MPC 동일) → GRU 비교 결과만 보고, 메커니즘은 "open question"으로 명시

**스크립트:** `04b_lstm_gru_mechanism.py`  
**기간:** 1주  
**우선순위:** 높음 (Phase 4와 병행 가능)

---

### Phase 4 — 예방 분류체계 + 소급 진단 도구 *(Decision_log.md D2 — 2026-09-15)*

> **설계 변경:** 원래 "Detector 개발"에서 "Option C"로 전환. Phase 5(예방 전략)를 흡수 통합.  
> **근거:** Phase 3B에서 H_mech 가설이 인과 확증됨 → 원인이 명료한 상태에서 실시간 detector보다 예방 분류체계가 1차 기여로서 가치가 높음. Phase 2에서 PDR 실시간 알람이 작동 불가(specificity=0.063)임이 이미 확인됨.

#### Part A — 예방 분류체계 (Prevention Taxonomy)

**목적:** Phase 1B·3·3B에서 확인된 모든 예방 개입을 MPC 감소율·RMSE non-inferiority·구현 비용 3축으로 체계적 비교.

| 방법 | 개입 수준 | 기대 MPC rate | 기존 증거 |
|------|----------|:----------:|------|
| GRU로 교체 | 아키텍처 | 0.00 | Phase 3 확인 |
| forget gate = 1 (V2_fg1) | 아키텍처 내부 | 0.00 | Phase 3B 확인 |
| forget gate clamp [ε, 1] | 아키텍처 내부 | TBD (≈0) | Phase 4 신규 |
| MAE loss | Loss function | 0.00 | Phase 1B 확인 |
| bias_init = train_mean | 초기화 | 0.00 | Phase 1B 확인 |
| per-seed random split | 프로토콜 | 0.00 | Phase 1A 확인 |
| patience = 30 | 프로토콜 | 0.20 | Phase 1B 확인 (부분) |
| early-stopping warm-up | 프로토콜 | TBD | Phase 4 확인 |

**forget gate clamp [ε, 1] 신규 실험:**
- Phase 3B의 fig3_fg_at_stop.png 데이터로 ε 후보 도출 (collapsed run의 forget gate 분포 상한)
- ε ∈ {0.001, 0.01, 0.05} 3수준 탐색
- V2_fg1(f=1 고정)과의 차이: clamp는 forget gate 기능(선택적 망각)을 유지하면서 gradient blocking만 제거 → 더 보수적이고 실용적인 개입

**성공 기준:**
- confirmatory conditions에서 MPC occurrence 통계적으로 유의한 감소
- 정상 프로토콜 대비 RMSE non-inferiority margin 이내
- 구현 비용(코드 변경 줄 수, 추가 학습시간) 정량 보고

#### Part B — 소급 진단 도구 (Retrospective Audit Tool)

**목적:** 이미 학습 완료된 모델의 출력값만으로 MPC 여부를 사후 진단하는 PDR/R² 임계값 검증.

> Phase 2에서 확인: PDR 실시간 알람 불가 (양쪽 모두 epoch 초기 PDR≈0). 소급 진단(ES 종료 시점 PDR)은 완벽 분리(collapsed≈0.000002, normal≈0.98).

1. Phase 1A calibration set에서 PDR, R², CBR, ISS 임계값 후보 고정 (test set 미사용).
2. Phase 3의 미사용 dataset×architecture 조건에서 임계값 고정 채 held-out 확인.
3. sensitivity, specificity, balanced accuracy, AUROC를 95% CI와 함께 보고.
4. 산출물: "이미 학습된 모델을 감사(audit)하는 체크리스트" — 실시간 알람이 아닌 **사후 품질 검진 도구**.

**스크립트:** `05_detector_validation.py` (파일명 유지, 내부 구조 재설계 — Part A/B 분리 구현)

---

### 6.7 실험 계획 전체 로드맵 (Experiment Plan)

아래 표는 Phase 0–5의 목적·데이터·기간·우선순위·산출물을 한눈에 정리한 것이다.

| Phase | 명칭 | 주요 목적 | 대상 데이터 | 아키텍처 | 기간 | 우선순위 | 스크립트 |
|---|---|---|---|---|---:|:---:|---|
| 0 | 재현 및 감사 | 붕괴 현상이 구현 오류가 아님 확인 | FD003 | LSTM (M0) | 1주 | 최상 | `00_reproduce_and_audit.py` |
| 1A | 메커니즘 선별 | validation × early stopping × labeling의 주효과·상호작용 정량화 | FD003 | LSTM | 2주 | 최상 | `01_mechanism_screen.py` |
| 1B | 최적화 후속 | patience, LR, bias init, loss의 MPC 기여 분리 | FD003 | LSTM | 1–2주 | 높음 | `02_optimization_followup.py` |
| 2 | Training Dynamics | epoch별 PDR·R²·gradient·checkpoint 궤적 수집 (Phase 1과 병행) | FD003 | LSTM | — | 최상 | `03_training_dynamics.py` |
| 3 | 일반화 확인 | LSTM·FD003 특이성 배제; dataset × architecture 재현 범위 확인 | FD001–FD004 | MLP / 1D-CNN / GRU | 2–3주 | 최상 | `04_generalization.py` |
| **3B** | **LSTM vs GRU 메커니즘** | **forget gate 가설 검증: LSTM만 취약한 구조적 원인 규명** | **FD003** | **LSTM 변형 / GRU** | **1주** | **높음** | **`04b_lstm_gru_mechanism.py`** |
| **4** | **예방 분류체계 + 소급 진단** *(D2)* | **Part A: 모든 예방책 비용-효과 비교 (forget gate clamp 포함); Part B: 학습 완료 모델 소급 감사 도구** | **Phase 1 calibration → Phase 3 held-out; FD001–FD004** | **전체** | **2–3주** | **높음** | **`05_detector_validation.py`** |
| — | 논문 작성 | 통계분석·그림·원고 | — | — | 3–4주 | 높음 | — |

**Phase별 주요 결과물:**

| Phase | 핵심 결과물 |
|---|---|
| 0 | 재현 확인 보고서, 단순 기준선 비교표, 구현 감사 체크리스트 |
| 1A | MPC occurrence table (3×3×3 factorial), "Silence" 케이스 비율, mixed-effects OR 추정치 |
| 1B | 최적화 요인별 MPC 기여 순위, sequential design 결과 |
| 2 | epoch별 PDR·RMSE trajectory plot (Figure 1 후보), checkpoint 선택 시점 중첩 시각화 |
| 3 | dataset × architecture별 MPC 발생률 (Figure 4 후보), go/no-go 판정 |
| **3B** | **forget gate activation trajectory (collapsed vs normal), LSTM 변형별 MPC rate 비교표, 인과 메커니즘 다이어그램 (Figure 4B 후보)** |
| **4** | **Part A: 예방책 분류체계 표 (방법×MPC감소율×RMSE×비용), forget gate clamp ε 분석, (Table 2 후보); Part B: Detector ROC/PR curve, held-out sensitivity·specificity·AUROC (Figure 5 후보)** |

**실행 순서와 의존성:**

```
Phase 0 (구현 오류 배제) ──→ Phase 1A + Phase 2 (병행)
                                  │
                          Phase 1B (Phase 1A 결과 기반 sequential)
                                  │
                          Phase 3 (go/no-go: 2+ dataset, 2+ architecture)
                                  │
                          Phase 3B (LSTM vs GRU 메커니즘) ← H_mech 확증 완료
                                  │
                   Phase 4 (예방 분류체계 + 소급 진단 도구) [D2]
                       Part A: Prevention Taxonomy
                       Part B: Retrospective Audit Tool
                                  │
                            논문 작성
```

**go/no-go 기준 (Phase 3 후):**

| 재현 범위 | 결정 |
|---|---|
| 2개 이상 dataset + 2개 이상 architecture | Full journal paper (mechanisms–detection–prevention) |
| **LSTM 한정 ← 현재 상태** | **architecture-specific failure로 범위 확정 → Phase 3B H_mech 확증으로 메커니즘 추가 확보 완료** |
| FD003 한정 | benchmark case study로 전환 |
| 구현 오류로 판명 (Phase 0) | reproducibility/implementation note로 범위 축소 |

**Phase 3B go/no-go — 결과 확정 (2026-09-14):**

| Phase 3B 결과 | 실제 결과 | 논문 영향 |
|---|---|---|
| forget gate 가설 확인 | **✅ 확증** — V2_fg1 MPC 0/10, V1_fb1 MPC 4/10, GRU 0/10 | **메커니즘 섹션 추가 — 기여 강화** |
| 가설 기각 | — | — |
| V1 forget bias=+1만으로 MPC 감소 | ✅ 부분 확인 (0.60→0.40) | forget gate clamp [ε,1] 실용 개입 근거 |

**인과 메커니즘 요약 (논문 §Mechanism에 포함):**
```
LSTM forget gate → 0  →  BPTT T스텝: f^30 ≈ 0  →  cell state gradient 지수 감쇠
→ early stopping이 trivial solution에서 종료  →  MPC 고착

fg_clamp [ε,1]: ε^30 ≈ 0 (Baseline과 동일) — f 하한 보장만으로는 지수 감쇠 불가 피
GRU: cell state 없음 → 이 BPTT 경로 부재 → MPC 0/10
V2_fg1: f=1 (CEC) → 1^30 = 1 → gradient decay 없음 → MPC 0/10
```
*(Phase 4 fg_clamp 결과로 수정된 설명 — BPTT 전개 기반)*

> 총 run 수는 Phase 0 pilot에서 평균 학습시간과 collapse prevalence를 확인한 뒤 확정한다. Phase 1A 결과만으로도 PHM Society Conference 발표 스코프 충족 — 빠른 커뮤니티 피드백 경로로 활용 가능.

---

## 7. 통계 분석 계획

### 7.1 주요 결과변수

- **Primary:** MPC occurrence 또는 continuous MPC severity
- **Secondary:** validation/test RMSE, NASA score, PDR, R², CBR, ISS, stopping epoch, 계산시간
- **Benchmark distortion:**

$$\Delta_{inflation} = \left(\frac{E_{collapsed} - E_{new}}{E_{collapsed}}\right) - \left(\frac{E_{valid} - E_{new}}{E_{valid}}\right)$$

`E_valid`는 정상성 검사를 통과한 기준선. 분모 불안정성으로 인해 absolute error difference도 함께 보고한다.

### 7.2 추론

- binary MPC: mixed-effects logistic regression, odds ratio와 95% CI.
- continuous outcomes: mixed-effects model 우선 사용; seed, split, dataset, architecture의 반복 구조를 random effect로 반영.
- 다중 비교: 사전 정의 주요 contrast에 Benjamini–Hochberg FDR 적용.
- 예방 효과: collapse risk difference/risk ratio와 RMSE non-inferiority를 함께 검정.
- detector: dataset 또는 architecture 단위 group split으로 leakage 방지, bootstrap CI 제공.
- 효과크기와 불확실성을 중심으로 보고하며 p-value만으로 결론을 내리지 않는다.

### 7.3 강건성 분석

- MPC threshold 및 equivalence margin 변화에 대한 sensitivity analysis.
- RUL clipping threshold, normalization, window length 및 target scaling 변화.
- train-mean constant 외에 median constant와 validation-blind simple baseline 비교.
- collapse 판정에서 R², ISS 또는 CBR 하나를 제외한 leave-one-metric-out 분석.

---

## 8. 재현성 및 품질관리

- split manifest와 engine ID를 저장하고 모든 결과 테이블에 run ID를 연결한다.
- Python, CUDA, PyTorch, GPU, 패키지 lockfile, git commit을 기록한다.
- deterministic option 사용 여부와 비결정적 연산을 명시한다.
- raw predictions, epoch log, checkpoint metadata 및 실패 run을 선택적으로 누락하지 않고 보존한다.
- 분석 코드는 결과 테이블 생성까지 자동화하며, confirmatory analysis 전 configuration을 동결한다.
- negative/null result와 예방 실패 조건도 보고한다.
- 기존 H6 논문과의 중복을 피하기 위해 본 연구의 목적, 데이터 재사용 범위 및 신규 분석을 명확히 기술한다.

---

## 9. 구현 자산과 폴더 구조

### 폴더 구조

```text
Collapse_Study\
├── Dataset\                    ← Junction → C:\BMAD_PY313\Dataset\
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

### 가상환경 및 데이터

```
가상환경:  C:\BMAD_PY313\NASA_TurboFan\Scripts\activate
데이터:    Collapse_Study\Dataset\train_FD00{1-4}.txt  (junction 통해 접근)
           Collapse_Study\Dataset\test_FD00{1-4}.txt
           Collapse_Study\Dataset\RUL_FD00{1-4}.txt
```

### 재사용 코드 (상위 프로젝트)

```
C:\BMAD_PY313\Data_Analysis\Code\H6_fault_mode\phase2_models\h6_p2_model_utils.py  ← LSTM backbone
C:\BMAD_PY313\Data_Analysis\Code\Ad-hoc_Analysis\04_run_h6_corrected.py            ← 수정 프로토콜 레퍼런스
C:\BMAD_PY313\Data_Analysis\Code\shared\op_condition_utils.py                       ← K-means 잔차화
```

### 실험 스크립트 (Data_Analysis\Code\)

| 파일 | Phase | 역할 |
|---|---|---|
| `00_reproduce_and_audit.py` | Phase 0 | 원 관찰 재현 및 구현 감사 |
| `01_mechanism_screen.py` | Phase 1A | validation × early stopping × labeling |
| `02_optimization_followup.py` | Phase 1B | patience, LR, initialization, loss |
| `03_training_dynamics.py` | Phase 2 | epoch trajectory 집계·시각화 |
| `04_generalization.py` | Phase 3 | dataset × architecture 확인 |
| `04b_lstm_gru_mechanism.py` | Phase 3B | forget gate 가설 검증 (LSTM 변형 + GRU gate logging) |
| `05_detector_validation.py` | Phase 4 | Part A: 예방 분류체계 (forget gate clamp 포함); Part B: 소급 감사 도구 calibration |

### 참조 결과 데이터

```
C:\BMAD_PY313\Data_Analysis\Results\H6_fault_mode\raw_predictions_M0_FD003_seed*.csv
C:\BMAD_PY313\Data_Analysis\Results\H6_corrected\h6_corrected_results.csv
```

---

## 10. 예상 핵심 표와 그림

1. **Figure 1:** collapsed vs normal run의 epoch별 validation RMSE, PDR 및 checkpoint 선택 시점.
2. **Figure 2:** target–prediction 산점도와 constant baseline 비교.
3. **Figure 3:** validation composition × early stopping × labeling의 MPC 발생 확률과 interaction.
4. **Figure 4:** dataset × architecture별 MPC 발생률 및 PDR 분포.
5. **Figure 5:** detector ROC/PR curve와 외부 확인 성능.
6. **Table 1:** MPC, CMC 및 관련 collapse 개념의 정의 차이.
7. **Table 2:** 예방 전략의 risk reduction, RMSE 변화, 시간·재시작 비용.
8. **Table 3:** 붕괴 기준선과 정상 기준선 사용 시 apparent improvement의 변화 (Δ_inflation).

---

## 11. 범위, 주장 수준 및 중단 기준

### 11.1 허용 가능한 주장

- 동일 구현에서 protocol-induced functional collapse가 발생하고 기준선 비교를 왜곡할 수 있다.
- 특정 조건에서 validation-time 지표가 해당 실패를 조기에 경고한다.
- 확인한 데이터셋과 아키텍처 범위 안에서 예방책의 효과가 재현된다.

### 11.2 피해야 할 주장

- MPC 또는 평균 붕괴 현상을 최초 발견했다.
- FD003 문헌의 40~50 RMSE가 다수이며 12~13만이 "올바른 값"이다.
- synthetic data가 MPC의 원인이다.
- N-CMAPSS가 실제 run-to-failure 엔진 데이터다.
- validation split이 training loss의 global minimum을 직접 만든다.
- `pred_std` 하나로 모든 도메인의 MPC를 99% 정확도로 진단할 수 있다.
- Huang(2026)을 반박하는 peer-reviewed 결론을 미리 전제한다.

### 11.3 의사결정 기준 (go/no-go)

- **구현 오류 확인:** 방법론 논문 대신 reproducibility/implementation note로 범위 축소.
- **LSTM 한정 재현:** 일반 회귀 pathology 주장 철회 → architecture-specific failure로 재정의.
- **FD003 한정 재현:** dataset-wide safeguard 주장 철회 → benchmark case study로 전환.
- **다수 family·dataset 재현:** mechanisms–detection–prevention의 전체 논문으로 진행.

### 11.4 솔직한 한계 (논문에서 명시 필요)

- 실제 항공 엔진 데이터는 엔진 간 다양성이 높아 MPC 발생 가능성 낮음.
- "실제 배포 시스템에서 MPC 발생" 주장은 현재 증거로 지지 어려움.
- **이 연구의 실무적 의미는 배포 시스템보다 연구 방법론 수준에 있음.** 이 한계를 서론 초반에 명시하면 논문의 범위가 명확해지고 신뢰도가 높아진다.

---

## 12. 일정과 실행 우선순위

| 단계 | 내용 | 예상 기간 | 우선순위 | 상태 |
|---|---|---:|---:|:---:|
| Phase 0 | 재현 및 구현 감사 | 1주 | 최상 | ✅ 완료 |
| Phase 1A | 핵심 메커니즘 선별 | 2주 | 최상 | ✅ 완료 |
| Phase 1B | 최적화 후속 분석 | 1~2주 | 높음 | ✅ 완료 |
| Phase 2 | training dynamics 분석 (Phase 1과 병행) | — | 최상 | ✅ 완료 |
| Phase 3 | dataset × architecture 일반화 | 2~3주 | 최상 | ✅ 완료 |
| Phase 3B | LSTM vs GRU 메커니즘 — H_mech 확증 | 1주 | 높음 | ✅ 완료 |
| Phase 4 | 예방 분류체계 + 소급 진단 도구 *(D2)* | 2~3주 | 높음 | 대기 중 |
| 논문 작성 | 통계분석·그림·원고 | 3~4주 | 높음 | 대기 중 |

총 run 수는 Phase 0 pilot의 평균 학습시간과 collapse prevalence를 확인한 뒤 확정한다.

---

## 13. 투고 전략

### 1순위: Reliability Engineering & System Safety (RESS)

"새 모델"보다 **기준선이 실제 예후 정보를 학습했는지 검증하는 방법론**을 논문의 중심에 둔다. RESS 직접 투고 여부는 최소한 두 architecture family와 C-MAPSS 두 개 이상에서 현상 또는 경계 조건이 확인된 뒤 결정한다.

### 대안

- IEEE Transactions on Reliability: 신뢰성 평가 프로토콜과 재현성 강조.
- Mechanical Systems and Signal Processing: 실제 bearing data와 신호분석을 포함할 경우.
- **PHM Society Conference:** Phase 0–1A의 mechanism 결과만으로도 학회 발표 스코프 해당. 빠른 피드백과 커뮤니티 경고 목적.

---

## 14. 주요 위험과 완화책

| 위험 | 영향 | 완화책 |
|---|---|---|
| 단순 코드 버그로 판명 | 독립 연구 novelty 감소 | Phase 0 감사와 독립 구현 검증 |
| MPC 판정의 임의성 | detector 신뢰성 저하 | 다중 기준, blind adjudication, threshold sensitivity |
| test leakage | 성능 과대평가 | validation-only calibration과 held-out confirmation |
| LSTM/FD003 특이성 | 일반화 주장 제한 | MLP·CNN·RNN 및 FD001–FD004 확인 (§11.3 go/no-go 적용) |
| 계산량 폭증 | 일정 지연 | sequential/fractional factorial 및 중간 의사결정 기준 |
| 최근 preprint 의존 | 이론적 기반 취약 | 고전적 squared-error·regression literature를 중심 인용 |
| 기존 H6 논문과 중복 | 출판 윤리·기여 모호 | 데이터 재사용 공개와 신규 질문·분석 명확화 |
| Undertuned-baseline literature 누락 | 리뷰어 "기존 문제" 처리 위험 | §2.6 계보 편입 — Phase 1 전 systematic search 필요 |

---

## 15. 참고문헌

> 번호 체계는 `Literature_Review.md`와 통합된 단일 체계(P1–P24)를 사용한다. 전체 목록·URL·인용 수는 Literature_Review.md §8 참조. 아래는 Research_Plan에서 직접 인용된 항목만 수록.

[P1] Dohmatob, E. et al. (2024). [Model Collapse Demystified: The Case of Regression](https://consensus.app/papers/details/95b1fdfb14f95d9e8d361847227d6ac4/?utm_source=claude_code). *ArXiv.* (88 citations)

[P2] Al-Selwi, S. et al. (2023). [LSTM Inefficiency in Long-Term Dependencies Regression Problems](https://consensus.app/papers/details/ce8d9e87bdba5132a1ebcf65f76cdd8c/?utm_source=claude_code). *JARA SET.* (157 citations)

[P3] Ellefsen, A. L. et al. (2019). [Remaining useful life predictions for turbofan engine degradation using semi-supervised deep architecture](https://consensus.app/papers/details/56b01d0f6d2b5cc6aa452e487a6d910d/?utm_source=claude_code). *RESS.* (466 citations)

[P4] Chao, M. A. et al. (2021). Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics. *Data*, 6(1), 5. — N-CMAPSS 원논문. (403 citations)

[P5] Elsherif, M. et al. (2025). [A deep learning-based prognostic approach for predicting turbofan engine degradation and remaining useful life](https://consensus.app/papers/details/d677483454575ba2b93a04b7a54a7184/?utm_source=claude_code). *Scientific Reports.* (29 citations) — CAELSTM, FD003 RMSE=13.40

[P6] Vabalas, A. et al. (2019). [Machine learning algorithm validation with a limited sample size](https://consensus.app/papers/details/0b90b9c26ad75cc9bc3c1200ef8f69cb/?utm_source=claude_code). *PLoS ONE.* (1,546 citations)

[P7] Lee, M. & Chen, T. (2025). [Systematic Bias of Machine Learning Regression Models and Correction](https://consensus.app/papers/details/a9eff9e0664b5cde852e28ed5fbbc822/?utm_source=claude_code). *IEEE TPAMI.* (25 citations)

[P8] Das, S. et al. (2024). [Uncertainty-aware deep learning for monitoring and fault diagnosis from synthetic data](https://consensus.app/papers/details/4025d77718a456b58ef100afbe2bb6c8/?utm_source=claude_code). *RESS.* (42 citations)

[P14] Prechelt, L. (1998). Early stopping—but when? In *Neural Networks: Tricks of the Trade.* (2,469 citations)

[P19] Bruna, J., Sprechmann, P., & LeCun, Y. (2015). Super-resolution with deep convolutional sufficient statistics. *arXiv:1511.05666.* (341 citations)

[P20] Mathieu, M., Couprie, C., & LeCun, Y. (2016). Deep multi-scale video prediction beyond mean square error. *ICLR.* (2,020 citations)

[P21] Huang, W. (2026). Resolving Multi-Modal Regression by Difference-Quotient-Based Clustering. *arXiv:2608.25467.* ⚠️ 서지정보·버전·철회 여부 최종 원고 전 확인 필수. 보조 이론으로만 인용.

[P22] [Are we really making much progress? A worrying analysis of recent neural recommendation approaches](https://consensus.app/papers/details/48856aa30214567eb3a05edb5381fe5c/?utm_source=claude_code) — Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. *RecSys 2019.* (694 citations)

[P23] [A Metric Learning Reality Check](https://consensus.app/papers/details/d7dad3a69d4550da87ee3fe451efd59e/?utm_source=claude_code) — Musgrave, K., Belongie, S., & Lim, S.-N. *ECCV 2020.* (554 citations)

[P24] [Are GANs Created Equal? A Large-Scale Study](https://consensus.app/papers/details/3c0cdb0064615c3e86fee6a05b174c70/?utm_source=claude_code) — Lučić, M., Kurach, K., Michalski, M., Gelly, S., & Bousquet, O. *NeurIPS 2018.* (1,120 citations) *(arXiv 2017, NeurIPS 2018 발표)*

[P25] Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37–46. — Cohen's κ 원출처. §5.3 판정 일치도 측정 근거.

[P26] Schulz, K.F., Altman, D.G., Moher, D., & CONSORT Group (2010). CONSORT 2010 Statement: Updated Guidelines for Reporting Parallel Group Randomised Trials. *BMJ*, 340, c332. (2,500+ citations) — 임상시험 blinded outcome assessment의 방법론적 표준. §5.3 blind adjudication 원칙의 출처.

[P27] Murphy, A.H. (1988). Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient. *Monthly Weather Review*, 116(12), 2417–2424. — CBR 기저 개념의 출처. MSE 기반 Skill Score(SS = 1 − MSE/MSE_ref)는 CBR과 수학적으로 동등 (CBR² = 1 − SS). 기상예보 분야에서 null predictor 대비 모델 성능 측정의 표준 프레임.

[P28] Zeiler, M.D., & Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. In *ECCV 2014*, LNCS 8689, 818–833. (19,000+ citations) — ISS의 입력 교란 기반 민감도 개념의 원출처. 공간적 occlusion sensitivity를 본 연구에서 시계열·연속 회귀 맥락(센서 교란 + 시간축 permutation)으로 일반화.

---

## 16. 한 문장 연구 요약

> **This study investigates when RUL training protocols select a functionally non-informative near-constant predictor, how that failure distorts benchmark comparisons, and how it can be detected and prevented without using the test set.**

---

## 17. 논문 목차 (Manuscript Table of Contents)

**제목 후보:**  
*Mean-Prediction Collapse in Deep Remaining Useful Life Regression: Mechanisms, Detection, and Prevention*

---

### 목차 구조

```
1. Abstract
2. Introduction
3. Related Work
4. Methodology
   4.1 Operational Definition and Metrics
   4.2 Experimental Setup
   4.3 Research Hypotheses
   4.4 Statistical Analysis Plan
5. Results: Mechanism and Generalization  [Phase 0–3B]
   5.1 Reproduction and Implementation Audit (Phase 0)
   5.2 Validation Composition × Early Stopping (Phase 1A/1B)
   5.3 Training Dynamics (Phase 2)
   5.4 Dataset and Architecture Generalization (Phase 3)
   5.5 LSTM Forget Gate Mechanism (Phase 3B)
6. Results: Prevention and Retrospective Detection  [Phase 4]
   6.1 Prevention Taxonomy
   6.2 Retrospective Audit Tool
7. External Protocol Audit  [Phase 5]
   7.1 Audit Design and Paper Selection
   7.2 MPC Prevalence in Published Protocols
   7.3 Benchmark Distortion Quantification (Δ_inflation)
8. Discussion
   8.1 Practical Implications for PHM Research
   8.2 Limitations and Scope
9. Conclusion
10. Data and Code Availability
11. References
```

---

### 섹션별 구성 방침

#### 1. Abstract
- MPC 정의 → 발생 조건 → 감지·예방 결과 → 외부 audit 결과 순
- 핵심 수치: LSTM 특이적, FD003 MPC rate, 예방 cost, audit 5편 MPC rate 60–90%

#### 2. Introduction
- 동기: FD003 M0 `RMSE=43.23` → `12.97` 역전 사례 (§1.1)
- MPC 정의 및 CMC·일반 회귀 편향과의 구분 (§1.2)
- 연구 질문 RQ1–RQ4 요약
- 기여 4가지 목록 (§3.3)
- 논문 구조 안내 (1문단)

#### 3. Related Work
- 순서: LSTM 불안정성 → val split 편향 → 회귀 평균 편향 → 합성 데이터 한계 → undertuned-baseline reproducibility 계보 → 연구 갭 표

> **왜 undertuned-baseline 계보(P23·P24)가 필요한가:** 리뷰어가 "기존에 알려진 문제"로 처리하는 위험을 차단. MPC가 단순 불충분 최적화가 아닌 기능적 붕괴(functional failure)임을 구별하는 논거.

#### 4. Methodology
- **4.1 Operational Definition and Metrics** — PDR, CBR, R², ISS 정의 + MPC 판정 원칙 (단일 임계값이 아닌 복합 기준)
- **4.2 Experimental Setup** — 공통 LSTM backbone, FD001–FD004, Phase 0–5 로드맵 개요
- **4.3 Research Hypotheses** — H1a–H4b (가설을 별도 섹션으로 내지 않고 여기에 통합)
- **4.4 Statistical Analysis Plan** — mixed-effects logistic, BH-FDR, equivalence test, blind adjudication

> Hypothesis를 별도 섹션으로 두지 않는 이유: RESS 스타일에서 가설은 Methodology 안에 실험 설계의 일부로 제시하는 것이 자연스럽다. 별도 섹션은 의학·사회과학 논문 스타일에 가까움.

#### 5. Results: Mechanism and Generalization
- **5.1 Phase 0** — 붕괴가 구현 오류가 아님 확인, 단순 baseline 비교 (Table 3 Δ_inflation 1차 제시)
- **5.2 Phase 1A/1B** — MPC 발생률 factorial table, "Silence" 케이스 비율, 최적화 요인별 기여 순위
- **5.3 Phase 2** — epoch별 PDR·RMSE trajectory (Figure 1), checkpoint 선택 시점 중첩
- **5.4 Phase 3** — dataset × architecture MPC 발생률 (Figure 4): LSTM 특이적 확증
- **5.5 Phase 3B** — forget gate activation trajectory (collapsed vs normal), V1/V2 변형 비교 (Figure 4B), 인과 메커니즘 다이어그램

#### 6. Results: Prevention and Retrospective Detection
- **6.1 Prevention Taxonomy** — 방법×MPC감소율×RMSE×비용 분류표 (Table 2), forget gate clamp ε 분석
- **6.2 Retrospective Audit Tool** — PDR/R² 임계값, held-out sensitivity·specificity·AUROC (Figure 5)

#### 7. External Protocol Audit ← 별도 섹션 이유
외부 audit이 독립 섹션을 받아야 하는 근거:

- **방법론 차이:** Phase 0–4는 통제 실험(controlled experiment), Phase 5는 복제 연구(replication study). 동일한 Results 섹션에 묶으면 방법론적 차이가 희석됨.
- **기여 가시성:** 5편 발표 논문의 프로토콜을 동일 조건으로 재실행하여 MPC 발생 확증 — 이것이 "연구실 외부에도 이 문제가 실재함"을 보여주는 핵심 생태 타당도(ecological validity) 증거. 서브섹션으로 묻히면 리뷰어가 중요성을 과소평가할 수 있음.
- **분리된 methodology 기술 필요:** 논문 선정 기준(EAAI/Expert Systems, FD003, stacked LSTM), protocol 추출 방법, 재현 충실도 기준을 별도로 기술해야 감사 결과의 신뢰성이 확보됨.

섹션 구성:
- **7.1 Audit Design and Paper Selection** — 선정 기준(저널, 연도, 아키텍처), 5편 목록, 프로토콜 추출 방법, 재현 충실도 기준
- **7.2 MPC Prevalence in Published Protocols** — 5편 × 10 seeds 결과표 (mpc_rate, RMSE, PDR, stop_epoch), patience 단조 효과, 25-epoch 역설
- **7.3 Benchmark Distortion Quantification** — Δ_inflation 계산: 붕괴 기준선 대비 개선폭 vs 정상 기준선 대비 실제 개선폭 비교

#### 8. Discussion
- **8.1 Practical Implications** — LSTM+고정val+no-warmup 조합 위험, 최소비용 예방 권고, 소급 감사 체크리스트 활용법, 벤치마크 설계 교훈
- **8.2 Limitations and Scope** — LSTM 특이적 (GRU 불해당), C-MAPSS 합성 데이터 한계, 실제 배포 적용 주의, N-CMAPSS·bearing 미포함

#### 9. Conclusion
- RQ1–RQ4 결론 요약 (각 1–2문장)
- 핵심 권고: validation-time PDR check + per-seed split or ES warmup
- Future work: 실제 센서 데이터, Transformer 계열 적용 가능성

#### 10. Data and Code Availability
- CMAPSS 공개 데이터셋 출처
- 실험 코드 저장소 (GitHub 예정)
- split manifests, epoch logs, raw predictions 공개 여부

#### 11. References
- P1–P28 + 추가 인용 통합

---

### 구조 결정 근거 요약

| 결정 | 선택 | 이유 |
|------|------|------|
| Hypothesis 섹션 | Methodology 4.3으로 통합 | RESS 스타일; 가설은 실험 설계의 일부 |
| External Audit 위치 | 섹션 7 (독립) | 방법론 차이 + 생태 타당도 기여 가시성 |
| Results 분할 | 섹션 5 (메커니즘) + 섹션 6 (예방·감지) | Phase 4가 Phase 0–3B와 다른 주장 유형 |
| Discussion 구성 | 2 subsection | 함의 vs 한계 분리; 과도한 세분화 방지 |
