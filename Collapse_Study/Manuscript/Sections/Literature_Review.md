# Literature Review
## Mean-Prediction Collapse in Synthetic RUL Datasets

**작성일:** 2026-09-10  
**목적:** 선행 연구 갭 식별 및 연구 질문 도출  
**관련 파일:** `Research_Plan.md` (RQ 정의), `Previous Studies/` (원문 보관)

---

## 1. 개요

본 리뷰는 Mean-Prediction Collapse(MPC) 연구의 선행 연구 기반을 다음 5개 주제 영역으로 분류·정리한다:

1. CMAPSS 기반 딥러닝 RUL 예측의 현황
2. 훈련 불안정성 — LSTM과 시계열 회귀
3. 조기 종료와 검증 분할 설계
4. 벤치마크 공정성과 재현성
5. 합성 데이터의 한계와 평균 편향

각 섹션 말미에 **연구 갭(Gap)**을 명시하고, 섹션 7에서 종합 갭과 연구 질문을 도출한다.

### 1.1 연구 대상 현상: Mean-Prediction Collapse (MPC)

**명명 프레임**
- 기존 문헌은 이 현상을 "regression-to-the-mean problem" [P19], "mean collapse" [P21], "blurry predictions under MSE" [P20] 등 산발적으로 서술해 왔다. 본 연구는 PHM/RUL 벤치마크 맥락에서 이 현상이 프로토콜 결함과 결합하여 완전 기능 소실로 극단화되는 사례를 처음으로 기록하며, 이 현상 계열을 **Mean-Prediction Collapse (MPC)** 로 통합 명명한다.

**정의**

> **Mean-Prediction Collapse (MPC)** refers to a degenerate regression behavior in which model predictions exhibit substantially reduced dispersion relative to the target distribution and concentrate around the population or conditional mean, thereby suppressing input-dependent variations and systematically underrepresenting extreme target values.

- *RUL 예측 맥락 적용:* In the context of RUL prediction, MPC manifests as a model outputting a near-constant value approximating the dataset mean, losing all prognostic discriminability across engines.
- *본 연구 관찰 사례:* H6 M0 FD003 — 5개 시드 전부 예측값 ≈ 87 (FD003 테스트 RUL 평균), pred_std < 0.0002, RMSE = 43.23 (정상 수렴 후 12.97)
- *붕괴 심각도 구분:* MPC가 극단화될 때, 모델은 조건부 평균 E[y|x]가 아닌 **주변 평균 E[y]** (marginal mean, 데이터셋 전체 RUL 평균 ≈ 87)로 수렴한다 — 입력 x를 완전히 무시하는 더 극단적인 붕괴 형태. Huang(2026)[P21]이 기술한 mean collapse는 E[y|x] 수준의 부분적 붕괴이며, 본 연구 MPC는 그보다 한 단계 더 심각하다.

**정의 구성 요소와 선행 연구 매핑**

| 정의 구성 요소 | 근거 논문 |
|---|---|
| "reduced dispersion relative to target distribution" | Mathieu et al. 2016 [P20] — "blurry predictions under MSE" |
| "concentrate around... conditional mean" | Huang 2026 [P21], Bruna et al. 2015 [P19] |
| "suppressing input-dependent variations" | Huang 2026 [P21] — "K>1 모드 시 mean이 모든 모드와 멀어짐" |
| "systematically underrepresenting extreme values" | Lee et al. 2025 [P7] — systematic mean bias |
| "degenerate" (MSE loss landscape) | Li et al. 2022 [P18], Papyan et al. 2020 [P17] |

**개념적 근거 (기존 문헌)**
- **MSE → 평균 예측 문제 [P19][P20][P21]:** 본 현상의 가장 직접적인 이론적 토대
  - [Huang 2026][P21]: "제곱 손실 하에서 제약되지 않은 회귀자는 조건부 평균으로 수렴하며, K>1 모드 존재 시 이 평균은 모든 모드로부터 멀리 떨어진다" — MPC 이론을 가장 명시적으로 서술한 최신 논문. **"mean collapse"** 를 실험 최악 기준선으로 수치화
  - [Bruna et al. 2015][P19]: 초해상도(super-resolution)에서 MSE 기반 point estimate가 **"regression-to-the-mean problem"** 을 겪는다고 직접 명명 — 조건부 분포의 다봉성(multimodality)을 MSE가 포착하지 못해 조건부 평균 E[y|x]으로 수렴 [341회 인용]
  - [Mathieu et al. 2016][P20]: 비디오 예측에서 MSE 손실이 "inherently blurry predictions"을 유발함을 실증하고, MSE를 넘어서는 손실 함수(adversarial, gradient difference) 설계 필요성 제시 [2,020회 인용]
  - **MPC와의 연결:** 세 논문이 입증한 "MSE → 조건부 평균 수렴" 메커니즘이 MPC의 이론적 기반. 다만 기존 연구는 blurry 이미지 또는 새로운 모델 설계에 집중하며, 프로토콜 결함이 RUL 벤치마크에서 이를 완전 붕괴로 극단화하는 조건은 미탐구
- **MSE 손실 지형 [P18]:** Li et al. (2022)은 분류 맥락에서 MSE 하의 Neural Collapse 해가 global minimizer임을 증명 — 상수 예측이 MSE 경관의 광범위한 stable minimum을 형성할 수 있다는 이론적 근거 [133회 인용]
- **Neural Collapse (분류 유사 개념) [P17]:** Papyan et al. (2020) — 훈련 말기 분류 모델의 last-layer feature가 클래스 평균으로 수렴하는 NC1 현상. 분류의 "클래스 평균 수렴"은 회귀의 "전체 레이블 평균 수렴"(MPC)의 개념적 대응물이나, NC는 정상 훈련의 종단 현상, MPC는 프로토콜 결함으로 인한 비정상 초기 수렴 [1,007회 인용, PNAS]

**선행 개념과의 구별** (학술적 독립성 근거)
- **Model collapse [P1]:** 생성 모델의 반복 자기 학습 시 출력 다양성 상실 [Dohmatob et al. 2024] → MPC는 단일 판별 훈련에서 발생, 반복 자기 훈련 불필요
- **체계적 평균 편향 [P7]:** 정상 학습 완료 후 분포 꼬리의 통계적 회귀 현상 [Lee et al. 2025] → MPC는 학습 자체의 최적화 실패, 꼬리 편향이 아닌 전범위 상수화
- **소실 기울기 [P2]:** 기울기 소멸로 인한 느린 수렴 [Al-Selwi et al. 2023] → MPC는 기울기 존재 상태에서 잘못된 해로 **즉각** 수렴
- **조기 종료 가정 [P14]:** 검증 손실이 일반화를 반영한다는 전제 [Prechelt 1998] → MPC는 이 전제를 정면으로 위반 (val loss 낮아도 예측 불능)
- **Neural Collapse [P17]:** 분류 모델의 정상 훈련 말기 현상 [Papyan et al. 2020] → MPC는 회귀 모델의 비정상 훈련 초기 현상

**왜 중요한가**
- MPC는 검증 손실 모니터링만으로는 감지할 수 없어, 기준선 성능을 인위적으로 저하시키고 후속 복잡 모델의 개선 폭을 부풀리는 **"Clever Hans" 효과**를 유발한다 — 벤치마크 연구의 결론 신뢰성 자체를 훼손한다.
- 실제 사례: H6 M0 FD003 RMSE = 43.23 (MPC 상태) vs. 12.97 (정상 수렴) → 허위 "65.8% 개선" 주장 발생

---

## 2. CMAPSS 기반 딥러닝 RUL 예측의 현황

### 2.1 벤치마크로서의 CMAPSS

- NASA C-MAPSS 시뮬레이터로 생성된 FD001~FD004 데이터셋은 PHM 분야의 표준 벤치마크
- 4개 서브셋은 운전 조건(1개 vs. 6개) × 결함 모드(1개 vs. 2개)의 2×2 설계
- 광범위한 딥러닝 RUL 예측 연구의 공통 비교 기반으로 수백 편의 논문에서 활용

### 2.2 주요 딥러닝 방법론 흐름

- **CNN 기반:** [Li et al. 2018][P16]은 DCNN을 RUL 예측에 적용, 원시 센서 데이터 직접 입력으로 특징 추출 자동화. CMAPSS 전 서브셋에서 당시 SOTA 달성 (1,561회 인용 — 이 분야 기초 논문)
- **반지도 학습:** [Ellefsen et al. 2019][P3]은 레이블 부족 환경에서 비지도 사전 학습 + 지도 fine-tuning 구조 제안. 수렴 안정성 개선 효과 확인 (466회 인용)
- **물리-데이터 융합:** [Chao et al. 2020][P4]은 N-CMAPSS를 도입하며 물리 기반 모델과 딥러닝의 결합 프레임워크 제안. "훈련 데이터의 제한된 대표성"을 핵심 한계로 지적 (403회 인용)
- **주의 메커니즘:** [Elsherif et al. 2025][P5]의 CAELSTM은 컨볼루션 오토인코더 + 주의 LSTM으로 FD003 RMSE=13.40 달성 (29회 인용)
- **베이지안 딥러닝:** [Zhuang et al. 2023][P15]은 예측 불확실성을 포함한 정비 의사결정 프레임워크 제안. CMAPSS를 검증 기반으로 활용 (173회 인용)

### 2.3 공통 관찰

- 대부분의 논문이 자체 제안 모델과 기준선 LSTM/CNN을 비교하는 구조
- **기준선 훈련 프로토콜이 논문마다 상이** — 검증 분할 방식, 조기 종료 조건, 클리핑 등이 명시되지 않는 경우 다수
- FD003에서 보고되는 기준선 RMSE 범위가 13~43대로 매우 넓음 → 프로토콜 차이가 원인으로 의심

> **Gap 1:** CMAPSS 기반 연구에서 기준선 훈련 프로토콜의 표준화가 부재하며, 이로 인한 기준선 RMSE 편차가 체계적으로 분석된 바 없다.

---

## 3. 훈련 불안정성 — LSTM과 시계열 회귀

### 3.1 소실 기울기 문제

- **[Al-Selwi et al. 2023][P2]** — LSTM이 CMAPSS 데이터에서 장기 의존성 학습 시 소실 기울기 문제(VGP)에 취약함을 실증 분석 (157회 인용)
  - 핵심: 기울기 소실로 인한 **느린 수렴 또는 수렴 실패**
  - MPC와의 구별: MPC는 반대로 **너무 빠른 수렴** — trivial solution으로의 즉각적 포획

### 3.2 평균 예측 편향

- **[Lee et al. 2025][P7]** — 회귀 모델이 분포 꼬리(극값)에서 평균 방향으로 체계적으로 편향된 예측 생성함을 이론화, 교정 방법 제안 (25회 인용, IEEE TPAMI)
  - 현상적 유사성: 예측이 평균으로 수렴
  - 근본 차이: Lee의 편향은 정상 학습 후 발생하는 **통계적 회귀 현상** / MPC는 **학습 자체가 trivial solution에 포획**되는 현상

### 3.3 생성 모델의 "Model Collapse"와의 구별

- **[Dohmatob et al. 2024][P1]** — 생성 모델이 자신의 출력으로 반복 훈련될 때 성능 퇴화하는 "model collapse" 이론화 (88회 인용, ArXiv)
  - 맥락: LLM·이미지 생성 모델의 반복 자기 학습 시나리오
  - **MPC와 메커니즘 전혀 다름** — 단일 훈련에서 discriminative 모델의 trivial solution 수렴
  - 용어 혼동 주의: 본 연구의 MPC는 Dohmatob의 model collapse와 명확히 구별되어야 함

### 3.4 MSE 손실과 평균 예측 — 구조적 예측 분야의 선례

- **[Bruna et al. 2015][P19]** — 초해상도(super-resolution)에서 MSE point estimate의 한계 분석 (341회 인용, CoRR)
  - 핵심 명제: "point estimates suffer from the **regression-to-the-mean problem**, result of their inability to capture the multi-modality of this conditional distribution"
  - 메커니즘: MSE 최솟값 = 조건부 평균 E[y|x]. 고해상도 이미지의 조건부 분포가 다봉(multimodal)일 때, MSE 최솟값은 흐릿한(blurry) 평균 이미지
  - 솔루션: 조건부 Gibbs distribution + deep convolutional sufficient statistics

- **[Mathieu et al. 2016][P20]** — 비디오 예측에서 MSE를 넘어서는 손실 함수 설계 (2,020회 인용, CoRR)
  - 핵심 명제: "inherently blurry predictions obtained from the **standard Mean Squared Error (MSE) loss function**"
  - MSE 한계의 세 가지 보완 전략: (1) multi-scale architecture, (2) adversarial training, (3) image gradient difference loss
  - 함의: MSE만으로는 구조적 예측의 출력 다양성을 보존할 수 없음

### 3.5 다중 모드 회귀에서의 평균 붕괴 — 최신 직접 선행 연구

- **[Huang 2026][P21]** — "Resolving Multi-Modal Regression by Difference-Quotient-Based Clustering: Fast Coarse Conditional-Label Assignment" (arXiv:2608.25467, 2026년 8월)
  - **현재까지 발견된 MPC와 가장 직접적으로 관련된 논문**
  - **문제 정의:** "제곱 손실 하에서 제약되지 않은 회귀자는 조건부 평균으로 수렴한다. K > 1개 모드가 존재할 때 이 평균은 모든 모드로부터 멀리 떨어져 있다" — MPC의 이론적 근거를 명시적으로 서술
  - **"평균 붕괴(mean collapse)"** 를 실험의 최악 기준선으로 명명·수치화 (MSE=1.33 vs 오라클 0.09)
  - **원인 분석:** "쌍별 모순(pairwise contradictions)" — 거의 동일한 입력에서 매우 다른 출력을 갖는 샘플 쌍이 MSE 최솟값을 평균으로 당김
  - **솔루션:** Difference-Quotient-Based Clustering(DQC) — 클러스터 내 출력-입력 불일치를 최소화하도록 데이터를 분할 후 클러스터별 조건부 네트워크 훈련 (복잡도 O(n²/2))

- **Huang 2026과의 연결 및 구분**
  - **공통 현상:** 두 연구 모두 MSE 손실 하의 "평균 수렴 = 기능 소실" 현상을 다루며, Huang의 "squared loss → conditional mean" 정리가 MPC의 이론적 토대가 됨
  - **원인 구조의 근본적 차이 (핵심 구별점):**
    - Huang(2026): 붕괴 원인 = **데이터 구조** — "pairwise contradictions"(같은 입력 x에 다른 출력 y가 공존) → MSE 최솟값이 평균으로 수렴하는 **구조적·필연적 붕괴**. 해결을 위해 새로운 아키텍처(DQC)가 필요
    - **본 연구:** 붕괴 원인 = **훈련 프로토콜 결함** — 고정 val split + MIN_EPOCHS 부재 → **교정 가능한 절차적 붕괴**. 핵심 반증: 수정 프로토콜 적용 후 동일 데이터·동일 모델(M0)에서 RMSE = 12.97로 정상 수렴. FD003의 K=2 결함 모드는 Huang의 "진정 모호한 입력(truly ambiguous input)" 조건이 아님 — 올바른 프로토콜 하에서 두 모드는 센서 궤적으로 구분 학습 가능
  - **붕괴 수준의 차이:**
    - Huang(2026): E[y|x](조건부 평균)로 수렴 — 입력 x 간 변별 능력 부분 유지
    - **본 연구 MPC:** E[y](주변 평균 = ~87)로 수렴 — 입력과 무관한 완전 상수 예측. 조건부 평균 추정도 실패한 더 극단적 붕괴 형태
  - **본 연구의 독자 기여 (Huang이 다루지 않는 영역):** 프로토콜 원인 인과 분석 / 훈련 완료 후 즉시 감지(pred_std) / 아키텍처 변경 없는 최소 비용 방어 / 벤치마크 오염 경로(기준선 붕괴 → 허위 개선 수치) / PHM 도메인 첫 실증 보고

- **MPC와의 연결 및 구별 (CV 선례와의 비교)**
  - **연결:** [P19][P20]이 확립한 "MSE → 조건부 평균 수렴" 메커니즘이 MPC의 이론적 기반
  - **구별:** CV 선례는 blurry 이미지(기능 저하, pred_std > 0) 수준. MPC는 고정 val split + 합성 데이터 균질성 + MIN_EPOCHS 부재 결합 시 완전한 상수 예측(pred_std ≈ 0, 기능 완전 소실)으로 극단화됨

> **Gap 2:** 딥러닝 회귀 모델이 단일 훈련 내에서 trivial solution(상수 예측)으로 즉각 수렴하는 MPC 현상은 Huang(2026)[P21]이 이론화한 다봉 회귀 붕괴의 특수 사례이나, **PHM/RUL 벤치마크에서 이 현상이 프로토콜 결함으로 인해 발생하고 전파되는 메커니즘은 어느 선행 연구에서도 분석·보고된 바 없다.**

---

## 4. 조기 종료와 검증 분할 설계

### 4.1 조기 종료의 고전적 이해

- **[Prechelt 1998][P14]** — 검증 손실 기반 조기 종료의 체계적 기준 수립. 훈련 시간 vs. 일반화 성능의 tradeoff 분석 (2,469회 인용 — 조기 종료 연구의 기초 논문)
  - 핵심 가정: 검증 손실이 진정한 일반화 성능을 반영한다
  - **MPC 시나리오에서 이 가정이 깨짐** — 고정 검증셋의 RUL 분포가 trivial solution을 가능하게 할 때, 검증 손실 감소가 일반화 향상이 아닌 상수 예측으로의 수렴을 의미

- **[Miseta et al. 2023][P12]** — 훈련/검증 손실의 Pearson 상관 기반 새로운 조기 종료 기준(CDSC) 제안. 표준 조기 종료의 한계(분기점 감지 실패) 보완 (60회 인용)
  - 함의: 조기 종료 기준이 여전히 개선 가능한 미해결 문제임을 시사

- **[Mahsereci et al. 2017][P13]** — 검증셋 없이 기울기 통계만으로 조기 종료하는 방법 제안 (113회 인용)
  - 함의: 검증셋 의존성 자체를 제거하는 방향의 연구

### 4.2 검증 분할 편향

- **[Vabalas et al. 2019][P6]** — 소규모 데이터셋에서 고정 K-fold CV가 성능 추정에 강한 편향 유발 (N=1,000까지도 지속). 훈련/테스트 분할 접근이 더 강건함 확인 (1,546회 인용, PLoS ONE)
  - 직접 연결: 고정 val split → 검증셋 RUL 분포 고정 → trivial solution이 val MSE minimum이 됨
  - 한계: 분류 문제 집중. RUL 회귀에서 고정 검증셋이 손실 landscape를 왜곡하는 메커니즘은 미탐구

> **Gap 3:** 고정 검증 분할이 시계열 회귀의 손실 landscape를 trivial solution 방향으로 왜곡하는 메커니즘, 그리고 MIN_EPOCHS 부재가 이를 가속하는 방식은 체계적으로 연구되지 않았다.

---

## 5. 벤치마크 공정성과 재현성

### 5.1 시계열 예측 벤치마크의 편향 문제

- **[Qiu et al. 2024][P9]** — TFB 제안: 데이터 도메인 불충분, 전통 방법 대비 고정관념 편향, 비일관적 평가 파이프라인이 기존 시계열 예측 벤치마크의 3대 문제로 지적 (345회 인용, VLDB)
  - 핵심 발견: 평가 파이프라인의 비일관성이 방법 간 비교를 신뢰하기 어렵게 만듦

- **[Wang et al. 2024][P10]** — Time Series Library(TSLib) 구축: 41개 모델, 30개 데이터셋, 5개 분석 태스크의 공정 벤치마크. "특정 구조의 모델이 특정 태스크에만 적합하다"는 발견 (308회 인용, IEEE TPAMI)

- **[Tan et al. 2025][P11]** — SynTSBench: 합성 데이터로 딥러닝 모델의 시간 패턴 학습 능력 체계 평가. "현재 딥러닝 모델이 모든 시간 특징에서 최적 성능에 근접하지 못함" 확인 (4회 인용, ArXiv)
  - 함의: 합성 데이터 기반 평가가 실제 모델 능력 파악에 유효한 도구임을 지지

### 5.2 PHM 분야 재현성의 현실

- CMAPSS 기반 논문들은 공통 데이터셋을 사용하면서도 전처리, 검증 분할, 조기 종료, 클리핑 값 등 핵심 프로토콜 변수를 상이하게 설정
- **[Li et al. 2018][P16]** 이후 수많은 논문이 이 논문의 결과를 "기준"으로 인용하지만, 동일 프로토콜 재현 여부는 불명확
- **[Elsherif et al. 2025][P5]** FD003 RMSE=13.40은 우리의 수정 기준선 M0(12.97)와 사실상 동등 → 복잡한 모델이 올바른 기준선을 실질적으로 초과하지 못할 수 있음

> **Gap 4:** CMAPSS를 활용한 PHM 연구에서 훈련 프로토콜 비일관성이 기준선 성능에 미치는 영향이 정량적으로 분석된 바 없으며, 특히 "기준선 붕괴"가 개선 주장을 인위적으로 부풀리는 메커니즘은 보고되지 않았다.

### 5.3 Undertuned-Baseline Reproducibility — 타 도메인의 교차 증거

본 연구의 문제 계보에서 가장 직접적인 선행 흐름은 PHM이 아닌 **ML 재현성(reproducibility) 문헌**에 있다. 특히 다음 두 논문이 핵심 계보를 이룬다.

- **[Musgrave et al. (2020)][P23]** — "A Metric Learning Reality Check" *(ECCV 2020, 554 citations)*
  - 거리 학습 분야의 여러 방법을 일관된 프로토콜(동일 백본, 동일 분할, 동일 데이터 증강)로 재평가.
  - **핵심 발견:** 동일 조건 하에서 방법 간 성능 차이가 거의 없음. 기존 "4년간 두 배 이상 향상"이라는 주장은 백본 크기, 분할 방식, 증강 정책의 불일치에서 비롯된 아티팩트.
  - *본 연구와의 연결:* "실험 조건 차이가 성능 차이를 만든다"는 구조가 동일하다. 우리 연구는 이를 RUL 도메인에서 **메커니즘 수준** — val split × early stopping 상호작용이 trivial solution을 고정하는 경로 — 으로 설명한다.

- **[Lučić et al. (2018)][P24]** — "Are GANs Created Equal? A Large-Scale Study" *(NeurIPS 2018, 1,120 citations)*
  - 공정한 하이퍼파라미터 탐색과 통계적 유의성 평가 하에서 다양한 GAN 변형(WGAN, BEGAN, LSGAN 등) 간 성능 차이가 유의미하지 않음을 대규모로 실증.
  - **핵심 발견:** "측정 프로토콜이 결론을 결정한다." 충분한 탐색 예산 하에서 단순 baseline이 복잡한 변형과 동등 수준 도달.
  - *본 연구와의 연결:* "기준선을 제대로 훈련했을 때 신규 방법의 우위가 사라지는" 패턴이 MPC 수정 실험(RMSE 43.23→12.97)과 직접 대응한다.

**재현성 인프라 관련 배경 참고:**

- **[Ferrari Dacrema et al. (2019)][P22]** — "Are We Really Making Much Progress?" *(RecSys 2019, 694 citations)*
  - 18개 신경망 추천 알고리즘을 재현 평가한 결과 대부분이 단순 기준선을 능가하지 못함.
  - **이 논문의 초점:** 코드 미공개, 비일관적 분할, 표준화 부재 등 **재현성 인프라** 문제. "어떤 프로토콜 조건이 기준선을 기능적으로 붕괴시키는가?"라는 메커니즘 분석은 이 논문의 주요 관심사가 아니다.
  - *본 연구와의 거리:* "알고리즘을 재현할 수 없다"는 문제와 "프로토콜 조건이 기능적 붕괴를 유발한다"는 문제는 성격이 다르다. P22는 재현성의 중요성을 체계화한 선행 맥락으로 참조하되, 직접적 계보 논문은 P23/P24다.

**비교 요약:**

| 연구 | 도메인 | 주 관심사 | 기준선 약화 성격 |
|---|---|---|---|
| Ferrari Dacrema 2019 [P22] | 추천 시스템 | 재현성 인프라 (코드·분할·표준화) | 재현 불가능 → 비교 자체가 불가 |
| Musgrave 2020 [P23] | 거리 학습 | 공정 비교 프로토콜 | 조건 불일치 → 주장된 개선 소멸 |
| Lucic 2018 [P24] | GAN | 탐색 예산 공정성 | 최적화 불충분 → 변형 간 차이 소멸 |
| **본 연구 (MPC)** | **PHM/RUL** | **프로토콜 조건 → 기능적 붕괴** | **trivial solution 포획 → 입력 무감각** |

P23/P24가 기술한 현상은 **"최적화 불충분"** — 더 충분히 최적화하면 해소된다. MPC는 그 극단 형태로, 기준선이 단순히 낮은 것이 아니라 **입력과 무관한 상수를 출력하는 기능적 실패 상태**로 포획된다. 이 구별이 PHM/RUL 맥락에서 중요한 이유는, 기능적 붕괴는 추가 튜닝만으로 해소되지 않고 특정 프로토콜 조건들의 복합 작용으로 발생하기 때문이다.

> **Gap 7:** PHM/RUL 도메인에서 거리 학습·GAN 분야가 확인한 "측정 프로토콜이 결론을 결정한다"는 패턴이 존재하는지, 그리고 이것이 단순 최적화 부족을 넘어 **기능적 붕괴(MPC)** 수준으로 극단화되는 조건과 메커니즘이 무엇인지는 아직 연구된 바 없다.

---

## 6. 합성 데이터의 한계와 균질성

### 6.1 합성 데이터 대표성 부족

- **[Chao et al. 2020][P4]** — N-CMAPSS 도입 동기: CMAPSS의 "제한된 대표성"이 실세계 적용의 주요 장벽. 물리-딥러닝 융합으로 대표성 갭 보완 시도
- **[Das et al. 2024][P8]** — 물리 기반 시뮬레이터 합성 데이터로 훈련된 DNN의 불확실성 정량화. 시스템 모델 불확실성이 DNN 예측 성능에 유의미한 영향 (42회 인용, RESS)
  - 공통 관점: 합성 데이터의 "실세계와의 갭"에 주목

### 6.2 균질성의 역설 — 미탐구 영역

- 기존 연구들은 합성 데이터가 "현실을 충분히 반영하지 못함(under-represent)"을 문제로 봄
- **본 연구의 역설적 관점:** 합성 데이터가 "너무 균질하여(over-homogeneous)" 딥러닝 모델의 학습이 trivial solution으로 수렴하게 됨
  - CMAPSS FD003: 단일 운전 조건 + 2개 결함 모드 → 엔진 간 센서 궤적이 거의 동일
  - 결과: MSE landscape에서 평균 예측이 광범위하고 안정적인 local minimum 형성
  - **구조 분해 (Huang 2026과의 긴장 해소):** G5 논거(과도한 균질성)와 Huang(2026)의 논거(다봉성이 collapse 유발)는 서로 다른 레벨을 가리킨다. 동일 결함 모드 내 엔진들은 **과도하게 균질**(intra-mode homogeneity) → trivial solution이 넓고 안정적인 minimum; 결함 모드 간은 **분포가 다봉**(inter-mode bimodality, K=2) → MSE 최솟값이 두 모드의 중간값(~87)으로 당겨짐. "내부 균질 + 외부 다봉"의 복합 구조가 MPC 취약성을 만드는 특수 조건

- **[Tan et al. 2025][P11]** — 합성 데이터 기반 평가의 한계로 "confounding factor"를 지적. 모델이 실제 패턴이 아닌 데이터 생성 프로세스의 인공물을 학습할 수 있음

> **Gap 5:** 합성 RUL 데이터셋의 과도한 균질성이 딥러닝 모델의 trivial solution 수렴을 유발한다는 "역설적 취약성"은 문헌에서 인식되거나 연구된 바 없다.

---

## 7. 연구 갭 종합 및 연구 질문 도출

### 7.1 갭 종합

| # | 갭 | 관련 문헌 | 영향 범위 |
|---|---|---------|---------|
| G1 | CMAPSS 기준선 프로토콜 비표준화 | Li 2018, Elsherif 2025 | 분야 전반의 비교 신뢰성 |
| G2 | MPC가 독립적 현상으로 연구된 바 없음 | Al-Selwi 2023, Dohmatob 2024, Lee 2025 | MPC 현상 자체의 이해 |
| G3 | 고정 val split의 RUL 손실 landscape 왜곡 미탐구 | Prechelt 1998, Vabalas 2019 | 훈련 프로토콜 설계 |
| G4 | 프로토콜 버그가 기준선 붕괴를 유발함이 미보고 | Qiu 2024, Wang 2024 | 벤치마크 재현성 |
| G5 | 합성 데이터 균질성의 역설적 취약성 미인식 | Chao 2020, Das 2024, Tan 2025 | 합성 벤치마크 설계 |
| G6 | Huang(2026)의 "다봉 데이터 + MSE → 붕괴 필연적" 주장이 프로토콜을 통제 변수로 포함 시 어디까지 성립하는지 PHM 도메인에서 미검증 (arXiv, 동료 심사 미완) | Huang 2026 [P21] | 평균 붕괴 이론의 적용 범위 |
| G7 | PHM/RUL 도메인에서 undertuned-baseline reproducibility 문제 및 기능적 붕괴(MPC)로의 극단화 조건 미연구 | Musgrave 2020 [P23], Lucic 2018 [P24] (배경: Ferrari Dacrema 2019 [P22]) | 도메인 간 재현성 문제 연결 |

### 7.2 핵심 연구 공백

> 거리 학습[P23]과 GAN 평가[P24] 분야는 "측정 프로토콜이 결론을 결정한다 — 공정한 조건에서 비교하면 주장된 개선이 사라진다"는 패턴을 독립적으로 확인해왔다. 이 현상은 PHM/RUL 도메인에서 체계적으로 연구된 바 없으며, 더 중요하게는 단순 최적화 부족을 넘어 **기능적 붕괴(MPC)** — 모델이 입력과 무관한 상수를 출력하는 상태 — 로 극단화되는 조건과 메커니즘이 어느 도메인에서도 규명되지 않았다.
>
> 합성 RUL 데이터셋에서 특정 프로토콜 조건들이 복합 작용하여 딥러닝 회귀 모델을 trivial solution으로 즉각 수렴시키는 MPC 현상 — 그 발생 조건, 훈련 동역학, 감지 지표, 최소 비용 예방책 — 은 기존 문헌 어디에서도 직접적으로 규명·분석된 바 없다.
>
> Huang(2026)[P21]은 "다봉 데이터 + MSE → 붕괴 필연적"이라는 이론적 주장을 제기했으나, 이는 arXiv 단계 미검증 논문이며 프로토콜을 통제 변수로 포함한 검증이 없다. 우리의 예비 증거(수정 프로토콜 후 M0 FD003 RMSE=12.97 정상 수렴)는 이 주장의 보편성에 이의를 제기하며, PHM 도메인에서의 경계 조건 규명이 요구된다.

### 7.3 도출된 연구 질문

- **RQ1 (메커니즘):** 고정 val split / MIN_EPOCHS 부재 / 단일 MSE loss 세 조건 중 MPC의 필요충분조건은 무엇이며, 각 조건의 기여도와 상호작용 효과는?
  - *근거: G2, G3*

- **RQ2 (취약성):** 데이터셋의 어떤 속성(운전 조건 수, 결함 모드 수, 엔진 간 센서 분산)이 MPC 취약성을 결정하는가?
  - *근거: G5*

- **RQ3 (감지):** 훈련 완료 직후 추가 실험 없이 MPC를 사후 감지할 수 있는 최소 지표 집합은? (예비 지표 수준; 복수 데이터셋 검증 후 경고 프레임으로 확장 검토)
  - *근거: G1, G4*

- **RQ4 (방어):** 계산 오버헤드 없이 MPC를 완전히 방지하는 단일 최소 조치는?
  - *근거: G3, G4*

- **RQ5 (일반화):** MPC는 FD003에 국한되는가, 아니면 C-MAPSS 전 서브셋 및 다른 아키텍처 계열(MLP, 1D-CNN)에서도 재현되는가?
  - *근거: G5, G7*

- **RQ6 (Huang 경계 조건):** Huang(2026)의 "다봉 데이터 + MSE → 붕괴 필연적" 주장은 프로토콜을 통제 변수로 포함할 때 PHM 도메인에서 어디까지 성립하는가? 프로토콜 수정만으로 붕괴를 방지할 수 있다면, 아키텍처 기반 해법은 해당 케이스에서 과잉 처방인가?
  - *근거: G6*

> **주:** RQ5의 범위는 C-MAPSS FD001–FD004 × 3 architecture 계열로 한정한다. N-CMAPSS 및 bearing 데이터는 Phase 3 범위 외(Decision_log.md D1).

---

## 8. 참고 문헌

[P1] [Model Collapse Demystified: The Case of Regression](https://consensus.app/papers/details/95b1fdfb14f95d9e8d361847227d6ac4/?utm_source=claude_code) — Dohmatob et al., ArXiv, 2024. (88 citations)

[P2] [LSTM Inefficiency in Long-Term Dependencies Regression Problems](https://consensus.app/papers/details/ce8d9e87bdba5132a1ebcf65f76cdd8c/?utm_source=claude_code) — Al-Selwi et al., J. Advanced Research in Applied Sciences and Engineering Technology, 2023. (157 citations)

[P3] [Remaining useful life predictions for turbofan engine degradation using semi-supervised deep architecture](https://consensus.app/papers/details/56b01d0f6d2b5cc6aa452e487a6d910d/?utm_source=claude_code) — Ellefsen et al., RESS, 2019. (466 citations)

[P4] [Fusing Physics-based and Deep Learning Models for Prognostics](https://consensus.app/papers/details/d1bb68cc83795ab7be02db6cea4bece4/?utm_source=claude_code) — Chao et al., RESS, 2020. (403 citations) *N-CMAPSS 원논문*

[P5] [A deep learning-based prognostic approach for predicting turbofan engine degradation and remaining useful life](https://consensus.app/papers/details/d677483454575ba2b93a04b7a54a7184/?utm_source=claude_code) — Elsherif et al., Scientific Reports, 2025. (29 citations) *CAELSTM, FD003 RMSE=13.40*

[P6] [Machine learning algorithm validation with a limited sample size](https://consensus.app/papers/details/0b90b9c26ad75cc9bc3c1200ef8f69cb/?utm_source=claude_code) — Vabalas et al., PLoS ONE, 2019. (1,546 citations)

[P7] [Systematic Bias of Machine Learning Regression Models and Correction](https://consensus.app/papers/details/a9eff9e0664b5cde852e28ed5fbbc822/?utm_source=claude_code) — Lee et al., IEEE TPAMI, 2025. (25 citations)

[P8] [Uncertainty-aware deep learning for monitoring and fault diagnosis from synthetic data](https://consensus.app/papers/details/4025d77718a456b58ef100afbe2bb6c8/?utm_source=claude_code) — Das et al., RESS, 2024. (42 citations)

[P9] [TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods](https://consensus.app/papers/details/87fd121e745e51a08910e9726ce5309f/?utm_source=claude_code) — Qiu et al., VLDB, 2024. (345 citations)

[P10] [Deep Time Series Models: A Comprehensive Survey and Benchmark](https://consensus.app/papers/details/7abc87fe1a3f5cad8e2f6b2b273e2963/?utm_source=claude_code) — Wang et al., IEEE TPAMI, 2024. (308 citations)

[P11] [SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](https://consensus.app/papers/details/9d247201588151028ba3b4a6f9ed0558/?utm_source=claude_code) — Tan et al., ArXiv, 2025. (4 citations)

[P12] [Surpassing early stopping: A novel correlation-based stopping criterion for neural networks](https://consensus.app/papers/details/c7819d242ea65de3925781badf77bc97/?utm_source=claude_code) — Miseta et al., Neurocomputing, 2023. (60 citations)

[P13] [Early Stopping without a Validation Set](https://consensus.app/papers/details/3fb54de549e951d3a2ae19960e61c4e8/?utm_source=claude_code) — Mahsereci et al., ArXiv, 2017. (113 citations)

[P14] [Early Stopping-But When?](https://consensus.app/papers/details/a0e6bcfa620155e6896022463386e75d/?utm_source=claude_code) — Prechelt, 1998. (2,469 citations) *조기 종료 기초 논문*

[P15] [A prognostic driven predictive maintenance framework based on Bayesian deep learning](https://consensus.app/papers/details/f39e662a1ee15425ae2987c24ce6581e/?utm_source=claude_code) — Zhuang et al., RESS, 2023. (173 citations)

[P16] [Remaining useful life estimation in prognostics using deep convolution neural networks](https://consensus.app/papers/details/0e1c60924457539ba1cd7e398592c5a9/?utm_source=claude_code) — Li et al., RESS, 2018. (1,561 citations) *CMAPSS 딥러닝 기초 논문*

[P21] [Resolving Multi-Modal Regression by Difference-Quotient-Based Clustering: Fast Coarse Conditional-Label Assignment](https://arxiv.org/abs/2608.25467) — Huang Weiquan, arXiv:2608.25467, 2026. *MPC와 가장 직접 관련된 최신 논문. "squared loss → conditional mean" 이론 명시, "mean collapse"를 실험 baseline으로 수치화. DQC 클러스터링으로 다봉 회귀 해결 제안*

[P19] [Super-Resolution with Deep Convolutional Sufficient Statistics](https://consensus.app/papers/details/6a4521ec2034590ea422e076dc1f4723/?utm_source=claude_code) — Bruna, Sprechmann & LeCun, CoRR, 2015. (341 citations) *초해상도에서 MSE loss의 "regression-to-the-mean problem" 최초 명명. MPC 이론적 기반*

[P20] [Deep multi-scale video prediction beyond mean square error](https://consensus.app/papers/details/a920f2b22dca53e68136f4fa2d110ad0/?utm_source=claude_code) — Mathieu, Couprie & LeCun, CoRR, 2016. (2,020 citations) *비디오 예측에서 MSE의 "inherently blurry predictions" 실증 — 다봉 조건부 분포 하의 MSE 한계*

[P17] [Prevalence of neural collapse during the terminal phase of deep learning training](https://consensus.app/papers/details/e7806ee3a88f52f08bcfb1278fa2418e/?utm_source=claude_code) — Papyan et al., PNAS, 2020. (1,007 citations) *Neural Collapse 원논문 — 분류 모델의 last-layer feature가 클래스 평균으로 수렴하는 NC1 현상 정의. MPC의 분류 유사 개념*

[P18] [On the Optimization Landscape of Neural Collapse under MSE Loss: Global Optimality with Unconstrained Features](https://consensus.app/papers/details/a6bdb2f6c7b056d3bfb411bcc913c97f/?utm_source=claude_code) — Li et al., ArXiv, 2022. (133 citations) *MSE 손실 하에서 NC 해(클래스 평균 예측)가 global minimizer임을 증명 — 회귀 평균 예측이 MSE 경관의 안정적 minimum을 형성한다는 이론적 근거*

[P22] [Are we really making much progress? A worrying analysis of recent neural recommendation approaches](https://consensus.app/papers/details/48856aa30214567eb3a05edb5381fe5c/?utm_source=claude_code) — Ferrari Dacrema, M., Cremonesi, P., & Jannach, D., *RecSys 2019.* (694 citations) *18개 신경망 추천 알고리즘 재현 평가 — 7개만 재현 가능, 그 중 6개가 단순 heuristic 기준선에 뒤짐. "undertuned baseline" reproducibility 계보의 기초 논문.*

[P23] [A Metric Learning Reality Check](https://consensus.app/papers/details/d7dad3a69d4550da87ee3fe451efd59e/?utm_source=claude_code) — Musgrave, K., Belongie, S., & Lim, S.-N., *ECCV 2020.* (554 citations) *일관된 프로토콜로 metric learning 방법 재평가 — "4년간 두 배 이상 성능 향상" 주장이 실험 조건 불일치 아티팩트임을 실증.*

[P24] [Are GANs Created Equal? A Large-Scale Study](https://consensus.app/papers/details/3c0cdb0064615c3e86fee6a05b174c70/?utm_source=claude_code) — Lučić, M., Kurach, K., Michalski, M., Gelly, S., & Bousquet, O., *NeurIPS 2018.* (1,120 citations) *충분한 하이퍼파라미터 탐색 하에서 GAN 변형 간 성능 차이가 유의미하지 않음. "개선은 알고리즘보다 계산 예산·튜닝에서 비롯된다"는 결론.*

**[판정 방법론]**

[P25] Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37–46. — Cohen's κ 원출처. MPC 판정 일치도(§5.3) 측정에 사용.

[P26] Schulz, K.F., Altman, D.G., Moher, D., & CONSORT Group (2010). CONSORT 2010 Statement: Updated Guidelines for Reporting Parallel Group Randomised Trials. *BMJ*, 340, c332. (2,500+ citations) — 임상시험 blinded outcome assessment 방법론적 표준. §5.3 blind adjudication 원칙의 출처. 이 연구는 임상시험 방법을 MPC 감지 순환 논리 문제에 적용하였음.

**[MPC 감지 지표 원출처]**

[P27] Murphy, A.H. (1988). Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient. *Monthly Weather Review*, 116(12), 2417–2424. — CBR(Constant-Baseline Ratio) 기저 개념의 원출처. MSE 기반 Skill Score(SS = 1 − MSE/MSE_ref)는 CBR과 수학적으로 동등 (CBR² = 1 − SS). 기상예보 분야에서 null predictor 대비 모델 성능 측정의 표준 프레임. PDR과 CBR 자체는 본 연구에서 MPC 목적으로 정의.

[P28] Zeiler, M.D., & Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. In *ECCV 2014*, LNCS 8689, 818–833. (19,000+ citations) — ISS(Input Sensitivity Score)의 입력 교란 기반 민감도 개념의 원출처. 공간적 occlusion sensitivity를 본 연구에서 시계열·연속 회귀 맥락(센서 교란 + 시간축 permutation)으로 일반화. ISS 명칭과 시계열 적용 방식은 본 연구에서 정의.
