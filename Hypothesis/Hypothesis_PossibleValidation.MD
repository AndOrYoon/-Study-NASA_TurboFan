# Hypothesis Possible Validation Report
## NASA CMAPSS TurboFan — 가설 학술적 검증 및 Research Gap 분석

> **기준일:** 2026-07-02  
> **최종 수정:** 2026-07-30 (H6 — M3 Attention-Gate 조기 라우팅 전제 배치 문헌 검토 추가)  
> **검색 범위:** Web Search (2024~2026) + **Consensus MCP peer-reviewed 논문 검색 (2026-07-02)**  
> **원본 가설 출처:** `Dataset/EDA_Report.MD` Section 7

---

## 개요 — 가설 상태 요약

> **검증 대상 확정 (2026-07-02):** H2, H5, H6, H7 — 리뷰어 검토(`Hypothesis_critics.md`) 기반 선정  
> H1(표준 전처리 고정), H3(H5 보조 요인), H4(제외) 는 독립 검증 불필요

| # | 가설 | 학계 관심도 | Research Gap | 검증 난이도 | **검증 여부** |
|---|------|------------|--------------|------------|------------|
| H1 | 상수 센서 제거 | ★★★★★ | 낮음 (확립) | 쉬움 | ⬛ 전처리 고정 |
| **H2** | **RUL 클리핑 임계값 최적화** | ★★★★★ | 중간 (adaptive 미탐구) | 중간 | ✅ **검증 대상** |
| H3 | 다중 조건 클러스터 정규화 | ★★★★☆ | 중간 (방법론 경쟁 중) | 중간 | ⬛ H5 보조 요인 |
| H4 | PCA 다중공선성 축약 | ★★★☆☆ | 중간 (attention으로 대체 추세) | 중간 | ❌ 제외 |
| **H5** | **Per-unit 정규화** | ★★★☆☆ | **높음 (미탐구)** | 높음 | ✅ **검증 대상** |
| **H6** | **복합 고장 모드 멀티-브랜치** | ★★★★☆ | **높음 (EDA 전제 확인)** | 높음 | ✅ **검증 대상** |
| **H7** | **비대칭 손실 함수** | ★★★★★ | 중간 (cross-dataset 갭) | 중간 | ✅ **검증 대상** |

---

## H1 — 상수 센서 제거

### 학계 현황
CMAPSS 연구에서 사실상 **표준 전처리 단계**로 확립되어 있다.  
대다수 논문이 21개 센서 중 분산이 낮은 센서를 제거하고 **14개 유효 센서 + op1·op2 = 16개 피처** 조합을 사용한다.

> "Some column features do not provide useful information for RUL estimation, so researchers typically select 14 of the 21 sensors data and two of the three working conditions, totalling 16 features."  
> — Analytics Vidhya, 2024

- **MDPI Applied Sciences (2025):** "Linear Methods for Predictive Maintenance" 에서 상수 센서 제거를 필수 전처리로 명시  
- **arXiv 2605.02507 (2025, TCN 기반):** preprocessing pipeline에서 가장 먼저 실행하는 step으로 기술  

### Research Gap 평가
**갭 없음.** 이 가설은 이미 학계 컨센서스이다.  
단, **어떤 기준(분산 임계값, 상관계수, 도메인 지식)으로 제거할지** 에 대한 체계적 비교는 드물어, 임계값 sensitivity analysis는 소논문 기여가 될 수 있다.

### 검증 전략
- Baseline(21개) vs 14개 vs domain-knowledge 선택 세트의 RMSE 비교 ablation
- 데이터셋별(FD001~FD004) 제거 효과 차이 정량화

### 참고 문헌
- [Linear Methods for Predictive Maintenance — MDPI 2025](https://www.mdpi.com/2076-3417/15/18/9945)
- [TCN Preprocessing-Driven RUL — arXiv 2025](https://arxiv.org/pdf/2605.02507)
- [CMAPSS Failure Classification — Analytics Vidhya 2024](https://www.analyticsvidhya.com/blog/2024/07/cmapss-jet-engine-failure-classification-based-on-sensor-data/)

---

## H2 — RUL 125 Clipping (Piece-wise Linear)

### 학계 현황
**학계 표준으로 완전히 수렴**되어 있다. 2024~2026년 발표된 거의 모든 CMAPSS 논문이 이 전략을 별도 설명 없이 채택한다.

> "Clipping the maximum RUL to 125 cycles reduces label variance in early lifecycle stages and facilitates comparison with existing methods."  
> — Uncertainty-Aware DL Framework, arXiv 2511.19124 (2025)

- **Nature Scientific Reports (2025):** CAELSTM 모델에서 piecewise linear degradation + clipping 적용
- **IEEE DataPort (2025):** Multi-Objective Hyperparameter Optimization 논문에서 125 clipping을 기본 전제로 설정
- **PMC / Machines (2025):** Autoencoder-LSTM 연구에서 window size 실험의 출발점으로 클리핑 사용

### Research Gap 평가
**갭 없음.** 클리핑 값 125가 왜 최적인지에 대한 **이론적 근거는 여전히 부재**하지만, 이 값의 실증적 우월성이 반복 검증되어 있다.  
**잠재 갭:** 클리핑 값 자체를 데이터셋별로 최적화하는 adaptive clipping threshold 연구는 찾기 어렵다.

### 검증 전략
- 클리핑 값(75 / 100 / 125 / **130** / no-clip)에 따른 FD001~FD004 NASA Score 비교 (130: arXiv 2604.13459 실증 사용값)
- 데이터셋별 최적 클리핑 값이 다른지 검증

### 참고 문헌
- [Uncertainty-Aware DL for Turbofan — arXiv 2511.19124](https://arxiv.org/html/2511.19124)
- [TCN Preprocessing Approach — arXiv 2605.02507](https://arxiv.org/html/2605.02507v1)
- [Autoencoder-LSTM Window/Batch Effects — Machines 2025](https://doi.org/10.3390/machines14020135)
- [Deep Learning Prognostic Turbofan — Nature 2025](https://www.nature.com/articles/s41598-025-09155-z)

---

## H3 — 다중 조건 클러스터 기반 정규화 (FD002, FD004)

### 학계 현황
운전 조건 정규화의 **필요성은 컨센서스**이나, 정규화 방법론에서 경쟁이 진행 중이다.

**현재 경쟁 중인 접근법:**

| 방법 | 설명 | 대표 연구 |
|------|------|-----------|
| 클러스터 기반 min-max | K-means로 6개 조건 클러스터 식별 후 클러스터 내 정규화 | ResearchGate 2023 figure |
| MLP Regression 정규화 | op1~op3를 입력으로 기대 센서값 예측 → 잔차를 피처로 사용 | 2024 연구들 |
| Min-Max (전체) | 클러스터 무시, 전체 train 범위로 정규화 | 구형 논문 |

> "A more sophisticated normalization approach explicitly removes operational condition effects while preserving degradation-related patterns."  
> — Temporal Convolutional Memory Networks, arXiv 1810.05644

> "Recent work adopted regression-based normalization procedures for FD002 and FD004, training MLP regression models that predict expected sensor values given operational condition variables."

### Consensus MCP 검색 결과 (2026-07-02)

Consensus 검색(`regression normalization operating condition CMAPSS`)으로 두 편의 중요 논문 확인:

**[MOC-based Normalization — Appl. Soft Comput. 2022, 20 citations]**  
"Two density-based clustering algorithms are integrated to be a new classifier for operational conditions clustering and identification in an unsupervised manner... MOC-based Normalization recalibrates the upward or downward abrupt jumps of sensor readings at the operational conditions change-points."  
→ **클러스터 기반 정규화를 peer-reviewed로 구현한 가장 직접적 선행 연구.** GRU backbone 기준 top-3 대비 ~10% RMSE 개선.

**[DFormer — Measurement 2024, 32 citations]**  
"Previous methods accomplish the normalization operation to attenuate the inherent non-stationarity of raw series for better predictability, which can be less instructive for bursty RUL prediction tasks."  
→ 정규화 자체를 회피하고 차분(differencing) + Transformer로 비정상성을 직접 학습. 새로운 대안 접근법으로 H3 실험 설계에 4번째 비교군으로 추가 가능.

**갭 업데이트:** MOC (2022)가 클러스터 방법을 구현했으나, **MLP regression 정규화와의 직접 ablation 비교** 및 **DFormer식 정규화 회피 접근과의 3자 비교**는 여전히 미탐구.

### Research Gap 평가
**갭 있음 (중간 수준, 재확인).** MOC-based Normalization (2022)이 클러스터 정규화의 효과를 입증했으나, 세 가지(클러스터 min-max / MLP regression / 정규화 회피 differencing) 방법의 **공정한 ablation 비교**가 단일 논문 내에서 수행된 연구는 없다.  
특히 **FD002와 FD004에서 각각 최적 방법이 다른지**는 여전히 미탐구 영역이다.

### 검증 전략
- 4가지 정규화 방법 × FD002·FD004 × 동일 모델(LSTM) 조합으로 RMSE/NASA Score 비교
  1. Global min-max (baseline)
  2. Cluster min-max (MOC 방식)
  3. MLP regression 정규화
  4. Differencing 기반 비정규화 (DFormer 방식)
- 가설: MLP regression 또는 DFormer 방식이 FD004(복합 고장 + 다중 조건)에서 더 유리할 것

### 참고 문헌
- [MOC-based Normalization for Time-varying Conditions — Appl. Soft Comput. 2022](https://consensus.app/papers/details/c50add3976c55431a872966d2a1caf13/?utm_source=claude_code)
- [DFormer: Non-stationary Feature Learning — Measurement 2024](https://consensus.app/papers/details/c059e5edfaf25750b8b7608f826a171e/?utm_source=claude_code)
- [Operating Condition Clusters FD002/FD004 — ResearchGate](https://www.researchgate.net/figure/Clusters-of-operating-conditions-for-datasets-FD002-and-FD004_fig4_373263679)
- [Temporal Convolutional Memory Networks — arXiv](https://arxiv.org/pdf/1810.05644)
- [Spatio-Temporal Attention GNN — arXiv 2401.15964](https://arxiv.org/pdf/2401.15964)
- [Unsupervised LSTM Autoencoders Early Fault Detection — arXiv 2601.10269](https://arxiv.org/pdf/2601.10269)

---

## H4 — PCA 다중공선성 축약

### 학계 현황
PCA+딥러닝 파이프라인은 **2020~2022년에 활발히 탐구**되었으나, 현재는 Attention 기반 feature weighting으로 연구 트렌드가 이동하고 있다.

> "Researchers proposed a PCA-LSTM method for predicting RUL of aero-engines that uses PCA to reduce dimensionality and improve prediction performance."  
> — PMC 2020 (Sensors MDPI)

> "Dimensionality reduction plays a critical role in RUL prediction and explainability, as it helps mitigate multicollinearity and improves the applicability of permutation-based XAI methods."  
> — Nature Scientific Reports 2025

- **2025 해석 가능 앙상블 연구 (PMC):** PCA를 XGBoost·Random Forest와 결합 → 해석 가능성 확보에 PCA가 유리
- **2026 Wiley — Explainable RUL with Divisive Feature Clustering:** PCA 대신 hierarchical feature clustering 제안

### Research Gap 평가
**갭 있음 (중간, 트렌드 전환 구간).** PCA 자체보다 **"왜 특정 주성분이 RUL과 연결되는가"를 물리 도메인에서 해석하는 연구**가 부족하다.  
또한 PCA vs Attention 기반 선택의 성능 비교를 **동일 backbone 모델**로 수행한 논문은 드물다.

### 검증 전략
- PCA(n_components 탐색) vs 상관계수 기반 선택 vs Attention 기반 선택을 동일 LSTM backbone에서 비교
- 주성분의 물리적 의미 해석 (부하율, 압력비 등과의 매핑)

### 참고 문헌
- [PCA + Explainable DNN — PMC Sensors 2020](https://www.mdpi.com/1424-8220/20/22/6626)
- [Interpretable Ensemble RUL — PMC 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12615660/)
- [Explainable Divisive Feature Clustering — Wiley 2026](https://onlinelibrary.wiley.com/doi/10.1002/asmb.70084)
- [Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-09155-z)

---

## H5 — Per-unit 정규화 vs Fleet 정규화

### 학계 현황
CMAPSS 연구 대부분은 **train 셋 전체 기준의 min-max 또는 standard scaling (fleet-level normalization)** 을 사용한다.  
엔진 개별 초기값 기준 정규화(per-unit normalization)를 **명시적으로 비교·연구한 논문은 찾기 어렵다.**

최근 논문들에서 언급된 수준:
- "standard scaling leads to faster convergence and slightly higher performance compared to min-max scaling" — fleet level 비교에 그침
- N-CMAPSS(더 큰 데이터셋) 관련 2025 연구에서는 residual feature(모델 예측값과 실측값의 차이)를 열화 신호로 사용 — 이것이 per-unit 아이디어와 가장 유사

### Consensus MCP 검색 결과 (2026-07-02)

Consensus 검색(`per-engine normalization per-unit baseline CMAPSS remaining useful life`)으로 20개 논문 스캔:

- **CNN-LSTM-Attention (Int. J. Comput. Intell. Syst. 2024, 69 citations):** fleet 기준 min-max 정규화 적용, per-unit 언급 없음
- **Deep LSTM C-MAPSS (IEEE Access 2022, 75 citations):** 개선된 piecewise degradation 모델 제안, 정규화는 fleet-level
- **MMEPTMIXER (J. Comput. Des. Eng. 2026, 1 citation):** multi-scale TCN 기반, 정규화 방법론 논의 없음

**결론:** peer-reviewed 논문 전체에서 per-unit vs fleet normalization을 **명시적으로 비교한 논문 0편 확인.** 웹 검색 결과와 동일하게 Research Gap 재확인.

### Research Gap 평가
**갭 높음 (peer-reviewed 비교 연구 부재 — Consensus MCP 검색으로 재확인).** Per-unit normalization이 실제로 fleet normalization보다 우수한지를 **공정한 ablation으로 검증한 CMAPSS 논문을 peer-reviewed 데이터베이스에서도 찾을 수 없었다.**  
특히 수명이 짧은 엔진(< 150 cycles)에서의 이점은 미탐구 상태이다.

**이 가설은 현재 시점에서 가장 직접적인 Research Gap에 해당한다.**

### 검증 전략
1. Fleet min-max / Fleet standard / Per-unit (초기 n cycles 기준) 세 가지 정규화를 동일 모델로 비교
2. 수명 구간별(< 150, 150~250, > 250 cycles) 서브그룹 성능 분석
3. Test 셋 Ground-truth RUL을 기준으로 짧은 수명 엔진에서의 예측 오차 집중 분석

### 참고 문헌
- [Multi-Scale Dilated Fusion Attention — MDPI 2025](https://doi.org/10.3390/app15179813)
- [N-CMAPSS DL Customized Penalty — SJSU 2025](https://scholarworks.sjsu.edu/etd_projects/1523/)
- [CNN-LSTM-Attention Aero Engine — Springer 2024](https://link.springer.com/article/10.1007/s44196-024-00639-w)
- [Survey on GNN for RUL — arXiv 2409.19629](https://arxiv.org/pdf/2409.19629)

---

## H6 — 복합 고장 모드 멀티-브랜치 모델 (FD003, FD004)

### 학계 현황
FD003/FD004의 **이중 고장 모드 문제가 단일 모델에 어렵다는 사실은 널리 관찰**되고 있으나, 이를 명시적으로 해결하는 연구는 초기 단계이다.

> "FD003 (single operating condition, two fault modes) shows the worst model performance across implementations, suggesting that fault mode diversity is more challenging than operational variability."

- **arXiv 2602.19263 (2025):** "Prognostics of Multisensor Systems with Unknown and Unlabeled Failure Modes via Bayesian Nonparametric Process Mixtures" — CMAPSS에서 고장 모드를 잠재 변수로 추정하는 Bayesian 접근 제안 → 이 분야에서 가장 직접적인 관련 논문
- **MDPI Aerospace (2024):** "FC-AMSLSTM for HPC Degradation" — HPC 단일 고장 모드에 특화된 구조 제안 (Fan degradation은 별도 미연구)
- **arXiv 2511.19124 (2025):** 불확실성 추정에서 FD003·FD004의 aleatoric uncertainty가 FD001·FD002보다 높음을 보고

### Consensus MCP 검색 결과 (2026-07-02)

Consensus 검색(`mixture model fault mode unknown label CMAPSS FD003 FD004 remaining useful life`)으로 20개 논문 스캔:

**[Multi-task Adaptive Mixture-of-Experts — Expert Syst. Appl. 2025, 27 citations]**  
"Multi-task model of adaptive multi-scale feature fusion and adaptive mixture-of-experts for equipment remaining useful life prediction and fault diagnosis"  
→ **현재까지 발견된 가장 관련성 높은 peer-reviewed 논문.** RUL 예측 + 고장 진단을 동시에 수행하는 mixture-of-experts 구조 채택. 단, **fault mode 레이블을 학습 시 사용**하므로 H6의 핵심 도전(미레이블 고장 모드 분리)을 직접 해결하지는 않음.

**[CAELSTM — Scientific Reports 2025, 18 citations]**  
FD001·FD003에서 CAE + Attention-LSTM 조합. FD003의 이중 고장 모드를 별도 처리 없이 단일 모델로 처리 — RMSE 13.40 달성.  
→ FD003 성능은 개선됐지만 고장 모드 구조적 분리는 없음.

**갭 업데이트:** Mixture-of-experts 구조(2025)가 등장했으나, **고장 모드 레이블 없이(unsupervised)** 잠재 고장 모드를 분리하는 딥러닝 구조는 peer-reviewed 논문에서 미발표 상태 재확인.

### Research Gap 평가
**갭 높음 (미탐구 핵심 구조 문제 — Consensus MCP로 재확인).** HPC + Fan 이중 고장 모드를 **레이블 없이 명시적으로 분리·조건부 처리하는 mixture model / multi-branch 구조**는 CMAPSS에서 체계적으로 연구된 사례가 없다.  
- Bayesian nonparametric 접근(arXiv 2602.19263, 2025): unsupervised에 가장 근접하지만 arXiv 단계  
- Multi-task MoE (Expert Syst. Appl. 2025): fault mode 레이블 사용, peer-reviewed 게재됨  
- **딥러닝 기반 unsupervised multi-branch 구조: 미발표 상태**

**이 가설은 H5와 함께 학술 기여 가능성이 가장 높은 영역이다.**

---

### M3 Attention-Gate 조기 라우팅 전제 — 배치 문헌 검토 (2026-07-30)

M3의 핵심 전제("초기 K=10 사이클만으로 고장 모드 라우팅 가능")에 배치되거나 제한 조건을 부과하는 선행 연구를 Consensus MCP 검색으로 확인함.

**배치 논점 1 — 전체 열화 궤적이 있어야 고장 모드 분리 가능:**

[Degradation Modeling and Prognostic Analysis Under Unknown Failure Modes (Fu et al., 2024, IEEE TASE, 6 citations)](https://consensus.app/papers/details/b1482d5fe9165695a53dbe92e89af4ee/?utm_source=claude_code)은 UMAP으로 **전체 열화 궤적**을 저차원에 투영하여 클러스터링한 뒤 고장 모드를 식별한다. 테스트 엔진에 대해서도 고장 모드 레이블과 RUL을 **동시 예측(joint prediction)** 방식으로 처리한다. M3가 초기 10 사이클만으로 라우팅을 결정하는 방식과 패러다임이 근본적으로 다르며, "고장 모드 판별에 전체 궤적이 필요하다"는 암묵적 전제를 내포한다.

[Joint Learning of Failure Mode Recognition and Prognostics (Wang et al., 2023, IEEE TASE, 19 citations)](https://consensus.app/papers/details/09338fbac2d755b3a6c11bf41874e788/?utm_source=claude_code)은 열화 메커니즘 기반 피처를 전체 시계열에서 추출하여 고장 모드 인식과 RUL 예측을 공동 학습(joint learning)으로 수행한다. "서로 다른 고장 모드에서 센서 신호는 서로 다른 열화 **패턴**을 나타낸다"고 명시하여, 고장 모드 판별이 초기 절대값이 아닌 시간 경과에 따른 패턴 변화에 의존할 수 있음을 시사한다.

> **핵심 취약점:** M3 EDA에서 Silhouette=0.634가 나온 근거는 후반부 20% 평균과 열화 기울기 — 즉 **전체 수명 데이터** — 로 계산된 것이다. GatingNet이 초기 10 사이클만 보고 동등한 판별력을 낼 수 있는지는 실험적으로 확인된 사항이지, 이론적으로 보장된 것이 아니다. Fu et al.[2024]과 Wang et al.[2023]의 전체 궤적 의존적 접근은 이 불일치를 직접 지적한다.

**배치 논점 2 — 조기 이상 탐지 자체가 어렵다:**

[Causal Inference-Based Fault Diagnosis for Aero-Engine (Dong et al., 2025, ICEIOM)](https://consensus.app/papers/details/7326b8750b3158278d462204fd073809/?utm_source=claude_code)은 N-CMAPSS(실제 비행 조건 기반)에서 **"복잡한 열역학적 결합과 비선형 열화 패턴으로 인해 조기 이상 식별이 어렵다"** 고 명시한다. M3가 의존하는 "초기 사이클의 고장 모드 신호"가 실제 엔진에서는 노이즈에 묻혀 판별이 불가능할 수 있음을 N-CMAPSS 실험으로 뒷받침한다.

[Early Fault Detection on CMAPSS with Unsupervised LSTM Autoencoders (Sánchez et al., 2026, arXiv)](https://consensus.app/papers/details/ed28c601ef495d1eb78655f346b885bd/?utm_source=claude_code)은 재구성 오류가 임계값을 **지속적으로** 초과할 때 비로소 이상 경보를 발생시키는 방식으로, 단 10 사이클로 즉각 판별하는 M3와 대조된다. 신뢰할 수 있는 이상 탐지에는 **반복적 패턴 축적**이 필요하다는 입장이다.

**배치 논점 3 — 순수 비지도 + 조기 사이클 접근의 한계:**

[Deep Learning-Based Sensor Selection for FM Recognition and Prognostics (Wang et al., 2025, IEEE TASE, 14 citations)](https://consensus.app/papers/details/d2cbf153c5bd509aaa71c531e095e1a8/?utm_source=claude_code)은 일부 유닛의 고장 모드 레이블(semi-supervised)과 도메인 적응을 결합해야 정확한 고장 모드 인식이 가능하다고 주장한다. 순수 비지도 + 초기 10 사이클만으로 라우팅하는 M3보다 훨씬 많은 정보를 요구하는 방식이다.

**종합 평가:**

| 배치 논점 | 강도 | CMAPSS 내 방어 논리 |
|---------|------|-------------------|
| 전체 궤적 없이는 판별력 부족 (Fu 2024, Wang 2023) | **중간-높음** | CMAPSS FD003에서 고장 모드가 초기 절대값(s15 BPR, |Δz|=31.3)에 반영됨 — EDA로 확인 |
| 조기 이상 탐지 자체가 어렵다 (Dong 2025) | **중간** | N-CMAPSS 기반 결과 — CMAPSS 합성 데이터에서는 완화될 수 있음 |
| 레이블 또는 충분한 관측 필요 (Wang 2025) | **약함** | 방법론 차이이지 직접 반증 아님; M3는 비지도를 명시적 차별점으로 주장 |

**논문 대응 전략:** Limitation 절에 "M3의 조기 라우팅 성공은 CMAPSS 고장 모드가 열화 패턴이 아닌 초기 기준값 차이(s15 BPR)에 의해 구분된다는 CMAPSS-특수적 성질에 의존한다. 고장 모드가 초기 사이클에서 판별 불가능한 실제 엔진(Dong et al. 2025; N-CMAPSS)이나 전체 궤적 기반 접근이 필수적인 환경(Fu et al. 2024; Wang et al. 2023)에서는 K를 늘리거나 2단계 라우팅 전략이 필요할 수 있다"를 명시하여 선제 대응.

### 검증 전략
1. 고장 모드 레이블 없이 GMM 또는 EM으로 잠재 고장 모드 클러스터 추정
2. 추정된 클러스터별 전용 LSTM 브랜치 또는 soft gating network 구성
3. FD003/FD004에서 단일 모델 및 Multi-task MoE 대비 RMSE·NASA Score 개선 검증
4. 추정된 고장 모드 클러스터가 실제 HPC/Fan 분리와 일치하는지 사후 분석

### 참고 문헌
- [Multi-task Adaptive Mixture-of-Experts — Expert Syst. Appl. 2025](https://consensus.app/papers/details/37634668bbcc57da862da2431a4fc990/?utm_source=claude_code)
- [Bayesian Nonparametric Failure Mode Mixture — arXiv 2602.19263](https://arxiv.org/pdf/2602.19263)
- [CAELSTM Turbofan RUL — Scientific Reports 2025](https://consensus.app/papers/details/d677483454575ba2b93a04b7a54a7184/?utm_source=claude_code)
- [FC-AMSLSTM HPC Degradation — MDPI Aerospace 2024](https://www.mdpi.com/2226-4310/11/4/293)
- [Uncertainty-Aware DL Turbofan — arXiv 2511.19124](https://arxiv.org/html/2511.19124v1)
- [Early Fault Detection CMAPSS LSTM AE — arXiv 2601.10269](https://arxiv.org/pdf/2601.10269)
- **[배치 문헌] [Degradation Modeling Under Unknown Failure Modes — IEEE TASE 2024](https://consensus.app/papers/details/b1482d5fe9165695a53dbe92e89af4ee/?utm_source=claude_code)** — 전체 궤적 기반 UMAP 클러스터링으로 고장 모드 분리; 조기 라우팅 대비 패러다임
- **[배치 문헌] [Joint Learning of FM Recognition and Prognostics — IEEE TASE 2023](https://consensus.app/papers/details/09338fbac2d755b3a6c11bf41874e788/?utm_source=claude_code)** — 열화 패턴 기반 공동 학습; 초기 절대값 아닌 시계열 패턴 의존
- **[배치 문헌] [Causal Inference-Based Fault Diagnosis for Aero-Engine — ICEIOM 2025](https://consensus.app/papers/details/7326b8750b3158278d462204fd073809/?utm_source=claude_code)** — N-CMAPSS에서 조기 이상 탐지 어려움 실증
- **[배치 문헌] [Deep Learning Sensor Selection for FM Recognition — IEEE TASE 2025](https://consensus.app/papers/details/d2cbf153c5bd509aaa71c531e095e1a8/?utm_source=claude_code)** — semi-supervised + 도메인 적응 필요성 주장

---

## H7 — 비대칭/가중치 손실 함수 (NASA Score 최적화)

### 학계 현황
**가장 활발히 연구 중인 영역** 중 하나이다. 2020~2026년에 걸쳐 지속적인 연구가 이루어지고 있다.

> "The scoring is asymmetric around the true time of failure such that late predictions are more heavily penalized than early predictions."

> "Dynamically-weighted loss functions achieve significant improvement for remaining useful life prediction and fault detection over non-weighted loss predictions."  
> — Deep Learning with Dynamically Weighted Loss, PMC/MDPI Sensors 2020

**최근 주요 발전:**

| 논문 | 방법 | 연도 |
|------|------|------|
| MDPI Sensors (PMC) | Dynamically weighted loss + focal loss 비교, 4개 DL 아키텍처 실험 | 2020 |
| arXiv 2604.13459 | Asymmetric-Loss CNN-BiLSTM-Attention + Failure Heatmap | 2025 |
| SJSU 2025 | Customized Penalty loss for N-CMAPSS | 2025 |
| IEEE DataPort 2025 | Multi-Objective Hyperparameter Optimization (timeliness 조절) | 2025 |

### Consensus MCP 검색 결과 (2026-07-02)

Consensus 검색(`time-weighted loss asymmetric NASA score CMAPSS RUL clipping interaction`)으로 20개 논문 스캔:

**[Asymmetric-Loss CNN-BiLSTM-Attention — arXiv 2026, 0 citations]** (기존 참고 문헌 arXiv 2604.13459 동일)  
"Standard symmetric loss functions inadequately penalize the safety-critical error of over-estimating residual life... piecewise-linear RUL labeling capped at **130 cycles**"  
→ 표준 125-cycle 대신 130-cycle cap 사용 — adaptive clipping threshold 연구 가능성 재확인. FD001만 평가, cross-dataset 없음.

**[RUL Dynamics via Time Transformation — Reliab. Eng. Syst. Saf. 2025, 3 citations]**  
"Asymmetrical confidence intervals... Higher slopes indicate faster degradation, leading to narrower confidence intervals"  
→ 불확실성 기반 접근, 손실 함수 설계가 아닌 분석적 방법론. H7과 직접 관련 없음.

**갭 업데이트:** 비대칭 손실 연구가 활발하나 모두 FD001 단독 평가. **클리핑 임계값(125 vs 130) × 손실 함수 상호작용**을 cross-dataset으로 분석한 논문 여전히 0편.

### Research Gap 평가
**갭 있음 (중간, 활발히 좁혀지는 중 — Consensus MCP 재확인).** 비대칭 손실 자체는 잘 연구되어 있으나:  
- **수명 구간별 시간 가중치(time-weighted)와 NASA 비대칭 패널티를 결합한 통합 손실 함수**는 미탐구  
- **클리핑 임계값(75/100/125/130)과 손실 함수 간 상호작용** 비교 연구 없음  
- **FD001~FD004별로 최적 손실 가중치가 다른지**에 대한 cross-dataset 분석 부재 (현존 논문 대부분 FD001만 평가)

### 검증 전략
1. MSE / NASA Score Loss / Dynamically Weighted / Focal / Time-Weighted 손실 함수 비교
2. RUL 클리핑 임계값(100 / 125 / 130 / 미적용) × 손실 함수 조합의 교차 실험
3. 후반부(life > 80%) 샘플 가중치를 λ 파라미터로 조절하며 NASA Score 변화 추적
4. FD001~FD004 전체에서 동일 실험 반복 → 최적 조합의 데이터셋 의존성 분석

### 참고 문헌
- [Asymmetric-Loss CNN-BiLSTM-Attention — arXiv 2604.13459 / 2026](https://consensus.app/papers/details/e958d628dd8f5b52a4840ca793c2cb3a/?utm_source=claude_code)
- [RUL Dynamics via Time Transformation — Reliab. Eng. Syst. Saf. 2025](https://consensus.app/papers/details/ebd3bbe9dd965881bafd24fa765086ba/?utm_source=claude_code)
- [Dynamically Weighted Loss DL — PMC/MDPI Sensors 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7038523/)
- [N-CMAPSS Customized Penalty — SJSU 2025](https://scholarworks.sjsu.edu/etd_projects/1523/)
- [Multi-Objective Hyperparameter Calibration — IEEE DataPort 2025](https://ieee-dataport.org/documents/calibrating-prediction-timeliness-through-multi-objective-hyperparameter-optimization)

---

## 종합 — Research Gap 우선순위 및 학술 기여 전략

### Tier 1 — 논문 기여 가능성 높음 (갭 명확)

| 가설 | 갭 요약 | 추천 전략 |
|------|---------|-----------|
| **H5** (Per-unit 정규화) | 명시적 ablation 연구 전무 | FD001~FD004 전체에 걸친 비교 + 짧은 수명 엔진 서브그룹 분석 |
| **H6** (복합 고장 모드) | 딥러닝 기반 multi-branch 미존재 | FD003/FD004 특화 mixture/gating 구조 제안 |

### Tier 2 — 기존 연구 보완 (갭 중간)

| 가설 | 갭 요약 | 추천 전략 |
|------|---------|-----------|
| **H3** (조건 정규화) | 클러스터 vs MLP regression 직접 비교 부재 | ablation study + FD002·FD004 교차 검증 |
| **H7** (가중 손실) | 클리핑×손실 상호작용, 데이터셋별 최적화 미탐구 | 통합 손실 함수 설계 + λ 탐색 실험 |
| **H4** (PCA 축약) | Attention 대비 공정 비교 없음 | 동일 backbone, 3가지 차원 축소 방법 비교 |

### Tier 3 — 확립된 기법 (재확인 실험 수준)

| 가설 | 상태 |
|------|------|
| **H1** (상수 센서 제거) | 표준 전처리, 재현 실험 가치만 존재 |
| **H2** (RUL 125 클리핑) | 학계 컨센서스, Adaptive threshold 연구만 신규 기여 가능 |

---

## Consensus MCP 검색 완료 — 결과 요약 (2026-07-02)

5개 우선순위 쿼리 × Consensus MCP(200M+ peer-reviewed 논문) 검색 완료.

| 쿼리 | 핵심 발견 | 갭 상태 변화 |
|------|---------|------------|
| H5 per-unit 정규화 | 20편 스캔, per-unit vs fleet 직접 비교 0편 | **갭 유지 (HIGH)** |
| H6 mixture/fault mode | Multi-task MoE (Expert Syst. Appl. 2025, 27 citations) 발견 — 레이블 있는 방식 | **갭 유지 (HIGH), 선행연구 1편 추가** |
| H3 조건 정규화 | MOC-based Normalization (Appl. Soft Comput. 2022) + DFormer (Measurement 2024) 발견 | **갭 중간 유지, 선행연구 구체화** |
| H7 비대칭 손실 | 130-cycle cap 사용 사례 확인, cross-dataset 비교 0편 | **갭 중간 유지** |
| 전반 survey | 체계적 open-problems survey 논문 미발견 | — |

### 최종 Research Gap 우선순위 (변경 없음)

| 순위 | 가설 | 근거 |
|------|------|------|
| **1위** | **H5** (Per-unit 정규화) | peer-reviewed 비교 연구 전무 (Consensus 재확인) |
| **2위** | **H6** (복합 고장 모드) | unsupervised multi-branch 미발표 (레이블 방식만 존재) |
| **3위** | **H3** (조건 정규화) | MOC 선행연구 존재, 3-way ablation 부재 |
| **4위** | **H7** (비대칭 손실) | cross-dataset 분석 및 clipping×loss 상호작용 부재 |
| **5위** | **H4** (PCA) | Attention 대비 공정 비교 부재 |

---

*Validated via WebSearch + Consensus MCP (2026-07-02) · Cross-referenced against EDA_Report.MD hypotheses*  
*2026-07-30: H6 M3 조기 라우팅 배치 문헌 검토 추가 (Fu 2024, Wang 2023, Dong 2025, Wang 2025)*
