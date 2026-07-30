# TII Virtual Submission Review Report
## "From Fleet Normalization to Fault-Mode Gating: A Cross-Dataset Ablation Study of Turbofan RUL Prediction"

> **생성일:** 2026-07-08
> **목적:** IEEE TII(Transactions on Industrial Informatics) 투고 전 가상 동료 심사 — TII 심사 기준 및 독자층을 반영한 3인 리뷰
> **참조 문서:** Pre-Review_Report.md (2026-07-03), Revision_Changelog.md (2026-07-06), manuscript_full_text.md (v2026-07-06)
> **주의:** 이 보고서는 사전 리뷰(2026-07-03)에서 이미 제기·해결된 이슈를 재확인하지 않는다. TII에 특화된 신규 우려 사항과 잔존 미해결 항목에 초점을 둔다.

---

## 0. Executive Summary

| | Reviewer TII-1 | Reviewer TII-2 | Reviewer TII-3 |
|---|---|---|---|
| **역할** | PHM/산업 AI 도메인 전문가 | 딥러닝 방법론 + 벤치마크 전문가 | 산업 배포·시스템 엔지니어링 전문가 |
| **캘리브레이션** | IEEE TII 2022–2026 PHM 게재 논문 + TII Editorial Board 기준 | IEEE TNNLS 2024–2026, ICML 2024, IEEE TII 방법론 논문 | IEEE Trans. Reliability, Industrial IoT 논문, TII 실증 연구 |
| **권고** | **Major Revision** | **Major Revision** | **Major Revision (Strong)** |
| **핵심 우려** | CMAPSS 단일 벤치마크; backbone 구식 | Power-limited null result 언어 추가 보강; 불확실성 정량화 부재 | 산업 배포 가능성 미입증; 계산 복잡도 미제시 |

**세 리뷰어 공통 결론:** 논문은 CMAPSS 커뮤니티에 실질적 기여를 제공하며 방법론적 엄밀성이 높다. 그러나 TII의 "outstanding and original" 기준 + 산업 정보학 실질 기여 요건을 충족하려면 (i) N-CMAPSS 또는 실 산업 데이터로의 외적 타당성 검증, (ii) 배포 가능성(계산 비용, 실시간 추론), (iii) 현대 아키텍처 대비 포지셔닝이 보강되어야 한다.


## 1. TII 투고 수락률 평가

### TII 수락 기준 (공식 + 경험적)

TII 공식 기준: *"outstanding and novel contributions to the field of industrial informatics"*; 순수 방법론·벤치마크 논문은 산업 시스템 맥락과 분리 불가능해야 함. 명시 수락률 **< 20%**.

### 현재 논문 강점-약점 평가

| 기준 | 상태 | TII 기여도 |
|------|------|-----------|
| 방법론적 신규성 (M3 아키텍처) | ✅ 강함 — 65.8% RMSE 개선, 비지도 early-cycle 라우팅 | 높음 |
| 통계 엄밀성 (BH-FDR, 5 seeds) | ✅ CMAPSS 문헌 최상위 수준 | 중간 (TII가 특별 요구하지 않음) |
| 산업 실용성 (3계층 계층구조) | ✅ 실무자 지향적 — 조치 가능한 지침 | 높음 |
| Cross-dataset 검증 | ✅ FD001–FD004 전 데이터셋 포함 | 높음 |
| 실 산업 데이터 검증 | ❌ CMAPSS 시뮬레이션 전용 | **낮음 — TII 취약 포인트** |
| 아키텍처 현대성 | ⚠️ 2018년 stacked LSTM (TII 2026 기준 구식) | 낮음 |
| 계산 복잡도 분석 | ❌ 훈련 시간, 추론 지연 미제시 | **낮음 — TII 필수 기대** |
| 불확실성 정량화 | ❌ 점추정 전용; 예측 구간 없음 | 낮음 |
| N-CMAPSS 또는 추가 벤치마크 | ❌ 미포함 | 낮음 |

### 수락 확률 추정

| 시나리오 | 확률 | 근거 |
|---------|------|------|
| **현재 상태로 투고 시** | **20–30%** | 방법론 강점이 있으나 산업 실증 부재, backbone 구식으로 TII "outstanding" 기준 경계선 |
| **주요 수정 후** (N-CMAPSS 파일럿 + 계산 비용 + 배포 논의 추가) | **45–55%** | Major Revision 결과 심사 가능성 상당히 향상 |
| **전면 강화 후** (실 산업 데이터 + Transformer backbone + UQ 추가) | 65–75% | 범위 밖 — 사실상 다른 논문 |

> **결론:** 현재 상태로 투고하면 **데스크 통과 후 Major Revision** 확률이 가장 높다. 데스크 리젝션 위험은 스코프 미스매치가 아닌 "충분히 outstanding하지 않음" 이유로 약 30% 수준이다. RESS(2순위)로 투고하면 동일 논문으로 더 높은 수락 확률이 예상된다.

---

## 2. Reviewer TII-1 — PHM/산업 AI 도메인 전문가

**캘리브레이션:** IEEE TII 2022–2026 PHM 및 조건 모니터링 게재 논문, TII Editorial Standards for Industrial AI

**권고: Major Revision**

---

### 2-1. 논문 요약 및 종합 평가

본 논문은 NASA CMAPSS 벤치마크에서 RUL 예측 파이프라인의 4개 설계 요소(클리핑, 정규화, 고장 모드 분리, 손실 함수)를 교차 데이터셋 ablation으로 분리·평가한다. 통제된 설계와 BH-FDR 보정 통계 검정, M3 Attention Gate 아키텍처의 65.8% RMSE 개선은 CMAPSS 커뮤니티에 실질적 기여를 제공한다. Pre-Submission 단계의 주요 결함(Ridge vs LSTM 불일치, BH 비교 횟수 오류, p-d 모순)이 개정판에서 해결된 것을 확인했다.

그러나 **TII에서 요구하는 "industrial informatics" 실질 기여**를 현재 형태로 충족하기 어렵다. 다음 두 가지가 결정적이다.

---

### 2-2. 주요 우려 사항

#### [TII-MAJOR-1] CMAPSS 시뮬레이션 전용 — 외적 타당성 미입증

**심각도:** TII 게재 가능성에 가장 큰 단일 장애물.

CMAPSS는 열역학 시뮬레이터(C-MAPSS)가 생성한 합성 데이터다. 논문의 핵심 실용적 주장인 "세 리뷰어가 동의하는 3계층 설계 계층구조"는 CMAPSS에서만 입증됐다. TII 에디터는 2026년 기준 CMAPSS-only 논문에 대해 N-CMAPSS 또는 PRONOSTIA, PHM2012, IMS, 실 터보팬 운영 데이터 중 하나 이상을 요구하는 경향이 강해졌다.

특히 FD003에서 M3가 65.8% RMSE 개선을 달성한다는 주장은 합성 데이터의 두 고장 모드가 명확하게 분리되기 때문에 가능하다. 실제 산업 환경에서 고장 모드는 혼재하고, EDA 단계에서 Silhouette = 0.761처럼 깔끔하게 분리되지 않는다. 이 한계는 논문의 Discussion §V.G에 제한사항으로 언급되어 있지만, TII 독자층은 적어도 하나의 현실적 데이터셋에서의 파일럿 결과를 요구할 것이다.

**요구 사항:**
- 옵션 A (권장): N-CMAPSS DS03 (실 비행 사이클, 건강 파라미터 레이블 포함)에서 M3 파일럿 실험 추가 (5 seeds). N-CMAPSS는 고장 모드 레이블이 있으므로 라우팅 정확도를 직접 측정 가능.
- 옵션 B: 제목에 "CMAPSS Simulation Study" 명시 + Abstract에 "Our findings are specific to the CMAPSS simulation and may not transfer to real fleet sensor data without additional calibration" 추가. 단, 이 경우 데스크 리젝션 위험이 높아진다.

---

#### [TII-MAJOR-2] 아키텍처 현대성 — TII 2026 포지셔닝 문제

현재 논문의 backbone은 2018년 Zheng et al.에서 확립된 stacked LSTM (hidden=64)이다. TII에 2026년 게재된 PHM 논문들은 대부분 Transformer, Mamba, GNN, 또는 하이브리드 아키텍처를 채택한다. Introduction에서 STAR Transformer [45]와 BiLSTM [9]을 최신 SOTA로 언급하면서 의도적으로 2018 LSTM을 사용하는 설계 선택이 논문 안에서 정당화되고 있으나 (ablation 방법론의 필요성), TII 리뷰어들은 이를 "방법론적 기여의 한계"로 볼 것이다.

**현재 논문이 주장하는 바:** "M3 gating 원리의 유효성은 backbone과 독립적"  
**TII 리뷰어가 요구하는 바:** "그렇다면 왜 stronger backbone에서 검증하지 않았는가?"

Pre-Review Reviewer C가 동일 우려를 제기했고, 저자는 "미래 연구"로 대응했다. 이 대응은 RESS/AEI에서는 수용될 수 있지만, TII에서는 "왜 지금 하지 않았는가"로 다시 돌아올 가능성이 높다.

**요구 사항:**
- 최소: Attention-based encoder (예: single-head self-attention over 30 cycles)로 M3를 FD003에서만 재실험하여 "backbone 업그레이드 시에도 M3 gating 원리가 유지됨"을 보일 것. 전체 ablation 재실험은 필요 없음.
- 또는: Discussion §V.H에서 구체적인 예비 결과 없이 "미래 연구"로만 처리하면 리뷰어가 수락하지 않을 것임을 사전 인지.

---

#### [TII-IMPORTANT-3] 산업 배포 프레임워크 — 서술적 수준에 그침

§V.F의 3계층 의사결정 프레임워크는 이 논문의 가장 실용적인 기여이나, TII 기준에서 보면 현재 서술 수준이 "알고리즘 의사결정 플로우차트"에 머물러 있다. TII는 산업 배포 가능성에 대한 정량적 근거를 기대한다.

구체적으로:
1. **Silhouette ≥ 0.5 임계값의 신뢰성:** Silhouette을 0.5 기준으로 M3 사용 여부를 결정하라고 권고하는데, Silhouette이 0.5~0.65인 경우 M3가 얼마나 이득/손실을 주는지 데이터로 보여야 한다. 이 구간에서 M3가 M0보다 나쁠 수 있다.
2. **실시간 배포에서 GMM fitting 비용:** 실제 PHM 시스템은 새 플릿이 도입될 때마다 GMM을 재학습해야 한다. 이 비용(데이터량, 시간)에 대한 discussion이 없다.
3. **False routing의 영향:** GatingNet이 잘못 분류했을 때 RMSE가 어떻게 변하는지 민감도 분석이 없다.

---

### 2-3. 소결

| 항목 | 점검 결과 |
|------|---------|
| Pre-Review 주요 이슈 해결 여부 | ✅ 대부분 해결됨 (A1–A19) |
| TII "outstanding and original" 기준 충족 | ⚠️ 경계선 — M3 novelty로 성립하나 외적 타당성 부재로 약함 |
| TII 산업 informatics 스코프 충족 | ⚠️ 3계층 프레임워크가 강점이나 실증 없이는 약함 |
| 데스크 리젝션 위험 | ~30% |

---

## 3. Reviewer TII-2 — 딥러닝 방법론/벤치마크 전문가

**캘리브레이션:** IEEE TNNLS 2024–2026, ICML 2024 reproducibility track, TII 방법론 논문

**권고: Major Revision**

---

### 3-1. Pre-Review 이후 잔존 방법론 이슈

Pre-Review에서 제기된 주요 통계 이슈들이 Revision_Changelog.md에 의거하여 대부분 해결된 것을 확인한다. 특히:
- Cohen's d와 p_BH 모순 해결 (N=5 동일 기반 확인) ✅
- BH 비교 횟수 84 → 96 수정 ✅
- H7 null result 언어 power-limited로 재구성 ✅
- H2 Ridge regression §III.K 명시 ✅
- H5/H6 FD003 불일치 4개 구조적 차이 설명 ✅

이 수정들은 방법론적 신뢰성을 크게 높였다. 아래는 **개정 후에도 남아 있는 방법론 이슈**다.

---

### 3-2. 신규 방법론 우려 사항

#### [TII-MAJOR-4] N = 5 Seed의 근본적 한계 — 결론 강도 재평가 필요

Pre-Review에서 이미 지적됐고 power-limited 언어로 일부 해결됐지만, TII 리뷰어는 다음을 추가 요구할 것이다.

H6 M3의 핵심 주장인 "65.8% RMSE 개선"은 5 seeds에서 나온 수치다. RMSE = 14.78 ± 1.32는 95% CI로 약 [12.1, 17.4]를 커버한다 (t-분포 기준). 이 구간의 상단(17.4)이 H5/N1의 FD003 낮은 seed 결과(~14)와 근접하므로, 5 seeds는 "guaranteed improvement" 주장에 취약하다.

현실적 요구: H6의 경우 최소 10 seeds (또는 bootstrap)를 추가해 주요 결론의 신뢰 구간을 축소할 것.

---

#### [TII-MAJOR-5] 불확실성 정량화(UQ) 부재 — TII 2026 기대 수준 미달

TII에서 2024–2026년 게재된 PHM 논문의 상당수가 prediction interval 또는 calibrated uncertainty를 보고한다. 이 논문은 점추정(mean ± std)만 제공하며, 이는:
1. 실제 PHM 시스템에서 의사결정에 불충분하다 (예: "언제 엔진을 교체할지"는 점추정이 아닌 신뢰 구간이 필요)
2. [43] RUL-QMoE가 이미 확률적 예측을 제공하고 있어, M3가 UQ를 제공하지 않으면 비교 열위에 놓임

**최소 요구:** Conformal Prediction 또는 MC Dropout으로 M3의 90% 예측 구간을 FD003에서 보고. 구현 비용 낮음 (기존 M3 위에 MC Dropout 적용 시 추가 코드 ~20줄).

---

#### [TII-IMPORTANT-6] H7 클리핑 × 손실함수 교호작용 해석의 인과성 주장

§V.E의 주장: "Pinball이 clip=None 하에서 3,000× NASA Score를 개선하는 것은 분위수 회귀 특성 때문"

이 해석은 그럴듯하지만 실험적으로 검증되지 않았다. Pinball이 L6(τ=0.25)이므로 MSE보다 보수적(early) 예측을 유발한다는 것은 알려져 있으나, CMAPSS 특정 맥락에서 이 메커니즘이 실제로 작동하는지는 확인되지 않았다. 예측 분포의 직접 시각화(예: predicted vs true scatter plot for L1 MSE vs L6 Pinball at clip=None)가 없으면 이는 post-hoc rationalization이다.

---

#### [TII-MINOR-7] 재현 가능성 패키지 부재

TII는 2025년부터 code availability를 적극 권장한다. 현재 논문은 §V.G(예상)에서 언급만 할 뿐 실제 GitHub 링크나 Zenodo DOI가 없다. 이는 리뷰어가 독립 재현을 시도할 수 없게 만든다. 투고 전 코드/데이터 공개 또는 "코드는 acceptance 후 공개 예정" 명시 필요.

---

### 3-3. 강점 (유지된 기여)

1. **Cross-dataset ablation 설계:** CMAPSS 4개 서브데이터셋 동시 평가 + BH-FDR은 이 분야에서 방법론 표준을 한 단계 올린다.
2. **Inter-seed variance 진단 원리:** "높은 시드 간 분산은 잠재적 범주 구조의 진단 신호"는 재현 가능한 일반화 가능 원리다.
3. **M3 K-sensitivity 분석:** K=5로도 충분하다는 결과는 배포 실용성에 중요한 기여다.
4. **H2–H7 통합 분석:** 4개 설계 요소를 동일 프레임에서 비교한 논문은 이전에 없다.

---

## 4. Reviewer TII-3 — 산업 배포/시스템 엔지니어링 전문가

**캘리브레이션:** IEEE Trans. Reliability (PHM 실증), Industrial IoT 시스템, TII 산업 검증 논문

**권고: Major Revision (Strong)**

---

### 4-1. 배경 평가

본 리뷰어의 관점: TII의 핵심 독자층은 항공우주/제조 산업 시스템 엔지니어와 PHM 배포 담당자다. 이들이 논문에서 요구하는 정보가 현재 논문에 체계적으로 누락되어 있다. 아래는 TII가 게재하는 산업 PHM 논문들이 반드시 포함하는 항목들과의 비교다.

---

### 4-2. 주요 우려 사항

#### [TII-MAJOR-8] 계산 복잡도 완전 부재 — TII 실용 논문 필수 항목

TII에 2023–2026년 게재된 PHM 논문 중 사용 가능한 데이터를 조회한 결과, 거의 모든 논문이 다음을 보고한다: 훈련 시간(GPU 및 epoch당), 추론 지연(ms), 파라미터 수, FLOPs.

현재 논문은 이 중 어느 것도 없다. M3의 GatingNet은 경량이지만, 두 개의 full-capacity LSTM 브랜치를 병렬 실행하므로 M0 대비 ~2× 파라미터와 ~2× 추론 비용이 발생한다. 이 비용이 "65.8% RMSE 개선"의 trade-off로 적합한지 판단 불가.

**구체적 요구 사항:**
- Table: M0–M3 각 모델의 파라미터 수, FLOPs, 훈련 시간(GPU, epoch당), 추론 시간(ms/engine)
- GPU 사양 및 배치 크기 명시
- FD003에서 "K=5로도 충분" 주장 → K에 따른 추론 지연 변화도 보고

---

#### [TII-MAJOR-9] 실시간 배포 가능성 분석 부재

§V.F의 3계층 프레임워크는 "turbofan 엔진 커미셔닝 시점에 고장 모드 라우팅 수행"을 제안한다. 이는 매력적인 주장이지만 TII 독자 기준에서 다음 질문들이 미답이다:

1. **스트리밍 추론:** 실제 항공 PHM 시스템은 비행마다 실시간으로 센서 데이터를 처리한다. GatingNet이 K=5 사이클에서 라우팅을 완료하면 이후 남은 비행 이력에 대해 RUL 추론은 선택된 브랜치만 사용하는가? 고장 모드가 비행 중 변화(FD004의 6개 운전 조건 전환)할 경우 대응 방안은?

2. **모델 드리프트 대응:** 엔진이 노후화되면 센서 특성이 변한다. GatingNet이 과거 학습 데이터와 다른 분포의 초기 사이클을 받을 경우의 robust성이 검증되지 않았다.

3. **Fleet 규모 확장성:** 현재 실험은 최대 260개 엔진. 실제 항공사는 수천 개의 엔진을 운영한다. K-means residualization과 GMM fitting의 확장성 논의가 없다.

---

#### [TII-IMPORTANT-10] 유지보수 의사결정 비용 분석 누락

TII §V.F의 "실용적 배포 프레임워크"가 실질적 의미를 갖으려면 RUL 예측 오차가 유지보수 일정에 미치는 비용을 정량화해야 한다. 현재 논문은 "1 RMSE cycle ≈ 1 maintenance scheduling uncertainty cycle"이라고 §III.J에서 언급하지만, M3가 M0 대비 RMSE를 28.45 cycles 개선할 때의 실제 운영 비용 절감 규모에 대한 논의가 없다.

TII 독자들이 기대하는 형태:
- "FD003-class 터보팬의 평균 수명 206 cycles, cycle당 유지보수 비용 $X 기준, M0 → M3 전환 시 연간 절감 추정: $Y/engine"
- 이 계산이 정확할 필요는 없지만, order-of-magnitude 추정을 통해 "industrial informatics" 실질 기여임을 입증해야 한다.

---

#### [TII-IMPORTANT-11] Online Learning / 적응형 모델 논의 부재

TII PHM 논문에서 2024년 이후 증가하는 요구: 초기 배포 후 실제 운영 데이터가 누적될 때 모델이 적응하는 방법. 현재 논문의 M3는 오프라인(batch) 학습만 지원하며, online adaptation이나 transfer learning에 대한 논의가 전무하다. 이것이 "미래 연구"로 명시되어야 TII 리뷰어가 현재 범위의 한계를 이해하고 수용할 수 있다.

---

### 4-3. 강점

1. **3계층 계층구조의 실용성:** PHM 실무자가 즉시 사용할 수 있는 의사결정 플로우차트 형태로 제시한 것은 TII 독자층에 직접 가치 있다.
2. **K=5 조기 라우팅:** 커미셔닝 시점(비행 5회 후)에 고장 모드 식별이 가능하다는 결과는 산업 PHM 배포에 실질적 함의를 갖는다.
3. **Cluster collapse 방지 메커니즘:** M1의 247:1 붕괴 문제를 M3가 구조적으로 회피하는 방식은 실제 배포 환경에서 중요한 안정성 기여다.

---

## 5. 통합 이슈 매트릭스 (TII 특화)

이전 Pre-Review에서 이미 다룬 이슈는 포함하지 않음. TII 투고에 특화된 신규 또는 잔존 이슈만 수록.

| # | 이슈 | 리뷰어 | 우선순위 |
|---|------|--------|--------|
| T1 | N-CMAPSS 또는 실 데이터 파일럿 실험 | TII-1 | 🔴 Critical |
| T2 | 계산 복잡도 표 (파라미터, FLOPs, 추론 시간) | TII-3 | 🔴 Critical |
| T3 | 현대 backbone에서 M3 원리 검증 (최소 1 dataset) | TII-1 | 🔴 Critical |
| T4 | 불확실성 정량화 (MC Dropout 또는 Conformal Prediction) | TII-2 | 🟡 Important |
| T5 | N=5 → 10 seeds 로 확장 (H6 핵심 실험) | TII-2 | 🟡 Important |
| T6 | False routing 시 성능 민감도 분석 | TII-1 | 🟡 Important |
| T7 | 산업 배포 비용 편익 추정 (order-of-magnitude) | TII-3 | 🟡 Important |
| T8 | 스트리밍/온라인 배포 가능성 논의 추가 | TII-3 | 🟡 Important |
| T9 | GitHub/Zenodo 코드 가용성 링크 | TII-2 | 🟢 Minor |
| T10 | Silhouette 0.5 임계값의 신뢰 구간 — M3 이익이 0.5–0.65 구간에서도 유지되는지 | TII-1 | 🟢 Minor |
| T11 | Pinball 3,000× 개선 메커니즘 시각화 검증 | TII-2 | 🟢 Minor |
| T12 | Fleet 규모 확장성 논의 (수천 엔진) | TII-3 | 🟢 Minor |

---

## 6. Pre-Review 대비 상태 비교

사전 리뷰(2026-07-03)에서 제기된 30개 항목과 현재 상태를 비교한다.

| 분류 | Pre-Review 항목 수 | 해결 완료 | 부분 해결 | 미해결 |
|------|-----------------|---------|---------|------|
| 🔴 Critical | 7 (A1–A7) | **7/7** ✅ | 0 | 0 |
| 🟡 Important | 12 (A8–A19) | 10 | 1 (A8 센서 이름) | 1 (A17 N-CMAPSS) |
| 🟢 Minor | 11 (A20–A30) | 7 | 2 | 2 (A20 제목, A23 violin plot) |

**핵심 발견:** Pre-Review Critical 이슈 7개가 모두 해결된 것은 긍정적이다. 단, TII 전용 이슈(T1–T12) 12개가 새로 제기되며, 이 중 🔴 Critical 3개(T1 실증 데이터, T2 계산 비용, T3 현대 backbone)는 TII 수락에 결정적이다.

---

## 7. Reviewer TII-1/2/3 공통 강점 인정

1. **통제된 4-way ablation 설계:** 동일 논문에서 4개 설계 요소를 독립 변수로 분리한 연구는 CMAPSS 문헌에서 선례가 없다.
2. **M3의 실용적 혁신성:** Early-cycle 비지도 라우팅 + test-time collapse 면역성은 산업 배포에 직접 연결되는 구조적 기여다.
3. **솔직한 Null Result:** H5(정규화)와 H7(손실 함수)의 기각 결과를 은폐하지 않고 메커니즘 해석까지 제공한 것은 PHM 분야의 출판 편향에 대한 가치 있는 기여다.
4. **Inter-seed variance 진단 원리:** "재현 불가능한 분산 = 잠재 범주 구조의 신호"는 모든 벤치마크 연구자에게 전이 가능한 방법론 원리다.
5. **BH-FDR 다중 비교 보정:** CMAPSS 문헌 최초 수준의 통계 엄밀성.

---

## 8. 결론 및 투고 전략 권고

### 수락 경로 시나리오

| 경로 | 요구 작업 | 예상 소요 시간 | TII 수락 확률 |
|------|---------|------------|------------|
| **경로 A: TII 즉시 투고** | 없음 (현재 상태) | — | 20–30% |
| **경로 B: TII 강화 후 투고** | T1 (N-CMAPSS) + T2 (계산 비용) + T4 (UQ) | 2–4주 | 45–55% |
| **경로 C: RESS 우선 투고** | 없음 (현재 상태로 RESS 투고) | — | 45–60% |
| **경로 D: RESS 투고 후 TII 재도전** | RESS 리뷰어 피드백 반영 + 위 강화 | 6–12개월 후 | 65–70% |

### 전략 권고

**단기 (즉시 투고 결정 시):**
- 경로 C를 권고. RESS는 레퍼런스 4편 기반 최적 저널이며 현재 논문이 수락 가능 수준이다.
- TII 투고는 경로 B 강화 작업 후가 적합하다. 가장 중요한 단일 작업은 **T1 (N-CMAPSS 파일럿)**이다.

**중기 (M3 확장 연구 병행 가능 시):**
- N-CMAPSS DS03에서 M3 파일럿 (5 seeds, 2–3주 작업) → TII Major Revision 생존 확률 크게 향상
- MC Dropout으로 불확실성 구간 추가 (1주 추가 작업)
- 위 두 항목 완료 후 TII 직접 투고 = 경로 B, 수락 확률 45–55%

---

*이 보고서는 공개된 TII 투고 기준, 2022–2026 TII PHM 게재 논문 패턴, 그리고 Pre-Review Report (2026-07-03) 및 Revision_Changelog (2026-07-06)를 종합하여 작성됐다. 실제 TII 리뷰어의 의견과 다를 수 있으며, 특히 TII-MAJOR-1 (실증 데이터 부재)에 대한 편집부의 실제 입장은 현 EiC의 성향에 따라 달라질 수 있다.*
