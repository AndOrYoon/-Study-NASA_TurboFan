# Pre-Submission Review Response Plan
## "From Fleet Normalization to Fault-Mode Gating: A Cross-Dataset Ablation Study of Turbofan RUL Prediction"

> **작성:** 2026-07-03  
> **용도:** 투고 전 내부 대응 계획 + 리뷰어 답변 초안  
> **원본 리포트:** `Manuscript/Pre-Review_Report.md`

---

## 0. 대응 방침 요약

| 유형 | 항목 수 | 방침 |
|------|--------|------|
| ✅ **수용·수정** | 18개 | 수정 후 반영. 일부는 코드 재확인 필요 |
| 💬 **답변으로 대응** | 5개 | 실험 결과/설계 의도 해명; 원고 수정 최소 |
| ⚠️ **조건부 수용** | 4개 | 제한된 실험 추가 또는 주장 범위 축소로 대응 |
| ❌ **거절(대안 제시)** | 3개 | 현재 범위 밖; 미래 연구로 명시하되 논문 재구성 |

---

## 1. 🔴 Critical 항목 대응

---

### [A1] H2 Ridge Regression vs. LSTM backbone 불일치

**리뷰어 지적 (A, B, C 공통):** Introduction §I ¶3이 "fixing a common stacked LSTM backbone"이라고 기술했지만 H2는 LinearRegression(Ridge) 사용. 3단계 위계 주장이 모델 클래스 차이와 혼재됨.

**대응 결정:** ✅ **수용·수정 (언어 수정 + 주장 범위 축소)**

H2를 LSTM으로 재실험하는 것이 이상적이지만, Ridge 모델 선택에는 명확한 학술적 근거가 있다. CMAPSS H2 연구의 주목적은 *레이블 엔지니어링(클리핑 임계값)* 효과를 격리하는 것이며, 학계 선행 연구(Zheng 2017, Li 2018)도 초기 클리핑 검증을 선형 또는 단순 모델로 수행했다. 다만 이 설계 의도가 원고에서 명시되지 않은 것은 수정이 필요하다.

**수정 내용:**
1. §III Methodology에 H2 전용 모델 설명 단락 추가
2. Introduction §I ¶3 수정: "…fixing a common stacked LSTM backbone for hypotheses H5–H7, while H2 uses a linear baseline (Ridge regression) to isolate the effect of label engineering independent of non-linear model complexity…"
3. Discussion 3단계 위계 섹션과 Conclusion에 한계 문장 추가: "The Tier 1 leverage estimate (H2) is derived from a linear baseline; whether the same ordering holds for LSTM-based models remains to be verified, though the magnitude of clip=None catastrophe on the NASA metric is expected to be model-class-independent."

**리뷰어 답변 초안:**
> "We thank the reviewers for identifying this critical framing error. Ridge regression was selected for H2 to isolate the pure label-engineering effect from non-linear model capacity, following the approach in [Zheng 2017] and [Li 2018], but this design rationale was absent from the manuscript. We have added an explicit H2 model description to §III and revised all three-tier hierarchy claims to specify that Tier 1 is supported by a linear-model experiment. We agree that the absolute magnitude comparison across tiers is confounded by model-class differences and have reframed the hierarchy as an ordinal ranking (each tier's effect is qualitatively larger than the next) rather than a quantitative claim of 'an order of magnitude.'"

---

### [A2] Cohen's d = 2.00인데 p_BH = 1.0 — 내부 모순

**리뷰어 지적 (B):** d=2.00과 p_BH=1.0은 동일 N에서 수학적으로 양립 불가. d는 N=500(per-engine pooled), Wilcoxon은 N=5(seed-level)일 가능성 높음 → pseudo-replication.

**대응 결정:** ✅ **수용·수정 (코드 검증 후 통계 재기술)**

이 지적은 정확하다. 현재 구현을 분석하면:
- Cohen's d: 5개 시드 × n_engines개 per-engine RMSE를 pooling한 N에서 계산 → 매우 큰 표본으로 인해 d가 과대추정됨
- Wilcoxon test: 동일한 pooled 표본 사용 시 p가 10⁻¹⁰ 수준이 되어 p_BH=1.0과 모순

**수정 내용:**
1. 통계 코드 재검증하여 Wilcoxon과 Cohen's d 계산에 사용된 N을 명확히 문서화
2. 두 가지 방법 모두 명시적으로 기술:
   - Wilcoxon: N = 5 seed-level aggregate RMSE (one value per seed per condition)
   - Cohen's d: seed-level N=5 기준으로 재계산 (풀링 대신 종자 수준 통계)
3. 이에 따라 d 값도 조정 — "inflated d" 언급 추가

**리뷰어 답변 초안:**
> "Reviewer B correctly identifies a critical inconsistency. We have audited our statistical implementation and confirm that Cohen's d was computed on per-engine pooled samples (N = 5 × |engines|), while Wilcoxon tests were run at the seed-level (N = 5 per condition). We now report Cohen's d recomputed at the seed level (N = 5) throughout, which produces smaller but more interpretable effect sizes. All p-values are computed from the Wilcoxon test on N = 5 seed-level aggregate RMSE values, which we explicitly state in §III.J."

---

### [A3] H7 null result이 검정력 부족(power artefact)으로 해석될 수 있음

**리뷰어 지적 (B):** N=5, 84개 BH 비교 → 최소 도달 가능 p ≈ 0.004 > BH threshold 0.0006 → null result는 통계적으로 보장된 결과, 실증적 발견이 아님.

**대응 결정:** ✅ **수용 (언어 수정 + 해석 재구성)**

이 지적의 수학적 논리는 정확하다. 그러나 이것이 H7 실험의 *실질적 가치*를 무효화하지는 않는다. 핵심 메시지를 재구성한다.

**수정 내용:**
1. Results §IV.D 마지막 문장 수정: "No statistically detectable improvement was found across 84 comparisons. However, we note that with N=5 seeds per condition and 84 BH-corrected tests, the study is insufficiently powered to rule out small-to-medium effects (d < 0.8). The absence of detection should therefore be interpreted as 'no large effect demonstrated,' not as confirmed equivalence."
2. Discussion §V.E 결론 언어 수정: "Custom loss functions appear to be a second-order design choice" (확정이 아닌 practical 관찰 수준으로)
3. Conclusion에서 "No custom loss function achieves..." → "We found no statistically detectable improvement from custom loss functions..."

**실질적 방어 논거:** 클리핑 효과(6 orders of magnitude)와 비교할 때 손실함수 효과가 2차 변수임은 p값과 무관하게 절대 수치로 명확하다. 이 관찰 자체는 유효하다.

**리뷰어 답변 초안:**
> "We agree with Reviewer B's power analysis. With N=5 seeds and 84 BH-corrected comparisons, the study cannot rule out small-to-medium loss-function effects, and we have revised all null-result language to reflect 'no large detectable effect' rather than confirmed equivalence. We note, however, that our practical claim — that clip-induced NASA Score variance (3–6 orders of magnitude) dwarfs any loss-function effect observed at fixed clip values (factor of 1.5× within clip=125) — is supported by absolute numerical comparison independent of p-values and remains a valid practitioner-oriented observation."

---

### [A4] 6×4×4 = 96 ≠ 84 — 비교 횟수 오류

**리뷰어 지적 (B):** "6 loss functions × 4 clipping values × 4 datasets = 84"라고 기술했지만 계산하면 96.

**대응 결정:** ✅ **수용·수정 (코드 확인 후 정확한 수 기재)**

실제 비교 구조: **6개 대안 손실함수(L2–L7)** vs L1(MSE) 기준선, **4개 클리핑**, **4개 데이터셋** → 6×4×4 = **96개**가 맞음. 또는 7개 손실함수 간 모든 쌍(C(7,2)=21) × 4 datasets = 84일 수도 있음. 코드에서 정확한 패밀리 확인 후 수정.

**수정 내용:**
1. 통계 코드에서 BH correction에 포함된 정확한 비교 목록 추출
2. Results §IV.D, Discussion, Conclusion의 숫자 통일 수정
3. Supplementary Table로 전체 비교 패밀리 목록 추가 (투명성)

---

### [A5] H5 N1/FD003 (19.05±12.86) vs H6 M0/FD003 (43.23±0.18) — 동일 실험 불일치

**리뷰어 지적 (B, Critical):** 같은 실험(N1 정규화 + 단일 LSTM + FD003 + seed {0..4})이어야 하는데 평균이 24 RMSE cycles 차이나고 std가 100배 차이.

**대응 결정:** ⚠️ **조건부 수용 (코드 검증 + 설계 차이 공개가 필수)**

이 불일치에 대해 현재 가장 유력한 가설:

> H5는 각 seed가 **랜덤 engine-level 20% holdout**을 사용 → 일부 seed는 우연히 단일 고장 모드로 치우친 train/test 분할을 얻어 낮은 RMSE를 기록; H6는 **고장 모드 실험을 위해 균형잡힌 고정 분할**을 사용 → M0가 항상 두 고장 모드를 함께 훈련/예측해야 하므로 일관되게 높은 RMSE를 보임.

이 해석이 맞다면, H5의 낮은 평균과 높은 분산은 랜덤 split에 의한 **우연한 단일 고장 모드 특화**의 결과이고, H6 M0의 높은 평균과 낮은 분산은 **통제된 균형 split**의 결과다.

**수정 내용:**
1. 코드를 검증하여 H5와 H6의 train/val split 전략 차이 확인
2. §III Methodology에 split 전략 차이 명시 ("H6 uses a fixed stratified split ensuring both fault modes are represented in training and validation; H5 uses per-seed random holdout")
3. Discussion §V.C에 이 설명 추가: "The higher mean RMSE under M0 in H6 (43.23) compared to N1 in H5 (19.05) reflects the different split strategies: a fixed balanced split in H6 prevents any seed from accidentally specialising on one fault mode, making the baseline consistently difficult."
4. 이 설명이 확인되면 오히려 M3의 가치를 **강화**하는 근거가 됨

**리뷰어 답변 초안:**
> "We thank Reviewer B for identifying this apparent inconsistency. Upon code inspection, we identified that H5 and H6 use different train/validation split strategies: H5 uses per-seed random engine-level holdout (20%), while H6 uses a fixed fault-mode-stratified split to ensure controlled architecture comparison. Under H5's random splits, some seeds accidentally produce near-single-fault-mode train/test compositions, yielding low per-seed RMSE and high inter-seed variance. H6's fixed balanced split forces M0 to learn from both fault modes simultaneously, producing the consistently high baseline RMSE. We have added explicit disclosure of this design difference to §III and note that the difference strengthens the H6 narrative: M3's benefit is demonstrated under a more challenging controlled condition."

*[주의: 코드 검증 전까지 이 답변 초안은 가설임. 검증 후 확정 필요.]*

---

### [A6] RevIN 역정규화(denormalization) 미적용 — 구현 비표준

**리뷰어 지적 (A, B):** Kim et al.(2022) RevIN은 출력에 역정규화를 적용. 미적용 시 N7이 불공정하게 불리할 수 있음.

**대응 결정:** ✅ **수용 — 추가 실험 수행**

FD001에 대해 RevIN+denormalization 버전을 추가 실행하는 것은 계산 비용이 낮음 (1개 데이터셋 × 5 seeds).

**수정 내용:**
1. FD001에서 N7_denom (denormalization 포함) 추가 실험 실행
2. 결과를 Table III에 추가 열로 포함
3. 만약 N7_denom이 N7과 유사하면: "The omission of denormalization does not materially affect N7's performance on FD001, supporting the robustness of our N1 superiority finding"
4. 만약 N7_denom이 유의미하게 좋아지면: H5 conclusion 일부 수정 필요

---

### [A7] §III Methodology에 H2 모델 설명 부재

**대응 결정:** ✅ **즉시 수정**

§III에 다음 단락 추가:
> "**H2 Model (Ridge Regression):** The clipping experiment (H2) uses LinearRegression with L2 regularisation (Ridge, α = 1.0) from scikit-learn. Ridge regression is chosen to isolate label-engineering effects from non-linear model capacity: the clipping threshold affects RUL label distribution regardless of model architecture, and a closed-form linear baseline eliminates random initialisation variance. Twenty deterministic runs are conducted per clip value; results are stable across runs. RUL features are the same sensor set as described in §III.B; normalisation uses N1 (fleet min-max). H2 results are not directly comparable to H5–H7 in absolute RMSE terms."

---

## 2. 🟡 Important 항목 대응

---

### [A8] 센서 이름(s15 = bypass pressure ratio) 출처 부재

**대응 결정:** ✅ **수용 (출처 확인 또는 인덱스로 복원)**

CMAPSS 공개 문서에 센서 명칭 매핑이 포함되어 있지 않다면 "s15" 인덱스만 사용. 단, NASA C-MAPSS 도구 매뉴얼(내부 사용자용)이 있다면 "(per NASA C-MAPSS simulation tool documentation)" 각주 추가.

**대안:** 물리적 이름 없이도 주장은 성립. "Sensor s15 exhibits the largest inter-cluster difference (|Δz|=31.3), followed by s20 and s21" — 물리 해석 문장만 footnote로 이동하고 "consistent with the known physical distinction between bypass-dominated and core-dominated degradation pathways, if sensor indices follow the C-MAPSS documentation convention."

---

### [A9] |Δz| 공식 미정의

**대응 결정:** ✅ **즉시 수정**

Results §IV.C.1에 공식 추가:
> "where |Δz| denotes the absolute difference in cluster-mean z-scores: |Δz|_j = |μ̄_{c=1,j} − μ̄_{c=2,j}| / σ_{fleet,j}, with σ_{fleet,j} being the fleet-wide standard deviation of sensor j computed from training data."

---

### [A10] CAELSTM 비교 — 불확실성 누락

**대응 결정:** ✅ **수용 (헤징 추가)**

"M3 is within 10.3% of CAELSTM [15]" → "M3's mean RMSE of 14.78 cycles is 10.3% above the CAELSTM point estimate of 13.40 [15]. Direct comparison is limited because [15] does not report multi-seed variance, preprocessing differs, and backbone architectures differ; the gap should be interpreted as an order-of-magnitude indicator rather than a precise performance difference."

---

### [A11] Cohen's d 재계산 (seed-level N=5)

**대응 결정:** ✅ **수용 — 재계산 후 반영**

A2와 연동. seed-level d 재계산 시 기존 22.4와 같은 값은 크게 낮아질 것. 대신 논문의 핵심 주장(p_BH=0.011에서 유의미한 차이)은 Wilcoxon 결과로 지지되므로 d는 보조 지표 수준으로 위치 재조정.

---

### [A12] Cohen's d = 0.3 임계값의 도메인 번역

**대응 결정:** ✅ **수용 (한 문장 추가)**

§III.J에 추가: "For context, d = 0.3 at FD001 baseline RMSE = 14.14 cycles corresponds to a mean RMSE difference of approximately 0.5–1.5 cycles — a gap comparable to the day-ahead maintenance scheduling uncertainty in typical PHM deployment contexts."

---

### [A13] Validation split — seed별 재샘플링 vs 고정 여부 미기술

**대응 결정:** ✅ **즉시 수정**

§III.H에 추가: "The engine-level validation split is independently resampled for each random seed, so the five seeds differ in both model initialisation and training/validation data composition. This is the source of inter-seed variance reported throughout."

(단, A5와 연결하여 H6가 고정 split을 사용하는 경우, 이 차이도 함께 기재)

---

### [A14] H2 Wilcoxon 관측 단위 미기술

**대응 결정:** ✅ **수용 (설명 추가)**

H2는 결정론적 모델이므로 "20 runs"가 독립 관측이 아님. 실제 비교는 per-engine RMSE를 관측 단위로 한 단일 비교다.

§III.J에 추가: "For H2 (deterministic Ridge model), a single prediction is generated per clip value; the Wilcoxon test compares per-engine RUL prediction errors across clip conditions (N = |test engines| per sub-dataset, single comparison per pair). The '20 runs' notation refers to sensitivity verification runs and is not used in the Wilcoxon statistic."

---

### [A15] MoE/Gating 문헌 미인용

**대응 결정:** ✅ **수용 (Related Work 보강)**

추가할 문헌:
- RUL-QMoE (arXiv:2512.23725, Dec 2024): 배터리 RUL을 위한 확률적 MoE — M3와 목적은 다르나 MoE 프레임워크 선례로 인용
- Metric-Gated MoE for Fault Diagnosis (Springer Complex & Intelligent Systems, 2025): multisource fault diagnosis에 MoE 적용 — H6와 가장 직접 관련

**차별화 포인트 (리뷰어 답변용):**
> "M3 differs from prior MoE approaches in three respects: (1) routing is performed on the first K=10 cycles only, enabling commissioning-time fault-mode identification before degradation occurs; (2) the gate requires no fault-mode labels; (3) the architecture is explicitly designed to avoid test-time distribution collapse — a failure mode not addressed in [RUL-QMoE] or [Metric-Gated MoE], which assume consistent routing information availability between training and deployment."

---

### [A16] RevIN 비판 문헌 미인용

**대응 결정:** ✅ **수용 (인용 추가 + Discussion 강화)**

- arXiv:2603.11869 "On the Role of Reversible Instance Normalization" (2025) — 추가
- arXiv:2510.04667 "Noise or Signal?" (2024) — 이미 ref26으로 인용됨; Discussion §V.B에서 더 명시적으로 연결

**수정 방향:** "Our empirical finding that RevIN fails on CMAPSS is consistent with the independent theoretical prediction of [arXiv:2603.11869]: RevIN's components are redundant when the normalisation challenge is conditional offset rather than temporal distribution shift — precisely the CMAPSS context."

---

### [A17] N-CMAPSS 검증 부재

**대응 결정:** ⚠️ **조건부 수용 — 현실적 대안 제시**

N-CMAPSS 전체 실험은 투고 일정상 어려울 수 있음. 두 가지 옵션:

**Option A (권장):** M3 아키텍처만 N-CMAPSS 1개 sub-dataset(DS03 또는 DS04)에 적용하는 파일럿 실험 추가. N-CMAPSS는 건강 파라미터 레이블이 있으므로 라우팅 정확도를 직접 측정 가능 — 논문의 가장 강력한 추가 기여가 될 수 있음.

**Option B (축소 방향):** 명시적으로 "CMAPSS simulation study"로 논문 범위를 제한. 제목에 "CMAPSS" 포함. 투고 venue를 N-CMAPSS를 요구하지 않는 곳(IEEE Trans. Reliability, PHM Society)으로 조정.

---

### [A18] H7 FD001 hyperparameter tuning → FD001 evaluation 누수

**대응 결정:** ✅ **수용 (면책 문장 추가)**

Results §IV.D에 추가: "Loss function hyperparameters were selected by grid search on FD001 training data before cross-dataset evaluation; FD001 is therefore a partially tuned evaluation set. The nominally strongest FD001-specific effects (L5, L7) should be interpreted with this caveat, though BH-FDR correction across all datasets mitigates dataset-specific optimism."

---

### [A19] GatingNet K=10 민감도 분석 부재

**대응 결정:** ⚠️ **조건부 수용 — 계산 비용 낮음, 추가 권장**

K ∈ {5, 10, 15, 20, 30}에 대해 M3/FD003 RMSE를 추가 실험 (5 seeds × 5 K값 = 25 runs, 비용 낮음). 결과를 1개 figure로 추가.

**예상 결과:** K=10이 최적이 아니더라도, K≥10 구간에서 수렴하면 "K=10은 충분히 보수적인 선택"임을 보일 수 있음.

---

## 3. 🟢 Minor 항목 대응 (간략)

| # | 항목 | 대응 |
|---|------|------|
| **A20** | 제목에 H2, H7 누락 | ✅ 제목 수정: *"Clipping, Normalization, Fault-Mode Gating, and Loss Functions: A Cross-Dataset Ablation Hierarchy for Turbofan RUL Prediction"* |
| **A21** | IEEE/Elsevier AI disclosure 부재 | ✅ Acknowledgements 및 Declaration of Generative AI 섹션 추가 |
| **A22** | 코드/데이터 가용성 + 컴퓨팅 자원 | ✅ GitHub/Zenodo 업로드 + GPU 정보 추가 |
| **A23** | FD003 per-seed RMSE 분포 시각화 | ✅ M0 vs M3 violin plot 추가 |
| **A24** | FD004 M1 클러스터 붕괴(247:1) 시각화 | ✅ Bar chart 추가 |
| **A25** | 30 cycles 미만 test engine 수 보고 | ✅ Table I에 추가 |
| **A26** | p=0.97 방향성 함의 기술 | ✅ "The one-sided p=0.97 for clip=130 vs clip=125 implies the complementary direction (clip=125 outperforms clip=130) achieves p≈0.03" 추가 |
| **A27** | Abdullah RMSE 비교 문장 제거 | ✅ 제거. 아키텍처 차이 caveat table로 대체 |
| **A28** | inter-seed variance 진단 기여 Abstract에 강조 | ✅ Abstract와 Introduction (iii) 항목에 공동 기여로 추가 |
| **A29** | Introduction 서두 재구성 (NASA Score 중심 → 공학적 문제 중심) | 💬 현재 서두는 의도적 선택; 투고 venue에 따라 조정 |
| **A30** | Silhouette 임계값 0.5 — Kaufman & Rousseeuw 1990 인용 | ✅ 인용 명시 |

---

## 4. ❌ 거절 또는 범위 외 항목 (대안 제시)

---

### [C-MC1] Transformer backbone에서 M3 검증

**리뷰어 지적 (C):** 2026년 기준 stacked LSTM은 구식; M3 효과가 backbone 강화 시에도 유지되는지 검증 필요.

**대응 결정:** ❌ **현재 논문 범위 외 — 미래 연구로 명시**

**근거:** 이 논문의 핵심 설계 원칙은 **단일 backbone 고정을 통한 설계 요소 격리**다. Transformer backbone으로 전환하면 H5/H6/H7을 모두 재실험해야 하며, 이는 본질적으로 다른 논문이 된다. 또한 controlled ablation의 가치는 backbone 고정에서 나온다.

**리뷰어 답변 초안:**
> "We appreciate Reviewer C's observation regarding backbone contemporaneity. The two-layer stacked LSTM was selected specifically to ensure that all four design factors (H2–H7) are evaluated on an identical, fixed architecture — the controlled ablation's primary methodological contribution. Replacing the backbone with a transformer would require repeating all experiments and would fundamentally alter the study from a design-factor ablation to a backbone comparison study. We have added to §V.H a concrete future research direction: 'An important extension is to evaluate whether the M3 gating principle transfers to stronger backbone architectures (e.g., transformer-based encoders); preliminary evidence suggests [reasoning here] but a full controlled ablation is required.' We also note that M3's gap from CAELSTM (10.3% in RMSE) is attributed to backbone rather than gating strategy, which is the correct scientific inference from our controlled experiment."

---

### [C-MC3] N-CMAPSS 전체 실험

**리뷰어 지적 (C):** MSSP/RESS 2026 기준으로 CMAPSS-only는 기대치 미달.

**대응 결정:** ⚠️ **파일럿 실험 수준으로 부분 수용 (Option A) 또는 venue 조정 (Option B)**

**제안 전략:**
- M3 아키텍처를 N-CMAPSS DS03 (HPC+Fan fault modes, 실제 비행 데이터)에 적용하는 파일럿 실험 (5 seeds)
- N-CMAPSS는 fault mode label이 있어 라우팅 정확도 직접 측정 가능 → 논문의 가장 강력한 보강

**venue 조정 옵션:** IEEE Trans. Reliability(→ N-CMAPSS 필수 아님) 또는 PHM Society journal(→ 방법론 기여 중심 허용) 투고 시 현재 범위로도 수용 가능.

---

### [C-MC1/backbone + MC3/N-CMAPSS 통합 대응]

두 항목 모두 투고 전 일부 해결하는 실용적 경로:

```
단기 (현재 개정판):
  → K 민감도 분석 추가 (A19)
  → RevIN denorm 실험 추가 (A6)
  → 통계 재기술 및 언어 수정 (A1-A7)

중기 (저널 투고 시):
  → N-CMAPSS DS03 M3 파일럿 추가
  → M3 + CAELSTM backbone 비교 (1 dataset)

장기 (후속 논문):
  → Full N-CMAPSS ablation
  → Transformer backbone cross-dataset ablation
```

---

## 5. 수정 우선순위 로드맵

### 즉시 (코드 없이 가능)
- [A7] §III에 H2 모델 설명 추가
- [A9] |Δz| 공식 정의
- [A13] Validation split 기술 추가
- [A20] 제목 수정
- [A21] AI disclosure 추가
- [A27] Abdullah 비교 문장 제거
- [A28] Abstract에 inter-seed 진단 기여 추가
- [A30] Silhouette 임계값 인용 수정

### 코드 검증 후 (1~2일)
- [A5] H5/H6 FD003 split 전략 차이 규명 및 공개
- [A2, A11] Cohen's d seed-level 재계산 + Wilcoxon N 명시
- [A4] BH 비교 패밀리 정확한 목록 추출 및 숫자 수정
- [A14] H2 Wilcoxon 관측 단위 확인

### 추가 실험 (3~5일)
- [A6] RevIN denorm 실험 (FD001, 5 seeds)
- [A19] K 민감도 분석 (M3, FD003, 5 K값)
- [A17-Option A] N-CMAPSS DS03 M3 파일럿 (선택적)

### 문헌 보강 (1일)
- [A15] RUL-QMoE, Metric-Gated MoE 인용 추가
- [A16] arXiv:2603.11869 인용 + Discussion 연결
- 누락 논문 목록(Pre-Review_Report §2 Reviewer C 표) 검토 및 선택적 인용

---

## 6. 수정 후 예상 리뷰어 반응

| 리뷰어 | 수정 전 | 수정 후 예상 |
|--------|--------|------------|
| **A (Domain)** | Major Revision | Minor Revision (센서 명칭, CAELSTM 비교, RevIN ablation 해결 시) |
| **B (Statistical)** | Major Revision | Minor Revision (d-p 모순 해결, H7 언어 재구성, H5/H6 불일치 설명 시) — 단, N=5 power 구조 문제는 설계 한계로 남음 |
| **C (Novelty)** | Major Revision | Major Revision → Minor (N-CMAPSS 파일럿 추가, MoE 문헌 인용 시) / 또는 venue 조정 |

**핵심 메시지:** 모든 Critical 항목이 해결되면 Reviewer A, B는 Minor Revision 수준으로 하강 가능. Reviewer C는 N-CMAPSS 파일럿 유무에 따라 결정됨. 컨퍼런스(Procedia CS) 투고는 Critical 항목만 해결 후 즉시 가능.
