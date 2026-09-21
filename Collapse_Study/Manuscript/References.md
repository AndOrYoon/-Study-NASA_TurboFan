# References
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression

**작성일:** 2026-09-21  
**버전:** v0.1  
**목적:** 모든 원고 파일의 인용 번호 단일 출처. 새 참조 추가 시 이 파일을 먼저 업데이트.

---

## 번호 체계

| 접두어 | 대상 | 범위 |
|--------|------|------|
| **P1–P33** | 문헌 참조 (Literature references) | 본문 인용 |
| **A1–A5** | Phase 5 외부 감사 대상 논문 | §7 External Audit |
| **[ANON]** | 투고 중인 선행 연구 (리뷰 기간 익명 처리) | §2 도입 관찰 |

> ⚠️ **충돌 주의:** `Phase5-Result.md`의 P1–P5 표기는 내부 실험 레이블. 원고에서는 반드시 **A1–A5** 사용.

---

## Part I. 문헌 참조 (P1–P33)

### 모델 붕괴 · 평균 예측 이론

**[P1]** Dohmatob, E. et al. (2024). *Model Collapse Demystified: The Case of Regression.* arXiv. (88 citations)  
→ 생성 모델 반복 자기 학습 시 분포 퇴화. MPC와 메커니즘 다름 (단일 판별 훈련 vs. 반복 생성).

**[P17]** Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. *PNAS*, 117(40), 24652–24663. (1,007 citations)  
→ 분류 모델의 terminal phase에서 last-layer feature가 클래스 평균으로 수렴(NC1). MPC의 분류 유사 개념 — 단, NC는 정상 훈련 말기 현상, MPC는 비정상 훈련 초기 현상.

**[P18]** Li, Z. et al. (2022). On the Optimization Landscape of Neural Collapse under MSE Loss: Global Optimality with Unconstrained Features. *arXiv:2203.01238.* (133 citations)  
→ MSE 손실 하의 NC 해(클래스 평균 예측)가 global minimizer임을 증명. 상수 예측이 MSE 경관의 안정적 minimum을 형성한다는 이론적 근거.

**[P19]** Bruna, J., Sprechmann, P., & LeCun, Y. (2015). Super-resolution with deep convolutional sufficient statistics. *arXiv:1511.05666.* (341 citations)  
→ MSE point estimate의 "regression-to-the-mean problem" 최초 명명. MPC 정의 구성 요소 *"concentrate around the conditional mean"* 근거.

**[P20]** Mathieu, M., Couprie, C., & LeCun, Y. (2016). Deep multi-scale video prediction beyond mean square error. *ICLR 2016.* (2,020 citations)  
→ 비디오 예측에서 MSE의 "inherently blurry predictions" 실증. MPC 정의 구성 요소 *"reduced dispersion relative to the target distribution"* 근거.

**[P21]** Huang, W. (2026). Resolving Multi-Modal Regression by Difference-Quotient-Based Clustering. *arXiv:2608.25467.*  
⚠️ arXiv 미검증 preprint — peer review 미완료. 보조 이론으로만 인용.  
→ "squared loss → conditional mean" 이론 명시. "mean collapse"를 실험 baseline으로 수치화(MSE=1.33 vs oracle=0.09). MPC와 달리 데이터 구조(multimodality)가 원인으로 설정.

---

### LSTM 훈련 불안정성 · 회귀 편향

**[P2]** Al-Selwi, S. et al. (2023). LSTM Inefficiency in Long-Term Dependencies Regression Problems. *J. Advanced Research in Applied Sciences and Engineering Technology.* (157 citations)  
→ CMAPSS에서 LSTM 소실 기울기(VGP) 실증 분석. **느린 수렴** 문제 — MPC의 **즉각적 trivial solution 수렴**과 구별.

**[P7]** Lee, M. & Chen, T. (2025). Systematic Bias of Machine Learning Regression Models and Correction. *IEEE TPAMI.* (25 citations)  
→ 정상 학습 후 분포 꼬리의 체계적 평균 편향(CMC). MPC 정의 구성 요소 *"underrepresenting extreme values"* 근거.  
⚠️ CMC ≠ MPC: CMC는 정상 학습 완료 후 통계 현상; MPC는 학습 자체가 trivial solution에 포획.

---

### C-MAPSS RUL 예측

**[P29]** Saxena, A., Goebel, K., Simon, D., & Ecker, W. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *2008 IEEE Int. Conf. on Prognostics and Health Management (PHM).* doi: 10.1109/PHM.2008.4711414  
→ **C-MAPSS 데이터셋 원본 논문.** FD001–FD004 생성 시뮬레이터 및 데이터 구조 기술.

**[P16]** Li, X. et al. (2018). Remaining useful life estimation in prognostics using deep convolution neural networks. *RESS*, 172, 1–11. (1,561 citations)  
→ C-MAPSS 딥러닝 RUL 예측 기초 논문. piecewise-linear RUL clip=125 표준 확립.  
*clip=125 근거 인용에 사용.*

**[P3]** Ellefsen, A. L. et al. (2019). Remaining useful life predictions for turbofan engine degradation using semi-supervised deep architecture. *RESS*, 183, 240–251. (466 citations)  
→ 반지도학습으로 수렴 안정성 개선.

**[P4]** Chao, M. A. et al. (2021). Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics. *Data*, 6(1), 5. (403 citations)  
→ N-CMAPSS 원논문. CMAPSS의 "제한된 대표성" 지적.

**[P5]** Elsherif, S. M. et al. (2025). A deep learning-based prognostic approach for predicting turbofan engine degradation and remaining useful life. *Scientific Reports*, 15, Art. 12959. doi: 10.1038/s41598-025-09155-z (29 citations)  
→ CAELSTM, FD003 RMSE = 13.40. **= 외부 감사 대상 A5.**

**[P8]** Das, S. et al. (2024). Uncertainty-aware deep learning for monitoring and fault diagnosis from synthetic data. *RESS*, 240, 109606. (42 citations)  
→ 합성 데이터 훈련 DNN의 불확실성 정량화.

**[P15]** Zhuang, L. et al. (2023). A prognostic driven predictive maintenance framework based on Bayesian deep learning. *RESS*, 234, 109181. (173 citations)

---

### 조기 종료 · 검증 분할

**[P14]** Prechelt, L. (1998). Early stopping — but when? In *Neural Networks: Tricks of the Trade.* Springer. (2,469 citations)  
→ 검증 손실 기반 조기 종료의 체계적 기준 수립. 핵심 가정: "검증 손실이 일반화를 반영한다" — MPC 시나리오에서 위반됨.

**[P6]** Vabalas, A. et al. (2019). Machine learning algorithm validation with a limited sample size. *PLoS ONE*, 14(11), e0224365. (1,546 citations)  
→ 소규모 데이터에서 고정 K-fold CV의 강한 성능 추정 편향 실증.

**[P12]** Miseta, T. et al. (2023). Surpassing early stopping: A novel correlation-based stopping criterion for neural networks. *Neurocomputing*, 556, 126627. (60 citations)

**[P13]** Mahsereci, M. et al. (2017). Early stopping without a validation set. *arXiv:1703.09580.* (113 citations)

---

### 벤치마크 공정성 · 재현성

**[P23]** Musgrave, K., Belongie, S., & Lim, S.-N. (2020). A Metric Learning Reality Check. *ECCV 2020.* (554 citations)  
→ "측정 프로토콜이 결론을 결정한다" — metric learning 4년간 개선이 실험 조건 아티팩트임을 실증.

**[P24]** Lučić, M. et al. (2018). Are GANs Created Equal? A Large-Scale Study. *NeurIPS 2018.* (1,120 citations)  
→ 충분한 하이퍼파라미터 탐색 하에서 GAN 변형 간 유의 차이 없음.

**[P22]** Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). Are we really making much progress? *ACM RecSys 2019.* (694 citations)  
→ 재현성 인프라 문제. 본 연구와의 직접 연결은 P23/P24보다 약함.

---

### 시계열 벤치마크

**[P9]** Qiu, X. et al. (2024). TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods. *VLDB 2024.* (345 citations)

**[P10]** Wang, S. et al. (2024). Deep Time Series Models: A Comprehensive Survey and Benchmark. *IEEE TPAMI.* (308 citations)

**[P11]** Tan, Y. et al. (2025). SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series. *arXiv.* (4 citations)

---

### 측정 지표 · 방법론

**[P25]** Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.  
→ Cohen's κ. MPC 판정 blind adjudication의 inter-rater agreement 측정.

**[P26]** Schulz, K. F., Altman, D. G., Moher, D., & CONSORT Group (2010). CONSORT 2010 Statement. *BMJ*, 340, c332. (2,500+ citations)  
→ Blinded endpoint assessment 방법론 표준. §4.1.3 blind adjudication 원칙의 출처.

**[P27]** Murphy, A. H. (1988). Skill scores based on the mean square error and their relationships to the correlation coefficient. *Monthly Weather Review*, 116(12), 2417–2424.  
→ CBR 기저 개념(MSE Skill Score). CBR² = 1 − SS.

**[P28]** Zeiler, M. D. & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV 2014*, LNCS 8689, 818–833. (19,000+ citations)  
→ ISS의 입력 교란 기반 민감도 개념 원출처. 본 연구에서 시계열·연속 회귀 맥락으로 일반화.

---

### 합성 데이터 · 불확실성

**[P1]** → 위 참조 (Dohmatob 2024)

---

### 선행 연구 (투고 중 — 리뷰 기간 익명 처리)

**[ANON]** Authors (2026). [Title anonymized for review]. *Reliability Engineering & System Safety* (under review).  
→ BMAD H6 실험에서 MPC 최초 관찰. 수정 프로토콜 후 RMSE 43.23 → 12.97 회복 확인. 본 연구의 동기가 되는 선행 연구.  
*최종 원고 제출 시 de-anonymization.*

---

### C-MAPSS 원본 (추가 등록)

**[P29]** → 위 참조 (Saxena & Goebel 2008)

---

## Part II. 외부 감사 대상 논문 (A1–A5)

> Phase 5 실험에서 재현한 5편. `Phase5-Result.md`의 P1–P5와 **1:1 대응** (원고에서는 A 번호 사용).

| Audit ID | Phase5 ID | Citation | 저널 | 확인 수준 |
|----------|:---------:|---------|------|:---:|
| **A1** | P1 | Meng et al. (2023) [P31] | Expert Systems with Applications | ✅ 확인 |
| **A2** | P2 | Qin et al. (2024) [P32] | Eng. Applications of AI (EAAI) | ✅ 확인 |
| **A3** | P3 | Zheng et al. (2017) [P33] | IEEE ICPHM (conference) | ✅ 확인 |
| **A4** | P4 | Representative pre-2020 protocol | EAAI/ESWA 대표 패턴 | 🟡 패턴만 확인 |
| **A5** | P5 | Elsherif et al. (2025) **= [P5]** | Scientific Reports | ✅ 확인 |

**[P31]** Meng, H. et al. (2023). Bayesian gated-transformer model for risk-aware prediction of aero-engine remaining useful life. *Expert Systems with Applications*, 238, Art. 121859. PII: S0957417423023618  
→ 감사 대상 A1. FD003 LSTM 비교 기준선 RMSE 미확인(전문 미접근). 프로토콜 A1+B2+C2+patience=15 추론.

**[P32]** Qin, Y. et al. (2024). Spatial and temporal attention-based and residual-driven long short-term memory networks with implicit features for remaining useful life prediction. *Engineering Applications of Artificial Intelligence*, 133, Art. 108563. PII: S0952197624007073  
→ 감사 대상 A2. 보고 FD003 RMSE = 12.14. 프로토콜 A1+B2+C2+patience=20 추론.

**[P33]** Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). Long short-term memory network for remaining useful life estimation. *2017 IEEE Int. Conf. on Prognostics and Health Management (ICPHM).* doi: 10.1109/ICPHM.2017.7998311  
→ 감사 대상 A3. CMAPSS 최다 인용 LSTM 기준선. 보고 FD003 RMSE = 16.18. 프로토콜 A1+B2+C2+patience=10 추론.

**A4 — Representative pre-2020 protocol pattern:**  
특정 단일 논문이 아닌 2018–2021년 EAAI/ESWA 논문의 지배적 프로토콜 패턴 (RUL clipping 미적용, A1+B2+C1). 특정 인용 없이 "대표 패턴"으로 기술. §7에서 별도 설명.

**[P5]** Elsherif et al. (2025) → Part I 참조. A5 = P5 동일 논문.

---

## Part III. 미해결 [REF] 플레이스홀더 추적

| 파일 | 위치 | 현재 표기 | 해결 방법 | 상태 |
|------|------|-----------|-----------|:----:|
| Introduction.md | §1 Context | `[REF]` — C-MAPSS 원본 | → `[P29]` | ✅ 해결 |
| Introduction.md | §2 Motivating Observation | `[REF to BMAD prior work, anon for review]` | → `[ANON]` | ✅ 해결 |
| Related_Work.md | §2.1 첫 단락 | `[REF: Saxena 2008]` | → `[P29]` | ✅ 해결 |
| Methodology.md | §4.2.1 첫 단락 | `[REF: Saxena 2008]` | → `[P29]` | ✅ 해결 |
| Methodology.md | §4.2.1 전처리 #4 | `[REF]` — clip=125 표준 | → `[P16]` | ✅ 해결 |
| Methodology.md | §4.2.4 Table 4 | `[REF]` × 5 — audit 논문 | → A1–A5 + P31/P32/P33/P5 | ✅ 해결 |

> **다음 작업:** 위 "해결 방법" 컬럼의 표기로 각 원고 파일의 `[REF]`를 실제 교체 필요.  
> 현재 원고 파일에는 아직 `[REF]`가 남아 있음 — 최종 원고 정리 단계에서 일괄 교체 권장.

---

## Part IV. 번호 현황 요약

| 번호 | 저자 (연도) | 주제 | 주요 사용 섹션 |
|:----:|------------|------|:---:|
| P1 | Dohmatob (2024) | 생성 모델 붕괴 | §2.3 |
| P2 | Al-Selwi (2023) | LSTM VGP C-MAPSS | §2.3, §4 |
| P3 | Ellefsen (2019) | 반지도 RUL | §2.1 |
| P4 | Chao (2021) | N-CMAPSS | §2.6 |
| P5 | Elsherif (2025) | CAELSTM FD003 | §2.1, §7(A5) |
| P6 | Vabalas (2019) | Val split bias | §2.4, §4.4 |
| P7 | Lee & Chen (2025) | 회귀 평균 편향 | §2.3, §3.1 |
| P8 | Das (2024) | 합성 데이터 불확실성 | §2.6 |
| P9 | Qiu (2024) | TFB benchmark | §2.5 |
| P10 | Wang (2024) | TSLib | §2.5 |
| P11 | Tan (2025) | SynTSBench | §2.6 |
| P12 | Miseta (2023) | CDSC early stopping | §2.4 |
| P13 | Mahsereci (2017) | ES without val | §2.4 |
| P14 | Prechelt (1998) | Early stopping | §2.4, §4 |
| P15 | Zhuang (2023) | Bayesian PHM | §2.1 |
| P16 | Li (2018) | DCNN RUL, clip=125 | §2.1, §4.2.1 |
| P17 | Papyan (2020) | Neural Collapse | §3.1 |
| P18 | Li (2022) | MSE NC landscape | §3.1 |
| P19 | Bruna (2015) | Regression-to-mean | §2.2, §3.1 |
| P20 | Mathieu (2016) | Blurry pred MSE | §2.2, §3.1 |
| P21 | Huang (2026) | Multimodal collapse | §2.2, §3.1 |
| P22 | Ferrari Dacrema (2019) | RecSys 재현성 | §2.5 |
| P23 | Musgrave (2020) | Metric learning | §2.5, §4 |
| P24 | Lučić (2018) | GAN equality | §2.5, §4 |
| P25 | Cohen (1960) | Cohen's κ | §4.1.3 |
| P26 | Schulz/CONSORT (2010) | Blind adjudication | §4.1.3 |
| P27 | Murphy (1988) | MSE Skill Score | §4.1.2 |
| P28 | Zeiler & Fergus (2014) | Input sensitivity | §4.1.2 |
| **P29** | **Saxena & Goebel (2008)** | **C-MAPSS 원본** | **§1, §2.1, §4.2.1** |
| **P30** | **[ANON] Authors (2026)** | **BMAD prior work** | **§2** |
| **P31** | **Meng (2023)** | **감사 A1, ESWA** | **§7** |
| **P32** | **Qin (2024)** | **감사 A2, EAAI** | **§7** |
| **P33** | **Zheng (2017)** | **감사 A3, ICPHM** | **§7** |

*굵은 글씨 = 이번에 신규 추가된 항목*
