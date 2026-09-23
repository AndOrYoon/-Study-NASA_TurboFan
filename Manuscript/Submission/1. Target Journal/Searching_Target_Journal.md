# Target Journal Search — Turbofan RUL Paper

**Paper title:** From Fleet Normalization to Fault-Mode Gating: A Cross-Dataset Ablation Study of Turbofan Remaining Useful Life Prediction
**Last updated:** 2026-09-23
**Author affiliation:** Electronics and Telecommunications Research Institute (ETRI)

---

## 0. Submission Strategy (Updated 2026-09-23)

> 🔴 **RESS 데스크 리젝션 (2026-09-23):** "manuscripts on signal processing and/or fault diagnosis are no longer included in the scope of this journal" + "missing a significant body of recent literature." 투고 전략 전면 재설정.

**투고 전략 재설정 근거:**
- RESS 제외 이유 ① Scope — fault diagnosis / signal processing 명시 배제
- RESS 제외 이유 ② 문헌 부족 — 타깃 저널 내 최신 literature 불충분, 기여 차별성 불명확
- ⚠️ **Cross-cutting issue:** 두 번째 이유는 저널 무관 공통 문제. **어느 저널이든 투고 전 문헌 보강 필수.**

| Priority | Journal | IF | Quartile | 레퍼런스 인용 | Rationale |
|----------|---------|-----|----------|------------|-----------|
| **1순위** | **Advanced Engineering Informatics (AEI)** | **11.5** | Q1 | **2건** ([30],[39]) | H6 직접 비교군([30]) 게재 저널; IF 대폭 상승(8→11.5); AI+공학 융합 스코프 완벽 일치 |
| **2순위** | **Expert Systems with Applications (ESWA)** | ~7.0–9.4 | Q1 | 미확인 | PHM/RUL 논문 다수; CMAPSS 선례 확인; 스코프 넓어 거절 위험 낮음 |
| **3순위** | **Mechanical Systems and Signal Processing (MSSP)** | **10.1** | Q1 | 미확인 | PHM·fault diagnosis 정중앙; **단 simulation-only 리스크 + 8–15p 제한** |
| **백업** | **IEEE Access** | 4.2 | Q2 | 1건 ([42]) | 빠른 게재; IF 낮음 |
| ~~탈락~~ | ~~RESS~~ | ~~7.2~~ | ~~Q1~~ | ~~4건~~ | ~~2026-09-23 데스크 리젝션 — scope 배제~~ |
| ~~보류~~ | ~~IEEE TII~~ | ~~12.3~~ | ~~Q1~~ | ~~0건~~ | ~~N-CMAPSS·UQ 미해결; 수락률 32–38% 상한; 10p 제한~~ |

```
투고 순서: AEI → (탈락) → ESWA → (탈락) → MSSP → (탈락) → IEEE Access
```

---

## 1. ETRI Journal — Fit Assessment

### Journal Profile

| Item | Detail |
|------|--------|
| Publisher | Wiley (Open Access) |
| Impact Factor (2026) | 2.0 (Q3 Engineering) |
| Core scope | Information, telecommunications, electronics, computing, networks |

### Verdict: **Not recommended**

- 스코프 미스매치: PHM·항공우주 도메인이 ICT 저널 주제와 불일치
- CMAPSS / RUL 게재 선례 없음 → 데스크 리젝션 위험
- IF 2.0 (Q3) — 논문 기여도 대비 저널 위상 낮음

---

## 2. IEEE Transactions on Industrial Electronics (TIE) — Ruled Out

### Journal Profile

| Item | Detail |
|------|--------|
| IF (2026) | ~7.5 (Q1) |
| Editor-in-Chief | Prof. Yang Shi; Co-EiCs: Xinkai Chen, G. Buticchi, A. G. Yepes (2026–2028) |
| Submission system | IEEE ScholarOne Manuscripts |
| APC | $2,645 |

### Page Limits (2025.01.01~)

| 유형 | 최초 제출 | 최종본 | 초과 과금 |
|------|-----------|--------|----------|
| Regular Paper | 10페이지 | 12페이지 | 11페이지부터 |
| Letter | 4페이지 | 6페이지 | 5~6페이지 |

### ⚠️ 결격 사유: 하드웨어 실험 요건

> 출처: [TIE Experimental Verification Policy](https://www.ieee-ies.org/pubs/transactions-on-industrial-electronics/287-experimental-verification)

*"All submissions must contain experimental verification of novel concepts. The experimental rig must contain hardware components other than a PC/laptop."*

- 본 논문: NASA CMAPSS 소프트웨어 시뮬레이션 → 요건 **미충족**
- 편집부가 능동적으로 검사 → 데스크 리젝션 위험 **높음**

### Verdict: **Not recommended — 하드웨어 요건 미충족**

---

## 3. IEEE Transactions on Industrial Informatics (TII) — 1순위 ✅

> 출처: [IEEE IES — TII](https://www.ieee-ies.org/pubs/transactions-on-industrial-informatics) / [TII Submission Guide 2026](https://manusights.com/blog/ieee-transactions-on-industrial-informatics-submission-guide)

### Journal Profile

| Item | Detail |
|------|--------|
| IF (2025) | ~11.7 (Q1 Engineering) |
| 심사 방식 | **Double-blind** |
| 제출 시스템 | **IEEE Author Portal** (ScholarOne 아님) |
| 기관 이메일 | 전 저자 필수 (2024.02.15~) |

### Page Limits (2025.01.01~)

| 유형 | 최초 제출 | 최종본 | 초과 과금 |
|------|-----------|--------|----------|
| Regular Paper | **10페이지** | 12페이지 | 11페이지부터 $250/페이지 ($200 IES회원) |
| Survey (EiC 사전 승인) | 12페이지 | — | — |
| Letter | 4페이지 | 6페이지 | 5~6페이지 |

### 스코프 적합성

공식 스코프: *"AI enhanced automation, industrial cyber-physical systems, industrial IoT, condition monitoring, predictive maintenance, knowledge-based automation"*

| 논문 기여 | TII 스코프 |
|----------|-----------|
| LSTM 기반 RUL 예측 | ✅ AI enhanced automation |
| 센서 전처리 ablation | ✅ Industrial informatics |
| 결함 모드 분리 (M3) | ✅ Condition monitoring |
| 4개 데이터셋 교차검증 | ✅ Industrial system validation |
| 하드웨어 요건 | ✅ 없음 |
| CMAPSS 게재 선례 | ✅ 다수 확인 |

### TII 제출 체크리스트 (공식 + 실무 종합)

> 출처: [TII Checklist page](https://www.ieee-ies.org/pubs/transactions-on-industrial-informatics/revised-submission?view=article&id=423) / [TII Submission Guide 2026](https://manusights.com/blog/ieee-transactions-on-industrial-informatics-submission-guide)
>
> **수락률 < 20%** — 기술적으로 올바른 논문도 50~70%가 "outstanding and original" 기준 미달로 리젝션. 기술적 정확성만으로는 부족함.

#### Stage 1 — 스코프 & 기여도 검토

| # | 항목 | 본 논문 상태 |
|---|------|------------|
| 1 | 원고가 industrial informatics 주제를 다루는가 (순수 전자공학·제어이론·범용 AI/ML 아님) | ✅ PHM·조건 모니터링·산업 AI |
| 2 | 기여가 **outstanding and original**인가 (기술적으로 맞는 것만으로는 불충분) | ✅ 65.8% RMSE 개선 + 4-way ablation |
| 3 | 산업 시스템 맥락과 분리 불가능한 연구인가 ("AI + 산업 데이터" 구성 아님) | ✅ (단, Cover Letter에서 명시 강화 필요) |
| 4 | TIE·TCST 등 자매 저널이 아닌 TII가 적합한 이유를 설명할 수 있는가 | ✅ TIE 하드웨어 요건 미충족; TII 조건모니터링 스코프 |

#### Stage 2 — 원고 형식

| # | 항목 | 본 논문 상태 |
|---|------|------------|
| 5 | 최초 제출 **10페이지 이하** (IEEE 두 열 형식 기준, 엄격 적용) | ⚠️ **미확인 — 변환 후 페이지 수 측정 필요** |
| 6 | IEEE TII 공식 두 열 템플릿 사용 | ⚠️ **현재 Markdown 원고 → 템플릿 적용 필요** |
| 7 | **Double-blind 준수:** 본문·각주·감사의 글·저자 약력에 저자 식별 정보 없음 | ⚠️ 플레이스홀더 처리됨 — 최종 확인 필요 |
| 8 | 파일 크기 40 MB 이하 | ✅ (예상 문제 없음) |
| 9 | 산업 시스템 아키텍처 또는 맥락을 보여주는 그림 포함 | ✅ Figure 포함 예정 |
| 10 | 초록이 TII 기준에 맞게 작성됨 | ✅ |

#### Stage 3 — 선언문 & 첨부 서류

| # | 항목 | 본 논문 상태 |
|---|------|------------|
| 11 | 이해충돌(Conflict of Interest) 공개 완료 | ⚠️ 작성 필요 |
| 12 | 데이터 가용성 선언 (공개 데이터: NASA CMAPSS URL 명시) | ⚠️ 작성 필요 (NASA CMAPSS는 공개 데이터) |
| 13 | 인간 대상 연구 해당 없음 (Ethics statement 불필요) | ✅ |
| 14 | 저자 기여도 및 연구비 지원 기재 | ⚠️ 작성 필요 |
| 15 | 추천 리뷰어 (선택 사항) | — |

#### Stage 4 — Cover Letter

| # | 항목 | 본 논문 상태 |
|---|------|------------|
| 16 | Outstanding-and-original 기여 근거 설명 | ✅ Cover Letter v2.0에 포함 |
| 17 | TIE·TCST 등 자매 저널과의 차별성 명시 | ⚠️ 추가 보강 권장 |
| 18 | Industrial informatics 관점이 핵심임을 명시 (부수적 아님) | ✅ Cover Letter v2.0에 포함 |

#### Stage 5 — 최종 제출 전 점검

| # | 항목 | 본 논문 상태 |
|---|------|------------|
| 19 | 제출 시스템: **IEEE Author Portal** (ScholarOne 아님, 2025.02 이후) | ⚠️ 계정 생성 필요 |
| 20 | 전 저자 **기관 이메일** 사용 (2024.02.15~ 필수) | 🚨 **현재 Gmail(spiceyoon@gmail.com) 사용 — ETRI 기관 이메일로 변경 필수. 미변경 시 심사 진입 불가** |
| 21 | 현재 EiC 이름 확인 후 Cover Letter 수신인 기재 | ⚠️ 제출 직전 확인 필요 |
| 22 | 파일 누락 없음 (원고 PDF + 소스파일 + 커버레터 + 선언문) | ⚠️ 제출 시 확인 |

#### 데스크 리젝션 주요 원인 (편집부 공개)

- 저자 신원 노출 (본문 내 이름·소속 등)
- 10페이지 초과
- 파일 누락 또는 선언문 미완성
- 스코프 미스매치 및 프레이밍 문제 (방법론 약점보다 **스코프 미스매치가 조기 리젝션의 주원인**)

### Verdict: **권장 — 1순위 확정**

---

## 4. IEEE Access — 2순위 (백업)

> 출처: [IEEE Access Journal Guide 2026](https://manusights.com/journals/ieee-access) / [IEEE Access Submission Guidelines](https://ieeeaccess.ieee.org/authors/submission-guidelines/)

### Journal Profile

| Item | Detail |
|------|--------|
| IF (2026) | 4.2 (Q2 Engineering) |
| 심사 방식 | Single-blind |
| 페이지 제한 | **없음** (20페이지 권장) |
| 하드웨어 요건 | 없음 |
| APC | $2,160 |
| 평균 결정 기간 | **4~8주** |
| 수락률 | ~27~45% |

### 장점
- 페이지 제한 없음 → 원고 길이 조정 불필요
- Single-blind → double-blind 익명화 작업 불필요
- 빠른 심사 (4~8주), 수락률 높음
- CMAPSS RUL 논문 다수 게재 확인

### 우려 사항
- IF 4.2 (Q2) — 논문 기여도 대비 저널 위상 낮음
- 연간 수만 편 게재 → 개별 논문 가시성 희석
- IEEE Transactions 대비 CV 신호 약함
- CAS 저널 등급 리스트 주의 등급 분류 이력

### Verdict: **TII 리젝션 시 2순위 백업으로 적합**

---

## 4b. Reliability Engineering & System Safety (RESS) — ~~1순위~~ 🔴 데스크 리젝션 (2026-09-23)

> **에디터 코멘트 원문:**
> "Your manuscript is missing a significant body of recent literature from relevant journals in the field considering both the fundamental methodology and application. As a result, the contributions and advantages of the proposed method with respect to the current state-of-the-art within the scope of this journal are not clear and the manuscript might be more appropriate for one of the other journals in the reference list."
>
> "**manuscripts on signal processing and/or fault diagnosis are no longer included in the scope of this journal.**"
>
> **판단:** Scope 불일치 (fault diagnosis/signal processing 명시 배제) + 문헌 부족 지적. 재투고 불가.

### Journal Profile

| Item | Detail |
|------|--------|
| Publisher | Elsevier |
| IF (2024) | ~7.2 (Q1 Engineering) |
| 심사 방식 | Single-blind |
| 페이지 제한 | **없음** |
| APC | 오픈 액세스 선택 시 $3,200 (비OA 무료) |
| 제출 시스템 | Elsevier Editorial Manager |

### 레퍼런스 내 RESS 게재 논문 (4건)

| 레퍼런스 | 논문 | 인용 이유 |
|---------|------|---------|
| [6] Li et al. 2018 | Deep CNN for RUL (RMSE FD001=12.61) | clip=125 최초 명시적 정당화 |
| [7] Ellefsen et al. 2019 | Semi-supervised deep architecture (453인용) | 비지도 사전학습 검증 |
| [10] Xu et al. 2022 | Spatio-temporal Transformer, FD002/FD004 | 운전조건 처리 비교군 |
| [11] Zhang et al. 2023 | Trend-Augmented Transformer (117인용) | 열화 추세 임베딩 선행연구 |

### 스코프 적합성

- "Reliability, safety and risk in industrial systems" 내 PHM·예측 유지보수 명시
- CMAPSS 딥러닝 ablation이 RESS 역대 최다 인용 논문군에 속함 ([6], [7] 포함)
- H2의 "clip=None → NASA Score 6 orders of magnitude 증폭" 결과가 RESS의 reliability risk 관점과 직접 부합
- 3계층 설계 위계를 신뢰성 엔지니어링 체크리스트로 재프레이밍 시 RESS 스코프 언어와 자연스럽게 정렬

### RESS 전환을 위한 추가 작업

| 우선순위 | 항목 | 내용 |
|--------|------|------|
| 필수 | Abstract + Introduction 재프레이밍 | "industrial informatics" → "reliability & PHM" 중심 재서술 |
| 필수 | Cover Letter RESS 버전 신규 작성 | TII 버전 전면 재작성 |
| 필수 | 표 번호 충돌 해결 | §III.E Table III ↔ Results §IV Table III–V 재번호 |
| 필수 | Elsevier 템플릿 적용 | Markdown → Elsevier LaTeX/Word 변환 |
| 권장 | §V.F 3계층 → 신뢰성 설계 체크리스트 재서술 | RESS 스코프 언어 정렬 |
| 선택 | T6: False routing 민감도 분석 | RESS 신뢰성 관점에서 유의미 |

### Verdict: **1순위 확정 — CMAPSS 딥러닝의 홈 저널, 예상 수락률 42–52%**

---

## 4c. Advanced Engineering Informatics (AEI) — 3순위

### Journal Profile

| Item | Detail |
|------|--------|
| Publisher | Elsevier |
| IF (2024) | ~8.0 (Q1 Engineering) |
| 심사 방식 | Single-blind |
| 페이지 제한 | **없음** |

### 레퍼런스 내 AEI 게재 논문 (2건)

| 레퍼런스 | 논문 | 인용 이유 |
|---------|------|---------|
| [30] (2024) | Multi-Task Fault Mode Feature Separation (AEI Vol.60) | H6 최근접 선행연구 — 비지도 오토인코더+K-means로 고장 모드 분리 후 멀티태스크 학습 |
| [39] Liu et al. 2021 | Unsymmetrical penalty loss for RUL (47인용) | H7 비대칭 손실 선행연구 |

### 스코프 적합성

- "AI-enhanced engineering systems, knowledge-based and data-driven engineering" 명시
- [30]이 AEI에 게재 → 리뷰어 풀이 H6 M3의 직접 비교군을 잘 이해하는 전문가들로 구성될 가능성 높음
- H6 M3 아키텍처 novelty를 중점으로 투고 시 적합

### Verdict: ~~3순위~~ → **1순위 ✅ (2026-09-23 재확정)** — RESS 탈락으로 순위 상승; IF 11.5로 대폭 상승

---

## 4d. Mechanical Systems and Signal Processing (MSSP) — 3순위 ⚠️ (신규 추가)

### Journal Profile

| Item | Detail |
|------|--------|
| Publisher | Elsevier |
| IF (2024) | **10.07** (Q1) |
| 심사 방식 | Single-blind |
| 페이지 제한 | **8–15페이지** (리서치 페이퍼 기준) |
| 제출 시스템 | Elsevier Editorial Manager |
| ML 논문 특별 가이드라인 | ✅ 있음 (2024-12 발행, PDF 확인) |

### 스코프 적합성

| 논문 기여 | MSSP 스코프 |
|----------|-----------|
| RUL 예측 (degradation prognostics) | ✅ 명시 |
| 센서 신호 전처리·정규화 (H5) | ✅ 신호처리 정중앙 |
| 결함 모드 라우팅 (H6) | ✅ fault diagnosis 명시 |
| CMAPSS 터보팬 엔진 시뮬레이션 | ✅ 항공·기계 시스템 |

- RESS가 명시적으로 배제한 "signal processing / fault diagnosis" 영역을 MSSP는 정중앙에서 커버

### ⚠️ 리스크 1: Simulation-Only 논문

MSSP 공식 ML 가이드라인 및 제출 가이드 경고:
> *"Manuscripts relying solely on public benchmark datasets without clear mechanical-system context face rejection. Editors want evidence the method solves an actual engineering problem, not just achieves high accuracy metrics."*

- **완화 전략:** CMAPSS = NASA 터보팬 시뮬레이션 (명백한 기계 시스템 맥락). Abstract/Introduction에서 "turbofan engine degradation engineering problem" 프레이밍 강조 필수. 정확도 지표보다 **설계 결정의 공학적 함의**를 전면에 배치.
- CMAPSS benchmark 논문이 MSSP에 게재된 직접 선례는 검색으로 미확인 → **투고 전 최근 5년 MSSP 게재 CMAPSS 논문 여부 수동 확인 필요**

### ⚠️ 리스크 2: 페이지 제한

- 8–15페이지 = 현재 원고 대비 상당한 압축 필요 (현재 원고 ~13,000 단어 이상)
- 투고 전 LaTeX 컴파일 후 MSSP 2열 템플릿 기준 페이지 수 측정 필수

### Verdict: **3순위 — IF 높지만 simulation-only 리스크 + 페이지 압축 부담이 큰 리스크; AEI·ESWA 모두 탈락 시 선택**

---

## 4e. Expert Systems with Applications (ESWA) — 2순위 ✅ (신규 추가)

### Journal Profile

| Item | Detail |
|------|--------|
| Publisher | Elsevier |
| IF (2026) | **~7.0–9.4** (Q1, 출처별 차이; 최신 JCR 재확인 권장) |
| 심사 방식 | Single-blind |
| 페이지 제한 | **없음** |
| 제출 시스템 | Elsevier Editorial Manager |
| APC | 비OA 무료 |

### 스코프 적합성

- "AI applications in engineering, deep learning, neural networks, data mining, machine learning" 명시
- PHM/RUL 논문 다수 게재 확인 — CMAPSS 선례 있음
- 하드웨어 요건 없음
- 스코프가 넓어 fault diagnosis / signal processing 기반 논문도 수용

### 장점

- AEI 탈락 시 가장 안전한 선택지 (scope 거절 위험 낮음)
- 페이지 제한 없음 → 원고 길이 조정 불필요
- CMAPSS benchmark 논문 수용 확인
- Elsevier 플랫폼 (기존 format 재활용 가능)

### 우려 사항

- IF가 AEI(11.5)·MSSP(10.1) 대비 낮음
- 폭넓은 스코프 → 개별 논문 가시성 희석 가능
- 주요 레퍼런스 인용 저널 아님 → Cover Letter에서 기여 차별성 별도 강조 필요

### Verdict: **2순위 ✅ — 안전하고 적합한 스코프; AEI 탈락 시 우선 선택**

---

## 5. 저널 최종 비교 (Updated 2026-09-23)

| 기준 | **AEI** | **ESWA** | **MSSP** | IEEE Access | ~~RESS~~ | TIE | TII |
|------|---------|---------|---------|-------------|------|-----|-----|
| IF | **11.5 (Q1)** | 7.0–9.4 (Q1) | **10.1 (Q1)** | 4.2 (Q2) | ~~7.2 (Q1)~~ | 7.5 (Q1) | 12.3 (Q1) |
| 스코프 적합 | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ~~❌ 탈락~~ | ❌ 하드웨어 | ★★★★★ |
| 레퍼런스 인용 | **2건** | 미확인 | 미확인 | 1건 | ~~4건~~ | 0건 | 0건 |
| 하드웨어 요건 | ✅ 없음 | ✅ 없음 | ⚠️ 권장 | ✅ 없음 | ~~✅ 없음~~ | ❌ 필수 | ✅ 없음 |
| 페이지 제한 | **없음** | **없음** | **8–15p** | **없음** | ~~없음~~ | 10p | 10p |
| 심사 방식 | Single | Single | Single | Single | ~~Single~~ | Single | Double |
| 예상 수락률 | 30–40% | 35–50% | 25–35% | 27–45% | ~~42–52%~~ | — | 32–38% |
| 권장 순위 | **✅ 1순위** | **✅ 2순위** | **⚠️ 3순위** | **백업** | ~~❌ 데스크 리젝션~~ | ❌ | ~~보류~~ |

---

## 6. Pre-Submission Action Items (AEI 1순위 기준, 2026-09-23 업데이트)

> Elsevier Editorial Manager 기준. RESS 투고 시 준비된 elsarticle LaTeX 패키지는 AEI도 Elsevier이므로 대부분 재활용 가능.

### 🔴 Cross-cutting — 저널 무관 필수 (RESS 에디터 지적 사항)

| # | 항목 | 조치 내용 |
|---|------|---------|
| 0a | **타깃 저널 내 최신 문헌 보강** | AEI 내 최근 3–5년 CMAPSS/RUL/PHM 논문 조사 → 인용 추가; 기여 차별성을 해당 저널 SOTA 대비 명시 |
| 0b | **Abstract + Introduction 재프레이밍** | "reliability & PHM" → "AI-enhanced engineering informatics / data-driven PHM" 중심으로 재서술; AEI 리뷰어 첫인상 결정 |
| 0c | **Cover Letter AEI 버전 신규 작성** | RESS 버전 전면 재작성 — AEI Editor-in-Chief 수신; [30] 동일 저널 선행연구와의 차별성 강조; "missing literature" 지적 대응하여 문헌 보강 내용 명시 |

### 🚨 제출 불가 — 즉시 조치 필요

| # | 항목 | 조치 내용 |
|---|------|---------|
| A | **AEI 저자 가이드라인 확인** | AEI 공식 Author Instructions 접속 → 단어 수 제한, 그림 형식, 선언문 요건 확인 |
| B | **표 번호 충돌 확인** | §III.E Table III ↔ Results §IV 번호 충돌 여부 재확인 (RESS LaTeX에서 수정했으나 AEI 버전 재점검 필요) |

### ⚠️ 필수 — 제출 전 완료

| # | 항목 | 조치 내용 |
|---|------|---------|
| C | **Elsevier 템플릿 재활용** | RESS `main.tex` (elsarticle) → AEI 저널명·ISSN만 교체; elsarticle.cls 호환 |
| D | **Abstract 단어 수 제한** | ✅ 194단어 (RESS 기준 완료) — AEI 제한 별도 확인 필요 |
| E | **Highlights 파일** | ✅ `Submission/RESS/highlights.txt` 기존 파일 → AEI 내용에 맞게 수정 |
| F | **Generative AI 선언문** | ✅ 기존 `declaration_ai_use.txt` 재활용 가능 — AI 사용 여부 최종 확인 |
| G | **CRediT 저자 기여도** | ✅ Elsevier 공통 양식 — 기존 내용 재활용 |
| H | **이해충돌 선언문** | ✅ Elsevier 공통 양식 — 기존 내용 재활용 |
| I | **연구비 기재** | ✅ RS-2026-25621690 (MSIT) — 기존 Funding 섹션 재활용 |
| J | **데이터 가용성 선언** | ✅ T9에서 이미 추가됨 — 재활용 |

### 💡 권장 — 수락률 향상

| # | 항목 | 조치 내용 |
|---|------|---------|
| K | **[30] AEI 선행연구와 명시적 비교** | §V Discussion에서 [30]과의 방법론 차이 및 우위를 AEI 리뷰어 관점에서 명확히 서술 |
| L | **추천 리뷰어 준비** | PHM/data-driven prognostics 분야 AEI 게재 경험 있는 전문가 2–3인; [30] 저자 제외 |
| M | **§V.F 3계층 → engineering informatics 체크리스트 재프레이밍** | AEI 스코프 언어("engineering decision-making", "knowledge-intensive engineering tasks")로 정렬 |

---

## 7. Decision Log

| Date | Action | Outcome |
|------|--------|---------|
| 2026-07-06 | ETRI Journal 적합성 검토 | 불가 — 스코프 미스매치, PHM 선례 없음 |
| 2026-07-06 | IEEE TIE 1순위 선정 (초기 결정) | 임시 결정 |
| 2026-07-06 | Cover letter 초안 작성 | `Cover_Letter/Cover_Letter_draft.md` (TIE 기준) |
| 2026-07-07 | IEEE TIE 투고 규정 검토 | **TIE 불가** — 하드웨어 실험 요건 미충족; 데스크 리젝션 위험 높음 |
| 2026-07-07 | 페이지 제한 정정 | TIE 최초 제출 10페이지 (구정보 8페이지는 오류) |
| 2026-07-07 | IEEE TII 투고 규정 검토 | 스코프·하드웨어 요건 모두 적합; double-blind 처리 필요 |
| 2026-07-07 | IEEE Access 투고 적합성 검토 | 백업으로 적합; IF·가시성은 TII 대비 낮음 |
| 2026-07-07 | **최종 확정** | **TII 1순위, IEEE Access 2순위** |
| 2026-07-07 | Cover letter TII 버전으로 수정 | `Cover_Letter/Cover_Letter_draft.md` 업데이트 |
| 2026-07-07 | TII 공식 체크리스트 항목 확인 및 반영 | 22개 항목 검토 완료; 🚨 기관 이메일(ETRI) 미설정, 페이지 수 미측정, 선언문 미작성 등 제출 전 필수 작업 확인 |
| 2026-07-08 | 레퍼런스 45개 기반 타깃 저널 재분석 | RESS(4건 인용)→2순위, AEI(2건 인용)→3순위로 추가; 투고 전략 TII→RESS→AEI→Access 4단계로 확장 |
| 2026-07-09 | TII Virtual Review Phase 1+2 완료 — TII 수락률 최종 평가 | 32–38% 한계 확인. N-CMAPSS 미착수·UQ 미해결 상태로 TII Critical 이슈 미해소. Phase 3 미진행 결정 |
| 2026-07-09 | **목표 저널 변경: TII → RESS 1순위 확정** | RESS 예상 수락률 42–52%; CMAPSS 딥러닝 홈 저널; 레퍼런스 4건 기게재; 페이지 제한 없음; 즉시 투고 가능 |
| 2026-07-09 | 투고 전략 3단계로 축소 | RESS → AEI → IEEE Access (TII 제외) |
| 2026-07-09 | §4b RESS 스코프 적합성 업데이트 | BH-FDR 차별화 포인트 과장 제거; H2 reliability risk 프레이밍 및 3계층 체크리스트 항목으로 교체 |
| 2026-07-09 | §6 Pre-Submission Action Items RESS 기준으로 전면 교체 | TII 특화 항목(IEEE 이메일·Double-blind·10p) 삭제; Elsevier/RESS 특화 항목으로 대체 |
| 2026-09-09 | RESS 투고 완료 | Elsevier Editorial Manager 통해 투고; 리뷰 대기 |
| 2026-09-23 | **RESS 데스크 리젝션 수신** | 에디터 사유: ① "signal processing/fault diagnosis are no longer included in the scope" ② "missing a significant body of recent literature" → 투고 전략 전면 재검토 |
| 2026-09-23 | MSSP, ESWA 신규 후보 조사 | MSSP IF=10.07(Q1), ML 가이드라인 있음, simulation-only 리스크 확인; ESWA IF≈7–9.4(Q1), PHM/CMAPSS 선례 확인 |
| 2026-09-23 | AEI IF 재확인 | IF ~8.0 → 11.5로 대폭 상승 (JCR 2026 기준) — 사실상 RESS 대비 동급 이상 |
| 2026-09-23 | **투고 전략 재확정** | **AEI 1순위 → ESWA 2순위 → MSSP 3순위 → IEEE Access 백업**; RESS·TIE 탈락 확정 |
| 2026-09-23 | **수락률 현실화 재검토** | 논문 기여 구조(방법론·ablation, null result H6/H7, CMAPSS 합성 데이터) 감안 시 AEI 수락 확률 15–20% 수준. **EAAI 1순위 재검토** 필요 — §8 참조 |

---

## 8. 현실적 수락 확률 기반 투고 전략 재검토 (2026-09-23)

> **배경:** RESS 데스크 리젝션 이후 스코프만이 아니라 논문 기여 수준 자체를 기준으로 수락 가능성을 재평가함.

### 논문 기여 구조의 현실적 평가

#### 강점

| 항목 | 내용 |
|------|------|
| 통계적 엄밀성 | BH-FDR 교정 + 5-seed 반복 설계 — 동급 CMAPSS 논문 중 명확한 차별점 |
| 실험 구조 | 4개 데이터셋 × 4요인 독립 통제 — 설계 자체는 탄탄 |
| 정직한 보고 | 65.8% 오류 자체 수정 및 재현성 기여 — 방법론 논문으로서의 신뢰도 |

#### 약점 (솔직한 평가)

| 약점 | 설명 |
|------|------|
| 주요 발견이 "이미 알려진 것의 재확인" | clip=125 표준, Fleet MinMax 기본 선택 — PHM 실무자에게 새로운 정보가 아님 |
| H6 핵심 주장이 null result | M3가 올바르게 훈련된 M0 대비 유의미한 성능 향상 없음 → "novel architecture"로 포지셔닝 어려움 |
| H7 완전 null result | 손실함수 전부 유의미하지 않음 → 기여가 "부정적 결과의 체계적 확인"에 그침 |
| CMAPSS 한계 | 합성 데이터셋 → Q1 고IF 저널 리뷰어 기대치와 갈수록 격차 |
| LSTM backbone | RMSE 수치 경쟁에서 SOTA 대비 열위 — 성능 주도 논문이 아님을 방어해야 함 |

> **핵심 문제:** 논문의 실질적 메시지는 "clip 먼저 잡고, Fleet MinMax 써라, M1은 쓰지 마라" — 방법론적으로 정직하고 유용하지만, IF 10+ 저널이 기대하는 novelty 수준과 간극이 있다.

---

### 저널별 수락 확률 현실적 추정

| 저널 | IF | 예상 수락률 | 평가 |
|------|----|-----------|------|
| AEI | 11.5 (Q1) | **15–20%** | 야심적. AEI 리뷰어는 novel architecture + 성능 선도를 기대. Null result H6/H7이 "기여가 뭔가?" 반응 유발 가능 |
| **EAAI** *(Engineering Applications of AI)* | **7.8 (Q1)** | **35–45%** | **실질적으로 가장 적합.** AI 방법론 + 공학 응용, ablation/methodology 논문 다수 게재. Elsevier이므로 elsarticle 재활용 가능 |
| ESWA | 7–9.4 (Q1) | **35–45%** | EAAI와 유사. 스코프 넓어 scope 거절 위험 낮음. CMAPSS 선례 있음 |
| IEEE Transactions on Reliability | 5.0 (Q1) | **30–40%** | PHM 정중앙. "신뢰성 설계 지침" 논문으로 완전히 리프레이밍 시 fit 좋음. IF는 낮지만 독자층 정합성 높음 |
| IEEE Access | 4.2 (Q2) | **60–70%** | 안전하지만 IF 희생 |

---

### 수락률 기반 최종 추천 투고 순서

```
1순위: EAAI — Engineering Applications of Artificial Intelligence (IF 7.8, Q1, Elsevier)
  → "AI-enhanced engineering methodology" 스코프 정중앙
  → Ablation study + null result 논문에 관대한 편집 문화
  → Elsevier elsarticle 템플릿 재활용 가능
  → 저널명·ISSN 교체만으로 현재 main_revision.tex 재활용 가능

2순위: ESWA — Expert Systems with Applications (IF ~8, Q1, Elsevier)
  → EAAI 탈락 시 동일 전략으로 즉시 재투고 가능

3순위: IEEE Transactions on Reliability (IF 5.0, Q1, IEEE)
  → "RUL 예측 파이프라인의 신뢰성 설계 체계" 각도로 완전 리프레이밍 필요
  → APC 및 IEEE 형식 변환 비용 감안

백업: IEEE Access (IF 4.2)
```

> **AEI 포지션:** 완전히 배제하지는 않으나, RESS 리젝션 직후 동급 Elsevier Q1 저널 재도전은 기여 수준 문제를 해결하지 않은 채 같은 리스크 반복. **문헌 보강 + 프레이밍 개선이 충분하다고 판단될 경우에만 1순위로 유지.**

---

## 9. EAAI · ESWA 투고 가이드라인 상세 검토 (2026-09-23)

> **배경:** §8의 현실적 수락률 분석에서 EAAI를 1순위로 확정함에 따라, EAAI 및 차순위 ESWA의 실제 투고 규정을 PDF 원문 기준으로 정밀 검토. 원본 파일 위치: `Manuscript/Manuscript_Guideline/`

---

### 9-A. EAAI (Engineering Applications of Artificial Intelligence)

**기본 정보**
- ISSN: 0952-1976 / Publisher: Elsevier / IF: ~7.8 (Q1) / IFAC 공식 저널
- 심사 방식: **Double anonymized** (저자·심사자 상호 익명)
- 제출 시스템: Elsevier Editorial Manager

#### 분량·형식 제한

| 항목 | 기준 | 현재 원고 상태 |
|------|------|--------------|
| **전체 페이지** | **최대 50페이지 (desk rejection)** | ⚠️ preprint 12pt = 58p, 11pt = 56p — **6~8페이지 초과** |
| 파일 크기 | 100 MB 이하 | ✅ 문제 없음 |
| Abstract | ≤250 단어 | ✅ ~240단어 |
| Keywords | **1–6개** | 🔴 현재 7개 → 1개 삭제 필요 |
| Highlights | 3–5개 bullet, 각 85자 이하 | ⚠️ 미작성 |
| 레퍼런스 형식 | 제출 시 어떤 일관된 형식도 허용 (게재 후 Harvard로 재포맷) | ✅ 현재 numbered 형식 그대로 제출 가능 |
| 컬럼 형식 | 단일 컬럼 필수 (LaTeX review/preprint 모드 = 단일 컬럼) | ✅ 현재 `[review,12pt]` = 단일 컬럼 |

#### 🔴 Desk Rejection 4대 조건 (미충족 시 심사 미진입)

| 조건 | 현재 상태 |
|------|---------|
| ① 비유적 메타휴리스틱 논문 아닐 것 | ✅ 해당 없음 |
| ② **Abstract에서 AI 기여와 공학 응용을 명확히 구분** | 🔴 **현재 미구분 → abstract 수정 필요** |
| ③ 제목·abstract 내 미정의 약어 금지 | 🔴 **CMAPSS·FDR·N1–N3 미정의** |
| ④ 단일 컬럼 형식 | ✅ |

#### Abstract 내 미정의 약어 목록

| 약어 | 현재 상태 | 처리 방법 |
|------|----------|---------|
| `RUL` | title에서만 정의, abstract 본문에서 재정의 없음 | abstract 첫 등장 시 "Remaining Useful Life (RUL)" 재정의 |
| `CMAPSS` | 미정의 | "C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)" |
| `FDR` | "BH FDR-corrected"에서만 등장 | "False Discovery Rate (FDR)" |
| `N1–N3` | 미정의 | 풀어쓰거나 삭제 |

#### 제출 전 필수 준비사항 (Desk Rejection 외)

| 항목 | 설명 |
|------|------|
| **Double anonymized 파일 분리** | Title page(저자 정보) + 익명 원고 별도 파일 제출 필수 |
| **Research data — Option C 의무** | 코드(GitHub/Zenodo DOI) 등록 + 원고 내 Data Availability Statement 추가 |
| **AI 사용 선언 섹션** | References 직전에 "Declaration of Generative AI…" 별도 섹션 추가 |
| **CRediT Author Contributions** | 필수 (Conceptualization, Formal analysis, Writing 등) |
| **Graphical abstract** | 권장 (531×1328 px) |

#### 스코프 적합성

- 핵심 범위: "Intelligent fault detection, fault analysis, diagnostics and monitoring" + "Deep learning and real-world applications" → **직접 부합**
- 공개 데이터셋 기반 재현성 실증 요구 → NASA CMAPSS ✅

#### EAAI 페이지 문제 해결 방안

| 방안 | 예상 페이지 | 비고 |
|------|-----------|------|
| preprint 10pt 제출 | **48p ✅** | 가독성 다소 저하 |
| 콘텐츠 트리밍 (~6p 절감) | **~50p ✅** | Discussion 압축, 부록 이동 등 |

---

### 9-B. ESWA (Expert Systems with Applications)

**기본 정보**
- ISSN: 0957-4174 / Publisher: Elsevier / IF: ~7.0–9.4 (Q1, JCR 연도별 차이)
- 심사 방식: **Double anonymized**
- 제출 시스템: Elsevier Editorial Manager

#### 분량·형식 제한

| 항목 | 기준 | 현재 원고 상태 |
|------|------|--------------|
| **전체 페이지** | **명시 없음** | ✅ 56p도 문제 없음 |
| Abstract | ≤250 단어 | ✅ |
| Keywords | **1–7개** | ✅ 현재 7개 그대로 가능 |
| Highlights | 3–5개 bullet, 각 85자 이하 | ⚠️ 미작성 |
| **레퍼런스 형식** | **APA 7판 필수** (author-year, 알파벳 정렬) | 🔴 **현재 numbered 형식 → 54개 전체 재변환 필요** |
| 컬럼 형식 | 단일 컬럼 (Word); LaTeX은 두 컬럼도 허용 | ✅ |

#### ESWA APA 레퍼런스 형식 예시

```
현재 형식 (numbered, in-text):
  "...LSTM network [5]..."

ESWA 필요 형식 (APA, in-text):
  "...LSTM network (Zheng et al., 2017)..."

현재 형식 (bibliography):
  [5] S. Zheng et al., "Long Short-Term Memory Network...", Proc. ICPHM, 2017.

ESWA 필요 형식 (APA bibliography):
  Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). Long Short-Term
  Memory Network for Remaining Useful Life Estimation. Proceedings of ICPHM, 88–95.
```

> ⚠️ **작업량 주의:** 현재 `main_revision.tex`는 수동 `\bibitem{ref01}` + `\bibliographystyle{elsarticle-num}` 방식. ESWA 투고 시 54개 레퍼런스 전체를 APA 형식으로 재작성 + 본문 내 모든 `\cite{ref##}` 호출을 author-year 방식으로 변경해야 함.

#### Desk Rejection 조건

ESWA는 EAAI와 달리 **명시적 desk rejection 조건을 제시하지 않음** → 심사 진입 가능성이 더 높음.

#### 제출 전 필수 준비사항

| 항목 | EAAI와 동일 여부 |
|------|--------------|
| Double anonymized 파일 분리 | 동일 필수 |
| Research data — Option C 의무 | 동일 필수 |
| AI 사용 선언 섹션 | 동일 필수 |
| CRediT Author Contributions | **권장** (EAAI는 필수, ESWA는 "encouraged") |

#### 스코프 적합성

- 핵심 범위: "Expert/intelligent systems in engineering; neural networks; data mining; knowledge discovery"
- PHM/RUL 논문 다수 게재 확인, CMAPSS 선례 있음
- 단, methodology/ablation 중심 논문보다 application 중심 논문을 더 선호하는 경향 → 스코프 적합도는 EAAI보다 약간 낮음

---

### 9-C. EAAI vs ESWA 종합 비교

| 항목 | EAAI | ESWA |
|------|------|------|
| IF | ~7.8 (Q1) | ~7–9.4 (Q1) |
| **페이지 제한** | **50p (desk rejection)** — 현재 초과 | **없음** ✅ |
| **레퍼런스 형식** | 제출 시 어떤 형식도 가능 ✅ | **APA 필수** — 54개 재변환 필요 🔴 |
| **Desk rejection 조건** | 4가지 명시 (abstract 구분 등) | 없음 ✅ |
| Keywords 제한 | 최대 6개 (1개 삭제 필요) | 최대 7개 ✅ |
| 스코프 적합도 | ★★★★★ (fault detection/diagnostics 직결) | ★★★★☆ (engineering AI 응용) |
| 심사 방식 | Double anonymized | Double anonymized |
| Research data | Option C (의무) | Option C (의무) |
| CRediT | 필수 | 권장 |
| 제출 준비 핵심 작업 | **페이지 축소 (~6p) + abstract 수정** | **레퍼런스 54개 APA 재변환** |

#### 결론

**EAAI 1순위 유지.** 스코프 적합도가 높고 레퍼런스 재포맷 부담이 없음. Abstract 수정(AI/engineering 구분, 약어 정의)과 페이지 축소(10pt 제출 or ~6p 트리밍)가 핵심 준비 작업.

**ESWA 2순위.** EAAI 탈락 시 APA 레퍼런스 재변환 작업(54개)을 수행하고 재투고. 페이지 제한이 없어 분량 부담은 없음. 스코프 거절 위험도 낮음.
