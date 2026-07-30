# Target Journal Search — Turbofan RUL Paper

**Paper title:** From Fleet Normalization to Fault-Mode Gating: A Cross-Dataset Ablation Study of Turbofan Remaining Useful Life Prediction
**Last updated:** 2026-07-09
**Author affiliation:** Electronics and Telecommunications Research Institute (ETRI)

---

## 0. Submission Strategy (Updated 2026-07-09)

TII Virtual Review 결과(예상 수락률 32–38%) 및 RESS 전환 검토 결과, **RESS를 1순위로 변경**. TII는 N-CMAPSS 부재·현대 backbone 미검증·UQ 부재 등 Critical 이슈가 잔존하며 현재 원고로 수락률 상한이 낮음. RESS는 CMAPSS 딥러닝 논문의 홈 저널로 레퍼런스 4건이 이미 게재되어 있고, 페이지 제한이 없어 현재 원고를 압축 없이 투고 가능. 예상 수락률 42–52%.

| Priority | Journal | IF | Quartile | 레퍼런스 인용 | Rationale |
|----------|---------|-----|----------|------------|-----------|
| **1순위** | **Reliability Engineering & System Safety (RESS)** | ~7.2 | Q1 | **4건** ([6],[7],[10],[11]) | CMAPSS 딥러닝 홈 저널; 페이지 제한 없음; PHM·신뢰성 스코프 최적; 예상 수락률 42–52% |
| **2순위** | **Advanced Engineering Informatics (AEI)** | ~8.0 | Q1 | **2건** ([30],[39]) | H6 최근접 선행연구([30])와 같은 저널; 엔지니어링+AI 융합 |
| **백업** | **IEEE Access** | 4.2 | Q2 | 1건 ([42]) | 빠른 게재, 스코프 거절 위험 없음; IF 낮음 |
| ~~검토 완료~~ | ~~IEEE TII~~ | ~~12.3~~ | ~~Q1~~ | ~~0건~~ | ~~수락률 32–38% 한계; N-CMAPSS·UQ 미해결로 미선택~~ |

```
투고 순서: RESS → (탈락) → AEI → (탈락) → IEEE Access
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

## 4b. Reliability Engineering & System Safety (RESS) — **1순위 ✅ (2026-07-09 확정)**

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

### Verdict: **H6 M3 novelty 강조 전략에서 TII·RESS 탈락 시 3순위**

---

## 5. 저널 최종 비교 (Updated 2026-07-09)

| 기준 | TIE | TII | **RESS** | **AEI** | IEEE Access | ETRI Journal |
|------|-----|---------|---------|---------|-------------|-------------|
| IF | 7.5 (Q1) | 12.3 (Q1) | **7.2 (Q1)** | 8.0 (Q1) | 4.2 (Q2) | 2.0 (Q3) |
| 스코프 적합 | ★★★☆☆ | ★★★★★ | **★★★★★** | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |
| 레퍼런스 인용 | 0건 | 0건 | **4건** | **2건** | 1건 | 0건 |
| 하드웨어 요건 | ❌ 필수 | ✅ 없음 | **✅ 없음** | ✅ 없음 | ✅ 없음 | ✅ 없음 |
| 페이지 제한 | 10p | 10p | **없음** | **없음** | **없음** | 별도 |
| 심사 방식 | Single | Double | **Single** | Single | Single | Single |
| 게재 속도 | 보통 | 보통 | **보통** | 보통 | **빠름** | 빠름 |
| 커뮤니티 가시성 | 높음 | **최고** | 높음 | 중간 | 중간 | 낮음 |
| 예상 수락률 | — | 32–38% | **42–52%** | — | — | — |
| 권장 순위 | ❌ | ~~검토 완료~~ | **1순위** | **2순위** | **백업** | ❌ |

---

## 6. Pre-Submission Action Items (RESS 기준, 2026-07-09 업데이트)

> TII → RESS 전환에 따라 제출 요건을 재정리. Elsevier Editorial Manager 기준.

### 🚨 제출 불가 — 즉시 조치 필요

| # | 항목 | 조치 내용 |
|---|------|---------|
| A | **Abstract + Introduction 재프레이밍** | "industrial informatics" → "reliability & PHM" 중심으로 재서술; RESS 리뷰어 첫인상 결정 |
| B | **Cover Letter RESS 버전 신규 작성** | TII 버전 전면 재작성 — RESS Editor-in-Chief 수신, CMAPSS PHM 관련성 강조 |
| C | **표 번호 충돌 해결** | §III.E Table III ↔ Results §IV Table III–V 번호 충돌 → 전체 재번호 필요 |

### ⚠️ 필수 — 제출 전 완료

| # | 항목 | 조치 내용 |
|---|------|---------|
| D | **Elsevier 템플릿 적용** | ✅ 완료 2026-07-09 — `build_ress_latex.py` 생성, `Submission/RESS/main.tex` 출력 (529줄, elsarticle review mode) |
| D1 | **Abstract 200-word 제한** | ✅ 완료 2026-07-09 — 265→194 단어로 트리밍; manuscript_full_text.md + Abstract_draft.md 업데이트 |
| D2 | **Highlights 파일** | ✅ 완료 2026-07-09 — `Submission/RESS/highlights.txt` 생성 (5 bullets, 각 ≤85자) |
| D3 | **Generative AI 선언문** | ✅ 완료 2026-07-09 — `Submission/RESS/declaration_ai_use.txt` 생성; AI 사용 여부 최종 확인 필요 |
| D4 | **전체 단어 수 확인** | ⚠️ RESS 최대 13,000 단어 — 현재 미측정; LaTeX 컴파일 후 wordcount 확인 필요 |
| E | **CRediT 저자 기여도 작성** | Elsevier 필수 항목 (Conceptualization, Methodology, Software, Writing 등 역할 명시) |
| F | **이해충돌 선언문 작성** | Elsevier 표준 COI 양식 작성 |
| G | **연구비 기재** | Funding 섹션 — 지원 없을 경우 "This research received no specific grant from any funding agency" |
| H | **데이터 가용성 선언** | 원고 내 Data Availability Statement 섹션 확인 (T9에서 이미 추가됨) |
| I | **Elsevier Editorial Manager 계정** | 제출 시스템 계정 생성 필요 |

### 💡 권장 — 수락률 향상

| # | 항목 | 조치 내용 |
|---|------|---------|
| J | **§V.F 3계층 → 신뢰성 설계 체크리스트 재서술** | "design hierarchy" → reliability engineering 관점 언어로 정렬 |
| K | **H2 reliability risk 프레이밍 복원** | clip=None → NASA Score 6 orders of magnitude = reliability failure risk; RESS에서 TII보다 강조 가능 |
| L | **추천 리뷰어 준비** | PHM/predictive maintenance 분야 RESS 게재 경험 있는 전문가 2~3인 |
| M | **T6: False routing 민감도 분석** (선택) | GatingNet 오분류 시 RMSE 변화 정량화; RESS 신뢰성 관점에서 TII보다 가치 높음 |

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
