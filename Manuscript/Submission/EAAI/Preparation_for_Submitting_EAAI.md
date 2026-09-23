# Preparation for Submitting to EAAI

**Target journal:** Engineering Applications of Artificial Intelligence (EAAI)
**ISSN:** 0952-1976 / Publisher: Elsevier (IFAC) / IF: ~7.8 (Q1)
**Created:** 2026-09-23
**Based on:** Guide for Authors PDF (`Manuscript/Manuscript_Guideline/`) + §9-A of `Searching_Target_Journal.md`

---

## 제출 전 필수 작업 7가지 (우선순위 순)

---

### ✅ Task 1 — Abstract 수정 〔DR-1 + DR-2〕

**Desk Rejection 위험 — 최우선 처리**

#### DR-1: AI 기여 vs. 공학 응용 구분 명시
EAAI 필수 요건: *"The abstract should clearly specify which is the contribution in AI and which is the application in engineering."*

현재 abstract는 이 구분이 없음. 아래 방향으로 수정:
- **AI contribution:** stacked LSTM 기반 controlled ablation 설계, BH-FDR 다중 비교 교정
- **Engineering application:** 터보팬 엔진 RUL 예측 파이프라인의 설계 인자 우선순위 결정

#### DR-2: Abstract 내 미정의 약어 해소
| 약어 | 현재 상태 | 수정 방향 |
|------|----------|---------|
| `RUL` | title에서만 정의, abstract 본문에서 재정의 없음 | 첫 등장 시 "Remaining Useful Life (RUL)" |
| `CMAPSS` | 미정의 | "C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)" |
| `FDR` | "BH FDR-corrected"에서만 등장 | "False Discovery Rate (FDR)" |
| `N1–N3` | 미정의 | 풀어쓰거나 삭제 |

**작업 파일:** `EAAI/Abstract_draft.md`
**단어 수 여유:** RESS 200단어 제한 → EAAI 250단어 허용 (50단어 여유)

---

### ✅ Task 2 — Keywords 1개 삭제 〔DR-3〕

**Desk Rejection 위험**

- EAAI 허용 범위: **최대 6개**
- 현재: 7개 (Remaining useful life prediction / Prognostics and health management / Turbofan engine / NASA CMAPSS / LSTM / Fault-mode gating / Normalization ablation)
- 삭제 후보: `Prognostics and health management` (제목 및 내용에서 이미 충분히 드러남) 또는 `Turbofan engine` (CMAPSS로 함의됨)

**작업 위치:** `main_EAAI.tex` → `\begin{keyword}` 블록

---

### ✅ Task 3 — 저널명 변경 〔DR-4〕

**Desk Rejection 위험**

`main_EAAI.tex` 1줄 수정:
```latex
% 변경 전
\journal{Reliability Engineering \& System Safety}

% 변경 후
\journal{Engineering Applications of Artificial Intelligence}
```

**작업 파일:** `Manuscript/Submission/EAAI/main_EAAI.tex` (RESS 버전 복사 후 작업)

---

### ✅ Task 4 — Highlights 작성 〔R-3〕

- 형식: 3–5개 bullet point, 각 **최대 85자** (공백 포함)
- 별도 파일로 제출 (파일명에 "highlights" 포함)
- 목적: 검색 엔진 노출 및 독자 첫인상

**초안 방향 (85자 제한 내):**
1. Controlled four-factor ablation of turbofan RUL pipeline across all CMAPSS sub-datasets
2. Fleet min-max normalization significantly outperforms per-unit and RevIN strategies
3. GMM hard routing degrades RMSE by 76–156%; soft/attention routing avoids collapse
4. No custom loss improves over MSE after Benjamini-Hochberg FDR correction
5. Upstream label and normalization choices dominate downstream architecture and loss

**작업 파일:** `Manuscript/Submission/EAAI/highlights.txt` (신규 작성)

---

### ✅ Task 5 — Double Anonymized 파일 분리 〔R-1〕

EAAI는 **이중 익명 심사** (저자·심사자 상호 익명).

제출 파일 2종 필요:
| 파일 | 포함 내용 | 제외 내용 |
|------|---------|---------|
| **Title page** | 저자명, 소속, 이메일, ORCID, Acknowledgements, Competing interests | — |
| **익명 원고** | 본문 전체 + References + Tables | 저자명, 소속, Acknowledgements, 기관 식별 정보 |

> ⚠️ Supplementary materials도 저자 식별 정보 없어야 함.

**작업:** `main_EAAI.tex`에서 `\author`, `\affiliation`, `\ead`, `\fntext` 블록을 title page 파일로 분리

---

### ✅ Task 6 — Research Data 등록 + Data Statement 추가 〔R-2〕

EAAI Research Data **Option C 의무** — 미이행 시 게재 후 문제 발생 가능.

#### 필요 조치
1. **코드 공개:** `Data_Analysis/Code/` (H2–H4 + shared utils) → GitHub repository 생성 → Zenodo DOI 등록
2. **Data Availability Statement** 원고 내 추가:
   ```
   The NASA C-MAPSS dataset used in this study is publicly available at
   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/
   pcoe-public-data-set-repository/. Experimental code and result files are
   available at [Zenodo DOI: XX.XXXXX/zenodo.XXXXXXX].
   ```
3. **원고 본문 인용:** 코드 저장소를 reference로 추가하고 Methodology 섹션에서 인용

**작업 파일:** 원고 내 `\section*{Data Availability}` 섹션 추가

---

### ✅ Task 7 — AI 사용 선언 + CRediT 저자 기여도 〔R-4 + R-5〕

#### R-4: Generative AI 사용 선언 (필수)
References 바로 앞에 별도 섹션으로 추가:

```latex
\section*{Declaration of Generative AI and AI-assisted technologies
          in the manuscript preparation process}
During the preparation of this work the authors used Claude (Anthropic)
in order to assist with manuscript drafting, literature synthesis, and
code development. After using this tool, the authors reviewed and edited
the content as needed and take full responsibility for the content of
the publication.
```

#### R-5: CRediT 저자 기여도 (필수)
```latex
\section*{Author Contributions}
\textbf{Young Seog Yoon:} Conceptualization, Methodology, Software,
Formal analysis, Writing – original draft, Writing – review \& editing.
\textbf{Eun Seo Lee:} Investigation, Validation, Writing – review \& editing.
\textbf{Hyeontae Kim:} Data curation, Software, Visualization.
\textbf{Ji Yeon Son:} Project administration, Funding acquisition,
Supervision.
```
> ※ 실제 기여도는 저자 협의 후 확정 필요.

---

## 진행 상태 추적

| # | Task | 항목 | 상태 | 완료일 |
|---|------|------|------|--------|
| — | References_EAAI.md 수정 | Section Usage Map 교정 + v1.1 갱신 | ✅ 완료 | 2026-09-23 |
| 1 | Abstract 수정 | DR-1 (AI/engineering 구분) + DR-2 (약어 정의) | ✅ 완료 | 2026-09-23 |
| 2 | Keywords 1개 삭제 | DR-3 | ✅ 완료 | 2026-09-23 |
| 3 | 저널명 변경 | DR-4 | ✅ 완료 | 2026-09-23 |
| 4 | Highlights 작성 | R-3 | ✅ 완료 | 2026-09-23 |
| 5 | Double Anonymized 파일 분리 | R-1 | ✅ 완료 | 2026-09-23 |
| 6 | Research Data + Data Statement | R-2 | 🔶 일부완료 | 2026-09-23 (원고 내 DAS 삽입 완료; Zenodo DOI 등록은 별도 실행 필요) |
| 7 | AI 선언 + CRediT | R-4 + R-5 | ✅ 완료 | 2026-09-23 |

---

## 페이지 제한 관련 메모

| 컴파일 모드 | 페이지 수 | EAAI 50p 제한 |
|------------|---------|--------------|
| review 12pt (현재) | 82p | 초과 |
| preprint 12pt | 58p | 초과 (+8p) |
| preprint 11pt | 56p | 초과 (+6p) |
| **preprint 10pt** | **48p** | **✅ 통과** |

> 단기 해결: 10pt로 제출 (48p). 근본 해결: Discussion 압축 등 ~6p 트리밍.

---

## 참고 링크

- EAAI Guide for Authors PDF: `Manuscript/Manuscript_Guideline/Guide for authors - Engineering Applications of Artificial Intelligence...pdf`
- 가이드라인 상세 분석: `Manuscript/Submission/1. Target Journal/Searching_Target_Journal.md` §9-A
- RESS 원본 LaTeX: `Manuscript/Submission/RESS/Revision/main_revision.tex`
