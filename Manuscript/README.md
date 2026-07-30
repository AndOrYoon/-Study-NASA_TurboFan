# Manuscript — IEEE TII 제출 초안

**현재 상태: 원고 완성 — IEEE TII 제출 준비 중**

---

## 투고 대상 저널

**IEEE Transactions on Industrial Informatics (TII)**

| 항목 | 내용 |
|------|------|
| 투고 방식 | IEEE Author Portal (더블블라인드) |
| 페이지 제한 | 10페이지 (2단 IEEE 포맷) |
| 심사 방식 | Double-blind peer review |
| 선정 근거 | Industrial informatics 중심성 (PHM + 산업 배포 가이드); TIE는 하드웨어 검증 요구 → CMAPSS 시뮬레이션 부적합 |

---

## 논문 제목

**From Fleet Normalization to Fault-Mode Gating: A Cross-Dataset Ablation Study of Turbofan Remaining Useful Life Prediction**

---

## 폴더 구조

```
Manuscript/
├── Sections/                            ← 섹션별 개별 초안 (마스터 소스)
│   ├── Abstract_draft.md               (v1.1, TII 타깃)
│   ├── Introduction.md                 (v1.0, I.)
│   ├── Methodology.md                  (v1.0, III.)
│   ├── Results.md                      (v1.0, IV.)
│   ├── Discussion_Implication.md       (v1.0, V.)
│   └── Conclusion.md                   (v1.0, VI.)
├── References.md                        ← 참고문헌 45개, 섹션별 사용 지도 포함
├── Citation/                            ← ref01–ref45 개별 RIS 파일 + fetch 스크립트
├── Tables_Figures.md                    ← Figure 1–8 / Table 1–5 캡션 (영문)
├── Full-Text_Manuscript/
│   ├── manuscript_full_text.md         ← 통합 완성 원고 (Compiled 2026-07-06)
│   ├── manuscript_draft(TII Submission).docx  ← 제출용 DOCX (더블블라인드)
│   ├── build_tii_submission.py         ← MD → IEEE DOCX 변환 스크립트
│   ├── ieee_template_base.docx         ← IEEE 스타일 템플릿 기반
│   └── md_to_docx.py                   ← 범용 변환 스크립트
├── Cover_Letter/
│   ├── Cover_Letter_draft.md           ← 커버레터 v3.0 (TII 특화)
│   ├── Cover_letter(IEEE_TII).docx     ← 제출용 커버레터 DOCX
│   └── convert_cover_letter.py         ← 커버레터 변환 스크립트
├── Figures/                             ← 최종 Figure 1–8 (manuscript용)
├── Tables/                              ← Table 1–5 CSV
├── Manuscript_Guideline/
│   └── Manuscript_Guideline.md         ← TII 투고 가이드라인 및 체크리스트
├── Pre-Review/
│   ├── Pre-Review_Report.md            ← 내부 사전 리뷰 (3인, Major Revision)
│   ├── Pre-Review_Response.md          ← 리뷰 대응 정리
│   ├── Code_Verification_Report.md     ← 코드-원고 일치성 검증
│   └── Revision_Changelog.md           ← 수정 이력
├── Internal_Review/
│   └── cross_section_consistency.md    ← 섹션 간 수치·용어 일관성 검토 (2026-07-06)
└── Submission/
    └── 1. Target Journal/
        └── Searching_Target_Journal.md ← 저널 선정 분석
```

**섹션 번호 규칙:** II (Related Work)는 I과 V에 통합 → 섹션 번호 I, III–VI

---

## 완성 상태

| 구성 요소 | 상태 | 파일 |
|----------|------|------|
| Abstract | ✅ 완성 (v1.1) | `Sections/Abstract_draft.md` |
| I. Introduction | ✅ 완성 (v1.0) | `Sections/Introduction.md` |
| III. Methodology | ✅ 완성 (v1.0) | `Sections/Methodology.md` |
| IV. Results | ✅ 완성 (v1.0) | `Sections/Results.md` |
| V. Discussion & Implications | ✅ 완성 (v1.0) | `Sections/Discussion_Implication.md` |
| VI. Conclusion | ✅ 완성 (v1.0) | `Sections/Conclusion.md` |
| References (46개) | ✅ 완성 | `References.md` |
| Figures (8개) | ✅ 생성 완료 | `Figures/` |
| Tables (5개) | ✅ 생성 완료 | `Tables/` |
| 통합 원고 MD | ✅ 완성 | `Full-Text_Manuscript/manuscript_full_text.md` |
| 제출용 DOCX | ✅ 생성 완료 | `manuscript_draft(TII Submission).docx` |
| 커버레터 DOCX | ✅ 생성 완료 | `Cover_letter(IEEE_TII).docx` |
| 섹션 간 일관성 검토 | ✅ 완료 (5개 수정) | `Internal_Review/cross_section_consistency.md` |

---

## 주요 연구 기여

| 기여 | 핵심 수치 |
|------|---------|
| (i) RUL 클리핑 cross-dataset 검증 | clip=125 최적; clip=None → FD003 NASA Score 4,014,723 |
| (ii) 정규화 전략 체계적 ablation | Fleet MinMax 최우수; FD003 std=12.86 = 고장 모드 신호 |
| (iii) M3 Attention Gate 아키텍처 | FD003 RMSE −65.8% (14.78±1.32 vs 43.23±0.18), K=5사이클로 충분 |
| (iv) 3계층 설계 계층구조 | 레이블 엔지니어링 > 아키텍처 > 손실 함수 |
| (v) 손실 함수 null 결과 | BH-FDR 보정 96개 비교 — MSE 대비 유의한 개선 없음 |

---

## 제출 전 잔여 항목

| # | 항목 | 비고 |
|---|------|------|
| 🚨 A | ETRI 기관 이메일로 IEEE Author Portal 계정 등록 | Gmail 미허용 |
| B | Word에서 10페이지 이내 확인 (IEEE 2단 포맷) | — |
| C | 커버레터 `[Author Name]`, `[Position/Title]` 기입 | — |
| D | References [30]–[36] 저자 정보 보완 (현재 "Authors not retrieved") | — |
| E | COI(이해충돌) 선언 준비 | IEEE 양식 |
| F | 데이터 가용성 선언 (NASA CMAPSS 공개 URL 추가) | — |
| G | 현 TII Editor-in-Chief 이름 확인 → 커버레터 수신인 수정 | ieee-ies.org |

---

## 파일 재생성 명령

```powershell
# 가상환경 활성화
..\NASA_TurboFan\Scripts\activate

# 제출용 DOCX 재생성 (MD → IEEE 2단 포맷)
python Full-Text_Manuscript\build_tii_submission.py

# 커버레터 DOCX 재생성
python Cover_Letter\convert_cover_letter.py
```
