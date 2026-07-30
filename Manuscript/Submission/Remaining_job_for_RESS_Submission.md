# RESS 투고 전 남은 작업 목록

> **작성일:** 2026-07-10  
> **투고 대상:** Reliability Engineering & System Safety (Elsevier)  
> **투고 시스템:** Elsevier Editorial Manager  
> **예상 수락률:** 45–60%  
> **현재 원고 상태:** Phase 1 + Phase 2 수정 완료 (TII Virtual Review 대응)

---

## 1. 사용자가 직접 처리해야 할 항목

### 1-1. 저자 정보 입력 (필수 — 투고 불가 상태)

| 파일 | 수정 내용 |
|------|---------|
| `RESS/main.tex` line 26 | `[Author Name]` → 실명 |
| `RESS/main.tex` line 27 | `spiceyoon@gmail.com` → ETRI 기관 이메일 |
| `RESS/credit_author_statement.txt` line 1 | `[Author Name]` → 실명 |
| `Cover_Letter/Cover_Letter_RESS_draft.md` | `[Author Name]`, `[Position/Title]` → 실명·직함 |

### 1-2. Funding 섹션 확인 (필수)

`RESS/main.tex` line 533:

```latex
\section*{Funding}
This research received no specific grant from any funding agency in the public,
commercial, or not-for-profit sectors.
```

- ETRI 기관 연구비(과제번호)가 있는 경우 → 기관명 + grant number로 교체
- 수령한 연구비가 없는 경우 → 현 문구 유지

### 1-3. AI 사용 선언 최종 결정 (필수)

`RESS/declaration_ai_use.txt` 에 두 가지 문구가 모두 있음. 하나를 선택:

- **AI 사용 공개 (현재 선택된 문구):** "Claude (Anthropic)을 원고 초안, 언어 편집, 문헌 검토에 사용"
- **AI 미사용 선언:** "생성형 AI 또는 AI 보조 기술을 사용하지 않았다"

→ 실제 작성 과정 기준으로 최종 결정 후 불필요한 문구 삭제

### 1-4. 참고문헌 "(Authors not retrieved)" 보완 (권장)

`RESS/main.tex`의 아래 항목들은 저자 미확인 상태:

- ref30, ref31, ref32, ref33 (line 602–608)
- ref34, ref35, ref36 (line 610–614)

→ DOI 또는 arXiv 링크로 실제 저자명 확인 후 교체 권장

### 1-5. elsarticle.cls 다운로드 (필수 — LaTeX 컴파일 불가)

- 현재 `RESS/` 폴더에 `elsarticle.cls` 없음
- Elsevier 공식 배포: [https://www.ctan.org/pkg/elsarticle](https://www.ctan.org/pkg/elsarticle)
- 다운로드 후 `Manuscript/Submission/RESS/elsarticle.cls`로 배치

### 1-6. Elsevier Editorial Manager 투고

- Editorial Manager 접속 → 신규 제출 → RESS 선택
- 업로드 파일: main.tex + 그림 파일들 (PDF는 시스템이 자동 생성)
- 추천 리뷰어 2–3인 입력 (PHM/RESS 게재 경험자)

---

## 2. Claude가 처리할 수 있는 항목

### 2-1. highlights.txt Bullet 2 수정 (85자 제한 초과)

현재: `"FD003 variance anomaly traces to latent fault-mode heterogeneity, not normalization"` = 83자 ✓  
→ 이미 `highlights.txt` 내 수정 제안 메모 있음. 실제 파일 반영 필요.

### 2-2. Figures 복사 (RESS 폴더에 그림 없음)

`Manuscript/Figures/` → `Manuscript/Submission/RESS/` 복사 필요:

- Fig1_degradation_trends.png
- Fig2_rul_clipping.png
- Fig3_H2_clipping_rmse.png
- Fig4_H5_normalization_rmse.png
- Fig5_H5_statistical_test.png
- Fig6_H6_gmm_clustering.png
- Fig7_H6_model_comparison.png
- Fig8_H7_clip_loss_interaction.png

### 2-3. 표 번호 최종 검증

`main.tex` 내 Table I~VII 참조 일관성 확인 필요  
(Phase 1에서 Table II 신규 추가 → 기존 Table III~V가 IV~VII로 밀렸을 가능성)

### 2-4. manuscript_full_text.md ↔ main.tex 동기화 확인

- `manuscript_full_text.md` 수정 시각: 2026-07-09 18:50
- `main.tex` 수정 시각: 2026-07-09 18:45 (5분 먼저)
- 마지막 5분간 변경 사항이 main.tex에 반영됐는지 확인 필요

---

## 3. 투고 파일 패키지 현황

| 파일 | 상태 |
|------|------|
| `RESS/main.tex` | ✅ 있음 (플레이스홀더 교체 필요) |
| `RESS/highlights.txt` | ⚠️ Bullet 2 수정 필요 |
| `RESS/declaration_ai_use.txt` | ⚠️ 최종 선택 필요 |
| `RESS/conflict_of_interest.txt` | ✅ 있음 |
| `RESS/credit_author_statement.txt` | ⚠️ [Author Name] 교체 필요 |
| `Cover_Letter/Cover_Letter_RESS_draft.md` | ⚠️ [Author Name] 교체 필요 |
| `RESS/elsarticle.cls` | ❌ 없음 (다운로드 필요) |
| `RESS/Fig1~8.png` | ❌ 없음 (복사 필요) |
| `.bib` 파일 | ✅ 불필요 (참고문헌 inline) |

---

## 4. 투고 준비 순서 (권장)

```
1. elsarticle.cls 다운로드 → RESS/ 배치
2. Figures 복사 (RESS/Fig1~8.png)
3. [Author Name] / 이메일 / Funding 교체
4. declaration_ai_use.txt 최종 결정
5. 참고문헌 "(Authors not retrieved)" 보완
6. LaTeX 컴파일 → PDF 확인 (표 번호, 그림 위치 검토)
7. Cover Letter 최종화
8. Editorial Manager 제출
```
