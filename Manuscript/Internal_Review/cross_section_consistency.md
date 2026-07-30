# Cross-Section Consistency Review
> Reviewed: 2026-07-06
> Sections examined: Abstract_draft, Introduction, Methodology, Results, Discussion_Implication, Conclusion

---

## Summary

5개의 실질적 불일치(factual / terminological inconsistency)가 발견되었으며, 모두 이 검토 내에서 수정 완료되었다. 구조적 불일치(섹션 간 논리 단절)는 없으며, 세 가지 주의 사항(caution)은 현재 상태를 유지한다.

---

## CRITICAL — 수정 완료

### C1. Abstract 표준편차 수치 오류
- **위치:** Abstract_draft.md
- **오류:** "RMSE std = 12.86 vs. ≤ 1.68 on other datasets"
- **근거:** Introduction §(ii), Results §B, Discussion §V.C 모두 "≤ 1.84"로 기술
- **수정:** `≤ 1.68` → `≤ 1.84`

### C2. Abstract NASA Score 증폭 배수 과장
- **위치:** Abstract_draft.md
- **오류:** "three to six orders of magnitude" (3~6 자릿수 증폭)
- **근거:** Results §A: FD004 = 35.4× (≈1.5 자릿수), FD003 = 306,000× (≈5.5 자릿수). 하한이 "three"가 아닌 약 "two"에 해당
- **수정:** "three to six orders of magnitude" → "up to six orders of magnitude (35-fold on FD004, 306,000-fold on FD003)"

### C3. Methodology §III.J — 통계 검정 방향 혼재
- **위치:** Methodology.md §III.J
- **오류:** "one-sided Wilcoxon rank-sum tests" — 범용 서술이지만 H2는 two-sided(§III.K)
- **수정:** H2에 대해 two-sided임을 명시; H5/H6/H7에 대해서는 one-sided(improvement) 유지

### C4. Methodology §III.J — 표본 크기 N 기술 오류
- **위치:** Methodology.md §III.J
- **오류:** "per-engine RMSE values pooled across five seeds (N = 5 × |engines|)"
- **근거:** 코드 검증(Code_Verification_Report.md Issue B) 확인: H7은 per-seed NASA Score (N=5); H2는 per-engine RMSE (N≈100–259)
- **수정:** 가설별로 N을 명시

### C5. Results §A — H2 검정 명칭 불일치
- **위치:** Results.md §A
- **오류:** "Wilcoxon rank-sum, p = 0.97" — H2는 실제로 Mann-Whitney U (two-sided)
- **근거:** Methodology §III.K에서 corrected
- **수정:** "rank-sum (Mann-Whitney U, two-sided)" 또는 "Mann-Whitney U"로 통일

---

## CAUTION — 현상 유지 (수정 불필요)

### P1. H5 vs H6 FD003 베이스라인 수치 차이
- H5 N1/FD003 RMSE = 19.05 ± 12.86, H6 M0/FD003 RMSE = 43.23 ± 0.18
- 언뜻 같은 실험처럼 보이지만 코드 검증에서 4가지 설계 차이(backbone, features, val split, clipping)가 확인됨
- Results §B에 parenthetical note 추가됨; Methodology Table II, Discussion §V.G(Limitation 5)에 설명 완료
- **판단:** 추가 수정 불필요

### P2. 총 실험 횟수 표현
- Abstract: "over 800 training runs"
- Abstract 초안 노트: 실제 ≈1,055회, 보수적 표현으로 의도 기술
- **판단:** 의도된 conservative phrasing — 수정 불필요

### P3. M3 std=1.32 해석 문장 (Results §C.2)
- "M0 (std = 0.18 is low because M0 consistently fails at the same level; M3 std reflects one outlier seed...)" — 다소 복잡한 parenthetical
- 사실관계는 정확하나 가독성 문제가 있음
- **판단:** Abstract에서 M0 std를 언급하지 않으므로 section 내에서만 발생 — 의미 전달은 충분

---

## 적용된 수정 목록 (이 리뷰 결과)

| # | 파일 | 변경 전 | 변경 후 |
|---|------|--------|--------|
| C1 | Abstract_draft.md | `≤ 1.68` | `≤ 1.84` |
| C2 | Abstract_draft.md | "three to six orders of magnitude" | "up to six orders of magnitude (35-fold on FD004, 306,000-fold on FD003)" |
| C3 | Methodology.md §III.J | "one-sided ... tests" (universal) | "one-sided for H5/H6/H7; two-sided for H2 (see §III.K)" |
| C4 | Methodology.md §III.J | "N = 5 × \|engines\|" (universal) | H2: per-engine N≈100–259; H5/H6/H7: per-seed aggregate N=5 |
| C5 | Results.md §A | "Wilcoxon rank-sum" | "Mann-Whitney U (two-sided)" |
