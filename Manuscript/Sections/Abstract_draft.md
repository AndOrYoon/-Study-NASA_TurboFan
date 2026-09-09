# Abstract — RESS Submission

> **Draft status:** v1.7 — 2026-08-25 (GPT Rephrasing P2/P3: "establish"→"suggest", "each tier's failure impact"→"each tier produced qualitatively larger effects", gate confidence ≥0.8 threshold removed, "risk-prioritised"→"structured starting point")

**Title:**
From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction

---

## Abstract

Reliable Prognostic and Health Management (PHM) systems for turbofan engines embed interdependent design decisions — RUL label clipping, sensor normalization, fault-mode architecture, and training loss function — whose contributions to predictive reliability are rarely isolated. A controlled ablation across all four NASA CMAPSS sub-datasets (FD001–FD004) covers five clipping thresholds, seven normalization strategies, four fault-mode architectures, and seven loss functions, with all comparisons Benjamini-Hochberg FDR-corrected.

Fleet min-max normalization outperforms all per-unit and instance-level alternatives on FD001, FD002, and FD004 but exhibits anomalously high inter-seed variance on FD003 (RMSE std = 12.86 vs. ≤1.84 elsewhere), traced to co-existing HPC and fan fault modes rather than normalization failure. An attention-gate model (M3) routes engines to fault-specific branches from ten initial cycles, reducing FD003 RMSE by 65.8% (14.78 ± 1.32 vs. 43.23 ± 0.18) and NASA Score by 98.8%, while remaining immune to the test-time cluster collapse that degrades Gaussian Mixture Model (GMM) hard-routing by 75.4% on FD004. No custom loss achieves statistically detectable improvement over MSE after multiple-comparison correction (N = 5 seeds); removing RUL clipping inflates NASA prognostic scores by up to 306,000-fold regardless of loss design.

These findings suggest a three-tier design priority — label engineering, fault-mode architecture, loss function — where each tier produced qualitatively larger effects than the tier below it in the present experiments; gate misclassification analysis identifies gate confidence as an indicator of routing decisiveness, summarising priorities across the four CMAPSS sub-datasets as a structured starting point requiring validation on representative fleet data before operational deployment.

---

## Metadata

| Item | Detail |
|------|--------|
| Word count | ~250 words (RESS limit: 250; 2026-09-09 기준 투고 시스템 카운트 기준 최적화 완료) |
| Target venue | Reliability Engineering & System Safety (RESS) |
| Framing | Reliability risk analysis + controlled ablation + fault-mode PHM contribution |
| Primary claim | Fleet normalization beats per-unit; fault-mode gating explains residual variance |
| Secondary claim | Clipping and architecture dominate; loss function does not matter post-BH-FDR |
| Key number | −65.8% RMSE on FD003 (14.78 ± 1.32 vs. 43.23 ± 0.18) |

## Keywords (추천 5개 — 2026-09-09)

| 순위 | 키워드 | 선택 이유 |
|------|--------|----------|
| 1 | Remaining useful life prediction | 핵심 주제, 최다 검색 |
| 2 | Prognostics and health management | RESS 저널 scope 핵심 용어 |
| 3 | Turbofan engine | 응용 도메인 |
| 4 | Fault-mode routing | 핵심 기여 (Fault-mode gating보다 검색 범위 넓음) |
| 5 | Sensor normalization | 핵심 기여 (Normalization ablation보다 일반적) |

> **제거된 키워드:** `NASA CMAPSS` (데이터셋명으로 검색성 낮음), `LSTM` (지나치게 범용)  
> **main.tex 현재 keywords:** 7개 → 위 5개로 교체 권장

---

## Narrative Logic (H2 → H3 causal chain)

1. **H2 finding**: Fleet MinMax (N1) is significantly best across FD001/FD002/FD004 — but FD003 has inexplicably high variance (std = 12.86).
2. **Bridge**: The high variance is not a normalization artifact; it is caused by two distinct fault modes (HPC degradation vs. fan degradation) mixed in FD003/FD004 training data.
3. **H3 solution**: An end-to-end attention gate (M3) learns to separate fault modes from early-cycle observations — no cluster labels required. This collapses the FD003 variance and yields the paper's headline result.
4. **H4 & H1 context**: Once clipping is set to 125 and fault modes are handled, neither asymmetric loss functions nor alternative normalization methods add measurable value. This positions the paper as providing actionable design guidance: "fix clipping and architecture; everything else is noise."

---

## Draft Notes

- "over 800 training runs" = H1 (20) + H2 (140) + H3 (40) + H4 (560+150+100+45) ≈ 1,055 runs total; conservative phrasing avoids counting overlap.
- If venue imposes a stricter word limit (e.g., 150 words), cut the loss-function sentence and compress the normalization finding to one clause.
- K sensitivity (K ∈ {5,10,15,20,30}) added to para 2 — supports "very first flight cycles" claim in Discussion §V.D.
- "first observed flight cycles" is intentionally vague (K-agnostic) since K=5 and K=10 both work; specific K cited in Results.
- RevIN failure on multi-condition datasets is implied by "instance-level alternatives" — can be made explicit if a reviewer asks.
- Word count target for conference: 150–300 words. At ~265 words this is within range; if a 200-word limit applies, cut the K-sensitivity clause and the 98.8% NASA number.
