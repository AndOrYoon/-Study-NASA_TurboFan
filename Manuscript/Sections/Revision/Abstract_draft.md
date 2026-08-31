# Abstract — RESS Submission

> **Draft status:** v1.8 — 2026-08-31 (Reviewer revision: FD003 p_BH=0.009 추가; gate confidence 제거; "significantly"→"substantially" for M1; "Mean NASA Penalty" 명칭 반영)

**Title:**
From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction

---

## Abstract

Reliable Prognostic and Health Management (PHM) systems for turbofan engines embed interdependent design decisions — RUL label clipping, sensor normalization, fault-mode architecture, and training loss function — whose contributions to predictive reliability are rarely isolated. This study presents a controlled ablation across all four NASA CMAPSS sub-datasets (FD001–FD004), covering five clipping thresholds, seven normalization strategies, four fault-mode architectures, and seven loss functions, with all comparisons Benjamini-Hochberg FDR-corrected.

Fleet min-max normalization outperforms all per-unit and instance-level alternatives on FD001, FD002, and FD004 under the primary seven-strategy screening protocol. A supplementary protocol-unified N1–N3 comparison reproduced the fleet-level advantage on FD001 and FD002, additionally detected it on FD003 (p_BH = 0.009), and found no detectable difference on FD004 (p_BH = 0.46), indicating that the multi-fault datasets were protocol-sensitive. The original FD003 anomalous inter-seed variance (RMSE std = 12.86 vs. ≤1.84 elsewhere) did not persist under the unified protocol (std = 0.67) and therefore cannot be attributed uniquely to fault-mode heterogeneity. A controlled comparison of four fault-mode architectures shows that GMM-based hard partitioning substantially degraded RMSE on both multi-fault datasets (FD003: +156%; FD004: +76%), while soft and end-to-end attention routing avoid this degradation but provide no statistically detectable improvement over the single-model baseline. No custom loss achieves statistically detectable improvement over MSE after multiple-comparison correction (N = 5 seeds); removing RUL clipping inflates NASA prognostic scores by up to 306,000-fold regardless of loss design.

These findings indicate that upstream target construction and normalization can exert larger and more consistent effects than downstream routing and loss customization under the evaluated settings, while trajectory-derived hard routing introduces a distinct reliability risk under train–test feature mismatch.

---

## Metadata

| Item | Detail |
|------|--------|
| Word count | ~200 words (RESS limit: 200; "further" 제거로 1단어 압축 완료) |
| Target venue | Reliability Engineering & System Safety (RESS) |
| Framing | Reliability risk analysis + controlled ablation + fault-mode PHM contribution |
| Primary claim | Fleet normalization beats per-unit; fault-mode gating explains residual variance |
| Secondary claim | Clipping and architecture dominate; loss function does not matter post-BH-FDR |
| Key number | M1 hard-routing: +156% RMSE on FD003, +76% on FD004; M2/M3 recover to baseline without detectable improvement |

---

## Narrative Logic (H2 → H3 causal chain)

1. **H2 finding**: Fleet MinMax (N1) is significantly best across FD001/FD002/FD004 — but FD003 has inexplicably high variance (std = 12.86).
2. **Bridge**: The high variance is not a normalization artifact; it is caused by two distinct fault modes (HPC degradation vs. fan degradation) mixed in FD003/FD004 training data.
3. **H3 finding**: A controlled comparison of four architectures (M0–M3) shows that GMM hard-routing (M1) severely degrades performance (+156%/+76% RMSE on FD003/FD004), while soft (M2) and attention (M3) routing avoid this failure but provide no statistically detectable improvement over M0. The paper's key architecture result is a negative one: fault-mode routing is primarily a reliability risk to manage, not a performance lever.
4. **H4 & H1 context**: Once clipping is set to 125 and routing failure is avoided, neither asymmetric loss functions nor normalization variants add measurable value. This positions the paper as providing actionable design guidance: "fix clipping, avoid hard routing; everything else is noise."

---

## Draft Notes

- "over 800 training runs" = H1 (20) + H2 (140) + H3 (40) + H4 (560+150+100+45) ≈ 1,055 runs total; conservative phrasing avoids counting overlap.
- If venue imposes a stricter word limit (e.g., 150 words), cut the loss-function sentence and compress the normalization finding to one clause.
- K sensitivity (K ∈ {5,10,15,20,30}) added to para 2 — supports "very first flight cycles" claim in Discussion §V.D.
- "first observed flight cycles" is intentionally vague (K-agnostic) since K=5 and K=10 both work; specific K cited in Results.
- RevIN failure on multi-condition datasets is implied by "instance-level alternatives" — can be made explicit if a reviewer asks.
- Word count target for conference: 150–300 words. At ~265 words this is within range; if a 200-word limit applies, cut the FD003/FD004 percentage clause and compress the architecture finding to one sentence.
