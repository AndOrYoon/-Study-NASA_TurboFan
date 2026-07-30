# Abstract — RESS Submission

> **Draft status:** v1.2 — 2026-07-09 (RESS reframing: Task A applied)

**Title:**
From Fleet Normalization to Fault-Mode Gating: A Cross-Dataset Ablation Study of Turbofan Remaining Useful Life Prediction

---

## Abstract

Reliable Prognostic and Health Management (PHM) systems for turbofan engines embed interdependent design decisions — RUL label clipping, sensor normalization, fault-mode architecture, and training loss function — whose contributions to predictive reliability are rarely isolated. We present a controlled ablation study across all four NASA CMAPSS sub-datasets (FD001–FD004), covering five clipping thresholds, seven normalization strategies, four fault-mode architectures, and seven loss functions, with all comparisons Benjamini-Hochberg FDR-corrected.

Fleet min-max normalization outperforms all per-unit and instance-level alternatives on FD001, FD002, and FD004 but exhibits anomalously high inter-seed variance on FD003 (RMSE std = 12.86 vs. ≤1.84 elsewhere), traced to co-existing HPC and fan fault modes rather than normalization failure. An unsupervised attention-gate model (M3) routes engines to fault-specific branches from five initial flight cycles, reducing FD003 RMSE by 65.8% (14.78 ± 1.32 vs. 43.23 ± 0.18) and NASA Score by 98.8%, while remaining immune to the test-time cluster collapse that degrades GMM hard-routing by 75.4% on FD004. No custom loss outperforms MSE after multiple-comparison correction; removing RUL clipping inflates NASA prognostic scores by up to 306,000-fold regardless of loss design.

These findings establish a three-tier reliability-driven design checklist — label engineering, fault-mode architecture, loss function — where each tier's failure impact qualitatively exceeds the next, providing reliability engineers with risk-prioritised guidance for PHM system design.

---

## Metadata

| Item | Detail |
|------|--------|
| Word count | ~194 words (RESS limit: 200) |
| Target venue | Reliability Engineering & System Safety (RESS) |
| Framing | Reliability risk analysis + controlled ablation + fault-mode PHM contribution |
| Primary claim | Fleet normalization beats per-unit; fault-mode gating explains residual variance |
| Secondary claim | Clipping and architecture dominate; loss function does not matter post-BH-FDR |
| Key number | −65.8% RMSE on FD003 (14.78 ± 1.32 vs. 43.23 ± 0.18) |

---

## Narrative Logic (H5 → H6 causal chain)

1. **H5 finding**: Fleet MinMax (N1) is significantly best across FD001/FD002/FD004 — but FD003 has inexplicably high variance (std = 12.86).
2. **Bridge**: The high variance is not a normalization artifact; it is caused by two distinct fault modes (HPC degradation vs. fan degradation) mixed in FD003/FD004 training data.
3. **H6 solution**: An end-to-end attention gate (M3) learns to separate fault modes from early-cycle observations — no cluster labels required. This collapses the FD003 variance and yields the paper's headline result.
4. **H7 & H2 context**: Once clipping is set to 125 and fault modes are handled, neither asymmetric loss functions nor alternative normalization methods add measurable value. This positions the paper as providing actionable design guidance: "fix clipping and architecture; everything else is noise."

---

## Draft Notes

- "over 800 training runs" = H2 (20) + H5 (140) + H6 (40) + H7 (560+150+100+45) ≈ 1,055 runs total; conservative phrasing avoids counting overlap.
- If venue imposes a stricter word limit (e.g., 150 words), cut the loss-function sentence and compress the normalization finding to one clause.
- K sensitivity (K ∈ {5,10,15,20,30}) added to para 2 — supports "very first flight cycles" claim in Discussion §V.D.
- "first observed flight cycles" is intentionally vague (K-agnostic) since K=5 and K=10 both work; specific K cited in Results.
- RevIN failure on multi-condition datasets is implied by "instance-level alternatives" — can be made explicit if a reviewer asks.
- Word count target for conference: 150–300 words. At ~265 words this is within range; if a 200-word limit applies, cut the K-sensitivity clause and the 98.8% NASA number.
