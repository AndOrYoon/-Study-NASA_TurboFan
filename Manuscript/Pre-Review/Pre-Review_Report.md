# Pre-Submission Review Report
## "From Fleet Normalization to Fault-Mode Gating: A Cross-Dataset Ablation Study of Turbofan RUL Prediction"

> **Generated:** 2026-07-03  
> **Reviewers:** A (Domain Specialist), B (Statistical Rigor), C (Academic Novelty)  
> **Calibration sources:** IEEE TIE / PHM Society (A), NeurIPS 2026 / ICML 2024 / TNNLS (B), MSSP / RESS / Information Fusion (C)

---

## 0. Executive Summary

| | Reviewer A | Reviewer B | Reviewer C |
|---|---|---|---|
| **Role** | CMAPSS/PHM Domain Specialist | Statistical Methodology Expert | Academic Breadth & Novelty |
| **Calibrated against** | IEEE Trans. Industrial Electronics, PHM Society | NeurIPS 2026, ICML 2024, Henderson et al. (2018) | MSSP, RESS, Information Fusion |
| **Recommendation** | **Major Revision** | **Major Revision** | **Major Revision** |
| **Tone** | Constructive — values controlled design and null results | Critical — flags internal statistical contradictions | Balanced — strongest on novelty gaps and missing literature |

All three reviewers independently identified the same two issues as most critical. The paper has genuine contributions (M3 architecture, normalisation ablation, FD003 diagnostic insight) but requires substantive revision before submission.

---

## 1. Cross-Reviewer Consensus: Must-Fix Issues

The following issues were flagged by **two or more reviewers independently** and represent the highest-priority revision targets.

### [CRITICAL-1] H2 Uses Ridge Regression, Not the Shared LSTM Backbone

**Flagged by:** Reviewer A (Major M1), Reviewer B (Major M1 & M8), Reviewer C (Major MC2)

Introduction §I, paragraph 3 states: *"Fixing a common stacked LSTM backbone, we independently vary (H2) five RUL clipping thresholds."* This is factually incorrect. H2 uses LinearRegression (Ridge), which is both deterministic and a qualitatively different model class. The methodology section never discloses this. Consequences:

- H2 RMSE values (FD001: 21.90) are not comparable to H5–H7 RMSE values (FD001: 14.14 ± 0.22).
- The "three-tier hierarchy, order of magnitude per tier" claim conflates model-complexity differences with design-factor effects.
- The Conclusion and Introduction contributions overstate what has been demonstrated.

**Required action (choose one):**
- (a) Re-run H2 with the stacked LSTM backbone to make all four hypotheses directly comparable; or
- (b) Add explicit H2 model description to §III, correct Introduction §I ¶3, and qualify all hierarchy magnitude claims to state that Tier 1 is supported by a linear-model experiment only.

---

### [CRITICAL-2] Internal Statistical Contradiction: Cohen's d = 2.00 but p_BH = 1.0

**Flagged by:** Reviewer B (Major M2, M3, M7)

Results §IV.D reports L7/HubA at clip=125 on FD001: RMSE 14.90 ± 0.33 vs 16.34 ± 0.40, Cohen's d = 2.00, p_BH = 1.0. This is statistically impossible under a single coherent testing framework:

- If d is computed on per-engine pooled values (N = 5 × 100 = 500), a Wilcoxon test at this sample size with d = 2.00 would yield p < 10⁻¹⁰ — not p_BH = 1.0.
- The only coherent resolution is that d is computed on the large pooled sample while the Wilcoxon test runs on N = 5 seed-level values. But N = 5 with 84 BH-corrected tests has a minimum achievable corrected p of ~0.004 / (84 × C(10,5)) — far above the BH threshold of 0.05/84 ≈ 0.0006. **The H7 null result is therefore mathematically guaranteed by the study design** regardless of the true underlying effect.

**Required action:** Specify the exact N used for each statistic. If d uses N = 500 and p uses N = 5, acknowledge: (a) Cohen's d values are inflated by √(n_engines) and should be recomputed at the seed-level, and (b) the H7 null result reflects insufficient power rather than evidence of no effect. Reframe "No custom loss function achieves significant improvement" as "We found no detectable improvement, though power was insufficient to rule out small-to-medium effects."

---

### [CRITICAL-3] Arithmetic Error: 6 × 4 × 4 = 96, Not 84

**Flagged by:** Reviewer B (Major M4)

Results §IV.D, Discussion §V.E, and Conclusion §VI all state "84 pairwise comparisons (6 loss functions × 4 clipping values × 4 datasets)." The product is 96. The number 84 is internally consistent (e.g., C(7 losses, 2) × 4 datasets = 84 pairwise comparisons among all loss functions per dataset), but the stated parenthetical derivation is wrong. Undisclosed exclusions from the BH family undermine the correction's validity.

**Required action:** State the exact family of tests included in the BH correction, provide a complete enumeration (possibly in Supplementary), and correct the count throughout.

---

### [CRITICAL-4] H5/FD003 and H6/M0/FD003 Baseline Inconsistency

**Flagged by:** Reviewer B (Major M5)

H5 reports N1/FD003 RMSE = 19.05 ± 12.86. H6 reports M0/FD003 RMSE = 43.23 ± 0.18. These should be the **same experiment** (identical backbone, N1 normalisation, FD003, seeds {0,1,2,3,4}). The mean difference is 24.18 cycles; the standard deviation difference is two orders of magnitude (12.86 vs 0.18). This inconsistency directly falsifies the paper's central mechanistic claim — that H5/FD003 high variance is caused by bimodal convergence — because M0 in H6 should reproduce that variance but shows effectively zero variance instead.

**Required action:** Identify the experimental difference between H5 and H6 for FD003 (different preprocessing? different val split? different data loading order?) and either reconcile the baselines or explicitly explain why two nominally identical experiments produce different results.

---

### [CRITICAL-5] RevIN Implementation Deviates from Original Paper Without Justification

**Flagged by:** Reviewer A (Major M3), Reviewer B (Minor m4)

§III.D states that for N7 (RevIN), "no denormalization is applied to the prediction." Kim et al. (2022)'s RevIN gains were partly attributed to the denormalisation restoring the output to the target distribution. Without denormalisation, N7 as implemented is a learnable input normalisation without output correction — which may explain the reported underperformance as an implementation artefact rather than RevIN's unsuitability for prognostics.

**Required action:** Run N7 with standard denormalisation on at least FD001 and report whether performance changes. If unchanged, the null finding is robust; if substantially different, the current comparison is unfair.

---

### [IMPORTANT-6] Sensor Name Citations Missing

**Flagged by:** Reviewer A (Major M4)

Results §IV.C.1 and Discussion §V.D name s15 as "bypass pressure ratio" and s12 as "fuel-to-pressure ratio." The original CMAPSS dataset files (Saxena & Goebel, 2008) do not include public sensor-name mappings. Using unsupported physical names is a reproducibility and credibility risk.

**Required action:** Provide an explicit citation for the sensor-name mapping (NASA technical memorandum, C-MAPSS simulation tool documentation, or a peer-reviewed secondary source). If no public citation exists, revert to sensor indices only.

---

### [IMPORTANT-7] Missing Literature: MoE/Gating, RevIN Critique, Recent CMAPSS

**Flagged by:** Reviewer C (Major MC4, MC5)

Gaps identified by web search:
- **RUL-QMoE** (arXiv:2512.23725, Dec 2024): probabilistic mixture-of-experts for PHM RUL — overlaps directly with M3's architecture claim.
- **Metric-Gated MoE for Fault Diagnosis** (Springer Complex & Intelligent Systems, 2025): MoE gating for multisource PHM.
- **On the Role of RevIN** (arXiv:2603.11869, 2025): independently concludes RevIN fails when the normalisation challenge is not temporal distribution shift — directly supports and must be cited for contribution (ii).
- **Noise or Signal? Deconstructing RevIN** (arXiv:2510.04667, 2024): identifies RevIN failure modes compatible with H5's finding.
- **Deep Domain Adaptation for Turbofan RUL** (arXiv:2510.03604, 2024): cross-dataset preprocessing comparison that partially overlaps with contribution (ii).
- **CMAPSS PHM Survey 2024** (MDPI Sensors 24(11):3454): required for contextualising ablation scope.

**Required action:** Add and cite the above in the Related Work and Discussion sections, and explicitly differentiate M3 from prior MoE/gating approaches.

---

## 2. Individual Reviewer Reports

---

### Reviewer A — Domain Specialist (IEEE Trans. Style)
**Recommendation: Major Revision**

**Review calibrated against:** IEEE Transactions on Industrial Electronics, PHM Society IJPHM standards

#### Summary
The controlled multi-dataset design, honest null results, and the FD003 inter-seed variance reframing are genuine contributions that advance reproducibility standards in the CMAPSS community. The revision is requested (not rejection) because the identified issues are correctable without redesigning the study.

#### Technical Correctness

**Constant sensor removal (§III.B):** Criterion for "constant" (threshold value, before/after residualisation, training-set only) is never stated. Must be made explicit for reproducibility.

**K-means residualisation (§III.C):** k = 6 is correct but its justification (six known operating regimes per Saxena & Goebel, 2008) is never stated — appears arbitrary to uninformed readers. Requires one sentence of citation. Also: does GMM clustering in H6 operate on residualised or raw sensor values? This must be clarified, as residualisation may remove information the GMM uses.

**Zero-padding:** How many FD001–FD004 test engines have fewer than 30 cycles? This affects prediction reliability and should be reported.

**RevIN (§III.D) — Non-standard modification [MAJOR]:** Omitting denormalisation for scalar output is a design choice but is not justified. Given N7 significantly underperforms on all three evaluable datasets, the risk is that poor performance reflects the implementation choice, not RevIN's structural unsuitability for prognostics. An ablation with standard denormalisation is required.

**Sensor semantics [MAJOR]:** s15 as "bypass pressure ratio" is physically plausible for HPC vs. fan discrimination but is unsupported without a citation. See [IMPORTANT-6] above.

**|Δz| statistic [MAJOR]:** Results §IV.C.1 reports "|Δz| = 31.3" without defining the formula. Reproducibility requires the exact formula (z-score difference? pooled-SD normalised difference? Cohen's d equivalent?).

#### Experimental Validity

**H2/H5–H7 backbone inconsistency [MAJOR]:** See [CRITICAL-1] above.

**H2 Wilcoxon for deterministic model [MAJOR]:** LinearRegression is deterministic. The source of variance across "20 runs" is never explained. If the 20 runs use the same data and model, all outputs are identical and the Wilcoxon test is meaningless. If the 20 runs use different random train/val splits, this must be stated. The per-engine error comparison framework must be explicitly described.

**Loss hyperparameter tuned on FD001, then evaluated on FD001 (§III.G):** The grid search selects parameters that maximise FD001 performance, then reports FD001 as one of four evaluation datasets. The nominally largest H7 effects on FD001 (L5, L7) are therefore unsurprising. A paragraph in §IV.D acknowledging this must be added.

**M3 GatingNet K = 10 sensitivity:** No ablation varying K. For a deployment claim based on K = 10 representing only 4% of average engine lifetime, some sensitivity evidence is needed.

**Wilcoxon pooled across seeds:** Pooling per-engine errors across 5 seeds treats seed replicates as independent observations. This overstates independence. The conservative alternative (N = 5 seed-level means) should at minimum be discussed as a robustness check.

#### Benchmark Comparison

**CAELSTM comparison:** CAELSTM's RMSE = 13.40 is a point estimate. M3's lower bound at mean − 1σ = 13.46 is nearly identical to it. Uncertainty of the reference value must be noted. Preprocessing differences (normalisation, backbone) must be tabulated.

**Abdullah (2026) comparison:** The sentence "a result our L7... numerically surpasses" should be removed. Architecture differences make the comparison uninformative.

**Rengasamy et al. comparison:** The paper's explanation (cross-dataset BH-FDR raises threshold) is plausible but should be framed as one of several possible interpretations, not the definitive account.

#### Strengths
1. Controlled multi-dataset ablation design — methodological gold standard for CMAPSS.
2. Honest null results for H5 and H7.
3. FD003 inter-seed variance as fault-mode diagnostic indicator — most transferable insight.
4. M3's immunity to test-time cluster collapse — sound and practically motivated.
5. Rigorous statistics (Wilcoxon + BH-FDR + Cohen's d) relative to CMAPSS literature norm.

#### Major Concerns (M1–M6)
| # | Issue | Section |
|---|-------|---------|
| M1 | H2 Ridge/LSTM inconsistency; hierarchy claim overstated | §I, §III, §VI |
| M2 | H2 Wilcoxon unit of observation undefined for deterministic model | §III.J, §IV.A |
| M3 | RevIN denormalisation ablation needed | §III.D |
| M4 | Sensor name citations absent | §IV.C.1, §V.D |
| M5 | \|Δz\| formula undefined | §IV.C.1 |
| M6 | CAELSTM comparison lacks uncertainty bound; preprocessing not tabulated | §IV.C.2 |

#### Minor Concerns (m1–m10)
- m1: Table I should include engine counts and total cycles per dataset
- m2: Validation split — fixed vs. resampled across seeds?
- m3: GatingNet capacity (10×14→32 vs 10×20→32) not varied
- m4: M1 cluster collapse (247:1 on FD004) should be shown as a figure
- m5: Decision-tree flowchart may not render in two-column IEEE format
- m6: Remove Abdullah RMSE comparison from §IV.D
- m7: FD003 per-seed RMSE bimodality should be shown as a table/figure
- m8: AI usage disclosure absent (required by IEEE and Elsevier)
- m9: Code/data availability statement and compute resources absent
- m10: H7 null result should note hyperparameter tuning on FD001 evaluation set

---

### Reviewer B — Statistical Methodology Specialist (NeurIPS/ICML Style)
**Recommendation: Major Revision**

**Review calibrated against:** NeurIPS 2026 Reviewer Guidelines, ICML 2024 Paper Guidelines, Henderson et al. (2018) "How Many Random Seeds?"

#### Experimental Design

**H2 model not disclosed in Methodology [MAJOR]:** Introduction §I ¶3 claims "fixing a common stacked LSTM backbone" for all four hypotheses. Methodology §III.E–H describes only the LSTM. The Ridge regression model used for H2 appears only in the Results footnote. This omission is not a presentation issue — it is a design validity issue because the RMSE values from H2 are the basis for Tier 1's "leverage" in the three-tier hierarchy. See [CRITICAL-1].

**H5/H6 FD003 baseline inconsistency [MAJOR]:** N1/FD003 RMSE = 19.05 ± 12.86 (H5) vs M0/FD003 RMSE = 43.23 ± 0.18 (H6) — same experimental configuration should produce the same result. See [CRITICAL-4]. This is the single most concerning technical inconsistency in the manuscript.

**H7 hyperparameter leakage [MAJOR]:** Grid search on FD001 to select τ, λ_t, λ_a, δ; FD001 then used as evaluation dataset. The largest H7 effects appear on FD001 (L5, L7). This is not randomised — it is designed to show FD001 gains. The null result after BH-FDR may be partly a consequence of FD001's parameters being suboptimal for the other three datasets.

#### Statistical Testing

**Cohen's d vs. Wilcoxon p-value internal contradiction [MAJOR]:** See [CRITICAL-2]. At d = 2.00 and N = 500 per group (per-engine pooled), the Wilcoxon p-value would be p < 10⁻¹⁰ — not p_BH = 1.0. This is a disqualifying internal contradiction that must be resolved before any statistical claim in the paper can be accepted. The most likely explanation (d on N=500, Wilcoxon on N=5) produces pseudo-replication in one direction and underpower in the other.

**H7 null result is a power artefact [MAJOR]:** At N = 5 per group (two-sample Wilcoxon), minimum achievable p = 1/C(10,5) = 0.004. BH threshold at rank 1 (most significant) over 84 tests = 0.05/84 ≈ 0.0006. Since 0.004 > 0.0006, no test can achieve BH significance with N = 5 and 84 comparisons. The null result in H7 is guaranteed by design — it cannot be reported as an empirical finding about loss functions. See [CRITICAL-2].

**Arithmetic error in comparison count [MAJOR]:** 6 × 4 × 4 = 96 ≠ 84. See [CRITICAL-3].

**One-sided test direction inconsistency [MODERATE]:** §III.J specifies one-sided testing ("treatment vs. baseline"). §IV.B reports per-unit strategies are "significantly inferior" (i.e., baseline > treatment). A pre-specified one-sided test in the direction "treatment outperforms baseline" cannot produce this result — the direction must be flipped. Either the test was applied bilaterally and reported as one-sided, or the directionality was determined post-hoc. Must be clarified.

**H2 Wilcoxon for deterministic model [MODERATE]:** LinearRegression is deterministic. The source of variance across 20 runs must be explained. If runs use the same data, all outputs are identical and the test is invalid.

#### Effect Size Reporting

**Cohen's d values of 7.6–22.4 are artefacts of sample size [MAJOR]:** d = 22.4 (N2 vs N1 on FD002) is physically impossible for a real experiment with genuine variance. These values result from computing Cohen's d on per-engine pooled samples (N ≈ 1500 for FD002 × 5 seeds), where the denominator is the within-condition inter-engine RMSE SD — a function of engine lifetime variation, not treatment effect. Appropriate d should be computed on seed-level aggregate RMSE (N = 5 per group), giving much smaller and interpretable values.

**d threshold of 0.3 not domain-justified [MODERATE]:** Cohen's 0.3 ("small" by psychology norms) should be translated to meaningful prognostics units (e.g., "0.3 SD ≈ X cycles ≈ Y maintenance planning hours") for an engineering venue.

#### Reproducibility and Claims vs. Evidence

**Validation split specification [MODERATE]:** §III.H does not state whether the 20% engine-level holdout is fixed across seeds or resampled per seed. If resampled, different seeds evaluate on different validation sets — complicating seed-to-seed comparison.

**Three-tier hierarchy magnitude claim unsupported [MAJOR]:** "Each tier's achievable leverage declines by at least an order of magnitude" — this claim mixes H2 (Ridge) and H5–H7 (LSTM) results and does not provide a formula or calculation. It is a narrative assertion.

**H7 null result language [MAJOR]:** "No custom loss function achieves statistically significant improvement" should become "We found no statistically detectable improvement; power was insufficient to rule out small-to-medium effects given N = 5 seeds and 84 BH-corrected comparisons."

**p = 0.97 for clip=130 without directional context [MINOR]:** A one-sided p = 0.97 implies the complementary direction (clip=125 > clip=130) has p ≈ 0.03 — borderline significant. This asymmetry should be noted.

#### Summary of Major Concerns
| # | Issue |
|---|-------|
| M1 | H2 model not disclosed in Methodology; Introduction claim false |
| M2 | d = 2.00 with p_BH = 1.0 is statistically impossible — internal contradiction |
| M3 | N=5 + 84 BH tests = guaranteed null; H7 result is power artefact |
| M4 | 6×4×4 = 96 ≠ 84 — arithmetic error in comparison family |
| M5 | H5 N1/FD003 RMSE 19.05±12.86 ≠ H6 M0/FD003 43.23±0.18 |
| M6 | H7 hyperparameter tuning on FD001 leaks into FD001 evaluation |
| M7 | Cohen's d 7.6–22.4 from per-engine N=500 not seed-level N=5 |
| M8 | Hierarchy order-of-magnitude claim mixes model classes; no calculation |

---

### Reviewer C — Academic Novelty & Breadth (MSSP/RESS Style)
**Recommendation: Major Revision**

**Review calibrated against:** MSSP, RESS, Information Fusion editorial standards; web search for 2024–2026 competing papers

#### Novelty Assessment by Contribution

**Contribution (i) — Clip = 125 cross-dataset validation: Incremental.**
Already the de-facto standard in the 2023–2026 literature (cited without justification in virtually all recent CMAPSS papers). The "catastrophic clip = None" finding is mechanistically obvious from the NASA Score formula; the value is in the quantification, not the discovery.

**Contribution (ii) — First controlled normalisation ablation: Genuinely novel.**
No prior publication conducts a systematic multi-strategy normalisation comparison across all four CMAPSS sub-datasets under controlled architecture and loss. The RevIN extension is timely. However:
- The RevIN critique literature is not engaged: arXiv:2603.11869 and arXiv:2510.04667 reach independently the same theoretical conclusion the authors state. These must be cited.
- The "first ablation" claim should be qualified with "to the best of our knowledge."

**Contribution (iii) — M3 early-cycle attention gate (−65.8% FD003 RMSE): Architecturally novel, diluted by MoE literature.**
The specific combination (first-K-cycle GatingNet + two-branch LSTM, no supervision, end-to-end) does not appear in prior CMAPSS literature. However:
- MoE architectures with learned soft gating are well-established (Jacobs et al., 1991; Shazeer et al., 2017). Recent PHM applications include RUL-QMoE (arXiv:2512.23725, Dec 2024) and Metric-Gated MoE for Fault Diagnosis (Springer 2025). These must be cited and differentiated.
- The backbone (two-layer stacked LSTM, 2018-era) is not competitive with 2026 state of the art (CAELSTM RMSE = 13.40 on FD003 vs M3's 14.78). The 10.3% gap attributed to "simpler backbone" is a hypothesis, not a verified claim.
- CAELSTM achieves a 10.3% lower RMSE using supervised approaches — and the gap may reflect backbone quality rather than gating quality. An experiment replacing LSTM branches with a stronger encoder would be needed to attribute the residual gap definitively.

**Contribution (iv) — Three-tier hierarchy: Valuable heuristic, not independently defensible as a research contribution.**
The hierarchy is post-hoc rationalisation of results from four non-commensurate experiments (H2 = Ridge, H5/H6/H7 = LSTM). Its practical value is real; its scientific status as "established fact" is overstated. See [CRITICAL-1].

#### Positioning Critique — Overlooked Literature

| Paper | Relevance | Status |
|-------|-----------|--------|
| RUL-QMoE (arXiv:2512.23725, Dec 2024) | Probabilistic MoE for PHM RUL — directly competes with M3's architecture claim | **Missing** |
| Metric-Gated MoE for Fault Diagnosis (Springer 2025) | MoE gating for multisource PHM | **Missing** |
| On the Role of RevIN (arXiv:2603.11869, 2025) | Same conclusion as contribution (ii) reached independently | **Missing** |
| Noise or Signal? Deconstructing RevIN (arXiv:2510.04667, 2024) | RevIN failure modes compatible with H5 finding | **Cited as ref26** — needs more engagement |
| Deep Domain Adaptation for Turbofan RUL (arXiv:2510.03604, 2024) | Cross-dataset preprocessing survey | **Missing** |
| CMAPSS PHM Survey 2024 (MDPI Sensors 24(11):3454) | Ablation scope context | **Missing** |
| STAR Transformer for Turbofan RUL (MDPI Sensors 24(3):824, 2024) | Attention-based SOTA on CMAPSS | **Missing** |
| Novel TCN Preprocessing Approach (arXiv:2605.02507, 2025) | Directly competes with H5 preprocessing claims | **Missing** |

#### Scope and Generalisability

**CMAPSS-only is serious but not fatal for MSSP/RESS** — provided:
- Limitations section remains honest (already achieved).
- At least one N-CMAPSS experiment is added. N-CMAPSS provides health parameter labels that would allow M3 routing accuracy to be directly measured rather than inferred from Silhouette scores.

**Backbone is 2018-era.** A stacked LSTM with 64 hidden units is well below the architectural frontier in 2026. This does not invalidate the controlled ablation design (holding backbone constant is correct methodology) but limits the relevance of absolute RMSE values and the generalisability of the hierarchy to modern architectures.

**Synthetic data assumption in contribution (ii).** Fleet normalisation wins *because* CMAPSS inter-engine variation is small by design. The finding may reverse on real fleets with calibration offsets. The Discussion correctly notes this; it should also appear in the Abstract.

#### Title and Framing

Current title ("From Fleet Normalization to Fault-Mode Gating") emphasises H5 and H6 but ignores H2 (clipping) and H7 (loss). Suggested alternative: *"Clipping, Normalisation, Fault-Mode Gating, and Loss Functions: A Cross-Dataset Ablation Hierarchy for Turbofan RUL Prediction"*

Opening sentence leads with the NASA metric formula — niche for a general MSSP/RESS readership. An engineering-problem framing (unscheduled engine removal costs, maintenance scheduling under uncertainty) would better serve these venues before narrowing to CMAPSS methodology.

The inter-seed variance diagnostic insight (§V.C) is **underemphasised** relative to performance results. It is the most transferable conceptual contribution and should be elevated in the Abstract and Introduction as a co-equal finding.

#### Major Concerns
| # | Issue |
|---|-------|
| MC1 | Stacked LSTM backbone uncompetitive in 2026; M3-on-transformer experiment needed to separate gating from backbone effects |
| MC2 | H2 model-class confound undermines three-tier hierarchy magnitude claims |
| MC3 | No N-CMAPSS validation; expected by MSSP/RESS area editors in 2026 |
| MC4 | MoE literature (RUL-QMoE, Metric-Gated MoE) not engaged; M3 novelty overstated |
| MC5 | RevIN critique literature (arXiv:2603.11869, arXiv:2510.04667) not fully cited |

#### Minor Concerns
- mn1: Violin/jitter plot for per-seed RMSE on FD003 (M0 vs M3) would make bimodality argument visually immediate
- mn2: "M3 is immune to cluster collapse" is conditional on K=10 being sufficient — state conditionality explicitly
- mn3: H7 grid search on FD001 introduces mild snooping; clarify whether tuning used training partition only
- mn4: Silhouette threshold 0.5 from [35] should be identified by paper title (Kaufman & Rousseeuw 1990)
- mn5: Title should include "CMAPSS" to set scope expectations
- mn6: Opening framing of Introduction should lead with engineering problem, not metric formula

---

## 3. Consolidated Action Item Matrix

Priority levels: 🔴 Critical (must fix for acceptance) | 🟡 Important (expected for revision) | 🟢 Minor (enhances quality)

| # | Issue | Reviewers | Priority |
|---|-------|-----------|----------|
| A1 | Re-run H2 with LSTM backbone **OR** explicitly limit all hierarchy claims to "linear-model evidence only" | A, B, C | 🔴 |
| A2 | Resolve Cohen's d = 2.00 vs. p_BH = 1.0 contradiction — specify exact N for each statistic | B | 🔴 |
| A3 | Reframe H7 null result as power-limited non-finding, not confirmed absence of effect | B | 🔴 |
| A4 | Fix arithmetic: 6×4×4 = 96, provide exact BH family enumeration | B | 🔴 |
| A5 | Reconcile H5/FD003 (19.05±12.86) vs H6/M0/FD003 (43.23±0.18) | B | 🔴 |
| A6 | RevIN denormalisation ablation (at least FD001) | A, B | 🔴 |
| A7 | Add H2 model to §III Methodology; correct Introduction §I ¶3 | A, B, C | 🔴 |
| A8 | Cite sensor name mapping for s15, s12 or revert to index labels | A | 🟡 |
| A9 | Define |Δz| formula in §IV.C.1 | A | 🟡 |
| A10 | Add uncertainty bound for CAELSTM reference RMSE; tabulate preprocessing differences | A | 🟡 |
| A11 | Recompute Cohen's d at seed-level (N=5); report both seed-level d and note inflated values | B | 🟡 |
| A12 | Add domain justification for d ≥ 0.3 threshold in prognostics units (cycles) | B | 🟡 |
| A13 | Clarify validation split: fixed vs. resampled per seed | A, B | 🟡 |
| A14 | Add H2 Wilcoxon unit of observation (per-engine? identical runs?) | A, B | 🟡 |
| A15 | Add MoE literature: cite RUL-QMoE (arXiv:2512.23725), Metric-Gated MoE (Springer 2025) | C | 🟡 |
| A16 | Add RevIN critique literature: arXiv:2603.11869 + arXiv:2510.04667 with discussion | C | 🟡 |
| A17 | Add N-CMAPSS pilot experiment (at minimum M3 on one sub-dataset) | C | 🟡 |
| A18 | Acknowledge H7 hyperparameter leakage (FD001 tuning + evaluation) in §IV.D | A, B | 🟡 |
| A19 | Add K sensitivity analysis for GatingNet (K = 5, 10, 15, 20) | A | 🟡 |
| A20 | Revise title to include all four design factors | C | 🟢 |
| A21 | Add IEEE/Elsevier AI disclosure statements | A | 🟢 |
| A22 | Add code/data availability + compute resources (GPU, training time) | A | 🟢 |
| A23 | Add violin/jitter plot: per-seed RMSE for M0 vs M3 on FD003 | C | 🟢 |
| A24 | Add bar chart: M1 cluster assignment distribution (train vs. test on FD004) | A | 🟢 |
| A25 | State zero-padding scope: how many test engines < 30 cycles? | A | 🟢 |
| A26 | State p = 0.97 directional implication for clip=130 | B | 🟢 |
| A27 | Remove or strongly hedge Abdullah RMSE comparison in §IV.D | A | 🟢 |
| A28 | Elevate inter-seed diagnostic insight to Abstract and Introduction contributions | C | 🟢 |
| A29 | Revise opening sentence of Introduction for MSSP/RESS readership | C | 🟢 |
| A30 | Identify Silhouette threshold source as Kaufman & Rousseeuw (1990) | C | 🟢 |

---

## 4. Strengths — What All Reviewers Agreed Works Well

1. **Controlled multi-dataset ablation design** is methodologically rigorous and fills a genuine gap in the CMAPSS literature (all three reviewers).
2. **Honest null results** for H5 and H7 — courageous and valuable for countering publication bias (A, C).
3. **FD003 inter-seed variance as diagnostic indicator** — most transferable conceptual contribution; underemphasised in current framing (A, C).
4. **M3 immunity to test-time cluster collapse** — deployment-relevant and structurally principled (A, C).
5. **Statistical rigor (Wilcoxon + BH-FDR + Cohen's d)** is well above CMAPSS literature norm, even with the implementation issues (B acknowledges the intent is correct).
6. **Acknowledged limitations** — synthetic data, single backbone, GMM labelling, H2/H5 model mismatch — all honest and appropriate (A).

---

## 5. Venue Fit Assessment

| Venue | Fit | Reasoning |
|-------|-----|-----------|
| **IEEE Trans. Industrial Electronics** | Good (post-revision) | Multi-dataset controlled ablation aligns with TIE scope; requires N-CMAPSS or backbone extension |
| **IEEE Trans. Reliability** | Good (post-revision) | Prognostics focus; null results valued; statistical rigor aligns with editorial expectations |
| **MSSP** | Moderate | Strong fit for M3 architecture contribution; synthetic data and 2018 backbone may concern area editors |
| **RESS** | Moderate | Values real-world applicability; CMAPSS-only paper needs N-CMAPSS pilot to survive review |
| **Information Fusion** | Weak | Multi-branch architecture could be reframed as fusion; requires stronger theoretical grounding |
| **Procedia CS (Conference)** | Good | Lower novelty bar; single-track venue appropriate for ablation study with null results |

**Recommended path:** IEEE Trans. Reliability or IEEE Trans. Industrial Electronics, after addressing 🔴 Critical items (A1–A7) and the highest-priority 🟡 items (A8–A19). Conference submission to Procedia CS is viable immediately after 🔴 items only.
