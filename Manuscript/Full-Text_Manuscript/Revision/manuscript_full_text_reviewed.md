# From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction

<!-- REVIEW SUMMARY — 2026-08-31
Overall assessment: The manuscript has improved materially: the 65.8% M3-superiority claim is gone, the strict three-tier framing has largely been replaced by attribution-based language, and the unified normalization result is acknowledged. It is not yet submission-ready because several claims in the Abstract, Results, Discussion, Limitations, and Conclusion remain mutually inconsistent.

CRITICAL items before submission:
1. Add the unified N1–N3 experiment explicitly to Methods and Results; it is currently invoked in the Abstract/Contributions without a reproducible protocol or complete result table.
2. Remove the causal attribution of the original FD003 variance to latent fault-mode mixing; the corrected protocol shows that the variance was protocol-sensitive.
3. Recompute H3 using a coherent two-sided test (paired if the same seed/split is shared). The reported p=0.0045 appears incompatible with the stated one-sided improvement test when the observed result is degradation.
4. Remove the claim that H4 significance was mathematically impossible. The stated 1/2^5 minimum does not describe scipy.stats.ranksums, and BH adjustment is not a simple multiplication by 96.
5. Remove deployment interpretations based on gate confidence >0.5. For a binary softmax, max(w0,w1) is mechanically >=0.5; without labels or a confidence–performance association, it is not evidence of useful routing.
6. Resolve the direct protocol contradiction: Results state that corrected H3 clips predictions to [0,125], whereas Limitations state that H3 did not.

Recommended positioning: a transparent four-factor attribution study showing robust upstream effects on selected datasets, protocol-sensitive normalization effects on FD003/FD004, and a clear reliability risk from trajectory-derived hard routing. Avoid presenting M3 or gate confidence as deployment-ready contributions.
-->

**Authors:** Young Seog Yoon¹\* (ORCID: 0000-0003-3796-8480), Eun Seo Lee¹, Hyeontae Kim¹, Ji Yeon Son²
**Affiliations:**
¹ Behavioral Intelligence for Autonomous Manufacturing Research Section, Electronics and Telecommunications Research Institute (ETRI), Daejeon, South Korea
² Autonomous Manufacturing Research Division, Electronics and Telecommunications Research Institute (ETRI), Daejeon, South Korea
**Correspondence:** isay@etri.re.kr (Young Seog Yoon)

> **Compiled:** 2026-07-06  
> **Last revised:** 2026-08-26 (GPT Rephrasing P1–P3 전 항목 반영: overclaim 완화, ≥0.8 처방 제거, 경제 수치 문단 삭제, "systemic reliability failure"→"most severe performance degradation", Conclusion 재작성, Cover Letter 일관성 반영)  
> **Status:** Complete draft — all sections revised and cross-checked for internal consistency  
> **Source sections:** `Manuscript/Sections/` (Abstract_draft, Introduction, Methodology, Results, Discussion_Implication, Conclusion)  
> **Section numbering note:** Sections are numbered I, III–VI; Section II (Related Work) is integrated into Sections I and V.

<!-- REVIEW — MINOR
Delete lines 9–13 and the final source-file note before submission. “GPT Rephrasing,” internal status, source directories, and numbering notes are production metadata, not manuscript content. Renumber III–VI sequentially or state the journal-compliant structure without an internal note.
-->

---

## Abstract

Reliable Prognostic and Health Management (PHM) systems for turbofan engines embed interdependent design decisions — RUL label clipping, sensor normalization, fault-mode architecture, and training loss function — whose contributions to predictive reliability are rarely isolated. This study presents a controlled ablation across all four NASA CMAPSS sub-datasets (FD001–FD004), covering five clipping thresholds, seven normalization strategies, four fault-mode architectures, and seven loss functions, with all comparisons Benjamini-Hochberg FDR-corrected.

Fleet min-max normalization outperforms all per-unit and instance-level alternatives on FD001, FD002, and FD004 under the primary seven-strategy screening protocol; a supplementary protocol-unified N1–N3 comparison confirmed cross-protocol robustness on FD001 and FD002 but found no detectable difference on FD004 (p_BH = 0.46), indicating protocol sensitivity on multi-condition datasets. The anomalously high inter-seed variance on FD003 (RMSE std = 12.86 vs. ≤1.84 elsewhere) resolved to std = 0.67 under a backbone-matched controlled protocol, indicating it reflects a training-protocol interaction rather than a normalization limitation. A controlled comparison of four fault-mode architectures shows that GMM-based hard partitioning significantly degrades RMSE on both multi-fault datasets (FD003: +156%; FD004: +76%), while soft and end-to-end attention routing avoid this degradation but provide no statistically detectable improvement over the single-model baseline. No custom loss achieves statistically detectable improvement over MSE after multiple-comparison correction (N = 5 seeds; minimum detectable d ≈ 2.0); removing RUL clipping inflates NASA prognostic scores by up to 306,000-fold regardless of loss design.

<!-- REVIEW — CRITICAL
The Abstract omits the positive unified FD003 result and calls the sensitivity “multi-condition,” although FD003 has one operating condition. Use “multi-fault datasets” or simply “FD003 and FD004.” Also remove “minimum detectable d ≈ 2.0” unless the exact power model is documented.

Suggested replacement:
“The seven-strategy screening favoured fleet min-max on FD001, FD002, and FD004. A protocol-unified N1–N3 comparison reproduced the fleet-level advantage on FD001 and FD002, additionally detected it on FD003 (p_BH=0.009), and found no detectable difference on FD004 (p_BH=0.46), indicating that the multi-fault datasets were protocol-sensitive. The original FD003 variance did not persist under the unified protocol and therefore cannot be attributed uniquely to fault-mode heterogeneity.”

Retain “significantly degrades” only after the H3 two-sided/paired analysis is recomputed.
-->

These findings suggest that upstream design decisions — RUL clipping and sensor normalisation — produce larger and more consistent performance differences than downstream choices — fault-mode routing and loss function — in the present CMAPSS experiments; gate confidence is proposed as a post-training diagnostic for routing decisiveness, with a practical design checklist summarising observed priorities as a structured starting point requiring validation on representative fleet data before operational deployment.

<!-- REVIEW — MAJOR
Delete gate confidence from the Abstract. The metric is not calibrated, has no ground-truth routing labels, and is not linked empirically to RMSE improvement. A safer ending is:
“These findings indicate that upstream target construction and normalization can exert larger and more consistent effects than downstream routing and loss customization under the evaluated settings, while trajectory-derived hard routing introduces a distinct reliability risk under train–test feature mismatch.”
-->

---

## I. Introduction

Remaining Useful Life (RUL) prediction for turbofan engines is a foundational task in Prognostic and Health Management (PHM), directly shaping safety-critical maintenance scheduling decisions in civil and military aviation. The NASA prognostic score reflects the underlying risk asymmetry explicitly: late predictions — where remaining life is over-estimated — carry an exponentially higher penalty than early ones, because an undetected engine failure costs orders of magnitude more than a precautionary shop visit. This asymmetric risk structure means that design choices in a turbofan PHM pipeline — how RUL targets are labelled, how sensor data are normalised, how multi-fault engines are handled — interact with the evaluation metric in ways that can constitute systemic reliability failures when misconfigured. The NASA CMAPSS benchmark [1], comprising four run-to-failure sub-datasets (FD001–FD004) under varying fault modes and operating conditions, has served as the canonical testbed for data-driven RUL estimation since 2008. As model architectures have evolved from linear regression [2] to stacked LSTM [4, 5] and attention-based encoders [7, 10, 11, 12, 41], reported RMSE on FD001 has fallen from above 25 cycles to below 13 [13]. Yet the pipeline design decisions responsible for these gains have rarely been isolated from one another [15], leaving a fundamental question unresolved: which design decisions — RUL label engineering, sensor normalisation, fault-mode architecture, and training loss function — are the primary drivers of predictive reliability, and in what priority order should they be addressed?

<!-- REVIEW — MAJOR
“systemic reliability failures” and “in what priority order” retain the old hierarchy framing. Suggested final question:
“Yet the individual effects and robustness boundaries of these pipeline decisions are rarely isolated, leaving unresolved which choices materially affect predictive reliability under controlled conditions.”
-->

Existing studies rarely disentangle individual design factors. The majority of the CMAPSS literature evaluates a single sub-dataset — most often FD001 — and bundles normalisation, loss function, and architecture into a single proposed system [8, 12, 13], making it impossible to attribute observed gains to any one component — a cross-study comparison problem that Ramasso and Saxena [3] documented after reviewing more than seventy publications, finding that absent benchmarking baselines and common dataset misunderstandings left multi-fault mode and operating-condition challenges unresolved despite years of collective effort. Deng et al. [25], for instance, reported RMSE = 13.91 on FD003 by combining CNN feature extraction, LSTM temporal modelling, and attention mechanisms into a single system; Asif et al. [38] showed that refined RUL labelling and sensor preprocessing alone substantially close the performance gap without architectural change. In both cases, it remains unclear whether the observed improvement originates from the preprocessing pipeline, the architectural design, or their interaction. Three specific gaps motivate this study: (i) no systematic comparison of normalisation strategies across all CMAPSS sub-datasets exists, despite the known operating-condition heterogeneity of FD002 and FD004 [9, 21]; (ii) the multi-fault structure of FD003/FD004 is acknowledged in prior work [6, 32] but rarely exploited through explicit architecture design; and (iii) the interaction between RUL clipping threshold and loss function has not been studied jointly, even though both directly shape the label distribution, gradient signal, and — as demonstrated in this study — the tail behaviour of the NASA prognostic risk metric, with misconfiguration capable of inflating prognostic scores by orders of magnitude.

This study addresses these gaps through a controlled ablation across all four CMAPSS sub-datasets. H1 uses linear regression (OLS) to isolate label-engineering effects from model capacity; this design choice ensures that the first-tier finding holds independently of backbone architecture — the label-misconfiguration failures quantified in §IV.A are not LSTM-specific effects. H2 uses a compact stacked LSTM (LSTM2 hidden=32). H3 and H4 use a full-capacity stacked LSTM (LSTM2 hidden=64). Within each hypothesis, all other factors are held constant while the design factor under study is varied, with all comparisons subject to Wilcoxon rank-sum tests corrected by Benjamini-Hochberg FDR (BH-FDR). Five RUL clipping thresholds (H1), seven normalisation strategies (H2), four fault-mode architectures (H3), and seven training loss functions (H4) were independently varied, enabling attribution of observed performance differences to the factor under study. The stacked LSTM was selected as the controlled backbone rather than attention-based alternatives for two reasons: (i) LSTM models represent the dominant baseline class in the CMAPSS literature [4, 5, 8], maintaining comparability with prior work; and (ii) pilot experiments found that attention mechanisms perform fault-mode separation implicitly, making the explicit early-cycle GatingNet (H3/M3) redundant over Transformer backbones — a finding that constitutes a separate research question (§IV.I) rather than a controllable factor within this ablation. The design-factor rankings observed below therefore apply specifically to stacked LSTM architectures.

<!-- REVIEW — MAJOR
Replace “first-tier finding” and “design-factor rankings.” Also, “pilot experiments found…” is an empirical claim without methods/results in the paper. Either report those experiments reproducibly or frame them as motivation/future work rather than a finding.

Suggested wording: “H1 tests whether the label effect is observable under a capacity-limited deterministic baseline; it does not by itself establish architecture-independence.” OLS evidence cannot prove that the same effect magnitude generalises to every backbone.
-->

H1 empirically confirms that clip = 125 cycles — the established literature standard [4, 5] — is optimal or statistically tied-optimal across all four sub-datasets. Omitting the clipping ceiling increased the FD003 NASA prognostic score by up to 306,000-fold, indicating that RUL label specification functions as a safety-critical design boundary in benchmark prognostic evaluation — a finding that motivates treating it as a resolved prerequisite rather than a tunable parameter [17]. Based on this result, clip = 125 is adopted as the first-tier prerequisite across all subsequent hypotheses, allowing each remaining design factor to be evaluated independently against a stable label-engineering baseline.

The main contributions of this paper are:

(i) A cross-dataset attribution study of four PHM pipeline design factors — RUL label clipping, sensor normalisation, fault-mode architecture, and loss function — showing that upstream target and data transformations (clipping, normalisation) produce larger and more consistent performance differences than downstream routing and loss-function modifications across the evaluated CMAPSS settings, with results expressed as a sequential design checklist for PHM pipeline configuration.

(ii) A seven-strategy normalisation ablation on CMAPSS, supplemented by a protocol-unified N1–N3 robustness analysis, showing that fleet-level min-max scaling significantly outperforms per-unit and RevIN strategies on FD001 and FD002 across both protocols (cross-protocol robust), with additional protocol-specific advantages on FD004 (original screening, p_BH=0.009) and FD003 (unified protocol, p_BH=0.009), and that FD003's anomalous inter-seed RMSE variance (std = 12.86 vs ≤1.84 elsewhere) reflects an interaction among backbone capacity, validation split method, and checkpoint-selection dynamics under the H2 experimental protocol — resolving to std = 0.67 under a backbone-matched controlled comparison — rather than a normalisation deficiency.

<!-- REVIEW — CRITICAL
The unified experiment compares N1 and N3 only, so it cannot establish cross-protocol superiority over RevIN. The sentence also attributes the FD003 variance to three mechanisms that were not individually ablated. Shorten and separate observation from mechanism.

Suggested replacement:
“(ii) A seven-strategy normalization screen, supplemented by a protocol-unified N1–N3 comparison, showing a cross-protocol fleet-level advantage on FD001 and FD002 and protocol-sensitive rankings on FD003 and FD004. The original FD003 variance did not persist under the unified protocol, identifying it as protocol-sensitive rather than a unique signature of normalization failure or latent fault structure.”
-->

(iii) A controlled comparison of four fault-mode architectures (M0–M3), showing that GMM-based hard partitioning significantly degrades prognostic performance on both multi-fault datasets (FD003: +156% RMSE; FD004: +76%), while soft and end-to-end attention routing avoid this degradation without statistically detectable improvement over the single-model baseline.

(iv) A confirmation across all four CMAPSS sub-datasets that no custom loss function achieves statistically detectable improvement over MSE after Benjamini-Hochberg FDR correction across 96 pairwise comparisons (6 loss functions × 4 clipping values × 4 sub-datasets), establishing loss function design as the lowest-priority tier of the risk-priority ordering. Removing RUL clipping inflated FD003 NASA Score by up to 306,000-fold regardless of loss design — confirming that label engineering subsumes loss function choice as the dominant control over tail-risk behaviour in the NASA prognostic metric.

<!-- REVIEW — MAJOR
Delete “lowest-priority tier” and avoid “confirmation” because H4 is power-limited. Suggested replacement:
“(iv) A four-dataset clipping–loss analysis in which no custom loss produced a statistically detectable improvement over MSE after BH-FDR correction, while removal of RUL clipping increased the FD003 prognostic penalty by several orders of magnitude. The loss comparisons are interpreted as power-limited null findings, not equivalence.”
-->

---

## III. Methodology

**Problem Formulation.** Engine $i$ ($i = 1, \ldots, N$) generates a multivariate sensor time series $\mathbf{X}^{(i)} \in \mathbb{R}^{T_i \times F}$, where $T_i$ is the run-to-failure lifetime in flight cycles and $F$ is the number of input features (varies by dataset and hypothesis; see §III.H). The model input at cycle $t$ is a fixed-length sliding window of 30 consecutive cycles:

$$\mathbf{W}_t^{(i)} = \mathbf{X}^{(i)}[t-29:t,\;:] \in \mathbb{R}^{30 \times F}$$

with front-zero-padding when $t < 30$. The RUL target follows a piecewise-linear formulation with clipping threshold $\tau$:

$$y_t^{(i)} = \min\!\bigl(T_i - t,\;\tau\bigr)$$

The prediction task is to learn $f : \mathbb{R}^{30 \times F} \to \mathbb{R}$ such that $f(\mathbf{W}_t^{(i)}) \approx y_t^{(i)}$ for all training cycles. At test time, $f$ is applied to the final observed window of each test engine and its output compared against the benchmark-provided ground-truth RUL. The four design variables studied in §IV are the clipping threshold $\tau$ (H1), the sensor preprocessing and normalisation strategy applied to $\mathbf{X}^{(i)}$ (H2), the architecture and routing mechanism of $f$ (H3), and the training objective used to optimise $f$ (H4).

### A. Dataset

The NASA CMAPSS benchmark was used, comprising four sub-datasets (FD001–FD004) simulating turbofan engine run-to-failure under varying operating conditions and fault modes (Table I). Each dataset records 21 raw sensor channels per flight cycle together with a held-out test set and ground-truth RUL values for the final observation of each test engine.

**Table I. CMAPSS sub-dataset characteristics.**

| Dataset | Train engines | Test engines | Op. conditions | Fault modes |
|---------|:------------:|:------------:|:--------------:|:-----------:|
| FD001 | 100 | 100 | 1 | HPC degradation only |
| FD002 | 260 | 259 | 6 | HPC degradation only |
| FD003 | 100 | 100 | 1 | HPC + Fan degradation |
| FD004 | 249 | 248 | 6 | HPC + Fan degradation |

Representative sensor degradation trajectories for FD003 — which contains both fault modes — are shown in Fig. 1, illustrating the visually distinct patterns that motivate fault-mode-aware modelling (H3) and fleet-level normalization (H2).

[FIGURE 1: Sensor Degradation Trajectories]
**Fig. 1.** Illustrative sensor degradation trajectories for representative engines in FD003 (single operating condition, two fault modes). Selected sensors with high RUL correlation (s2, s3, s4, s7, s11, s12) show visually distinct degradation patterns between HPC-fault and fan-fault engines, motivating both fault-mode-aware modeling (H3) and fleet-level normalization that preserves inter-engine degradation contrast (H2).

### B. Data Preprocessing

Seven constant-variance sensor channels were removed per dataset. For FD001 and FD003, sensors s1, s5, s6, s10, s16, s18, and s19 were discarded, leaving 14 input features; for FD002 and FD004, only s16 is constant, yielding 20 features. RUL targets followed a piecewise-linear formulation: for each training engine, cycles where the remaining life exceeds a clipping threshold τ receive a constant label of τ, while the final segment decreases linearly to zero. Five clipping values were evaluated (τ ∈ {75, 100, 125, 130, ∞}), with τ = 125 cycles serving as the standard baseline per established literature [4, 5]. Ground-truth test RUL values for each test engine's final observation are provided directly by the benchmark; test performance was evaluated by comparing model RUL predictions against these ground-truth labels. The piecewise labeling scheme and the effect of varying the clipping threshold are illustrated in Fig. 2.

[FIGURE 2: Piecewise Linear RUL Label and Clipping]
**Fig. 2.** Piecewise linear RUL labeling scheme with threshold-based clipping. The raw RUL decreases linearly from the maximum cycle but is clipped at threshold τ to account for the healthy phase. The shaded region illustrates the effect of varying τ ∈ {75, 100, 125, 130, ∞}; over-clipping truncates degradation information while removing the ceiling entirely permits unbounded targets that destabilize the NASA prognostic score.

### C. Operating-Condition Residualization

FD002 and FD004 contain six discrete operating conditions that shift absolute sensor levels by tens to hundreds of units. Although these conditions are identifiable via the three operating-condition columns (op1, op2, op3), those columns carry continuous floating-point values with cycle-level simulation noise — op1 takes 536 unique values and op2 takes 105 unique values across FD002 — making deterministic condition assignment by exact value matching infeasible. K-means (k = 6) was therefore applied to recover the six conditions unsupervised and to decouple degradation signals from operating-point offsets. A K-means model was fitted on the standardised op columns of the training set; per-cluster sensor means were computed from training data only and subtracted from each observation:

$$z_{i,j} = x_{i,j} - \mu_{c(i),\,j}$$

where c(i) denotes the cluster assignment of cycle i and μ_{c,j} is the training-set mean of sensor j within cluster c. This step precedes all normalization and is applied identically to training and test data using statistics derived exclusively from training engines. Post-hoc inspection confirms that each of the six K-means clusters maps to exactly one operating condition on both FD002 and FD004 (cluster purity = 1.0), validating that K-means recovers the true condition structure from the raw continuous op values. FD001 and FD003 present a single operating condition and therefore do not require this step.

### D. Normalization Strategies (H2)

Seven normalization strategies (N1–N7) were compared to evaluate the effect of the reference-statistics choice. Fleet-level methods compute statistics across all training engines: N1 applies min-max scaling to [0, 1] and N2 applies z-score standardisation. Per-unit methods (N3–N6) use each engine's own early-cycle observations as the reference baseline, removing initial-condition offsets before any degradation signal is visible: N3 and N5 apply min-max and z-score over the first 5 cycles; N4 and N6 apply the same transforms over the first 10 cycles.

<!-- REVIEW — CRITICAL
Add a reproducible subsection for the protocol-unified N1–N3 analysis: backbone, features, operating-condition columns, residualization, validation split, minimum epochs, evaluation clipping, seeds, comparison family, and exact run count. At present the Abstract and Contribution use unified results that are absent from Methods and Results. Also reconcile “50 runs”: 2 methods × 4 datasets × 5 seeds equals 40 runs unless additional conditions are included.
-->

N7 implements a forward-only variant of Reversible Instance Normalization (RevIN [20]) as a learnable module within the model. Each 30-cycle inference window is normalised by its instantaneous mean and standard deviation, with trainable affine parameters (γ, β). The inverse transform is not applied to the scalar RUL output: RevIN's inverse is designed to restore multi-step sensor forecasts to their original measurement units — an operation defined for outputs that live in sensor space. Scalar RUL predictions are not in sensor space; applying the inverse transform is therefore undefined for this output type. N7 accordingly implements only the forward (normalisation) pass of RevIN, making it equivalent to per-window instance normalisation with learnable affine parameters.

Formally, using the problem-formulation notation defined above: fleet-level statistics (N1, N2) aggregate over all training engines $i$ and all cycles $t$ per sensor $j$; per-unit statistics (N3–N6) aggregate over engine $i$'s first $K_0 \in \{5, 10\}$ cycles per sensor $j$; N7 aggregates over each 30-cycle inference window $\mathbf{W}_t^{(i)} \in \mathbb{R}^{30 \times F}$ per sensor $j$, with learnable affine output $\hat{x}_{\tau,j} = \gamma_j(x_{\tau,j} - \mu_{w,j})/\sigma_{w,j} + \beta_j$.

N1 served as the primary comparison baseline. All strategies were evaluated on the identical backbone with all other experimental factors fixed.

| ID | Name | Statistics axis |
|----|------|----------------|
| N1 | Fleet min-max | All engines $i$, all cycles $t$, per sensor $j$ |
| N2 | Fleet z-score | All engines $i$, all cycles $t$, per sensor $j$ |
| N3 | Per-unit min-max (5 cy) | Engine $i$, $t \le 5$, per sensor $j$ |
| N4 | Per-unit min-max (10 cy) | Engine $i$, $t \le 10$, per sensor $j$ |
| N5 | Per-unit z-score (5 cy) | Engine $i$, $t \le 5$, per sensor $j$ |
| N6 | Per-unit z-score (10 cy) | Engine $i$, $t \le 10$, per sensor $j$ |
| N7 | RevIN-style fwd-only (learnable) | $\mathbf{W}_t^{(i)} \in \mathbb{R}^{30 \times F}$, per sensor $j$ |

### E. LSTM Backbone Architectures

The stacked LSTM was selected as the controlled backbone for this ablation for three reasons. First, LSTM-based models constitute the dominant baseline class in the CMAPSS RUL literature [4, 5, 8], making results directly comparable with prior single-factor studies. Second, a controlled ablation requires the backbone to remain fixed across conditions; substituting an attention-based encoder would conflate backbone capacity with the design factor under study, preventing clean attribution of observed differences. Third, pilot experiments in which the backbone was replaced with a Transformer encoder or a self-attention LSTM found that attention mechanisms perform fault-mode separation implicitly — rendering the explicit early-cycle GatingNet (M3) redundant — a finding that constitutes a separate research question rather than a controllable variable within the present study. The design-factor rankings observed in §IV.A therefore apply specifically to stacked LSTM architectures; whether and how the tier ordering changes for attention-based backbones is identified as a priority open question in §IV.I.

H2 used a compact LSTM backbone and H3/H4 used a full-capacity backbone; within each hypothesis, the backbone was held fixed so that observed differences reflected only the design factor under study. Each network takes a sliding window of 30 consecutive cycles as input and produces a scalar RUL estimate:

```
H2 Backbone (compact):
Input  (B × 30 × 17_or_18_or_23)
→ LSTM₁ (hidden=64, returns full sequence) → Dropout(0.2)
→ LSTM₂ (hidden=32, returns last time step) → Dropout(0.2)
→ Linear(32 → 16) → ReLU → Linear(16 → 1)

H3/H4 Backbone (full-capacity):
Input  (B × 30 × 15_or_20)
→ LSTM₁ (hidden=64, returns full sequence) → Dropout(0.2)
→ LSTM₂ (hidden=64, returns last time step) → Dropout(0.2)
→ Linear(64 → 32) → ReLU → Linear(32 → 1)
```

LSTM₁ passes its full sequence of hidden states to LSTM₂, preserving temporal context across layers. Test sequences shorter than 30 cycles were zero-padded at the front. Input dimensions vary by hypothesis and dataset; see Table III in Section H.

**Note:** The difference in backbone capacity means that H2 and H3/H4 RMSE values are not directly comparable in absolute terms. Each hypothesis is evaluated internally relative to its own controlled baseline.

### F. Fault-Mode Architectures (H3)

Four architectures were compared on FD003 and FD004 to assess whether explicit fault-mode separation improves RUL accuracy. When two fault modes coexist within a dataset, a single-branch model must simultaneously minimise loss across both distributions, settling in a parameter compromise that fits neither mode optimally; routing engines to fault-specific branches eliminates this competition and allows each branch to specialise. For M1 and M2, this routing follows a fixed three-step pre-training pipeline: (1) degradation-slope features are extracted from seven discriminant sensors per engine; (2) a GMM is fit on training engines only to obtain cluster assignments; (3) each engine is assigned to a branch deterministically (M1) or blended by soft posterior weights (M2). M3 bypasses GMM fitting entirely, learning routing end-to-end from only the first K = 10 observed cycles.

**M0 (Baseline)** is a single-branch model using the shared backbone without any fault-mode handling.

**M1 (Hard Routing)** trains two independent LSTM branches. Each training engine was assigned deterministically to one branch via the argmax of its GMM posterior probability; test engines were routed identically. Branches were trained independently with standard MSE. A fairness-control variant (M1_kprefix) — in which GMM routing features were restricted to the same first K = 10 cycles used by M3's GatingNet — was evaluated separately to isolate information-horizon effects from architectural differences (§IV.C.2).

**M2 (Soft Gating)** retains two branches but replaces hard assignment with a weighted sum:

$$\hat{y}_{\text{final}} = p_0 \cdot \hat{y}_0 + p_1 \cdot \hat{y}_1$$

where [p₀, p₁] are the GMM posterior probabilities. The training loss augments the final MSE with auxiliary branch supervision:

$$\mathcal{L}_{\text{M2}} = \text{MSE}(\hat{y}_{\text{final}},\, y) + 0.1\cdot\text{MSE}(\hat{y}_0,\, y) + 0.1\cdot\text{MSE}(\hat{y}_1,\, y)$$

**M3 (Attention Gate)** eliminates the GMM entirely. A lightweight GatingNet reads only the first K = 10 observed cycles and produces end-to-end soft routing weights:

$$\text{GatingNet}: (B \times K \times F) \to \text{Flatten} \to \text{Linear}(K{\cdot}F \to 32) \to \text{ReLU} \to \text{Linear}(32 \to 2) \to \text{Softmax}$$

The final prediction is $\hat{y}_{\text{final}} = w_0\hat{y}_0 + w_1\hat{y}_1$ with training loss:

$$\mathcal{L}_{\text{M3}} = \text{MSE}(\hat{y}_{\text{final}},\, y) + 0.05\cdot\text{MSE}(\hat{y}_0,\, y) + 0.05\cdot\text{MSE}(\hat{y}_1,\, y)$$

By reading only early-cycle data, M3 avoids any dependence on late-cycle observations that are unavailable at real deployment time and is immune to the test-time cluster-distribution collapse that makes M1 unreliable on FD004. GatingNet confidence, defined as mean max(w₀, w₁) over the training set, is proposed as a post-training indicator of routing decisiveness.

**Table II. Computational complexity of H3 architectures (FD003, F = 15 features).**
*Parameter counts verified by direct model inspection. FLOPs computed analytically per inference window (batch = 1, window = 30 cycles) using the standard LSTM FLOPs formula: 8 × (input + hidden) × hidden per timestep. Inference latency measured on NVIDIA RTX GPU (2,000 runs, batch = 1, after 200-run warm-up; mean ± std reported). Training time is GPU compute per epoch on FD003 (~13,600 training sequences, batch = 256), excluding data loading. M1 routes each test engine to a single branch via GMM argmax, so its inference FLOPs equal M0's.*

| Model | Parameters | FLOPs / window | Inference (measured) | Training (GPU) |
|-------|-----------|---------------|---------------------|----------------|
| M0 (single branch) | 56.1K | 3.18M | 0.24 ± 0.05 ms | ~0.1 s/epoch |
| M1 (hard routing) | 112.3K | 3.18M† | 0.25 ± 0.06 ms† | ~0.2 s/epoch‡ |
| M2 (soft gating) | 112.3K | 6.37M | 0.57 ± 0.24 ms | ~0.2 s/epoch |
| M3 (attention gate, K=10) | 117.2K | 6.38M | 0.64 ± 0.21 ms | ~0.2 s/epoch |

†M1 activates one branch at inference (GMM argmax); both branches (112.3K) reside in memory but only one branch forward pass executes.  ‡M1 trains two branches sequentially; reported time is total per epoch.

M3's GatingNet adds only 4.9K parameters above M1/M2 (< 5% overhead), confirming that any performance differences across architectures are not attributable to additional model capacity. The approximately 2× parameter increase from M0 to M3 reflects the two independent prediction branches rather than the gating mechanism itself. All four architectures are well within the computational budget of embedded PHM controllers, which typically support models of up to several hundred thousand parameters.

For M1 and M2, GMM cluster assignments (k = 2, full covariance) were derived by unsupervised fitting on degradation-slope features of seven discriminant sensors identified by EDA (s15, s20, s21, s7, s12, s2, s4). The inter-cluster discriminability of each sensor is quantified by its inter-cluster z-score:

$$|\Delta z|_j = \frac{|\bar{\mu}_{c=1,j} - \bar{\mu}_{c=2,j}|}{\sigma_{\text{fleet},j}}$$

where $\bar{\mu}_{c,j}$ is the training-set mean of sensor $j$ within cluster $c$, and $\sigma_{\text{fleet},j}$ is the fleet-wide standard deviation of sensor $j$. Both branches in M2 and M3 use the full shared backbone.

### G. Loss Functions (H4)

Seven training loss functions were evaluated to determine whether asymmetric or dynamically weighted objectives improve over standard MSE. All functions share the signature `loss(pred, true, life_ratio=None)`, where life_ratio = 1 − RUL/RUL_max is a training-time weighting signal computed from training-set cycle counts only; it is set to None at inference time, eliminating any dependency on unknown test-set engine lifetimes.

| ID | Name | Key formulation |
|----|------|----------------|
| L1 | MSE (baseline) | (1/N) Σ (ŷ−y)² |
| L2 | NASA Score Loss | Differentiable approx. of s(d) |
| L3 | DynMSE | (1 + λ_dyn · r) · (ŷ−y)²; λ_dyn = 1.0 |
| L4 | Focal-RUL | (‖ŷ−y‖/(‖ŷ−y‖+1))^γ · (ŷ−y)²; γ = 2 |
| L5 | TWA | w_time · w_asym · (ŷ−y)²; λ_t = 10, λ_a = 1 |
| L6 | Pinball | τ·max(0, y−ŷ) + (1−τ)·max(0, ŷ−y); τ = 0.35 (default); τ = 0.25 (FD001/FD003 grid search) |
| L7 | HubA | Huber(δ=20) · w_asym; λ_a = 3 |

For L5 (TWA), w_time = 1 + λ_t · r and w_asym = 1 if d < 0, else λ_a. For L7, w_asym = 1 if d < 0, else λ_a, where d = ŷ − y. Key hyperparameters (τ, λ_t, λ_a, δ) were selected via grid search on FD001 before cross-dataset evaluation. FD001 therefore functions as a partially tuned evaluation dataset for H4; nominally strong FD001-specific effects (particularly L5 and L7) should be interpreted with this caveat. BH-FDR correction across all four datasets partially mitigates this optimism.

### H. Training Configuration

All models were optimised with Adam (learning rate 1×10⁻³, weight decay 1×10⁻⁴) with a batch size of 256, for a maximum of 100 epochs. Early stopping monitored validation loss with patience of 15 epochs and restored the best-performing checkpoint. The validation set was constructed by engine-level holdout: 20% of training engines were withheld, and all cycles of those engines were excluded from training. This prevented the RUL distribution mismatch that arises from cycle-level splitting. Each experimental configuration was run with five random seeds (0, 1, 2, 3, 4); results are reported as mean ± standard deviation across seeds.

<!-- REVIEW — MAJOR
This paragraph overgeneralises because H1 is deterministic and H2 uses a deterministic last-20%-by-ID split, while corrected H3 includes a 30-epoch minimum warm-up that is not stated here. Rewrite by hypothesis or explicitly list exceptions. The unified protocol must also be represented in Table III.
-->

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1×10⁻³ |
| Weight decay | 1×10⁻⁴ |
| Batch size | 256 |
| Max epochs | 100 |
| Early-stop patience | 15 (val loss) |
| Validation split | Engine-level, 20% of engines |
| Sliding window | 30 cycles |
| Random seeds | {0, 1, 2, 3, 4} |

The feature set and validation-split method differ across hypotheses, as shown in Table III. These differences were established independently during implementation and mean that the hypotheses are not directly cross-comparable in absolute RMSE.

**Table III. Feature set, validation split, and backbone by hypothesis.**

| Hypothesis | Feature set | Val split method | Backbone |
|-----------|------------|-----------------|---------|
| H1 | Sensors + op cols (OLS) | N/A (deterministic OLS) | Linear regression (OLS) |
| H2 | Sensors + op cols (incl. op1/op2/op3) | Last 20% by unit ID (deterministic) | Compact LSTM (LSTM₂ hidden=32) |
| H3 | Sensors only (no op cols) | Random 20% (per-run training seed) | Full LSTM (LSTM₂ hidden=64) |
| H4 | Sensors only (no op cols) | Random 20% (per-run training seed) | Full LSTM (LSTM₂ hidden=64) |

For H2 the resulting input dimension F is 17 (FD001: 14 sensors + 3 op), 18 (FD003: 15 sensors + 3 op), or 23 (FD002/FD004 after residualisation: 20 sensors + 3 op). For H3/H4 F is 14 (FD001), 15 (FD003), or 20 (FD002/FD004 after residualisation). **Note:** The feature and split differences between H2 and H3 reflect independent implementation choices made prior to analysis; they mean that the two hypotheses are not directly cross-comparable in absolute RMSE terms. Each hypothesis is interpreted relative to its own baseline condition.

### I. Evaluation Metrics

Two metrics were reported across all experiments. RMSE is the primary performance indicator:

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$$

The NASA prognostic score penalises late predictions more severely than early ones:

$$s(d) = \begin{cases} e^{-d/13} - 1 & d < 0 \text{ (early prediction)} \\ e^{d/10} - 1 & d \geq 0 \text{ (late prediction)} \end{cases}, \quad \text{NASA Score} = \frac{1}{N}\sum_{i=1}^{N} s(\hat{y}_i - y_i)$$

<!-- REVIEW — MAJOR
The conventional C-MAPSS NASA score is generally reported as the sum of engine penalties, whereas this equation reports the mean. If the mean is intentional, rename it “mean NASA penalty per test engine” and state that it is not directly comparable with published summed scores. If literature comparability is intended, remove 1/N and recompute absolute score values. Ratios are unaffected, but absolute claims and comparisons are not.
-->

where d = ŷ − y and N is the number of test engines. NASA Score is lower-is-better; a perfect prediction yields zero.

### J. Statistical Testing

Statistical comparisons differed in sample unit by hypothesis. For H1 (linear regression, deterministic), comparisons used a two-sided Mann-Whitney U test (`scipy.stats.ranksums`) on per-engine RMSE values (N ≈ 100–259 per dataset; see §III.K). For H2, H3, and H4 (LSTM, stochastic), comparisons used one-sided Wilcoxon rank-sum tests on per-seed aggregate metrics (N = 5), testing whether the treatment condition improves over baseline. All tests used α = 0.05 before correction. When multiple treatment conditions are compared simultaneously within one hypothesis, raw p-values were corrected using the Benjamini-Hochberg (BH) procedure at α_FDR = 0.05. A result was considered statistically meaningful when both p_BH < 0.05 **and** Cohen's d ≥ 0.3 (small effect threshold). Conditions satisfying the p-value criterion but yielding |d| < 0.1 are reported as "statistically significant but practically negligible" to distinguish statistical from practical significance. For context, a Cohen's d of 0.3 at the FD001 RMSE baseline of approximately 14–16 cycles corresponds to a mean RMSE difference of approximately 0.6–1.0 cycles — a gap comparable to one cycle of maintenance scheduling uncertainty in typical PHM deployment contexts.

<!-- REVIEW — CRITICAL
The one-sided “improvement” test is incompatible with declaring M1 degradation significant at p=0.0045. Inspect the code and recompute H3 with a pre-specified two-sided test. If the same seed produces matched splits across M0–M3, use a paired test (e.g., Wilcoxon signed-rank or paired permutation) and report the limited exact-p resolution at N=5. Reapply BH within a clearly defined H3 family.

Also clarify that scipy.stats.ranksums is not Mann–Whitney U (`mannwhitneyu` is the SciPy function for that test) and does not exploit pairing. Cohen's d should be paired d_z when observations are paired. Do not retain p=0.0045 until the test direction and dependence structure are resolved.
-->

### K. H1 Baseline Model (Linear Regression)

The RUL clipping study (H1) used ordinary least squares linear regression (`sklearn.linear_model.LinearRegression`, no regularisation) as its predictive model. Linear regression was chosen to isolate the effect of label engineering from non-linear model capacity: the clipping threshold's effect on RUL label distribution is architecture-independent, and a deterministic closed-form baseline eliminates random-initialisation variance. Each of the 20 experimental configurations (5 clip values × 4 datasets) was run exactly once; the model has no random state and produces identical results on identical data. The statistical comparison for H1 used the Mann-Whitney U test (`scipy.stats.ranksums`, unpaired, two-sided) on per-engine RMSE values (N ≈ 100–259 per dataset depending on sub-dataset). Note that this is technically an unpaired test; a paired Wilcoxon signed-rank test would be marginally more statistically efficient since the same test engines are evaluated under both clip conditions, and this limitation should be borne in mind when interpreting H1 significance levels.

---

## IV. Results

> **Models:** H1 = LinearRegression (deterministic, 20 runs); H2/H3/H4 = Stacked LSTM (5 seeds, mean ± std)  
> **Baseline:** clip = 125 cycles, Fleet min-max (N1), single-branch LSTM (M0), MSE (L1) throughout unless noted

### A. Effect of RUL Clipping (H1)

**clip = 125 cycles yielded the lowest or tied-lowest RMSE on all four sub-datasets.** The full matrix of results is presented in Table IV. On FD001 and FD003 (single operating condition), clip = 125 was the clear optimum (RMSE = 21.90 and 21.62, respectively). On FD002 and FD004, clip = 130 produced a marginally lower RMSE (Δ = −0.57 cycles on both), but the difference did not approach statistical significance (Mann-Whitney U, two-sided, p = 0.97 on each; BH-FDR applied). No tested alternative threshold achieved a statistically significant improvement over clip = 125, confirming its status as the practically validated standard [4, 5]. Aggressive clipping at clip = 75 was significantly inferior on all four datasets (Δ RMSE = +10.5 to +16.2 cycles; p ≤ 0.028), indicating that excessive truncation discards degradation signal in the upper RUL range. These findings were consistent with earlier single-dataset reports that converged empirically on 125 cycles [4, 5] and extend them to a rigorous four sub-dataset evaluation.

**Removing the RUL ceiling entirely was catastrophic on FD003.** At clip = None, the NASA prognostic score on FD003 reached 4,014,724 — a 306,000-fold increase over clip = 125 (13.09) — while FD004 exploded to 1,759 (35.4×). The mechanism is the asymmetric exponential NASA metric: late predictions are penalised by exp(d/10), which grows unboundedly as the model, trained on uncapped labels exceeding 500 cycles, systematically over-predicts RUL throughout the test trajectory.

RMSE was also significantly elevated on FD001 (31.90 vs 21.90; p = 0.006) and FD004 (46.99 vs 34.61; p = 0.002), though less extreme than the NASA Score collapse. FD002 was substantially more robust (RMSE = 33.05; p = 0.054), likely because its 260 training engines and six operating conditions provided sufficient distributional support for the model to learn a conservative bias without an explicit ceiling. FD003's disproportionate RMSE increase — 159% versus FD001's 46%, despite both having a single operating condition — may in part reflect FD003's bimodal fault-mode structure (Section C), where unbounded targets spanning two distinct degradation-lifetime populations could compound the over-prediction bias.

This dataset-dependent sensitivity confirmed that clip = 125 cannot be treated as universally optimal, but it remains the safest default across the full benchmark. Figure 3 presents the complete RMSE matrix.

[FIGURE 3: Effect of RUL Clipping on Prediction Accuracy]
**Fig. 3.** RMSE heatmap across four CMAPSS sub-datasets (FD001–FD004) and five clipping thresholds (mean over 20 runs, linear regression baseline). clip = 125 achieves the lowest or near-lowest RMSE in all datasets. The absence of clipping (clip = None) produces catastrophic RMSE on FD003 (56.1 cycles) where unbounded targets amplify early-life residuals exponentially.

### B. Effect of Normalization Strategy (H2)

**Normalization outcomes differ sharply between FD003 and the remaining three sub-datasets.** FD003 serves as the sole exception to an otherwise consistent pattern and is therefore addressed first.

On FD001, FD002, and FD004, fleet min-max (N1) achieved the lowest RMSE: 14.14 ± 0.22, 14.31 ± 0.10, and 14.60 ± 0.30 cycles, respectively. All per-unit strategies (N3–N6) were significantly inferior on FD001 (ΔRMSE = +3.8 to +6.6; p_BH = 0.011; d = 2.3–7.6) and FD002 (ΔRMSE = +1.2 to +4.2; p_BH = 0.011; d = 4.2–22.4). On CMAPSS, inter-engine sensor variation is modest relative to degradation magnitude; fleet-level statistics therefore preserve more discriminative information than engine-local references.

RevIN (N7) was likewise significantly inferior on FD001 (ΔRMSE = +0.78; p_BH = 0.011; d = 1.66), FD002 (+3.85; d = 20.0), and FD004 (+3.68; d = 6.19), despite its learnable affine parameters (Table V). RevIN [20] excels at multi-step forecasting under temporal distribution shift [24], but on CMAPSS the dominant challenge is extracting a shared degradation signature across engines — a context in which per-window adaptation is counterproductive. Fleet z-score (N2) showed no significant difference from N1 on FD001/FD002/FD003, indicating that the fleet-vs-per-unit distinction matters more than the specific scaling transform. Figures 4 and 5 visualise the full RMSE matrix and the pairwise statistical comparison against N1.

[FIGURE 4: RMSE Comparison Across Normalization Strategies]
**Fig. 4.** Mean RMSE heatmap (5 seeds) for seven normalization strategies (N1–N7) across four CMAPSS sub-datasets. Fleet min-max (N1) achieves the lowest RMSE on FD001 (14.14), FD002 (14.31), and FD004 (14.60). Per-unit methods (N3–N6) consistently underperform by removing between-engine degradation contrast. The anomalously high variance of N1 on FD003 (std = 12.86) is attributed to latent fault-mode mixing, resolved in H3.

<!-- REVIEW — CRITICAL
Delete the causal clause “attributed to latent fault-mode mixing, resolved in H3.” The corrected protocol narrows M0 variance without fault-mode separation, so H3 does not establish that routing resolved the anomaly. Caption the original-screen result descriptively and refer to the unified robustness subsection.
-->

[FIGURE 5: Statistical Significance of Normalization Differences (vs N1)]
**Fig. 5.** Pairwise statistical comparison of each normalization method against fleet min-max (N1) using the Wilcoxon rank-sum test with Benjamini-Hochberg FDR correction (α = 0.05). Effect sizes are reported as Cohen's *d*. Per-unit methods (N3–N6) and RevIN (N7) are significantly inferior on FD001, FD002, and FD004 (|*d*| ≥ 1.66 in all significant cases). FD003 shows no significant differences due to high seed-to-seed variance from mixed fault modes.

<!-- REVIEW — CRITICAL
Replace the final causal phrase with “under the original H2 protocol.” Mixed fault modes were not isolated as the cause. Add the unified N1–N3 table immediately after the original H2 screen; otherwise the manuscript foregrounds only the protocol that supports the stronger claim.
-->

**FD003 was the sole dataset where no normalization strategy separated from N1.** N1 achieved RMSE = 19.05 ± 12.86 on FD003 — the inter-seed standard deviation of 12.86 anomalously high relative to ≤ 1.84 on all other dataset-normaliser combinations — and all competitors showed p_BH ≥ 0.14. (Note: H2 N1/FD003 RMSE = 19.05 ± 12.86 and H3 M0/FD003 RMSE = 12.97 ± 0.67 reflect genuinely different experimental conditions — backbone architecture, input features, validation split method, and test-prediction clipping all differ between the two hypotheses (Table III). Each baseline is interpreted only against its own controlled conditions.)

Notably, however, this elevated variance was confined to N1 alone: the RevIN-style forward normalisation (N7) achieved 16.44 ± 0.36 and N4 achieved 17.35 ± 0.54, both with narrow distributions. N1's per-seed RMSE spanned approximately 7 to 43 cycles — a profile inconsistent with normalisation failure and indicative of a latent bimodal data structure rather than optimisation noise.

As detailed in Section C, FD003 contains two concurrent fault modes (HPC and fan degradation). The training sample drawn by each random seed determined which fault mode the single-branch model primarily learned, producing the observed inter-seed bimodality. This interpretation is supported by the fault-mode experiments below.

<!-- REVIEW — CRITICAL
These two paragraphs overstate causality and conflict with the unified/corrected result. A 7–43 range is fully compatible with training/checkpoint collapse; narrow variance under the corrected single-branch M0 weakens the “which fault mode was learned” explanation.

Suggested replacement:
“The variance spike was confined to N1 under the original H2 protocol and did not persist under the unified protocol. It is therefore interpreted as protocol-sensitive instability. FD003’s heterogeneous fault structure may contribute to this sensitivity, but the present experiments do not isolate its causal contribution.”
-->

### C. Fault-Mode Architectures (H3)

#### C.1 Unsupervised Cluster Quality

Before evaluating branch architectures, the presence of two distinct fault modes in FD003 and FD004 was verified. GMM clustering (k = 2) on combined late-cycle means and degradation slopes of seven discriminant sensors yielded Silhouette scores of 0.761 (FD003, AB_full variant) and 0.750 (FD004, AB_full), both well above the conventional quality threshold of 0.50 [31]. Slope-only clustering (AB_slope), which controls for possible life-length confounding, yielded Silhouette = 0.702 (FD003) and 0.683 (FD004), confirming that the two clusters reflect genuine sensor-trajectory differences rather than an artefact of unequal engine lifetimes. Sensor s15 (bypass pressure ratio) exhibited the largest inter-cluster z-score (|Δz| = 31.3), followed by s20 (HPT bleed, 16.0) and s21 (LPT bleed, 15.2), consistent with the known distinction between HPC-dominated and fan-dominated degradation pathways in CMAPSS FD003/FD004 [32]. Figure 6 summarises the cluster quality metrics across all feature variants.

[FIGURE 6: GMM Fault-Mode Cluster Quality (Phase 1)]
**Fig. 6.** GMM (*K* = 2) clustering results for FD003 and FD004. (a) Silhouette scores across three feature variants (AB_full, AB_slope, AB_late); AB_late achieves the highest separation (FD003: 0.858, FD004: 0.855). (b) BIC scores confirming *K* = 2 as the optimal cluster count. The AB_full variant (Silhouette ≥ 0.75 on both datasets) is used in Phase 2 to retain temporal diversity across the full engine lifetime.

#### C.2 FD003: Hard Routing Failure and Fault-Mode Architecture Results

**GMM hard-routing (M1) significantly degraded FD003 RMSE relative to the single-branch baseline.** Detailed results appear in Table VI. Under a corrected experimental protocol — using the per-run training seed for validation split, a 30-epoch minimum warmup, and evaluation clipping to [0, 125] cycles — M0 yielded RMSE = 12.97 ± 0.67 cycles and NASA Score = 497 ± 149. GMM hard-routing (M1) regressed substantially to RMSE = 33.25 ± 8.74 cycles (+156%; Wilcoxon p = 0.0045, d = 2.82). The large standard deviation (8.74 cycles) indicates that hard-routing quality was highly sensitive to the random training split: GMM labels derived from trajectory-level statistics produced markedly different branch assignments across seeds, leading to unstable per-seed RMSE outcomes.

<!-- REVIEW — CRITICAL
The effect magnitude is compelling, but “significantly” and p=0.0045 depend on the unresolved one-sided test. After recomputation, either report the corrected two-sided/paired p_BH or use “large and directionally consistent degradation” if exact N=5 inference cannot cross 0.05. Also distinguish raw p from BH-adjusted p.
-->

GMM soft-gating (M2) recovered to RMSE = 12.28 ± 0.57 cycles, comparable to M0 (p = 0.31; d = −1.06), confirming that weighted branch aggregation avoids the assignment instability of hard partitioning without providing statistically detectable improvement over the single-model baseline. M3, which replaced the GMM entirely with a lightweight GatingNet trained end-to-end on the first K = 10 observed cycles, achieved RMSE = 13.24 ± 1.69 cycles (NASA Score = 550 ± 283). M3 did not differ detectably from M0 (p = 0.75; d = 0.24); the 0.27-cycle mean difference is practically negligible. The inter-seed spread of M3 (std = 1.69) was slightly wider than M0 (std = 0.67), reflecting seed-to-seed variability in GatingNet random initialisation.

<!-- REVIEW — MAJOR
M2 has d=-1.06, so “comparable” and “confirming” are too strong despite p=0.31. Use: “M2 yielded a numerically lower RMSE than M0 (12.28 vs 12.97; d=-1.06), but the difference was not statistically detectable with five seeds.” Avoid interpreting non-significance as similarity.
-->

A post-hoc sensitivity analysis varying K ∈ {5, 10, 15, 20, 30} revealed that M3 RMSE was largely insensitive to the number of initial cycles used for routing. K = 5 already achieved RMSE = 14.23 ± 0.34, confirming that fault-mode identity is detectable from as few as five initial flight cycles. Figure 7 compares all four architectures on FD003 and FD004 in terms of RMSE mean and inter-seed variance.

<!-- REVIEW — MAJOR
Without ground-truth fault-mode labels, RMSE at K=5 cannot “confirm fault-mode identity.” It only shows that the end-to-end gate operates with a five-cycle prefix without a large additional RMSE penalty. Suggested wording: “K=5 yielded RMSE=14.23±0.34, indicating limited sensitivity of prognostic performance to prefix length within this benchmark.”
-->

GatingNet confidence on FD003 was 0.727 ± 0.165 (mean max(w₀, w₁) over training engines), indicating moderately decisive routing — consistent with the Silhouette score of 0.761 confirming fault-mode separability on this dataset.

[FIGURE 7: Fault-Mode Separation Model Comparison (M0–M3)]
**Fig. 7.** RMSE comparison (mean ± std, 5 seeds) of four fault-mode separation architectures on FD003 and FD004. M1 (GMM Hard Routing) significantly degrades to 33.25 ± 8.74 on FD003 (+156% vs M0 at 12.97 ± 0.67) and collapses to 33.28 ± 2.05 on FD004 (+76% vs M0 at 18.96 ± 3.97) due to test-time cluster assignment collapse (247:1 ratio). M2 and M3 recover to single-model baseline levels on both datasets; M3 achieves RMSE = 13.24 ± 1.69 on FD003 and 17.30 ± 1.04 on FD004, and did not differ detectably from M0 on both (p > 0.3).

#### C.3 FD004: Cluster Collapse Under Hard Routing

**Hard routing (M1) degraded below the baseline on FD004.** M0 achieved RMSE = 18.96 ± 3.97 on FD004; M1 regressed to 33.28 ± 2.05, a 75.5% increase (p = 0.0045, d = 3.64). Post-hoc inspection of test-time cluster assignments revealed the root cause: 247 of 248 test engines were assigned to the same branch by the GMM argmax, compared with an approximately balanced assignment during training. The root mechanism is a feature distribution shift at test time: the GMM was fitted on complete run-to-failure trajectories (computing late-cycle means and degradation slopes across each training engine's full life), but test-engine routing had to use only the last 30-cycle observation window. On FD004, which spans six operating conditions, the sensor statistics of a 30-cycle window vary substantially depending on which operating condition the engine encountered in its final cycles — a variation unrelated to fault mode. This operating-condition-driven variability overwhelmed the fault-mode signal in the GMM features, causing the decision boundary — learned on trajectory-level statistics — to assign virtually all test engines to a single branch. Soft-gating (M2) recovered at RMSE = 18.82 ± 1.56, comparable to M0 (p = 0.44; d = −0.06), confirming that soft weighting avoids the collapse without improving over the baseline.

M3 achieved RMSE = 17.30 ± 1.04 on FD004, and did not differ detectably from M0 (p = 0.31; d = −0.48). Because M3's GatingNet reads only the first 10 cycles of each engine — information that is equally available in training and test settings — it was immune to the test-time distribution collapse that undermined M1. On both FD003 and FD004, M2 and M3 successfully avoided the hard-routing failure but provided no statistically detectable improvement over the single-branch baseline. This result suggests that the early-cycle attention gate avoids a major reliability risk (catastrophic hard-routing degradation) without introducing the performance overhead of a dedicated fault-mode architecture for deployments where fault-mode structure is weak or early-cycle discriminability is low.

### D. Loss Function Comparison (H4)

**A statistical constraint shapes how H4 results should be read.** With N = 5 seeds per condition, the Wilcoxon rank-sum test cannot produce raw p-values smaller than 1/2⁵ = 0.031; after BH-FDR correction across 96 comparisons, this minimum raw p is insufficient to reach p_BH < 0.05. Statistical significance is therefore mathematically unachievable in this experimental design regardless of the true effect size (post-hoc power computation indicates minimum detectable Cohen's d ≈ 2.0 at this family size and seed count; see §V.E). H4 results are exploratory: patterns indicate directional signals, not confirmed effects.

<!-- REVIEW — CRITICAL
Replace this paragraph. The 1/2^5 bound does not describe a two-sample scipy.stats.ranksums test, and BH adjustment depends on ordered p-values rather than multiplying every p by 96. Therefore “mathematically unachievable regardless of effect size” is not established.

Suggested replacement:
“With only five seeds per condition and a large multiple-comparison family, H4 had limited power to detect anything other than large and consistent effects. Non-significant results are therefore interpreted as an absence of detectable improvement rather than evidence of equivalence.”

Document the power-analysis test, alpha, sidedness, pairing, correction assumption, and software before retaining d≈2.0.
-->

**No statistically detectable improvement from any custom loss function was found.** Across all 96 pairwise comparisons (6 loss functions × 4 clipping values × 4 datasets), zero reached p_BH < 0.05; the minimum corrected p-value was 0.176.

Some individual combinations showed nominally lower scores: L5 (TWA) at clip = 125 on FD001 achieved NASA Score = 3.40 ± 0.32 vs 5.66 ± 1.12 for MSE (d = −1.54); L7 (HubA) reduced RMSE to 14.90 ± 0.33 vs 16.34 ± 0.40 but worsened NASA Score (d = +2.00), because HubA's symmetric Huber penalty does not sufficiently penalise late predictions. Neither survived BH-FDR correction (p_BH = 1.0). Positive d denotes a worse NASA Score than MSE; negative d denotes improvement. The null result diverged from single-dataset gains reported by Rengasamy et al. [33, 35] because cross-sub-dataset BH-FDR substantially raises the significance threshold. Clipping threshold findings are independent of backbone architecture; H1 used linear regression (OLS) specifically to isolate label-engineering effects from model capacity.

The dominant driver of NASA Score variance across this study was RUL clipping, not loss function choice. Table VII illustrates this: at clip = 125, all seven loss functions converged to within a factor of 1.5× of one another in NASA Score on FD001 (range: 3.40–6.14). Removing the clip inflated FD003 NASA Score to between 3,124 (L6, Pinball) and 9,613,539 (L1, MSE), a span of three orders of magnitude that dwarfs any inter-loss difference at a fixed clip value. Even the loss functions designed to suppress late predictions (L5 TWA, L6 Pinball, L7 HubA) reduced but did not eliminate the clip = None catastrophe on FD003: L6 Pinball achieved NASA = 3,124 vs L1 MSE's 9,614,000 — a 3,000× improvement within the clip-free condition, yet still 630× worse than the worst result at clip = 125. This finding implies that loss-function engineering is a second-order design choice: it cannot substitute for proper label engineering (RUL clipping) as a mechanism for controlling the tail behaviour of the NASA penalisation function. Practitioners should fix the clipping threshold before considering custom loss functions. Figure 8 illustrates the clip-dominance effect across all seven loss functions and four clipping values.

[FIGURE 8: RUL Clipping × Loss Function Interaction (NASA Score)]
**Fig. 8.** Mean NASA prognostic score (lower is better) across all four datasets as a function of RUL clipping threshold and loss function (560 LSTM training runs, 5 seeds per configuration). clip = None yields catastrophic scores (up to 2.4 × 10⁶) regardless of loss function, while clip ∈ {125, 130} stabilises results for all losses. Among clipped configurations, no custom loss achieves statistically significant improvement over MSE after BH-FDR correction (α = 0.05).

---

## V. Discussion and Implications

### A. Which Design Choices Actually Move the Needle

Which design choices actually move the needle in turbofan RUL prediction? Across more than 800 training runs spanning four sub-datasets, seven normalization strategies, seven loss functions, and four architectural variants, the answer is consistent across all four CMAPSS sub-datasets: RUL clipping threshold and fault-mode architecture account for virtually all achievable variance in RMSE on this benchmark. Normalization strategy and loss function, by contrast, function as residual variables — each capable of degrading performance when misapplied, but neither capable of improving on a well-configured baseline by a statistically detectable margin once clipping is fixed and fault structure is handled. This ordering — clipping and architecture as load-bearing, normalization and loss as residual — has not previously been examined through a controlled cross-scenario evaluation and carries immediate implications for how practitioners and benchmark designers should allocate modelling effort [4, 5]. Because H2 (normalization) and H3 (architecture) were evaluated under different experimental configurations — backbone capacity, feature set, and validation protocol differ between the two hypotheses (§II.H, §IV.G) — the tier ranking reflects aggregated within-hypothesis evidence rather than a single unified cross-factor ablation; practitioners applying the tiers in sequence should validate the combined pipeline on their own data. The first tier is exempt from this caveat: H1's linear regression (OLS) baseline — a deterministic model with no capacity to compensate for label ceiling removal — demonstrates that label misconfiguration is a dataset-level reliability failure, not a backbone artefact; the 306,000-fold NASA Score inflation at clip = None generalises beyond the stacked-LSTM backbone used in H2–H4. The broader significance is methodological: single-dataset experiments, which constitute the majority of the CMAPSS literature [8, 12, 13], cannot isolate these hierarchical effects because a confound that is large on one sub-dataset may be negligible on another and vice versa.

<!-- REVIEW — CRITICAL
This is the old hierarchy and directly contradicts the new Abstract/Contribution, which elevate normalization and state that routing did not improve on M0. Rewrite the entire paragraph. Do not claim “virtually all achievable variance” without a common factorial model or variance decomposition, and do not claim OLS results “generalise beyond” LSTM.

Suggested core:
“The four studies identify different types of sensitivity rather than a universal ranking. RUL clipping produced the largest observed prognostic-penalty changes; normalization effects were robust on FD001/FD002 but protocol-sensitive on FD003/FD004; hard routing introduced large degradation; and no custom loss produced a detectable gain under the present power-limited design. Because the factors were not evaluated in a unified factorial experiment, these results support an attribution-oriented checklist rather than a strict quantitative hierarchy.”
-->

### B. Why Fleet Normalization Beats Per-Unit Strategies

On CMAPSS, inter-engine sensor variation is small relative to degradation magnitude — a property that consistently favours fleet-level statistics over engine-local references. Per-unit normalisation methods (N3–N6) subtract engine-specific initial-state baselines, an operation that removes genuine absolute degradation information when initial-condition variation across engines is small relative to fault progression magnitude. On a homogeneous synthetic fleet such as CMAPSS, a fleet-trained min-max scaler encodes a cross-engine map of the full degradation trajectory; engine-local normalisation replaces this shared map with a zero-mean engine-local reference, discarding the information about where an engine sits in the fleet distribution. The RevIN-style forward normalization [20] failure compounds this problem: RevIN's per-window mean subtraction applies the local-baseline removal at every inference step, progressively erasing the degradation trend across each 30-cycle window. Instance-normalisation approaches — including RevIN [20] and instance-normalisation flows for non-stationary time series [24] — demonstrate strong gains on multi-step forecasting tasks where distribution shift across training and test windows is the dominant challenge; on CMAPSS, the dominant challenge is extracting a common degradation signature from a fleet — a context in which instance-level adaptation is counterproductive rather than beneficial. A recent theoretical critique of RevIN [22, 23] reached a compatible conclusion: RevIN's components are redundant when the primary normalisation challenge is not temporal distribution shift but conditional offset. The four-dataset results are consistent with this theoretical prediction in the prognostics setting. Whether a heuristic affine rescaling of the scalar RUL prediction — applying the learnable parameters as $\hat{y}_{\text{rescaled}} = \gamma \cdot \hat{y}_{\text{norm}} + \beta$, as a global scaling without the theoretical motivation of restoring sensor units — would empirically recover some performance remains a targeted ablation not conducted in this study and may merit evaluation in future work.

The relative performance of fleet- and per-unit normalization is likely to depend on dataset characteristics. Practitioners working on real engine fleets where inter-engine manufacturing tolerance produces substantial initial-state variation — including variable rotor clearances, turbine blade wear, and sensor calibration offsets — may find per-unit strategies more competitive than the CMAPSS results suggest [25]. On CMAPSS, the observed level of inter-engine variation favours fleet-level statistics; this ranking may differ on other fleets. Inter-engine variability relative to degradation range should therefore be considered when selecting a normalization level. Future work could quantify how the relative performance of the two approaches varies with inter-engine variability — a direction requiring N-CMAPSS or real-fleet data with known manufacturing tolerances.

### C. Why FD003 Inter-Seed Variance Signals Latent Fault Structure

<!-- REVIEW — CRITICAL
Rename this section. The corrected evidence does not support the title’s causal claim. Suggested title: “C. Protocol Sensitivity of FD003 Inter-Seed Variance.”
-->

The N1/FD003 standard deviation of 12.86 cycles is not a modelling failure; it is consistent with the presence of heterogeneous fault-related structure in the training data. A single-branch LSTM trained on mixed-fault data converges to one of two local optima depending on which fault mode is overrepresented in the random training subsample drawn by a given seed, producing approximately bimodal per-seed RMSE outcomes. If this interpretation is correct, two patterns would be expected: first, separating fault modes should collapse the inter-seed variance towards single-fault benchmark levels; second, no analogous variance spike should appear on single-fault datasets (FD001/FD002). The second pattern was confirmed: no dataset-normalizer combination outside FD003 exceeded std = 1.84, consistent with FD003 being uniquely affected by fault-mode heterogeneity. Under the corrected H3 protocol, M0 itself achieved narrow std = 0.67 on FD003 once the training procedure was stabilised, indicating that the H2 variance anomaly (std = 12.86) originated from the training protocol rather than from an unresolvable bimodality in the data. M3 showed slightly wider spread (std = 1.69), reflecting seed-to-seed variability in GatingNet initialisation. GatingNet confidence on FD003 was 0.727 ± 0.165, indicating moderately decisive routing. More generally, unusually high inter-run variance may motivate an examination of latent data heterogeneity before it is attributed solely to optimisation variability. Inter-seed variance may therefore provide useful diagnostic information in addition to mean performance.

### D. What the Architecture Comparison Reveals About Fault-Mode Routing

**Hard GMM routing introduces substantial reliability risk; soft and attention-based routing avoid this degradation but provide no statistically detectable improvement over the single-model baseline.** M1's collapse on both FD003 (+156% RMSE) and FD004 (+76%) establishes that unsupervised trajectory-level GMM partitioning is an unreliable routing strategy under realistic train-test conditions. The failure mechanism differs between the two datasets: on FD003 the training-split sensitivity of GMM labels causes high seed-to-seed variance (std = 8.74), while on FD004 the operating-condition-driven feature distribution shift at test time causes near-complete assignment collapse (247:1 ratio). Both failure modes are attributable to the same root cause — GMM features derived from complete run-to-failure trajectories are unavailable or unreliable at test time.

M2 and M3 avoid this failure by design: M2 uses soft posterior weights (no hard assignment), and M3 reads only early-cycle observations that are identically distributed between training and test. The result is that M2 and M3 recover to single-model baseline performance on both datasets, with no statistically detectable improvement over M0 (all p > 0.3). This null improvement is not a failure of the routing concept per se; it reflects a well-calibrated single-model baseline (M0) once the training protocol is properly configured. Unsupervised fault-mode information, under these conditions, does not provide incremental predictive benefit beyond what a well-trained single-model already achieves.

**In FD003, fault-mode identity is detectable from early-cycle observations.** A K sensitivity analysis (K ∈ {5, 10, 15, 20, 30}) showed that M3 RMSE was largely insensitive to K, and K = 5 already achieved RMSE = 14.23 ± 0.34, confirming that fault-mode membership is encoded in initial sensor baseline state (s15 bypass pressure ratio, |Δz| = 31.3) rather than in degradation trajectory shape. Inter-seed variance decreased with K (std: 0.34 at K=5, 0.27 at K=30), suggesting that longer context improves routing stability but not mean accuracy; K = 5–10 was sufficient within the present CMAPSS experiments.

GatingNet confidence (mean max(w₀, w₁) over training engines) can be computed as a post-training indicator of routing decisiveness: FD003 yielded confidence = 0.727 ± 0.165 and FD004 yielded 0.676 ± 0.055. Both values are above 0.5 (pure random gating), with FD003 showing higher decisiveness consistent with its stronger Silhouette score (0.761 vs 0.750). This metric is available before deployment and can serve as a diagnostic for whether the GatingNet has identified a reliable early-cycle discriminant; if confidence remains near 0.5 after convergence, M3 has not found a useful routing signal and M0 may be more appropriate.

<!-- REVIEW — CRITICAL
For a two-class softmax, max(w0,w1) is always >=0.5 by construction; 0.5 is not an empirical “random-gating baseline.” Higher confidence can also reflect overconfidence or branch collapse. Without ground-truth mode labels or a demonstrated confidence–RMSE association, retain this only as a descriptive statistic and remove deployment language.

Suggested wording: “Mean maximum gate weight was 0.727±0.165 on FD003 and 0.676±0.055 on FD004. Because ground-truth routing labels and calibration targets are unavailable, these values are reported descriptively and are not interpreted as routing accuracy or deployment thresholds.”
-->

From an architecture standpoint, M3 is an instance of mixture-of-experts (MoE) inference [28, 39, 40] with an early-cycle context encoder as the gating network and no label supervision for the gate. Unlike RUL-QMoE [39], which targets probabilistic interval prediction for battery degradation using supervised material labels, and PSMMoEs [40], which applies metric-learning gating for cross-domain fault classification, M3 performs deterministic point-prediction RUL routing in a single unlabelled turbofan dataset. M3's practical advantage lies in its architectural efficiency: the GatingNet contributes only 4.9K parameters — less than 5% overhead above the two-branch baseline (Table II) — and requires no modification to the LSTM backbone. For practitioners operating existing LSTM-based prognostic pipelines, fault-mode-aware protection against hard-routing collapse is therefore achievable without backbone migration, at negligible additional computational cost. Because M3 uses an identical two-branch architecture on both FD003 and FD004, any capacity or ensemble benefit would appear equally on both sub-datasets; the statistical equivalence of M3 and M0 on both datasets provides an implicit cross-dataset capacity control, ruling out model-size artefacts as an explanation for any inter-model difference. Whether the gating principle extends to attention-based encoders [7, 11, 12] — where global attention may inherently capture early-cycle fault patterns through its query-key-value mechanism — is an open question addressed in Section H.

<!-- REVIEW — CRITICAL
“Statistical equivalence” remains and must be removed. Non-significance is not equivalence without a pre-specified margin and equivalence test. The capacity-control argument is also invalid: equal architecture across datasets does not rule out dataset-specific capacity or ensemble effects. Most importantly, M0 already avoids hard-routing collapse with lower complexity, so M3 is not demonstrated to provide practical protection beyond choosing M0.

Recommended positioning: M0 is the default; M2/M3 are alternatives that did not reproduce M1’s degradation but did not show incremental benefit.
-->

The early-cycle routing premise warrants an important caveat regarding broader applicability. Alternative unsupervised approaches to fault mode separation — notably UMAP-based trajectory clustering [42] and joint learning frameworks that operate over the full degradation history [43] — implicitly assume that fault mode identity becomes unambiguous only as degradation progresses, and they route or classify test units accordingly using all available observation context. This view is corroborated by causal-inference experiments on N-CMAPSS [44], which report that early anomalies are "difficult to identify due to complex thermodynamic couplings and nonlinear degradation patterns" in real-engine data; and by semi-supervised sensor selection work [45] showing that accurate fault mode recognition benefits substantially from partial supervisory labels. Taken together, these studies suggest that pure unsupervised early-cycle separation may be a property specific to the CMAPSS simulation environment rather than a universal prognostic principle.

M3's five-cycle discriminability on CMAPSS is traceable to a concrete data-generative property: the two fault modes differ primarily in absolute baseline sensor level (s15 bypass pressure ratio, |Δz| = 31.3) rather than in degradation rate or trajectory shape. In a real-engine fleet where fault modes manifest as progressive deviations from a shared healthy baseline — rather than as distinct initial operating points — the GatingNet faces a harder discrimination task, and the required early-cycle window K may be substantially larger or routing performance lower. Silhouette screening may provide one way to assess whether early-cycle separation is sufficiently distinct: if GMM clustering on K-cycle prefixes yields Silhouette < 0.5, early-cycle gating may be less appropriate and should be validated against full-trajectory routing strategies such as those in [42, 43] or a two-stage approach that begins routing only after a short burn-in window.

Similar early-cycle routing may be worth investigating in other PHM applications where fault modes differ in their initial sensor states — for instance, battery cells with different cathode compositions exhibiting divergent early-discharge voltage profiles, or induction motors with pre-existing stator-winding asymmetries detectable from start-up current signatures. The proposed screening measures could be evaluated as candidate pre-deployment indicators in such systems.

### E. Why Loss Functions Cannot Substitute for Label Engineering

H4's null result is best read as a power constraint: with N = 5 seeds and a 96-test BH family, no comparison can achieve statistical significance regardless of effect size — the finding is "no large effect was detectable" rather than confirmed equivalence. Post-hoc power computation indicates that the minimum detectable Cohen's d at this family size and sample count is approximately 2.0 — the study is powered only to detect very large effects, not the moderate effects (d ≈ 0.5–1.0) that practical loss-function improvements would typically be expected to produce. Despite this limitation, the observed effect sizes provide exploratory information: once RUL clipping is correctly applied at τ = 125, the MSE gradient already penalises late predictions disproportionately, because the clipped label distribution is right-skewed and CMAPSS test engines are predominantly in the low-RUL regime at the end of their trajectories. Asymmetric losses may therefore provide only an additional adjustment once clipping is applied. One possible interpretation is that clipping already introduces asymmetry into the target distribution, reducing the additional benefit of an explicitly asymmetric loss. These results raise the possibility that the apparent benefit of asymmetric losses in single-dataset studies [33, 34, 35] depends partly on the clipping strategy used in the baseline. Across the tested clip–loss combinations, clipping produced larger performance changes than loss-function selection, consistently across all four sub-datasets — a cross-term that has not been previously reported in the CMAPSS literature.

<!-- REVIEW — CRITICAL
Two issues: (1) remove the mathematically-impossible claim as above; (2) MSE is symmetric in positive and negative residuals. A right-skewed/clipped label distribution does not make the MSE gradient penalise late errors more than equally sized early errors. Therefore the proposed mechanism “clipping already introduces asymmetry” is not established.

Safer interpretation: clipping bounds the target range and reduces extreme overprediction exposure under the asymmetric evaluation metric; it does not make MSE itself asymmetric. Present any clipping–loss interaction as empirical and exploratory, not as a proven gradient mechanism.
-->

One exception deserves mention. Under clip = None, L6 (Pinball, τ = 0.25) reduced FD003 NASA Score from 9,614,000 to 3,124 — a 3,000-fold improvement relative to MSE within the clip-free condition, reflecting Pinball's quantile-regression property of producing conservative (early) predictions [36]. However, even this gain leaves FD003 630× worse than the worst result at clip = 125, confirming that no loss function can compensate for missing label engineering. Chung et al.'s [37] theoretical critique of blind Pinball minimisation reinforces this conclusion: calibrated quantile coverage requires the label distribution to be well-conditioned, which is not the case for an uncapped RUL target that can exceed 500 cycles.

### F. A Risk-Oriented Interpretation of PHM Pipeline Design Choices

The four hypotheses collectively indicate that upstream design decisions — RUL label clipping and sensor normalisation — produced larger and more consistent performance differences than downstream choices — fault-mode routing and loss function — across the present CMAPSS experiments. This section translates that finding into a practical design checklist. Each step is calibrated to CMAPSS and should be validated on fleet-specific data before operational commitment.

```
STEP 1 — Verify RUL Label Ceiling
  Confirm clip = 125 cycles (τ) is appropriate for the target fleet.
  ─────────────────────────────────────────────────────────────────────
  Domain evidence that fleet average lifetime differs substantially
  from 206–247 cycles (the CMAPSS range)?
    YES → consider adaptive clip = 70th-percentile fleet lifetime
    NO  → use τ = 125 (confirmed optimal across all CMAPSS sub-datasets)

STEP 2 — Compare Fleet vs Per-Unit Normalization
  Fit fleet min-max (N1) and per-unit min-max (N3) on training data.
  Evaluate on a held-out engine-level validation split.
  ─────────────────────────────────────────────────────────────────────
  Fleet normalization significantly better on validation split?
    YES → use fleet min-max (N1)
    NO  → assess whether inter-engine manufacturing variability
          exceeds within-engine degradation variance; if so,
          per-unit normalization may be more competitive

STEP 3 — Assess Multi-Fault Risk Before Choosing Architecture
  Default architecture: single-branch baseline (M0).
  Fit GMM (k = 2) on early-cycle sensor slopes; compute Silhouette S.
  ─────────────────────────────────────────────────────────────────────
  S >= 0.5? (latent fault structure may be present)
    YES → DO NOT USE hard GMM routing (M1): proved significantly
          harmful on both multi-fault CMAPSS sub-datasets
          (+156% RMSE on FD003; +76% on FD004)
        → Adopt M3-style early-cycle GatingNet ONLY IF a held-out
          validation set shows incremental RMSE benefit over M0.
          Gate confidence (mean max(w0,w1) > 0.5 after training?)
          can serve as a routing-quality diagnostic.

<!-- REVIEW — CRITICAL
Remove these two lines from the checklist. The >0.5 condition is tautological for a binary softmax maximum and is not a validated routing-quality criterion.
-->
    NO  → retain single-branch baseline (M0)

STEP 4 — Loss Function (lowest-priority factor)
  Use MSE (L1) as default.
  ─────────────────────────────────────────────────────────────────────
  No custom loss achieved statistically detectable improvement
  over MSE after BH-FDR correction in the present study.
  If deployment metric is NASA Score (asymmetric):
    → L7 (HubA, δ=20, λ_a=3) or L5 (TWA, λ_t=10)
      showed directional improvement on FD001; validate with >=10 seeds.
    ONLY adopt if BH-FDR-corrected gains survive on all target datasets.
```

A label engineering error — omitting or miscalibrating the RUL clipping threshold — inflates the NASA prognostic score by up to six orders of magnitude, constituting the most severe performance degradation observed across all experiment conditions. Normalisation strategy produced large and consistent RMSE differences on FD001–FD003 (fleet vs per-unit: ΔRMSE = 2.1–8.4 cycles; p_BH = 0.009). By contrast, no routing architecture (M0–M3) produced a statistically detectable RMSE improvement over the single-model baseline: M2 and M3 avoid the hard-routing collapse without improving over M0 (all p > 0.3). Loss-function selection showed no detectable effect after BH-FDR correction. The primary design value of M2 and M3 is therefore protection against the reliability risk introduced by hard GMM routing, not performance improvement over a well-configured baseline. This risk-protection perspective is conceptually analogous to severity-tier frameworks in reliability engineering (such as FMEA), though no formal methodological equivalence is claimed.

<!-- REVIEW — MAJOR
Qualify FD003 as a unified-protocol finding; it was not cross-protocol robust. Also, M2/M3 do not protect M0 from a risk it does not have. Replace the last two sentences with: “The results support M0 as the default architecture and identify trajectory-derived hard routing as an avoidable risk. M2 and M3 did not reproduce the degradation but showed no detectable incremental benefit over M0.”
-->

Gate confidence (mean max(w₀, w₁)) can be examined after GatingNet training as a diagnostic for routing decisiveness: FD003 yielded 0.727 ± 0.165 and FD004 yielded 0.676 ± 0.055, both above the random-gating baseline of 0.5. However, because ground-truth fault-mode labels are unavailable on CMAPSS, the relationship between confidence and routing accuracy cannot be established from these data alone; confidence should not be interpreted as a calibrated deployment threshold. If confidence remains near 0.5 after convergence, M3 has not found a reliable early-cycle discriminant and M0 should be preferred.

For practitioners encountering S ∈ [0.5, 0.65], this range may be considered inconclusive given the limited number of datasets evaluated. If fault-mode differences emerge only in late life (slope-driven rather than baseline-driven), the GatingNet may not reliably separate modes at K = 10 cycles; a larger K or a two-stage routing strategy should be evaluated. A held-out validation comparing M3 against M0 is advisable before committing to fault-mode routing in this marginal range.

### G. Limitations

**Several limitations constrain the generalisability of these findings.** First, and most critically, all experiments were conducted on the NASA CMAPSS simulation benchmark. Real turbofan engines exhibit sensor noise, calibration drift, maintenance-induced sensor resets, and operating-history effects that are absent from the synthetic CMAPSS environment. In particular, real fleets typically exhibit greater inter-engine manufacturing variability than CMAPSS, which may alter the fleet-vs-per-unit normalization ranking established in H2. The recently introduced N-CMAPSS dataset [14] addresses some of these gaps by including realistic degradation trajectories, variable flight conditions, and turbofan-specific sensor physics; whether the observed attribution-based priority ordering transfers to N-CMAPSS — and to real operational sensor streams — is the primary open question from this study and the most important direction for validation prior to industrial deployment.

Practitioners intending to apply this design priority ordering should follow a staged validation roadmap before committing to production: (1) replicate the ablation on N-CMAPSS to verify that tier rankings survive realistic sensor noise and variable flight profiles; (2) validate GMM cluster assignments against post-teardown fault records on at least one fleet sample to confirm that unsupervised routing corresponds to physically distinct failure modes rather than a statistical partition; and (3) measure inter-engine sensor coefficient of variation on the target fleet — if substantially higher than the CMAPSS baseline (~5%), per-unit normalisation strategies should be re-evaluated before committing to fleet min-max at Tier 1.

<!-- REVIEW — MAJOR
Old terms remain: “tier rankings” and “Tier 1.” Replace with “factor-specific findings” and “fleet min-max.” Verify the ~5% CMAPSS coefficient-of-variation figure or remove it if it is not reported in Methods/Results.
-->

Second, all hypotheses except H1 used a common two-layer stacked LSTM backbone; architectural choices and design-factor rankings may interact differently with Transformer-based or graph-neural-network encoders [7, 10, 11, 12], which have shown competitive performance on CMAPSS in recent work.

Third, H3 relied on unsupervised cluster quality (Silhouette ≥ 0.5) as a proxy for genuine physical fault categories, because CMAPSS provides no fault-mode labels. Whether M3's routing corresponds to HPC vs. fan degradation in a physically meaningful sense, or merely to a statistical partition, cannot be verified from the benchmark data alone.

Fourth, the H1 clipping analysis employed LinearRegression (OLS, no regularisation) for computational tractability, while H2 and H4 used the LSTM backbone; direct cross-hypothesis RMSE comparisons therefore confound model complexity with the design factor under study, and the absolute RMSE values in Table IV should not be compared with those in Tables V–VII.

Fifth, within-study comparisons between H2 and H3 were constrained by implementation differences not unified prior to analysis: H2 used a compact backbone (LSTM₂ hidden=32, FC 32→16→1) and 17–23 input features (sensors plus op1/op2/op3; dataset-dependent — see Table III), while H3/H4 used a full-capacity backbone (LSTM₂ hidden=64, FC 64→32→1) and 15 features (sensors only). H2 used a deterministic validation split (last-20% of engines by unit ID) while H3 used a random split (per-run training seed). H2 clipped test predictions to [0, 125] at evaluation; H3 did not. These differences mean that the H2 FD003 baseline RMSE (19.05 ± 12.86) and H3 M0 baseline RMSE (12.97 ± 0.67) reflect genuinely different experimental conditions; cross-hypothesis absolute RMSE comparisons should not be taken at face value.

<!-- REVIEW — CRITICAL
Direct contradiction: Results C.2 says corrected H3 used evaluation clipping to [0,125], but this paragraph says H3 did not. Correct Table III/Methods/Limitations to reflect the final code. Also explain how the protocol-unified N1–N3 analysis addresses these differences and which discrepancies remain.
-->

Sixth, the current implementation assumed a static batch-trained model deployed without further adaptation. In practice, sensor characteristics and fleet composition evolve over time; online learning mechanisms — such as periodic fine-tuning triggered by newly labeled flight cycles — would be required to maintain routing accuracy over extended operational periods. M3's lightweight GatingNet is architecturally compatible with such periodic updates, and this extension is reserved for future work.

Seventh, M3's early-cycle routing relies on fault modes being distinguishable from initial baseline sensor state rather than from accumulated degradation patterns. Approaches that leverage full degradation trajectories for fault mode identification [42, 43] — including UMAP-based clustering over complete run-to-failure histories and joint learning frameworks that process the entire temporal sequence — suggest that the early-cycle discriminability observed on CMAPSS may not generalise to real-engine environments where initial sensor readings are contaminated by noise, calibration offsets, or operational variability. Causal-inference experiments on N-CMAPSS [44] explicitly characterise early anomaly detection as difficult in realistic flight conditions, and semi-supervised frameworks [45] report that partial fault-mode labels substantially improve mode recognition accuracy beyond what unsupervised early-cycle signals alone can provide. Practitioners should therefore verify early-cycle cluster discriminability (e.g., Silhouette ≥ 0.5 on K-cycle prefixes) before adopting K-cycle gating in new deployment contexts, and should consider full-trajectory routing alternatives when this threshold is not met.

### H. Industrial and Deployment Implications

The observed attribution-based priority ordering may provide a useful structure for reliability risk management in turbofan PHM deployment. Neglecting label engineering (clip = None) inflates the NASA prognostic score by up to 306,000-fold on FD003 — equivalent in practice to a prognostics system that chronically over-predicts remaining life and delays maintenance intervention, the highest-cost failure mode in turbofan fleet operations where an undetected in-flight shutdown typically costs orders of magnitude more than a precautionary shop visit. This finding highlights the importance of RUL clipping calibration as a design decision that should precede architectural or loss-function work.

The fault-mode architecture comparison reveals that GMM hard-routing introduces substantial reliability risk — quantified at +156% RMSE on FD003 and +76% on FD004 — while soft and attention-based routing avoid this degradation without adding measurable architectural cost relative to the single-model baseline. Because M3's GatingNet reads only the first ten flight cycles, fault-mode routing may be assessable at or near engine commissioning — before significant degradation has accumulated — with no burn-in observation window required on CMAPSS-class data. Gate confidence (mean max(w₀, w₁) over training engines) can be examined before deployment as an indicator of routing decisiveness (FD003: 0.727; FD004: 0.676 in the present experiments); a value near 0.5 signals that the GatingNet has not found a reliable early-cycle discriminant and M0 may be more appropriate. On computational scale, K-means residualization and GMM fitting scale linearly with engine count; Mini-Batch K-means variants are expected to substantially reduce fitting time for fleets of thousands of engines, though empirical runtime profiling at operational scale has not been conducted and remains a direction for validation.

<!-- REVIEW — CRITICAL
Remove the commissioning/deployment recommendation and gate-confidence rule. M3 did not improve on M0, routing correctness is unobserved, and CMAPSS initial-state separability may be simulation-specific. Also “without adding measurable architectural cost relative to M0” is inaccurate: M3 adds branches/parameters/FLOPs even if overhead is modest. State the measured overhead numerically instead.
-->

The H4 results suggest that extensive loss-function tuning may be a lower experimental priority than label and architecture choices, although the comparison is power-limited by the within-hypothesis design. Engineering effort is more productively directed to clipping calibration or multi-fault screening before extensive loss-function search. This priority ordering thus provides a structured starting point: verify labelling first, address fault-mode structure second, and treat loss selection as a later-stage refinement subject to cross-dataset validation.

---

### I. Future Research Directions

**Three directions emerge directly from the limitations and null results of this study.**

<!-- REVIEW — MINOR
Five directions follow, not three. Change to “Five directions” or consolidate the subsections.
-->

**Adaptive RUL clipping.** Our results confirm that a single threshold is not universally optimal: clip = 130 is marginally better on FD002/FD004 (ΔRMSE = −0.57, non-significant) while clip = 125 is unambiguously best on FD001/FD003, and the dataset-specific lifetime distributions differ (FD001/FD002 mean ≈ 206 cycles; FD003/FD004 mean ≈ 247 cycles). A data-driven adaptive threshold — for instance, set at the 70th-percentile engine lifetime in the training fleet — would be principled, require no manual tuning, and directly address the H1 research gap identified in the literature [16, 18]. An adaptive scheme would also handle the clip = None catastrophe on FD003 automatically, since the 70th-percentile of FD003 lifetimes is approximately 137 cycles, close to the empirically validated 125.

**M3 on real-data and N-CMAPSS benchmarks.** Validating the early-cycle GatingNet on N-CMAPSS (2021), which includes sensor noise, maintenance resets, and variable flight-envelope profiles, would test whether the fault-mode routing principle transfers beyond the CMAPSS simulation environment. N-CMAPSS also provides turbofan health parameter labels (HPT/Fan degradation) that would allow the cluster-quality proxy to be replaced by supervised routing accuracy, enabling a direct comparison between M3's unsupervised gating and a fully supervised fault-mode classifier. A parallel question is whether M3's explicit GatingNet remains necessary when the backbone itself is replaced with an attention-based encoder [7, 10, 11, 12]; global attention mechanisms may inherently perform early-cycle fault separation, in which case the attribution-based priority framework's architectural factor would require reformulation for Transformer-class models. Comparing M3's unsupervised gating against a supervised fault-mode classifier — given ground-truth HPT/Fan health labels available in N-CMAPSS — would quantify the reliability cost of label-free routing: the RMSE gap between M3 (RMSE = 13.24 ± 1.69 on FD003) and the supervised upper bound constitutes the measurable cost of unsupervised deployment. Separately, the internal design choices of the GatingNet — hidden units (currently 32), auxiliary loss weights (0.05), and Softmax normalisation — were not ablated within this study; a targeted sensitivity analysis of these parameters is a natural extension before adapting M3 to new deployment contexts. An orthogonal design option not evaluated in this study is using GMM posterior probabilities directly as auxiliary input features — appended to the sensor window — rather than as routing weights; comparing this feature-augmentation approach against M2 and M3 would quantify whether soft routing and feature-based fault-mode representations yield different inductive biases for RUL prediction.

**Deep learning backbone exploration and ensemble strategies for real operational data.** The attribution-based priority ordering was validated on a stacked LSTM backbone applied to a controlled simulation benchmark. As industrial PHM systems move towards deployment on real sensor streams, empirical exploration of diverse backbone architectures — including Transformer encoders [7, 10, 11, 12], temporal convolutional networks, and graph neural networks — is necessary to establish whether the design hierarchy and the architectural ranking generalise across model families. Real operational data from heterogeneous fleets introduces irregular degradation patterns, sensor calibration drift, and distributional shifts that may interact differently with each architecture; no single backbone can be assumed optimal without validation on representative operational data. Beyond single-model approaches, ensemble and blending strategies that combine predictions across multiple architecture families are likely to be required for robust coverage across diverse fault types, operating regimes, and fleet compositions encountered in practice. The four-factor design checklist provides a structured starting point — establishing label engineering and fault-mode screening as prerequisites — but the specific architectural choice should be treated as an open variable to be resolved empirically for each new deployment context.

**Scalable mixture-of-experts with unknown k.** M3 currently assumes exactly k = 2 fault modes, a strong prior that may not hold for heterogeneous industrial fleets. Extending the gating mechanism to an unknown and unbounded number of fault modes via Bayesian nonparametric priors — for example, a Dirichlet process mixture model on early-cycle sensor trajectories [30] — would eliminate the cluster-count hyperparameter and adapt gracefully to fleets with diverse fault histories. Recent work on Bayesian nonparametric process mixtures for unlabelled failure modes [30] provides a natural prior framework for this extension.

**Loss × clipping interaction study.** The 4 × 7 cross-tabulation presented here reveals that clipping dominates, but the sample size per cell (5 seeds × 4 datasets) is insufficient to detect small interaction effects. A targeted study that varies clipping continuously (e.g., τ ∈ {90, 100, 110, 120, 125, 130, 140} cycles) and pairs each level with theoretically motivated losses could isolate whether any custom loss provides a statistically detectable benefit after optimal clipping is identified for each sub-dataset. This design would directly resolve the confound identified by comparing the null result of this study with the single-dataset gains reported by Rengasamy et al. [33] and Abdullah [19]. As a lower-cost intermediate step, a targeted 15–20 seed re-run on FD001 alone would resolve whether asymmetric losses produce a detectable single-dataset effect that disappears under cross-sub-dataset BH-FDR correction, providing direct evidence for or against the asymmetric loss hypothesis on the most-studied sub-dataset.

---

## VI. Conclusion

Four PHM pipeline design factors — RUL label clipping, sensor normalisation, fault-mode architecture, and loss function — were evaluated in isolation across all four NASA CMAPSS sub-datasets (FD001–FD004), with all comparisons Benjamini-Hochberg FDR-corrected. H1 confirmed that clip = 125 cycles — the established standard [4, 5] — produces the lowest or statistically tied-lowest RMSE on all four sub-datasets; removing the RUL ceiling inflates FD003 NASA Score by 306,000-fold, establishing label engineering as a resolved prerequisite throughout the remaining analyses.

Fleet-level min-max normalisation significantly outperformed all per-unit and RevIN strategies on FD001, FD002, and FD004. On FD003, the anomalously high inter-seed variance observed under fleet normalisation (std = 12.86) resolved to std = 0.67 under a backbone-matched controlled protocol, indicating it reflects an interaction among backbone capacity, validation split method, and checkpoint-selection dynamics rather than a normalisation deficiency or solely a fault-mode heterogeneity effect. A controlled comparison of four fault-mode architectures (M0–M3) revealed that GMM-based hard partitioning significantly degraded RMSE on both multi-fault datasets (FD003: +156%; FD004: +76%), while soft and end-to-end attention routing avoided this degradation without providing statistically detectable improvement over the single-model baseline. Gate confidence (mean max(w₀, w₁)) is proposed as a post-training indicator of routing decisiveness — values above 0.5 indicate that the GatingNet has found a useful early-cycle discriminant, with FD003 (0.727) showing higher decisiveness than FD004 (0.676), consistent with the stronger fault-mode separability on that dataset.

<!-- REVIEW — CRITICAL
The Conclusion must report both protocols: robust N1 advantage on FD001/FD002 and protocol-sensitive results on FD003/FD004. Do not attribute the variance to specific mechanisms not isolated experimentally. Delete the gate-confidence sentences. Retain “significantly degraded” only after the H3 test is corrected.
-->

No statistically detectable improvement from custom loss functions was found after Benjamini-Hochberg correction across 96 pairwise comparisons (6 losses × 4 clips × 4 datasets). It should be noted that with N = 5 seeds and a 96-test BH family, this study was powered only to detect very large effects (minimum detectable Cohen's d ≈ 2.0); the null result should be interpreted as 'no large effect was detectable' rather than confirmed equivalence. The dominant source of variance in all prognostic metrics was RUL clipping rather than loss design. Together, the four hypotheses suggest an attribution-based priority ordering: upstream design decisions — RUL label clipping and sensor normalisation — produced larger and more consistent performance differences than downstream choices — fault-mode routing and loss-function modifications — across the present CMAPSS experiments. Silhouette score and gate confidence serve as sequential diagnostic indicators (not calibrated thresholds) for the routing decision. This ordering may guide where PHM research and engineering effort is most productively directed.

<!-- REVIEW — MAJOR
Retain the power limitation only after documenting the calculation; otherwise say “limited power with five seeds and extensive multiplicity.” Remove gate confidence as a diagnostic indicator. “Dominant source of variance” implies a formal variance decomposition that was not performed; use “largest observed performance changes.”
-->

Collectively, three findings carry direct implications for PHM practice. First, label engineering is a safety-relevant design decision: miscalibrated RUL clipping produced the most severe performance degradation observed across all experiment conditions, dominating all other design choices, and must be resolved before architectural or loss-function work begins. Second, GMM hard-routing is a reliability risk on multi-fault datasets; soft and early-cycle attention routing avoid catastrophic failure at negligible computational overhead, and gate confidence (mean max(w₀, w₁)) can be examined before deployment as a diagnostic for routing decisiveness. Third, custom loss functions offer no statistically recoverable benefit once clipping is correctly applied, suggesting that engineering effort is better redirected to the higher-priority tiers. The controlled structure of this ablation also provides an attribution framework for interpreting gains reported by state-of-the-art bundled systems [12, 25, 38]: the evidence indicates that correct label engineering and fault-mode routing account for the majority of systematic performance variation on CMAPSS, while architectural depth and loss-function choice are secondary factors that become distinguishable only after higher-priority tiers are resolved. Taken together, the four-factor design checklist — label engineering, sensor normalisation, fault-mode architecture, and loss function — offers evidence-based design priorities for PHM system configuration, where upstream misconfigurations produce qualitatively larger performance consequences than downstream ones.

<!-- REVIEW — CRITICAL
This paragraph reintroduces the rejected hierarchy and overstates fault-mode routing: routing did not improve over M0. Remove “higher-priority tiers,” “majority of systematic performance variation,” gate-confidence deployment language, and “negligible” unless quantified against M0. A defensible conclusion is that hard routing is avoidable risk, while M0 remains the default and loss findings are power-limited.
-->

---

## Funding

This work was supported by the National IT Industry Promotion Agency (NIPA) grant funded by the Ministry of Science and ICT (No. XXXX, Development of an Integrated Full-Lifecycle Operation Platform Technology for Ensuring Manufacturing Data Reliability).

---

## Data and Code Availability

The NASA CMAPSS dataset used in this study is publicly available at the NASA Prognostics Center of Excellence Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). No data were generated or modified; the benchmark is used as-is under its public access terms. The experimental code, including all preprocessing pipelines, model implementations (M0–M3), and statistical testing routines, will be made publicly available on GitHub upon acceptance of this manuscript.

---

## References

[1] A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," in *Proc. 1st Int. Conf. Prognostics and Health Management (PHM)*, Denver, CO, 2008.

[2] F. O. Heimes, "Recurrent Neural Networks for Remaining Useful Life Estimation," in *Proc. 1st Int. Conf. Prognostics and Health Management (PHM)*, IEEE, 2008, doi: 10.1109/PHM.2008.4711422.

[3] E. Ramasso and A. Saxena, "Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets," *Int. J. Prognostics Health Manage.*, vol. 5, no. 2, 2014, doi: 10.36001/ijphm.2014.v5i2.2236.

[4] S. Zheng, K. Ristovski, A. Farahat, and C. Gupta, "Long Short-Term Memory Network for Remaining Useful Life Estimation," in *2017 IEEE Int. Conf. Prognostics and Health Management (ICPHM)*, pp. 88–95, doi: 10.1109/ICPHM.2017.7998311.

[5] X. Li, Q. Ding, and J.-Q. Sun, "Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks," *Rel. Eng. Syst. Safety*, vol. 172, pp. 1–11, 2018, doi: 10.1016/j.ress.2017.11.021.

[6] A. L. Ellefsen et al., "Remaining Useful Life Predictions for Turbofan Engine Degradation Using Semi-Supervised Deep Architecture," *Rel. Eng. Syst. Safety*, vol. 183, pp. 240–251, 2019, doi: 10.1016/j.ress.2019.01.016.

[7] Y. Mo et al., "Remaining Useful Life Estimation via Transformer Encoder Enhanced by a Gated Convolutional Unit," *J. Intell. Manuf.*, vol. 35, pp. 1997–2012, 2021, doi: 10.1007/s10845-021-01750-x.

[8] R. Jin et al., "Bi-LSTM-Based Two-Stream Network for Machine Remaining Useful Life Prediction," *IEEE Trans. Instrum. Meas.*, vol. 71, 2022, doi: 10.1109/TIM.2022.3163761.

[9] D. Xu et al., "Spatio-Temporal Degradation Modeling and Remaining Useful Life Prediction Under Multiple Operating Conditions Based on Attention Mechanism and Deep Learning," *Rel. Eng. Syst. Safety*, vol. 225, 2022, doi: 10.1016/j.ress.2022.108648.

[10] Y. Zhang et al., "Trend-Augmented and Temporal-Featured Transformer Network with Multi-Sensor Signals for Remaining Useful Life Prediction," *Rel. Eng. Syst. Safety*, vol. 235, 2023, doi: 10.1016/j.ress.2023.109258.

[11] H. Wang et al., "Comprehensive Dynamic Structure Graph Neural Network for Aero-Engine Remaining Useful Life Prediction," *IEEE Trans. Instrum. Meas.*, vol. 72, 2023, doi: 10.1109/TIM.2023.3312337.

[12] K. You et al., "A 3-D Attention-Enhanced Hybrid Neural Network for Turbofan Engine Remaining Life Prediction Using CNN and BiLSTM Models," *IEEE Sensors J.*, vol. 24, no. 3, 2024, doi: 10.1109/JSEN.2023.3335994.

[13] S. M. Elsherif, B. Hafiz, M. A. Makhlouf, and O. Farouk, "A Deep Learning-Based Prognostic Approach for Predicting Turbofan Engine Degradation and Remaining Useful Life," *Sci. Rep.*, 2025, doi: 10.1038/s41598-025-09155-z.

[14] F. Wu et al., "Remaining Useful Life Prediction Based on Deep Learning: A Survey," *Sensors*, vol. 24, no. 11, art. 3454, 2024, doi: 10.3390/s24113454.

<!-- REVIEW — CRITICAL CITATION
The manuscript cites [14] as the primary N-CMAPSS dataset source, but [14] is a survey. Replace/add the actual N-CMAPSS dataset paper and cite it where the dataset is described. Verify all 2025–2026 references and avoid relying on an anonymous arXiv item [23] for a central theoretical claim.
-->

[15] H. Li et al., "A Review on Physics-Informed Data-Driven Remaining Useful Life Prediction: Challenges and Opportunities," *Mech. Syst. Signal Process.*, vol. 209, 2024, doi: 10.1016/j.ymssp.2024.111120.

[16] F. Imbert, T. Adewumi, and H. Han, "A Novel Preprocessing-Driven Approach to Remaining Useful Life (RUL) Prediction Using Temporal Convolutional Networks (TCN)," in *IEEE 37th Int. Conf. Tools Artif. Intell. (ICTAI 2025)*, 2026, doi: 10.1109/ICTAI66417.2025.00160.

[17] K. Ensarioğlu, T. İnkaya, and E. Emel, "Remaining Useful Life Estimation of Turbofan Engines with Deep Learning Using Change-Point Detection Based Labeling and Feature Engineering," *Appl. Sci.*, vol. 13, no. 21, art. 11893, 2023, doi: 10.3390/app132111893.

[18] A. Srinivasan, J. C. Andresen, and A. Holst, "Ensemble Neural Networks for Remaining Useful Life (RUL) Prediction," *Asia Pacific Conf. PHM Society*, vol. 4, no. 1, 2023, doi: 10.36001/phmap.2023.v4i1.3611.

[19] M. E. B. Abdullah, "Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL Prediction with Interpretable Failure Heatmaps," arXiv preprint arXiv:2604.13459, April 2026.

[20] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift," in *Int. Conf. Learning Representations (ICLR)*, 2022.

[21] Z. Zhang et al., "A Framework for Predicting the Remaining Useful Life of Machinery Working under Time-Varying Operational Conditions," *Appl. Soft Comput.*, 2022.

[22] G. Berthelier et al., "On the Role of Reversible Instance Normalization," arXiv preprint arXiv:2603.11869, 2026.

[23] Anonymous, "Noise or Signal? Deconstructing Contradictions and An Adaptive Remedy for Reversible Normalization in Time Series Forecasting," arXiv preprint arXiv:2510.04667, October 2025.


[24] Z. Sun et al., "IN-Flow: Instance Normalization Flow for Non-Stationary Time Series Forecasting," in *Proc. 31st ACM SIGKDD Conf. Knowledge Discovery and Data Mining*, 2025, doi: 10.1145/3690624.3709260.

[25] S. Deng et al., "Prediction of Remaining Useful Life of Aero-Engines Based on CNN-LSTM-Attention," *Int. J. Comput. Intell. Syst.*, 2024, doi: 10.1007/s44196-024-00639-w.

[26] D. Xu, J. Shang, C. Jiang, X. Shang, H. Qiu, and L. Gao, "A Novel Multi-Task Learning Framework with Fault Mode Feature Separation for Remaining Useful Life Estimation of Mechanical Systems," *Adv. Eng. Informatics*, vol. 64, art. 103053, 2025, doi: 10.1016/j.aei.2024.103053.

[27] T. Lodygowski and S. Szrama, "Unsupervised Classification and Remaining Useful Life Prediction for Turbofan Engines Using Autoencoders and Gaussian Mixture Models: A Comprehensive Framework for Predictive Maintenance," *Appl. Sci.*, vol. 15, no. 14, art. 7884, 2025, doi: 10.3390/app15147884.

[28] Y. Liu, B. Xu, and Y.-a. Geng, "Multi-Condition Remaining Useful Life Prediction Based on Mixture of Encoders," *Entropy*, vol. 27, no. 1, art. 79, 2025, doi: 10.3390/e27010079.

[29] Z. Peng, Q. Wang, Z. Liu, and R. He, "Remaining Useful Life Prediction for Aircraft Engines under High-Pressure Compressor Degradation Faults Based on FC-AMSLSTM," *Aerospace*, vol. 11, no. 4, art. 293, 2024, doi: 10.3390/aerospace11040293.

[30] K. Fu, S. S. Disanayaka Mudiyanselage, C. Dai, and M. Kim, "Prognostics of Multisensor Systems with Unknown and Unlabeled Failure Modes via Bayesian Nonparametric Process Mixtures," arXiv preprint arXiv:2602.19263, February 2026.

[31] J. Cohen, X. Huan, and J. Ni, "Fault Prognosis of Turbofan Engines: Eventual Failure Prediction and Remaining Useful Life Estimation," *Int. J. Prognostics Health Manage. (IJPHM)*, 2023, doi: 10.36001/ijphm.2023.v14i2.3486.

[32] H. Özcan, "Interpretable Ensemble Remaining Useful Life Prediction Enables Dynamic Maintenance Scheduling for Aircraft Engines," *Sci. Rep.*, 2025, doi: 10.1038/s41598-025-23473-2.

[33] D. Rengasamy, M. Jafari, B. Rothwell, X. Chen, and G. P. Figueredo, "Deep Learning with Dynamically Weighted Loss Function for Sensor-Based Prognostics and Health Management," *Sensors*, vol. 20, no. 3, art. 723, 2020, doi: 10.3390/s20030723.

[34] D. Rengasamy, H. Bhatt, B. Rothwell, X. Chen, and G. P. Figueredo, "Asymmetric Loss Functions for Deep Learning Early Predictions of Remaining Useful Life in Aerospace Gas Turbine Engines," in *2020 Int. Joint Conf. Neural Networks (IJCNN)*, 2020.

[35] Z. Liu et al., "A Multi-Head Neural Network with Unsymmetrical Constraints for Remaining Useful Life Prediction," *Adv. Eng. Informatics*, 2021.

[36] R. Diao et al., "Turbofan Engine Remaining Useful Life Prediction with Reliable Prediction Intervals via LSTM-Based Quantile Regression and Conformal Calibration," *Sensors*, vol. 26, no. 7, art. 2249, 2026.

[37] Y. Chung et al., "Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.

[38] O. Asif et al., "A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset," *IEEE Access*, 2022.

[39] S. Ly, R. Yang, N. Dixit, and H. D. Nguyen, "RUL-QMoE: Multiple Non-crossing Quantile Mixture-of-Experts for Probabilistic Remaining Useful Life Predictions of Varying Battery Materials," arXiv preprint arXiv:2512.23725, December 2025.

[40] B. Yang, J. Zhang, R. Liu, D. Lin, P. Li, and C. L. P. Chen, "Point-to-Set Metric-Gated Mixture of Experts for Multisource Domain Adaptation Fault Diagnosis," *IEEE Trans. Neural Netw. Learning Syst.*, Early Access, March 2025, doi: 10.1109/TNNLS.2025.3548894.

[41] Z. Fan, W. Li, and K.-C. Chang, "A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine Remaining Useful Life Prediction," *Sensors*, vol. 24, no. 3, art. 824, 2024, doi: 10.3390/s24030824.

[42] Y. Fu, Z. Feng, J. Zhou, and Q. Shi, "Degradation Modeling and Prognostic Analysis Under Unknown Failure Modes," *IEEE Trans. Autom. Sci. Eng.*, 2024, doi: 10.1109/TASE.2024.3371143.

[43] D. Wang, Q. Zhao, B. Yang, and K.-L. Tsui, "Joint Learning of Failure Mode Recognition and Prognostics for Degradation Processes," *IEEE Trans. Autom. Sci. Eng.*, vol. 20, no. 4, pp. 2790–2802, 2023, doi: 10.1109/TASE.2022.3197094.

[44] K. Dong, Z. Liu, Y. Zhang, and J. Liu, "Causal Inference-Based Fault Diagnosis and Abnormal Degradation Detection for Aero-Engine," in *2025 Int. Conf. Equipment Intelligent Operation and Maintenance (ICEIOM)*, 2025.

[45] Y. Wang, X. Liu, C. Sun, R. Yan, and X. Chen, "Deep Learning-Based Sensor Selection for Failure Mode Recognition and Prognostics Under Time-Varying Operating Conditions," *IEEE Trans. Autom. Sci. Eng.*, vol. 22, pp. 6993–7007, 2025, doi: 10.1109/TASE.2024.3469052.

---

*End of manuscript. Source files in `Manuscript/Sections/`. Figures and tables in `Manuscript/Figures/` and `Manuscript/Tables/`. Citation RIS files in `Manuscript/Citation/`.*
