# Proposed Structure for Section II. Related Work

## 1. Recommendation

Creating a dedicated **Section II. Related Work** is recommended. It resolves the current numbering gap between Sections I and III and, more importantly, separates three functions that are presently mixed across the Introduction and Discussion:

1. explaining why design-factor attribution matters;
2. reviewing how prior studies treated each design factor; and
3. positioning the present controlled ablation against the unresolved gaps.

The section should be created by **redistributing and consolidating existing literature discussion**, rather than merely adding more text after the current Introduction. This avoids duplication and keeps the manuscript focused on controlled design-factor attribution rather than on proposing a new state-of-the-art model.

## 2. Recommended Manuscript Flow

1. **Introduction:** Why design-factor attribution matters in turbofan RUL prediction
2. **Related Work:** How prior studies treated the four factors and what remains unresolved
3. **Methodology:** How the four factors are isolated experimentally
4. **Results:** Which factors materially affect performance and reliability
5. **Discussion:** How the findings modify, qualify, or challenge prior assumptions
6. **Conclusion:** Upstream configuration priorities and the reliability risk of hard routing

This produces a coherent progression:

> Benchmarking problem → prior approaches and assumptions → unresolved attribution gap → controlled experiments → reliability-oriented interpretation

## 3. Recommended Section II Structure

### II.A. Benchmarking and Design-Factor Attribution in RUL Prediction

This subsection should establish the overarching problem: many C-MAPSS studies combine preprocessing, architecture, and loss design into a single proposed system, making the source of reported improvement difficult to identify.

Relevant material includes:

- the C-MAPSS benchmarking and reproducibility issues discussed by Ramasso and Saxena [3];
- bundled CNN–LSTM–attention systems such as Deng et al. [25] and DA-LSTM Shi et al. (2024, RESS, 156 citations) ★NEW;
- studies showing that label and preprocessing changes alone can substantially affect performance, such as Asif et al. [38]; and
- the broader distinction between predictive performance and controlled causal attribution.

**★NEW — 재현성 문제 (Freitas et al., 2026):** A reproducibility study of 21 Transformer-based FD001 models found that AGATT (originally reported RMSE = 11.45) reproduced as 15.99 ± 1.39 under 30-run execution — confirming that single-seed RMSE values systematically underrepresent central tendency. This directly motivates the present study's five-seed repeated design and BH-FDR correction.

Recommended central sentence:

> Existing studies demonstrate strong predictive performance, but their bundled experimental designs make it difficult to determine whether the reported gains originate from target construction, normalization, architecture, or loss design; and single-seed evaluation protocols introduce reproducibility concerns that further complicate cross-study comparison.

The purpose is not to discount previous performance gains, but to show that **factor-level attribution remains unresolved** and that **statistical reliability of reported results is an independent concern**.

### II.B. RUL Label Construction and Sensor Normalization

This subsection should review the upstream design choices evaluated in H1 and H2:

- piecewise-linear RUL labeling and the conventional 125-cycle clipping threshold [4, 5];
- prior work on clipping or adaptive label construction [16–18];
- fleet-level, per-unit, and instance-level normalization;
- RevIN and non-stationary time-series normalization [20, 22–24];
- operating-condition clustering and residualization for FD002/FD004 [9, 21]; and
- **★NEW — Ruvaifa et al. (2026, *Array*):** the most systematic preprocessing ablation to date, comparing sensor selection criteria, sequence-length design, and normalization strategies across all four sub-datasets. Key finding: preprocessing impact exceeds architectural complexity in many cases. **Gap:** no pairwise statistical significance testing; normalization strategy is one of several factors rather than the primary controlled variable; operating-condition–normalization interaction on FD002/FD004 is not isolated.

Recommended gap statement:

> Although clipping and normalization are routinely adopted, and Ruvaifa et al. (2026) recently demonstrated their importance, no prior study has conducted pairwise statistical comparison of fleet-level, per-unit, and instance-level normalisation strategies across all four C-MAPSS sub-datasets under a controlled backbone-matched protocol with multiple-comparison correction.

The Related Work section should not state in advance that fleet normalization is superior. It should explain **why a controlled comparison is necessary**; the empirical ranking and protocol sensitivity belong in the Results.

### II.C. Multi-Fault Prognostics and Routing Architectures

This subsection should connect the multi-fault characteristics of FD003/FD004 with alternative routing architectures:

- multi-fault C-MAPSS challenges [6, 32];
- clustering-based fault-mode separation;
- hard routing and posterior-weighted soft routing;
- mixture-of-experts and learned gating [28, 39, 40];
- **★NEW — Xiong et al. (2023, *RESS*, 67 citations):** adaptive framework combining physics-informed FM classifier with DCNN and operating-condition-aware smoothing; ~7% RUL accuracy improvement on multi-fault sub-datasets. Relevant as a more principled FM-recognition approach than hard GMM routing, but does not isolate the effect of routing accuracy from the smoothing component;
- full-trajectory clustering and joint fault-recognition/prognosis methods [42, 43]; and
- limitations of early fault-mode recognition [44, 45].

The logical progression should be:

1. fault-specific experts may reduce competition between heterogeneous degradation modes;
2. this potential benefit depends on reliable routing;
3. deployment limits the trajectory information available at routing time; and
4. the reliability boundary among hard, soft, and end-to-end routing remains insufficiently tested.

Recommended gap statement:

> Prior studies suggest that fault-mode-aware modelling can improve specialization, but the reliability boundary between hard partitioning, posterior-weighted routing, and end-to-end gating has not been systematically evaluated under matched prognostic conditions.

This prepares the reader for the eventual finding that hard GMM routing degrades performance, while M2/M3 avoid the degradation without detectably outperforming M0.

### II.D. Loss Engineering and Asymmetric Prognostic Risk

This subsection should cover:

- MSE as the conventional baseline;
- NASA-inspired asymmetric objectives;
- time-weighted and dynamically weighted losses;
- Pinball or quantile loss;
- asymmetric Huber-type objectives; and
- prior single-dataset claims of loss-function improvement [33–37].

The principal unresolved issue is not whether custom losses exist, but whether their reported gains remain after the RUL clipping condition is controlled.

Recommended gap statement:

> Existing loss-function studies typically assume a fixed RUL-labeling scheme, leaving unresolved whether reported gains remain detectable after the clipping threshold is properly controlled and multiplicity across datasets is considered.

This framing allows H4's null result to function as a direct answer to a literature gap rather than as an unsuccessful model-development attempt.

### II.E. Research Gap and Positioning of This Study

This final subsection should synthesize the preceding literature rather than introduce another group of studies. The current Introduction paragraph beginning with “Three specific gaps motivate this study” can be moved here and expanded.

The research gaps can be summarized as follows:

0. **(★NEW — Cross-cutting) Reproducibility and multi-seed evaluation:** Single-seed RMSE reporting — standard across virtually all CMAPSS literature — has been shown to systematically underrepresent central tendency (Freitas et al., 2026), making cross-study comparisons of factor-level effects statistically unreliable without repeated-run designs.
1. **Label clipping and normalization** are not consistently controlled before architecture and loss comparisons, preventing clean attribution of reported gains.
2. **Normalization strategies** have not been compared with pairwise statistical significance testing across all four sub-datasets; Ruvaifa et al. (2026) approached this but without BH-FDR correction.
3. **The reliability boundary among hard, soft, and end-to-end routing** has not been directly evaluated under matched prognostic conditions on both multi-fault CMAPSS sub-datasets.
4. **The clipping–loss interaction** has not been tested under cross-dataset multiple-comparison correction, leaving open whether reported single-dataset loss gains survive the first-tier prerequisite.

Recommended closing research question:

> Accordingly, this study asks which pipeline design choices materially alter predictive reliability, which introduce avoidable failure risks, and which provide no detectable incremental benefit under controlled, multi-seed C-MAPSS experiments with multiple-comparison correction.

This sentence should lead directly into Section III and the H1–H4 experimental design.

## 4. Revised Role of the Introduction

After creating Related Work, the Introduction should be shortened to approximately five paragraphs:

1. importance of RUL prediction and reliability-oriented maintenance;
2. progress in C-MAPSS research and the problem of bundled pipeline decisions;
3. the central design-factor attribution question;
4. a concise overview of the H1–H4 experimental structure; and
5. the four contributions.

Detailed descriptions of Deng et al., Asif et al., RevIN, MoE, and custom-loss studies should move to Section II. The study purpose and contribution statements should remain in the Introduction.

The current result-oriented statement:

> H1 empirically confirms that clip = 125 cycles...

should be replaced in the Introduction with a design-oriented statement:

> H1 first evaluates the RUL clipping threshold using a capacity-limited deterministic baseline. The selected threshold is then fixed in subsequent hypotheses so that normalization, architecture, and loss effects can be evaluated against a stable label-engineering condition.

Numerical findings and confirmatory language should remain in the Abstract and Results.

## 5. Material to Move from the Discussion

The following literature-heavy material in the current Discussion can be relocated or shortened:

- comparisons between M3 and prior MoE approaches [28, 39, 40];
- full-trajectory clustering and joint-learning methods [42, 43]; and
- evidence concerning the difficulty of early fault recognition [44, 45].

The division of roles should be:

- **Related Work:** explain what prior methods assume and how they operate;
- **Discussion:** explain how the present empirical findings support, qualify, or challenge those assumptions.

For example, Related Work may state that some methods rely on full degradation histories, while Discussion should interpret the M1_kprefix result as evidence that correcting the information horizon alone does not recover hard-routing performance.

## 6. Recommended Final Table of Contents

```text
I. Introduction

II. Related Work
    A. Benchmarking and Design-Factor Attribution in RUL Prediction
    B. RUL Label Construction and Sensor Normalization
    C. Multi-Fault Prognostics and Routing Architectures
    D. Loss Engineering and Asymmetric Prognostic Risk
    E. Research Gap and Study Positioning

III. Methodology
IV. Results
V. Discussion and Implications
VI. Conclusion
```

## 7. Length and Writing Guidance

- Target approximately **1,200–1,600 words** for Section II.
- Reuse and consolidate existing literature discussion to avoid increasing the total manuscript length substantially.
- Organize by research problem and methodological assumption rather than listing studies chronologically.
- End every subsection with an explicit unresolved issue that motivates the corresponding hypothesis.
- Avoid presenting the current study's empirical conclusions before the Results.
- Preserve the distinction between a literature-supported hypothesis and a finding established by the present experiments.

## 8. Final Recommendation

Section II should be added. With 45 references and four distinct experimental factors, a dedicated Related Work section will make the manuscript's novelty easier to evaluate and will naturally resolve the current I-to-III numbering gap. Its central function should be to demonstrate that prior work has achieved predictive advances but has not adequately isolated the effects and reliability boundaries of upstream label and normalization choices, routing architecture, and loss design.
