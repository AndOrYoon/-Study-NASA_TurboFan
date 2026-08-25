# GPT Review Notes for RESS Manuscript

## Manuscript
**Title:** *From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction*

## Target Journal
**Reliability Engineering & System Safety (RESS)**

---

## Overall Recommendation

**Recommendation: Major Revision**

The manuscript addresses an important and practically relevant question in turbofan PHM: which pipeline design choices actually matter most for predictive reliability? Its strongest feature is that it moves beyond a conventional “new architecture + benchmark improvement” paper and instead performs a structured examination of RUL clipping, normalization, fault-mode architecture, and loss-function design across all four NASA CMAPSS sub-datasets.

The most interesting scientific story is not simply the performance of M3. It is the sequence:

**Normalization ablation → anomalous FD003 inter-seed variance → latent-structure hypothesis → fault-mode analysis → gating intervention → performance/variance recovery → false-routing perturbation analysis.**

This is a strong and potentially publishable story for RESS.

However, the manuscript currently states several conclusions more strongly than the experimental design supports. The main issues are:
- the proposed three-tier hierarchy is not based on a single unified cross-factor experiment;
- H4 is statistically underpowered by construction;
- the causal interpretation of FD003 variance is too strong;
- the GatingNet confidence threshold of 0.8 is insufficiently calibrated;
- the M1 comparison on FD004 is affected by a train/test representation mismatch;
- some industrial deployment claims overreach what CMAPSS evidence can support.

---

# Major Comments

## 1. The three-tier hierarchy is not established by a unified cross-factor ablation

The manuscript presents the central ordering:

**Label engineering → Fault-mode architecture → Loss function**

and repeatedly describes this as an “established” hierarchy.

The difficulty is that H1–H4 are not evaluated under the same experimental configuration:
- H1 uses Ridge regression;
- H2 uses a compact stacked LSTM;
- H3/H4 use a larger stacked LSTM;
- H2 includes operating-condition variables, while H3/H4 use sensors only;
- validation split procedures differ;
- H2 clips predictions to [0, 125] at evaluation, while H3 does not.

The manuscript itself acknowledges that absolute RMSE values cannot be directly compared across hypotheses.

### Reviewer concern
If the experimental conditions differ enough that cross-hypothesis absolute effects are not directly comparable, the manuscript cannot strictly claim that one design factor has a larger effect than another in a unified quantitative sense.

### Recommended action
Preferred:
- run a smaller unified confirmatory experiment under one common backbone, preprocessing pipeline, split strategy, and output treatment.

If that is too expensive:
- weaken the framing from “established hierarchy” to something like:
  - **“empirically observed risk-priority ordering”**
  - **“aggregated within-hypothesis evidence suggests…”**
  - **“a practical sequencing framework”**

This single framing adjustment would materially reduce reviewer resistance.

---

## 2. H4 cannot statistically support the strongest loss-function conclusion

The manuscript explicitly notes that with **N = 5 seeds** and **96 BH-FDR corrected comparisons**, statistical significance is mathematically unattainable regardless of true effect size.

That is a serious inferential limitation.

Yet the manuscript also states that custom loss functions do not outperform MSE and places loss engineering as the lowest-priority tier.

### Reviewer concern
“Failure to detect significance” is not equivalent to “evidence of no meaningful difference,” especially when the testing design has insufficient power by construction.

### Recommended action
Best option:
- add a focused confirmatory H4 experiment with more seeds.

A practical compromise would be:
- restrict to the most relevant clipping values (e.g., 125 and/or 130);
- compare MSE with only the strongest theoretically motivated custom losses;
- run 15–20 seeds for the focused subset.

If no new experiments are added:
- weaken the claim to:
  - **“no large, consistently detectable benefit was observed”**
  - **“the present study does not provide evidence that custom losses consistently outperform MSE”**
  - avoid any equivalence-style conclusion.

---

## 3. The FD003 “causal explanation” is too strong

The manuscript interprets high FD003 inter-seed variance as a signature of latent fault-mode heterogeneity and uses terms such as “causal explanation” and “validated.”

The evidence is interesting and mechanistically suggestive, but CMAPSS does not provide direct ground-truth fault-mode labels that verify the unsupervised cluster identities physically.

### Reviewer concern
The observed clustering and performance recovery are consistent with a fault-structure explanation, but they do not prove that the discovered clusters correspond causally to HPC and fan degradation modes.

### Recommended action
Replace strong causal language with:
- **“consistent with”**
- **“supports the hypothesis that…”**
- **“provides mechanistic evidence for…”**
- **“latent degradation-mode structure”**

Also replace phrases like:
- “correct fault-specific branch”

with:
- “distinct latent branch”
- “consistently separated latent subpopulation”

unless direct physical labels are available.

---

## 4. M3 requires a stronger control to establish that routing—not capacity—is the source of improvement

The M3 result on FD003 is impressive:

- M0 RMSE: **43.23 ± 0.18**
- M3 RMSE: **14.78 ± 1.32**
- RMSE reduction: **65.8%**
- NASA Score reduction: **98.8%**

The manuscript notes that the gating network itself adds only a small parameter overhead compared with M1/M2.

However, M3 still uses a two-branch architecture and has roughly twice the parameters of M0.

### Reviewer concern
Could the improvement be due partly to added capacity or ensemble-like behavior rather than meaningful routing?

### Recommended action
Add at least one of:
- parameter-matched single-branch LSTM;
- two-branch fixed 50/50 ensemble;
- random-gate two-branch control.

If M3 still dominates, the routing interpretation becomes much stronger.

---

## 5. GatingNet confidence ≥ 0.8 is not sufficiently calibrated as a deployment threshold

The proposed threshold is effectively inferred from two observed points:
- FD003 confidence = 0.840, routing strongly beneficial;
- FD004 confidence = 0.677, routing not beneficial.

The manuscript itself acknowledges that 0.8 is an indicative rather than calibrated threshold.

### Reviewer concern
Two datasets are not enough to support a universal or deployable numerical threshold.

### Recommended action
Keep the **confidence metric** as a contribution, but weaken the threshold claim.

Suggested framing:
> Gate confidence is proposed as a candidate diagnostic indicator; the observed values of 0.840 and 0.677 motivate, but do not calibrate, a deployment threshold.

Better still:
- show sensitivity across multiple candidate thresholds;
- relate confidence to routing-error risk or RMSE penalty;
- reserve 0.8 as an empirical reference point only.

---

## 6. M1 on FD004 suffers from a train/test feature-definition mismatch

The manuscript explains that:
- GMM was fitted using full training trajectories;
- test routing used only the last 30-cycle window;
- this mismatch led to near-total branch collapse (247 of 248 engines assigned to one branch).

### Reviewer concern
This may partly be an artifact of inconsistent feature construction rather than an inherent weakness of hard routing.

### Recommended action
Add a fairer baseline where the same information horizon is used in both training and test routing.

For example:
- fit GMM using K-cycle prefixes for training units;
- route test units using the same K-cycle representation.

If M3 still outperforms this fair baseline, the deployment argument becomes much stronger.

---

## 7. H1 should use a paired statistical test

The same test engines are evaluated under different clipping thresholds, but the manuscript uses an unpaired Mann–Whitney/rank-sum procedure.

The manuscript already acknowledges that a paired Wilcoxon signed-rank test would be more appropriate.

### Recommended action
Re-run the H1 significance analysis with a paired Wilcoxon signed-rank test.

This is a relatively low-cost fix and should be completed before submission.

---

## 8. RevIN comparison should be labeled more carefully

N7 is referred to as RevIN, but the inverse transformation is not applied.

The manuscript correctly explains why the inverse operation is problematic for scalar RUL regression and notes that the implementation is effectively forward-only instance normalization with learnable affine parameters.

### Reviewer concern
Calling this “RevIN” may imply a full implementation of the original method and could make the conclusion “RevIN underperforms” appear stronger than warranted.

### Recommended action
Use terminology such as:
- **“RevIN-style forward normalization”**
- **“RevIN-inspired learnable instance normalization”**
- **“forward-only RevIN variant”**

and narrow the conclusion accordingly.

---

## 9. Industrial and deployment claims overreach the CMAPSS evidence

The industrial implications section is useful in principle, but some claims are too strong for a simulation-benchmark study.

Potentially problematic examples include:
- “fault-mode classification is complete at engine commissioning”;
- pre-deployment qualification statements based on confidence ≥ 0.8;
- financial extrapolations to fleet-level savings of millions of dollars.

### Reviewer concern
RMSE improvement on CMAPSS does not directly establish reduction in unscheduled shop visits or economic savings.

### Recommended action
Retain the industrial implications section, but make it more conservative:
- use “may enable” instead of “enables”;
- describe economic values as illustrative scenarios rather than implied outcomes;
- consider removing the shop-visit-to-dollar extrapolation unless supported by a defensible maintenance decision model.

---

## 10. Use “cross-scenario” or “across CMAPSS sub-datasets” consistently

The title has already been improved by using:

**“A Systematic Ablation Study”**

instead of “Cross-Dataset Ablation Study.”

However, the body still uses phrases such as:
- cross-dataset ablation;
- cross-dataset confirmation;
- four-dataset cross-validation.

FD001–FD004 are four CMAPSS sub-datasets/scenarios within the same benchmark family.

### Recommended action
Prefer:
- **“across all four CMAPSS sub-datasets”**
- **“cross-scenario evaluation”**
- **“four-scenario benchmark evaluation”**

This is more precise and removes an avoidable reviewer attack point.

---

# Minor Comments

## 1. K = 5 vs K = 10 wording is inconsistent
The base M3 model uses **K = 10**, while K = 5 appears in a post-hoc sensitivity analysis.

Suggested fix:
- Abstract: “as few as five initial cycles”
- Methods: “default K = 10”
- Results: clearly distinguish default configuration from sensitivity findings.

---

## 2. Pinball τ inconsistency
The methodology table lists **τ = 0.25**, while Table VII lists **τ = 0.35**.

This must be checked and corrected.

---

## 3. Table IV “20 runs per configuration” appears inconsistent
The methods state that H1 Ridge regression is deterministic and each configuration was run once.

The caption should be corrected if it implies 20 repeated runs per configuration.

---

## 4. “Attention Gate” may be a misleading name
The GatingNet architecture is described as:

Flatten → Linear → ReLU → Linear → Softmax

This is not obviously an attention mechanism in the conventional neural-attention sense.

Consider renaming to:
- **Early-cycle GatingNet**
- **Soft Gating Network**
- **Early-cycle mixture gate**

unless explicit attention weighting over time/features exists elsewhere.

---

## 5. Avoid “correct fault-specific branch”
Without ground-truth fault-mode labels, use:
- “latent branch”
- “learned expert branch”
- “inferred degradation-mode branch”

---

## 6. Consider review-stage code release
The manuscript states that code will be released upon acceptance.

Because the study is ablation-heavy and reproducibility is central to its value, an anonymized review repository would strengthen the paper.

---

# Strongest Contributions

## 1. The scientific story around FD003 variance is strong
The most compelling sequence is:

**Normalization study → unexplained FD003 variance → structural hypothesis → fault-mode separation → gating solution → perturbation analysis**

This is more interesting than a straightforward model-performance paper.

---

## 2. The paper emphasizes engineering priorities rather than SOTA chasing
The study asks:
> What should a PHM engineer fix first?

That is a strong RESS-oriented framing.

The paper is most compelling when it argues that:
- label engineering can create catastrophic failure;
- fault structure can dominate model behavior;
- loss engineering may be a secondary refinement.

---

## 3. Negative results are potentially valuable
The result that custom losses do not show a consistent advantage over MSE could be practically important.

However, the value of this negative result depends heavily on fixing the H4 power problem or moderating the interpretation.

---

## 4. The limitations section is unusually transparent
The manuscript already acknowledges:
- CMAPSS-to-real-fleet generalization limits;
- backbone dependence;
- lack of physical cluster labels;
- differences between H1/H2/H3/H4 experimental setups;
- early-cycle routing limitations.

This is a strength and should be preserved.

---

# Suggested Claim Reframing

## Current-style claim
> The experiments establish a three-tier reliability-driven design hierarchy.

## Safer version
> The aggregated within-hypothesis evidence supports a practical risk-priority ordering in which label engineering should be addressed first, followed by fault-mode structure, with loss-function refinement treated as a lower-priority optimization step.

---

## Current-style claim
> GatingNet confidence ≥ 0.8 provides a deployable routing reliability indicator.

## Safer version
> GatingNet confidence is proposed as a candidate routing-diagnostic metric; the observed contrast between FD003 and FD004 suggests that higher confidence may identify cases where explicit routing is consequential, although the numerical threshold requires calibration on additional datasets.

---

## Current-style claim
> M3 routes engines to the correct fault-specific branch.

## Safer version
> M3 learns a stable early-cycle partition into latent expert branches that is consistent with the known multi-fault structure of FD003.

---

## Current-style claim
> No custom loss outperforms MSE.

## Safer version
> Under the present experimental design, no custom loss showed a consistently detectable advantage over MSE after multiple-comparison correction.

---

# Recommended Revision Priorities

| Priority | Issue | New experiment needed? | Expected impact |
|---|---|---:|---|
| 1 | Reframe or validate the three-tier hierarchy | Preferably | Very high |
| 2 | Fix H4 statistical power | Yes | Very high |
| 3 | Add M3 control for capacity/routing | Recommended | High |
| 4 | Downgrade the 0.8 threshold claim | No | High |
| 5 | Re-run H1 with paired Wilcoxon | Analysis only | Medium-high |
| 6 | Fairer M1 train/test routing baseline | Recommended | High |
| 7 | Rename RevIN variant more precisely | No | Medium |
| 8 | Remove/soften unsupported deployment ROI claims | No | Medium |
| 9 | Harmonize K=5/K=10 terminology | No | Medium |
| 10 | Fix numerical/table inconsistencies | No | Medium |

---

# Submission Readiness Assessment

## Current version
**Likely recommendation:** Major Revision, with meaningful Reject/Resubmit risk.

Estimated subjective final acceptance probability if submitted unchanged:
**~25–35%**

## After addressing the highest-priority issues
If the manuscript:
- softens the hierarchy claim or adds a unified confirmatory experiment;
- strengthens H4 with more seeds;
- adds one M3 capacity/routing control;
- reframes the 0.8 threshold as indicative;
- fixes H1 paired statistics;

then the manuscript would become substantially more defensible.

Estimated subjective final acceptance probability after strong revision:
**~45–60%**

These percentages are judgment-based estimates, not journal statistics.

---

# Recommended Editorial Positioning

The paper should not be positioned primarily as:

> another improved RUL architecture on CMAPSS.

A stronger positioning is:

> **a reliability-oriented systematic study of how PHM design choices fail, interact, and should be prioritized.**

The strongest intellectual message is:

> **The major value of the study is not merely identifying a better model, but showing that poor pipeline choices can create qualitatively different levels of prognostic risk, and that unexplained variance itself can reveal hidden structure that should influence architecture design.**

---

# Final Reviewer-Style Summary

The manuscript addresses an important and practically relevant question in turbofan PHM and contains several interesting empirical findings, particularly the diagnosis of FD003 inter-seed instability and the early-cycle gating experiments. However, the central three-tier hierarchy is currently stated more strongly than warranted by the heterogeneous cross-hypothesis experimental designs, and the H4 statistical design is underpowered to support the claimed lowest-priority status of loss-function engineering. The proposed gate-confidence threshold also requires more cautious interpretation. I therefore recommend **Major Revision**. The manuscript could become a valuable contribution to RESS if the claims are aligned more closely with the evidence and selected control experiments are added.
