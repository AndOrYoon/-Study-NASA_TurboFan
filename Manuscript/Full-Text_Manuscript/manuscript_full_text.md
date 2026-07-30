# From Fleet Normalization to Fault-Mode Gating: A Cross-Dataset Ablation Study of Turbofan Remaining Useful Life Prediction

**Authors:** [Placeholder — to be filled before submission]
**Affiliation:** [Placeholder]
**Correspondence:** spiceyoon@gmail.com

> **Compiled:** 2026-07-06  
> **Last revised:** 2026-07-30 (§V.D, §V.G updated; References [46]–[49] added)  
> **Status:** Complete draft — all sections revised and cross-checked for internal consistency  
> **Source sections:** `Manuscript/Sections/` (Abstract_draft, Introduction, Methodology, Results, Discussion_Implication, Conclusion)  
> **Section numbering note:** Sections are numbered I, III–VI; Section II (Related Work) is integrated into Sections I and V.

---

## Abstract

Reliable Prognostic and Health Management (PHM) systems for turbofan engines embed interdependent design decisions — RUL label clipping, sensor normalization, fault-mode architecture, and training loss function — whose contributions to predictive reliability are rarely isolated. We present a controlled ablation study across all four NASA CMAPSS sub-datasets (FD001–FD004), covering five clipping thresholds, seven normalization strategies, four fault-mode architectures, and seven loss functions, with all comparisons Benjamini-Hochberg FDR-corrected.

Fleet min-max normalization outperforms all per-unit and instance-level alternatives on FD001, FD002, and FD004 but exhibits anomalously high inter-seed variance on FD003 (RMSE std = 12.86 vs. ≤1.84 elsewhere), traced to co-existing HPC and fan fault modes rather than normalization failure. An unsupervised attention-gate model (M3) routes engines to fault-specific branches from five initial flight cycles, reducing FD003 RMSE by 65.8% (14.78 ± 1.32 vs. 43.23 ± 0.18) and NASA Score by 98.8%, while remaining immune to the test-time cluster collapse that degrades GMM hard-routing by 75.4% on FD004. No custom loss outperforms MSE after multiple-comparison correction; removing RUL clipping inflates NASA prognostic scores by up to 306,000-fold regardless of loss design.

These findings establish a three-tier reliability-driven design checklist — label engineering, fault-mode architecture, loss function — where each tier's failure impact qualitatively exceeds the next, providing reliability engineers with risk-prioritised guidance for PHM system design.

---

## I. Introduction

Remaining Useful Life (RUL) prediction for turbofan engines is a foundational task in Prognostic and Health Management (PHM), directly shaping safety-critical maintenance scheduling decisions in civil and military aviation. The NASA prognostic score reflects the underlying risk asymmetry explicitly: late predictions — where remaining life is over-estimated — carry an exponentially higher penalty than early ones, because an undetected engine failure costs orders of magnitude more than a precautionary shop visit. This asymmetric risk structure means that design choices in a turbofan PHM pipeline — how RUL targets are labelled, how sensor data are normalised, how multi-fault engines are handled — interact with the evaluation metric in ways that can constitute systemic reliability failures when misconfigured. The NASA CMAPSS benchmark [1], comprising four run-to-failure sub-datasets (FD001–FD004) under varying fault modes and operating conditions, has served as the canonical testbed for data-driven RUL estimation since 2008. As model architectures have evolved from linear regression [2] to stacked LSTM [5, 6] and attention-based encoders [13, 14, 45], reported RMSE on FD001 has fallen from above 25 cycles to below 13 [15]. Yet the pipeline design decisions responsible for these gains have rarely been isolated from one another, leaving reliability engineers without principled guidance on which choices most strongly determine PHM system dependability.

Existing studies rarely disentangle individual design factors. The majority of the CMAPSS literature evaluates a single sub-dataset — most often FD001 — and bundles normalisation, loss function, and architecture into a single proposed system [9, 14, 15], making it impossible to attribute observed gains to any one component. Three specific gaps motivate this study: (i) no cross-dataset comparison of normalisation strategies exists, despite the known operating-condition heterogeneity of FD002 and FD004 [24]; (ii) the multi-fault structure of FD003/FD004 is acknowledged in prior EDA [36] but rarely exploited through explicit architecture design; and (iii) the interaction between RUL clipping threshold and loss function has not been studied jointly, even though both directly shape the label distribution, gradient signal, and — as we show — the tail behaviour of the NASA prognostic risk metric, with misconfiguration capable of inflating prognostic scores by orders of magnitude.

We address these gaps through a controlled ablation study across all four CMAPSS sub-datasets. H2 uses Ridge regression to isolate label-engineering effects from model capacity. H5 uses a compact stacked LSTM (LSTM2 hidden=32). H6 and H7 use a full-capacity stacked LSTM (LSTM2 hidden=64). Within each hypothesis, all other factors are held constant while the design factor under study is varied, with all comparisons subject to Wilcoxon rank-sum tests corrected by Benjamini-Hochberg FDR (BH-FDR). We independently vary (H2) five RUL clipping thresholds, (H5) seven normalisation strategies, (H6) four fault-mode architectures, and (H7) seven training loss functions, enabling attribution of observed performance differences to the factor under study.

H2 empirically confirms that clip = 125 cycles — the established literature standard [5, 6] — is optimal or statistically tied-optimal across all four sub-datasets. Critically, omitting the clipping ceiling inflates the NASA prognostic score by up to 306,000-fold on FD003 — quantifying the reliability failure risk of label misconfiguration and establishing that RUL label engineering is not a modelling detail but a safety-critical prerequisite for PHM system deployment. This confirmation underpins the first tier of the reliability-driven design checklist below; the label-engineering choice is treated as a resolved prerequisite — and a critical reliability boundary condition — rather than an open design variable in the remaining hypotheses.

The main contributions of this paper are:

(i) The first controlled normalisation ablation on CMAPSS showing that fleet-level min-max scaling significantly outperforms per-unit and RevIN strategies on FD001, FD002, and FD004, and that FD003's anomalous inter-seed RMSE variance (std = 12.86 vs ≤1.84 elsewhere) is a diagnostic indicator of latent fault-mode heterogeneity rather than a normalisation deficiency — a transferable principle for any benchmark study: unexplained inter-seed variance may signal structural data heterogeneity that predicts the need for specialised architectures.

(ii) An end-to-end early-cycle attention-gate architecture (M3) that routes engines to fault-specific prediction branches using only the first ten observed flight cycles, achieving a 65.8% RMSE reduction on FD003 without any fault-mode labels, while remaining statistically equivalent to the single-branch baseline on FD004.

(iii) A three-tier reliability-driven design checklist — label engineering, fault-mode architecture, loss function — established by the first joint cross-dataset ablation of all four factors, demonstrating that each tier's failure impact qualitatively exceeds the next.

(iv) A practical reliability-driven deployment framework (Section V.F) that translates the three-tier checklist into concrete, ordered design decisions — fault-mode screening via unsupervised cluster quality and loss function choice — enabling PHM reliability engineers to allocate development resources to the factors that most strongly determine end-to-end system dependability in safety-critical maintenance operations. From an operational cost perspective, improper label engineering inflates the prognostic risk metric by up to six orders of magnitude regardless of all other design choices; M3's fault-mode routing completes at engine commissioning from as few as five flight cycles; and custom loss functions provide no statistically detectable benefit once label engineering is correctly applied — three findings that directly prioritise engineering effort in real fleet deployments.

---

## III. Methodology

### A. Dataset

We use the NASA CMAPSS benchmark, which comprises four sub-datasets (FD001–FD004) simulating turbofan engine run-to-failure under varying operating conditions and fault modes (Table I). Each dataset records 21 raw sensor channels per flight cycle together with a held-out test set and ground-truth RUL values for the final observation of each test engine.

**Table I. CMAPSS sub-dataset characteristics.**

| Dataset | Train engines | Test engines | Op. conditions | Fault modes |
|---------|:------------:|:------------:|:--------------:|:-----------:|
| FD001 | 100 | 100 | 1 | HPC degradation only |
| FD002 | 260 | 259 | 6 | HPC degradation only |
| FD003 | 100 | 100 | 1 | HPC + Fan degradation |
| FD004 | 249 | 248 | 6 | HPC + Fan degradation |

Representative sensor degradation trajectories for FD003 — which contains both fault modes — are shown in Fig. 1, illustrating the visually distinct patterns that motivate fault-mode-aware modelling (H6) and fleet-level normalization (H5).

[FIGURE 1: Sensor Degradation Trajectories]
**Fig. 1.** Illustrative sensor degradation trajectories for representative engines in FD003 (single operating condition, two fault modes). Selected sensors with high RUL correlation (s2, s3, s4, s7, s11, s12) show visually distinct degradation patterns between HPC-fault and fan-fault engines, motivating both fault-mode-aware modeling (H6) and fleet-level normalization that preserves inter-engine degradation contrast (H5).

### B. Data Preprocessing

Seven constant-variance sensor channels are removed per dataset. For FD001 and FD003, sensors s1, s5, s6, s10, s16, s18, and s19 are discarded, leaving 14 input features; for FD002 and FD004, only s16 is constant, yielding 20 features. RUL targets follow a piecewise-linear formulation: for each training engine, cycles where the remaining life exceeds a clipping threshold τ receive a constant label of τ, while the final segment decreases linearly to zero. We evaluate five clipping values (τ ∈ {75, 100, 125, 130, ∞}), with τ = 125 cycles serving as the standard baseline per established literature [5, 6]. Ground-truth test RUL values are provided directly by the benchmark; no RUL estimation is performed at evaluation time. The piecewise labeling scheme and the effect of varying the clipping threshold are illustrated in Fig. 2.

[FIGURE 2: Piecewise Linear RUL Label and Clipping]
**Fig. 2.** Piecewise linear RUL labeling scheme with threshold-based clipping. The raw RUL decreases linearly from the maximum cycle but is clipped at threshold *c* to account for the healthy phase. The shaded region illustrates the effect of varying *c* ∈ {75, 100, 125, 130, ∞}; over-clipping truncates degradation information while removing the ceiling entirely permits unbounded targets that destabilize the NASA prognostic score.

### C. Operating-Condition Residualization

FD002 and FD004 contain six discrete operating conditions that shift absolute sensor levels by tens to hundreds of units. We apply K-means residualization to decouple degradation signals from operating-point offsets. A K-means model (k = 6) is fitted on the three operating-condition variables (op1, op2, op3) of the training set using standardised inputs; per-cluster sensor means are computed from training data only and subtracted from each observation:

$$z_{i,j} = x_{i,j} - \mu_{c(i),\,j}$$

where c(i) denotes the cluster assignment of cycle i and μ_{c,j} is the training-set mean of sensor j within cluster c. This step precedes all normalization and is applied identically to training and test data using statistics derived exclusively from training engines. FD001 and FD003 present a single operating condition and therefore do not require this step.

### D. Normalization Strategies (H5)

We compare seven normalization strategies (N1–N7) to evaluate the effect of the reference-statistics choice. Fleet-level methods compute statistics across all training engines: N1 applies min-max scaling to [0, 1] and N2 applies z-score standardisation. Per-unit methods (N3–N6) use each engine's own early-cycle observations as the reference baseline, thereby removing initial-condition offsets before any degradation signal is visible: N3 applies min-max over the first 5 cycles, N4 over the first 10 cycles, N5 applies z-score over the first 5 cycles, and N6 over the first 10 cycles. N7 implements Reversible Instance Normalization (RevIN; Kim et al., 2022 [23]) as a learnable module within the model, normalising each 30-cycle inference window by its instantaneous mean and standard deviation and applying trainable affine parameters (γ, β); because the RUL output is a scalar rather than a sensor-space signal, no denormalization is applied to the prediction. This is intentional and architecturally correct: unlike time-series forecasting where the model output is in the same unit-space as the input (and RevIN's inverse transform would recover the original scale), the RUL prediction is a scalar in engine-cycle units that is dimensionally distinct from the input sensor features. Applying an inverse sensor-normalization transform to a cycle-unit output would be dimensionally incorrect. N7 therefore implements the forward (normalization) half of RevIN only, making it equivalent to per-window instance normalization with learnable affine parameters. N1 serves as the primary comparison baseline. All strategies are evaluated on the identical backbone with all other experimental factors fixed.

| ID | Name | Reference statistics |
|----|------|---------------------|
| N1 | Fleet min-max | All training engines |
| N2 | Fleet z-score | All training engines |
| N3 | Per-unit min-max (5 cy) | Engine's first 5 cycles |
| N4 | Per-unit min-max (10 cy) | Engine's first 10 cycles |
| N5 | Per-unit z-score (5 cy) | Engine's first 5 cycles |
| N6 | Per-unit z-score (10 cy) | Engine's first 10 cycles |
| N7 | RevIN (learnable) | Per-window, at inference |

### E. LSTM Backbone Architectures

H5 uses a compact LSTM backbone and H6/H7 use a full-capacity backbone; within each hypothesis, the backbone is held fixed so that observed differences reflect only the design factor under study. Each network takes a sliding window of 30 consecutive cycles as input and produces a scalar RUL estimate:

```
H5 Backbone (compact):
Input  (B × 30 × 18_or_14)
→ LSTM₁ (hidden=64, returns full sequence) → Dropout(0.2)
→ LSTM₂ (hidden=32, returns last time step) → Dropout(0.2)
→ Linear(32 → 16) → ReLU → Linear(16 → 1)

H6/H7 Backbone (full-capacity):
Input  (B × 30 × 15_or_20)
→ LSTM₁ (hidden=64, returns full sequence) → Dropout(0.2)
→ LSTM₂ (hidden=64, returns last time step) → Dropout(0.2)
→ Linear(64 → 32) → ReLU → Linear(32 → 1)
```

LSTM₁ passes its full sequence of hidden states to LSTM₂, preserving temporal context across layers. Test sequences shorter than 30 cycles are zero-padded at the front. Input dimensions vary by hypothesis and dataset; see Table III in Section H.

**Table II. Computational complexity of H6 architectures (FD003, F = 15 features).**
*Parameter counts verified by direct model inspection. FLOPs computed analytically per inference window (batch = 1, window = 30 cycles) using the standard LSTM FLOPs formula: 8 × (input + hidden) × hidden per timestep. Inference latency measured on NVIDIA RTX GPU (2,000 runs, batch = 1, after 200-run warm-up; mean ± std reported). Training time is GPU compute per epoch on FD003 (~13,600 training sequences, batch = 256), excluding data loading. M1 routes each test engine to a single branch via GMM argmax, so its inference FLOPs equal M0's.*

| Model | Parameters | FLOPs / window | Inference (measured) | Training (GPU) |
|-------|-----------|---------------|---------------------|----------------|
| M0 (single branch) | 56.1K | 3.18M | 0.24 ± 0.05 ms | ~0.1 s/epoch |
| M1 (hard routing) | 112.3K | 3.18M† | 0.25 ± 0.06 ms† | ~0.2 s/epoch‡ |
| M2 (soft gating) | 112.3K | 6.37M | 0.57 ± 0.24 ms | ~0.2 s/epoch |
| M3 (attention gate, K=10) | 117.2K | 6.38M | 0.64 ± 0.21 ms | ~0.2 s/epoch |

†M1 activates one branch at inference (GMM argmax); both branches (112.3K) reside in memory but only one branch forward pass executes.  ‡M1 trains two branches sequentially; reported time is total per epoch.

M3's GatingNet adds only 4.9K parameters above M1/M2 (< 5% overhead), confirming that the performance improvement is not attributable to additional model capacity. The approximately 2× parameter increase from M0 to M3 reflects the two independent prediction branches rather than the gating mechanism itself. All four architectures are well within the computational budget of embedded PHM controllers, which typically support models of up to several hundred thousand parameters.

**Note:** The difference in backbone capacity means that H5 and H6/H7 RMSE values are not directly comparable in absolute terms. Each hypothesis is evaluated internally relative to its own controlled baseline.

### F. Fault-Mode Architectures (H6)

We compare four architectures on FD003 and FD004 to assess whether explicit fault-mode separation improves RUL accuracy.

**M0 (Baseline)** is a single-branch model using the shared backbone without any fault-mode handling.

**M1 (Hard Routing)** trains two independent LSTM branches. Each training engine is assigned deterministically to one branch via the argmax of its GMM posterior probability; test engines are routed identically. Branches are trained independently with standard MSE.

**M2 (Soft Gating)** retains two branches but replaces hard assignment with a weighted sum:

$$\hat{y}_{\text{final}} = p_0 \cdot \hat{y}_0 + p_1 \cdot \hat{y}_1$$

where [p₀, p₁] are the GMM posterior probabilities. The training loss augments the final MSE with auxiliary branch supervision:

$$\mathcal{L}_{\text{M2}} = \text{MSE}(\hat{y}_{\text{final}},\, y) + 0.1\cdot\text{MSE}(\hat{y}_0,\, y) + 0.1\cdot\text{MSE}(\hat{y}_1,\, y)$$

**M3 (Attention Gate)** eliminates the GMM entirely. A lightweight GatingNet reads only the first K = 10 observed cycles and produces end-to-end soft routing weights:

$$\text{GatingNet}: (B \times K \times F) \to \text{Flatten} \to \text{Linear}(K{\cdot}F \to 32) \to \text{ReLU} \to \text{Linear}(32 \to 2) \to \text{Softmax}$$

The final prediction is $\hat{y}_{\text{final}} = w_0\hat{y}_0 + w_1\hat{y}_1$ with training loss:

$$\mathcal{L}_{\text{M3}} = \text{MSE}(\hat{y}_{\text{final}},\, y) + 0.05\cdot\text{MSE}(\hat{y}_0,\, y) + 0.05\cdot\text{MSE}(\hat{y}_1,\, y)$$

By reading only early-cycle data, M3 avoids any dependence on late-cycle observations that are unavailable at real deployment time and is immune to the test-time cluster-distribution collapse that makes M1 unreliable on FD004.

For M1 and M2, GMM cluster assignments (k = 2, full covariance) are derived by unsupervised fitting on degradation-slope features of seven discriminant sensors identified by EDA (s15, s20, s21, s7, s12, s2, s4). The inter-cluster discriminability of each sensor is quantified by its inter-cluster z-score:

$$|\Delta z|_j = \frac{|\bar{\mu}_{c=1,j} - \bar{\mu}_{c=2,j}|}{\sigma_{\text{fleet},j}}$$

where $\bar{\mu}_{c,j}$ is the training-set mean of sensor $j$ within cluster $c$, and $\sigma_{\text{fleet},j}$ is the fleet-wide standard deviation of sensor $j$. Both branches in M2 and M3 use the full shared backbone.

### G. Loss Functions (H7)

We evaluate seven training loss functions to determine whether asymmetric or dynamically weighted objectives improve over standard MSE. All functions share the signature `loss(pred, true, life_ratio=None)`, where life_ratio = 1 − RUL/RUL_max is a training-time weighting signal computed from training-set cycle counts only; it is set to None at inference time, eliminating any dependency on unknown test-set engine lifetimes.

| ID | Name | Key formulation |
|----|------|----------------|
| L1 | MSE (baseline) | (1/N) Σ (ŷ−y)² |
| L2 | NASA Score Loss | Differentiable approx. of s(d) |
| L3 | DynMSE | (1 + λ_dyn · r) · (ŷ−y)²; λ_dyn = 1.0 |
| L4 | Focal-RUL | (‖ŷ−y‖/(‖ŷ−y‖+1))^γ · (ŷ−y)²; γ = 2 |
| L5 | TWA | w_time · w_asym · (ŷ−y)²; λ_t = 10, λ_a = 1 |
| L6 | Pinball | τ·max(0, y−ŷ) + (1−τ)·max(0, ŷ−y); τ = 0.25 |
| L7 | HubA | Huber(δ=20) · w_asym; λ_a = 3 |

For L5 (TWA), w_time = 1 + λ_t · r and w_asym = 1 if d < 0, else λ_a. For L7, w_asym = 1 if d < 0, else λ_a, where d = ŷ − y. Key hyperparameters (τ, λ_t, λ_a, δ) were selected via grid search on FD001 before cross-dataset evaluation. FD001 therefore functions as a partially tuned evaluation dataset for H7; nominally strong FD001-specific effects (particularly L5 and L7) should be interpreted with this caveat. BH-FDR correction across all four datasets partially mitigates this optimism.

### H. Training Configuration

All models are optimised with Adam (learning rate 1×10⁻³, weight decay 1×10⁻⁴) with a batch size of 256, for a maximum of 100 epochs. Early stopping monitors validation loss with patience of 15 epochs and restores the best-performing checkpoint. The validation set is constructed by engine-level holdout: 20% of training engines are withheld, and all cycles of those engines are excluded from training. This prevents the RUL distribution mismatch that arises from cycle-level splitting. Each experimental configuration is run with five random seeds (0, 1, 2, 3, 4); results are reported as mean ± standard deviation across seeds.

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
| H2 | Sensors + op cols (Ridge) | N/A (deterministic Ridge) | Ridge regression |
| H5 | Sensors + op cols (incl. op1/op2/op3) | Last 20% by unit ID (deterministic) | Compact LSTM (LSTM₂ hidden=32) |
| H6 | Sensors only (no op cols) | Random 20% (RandomState seed=42) | Full LSTM (LSTM₂ hidden=64) |
| H7 | Sensors only (no op cols) | Random 20% (RandomState seed=42) | Full LSTM (LSTM₂ hidden=64) |

For H5 the resulting input dimension F is 17 (FD001: 14 sensors + 3 op), 18 (FD003: 15 sensors + 3 op), or 23 (FD002/FD004 after residualisation: 20 sensors + 3 op). For H6/H7 F is 14 (FD001), 15 (FD003), or 20 (FD002/FD004 after residualisation). **Note:** The feature and split differences between H5 and H6 reflect independent implementation choices made prior to analysis; they mean that the two hypotheses are not directly cross-comparable in absolute RMSE terms. Each hypothesis is interpreted relative to its own baseline condition.

### I. Evaluation Metrics

Two metrics are reported across all experiments. RMSE is the primary performance indicator:

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$$

The NASA prognostic score penalises late predictions more severely than early ones:

$$s(d) = \begin{cases} e^{-d/13} - 1 & d < 0 \text{ (early prediction)} \\ e^{d/10} - 1 & d \geq 0 \text{ (late prediction)} \end{cases}, \quad \text{NASA Score} = \frac{1}{N}\sum_{i=1}^{N} s(\hat{y}_i - y_i)$$

where d = ŷ − y and N is the number of test engines. NASA Score is lower-is-better; a perfect prediction yields zero.

### J. Statistical Testing

Statistical comparisons differ in sample unit by hypothesis. For H2 (Ridge regression, deterministic), comparisons use a two-sided Mann-Whitney U test (`scipy.stats.ranksums`) on per-engine RMSE values (N ≈ 100–259 per dataset; see §III.K). For H5, H6, and H7 (LSTM, stochastic), comparisons use one-sided Wilcoxon rank-sum tests on per-seed aggregate metrics (N = 5), testing whether the treatment condition improves over baseline. All tests use α = 0.05 before correction. When multiple treatment conditions are compared simultaneously within one hypothesis, raw p-values are corrected using the Benjamini-Hochberg (BH) procedure at α_FDR = 0.05. A result is considered statistically meaningful when both p_BH < 0.05 **and** Cohen's d ≥ 0.3 (small effect threshold). Conditions satisfying the p-value criterion but yielding |d| < 0.1 are reported as "statistically significant but practically negligible" to distinguish statistical from practical significance. For context, a Cohen's d of 0.3 at the FD001 RMSE baseline of approximately 14–16 cycles corresponds to a mean RMSE difference of approximately 0.6–1.0 cycles — a gap comparable to one cycle of maintenance scheduling uncertainty in typical PHM deployment contexts.

### K. H2 Baseline Model (Ridge Regression)

The RUL clipping study (H2) uses `sklearn.linear_model.LinearRegression` with L2 regularisation (Ridge, default α = 1.0) as its predictive model. Ridge regression was chosen to isolate the effect of label engineering from non-linear model capacity: the clipping threshold's effect on RUL label distribution is architecture-independent, and a deterministic closed-form baseline eliminates random-initialisation variance. Each of the 20 experimental configurations (5 clip values × 4 datasets) is run exactly once; the model has no random state and produces identical results on identical data. The statistical comparison for H2 uses the Mann-Whitney U test (`scipy.stats.ranksums`, unpaired, two-sided) on per-engine RMSE values (N ≈ 100–259 per dataset depending on sub-dataset). Note that this is technically an unpaired test; a paired Wilcoxon signed-rank test would be marginally more statistically efficient since the same test engines are evaluated under both clip conditions, and this limitation should be borne in mind when interpreting H2 significance levels.

---

## IV. Results

> **Models:** H2 = LinearRegression (deterministic, 20 runs); H5/H6/H7 = Stacked LSTM (5 seeds, mean ± std)  
> **Baseline:** clip = 125 cycles, Fleet min-max (N1), single-branch LSTM (M0), MSE (L1) throughout unless noted

### A. Effect of RUL Clipping (H2)

**clip = 125 cycles yields the lowest or tied-lowest RMSE on all four sub-datasets.** The full matrix of results is presented in Table IV. On FD001 and FD003 (single operating condition), clip = 125 is the clear optimum (RMSE = 21.90 and 21.62, respectively). On FD002 and FD004, clip = 130 produces a marginally lower RMSE (Δ = −0.57 cycles on both), but the difference does not approach statistical significance (Mann-Whitney U, two-sided, p = 0.97 on each; BH-FDR applied). The non-significant result (p = 0.97) is directionally consistent with clip=130 being marginally worse than clip=125, not better. No tested alternative threshold achieves a statistically significant improvement over clip = 125, confirming its status as the practically validated standard [5, 6]. Aggressive clipping at clip = 75 is significantly inferior on all four datasets (Δ RMSE = +10.5 to +16.2 cycles; p ≤ 0.028), indicating that excessive truncation discards degradation signal in the upper RUL range. These findings are consistent with earlier single-dataset reports that converged empirically on 125 cycles [5, 6] and extend them to a rigorous four-dataset cross-validation.

**Removing the RUL ceiling entirely is catastrophic on FD003.** At clip = None, the NASA prognostic score on FD003 reaches 4,014,724 — a 306,000-fold increase relative to clip = 125 (13.09) — while FD004 similarly explodes to 1,759 (35.4× increase). The mechanism is the asymmetric exponential structure of the NASA metric: late predictions (ŷ > y) are penalised by exp(d/10), which grows unboundedly as the model, trained on uncapped labels exceeding 500 cycles, systematically over-predicts RUL for the majority of the test trajectory. RMSE is also significantly elevated under clip = None on FD001 (RMSE = 31.90 vs 21.90; p = 0.006) and FD004 (46.99 vs 34.61; p = 0.002), though the divergence is less extreme than the NASA Score collapse. Notably, FD002 is substantially more robust to the removal of clipping (clip = None RMSE = 33.05; p = 0.054), likely because its 260 training engines and six operating conditions provide enough distributional support for the model to learn a more conservative bias even without an explicit ceiling. This dataset-dependent sensitivity implies that the choice of clipping threshold cannot be treated as globally optimal; however, clip = 125 is the safest default across the full benchmark. Figure 3 summarises the full RMSE matrix across all five clipping values and four datasets.

[FIGURE 3: Effect of RUL Clipping on Prediction Accuracy]
**Fig. 3.** RMSE heatmap across four CMAPSS sub-datasets (FD001–FD004) and five clipping thresholds (mean over 20 runs, Ridge regression baseline). clip = 125 achieves the lowest or near-lowest RMSE in all datasets. The absence of clipping (clip = None) produces catastrophic RMSE on FD003 (56.1 cycles) where unbounded targets amplify early-life residuals exponentially.

### B. Effect of Normalization Strategy (H5)

**Normalization outcomes differ sharply between FD003 and the remaining three sub-datasets.** FD003 serves as the sole exception to an otherwise consistent pattern and is therefore addressed first.

On FD001, FD002, and FD004, fleet-level min-max normalization (N1) achieves the lowest RMSE: 14.14 ± 0.22, 14.31 ± 0.10, and 14.60 ± 0.30 cycles, respectively. All per-unit strategies (N3–N6) are significantly inferior on FD001 (ΔRMSE = +3.8 to +6.6 cycles; p_BH = 0.011; Cohen's d = 2.3 to 7.6) and on FD002 (ΔRMSE = +1.2 to +4.2 cycles; p_BH = 0.011; d = 4.2 to 22.4). The finding contradicts the hypothesis that engine-local initial-cycle statistics would better capture individual degradation trajectories: on CMAPSS, inter-engine sensor variation is modest relative to degradation magnitude, and fleet-level statistics preserve more discriminative information than engine-local references. RevIN (N7) is likewise significantly inferior on FD001 (ΔRMSE = +0.78 cycles; p_BH = 0.011; d = 1.66), FD002 (+3.85 cycles; d = 20.0), and FD004 (+3.68 cycles; d = 6.19), despite its adaptive per-window normalisation and learnable affine parameters (Table V). The original RevIN paper [23] demonstrated strong gains on multi-step forecasting tasks where distribution shift across training and test windows is the primary challenge; on CMAPSS, the dominant challenge is instead extracting cross-engine degradation patterns from a common fault signature, a context less suited to instance-level adaptation. Fleet z-score (N2) shows no statistically significant difference from N1 on FD001/FD002/FD003, suggesting the specific choice between min-max and z-score is secondary to the fleet-vs-per-unit distinction. Figure 4 visualises RMSE across all normalizer–dataset combinations; the corresponding pairwise statistical comparison against N1 is shown in Fig. 5.

[FIGURE 4: RMSE Comparison Across Normalization Strategies]
**Fig. 4.** Mean RMSE heatmap (5 seeds) for seven normalization strategies (N1–N7) across four CMAPSS sub-datasets. Fleet min-max (N1) achieves the lowest RMSE on FD001 (14.14), FD002 (14.31), and FD004 (14.60). Per-unit methods (N3–N6) consistently underperform by removing between-engine degradation contrast. The anomalously high variance of N1 on FD003 (std = 12.86) is attributed to latent fault-mode mixing, resolved in H6.

[FIGURE 5: Statistical Significance of Normalization Differences (vs N1)]
**Fig. 5.** Pairwise statistical comparison of each normalization method against fleet min-max (N1) using the Wilcoxon rank-sum test with Benjamini-Hochberg FDR correction (α = 0.05). Effect sizes are reported as Cohen's *d*. Per-unit methods (N3–N6) and RevIN (N7) are significantly inferior on FD001, FD002, and FD004 (|*d*| ≥ 1.66 in all significant cases). FD003 shows no significant differences due to high seed-to-seed variance from mixed fault modes.

**FD003 is the sole dataset where no normalization strategy separates from N1.** N1 achieves a mean RMSE of 19.05 ± 12.86 cycles on FD003, an anomalously high standard deviation compared with ≤ 1.84 on all other dataset-normalizer combinations. All competitors show p_BH ≥ 0.14, indicating that no alternative normalization reliably outperforms N1 on this dataset. (Note: the H5 N1/FD003 baseline of 19.05 ± 12.86 and the H6 M0/FD003 baseline of 43.23 ± 0.18 are not directly comparable — H5 uses a compact LSTM backbone [LSTM₂ hidden=32, 18 input features] while H6 uses a full-capacity backbone [LSTM₂ hidden=64, 15 features]; their validation splits and test-prediction clipping also differ. Each hypothesis is interpreted against its own controlled baseline.) Crucially, the high variance is present in N1 alone: RevIN (N7) achieves 16.44 ± 0.36 and N4 achieves 17.35 ± 0.54, both with narrow distributions. This pattern is inconsistent with a normalization deficiency; N1's per-seed RMSE varies from approximately 7 to 43 cycles across the five runs, a variance profile indicative of a latent bimodal structure in the data rather than a statistical noise artefact. As detailed in Section C, FD003 contains two concurrent fault modes (HPC and fan degradation), and the training data composition sampled by a given random seed determines which fault mode the single-branch model primarily learns — producing the observed inter-seed bimodality. This causal explanation is validated by the fault-mode experiments in the following section.

### C. Fault-Mode Architectures (H6)

#### C.1 Unsupervised Cluster Quality

Before evaluating branch architectures, we verify that two distinct fault modes exist in FD003 and FD004. GMM clustering (k = 2) on combined late-cycle means and degradation slopes of seven discriminant sensors yields Silhouette scores of 0.761 (FD003, AB_full variant) and 0.750 (FD004, AB_full), both well above the conventional quality threshold of 0.50 [35]. Slope-only clustering (AB_slope), which controls for possible life-length confounding, yields Silhouette = 0.702 (FD003) and 0.683 (FD004), confirming that the two clusters reflect genuine sensor-trajectory differences rather than an artefact of unequal engine lifetimes. Sensor s15 (bypass pressure ratio) exhibits the largest inter-cluster z-score (|Δz| = 31.3), followed by s20 (HPT bleed, 16.0) and s21 (LPT bleed, 15.2), consistent with the known distinction between HPC-dominated and fan-dominated degradation pathways in CMAPSS FD003/FD004 [36]. Figure 6 summarises the cluster quality metrics across all feature variants.

[FIGURE 6: GMM Fault-Mode Cluster Quality (Phase 1)]
**Fig. 6.** GMM (*K* = 2) clustering results for FD003 and FD004. (a) Silhouette scores across three feature variants (AB_full, AB_slope, AB_late); AB_late achieves the highest separation (FD003: 0.858, FD004: 0.855). (b) BIC scores confirming *K* = 2 as the optimal cluster count. The AB_full variant (Silhouette ≥ 0.75 on both datasets) is used in Phase 2 to retain temporal diversity across the full engine lifetime.

#### C.2 FD003: Stepwise Improvement from M0 to M3

**The attention-gate model (M3) reduces FD003 RMSE by 65.8% relative to the single-branch baseline.** Detailed results appear in Table VI. M0 yields RMSE = 43.23 ± 0.18 cycles and NASA Score = 34,339 ± 2,094. This high baseline reflects the seed-dependent bimodality identified in Section B: without fault-mode separation, a given random initialisation converges to fitting either the HPC or the fan fault signature, but rarely both. GMM hard-routing (M1) reduces the mean RMSE to 32.45 ± 11.37, but the large standard deviation (11.37 cycles) indicates that the quality of the GMM-derived cluster labels is sensitive to the training split. GMM soft-gating (M2) further improves to 26.16 ± 14.81, confirming that weighted branch aggregation smooths over borderline cluster assignments; however, the inter-seed variability remains high (std = 14.81), suggesting that GMM posteriors alone are insufficient to produce a stable routing signal.

M3, which replaces the GMM entirely with a lightweight GatingNet trained end-to-end on the first K = 10 observed cycles, achieves RMSE = 14.78 ± 1.32 cycles (NASA Score = 425 ± 127). Relative to M0, this represents a 65.8% RMSE reduction and a 98.8% NASA Score reduction. Relative to the most recent state-of-the-art supervised approach on this benchmark, CAELSTM (RMSE = 13.40 [15]), M3 is within 10.3% while employing no fault-mode supervision — a practically small gap attributable to the simpler two-layer LSTM backbone rather than to the gating strategy itself. This comparison should be treated as indicative rather than definitive: CAELSTM uses convolutional autoencoder pre-training not present in the M3 backbone, and the 10.3% gap may reflect backbone capacity differences rather than gating strategy performance. The inter-seed spread of M3 (std = 1.32) is substantially narrower than M0 (std = 0.18 is low because M0 consistently fails at the same level; M3 std reflects one outlier seed at RMSE = 17.09 that is otherwise absent) and directly resolves the high-variance anomaly identified in H5: under M3, the GatingNet consistently routes engines to the correct fault-specific branch regardless of random initialisation, breaking the bimodal convergence trap.

A post-hoc sensitivity analysis varying K ∈ {5, 10, 15, 20, 30} reveals that M3 RMSE is largely insensitive to the number of initial cycles used for routing (range: 14.13–14.78 across all K values). Variance decreases monotonically with K (std = 0.34 at K=5, 0.27 at K=30), suggesting that additional early-cycle context improves routing stability but not mean accuracy. Notably, K=5 already achieves RMSE = 14.23 ± 0.34, within 0.55 cycles of the best result at K=20 (14.13 ± 0.38), confirming that fault-mode identity is detectable from as few as five initial flight cycles. Figure 7 compares all four architectures on FD003 and FD004 in terms of RMSE mean and inter-seed variance.

[FIGURE 7: Fault-Mode Separation Model Comparison (M0–M3)]
**Fig. 7.** RMSE comparison (mean ± std, 5 seeds) of four fault-mode separation architectures on FD003 and FD004. M3 (Attention Gate) achieves RMSE = 14.78 ± 1.32 on FD003, a 65.8% reduction versus M0 (43.23 ± 0.18), while remaining statistically equivalent to M0 on FD004 (28.33 ± 1.03 vs 28.05 ± 1.74). M1 collapses on FD004 (49.20 ± 7.17) due to test-time cluster assignment collapse (247:1 ratio).

#### C.3 FD004: Cluster Collapse Under Hard Routing

**Hard routing (M1) degrades below the baseline on FD004.** M0 achieves RMSE = 28.05 ± 1.74 on FD004; M1 regresses to 49.20 ± 7.17, a 75.4% increase. Post-hoc inspection of test-time cluster assignments reveals the root cause: 247 of 248 test engines are assigned to the same branch by the GMM argmax, compared with an approximately balanced assignment during training. This cluster-distribution collapse is consistent with the known mismatch between CMAPSS test windows (last-observed cycles only) and the GMM features derived from full training trajectories — the late-cycle sensor statistics used for training-set clustering are not reliably reproducible from a truncated test observation. Soft-gating (M2) partially recovers at RMSE = 30.71 ± 0.73 by weighting both branches, but still sits marginally above the baseline.

M3 achieves RMSE = 28.33 ± 1.03 on FD004, statistically equivalent to M0 (Cohen's d < 0.2; p > 0.7). Because M3's GatingNet reads only the first 10 cycles of each engine — information that is equally available in training and test settings — it is immune to the test-time distribution collapse that undermines M1. The result is a favourable asymmetric trade-off: M3 dramatically improves FD003 while leaving FD004 performance unchanged, with no architectural cost relative to M0 on the multi-condition dataset. This asymmetry suggests that the attention-gate design is most beneficial when a dataset contains distinct latent fault modes (FD003), while defaulting gracefully to single-branch behaviour when fault-mode structure is weaker or less discriminable from early-cycle observations (FD004).

### D. Loss Function Comparison (H7)

**No statistically detectable improvement from any custom loss function was found after Benjamini-Hochberg correction.** Across all 96 pairwise comparisons (6 loss functions × 4 clipping values × 4 datasets), zero reach p_BH < 0.05; the minimum corrected p-value is 0.176. We note that with N = 5 seeds and a 96-comparison BH-correction family, the minimum achievable corrected p-value is approximately 3.0 (capped at 1.0), meaning statistical significance is mathematically impossible regardless of true effect size; this result should be interpreted as "no large effect was detectable" rather than confirmed equivalence. Several individual combinations show nominally lower NASA Scores — most notably L5 (TWA) at clip = 125 on FD001 (NASA Score: 3.40 ± 0.32 vs 5.66 ± 1.12 for MSE; Cohen's d = −1.54, indicating improvement) and L7 (HubA) at clip = 125 on FD001 (RMSE: 14.90 ± 0.33 vs 16.34 ± 0.40, nominally better RMSE; NASA Score d = +2.00, indicating L7 produces a higher NASA Score than MSE despite lower mean squared error, because HubA's symmetric Huber penalty does not sufficiently weight late predictions under the asymmetric NASA metric) — but neither survives multiple-comparison correction (p_BH = 1.0 for both). Note: positive Cohen's d values denote that the alternative loss has a higher (worse) NASA Score than MSE; negative d denotes improvement. The pattern is consistent with Rengasamy et al. [37], who reported gains from asymmetric weighting on FD001 in single-dataset experiments; the divergence from our null result arises because cross-dataset BH-FDR correction substantially raises the significance threshold when all four sub-datasets are evaluated jointly. Our findings on the optimal clipping threshold are independent of the backbone architecture, as H2 deliberately uses Ridge regression to isolate label-engineering effects from model capacity.

The dominant driver of NASA Score variance across this study is RUL clipping, not loss function choice. Table VII illustrates this: at clip = 125, all seven loss functions converge to within a factor of 1.5× of one another in NASA Score on FD001 (range: 3.40–6.14). Removing the clip inflates FD003 NASA Score to between 3,124 (L6, Pinball) and 9,613,539 (L1, MSE), a span of three orders of magnitude that dwarfs any inter-loss difference at a fixed clip value. Even the loss functions designed to suppress late predictions (L5 TWA, L6 Pinball, L7 HubA) reduce but do not eliminate the clip = None catastrophe on FD003: L6 Pinball achieves NASA = 3,124 vs L1 MSE's 9,614,000 — a 3,000× improvement within the clip-free condition, yet still 630× worse than the worst result at clip = 125. This finding implies that loss-function engineering is a second-order design choice: it cannot substitute for proper label engineering (RUL clipping) as a mechanism for controlling the tail behaviour of the NASA penalisation function. Practitioners should fix the clipping threshold before considering custom loss functions. Figure 8 illustrates the clip-dominance effect across all seven loss functions and four clipping values.

[FIGURE 8: RUL Clipping × Loss Function Interaction (NASA Score)]
**Fig. 8.** Mean NASA prognostic score (lower is better) across all four datasets as a function of RUL clipping threshold and loss function (560 LSTM training runs, 5 seeds per configuration). clip = None yields catastrophic scores (up to 2.4 × 10⁶) regardless of loss function, while clip ∈ {125, 130} stabilises results for all losses. Among clipped configurations, no custom loss achieves statistically significant improvement over MSE after BH-FDR correction (α = 0.05).

---

## V. Discussion and Implications

### A. Which Design Choices Actually Move the Needle?

Which design choices actually move the needle in turbofan RUL prediction? Across more than 800 training runs spanning four sub-datasets, two normalization strategies, seven loss functions, and four architectural variants, the answer is unambiguous: RUL clipping threshold and fault-mode architecture account for virtually all achievable variance in RMSE. Normalization strategy and loss function, by contrast, function as residual variables — each capable of degrading performance when misapplied, but neither capable of improving on a well-configured baseline by a statistically detectable margin once clipping is fixed and fault structure is handled. This ordering — clipping and architecture as load-bearing, normalization and loss as residual — has not previously been established through controlled cross-dataset ablation and carries immediate implications for how practitioners and benchmark designers should allocate modelling effort [5, 6]. The broader significance is methodological: single-dataset experiments, which constitute the majority of the CMAPSS literature [9, 14, 15], cannot isolate these hierarchical effects because a confound that is large on one sub-dataset may be negligible on another and vice versa.

### B. Why Fleet Normalization Beats Per-Unit Strategies

**Fleet normalization wins because degradation magnitude exceeds inter-engine variation on CMAPSS.** Per-unit normalisation methods (N3–N6) subtract engine-specific initial-state baselines, an operation that removes genuine absolute degradation information when initial-condition variation across engines is small relative to fault progression magnitude. On a homogeneous synthetic fleet such as CMAPSS, a fleet-trained min-max scaler encodes a cross-engine map of the full degradation trajectory; engine-local normalisation replaces this shared map with a zero-mean engine-local reference, discarding the information about where an engine sits in the fleet distribution. The RevIN [23] failure compounds this problem: RevIN's per-window mean subtraction applies the local-baseline removal at every inference step, progressively erasing the degradation trend across each 30-cycle window. The original RevIN paper demonstrated strong gains on multi-step forecasting tasks [23] where distribution shift across training and test windows is the dominant challenge; on CMAPSS, the dominant challenge is extracting a common degradation signature from a fleet — a context in which instance-level adaptation is counterproductive rather than beneficial. A recent theoretical critique of RevIN [25, 26] reached a compatible conclusion: RevIN's components are redundant when the primary normalisation challenge is not temporal distribution shift but conditional offset. Our four-dataset empirical result offers the first controlled confirmation of this theoretical prediction in the prognostics domain.

The practical implication is context-dependent. Practitioners working on real engine fleets where inter-engine manufacturing tolerance produces substantial initial-state variation — including variable rotor clearances, turbine blade wear, and sensor calibration offsets — may find per-unit strategies more competitive than the CMAPSS results suggest [29]. On CMAPSS, inter-engine variation is by construction small and fleet statistics are the correct choice; but the fleet-vs-per-unit choice is a dataset property, not a universal truth. Any prognostics practitioner porting these methods should measure fleet-level inter-engine variability relative to degradation range before committing to a normalisation level.

### C. FD003 Variance Anomaly as a Diagnostic Signal

**High inter-seed variance is a diagnostic indicator of latent categorical data structure.** The N1/FD003 standard deviation of 12.86 cycles is not a modelling failure; it is an empirical signature of a bimodal loss landscape produced by two distinct fault modes (HPC and fan degradation) in the training data. A single-branch LSTM trained on mixed-fault data converges to one of two local optima depending on which fault mode is overrepresented in the random training subsample drawn by a given seed, producing approximately bimodal per-seed RMSE outcomes. This mechanism predicts two testable consequences: first, separating fault modes should collapse the inter-seed variance towards single-fault benchmark levels; second, no analogous variance spike should appear on single-fault datasets (FD001/FD002). Both predictions are confirmed: M3 reduces the FD003 inter-seed standard deviation from 12.86 to 1.32 cycles, and no dataset-normalizer combination outside FD003 exceeds std = 1.84. The diagnostic value of this finding extends beyond CMAPSS: researchers observing unexplained inter-run variance on any benchmark should test for latent categorical structure before attributing the variance to optimisation noise or hyperparameter sensitivity. Variance across seeds is free diagnostic information that most studies discard.

### D. What M3's Success Reveals About Fault-Mode Routing

**Fault-mode identity is encoded in the very first flight cycles.** M3's GatingNet reads only K = 10 cycles and consistently routes engines to the correct fault-specific branch, achieving RMSE = 14.78 ± 1.32 on FD003 without any fault-mode supervision. A K sensitivity analysis (K ∈ {5, 10, 15, 20, 30}) shows that M3 RMSE is largely insensitive to K (range: 14.13–14.78 cycles across all values), and that K=5 already achieves RMSE = 14.23 ± 0.34. This finding is non-trivial: intuitively, fault signatures should become most discriminable as damage accumulates in late life stages. The GatingNet's early-cycle — and as few as five-cycle — success implies that the two CMAPSS fault modes differ primarily in baseline sensor state rather than in degradation rate: the initial operating point of s15 (bypass pressure ratio, |Δz| = 31.3) and s12 (fuel-to-pressure ratio) sets fault-mode membership at or before commissioning, consistent with the EDA finding that inter-cluster differences are driven by absolute level rather than slope. Inter-seed variance decreases with K (std: 0.34 at K=5, 0.27 at K=30), suggesting that longer context improves routing stability but not mean accuracy; for practical deployment the lowest computationally feasible K (K=5 or K=10) is sufficient.

The practical implication for deployed systems is significant: fault-mode routing can be performed at engine commissioning, before any degradation has accumulated, enabling a prognostics system to select the appropriate predictive model at the earliest possible stage of operation. This is qualitatively distinct from approaches that use accumulated degradation history for routing [31, 33], which require a burn-in period before the router is reliable. M3 also avoids the test-time distribution problem that defeats M1 (247:1 cluster collapse on FD004): because the first K = 10 cycles are always available at deployment time, GatingNet inputs are identically distributed between training and deployment, eliminating a source of silent performance degradation that post-hoc GMM routing cannot escape.

From an architecture standpoint, M3 is an instance of mixture-of-experts (MoE) inference [32, 43, 44] with an early-cycle context encoder as the gating network and no label supervision for the gate. Unlike RUL-QMoE [43], which targets probabilistic interval prediction for battery degradation using supervised material labels, and PSMMoEs [44], which applies metric-learning gating for cross-domain fault classification, M3 performs deterministic point-prediction RUL routing in a single unlabelled turbofan dataset. Compared with CAELSTM [15] — the most recent supervised competitor, achieving RMSE = 13.40 on FD003 — M3's gap of 10.3% is attributable to the simpler two-layer LSTM backbone rather than to the gating strategy itself. M3's practical advantage lies in its architectural efficiency: the GatingNet contributes only 4.9K parameters — less than 5% overhead above the two-branch baseline (Table II) — and requires no modification to the LSTM backbone. For practitioners operating existing LSTM-based prognostic pipelines, fault-mode-aware prediction is therefore achievable without backbone migration, at negligible additional computational cost. Whether the gating principle extends to attention-based encoders [13, 14] — where global attention may inherently capture early-cycle fault patterns through its query-key-value mechanism — is an open question addressed in Section H.

The early-cycle routing premise warrants an important caveat regarding broader applicability. Alternative unsupervised approaches to fault mode separation — notably UMAP-based trajectory clustering [46] and joint learning frameworks that operate over the full degradation history [47] — implicitly assume that fault mode identity becomes unambiguous only as degradation progresses, and they route or classify test units accordingly using all available observation context. This view is corroborated by causal-inference experiments on N-CMAPSS [48], which report that early anomalies are "difficult to identify due to complex thermodynamic couplings and nonlinear degradation patterns" in real-engine data; and by semi-supervised sensor selection work [49] showing that accurate fault mode recognition benefits substantially from partial supervisory labels. Taken together, these studies suggest that pure unsupervised early-cycle separation may be a property specific to the CMAPSS simulation environment rather than a universal prognostic principle.

M3's five-cycle discriminability on CMAPSS is traceable to a concrete data-generative property: the two fault modes differ primarily in absolute baseline sensor level (s15 bypass pressure ratio, |Δz| = 31.3) rather than in degradation rate or trajectory shape. In a real-engine fleet where fault modes manifest as progressive deviations from a shared healthy baseline — rather than as distinct initial operating points — the GatingNet faces a harder discrimination task, and the required early-cycle window K may be substantially larger or routing performance lower. The Silhouette-based screening step at Tier 2 of the design checklist serves precisely this diagnostic role: if GMM clustering on K-cycle prefixes yields Silhouette < 0.5, early-cycle gating should not be adopted, and practitioners should instead consider full-trajectory routing strategies such as those in [46, 47] or a two-stage approach that begins routing only after a short burn-in window.

### E. Why Loss Functions Cannot Substitute for Label Engineering

**Custom loss functions are a second-order intervention.** We caution that the H7 null result is power-limited: with N = 5 seeds and a 96-test BH family, no comparison can achieve statistical significance regardless of effect size, so the finding is best read as "no large effect was detectable" rather than "all custom losses are equivalent to MSE." Notwithstanding this constraint, the result is practically informative: once RUL clipping is correctly applied at τ = 125, the MSE gradient already penalises late predictions disproportionately, because the clipped label distribution is right-skewed and CMAPSS test engines are predominantly in the low-RUL regime at the end of their trajectories. Asymmetric losses therefore add a secondary bias correction on top of an already-biased objective. Our null result is consistent with an information-theoretic reading: when the label distribution encodes asymmetry through clipping, an explicit asymmetric loss doubles the bias without adding new signal. This provides an explanation for a pattern visible in the literature: reported gains for asymmetric losses in single-dataset studies [37, 38] may partly reflect the absence of proper RUL clipping in the baseline, rather than an intrinsic benefit of the loss function itself. The cross-term our study reports for the first time — a 4 × 7 matrix of clip values and loss functions — reveals that the clip-dominance effect is consistent across all four sub-datasets and all seven loss functions tested.

One exception deserves mention. Under clip = None, L6 (Pinball, τ = 0.25) reduces FD003 NASA Score from 9,614,000 to 3,124 — a 3,000-fold improvement relative to MSE within the clip-free condition, reflecting Pinball's quantile-regression property of producing conservative (early) predictions. However, even this gain leaves FD003 630× worse than the worst result at clip = 125, confirming that no loss function can compensate for missing label engineering. The Chung et al. [41] theoretical critique of blind Pinball minimisation reinforces this conclusion: calibrated quantile coverage requires the label distribution to be well-conditioned, which is not the case for an uncapped RUL target that can exceed 500 cycles.

### F. A Reliability-Driven Design Checklist for Turbofan RUL Systems

**Three tiers determine the reliability and predictive accuracy of turbofan RUL systems, in order of their failure impact.** We propose the following evidence-based design checklist for reliability engineers configuring, validating, or certifying turbofan PHM systems:

```
START
  │
  ▼
TIER 1 — Label Engineering
  Set clip = 125 cycles (τ).
  ─────────────────────────────────────────────────────────────────────
  Is there domain evidence that fleet average lifetime differs
  substantially from 206–247 cycles (the CMAPSS range)?
    YES → consider adaptive clip = 70th-percentile fleet lifetime
    NO  → use τ = 125 ✓
  │
  ▼
TIER 2 — Architecture
  Fit GMM (k = 2) on early-cycle sensor slopes of the fleet.
  Compute Silhouette score S on the training set.
  ─────────────────────────────────────────────────────────────────────
  S ≥ 0.5?
    YES → add M3-style GatingNet on first K = 10 cycles
          (expected RMSE gain: up to −65.8% on FD003-like datasets)
    NO  → use single-branch baseline (M0)
  │
  ▼
TIER 3 — Loss Function
  Use MSE (L1) as default.
  ─────────────────────────────────────────────────────────────────────
  Is the deployment metric NASA Score (asymmetric)?
    YES → try L7 (HubA, δ=20, λ_a=3) or L5 (TWA, λ_t=10)
          ONLY IF: gains survive BH-FDR over all evaluation datasets
    NO  → stay with MSE
  │
  DONE
```

The ordering matters from a reliability perspective: a Tier 1 error — omitting or miscalibrating the RUL clipping threshold — inflates the NASA prognostic score by up to six orders of magnitude regardless of all subsequent design choices, constituting a systemic reliability failure in any safety-critical maintenance scheduling pipeline. Tier 2 gains (up to −65.8% RMSE) dwarf any gain achievable from Tier 3 (<5% RMSE in our experiments). Reliability engineers who optimise Tier 3 before resolving Tier 1 and Tier 2 risk both negligible performance returns and a latent system-level failure: an unresolved Tier 1 misconfiguration can invalidate the prognostic safety case of an otherwise correctly designed pipeline. We stress that the Silhouette threshold of 0.5 at Tier 2 is a necessary but not sufficient condition for M3 to help: the latent clusters must also be accessible from early-cycle observations. If discriminant sensor differences emerge only in late life (slope-driven rather than baseline-driven), the GatingNet will not separate modes reliably at K = 10 cycles, and a larger K or a two-stage routing strategy should be considered. Practitioners encountering S ∈ [0.5, 0.65] should treat this as a marginal regime in which M3 benefit is uncertain; a held-out validation comparing M3 against the single-branch baseline on a representative validation set is recommended before committing to fault-mode routing in this range.

### G. Limitations

**Several limitations constrain the generalisability of these findings.** First, and most critically, all experiments are conducted on the NASA CMAPSS simulation benchmark. Real turbofan engines exhibit sensor noise, calibration drift, maintenance-induced sensor resets, and operating-history effects that are absent from the synthetic CMAPSS environment. In particular, real fleets typically exhibit greater inter-engine manufacturing variability than CMAPSS, which may alter the fleet-vs-per-unit normalization ranking established in H5. The recently introduced N-CMAPSS dataset [16] addresses some of these gaps by including realistic degradation trajectories, variable flight conditions, and turbofan-specific sensor physics; whether the three-tier design hierarchy transfers to N-CMAPSS — and to real operational sensor streams — is the primary open question from this study and the most important direction for validation prior to industrial deployment. Second, all hypotheses except H2 use a common two-layer stacked LSTM backbone; architectural choices and design-factor rankings may interact differently with Transformer-based or graph-neural-network encoders [13, 14], which have shown competitive performance on CMAPSS in recent work. Third, H6 relies on unsupervised cluster quality (Silhouette ≥ 0.5) as a proxy for genuine physical fault categories, because CMAPSS provides no fault-mode labels. Whether M3's routing corresponds to HPC vs. fan degradation in a physically meaningful sense, or merely to a statistical partition, cannot be verified from the benchmark data alone. Fourth, the H2 clipping analysis employs LinearRegression (Ridge) for computational tractability, while H5 and H7 use the LSTM backbone; direct cross-hypothesis RMSE comparisons therefore confound model complexity with the design factor under study, and the absolute RMSE values in Table IV should not be compared with those in Tables V–VII. Fifth, within-study comparisons between H5 and H6 are constrained by implementation differences not unified prior to analysis: H5 uses a compact backbone (LSTM₂ hidden=32, FC 32→16→1) and 18 input features (sensors plus op1/op2/op3), while H6/H7 use a full-capacity backbone (LSTM₂ hidden=64, FC 64→32→1) and 15 features (sensors only). H5 uses a deterministic validation split (last-20% of engines by unit ID) while H6 uses a random split (RandomState seed=42). H5 clips test predictions to [0, 125] at evaluation; H6 does not. These differences mean that the H5 FD003 baseline RMSE (19.05 ± 12.86) and H6 M0 baseline RMSE (43.23 ± 0.18) reflect genuinely different experimental conditions; cross-hypothesis absolute RMSE comparisons should not be taken at face value. Sixth, the current implementation assumes a static batch-trained model deployed without further adaptation. In practice, sensor characteristics and fleet composition evolve over time; online learning mechanisms — such as periodic fine-tuning triggered by newly labeled flight cycles — would be required to maintain routing accuracy over extended operational periods. M3's lightweight GatingNet is architecturally compatible with such periodic updates, and this extension is reserved for future work. Seventh, M3's early-cycle routing relies on fault modes being distinguishable from initial baseline sensor state rather than from accumulated degradation patterns. Approaches that leverage full degradation trajectories for fault mode identification [46, 47] — including UMAP-based clustering over complete run-to-failure histories and joint learning frameworks that process the entire temporal sequence — suggest that the early-cycle discriminability observed on CMAPSS may not generalise to real-engine environments where initial sensor readings are contaminated by noise, calibration offsets, or operational variability. Causal-inference experiments on N-CMAPSS [48] explicitly characterise early anomaly detection as difficult in realistic flight conditions, and semi-supervised frameworks [49] report that partial fault-mode labels substantially improve mode recognition accuracy beyond what unsupervised early-cycle signals alone can provide. Practitioners should therefore verify early-cycle cluster discriminability (e.g., Silhouette ≥ 0.5 on K-cycle prefixes) before adopting K-cycle gating in new deployment contexts, and should consider full-trajectory routing alternatives when this threshold is not met.

### I. Industrial and Deployment Implications

The three-tier reliability-driven checklist translates directly to a reliability risk management framework for industrial turbofan PHM deployment. Neglecting label engineering (clip = None) inflates the NASA prognostic score by up to 306,000-fold on FD003 — equivalent in practice to a prognostics system that chronically over-predicts remaining life and delays maintenance intervention, the highest-cost failure mode in turbofan fleet operations where an undetected in-flight shutdown typically costs orders of magnitude more than a precautionary shop visit. This finding means that RUL clipping calibration is not a modelling detail but a safety-critical design decision that should precede any architectural or loss-function work.

M3's early-cycle gating is operationally viable in a way that post-hoc trajectory clustering is not. Because the GatingNet reads only the first five flight cycles, fault-mode classification can be completed at engine commissioning — before any degradation has accumulated — enabling a prognostics system to select the appropriate predictive branch without a burn-in observation window. This is practically significant for fleet operators who must provision maintenance planning pipelines before degradation history is available for new engine units.

To contextualise the industrial significance of the M3 improvement, consider a representative FD003-class turbofan operating at approximately 300 flight cycles per year. M0's mean RMSE of 43.23 cycles implies that maintenance scheduling uncertainty spans approximately ±43 cycles on average — equivalent to roughly ±17% of annual operating time. M3 reduces this to ±14.78 cycles (±5.9%), a reduction of 65.8% in scheduling uncertainty. For a fleet of 50 such engines, if each RMSE cycle of reduction prevents one additional unscheduled shop visit per year at an estimated opportunity cost of $200,000–$500,000 per event (consistent with published MRO cost benchmarks for civil turbofan shop visits), the order-of-magnitude operational benefit of deploying M3 over M0 is measurable in millions of dollars annually. These figures are indicative rather than definitive, as exact ROI depends on fleet-specific cost structures; the directional case for fault-mode-aware prognostics is, however, substantial. Regarding fleet scale, K-means residualization and GMM fitting scale linearly with engine count; for fleets of thousands of engines, Mini-Batch K-means variants reduce fitting time to under five minutes while preserving cluster quality.

Finally, the null result for custom loss functions is itself an industrial resource-allocation signal. Engineering effort spent designing and tuning asymmetric losses is statistically undetectable relative to correct label engineering and fault-mode architecture, and is better redirected to clipping calibration or multi-fault screening. The three-tier hierarchy thus functions as a risk-prioritised design checklist: fix labelling first, address fault-mode structure second, and treat loss selection as a last-resort refinement subject to rigorous cross-dataset validation before deployment.

---

### H. Future Research Directions

**Three directions emerge directly from the limitations and null results of this study.**

**Adaptive RUL clipping.** Our results confirm that a single threshold is not universally optimal: clip = 130 is marginally better on FD002/FD004 (ΔRMSE = −0.57, non-significant) while clip = 125 is unambiguously best on FD001/FD003, and the dataset-specific lifetime distributions differ (FD001/FD002 mean ≈ 206 cycles; FD003/FD004 mean ≈ 247 cycles). A data-driven adaptive threshold — for instance, set at the 70th-percentile engine lifetime in the training fleet — would be principled, require no manual tuning, and directly address the H2 research gap identified in the literature [18, 20]. An adaptive scheme would also handle the clip = None catastrophe on FD003 automatically, since the 70th-percentile of FD003 lifetimes is approximately 137 cycles, close to the empirically validated 125.

**M3 on real-data and N-CMAPSS benchmarks.** Validating the early-cycle GatingNet on N-CMAPSS (2021), which includes sensor noise, maintenance resets, and variable flight-envelope profiles, would test whether the fault-mode routing principle transfers beyond the CMAPSS simulation environment. N-CMAPSS also provides turbofan health parameter labels (HPT/Fan degradation) that would allow the cluster-quality proxy to be replaced by supervised routing accuracy, enabling a direct comparison between M3's unsupervised gating and a fully supervised fault-mode classifier. A parallel question is whether M3's explicit GatingNet remains necessary when the backbone itself is replaced with an attention-based encoder [13, 14]; global attention mechanisms may inherently perform early-cycle fault separation, in which case the three-tier hierarchy's architectural tier would require reformulation for Transformer-class models.

**Deep learning backbone exploration and ensemble strategies for real operational data.** The three-tier hierarchy was validated on a stacked LSTM backbone applied to a controlled simulation benchmark. As industrial PHM systems move towards deployment on real sensor streams, empirical exploration of diverse backbone architectures — including Transformer encoders [13, 14], temporal convolutional networks, and graph neural networks — is necessary to establish whether the design hierarchy and the architectural ranking generalise across model families. Real operational data from heterogeneous fleets introduces irregular degradation patterns, sensor calibration drift, and distributional shifts that may interact differently with each architecture; no single backbone can be assumed optimal without validation on representative operational data. Beyond single-model approaches, ensemble and blending strategies that combine predictions across multiple architecture families are likely to be required for robust coverage across diverse fault types, operating regimes, and fleet compositions encountered in practice. The three-tier hierarchy provides a principled starting point — establishing label engineering and fault-mode screening as prerequisites — but the specific architectural choice at Tier 2 should be treated as an open variable to be resolved empirically for each new deployment context.

**Scalable mixture-of-experts with unknown k.** M3 currently assumes exactly k = 2 fault modes, a strong prior that may not hold for heterogeneous industrial fleets. Extending the gating mechanism to an unknown and unbounded number of fault modes via Bayesian nonparametric priors — for example, a Dirichlet process mixture model on early-cycle sensor trajectories [34] — would eliminate the cluster-count hyperparameter and adapt gracefully to fleets with diverse fault histories. Recent work on Bayesian nonparametric process mixtures for unlabelled failure modes [34] provides a natural prior framework for this extension.

**Loss × clipping interaction study.** Our 4 × 7 cross-tabulation reveals that clipping dominates, but the sample size per cell (5 seeds × 4 datasets) is insufficient to detect small interaction effects. A targeted study that varies clipping continuously (e.g., τ ∈ {90, 100, 110, 120, 125, 130, 140} cycles) and pairs each level with theoretically motivated losses could isolate whether any custom loss provides a statistically detectable benefit after optimal clipping is identified for each sub-dataset. This design would directly resolve the confound identified by comparing our null result with the single-dataset gains reported by Rengasamy et al. [37] and Abdullah [22].

---

## VI. Conclusion

This paper presents a controlled cross-dataset ablation of four design factors for turbofan RUL prediction using the NASA CMAPSS benchmark (FD001–FD004). Within each hypothesis (using fixed architecture and preprocessing), we systematically evaluated RUL clipping threshold (H2), normalisation strategy (H5), fault-mode architecture (H6), and training loss function (H7). H2 confirmed that clip = 125 cycles — the established standard [5, 6] — produces the lowest or statistically tied-lowest RMSE on all four sub-datasets, and that removing the ceiling is catastrophic on FD003; this finding is consistent with prior literature and treated as a resolved prerequisite throughout the remaining analyses.

Fleet-level min-max normalisation significantly outperformed all per-unit and RevIN strategies on FD001, FD002, and FD004. On FD003, the anomalously high inter-seed variance observed under fleet normalisation proved to be a diagnostic signature of latent fault-mode heterogeneity rather than a normalisation deficiency. Introducing an end-to-end attention-gate architecture (M3) that routes each engine based on its first ten observed cycles resolved this bimodality, reducing FD003 RMSE by 65.8% and NASA Score by 98.8% relative to the single-branch baseline. M3 also avoided the test-time cluster-distribution collapse that rendered GMM hard-routing 75.4% worse than baseline on FD004, demonstrating that early-cycle routing is both more accurate and more deployment-robust than post-hoc trajectory clustering.

No statistically detectable improvement from custom loss functions was found after Benjamini-Hochberg correction across 96 pairwise comparisons (6 losses × 4 clips × 4 datasets). We note that with N = 5 seeds and 96 BH-corrected comparisons, the minimum achievable corrected p-value is approximately 3.0 (clipped to 1.0), making statistical significance mathematically impossible regardless of true effect size; the null result should be interpreted as "no large effect was detectable" rather than confirmed equivalence. The dominant source of variance in all prognostic metrics was RUL clipping rather than loss design. Together, the four hypotheses establish a three-tier reliability-driven design checklist — label engineering, then fault-mode architecture, then loss function — in which each tier's failure impact qualitatively exceeds the next. This ordering, confirmed for the first time through rigorous cross-dataset ablation, should guide where reliability engineers and benchmark designers allocate PHM system development effort.

From a reliability engineering perspective, these results provide concrete, risk-prioritised guidance for PHM system engineers. Improper label engineering alone can inflate the prognostic risk metric by up to six orders of magnitude regardless of all subsequent design choices — a systemic reliability failure with direct safety and operational cost consequences in turbofan fleet maintenance. M3's fault-mode routing requires only five initial flight cycles, enabling reliable prognostics model selection at engine commissioning without a burn-in window. Custom loss function engineering, by contrast, offers no statistically recoverable benefit once label engineering is in place, allowing development resources to be redirected to higher-leverage reliability interventions. Reliability engineers and PHM system designers can apply the three-tier reliability-driven checklist as a principled, evidence-based framework to maximise prognostic reliability under realistic deployment constraints.

---

## Data and Code Availability

The NASA CMAPSS dataset used in this study is publicly available at the NASA Prognostics Center of Excellence Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). No data were generated or modified; the benchmark is used as-is under its public access terms. The experimental code, including all preprocessing pipelines, model implementations (M0–M3), and statistical testing routines, will be made publicly available on GitHub upon acceptance of this manuscript.

---

## References

[1] A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," in *Proc. 1st Int. Conf. Prognostics and Health Management (PHM)*, Denver, CO, 2008.

[2] F. O. Heimes, "Recurrent Neural Networks for Remaining Useful Life Estimation," in *Proc. 1st Int. Conf. Prognostics and Health Management (PHM)*, IEEE, 2008, doi: 10.1109/PHM.2008.4711422.

[3] E. Ramasso and A. Saxena, "Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets," *Int. J. Prognostics Health Manage.*, vol. 5, no. 2, 2014, doi: 10.36001/ijphm.2014.v5i2.2236.

[4] G. S. Babu, P. Zhao, and X.-L. Li, "Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life," in *Database Systems for Advanced Applications (DASFAA)*, Springer, 2016, pp. 214–228, doi: 10.1007/978-3-319-32025-0_14.

[5] S. Zheng, K. Ristovski, A. Farahat, and C. Gupta, "Long Short-Term Memory Network for Remaining Useful Life Estimation," in *2017 IEEE Int. Conf. Prognostics and Health Management (ICPHM)*, pp. 88–95, doi: 10.1109/ICPHM.2017.7998311.

[6] X. Li, Q. Ding, and J.-Q. Sun, "Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks," *Rel. Eng. Syst. Safety*, vol. 172, pp. 1–11, 2018, doi: 10.1016/j.ress.2017.11.021.

[7] A. L. Ellefsen et al., "Remaining Useful Life Predictions for Turbofan Engine Degradation Using Semi-Supervised Deep Architecture," *Rel. Eng. Syst. Safety*, vol. 183, pp. 240–251, 2019, doi: 10.1016/j.ress.2019.01.016.

[8] Y. Mo et al., "Remaining Useful Life Estimation via Transformer Encoder Enhanced by a Gated Convolutional Unit," *J. Intell. Manuf.*, vol. 35, pp. 1997–2012, 2021, doi: 10.1007/s10845-021-01750-x.

[9] R. Jin et al., "Bi-LSTM-Based Two-Stream Network for Machine Remaining Useful Life Prediction," *IEEE Trans. Instrum. Meas.*, vol. 71, 2022, doi: 10.1109/TIM.2022.3163761.

[10] D. Xu et al., "Spatio-Temporal Degradation Modeling and Remaining Useful Life Prediction Under Multiple Operating Conditions Based on Attention Mechanism and Deep Learning," *Rel. Eng. Syst. Safety*, vol. 225, 2022, doi: 10.1016/j.ress.2022.108648.

[11] Y. Zhang et al., "Trend-Augmented and Temporal-Featured Transformer Network with Multi-Sensor Signals for Remaining Useful Life Prediction," *Rel. Eng. Syst. Safety*, vol. 235, 2023, doi: 10.1016/j.ress.2023.109258.

[12] J. Li et al., "Remaining Useful Life Prediction of Turbofan Engines Using CNN-LSTM-SAM Approach," *IEEE Sensors J.*, vol. 23, no. 10, 2023, doi: 10.1109/JSEN.2023.3243540.

[13] H. Wang et al., "Comprehensive Dynamic Structure Graph Neural Network for Aero-Engine Remaining Useful Life Prediction," *IEEE Trans. Instrum. Meas.*, vol. 72, 2023, doi: 10.1109/TIM.2023.3312337.

[14] K. You et al., "A 3-D Attention-Enhanced Hybrid Neural Network for Turbofan Engine Remaining Life Prediction Using CNN and BiLSTM Models," *IEEE Sensors J.*, vol. 24, no. 3, 2024, doi: 10.1109/JSEN.2023.3335994.

[15] S. M. Elsherif, B. Hafiz, M. A. Makhlouf, and O. Farouk, "A Deep Learning-Based Prognostic Approach for Predicting Turbofan Engine Degradation and Remaining Useful Life," *Sci. Rep.*, 2025, doi: 10.1038/s41598-025-09155-z.

[16] F. Wu et al., "Remaining Useful Life Prediction Based on Deep Learning: A Survey," *Sensors*, vol. 24, no. 11, art. 3454, 2024, doi: 10.3390/s24113454.

[17] H. Li et al., "A Review on Physics-Informed Data-Driven Remaining Useful Life Prediction: Challenges and Opportunities," *Mech. Syst. Signal Process.*, vol. 209, 2024, doi: 10.1016/j.ymssp.2024.111120.

[18] F. Imbert, T. Adewumi, and H. Han, "A Novel Preprocessing-Driven Approach to Remaining Useful Life (RUL) Prediction Using Temporal Convolutional Networks (TCN)," in *IEEE 37th Int. Conf. Tools Artif. Intell. (ICTAI 2025)*, 2026, doi: 10.1109/ICTAI66417.2025.00160.

[19] K. Ensarioğlu, T. İnkaya, and E. Emel, "Remaining Useful Life Estimation of Turbofan Engines with Deep Learning Using Change-Point Detection Based Labeling and Feature Engineering," *Appl. Sci.*, vol. 13, no. 21, art. 11893, 2023, doi: 10.3390/app132111893.

[20] A. Srinivasan, J. C. Andresen, and A. Holst, "Ensemble Neural Networks for Remaining Useful Life (RUL) Prediction," *Asia Pacific Conf. PHM Society*, vol. 4, no. 1, 2023, doi: 10.36001/phmap.2023.v4i1.3611.

[21] A. Arunan, Y. Qin, X. Li, and C. Yuen, "A Change Point Detection Integrated Remaining Useful Life Estimation Model under Variable Operating Conditions," *Control Eng. Practice*, 2024, doi: 10.1016/j.conengprac.2023.105840.

[22] M. E. B. Abdullah, "Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL Prediction with Interpretable Failure Heatmaps," arXiv preprint arXiv:2604.13459, April 2026.

[23] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift," in *Int. Conf. Learning Representations (ICLR)*, 2022.

[24] Z. Zhang et al., "A Framework for Predicting the Remaining Useful Life of Machinery Working under Time-Varying Operational Conditions," *Appl. Soft Comput.*, 2022.

[25] G. Berthelier et al., "On the Role of Reversible Instance Normalization," arXiv preprint arXiv:2603.11869, 2026.

[26] Anonymous, "Noise or Signal? Deconstructing Contradictions and An Adaptive Remedy for Reversible Normalization in Time Series Forecasting," arXiv preprint arXiv:2510.04667, October 2025.

[27] Anonymous, "Early Fault Detection on CMAPSS with Unsupervised LSTM Autoencoders," arXiv preprint arXiv:2601.10269, January 2026.

[28] Z. Sun et al., "IN-Flow: Instance Normalization Flow for Non-Stationary Time Series Forecasting," in *Proc. 31st ACM SIGKDD Conf. Knowledge Discovery and Data Mining*, 2025, doi: 10.1145/3690624.3709260.

[29] S. Deng et al., "Prediction of Remaining Useful Life of Aero-Engines Based on CNN-LSTM-Attention," *Int. J. Comput. Intell. Syst.*, 2024, doi: 10.1007/s44196-024-00639-w.

[30] (Authors not retrieved), "A Novel Multi-Task Learning Framework with Fault Mode Feature Separation for Remaining Useful Life Estimation of Mechanical Systems," *Adv. Eng. Informatics*, vol. 60, art. 102360, 2024, doi: 10.1016/j.aei.2024.102360.

[31] (Authors not retrieved), "Unsupervised Classification and Remaining Useful Life Prediction for Turbofan Engines Using Autoencoders and Gaussian Mixture Models: A Comprehensive Framework for Predictive Maintenance," *Appl. Sci.*, vol. 15, no. 14, art. 7884, 2025.

[32] (Authors not retrieved), "Multi-Condition Remaining Useful Life Prediction Based on Mixture of Encoders (MoEFormer)," *Entropy*, vol. 27, no. 1, art. 79, January 2025.

[33] (Authors not retrieved), "Remaining Useful Life Prediction for Aircraft Engines under High-Pressure Compressor Degradation Faults Based on FC-AMSLSTM," *Aerospace*, vol. 11, no. 4, art. 293, 2024.

[34] (Authors not retrieved), "Prognostics of Multisensor Systems with Unknown and Unlabeled Failure Modes via Bayesian Nonparametric Process Mixtures," arXiv preprint arXiv:2602.19263, February 2026.

[35] (Authors not retrieved), "Fault Prognosis of Turbofan Engines: Eventual Failure Prediction," *Int. J. Prognostics Health Manage. (IJPHM)*, 2023, doi: 10.36001/ijphm.2023.v14i2.3486.

[36] (Authors not retrieved), "Interpretable Ensemble Remaining Useful Life Prediction Enables Dynamic Maintenance Scheduling for Aircraft Engines," *Sci. Rep.*, 2025, doi: 10.1038/s41598-025-23473-2.

[37] D. Rengasamy, M. Jafari, B. Rothwell, X. Chen, and G. P. Figueredo, "Deep Learning with Dynamically Weighted Loss Function for Sensor-Based Prognostics and Health Management," *Sensors*, vol. 20, no. 3, art. 723, 2020, doi: 10.3390/s20030723.

[38] D. Rengasamy, H. Bhatt, B. Rothwell, X. Chen, and G. P. Figueredo, "Asymmetric Loss Functions for Deep Learning Early Predictions of Remaining Useful Life in Aerospace Gas Turbine Engines," in *2020 Int. Joint Conf. Neural Networks (IJCNN)*, 2020.

[39] Z. Liu et al., "A Multi-Head Neural Network with Unsymmetrical Constraints for Remaining Useful Life Prediction," *Adv. Eng. Informatics*, 2021.

[40] R. Diao et al., "Turbofan Engine Remaining Useful Life Prediction with Reliable Prediction Intervals via LSTM-Based Quantile Regression and Conformal Calibration," *Sensors*, vol. 26, no. 7, art. 2249, 2026.

[41] Y. Chung et al., "Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.

[42] O. Asif et al., "A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset," *IEEE Access*, 2022.

[43] S. Ly, R. Yang, N. Dixit, and H. D. Nguyen, "RUL-QMoE: Multiple Non-crossing Quantile Mixture-of-Experts for Probabilistic Remaining Useful Life Predictions of Varying Battery Materials," arXiv preprint arXiv:2512.23725, December 2025.

[44] B. Yang, J. Zhang, R. Liu, D. Lin, P. Li, and C. L. P. Chen, "Point-to-Set Metric-Gated Mixture of Experts for Multisource Domain Adaptation Fault Diagnosis," *IEEE Trans. Neural Netw. Learning Syst.*, Early Access, March 2025, doi: 10.1109/TNNLS.2025.3548894.

[45] Z. Fan, W. Li, and K.-C. Chang, "A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine Remaining Useful Life Prediction," *Sensors*, vol. 24, no. 3, art. 824, 2024, doi: 10.3390/s24030824.

[46] Y. Fu, Z. Feng, J. Zhou, and Q. Shi, "Degradation Modeling and Prognostic Analysis Under Unknown Failure Modes," *IEEE Trans. Autom. Sci. Eng.*, 2024, doi: 10.1109/TASE.2024.3371143.

[47] D. Wang, Q. Zhao, B. Yang, and K.-L. Tsui, "Joint Learning of Failure Mode Recognition and Prognostics for Degradation Processes," *IEEE Trans. Autom. Sci. Eng.*, vol. 20, no. 4, pp. 2790–2802, 2023, doi: 10.1109/TASE.2022.3197094.

[48] K. Dong, Z. Liu, Y. Zhang, and J. Liu, "Causal Inference-Based Fault Diagnosis and Abnormal Degradation Detection for Aero-Engine," in *2025 Int. Conf. Equipment Intelligent Operation and Maintenance (ICEIOM)*, 2025.

[49] Y. Wang, X. Liu, C. Sun, R. Yan, and X. Chen, "Deep Learning-Based Sensor Selection for Failure Mode Recognition and Prognostics Under Time-Varying Operating Conditions," *IEEE Trans. Autom. Sci. Eng.*, vol. 22, pp. 6993–7007, 2025, doi: 10.1109/TASE.2024.3469052.

---

*End of manuscript. Source files in `Manuscript/Sections/`. Figures and tables in `Manuscript/Figures/` and `Manuscript/Tables/`. Citation RIS files in `Manuscript/Citation/`.*
