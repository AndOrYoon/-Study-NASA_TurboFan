# III. Methodology

> **Draft status:** v1.9 — 2026-08-31 (Synced with manuscript_full_text_clean.md: §III.D unified protocol 단락 추가, §III.E three→two reasons, §III.H per-hypothesis 설정 추가, §III.K 검정명 수정)
> **Style:** Elsevier single-column (elsarticle, review mode) — Markdown source; compiled to LaTeX via build_ress_latex.py

---

**Problem Formulation.** Engine $i$ ($i = 1, \ldots, N$) generates a multivariate sensor time series $\mathbf{X}^{(i)} \in \mathbb{R}^{T_i \times F}$, where $T_i$ is the run-to-failure lifetime in flight cycles and $F$ is the number of input features (varies by dataset and hypothesis; see §III.H). The model input at cycle $t$ is a fixed-length sliding window of 30 consecutive cycles:

$$\mathbf{W}_t^{(i)} = \mathbf{X}^{(i)}[t-29:t,\;:] \in \mathbb{R}^{30 \times F}$$

with front-zero-padding when $t < 30$. The RUL target follows a piecewise-linear formulation with clipping threshold $\tau$:

$$y_t^{(i)} = \min\!\bigl(T_i - t,\;\tau\bigr)$$

The prediction task is to learn $f : \mathbb{R}^{30 \times F} \to \mathbb{R}$ such that $f(\mathbf{W}_t^{(i)}) \approx y_t^{(i)}$ for all training cycles. At test time, $f$ is applied to the final observed window of each test engine and its output compared against the benchmark-provided ground-truth RUL. The four design variables studied in §IV are the clipping threshold $\tau$ (H1), the sensor preprocessing and normalisation strategy applied to $\mathbf{X}^{(i)}$ (H2), the architecture and routing mechanism of $f$ (H3), and the training objective used to optimise $f$ (H4).

---

## A. Dataset

The NASA CMAPSS benchmark was used, comprising four sub-datasets (FD001–FD004) simulating turbofan engine run-to-failure under varying operating conditions and fault modes (Table I). Each dataset records 21 raw sensor channels per flight cycle together with a held-out test set and ground-truth RUL values for the final observation of each test engine.

---

## B. Data Preprocessing

Constant-variance sensor channels were removed per dataset based on zero-variance inspection of the training split. For FD001, seven channels (s1, s5, s6, s10, s16, s18, s19) were discarded, leaving 14 input features. For FD003, six channels (s1, s5, s10, s16, s18, s19) were discarded — s6 exhibits non-zero variance in FD003 — leaving 15 input features. For FD002 and FD004, only s16 is constant, yielding 20 features. RUL targets followed a piecewise-linear formulation: for each training engine, cycles where the remaining life exceeds a clipping threshold τ receive a constant label of τ, while the final segment decreases linearly to zero. Five clipping values were evaluated (τ ∈ {75, 100, 125, 130, ∞}), with τ = 125 cycles serving as the standard baseline per established literature [5, 6]. Ground-truth test RUL values for each test engine's final observation are provided directly by the benchmark; test performance was evaluated by comparing model RUL predictions against these ground-truth labels.

---

## C. Operating-Condition Residualization

FD002 and FD004 contain six discrete operating conditions that shift absolute sensor levels by tens to hundreds of units. Although these conditions are identifiable via the three operating-condition columns (op1, op2, op3), those columns carry continuous floating-point values with cycle-level simulation noise — op1 takes 536 unique values and op2 takes 105 unique values across FD002 — making deterministic condition assignment by exact value matching infeasible. K-means (k = 6) was therefore applied to recover the six conditions unsupervised and to decouple degradation signals from operating-point offsets. A K-means model was fitted on the standardised op columns of the training set; per-cluster sensor means were computed from training data only and subtracted from each observation:

$$z_{i,j} = x_{i,j} - \mu_{c(i),\,j}$$

where c(i) denotes the cluster assignment of cycle i and μ_{c,j} is the training-set mean of sensor j within cluster c. This step precedes all normalization and is applied identically to training and test data using statistics derived exclusively from training engines. Post-hoc inspection confirms that each of the six K-means clusters maps to exactly one operating condition on both FD002 and FD004 (cluster purity = 1.0), validating that K-means recovers the true condition structure from the raw continuous op values. FD001 and FD003 present a single operating condition and therefore do not require this step.

---

## D. Normalization Strategies (H2)

Seven normalization strategies (N1–N7) were compared to evaluate the effect of the reference-statistics choice. Fleet-level methods compute statistics across all training engines: N1 applies min-max scaling to [0, 1] and N2 applies z-score standardisation. Per-unit methods (N3–N6) use each engine's own early-cycle observations as the reference baseline, removing initial-condition offsets before any degradation signal is visible: N3 and N5 apply min-max and z-score over the first 5 cycles; N4 and N6 apply the same transforms over the first 10 cycles.

N7 implements a forward-only variant of Reversible Instance Normalization (RevIN; Kim et al. [23]) as a learnable module within the model. Each 30-cycle inference window is normalised by its instantaneous mean and standard deviation, with trainable affine parameters (γ, β). The inverse transform is not applied to the scalar RUL output: RevIN's inverse is designed to restore multi-step sensor forecasts to their original measurement units — an operation defined for outputs that live in sensor space. Scalar RUL predictions are not in sensor space; applying the inverse transform is therefore undefined for this output type. N7 accordingly implements only the forward (normalisation) pass of RevIN, making it equivalent to per-window instance normalisation with learnable affine parameters.

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

**Protocol-unified N1–N3 robustness analysis.** To assess whether N1's advantage over per-unit normalization persisted across experimental protocols, N1 and N3 were directly compared using the H3 backbone configuration (full-capacity LSTM, sensors-only features, random 20% engine validation split, 30-epoch minimum warm-up, and prediction clipping to [0, 125]; 5 seeds per strategy per dataset). Results are reported in §IV.B.2 (40 runs: 2 strategies × 4 datasets × 5 seeds).

---

## E. LSTM Backbone Architectures

The stacked LSTM was selected as the controlled backbone for this ablation for two reasons. First, LSTM-based models constitute the dominant baseline class in the CMAPSS RUL literature [4, 5, 8], making results directly comparable with prior single-factor studies. Second, a controlled ablation requires the backbone to remain fixed across conditions; substituting an attention-based encoder would conflate backbone capacity with the design factor under study, preventing clean attribution of observed differences. Whether attention mechanisms perform fault-mode separation implicitly — rendering the explicit early-cycle GatingNet (M3) redundant over Transformer backbones — is identified as a priority open question in §V.I rather than a verified finding within the present study.

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

---

## F. Fault-Mode Architectures (H3)

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

By reading only early-cycle data, M3 avoids any dependence on late-cycle observations that are unavailable at real deployment time and is immune to the test-time cluster-distribution collapse that makes M1 unreliable on FD004. Mean maximum gate weight (mean max(w₀, w₁) over training engines) is recorded as a descriptive statistic for routing decisiveness; see §IV.C.2.

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

---

## G. Loss Functions (H4)

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

---

## H. Training Configuration

Training configuration differed across hypotheses as follows:

- **H1 (Linear Regression):** Deterministic OLS — no random seeds, no early stopping, no epochs.
- **H2 (Normalization):** Compact LSTM; validation split is deterministic (last 20% of engines by unit ID); 5 seeds; 100 max epochs; patience = 15; test predictions clipped to [0, 125].
- **H3 (Fault-mode architecture):** Full-capacity LSTM; validation split is deterministic (engine-level shuffle, seed = 42, fixed across all model seeds); minimum 30-epoch warm-up before early stopping; 5 seeds; test predictions clipped to [0, 125].
- **H4 (Loss functions):** Full-capacity LSTM; validation split is deterministic (last 20% of engines by unit ID); 5 seeds; test predictions clipped to [0, 125]; no minimum warm-up.

General settings applied within each hypothesis unless overridden above:

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1×10⁻³ |
| Weight decay | 1×10⁻⁴ |
| Batch size | 256 |
| Max epochs | 100 |
| Early-stop patience | 15 (val loss) |
| Sliding window | 30 cycles |
| Random seeds | {0, 1, 2, 3, 4} |

The feature set and validation-split method differ across hypotheses, as shown in Table III. These differences were established independently during implementation and mean that the hypotheses are not directly cross-comparable in absolute RMSE.

**Table III. Feature set, validation split, and backbone by hypothesis.**

| Hypothesis | Feature set | Val split method | Backbone |
|-----------|------------|-----------------|---------|
| H1 | Sensors + op cols (OLS) | N/A (deterministic OLS) | Linear regression (OLS) |
| H2 | Sensors + op cols (incl. op1/op2/op3) | Last 20% by unit ID (deterministic) | Compact LSTM (LSTM₂ hidden=32) |
| H3 | Sensors only (no op cols); predictions clipped to [0, 125] at evaluation | Deterministic 20% (engine shuffle, seed = 42); 30-epoch minimum warm-up | Full LSTM (LSTM₂ hidden=64) |
| H4 | Sensors only (no op cols) | Deterministic 20% (last 20% by unit ID) | Full LSTM (LSTM₂ hidden=64) |

For H2 the resulting input dimension F is 17 (FD001: 14 sensors + 3 op), 18 (FD003: 15 sensors + 3 op), or 23 (FD002/FD004 after residualisation: 20 sensors + 3 op). For H3/H4 F is 14 (FD001), 15 (FD003), or 20 (FD002/FD004 after residualisation). **Note:** The feature and split differences between H2 and H3 reflect independent implementation choices made prior to analysis; they mean that the two hypotheses are not directly cross-comparable in absolute RMSE terms. Each hypothesis is interpreted relative to its own baseline condition.

---

## I. Evaluation Metrics

Two metrics were reported across all experiments. RMSE is the primary performance indicator:

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$$

The NASA prognostic score penalises late predictions more severely than early ones:

$$s(d) = \begin{cases} e^{-d/13} - 1 & d < 0 \text{ (early prediction)} \\ e^{d/10} - 1 & d \geq 0 \text{ (late prediction)} \end{cases}, \quad \text{Mean NASA Penalty} = \frac{1}{N}\sum_{i=1}^{N} s(\hat{y}_i - y_i)$$

where d = ŷ − y and N is the number of test engines. This study reports the mean prognostic penalty per test engine, which normalises for the different test-set sizes across sub-datasets (N = 100–259). This formulation differs from the conventional sum-of-penalties reported in much of the CMAPSS literature; absolute values are not directly comparable with published summed scores, though ratios and orderings across conditions are unaffected. Mean NASA Penalty is lower-is-better; a perfect prediction yields zero.

---

## J. Statistical Testing

Statistical comparisons differed in sample unit by hypothesis. For H1 (linear regression, deterministic), comparisons used a two-sided Wilcoxon rank-sum test (`scipy.stats.ranksums`) on per-engine RMSE values (N ≈ 100–259 per dataset; see §III.K). Note: `scipy.stats.ranksums` implements the Wilcoxon rank-sum test; the SciPy function for Mann-Whitney U is `scipy.stats.mannwhitneyu`. The two tests are mathematically equivalent on continuous data but named differently in the library.

For H2 (normalization, LSTM stochastic), comparisons used two-sided Wilcoxon rank-sum tests (`scipy.stats.ranksums`) on per-seed RMSE aggregate metrics (N = 5) to detect differences in either direction. For H4 (loss functions, LSTM stochastic), comparisons used one-sided paired Wilcoxon signed-rank tests (`scipy.stats.wilcoxon`, `alternative='less'`) on per-seed NASA Score aggregate metrics (N = 5), testing the pre-specified directional hypothesis that the custom loss reduces the NASA prognostic penalty relative to MSE; RMSE is reported descriptively for H4. All 96 H4 pairwise NASA Score comparisons (6 loss functions × 4 clipping values × 4 datasets) were jointly BH-FDR-corrected. For H3, two-sided tests are used to capture both improvement and degradation directions. Because M1 trains two branches on per-cluster engine subsets, its validation split drew from cluster-specific pools (seed=42 per cluster), whereas M0's validation split drew from the full engine pool (seed=42). Although the same random seed was used, the non-overlapping engine pools make per-seed outcomes effectively independent; M0-vs-M1 comparisons therefore used an independent two-sided Mann-Whitney U test (`scipy.stats.mannwhitneyu`, N = 5 per group). This pooling difference is a design limitation: M1's cluster-restricted training pools are smaller than M0's single-pool split, which may modestly disadvantage M1; re-running with a shared outer split was not feasible within the current study scope. M0-vs-M2 and M0-vs-M3 comparisons used paired two-sided Wilcoxon signed-rank tests (`scipy.stats.wilcoxon`, N = 5 pairs), exploiting the identical per-seed validation split shared by M0, M2, and M3. All six H3 p-values were jointly BH-FDR-corrected.

All tests used α = 0.05 before correction. When multiple treatment conditions are compared simultaneously within one hypothesis, raw p-values were corrected using the Benjamini-Hochberg (BH) procedure at α_FDR = 0.05. A result was considered statistically meaningful when both p_BH < 0.05 **and** Cohen's d ≥ 0.3 (small effect threshold). Conditions satisfying the p-value criterion but yielding |d| < 0.1 are reported as "statistically significant but practically negligible" to distinguish statistical from practical significance.

---

## K. H1 Baseline Model (Linear Regression)

The RUL clipping study (H1) used ordinary least squares linear regression (`sklearn.linear_model.LinearRegression`, no regularisation) as its predictive model. Linear regression was chosen to isolate the effect of label engineering from non-linear model capacity: the clipping threshold's effect on RUL label distribution is architecture-independent, and a deterministic closed-form baseline eliminates random-initialisation variance. Each of the 20 experimental configurations (5 clip values × 4 datasets) was run exactly once; the model has no random state and produces identical results on identical data. The statistical comparison for H1 used the Wilcoxon rank-sum test (`scipy.stats.ranksums`, unpaired, two-sided) on per-engine RMSE values (N ≈ 100–259 per dataset depending on sub-dataset). Note that this is technically an unpaired test; a paired Wilcoxon signed-rank test would be marginally more statistically efficient since the same test engines are evaluated under both clip conditions, and this limitation should be borne in mind when interpreting H1 significance levels.
