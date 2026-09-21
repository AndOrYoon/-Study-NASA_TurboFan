# Methodology
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression

**작성일:** 2026-09-21  
**버전:** Draft v0.1  
**관련 파일:** `Research_Plan.md` §5–8, `CLAUDE.md` (backbone 아키텍처·데이터 규격)

---

> **작성 방침:** 단순 문장, 능동태. §4.1 지표 정의 → §4.2 실험 설정 → §4.3 가설 → §4.4 통계 계획 순.

---

## 4. Methodology

### 4.1 Operational Definition and Metrics

#### 4.1.1 MPC Definition

We adopt the following operational definition of MPC throughout this study.

> **Mean-Prediction Collapse (MPC)** refers to a degenerate regression behavior in which model predictions exhibit substantially reduced dispersion relative to the target distribution and concentrate around the population or conditional mean, thereby suppressing input-dependent variations and systematically underrepresenting extreme target values.

This definition was introduced in Section 3.1. We repeat it here to anchor the four metrics that follow.

#### 4.1.2 Metrics

We use four metrics to quantify MPC severity. The first three are computable from validation outputs alone, without test-set access. The fourth requires a small input perturbation experiment at training completion.

**Prediction Dispersion Ratio (PDR)**

$$\mathrm{PDR}_V = \frac{s(\hat{Y}_V)}{s(Y_V) + \varepsilon}$$

where $s(\cdot)$ denotes sample standard deviation, $\hat{Y}_V$ is the vector of model predictions on the validation set $V$, $Y_V$ is the vector of true RUL values, and $\varepsilon = 10^{-8}$ prevents division by zero. PDR normalizes prediction spread by target spread. A normally trained model yields PDR ≈ 0.7–1.0. An MPC model yields PDR ≈ 0. In the collapsed FD003 baseline that motivated this study, PDR < 0.0001.

PDR operationalizes the definition component *"substantially reduced dispersion relative to the target distribution."*

**Constant-Baseline Ratio (CBR)**

$$\mathrm{CBR}_V = \frac{\mathrm{RMSE}(Y_V,\, \hat{Y}_V)}{\mathrm{RMSE}(Y_V,\, c_{\mathrm{train}}) + \varepsilon}, \qquad c_{\mathrm{train}} = \bar{Y}_{\mathrm{train}}$$

$c_{\mathrm{train}}$ is the training-set mean RUL, computed before training and held fixed. CBR measures the model's error relative to a trivial constant predictor. CBR = 1.0 means the model is no better than the constant. CBR > 1 means the model is worse. CBR < 0.5 indicates a well-performing model.

CBR is mathematically related to the MSE Skill Score defined by Murphy [P27]: CBR² = 1 − SS, where SS = 1 − MSE/MSE_ref. We use CBR rather than SS because CBR is expressed as a ratio and is directly interpretable as "how many times worse than the constant predictor."

CBR operationalizes *"concentrate around the population or conditional mean"* — a model predicting $c_{\mathrm{train}}$ exactly would yield CBR = 1.0.

**Coefficient of Determination (R²)**

$$R^2_V = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y}_V)^2}$$

R² ≤ 0 means the model explains none of the variance in the validation targets. The trivial constant predictor $\bar{y}_V$ yields R² = 0 by definition. An MPC model yields R² ≤ 0 because it predicts $c_{\mathrm{train}} \neq \bar{y}_V$ in general.

R² operationalizes *"suppressing input-dependent variations"* — a model with input-dependent outputs should explain a positive fraction of target variance.

**Input Sensitivity Score (ISS)**

$$\mathrm{ISS}_V = \frac{1}{|V|} \sum_{i \in V} \frac{\|\hat{f}(x_i + \delta_i) - \hat{f}(x_i)\|}{\|\delta_i\| + \varepsilon}$$

where $\delta_i$ is a small sensor perturbation drawn from a pre-defined distribution (standard deviation = 5% of each sensor's training range, applied independently per sensor). ISS measures the average change in model output per unit change in input. ISS → 0 means the model output is insensitive to input — the model is functionally constant even if its nominal output is not exactly equal to the training mean.

ISS extends the occlusion sensitivity framework of Zeiler and Fergus [P28] from spatial image patches to multivariate time-series inputs. The naming and time-series formulation are original to this study.

ISS operationalizes *"suppressing input-dependent variations"* at the output level — complementing R² which operates at the variance level.

#### 4.1.3 MPC Adjudication Criteria

MPC is adjudicated using a composite criterion, not a single threshold. A model is classified as MPC-positive if it satisfies all three of the following conditions:

1. **Low dispersion:** PDR < θ₁, where θ₁ is fixed on the calibration set (Phase 1A) before any confirmatory analysis.
2. **Non-informative output:** R² ≤ 0 or ISS < θ₂.
3. **Constant-predictor equivalence:** CBR is within a pre-specified equivalence interval [1 − δ, 1 + δ] centered at 1.

Threshold values θ₁, θ₂, and δ are derived from the Phase 1A calibration set and held fixed thereafter. They are not adjusted based on outcomes in confirmatory conditions.

To avoid circular reasoning — a detector that defines itself as ground truth — MPC labels for calibration runs are assigned by blind adjudication. Two assessors independently review prediction-versus-target scatter plots, epoch-level PDR and R² trajectories, input perturbation responses, and paired comparisons against the constant predictor. They do not have access to test-set results or final PDR/CBR scores during adjudication. Inter-rater agreement is measured with Cohen's κ [P25]. Disagreements are resolved by a pre-specified protocol. This procedure is adapted from blinded endpoint assessment in clinical trials [P26].

MPC severity is also reported as a continuous measure (PDR, R², CBR values) alongside the binary classification, to avoid artificial thresholding effects.

---

### 4.2 Experimental Setup

#### 4.2.1 Datasets

We use all four C-MAPSS subsets [P29]. Table 2 summarizes their key properties.

**Table 2. C-MAPSS dataset characteristics.**

| Subset | Operating conditions | Fault modes | Train units | Test units | Features after dropping constants |
|--------|:---:|:---:|:---:|:---:|:---:|
| FD001 | 1 | 1 | 100 | 100 | 14 |
| FD002 | 6 | 1 | 260 | 259 | 20 |
| FD003 | 1 | 2 | 100 | 100 | 15 |
| FD004 | 6 | 2 | 249 | 248 | 20 |

C-MAPSS is a thermodynamic simulator. Each row corresponds to one operating cycle. Each engine degrades from a healthy initial state until a failure threshold is reached. The test set provides the last observation window per engine; ground-truth RUL values are provided separately.

The primary dataset for mechanism studies (Phases 0–2) and the external audit (Phase 5) is **FD003**. This subset was the source of the motivating observation. It has one operating condition and two fault modes, making it the most homogeneous subset. FD001–FD004 are all used in Phase 3 generalization experiments.

**Preprocessing.** We apply the following steps in order, fitting all parameters on training data only:

1. *Constant sensor removal.* Sensors with zero variance in the training set are dropped: FD001 drops s1, s5, s6, s10, s16, s18, s19; FD003 drops s1, s5, s10, s16, s18, s19; FD002 and FD004 drop s16.
2. *Operating condition residualization (FD002, FD004 only).* We apply K-means clustering (k = 6) on the three operating condition columns, then subtract cluster means from sensor readings. This removes operating-condition offsets before normalization.
3. *Min-Max normalization.* Sensor values are scaled to [0, 1] using training-set min and max.
4. *RUL labeling.* Piecewise-linear RUL with clip = 125 cycles. This threshold is established as the standard for C-MAPSS [P16].
5. *Sliding window.* Window size = 30 cycles. Test sequences shorter than 30 cycles are zero-padded from the front.

**Validation split.** Unless a condition explicitly specifies a fixed split (condition A1_fixed), we use a per-seed random split: 80% of training engines form the train set; the remaining 20% form the validation set. The split is stratified by engine lifetime to maintain the RUL distribution. The random seed for the split differs from the model initialization seed, ensuring that split variation and parameter initialization variation are independent sources of randomness.

#### 4.2.2 Model Architecture

All experiments use the same stacked LSTM backbone:

```
Input: [batch, 30, n_features]
  → LSTM₁(hidden=64, return_sequences=True)
  → Dropout(0.2)
  → LSTM₂(hidden=64, return_sequences=False)   # last timestep only
  → Dropout(0.2)
  → Linear(64 → 32) → ReLU
  → Linear(32 → 1)
Output: scalar RUL prediction
```

LSTM₁ returns the full sequence (all 30 timesteps) to LSTM₂. Passing only the final hidden state of LSTM₁ to LSTM₂ is a known implementation error that degrades performance; we verified this in Phase 0.

**Training configuration:** Adam optimizer, learning rate = 1×10⁻³, weight decay = 1×10⁻⁴, batch size = 256, loss = MSE. Early stopping monitors validation MSE.

Phase 1B varies patience, learning rate, output bias initialization, and loss function systematically. Phase 3 additionally evaluates MLP and 1D-CNN architectures with comparable parameter counts.

#### 4.2.3 Experimental Phases

The study proceeds in six phases. Table 3 summarizes their purpose, scope, and relationship to the research questions.

**Table 3. Experimental phase overview.**

| Phase | Name | Primary purpose | Datasets | RQ |
|-------|------|----------------|----------|-----|
| 0 | Reproduction and audit | Confirm that MPC is not caused by a code error; establish simple baselines | FD003 | — |
| 1A | Mechanism screen | Quantify main effects and interactions of validation split × early stopping × RUL labeling on MPC occurrence | FD003 | RQ1 |
| 1B | Optimization follow-up | Isolate contributions of patience, learning rate, bias initialization, and loss function | FD003 | RQ1 |
| 2 | Training dynamics | Collect epoch-level PDR, R², CBR, gradient norms, and checkpoint trajectories during Phase 1 training | FD003 | RQ1 |
| 3 | Generalization | Replicate mechanism findings across FD001–FD004 and three architecture families (LSTM, MLP, 1D-CNN) | FD001–FD004 | RQ2 |
| 3B | LSTM vs GRU mechanism | Test whether LSTM forget gate → 0 is the structural cause; compare LSTM variants and GRU under identical conditions | FD003 | RQ1, RQ2 |
| 4A | Prevention taxonomy | Compare all prevention interventions on MPC reduction rate, RMSE non-inferiority, and implementation cost | FD001–FD004 | RQ4 |
| 4B | Retrospective audit tool | Calibrate PDR/R²/CBR/ISS thresholds on Phase 1A data; validate on held-out Phase 3 conditions | Phase 1A → Phase 3 | RQ3 |
| 5 | External protocol audit | Replicate five published training protocols; measure real-world MPC prevalence | FD003 | All |

**Phase 0** serves as a pre-condition check. We reproduce the original five-seed collapsed run (RMSE = 43.23) and the corrected run (RMSE = 12.97) from the same codebase. We also evaluate constant predictors (training mean, validation mean), a linear regression baseline, and a persistence baseline. Phase 1 proceeds only if the collapsed run is reproduced and confirmed not attributable to a code error.

**Phases 1A and 1B** use a factorial design. Phase 1A varies three binary factors: validation split (fixed vs. per-seed), early stopping mode (from epoch 0 vs. after MIN_EPOCHS warmup), and RUL labeling (unclipped vs. clip = 125). Phase 1B examines patience (5, 10, 20, 30), learning rate (0.1×, 1×, 10× baseline), output bias initialization (zero, training mean, random calibrated), and loss function (MSE, MAE, auxiliary branch loss). Phase 1B uses sequential design: factors identified as large in an initial screen are held fixed while remaining factors are explored.

**Phase 2** runs in parallel with Phases 1A and 1B. Every training run records: train and validation RMSE and loss per epoch; validation PDR, R², CBR per epoch; output-layer gradient norm and weight norm per epoch; early stopping counter state.

**Phase 3B** extends Phase 3 to investigate the structural cause of LSTM-specific MPC. We compare three LSTM variants — standard LSTM (V3_base), LSTM with forget bias initialized to +1 (V1_fb1), and LSTM with the forget gate fixed to 1 (V2_fg1, equivalent to the Constant Error Carousel) — against GRU under the MPC-inducing condition (A1_fixed + B2_from0 + C2_clip125). Gate activation values (forget gate mean per epoch) are logged for all runs.

#### 4.2.4 Phase 5: External Protocol Audit

Phase 5 replicates training protocols from published papers on C-MAPSS FD003 and measures MPC occurrence under each protocol.

**Paper selection criteria.** We include a paper if it satisfies all of the following:
- Published in EAAI or Expert Systems with Applications (the two highest-volume CMAPSS venues identified in our literature survey).
- Uses C-MAPSS FD003 as a benchmark dataset.
- Uses a stacked or single LSTM as the primary model or a key baseline.
- Provides sufficient protocol detail to extract: validation split method, patience, max training epochs, and RUL clipping.

Five papers meeting these criteria were identified. Table 4 summarizes their protocols (see Section 7 for full results).

**Table 4. Protocols replicated in Phase 5.**

| Audit ID | Citation | Val split | Patience | Max epochs | RUL clip |
|----------|----------|:---------:|:--------:|:----------:|:--------:|
| A1 | Meng et al. (2023) [P31] | Fixed, seed=42 | 15 | 200 | 125 |
| A2 | Qin et al. (2024) [P32] | Fixed, seed=42 | 20 | 200 | 125 |
| A3 | Zheng et al. (2017) [P33] | Fixed, seed=42 | 10 | 200 | 125 |
| A4 | Representative pre-2020 pattern | Fixed, seed=42 | 15 | 200 | None |
| A5 | Elsherif et al. (2025) [P5] | Fixed, seed=42 | N/A | 25 (fixed) | 125 |

**Replication procedure.** For each protocol, we train 10 independent models using random seeds 0–9. Model initialization seeds and protocol seeds are applied exactly as described in the source paper where specified; otherwise, seeds 0–9 are used systematically. The backbone architecture is held identical across all five protocols (Section 4.2.2). Protocol parameters are the only variable.

**Fidelity assessment.** We verify replication fidelity by checking that the median RMSE across non-collapsed seeds falls within 15% of the value reported in the source paper. If a paper does not report seed-level results, we compare against the paper's mean RMSE.

---

### 4.3 Research Hypotheses

The four primary hypotheses correspond to the four research questions stated in Section 5 of the Introduction.

**H1 (Mechanism):**  
The combination of a fixed validation split (A1_fixed) and early stopping from epoch 0 (B2_from0) increases MPC occurrence probability significantly more than either condition alone. This interaction effect is specific to LSTM architectures because the LSTM forget gate approaches zero under these conditions, blocking gradient flow through the cell state. GRU, which has no separate cell state, does not exhibit MPC under the same protocol conditions.

*Operationalization:* We test H1 using Phase 1A factorial data (mixed-effects logistic regression) and Phase 3B gate-activation data (forget gate mean at stopping epoch, collapsed vs. normal runs).

**H2 (Vulnerability):**  
MPC susceptibility across C-MAPSS subsets is better explained by label distribution characteristics — specifically, RUL label variance, clipping ratio, and inter-unit variability — than by the number of operating conditions or fault modes alone.

*Operationalization:* We test H2 using Phase 3 data. We compute Spearman correlations between dataset-level statistics (label variance, clipping ratio, inter-unit RMSE variability) and observed MPC rates under the A1+B2 protocol.

**H3 (Detection):**  
The composite indicator (PDR + R² + CBR + ISS) detects MPC from validation-only outputs with higher balanced accuracy than any single metric alone. Calibration-set thresholds generalize to held-out dataset × architecture combinations.

*Operationalization:* We test H3 using Phase 4B data. We compute AUROC and balanced accuracy for each metric individually and for the composite rule. Held-out conditions are Phase 3 dataset × architecture combinations not used in calibration.

**H4 (Prevention):**  
Per-seed random validation split, early-stopping warmup (MIN_EPOCHS ≥ 30), and GRU substitution each reduce MPC occurrence to zero or near-zero without statistically significant RMSE deterioration in non-collapsed runs. No single protocol change is universally sufficient; multiple effective interventions exist.

*Operationalization:* We test H4 using Phase 4A data. MPC reduction is assessed with a chi-squared test or Fisher's exact test. RMSE non-inferiority is assessed with a one-sided equivalence test using a 10% margin relative to the best-performing non-collapsed condition.

---

### 4.4 Statistical Analysis Plan

#### 4.4.1 Primary and Secondary Outcomes

The primary outcome is MPC occurrence (binary, adjudicated per Section 4.1.3).

Secondary outcomes are:
- Validation and test RMSE (lower is better)
- NASA prognostic score (lower is better; penalizes late predictions more than early)
- PDR, R², CBR, ISS (continuous MPC severity)
- Training stopping epoch
- Benchmark distortion index Δ_inflation (defined below)

**Benchmark distortion index.** We quantify the inflation of apparent performance gains caused by a collapsed baseline:

$$\Delta_{\mathrm{inflation}} = \left(\frac{E_{\mathrm{collapsed}} - E_{\mathrm{new}}}{E_{\mathrm{collapsed}}}\right) - \left(\frac{E_{\mathrm{valid}} - E_{\mathrm{new}}}{E_{\mathrm{valid}}}\right)$$

where $E_{\mathrm{collapsed}}$ is the RMSE of the collapsed baseline, $E_{\mathrm{valid}}$ is the RMSE of the same model trained under a corrected protocol, and $E_{\mathrm{new}}$ is the RMSE of a subsequent model. We also report absolute error differences alongside Δ_inflation to guard against denominator instability.

#### 4.4.2 Models for Binary Outcomes

MPC occurrence rates across protocol conditions are modeled with mixed-effects logistic regression. Fixed effects include the protocol factors under investigation (validation split type, early stopping mode, patience, learning rate, etc.). Random effects account for the repeated-measures structure (multiple seeds per condition, multiple datasets per architecture). We report odds ratios with 95% confidence intervals.

#### 4.4.3 Models for Continuous Outcomes

Continuous outcomes (RMSE, PDR, stopping epoch) are modeled with mixed-effects linear regression where normality is plausible, and with robust linear mixed models or beta regression where it is not. We use the same random-effect structure as in 4.4.2.

#### 4.4.4 Multiple Comparison Correction

All significance tests that involve more than one pre-specified contrast are corrected with the Benjamini–Hochberg false discovery rate (BH-FDR) procedure at α = 0.05. The set of contrasts is pre-specified per phase before data collection. We report both uncorrected p-values and BH-adjusted p-values.

#### 4.4.5 Non-Inferiority Testing

For each prevention intervention, we test RMSE non-inferiority relative to the best non-collapsed reference condition. The non-inferiority margin is set to 10% of the reference RMSE (i.e., the intervention is acceptable if its mean RMSE does not exceed 1.10 × RMSE_reference). We use a one-sided t-test or Wilcoxon test at α = 0.05.

#### 4.4.6 Detector Validation

For Phase 4B, detector performance (sensitivity, specificity, balanced accuracy, AUROC) is estimated with 10-fold cross-validation on the calibration set, using dataset or architecture as the fold unit to prevent leakage. Performance on the held-out confirmatory conditions is reported with 95% bootstrap confidence intervals (2,000 resamples).

#### 4.4.7 Seed Counts and Statistical Power

Discovery phases use 10 seeds per condition. This provides adequate power to detect MPC rates ≥ 50% (power > 0.90, one-sample proportion test, H₀: rate ≤ 10%, α = 0.05). Confirmatory phase seed counts are determined by a prospective power analysis using Phase 1A pilot estimates of MPC rate and its variance.

#### 4.4.8 Reporting Conventions

- All results are reported as mean ± standard deviation over seeds, unless otherwise noted.
- Effect sizes accompany all significance tests: Cohen's d for continuous outcomes, risk difference and relative risk for binary outcomes.
- We report negative and null results in full. Non-significant findings are stated explicitly with their effect size confidence intervals.
- We do not engage in post-hoc threshold adjustment or selective reporting of seeds.

---

## Reference Placeholders

*(번호 체계는 Literature_Review.md와 동일)*

- [P6] Vabalas et al. (2019) — validation split bias
- [P14] Prechelt (1998) — early stopping
- [P25] Cohen (1960) — Cohen's κ
- [P26] Schulz et al. / CONSORT (2010) — blind adjudication
- [P27] Murphy (1988) — MSE Skill Score (CBR 기저 개념)
- [P28] Zeiler & Fergus (2014) — input sensitivity (ISS 기저 개념)
- [C-MAPSS] Saxena & Goebel (2008) — C-MAPSS dataset
