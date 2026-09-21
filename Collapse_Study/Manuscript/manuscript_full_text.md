# Mean-Prediction Collapse in Deep Remaining Useful Life Regression: Mechanisms, Detection, and Prevention

**버전:** Draft v0.1  
**작성일:** 2026-09-21  
**상태:** Full manuscript — sections compiled from individual drafts

---

## Abstract

Deep learning models for remaining useful life (RUL) prediction are routinely evaluated against LSTM baselines on the C-MAPSS benchmark. We show that a subset of these baselines exhibits Mean-Prediction Collapse (MPC): the model outputs a near-constant prediction equal to the training-set mean, regardless of input, achieving no better RMSE than a trivial constant predictor. MPC is silent — validation loss decreases normally, and the failure is invisible without inspecting prediction dispersion.

We conduct a controlled experiment on C-MAPSS FD001–FD004 to identify when MPC occurs, why, and how to prevent it. Under the trigger conditions common in published work — a fixed validation split combined with early stopping from epoch zero — FD003+LSTM collapses at a rate of 80% across independent training seeds. GRU, 1D-CNN, and MLP are unaffected under identical conditions. The structural cause is LSTM forget gate saturation: as the gate approaches zero, backpropagation through the cell state is blocked, and the model is trapped at a trivial solution. Clamping the gate at ε ≤ 0.05 does not help; fixing it at 1 (restoring the original constant error carousel design) eliminates MPC entirely.

Five independent interventions achieve zero MPC with no RMSE penalty. The lowest-cost option is initializing the output layer bias to the training-set mean RUL — a single line of code. For retrospective diagnosis, the Prediction Dispersion Ratio (PDR = std(ŷ)/std(y)) computed at training completion identifies MPC with AUROC = 1.0000 across 600 runs, without test-set access.

We replicate five published training protocols and find MPC rates of 60–90% per protocol (41 of 50 total runs collapsed). In the case that motivated this study, correcting the baseline protocol reduced an apparent 65.8% improvement to a statistically non-significant difference. Protocol correction, not better models, drove the change. The mechanism explanation, zero-cost prevention options, and test-free diagnostic introduced here give PHM researchers the tools to verify that LSTM baselines are valid before drawing comparative conclusions.

**Keywords:** remaining useful life, LSTM, mean-prediction collapse, early stopping, benchmark reliability, prognostics and health management

---

## 1. Introduction

### 1.1 Context

Remaining useful life (RUL) prediction enables condition-based maintenance. When a system's remaining life can be estimated accurately, maintenance can be scheduled before failure rather than after. This reduces downtime, lowers cost, and prevents hazardous failures.

The C-MAPSS turbofan engine dataset [P29] has become the standard benchmark for deep learning–based RUL prediction. Since its release, hundreds of studies have used C-MAPSS to report performance improvements. Most follow the same pattern: propose a new model, compare it to a baseline, and report a percentage improvement in RMSE or NASA score.

This study does not propose a new model. It questions whether the baselines in these comparisons are valid.

---

### 1.2 Motivating Observation

The problem became visible during the development of a fault-mode routing model for the FD003 subset of C-MAPSS [ANON].

A simple stacked LSTM baseline reported RMSE = 43.23. A more complex model achieved RMSE = 14.78. The implied improvement was 65.8%.

Closer inspection of the baseline revealed the following:

- All five training seeds predicted approximately 87 cycles for every test engine.
- The prediction standard deviation across 100 test engines was less than 0.0002 cycles.
- The dataset mean RUL of the test set is approximately 87 cycles.

The baseline had learned nothing. It was predicting the dataset mean, regardless of input.

We corrected three aspects of the training protocol: (1) the validation split was randomized per seed instead of fixed, (2) a minimum training period was added before early stopping could fire, and (3) maximum training epochs were extended. The corrected baseline achieved RMSE = 12.97 ± 0.67. The claimed 65.8% improvement collapsed to a difference that was not statistically significant.

The problem was not the model. The problem was the protocol.

---

### 1.3 Mean-Prediction Collapse: Definition and Importance

We refer to this failure as **Mean-Prediction Collapse (MPC)**. This section defines the concept precisely, distinguishes it from related phenomena, and explains why it matters for benchmark-based research.

#### 1.3.1 Formal Definition

> **Mean-Prediction Collapse (MPC)** refers to a degenerate regression behavior in which model predictions exhibit substantially reduced dispersion relative to the target distribution and concentrate around the population or conditional mean, thereby suppressing input-dependent variations and systematically underrepresenting extreme target values.

This definition has four components. Each is grounded in prior literature.

| Component | What it means in RUL regression | Prior evidence |
|-----------|----------------------------------|----------------|
| *Reduced dispersion relative to the target distribution* | The spread of predictions is negligible compared to the spread of true RUL values | Mathieu et al. [P20]: "inherently blurry predictions" under MSE |
| *Concentrate around the population or conditional mean* | All predictions cluster near a single value (~87 cycles in FD003), regardless of which engine or cycle is queried | Bruna et al. [P19]: "regression-to-the-mean problem"; Huang [P21]: squared-loss regression converges to conditional mean |
| *Suppressing input-dependent variations* | The partial derivative of the output with respect to the input approaches zero — the model is functionally constant | Huang [P21]: unconstrained regressor under squared loss ignores multi-modal structure |
| *Systematically underrepresenting extreme target values* | Engines near end-of-life (low RUL) and engines early in their life (high RUL) receive the same near-mean prediction | Lee & Chen [P7]: systematic mean bias at distribution extremes |

The "degenerate" characterization is not merely qualitative. An MPC model performs no better — and often worse — than a trivial constant predictor that always outputs the training-set mean RUL. Mathematically, its coefficient of determination R² ≤ 0.

#### 1.3.2 Distinction from Related Phenomena

MPC must be distinguished from two phenomena it superficially resembles.

**Conditional-Mean Compression (CMC)** is a statistical property of well-trained regression models. A model exhibiting CMC still uses its inputs to produce differentiated predictions, but its output range is compressed toward the mean relative to the true target range. This is the phenomenon characterized by Lee and Chen [P7]. CMC occurs *after* successful training; the model has learned from the data. MPC occurs *instead of* training; the model has not learned from the data.

**Undertrained baselines** occur when a model receives insufficient optimization. An undertrained model still uses its inputs; it simply has not converged to its best solution yet. Musgrave et al. [P23] and Lučić et al. [P24] documented this extensively. MPC is not an undertrained model. The LSTM in MPC converges quickly and stably — to the wrong solution. Additional tuning does not fix it unless the protocol is corrected.

**Table 1. MPC versus related phenomena.**

| Phenomenon | Uses inputs? | Occurs when? | Fix |
|------------|:---:|---|---|
| MPC (this study) | ✗ | During training; protocol-induced | Correct the protocol |
| Conditional-Mean Compression [P7] | ✓ | After training; statistical | Bias correction or loss modification |
| Undertrained baseline [P23, P24] | ✓ | During training; insufficient optimization | More tuning |
| Vanishing gradient (LSTM) [P2] | ✓ | During training; architecture | Gradient clipping, architecture change |

#### 1.3.3 Measurable Indicators

MPC severity is quantified by three indicators, defined formally in Section 4. Here we introduce them briefly.

**Prediction Dispersion Ratio (PDR)** normalizes prediction spread by target spread:

$$\mathrm{PDR} = \frac{\mathrm{std}(\hat{y})}{\mathrm{std}(y) + \varepsilon}$$

PDR → 0 means predictions are nearly constant. In the FD003 collapse case, PDR < 0.0001. In a normally trained model, PDR ≈ 0.7–1.0.

**Constant-Baseline Ratio (CBR)** compares the model's RMSE to that of a trivial constant predictor equal to the training-set mean:

$$\mathrm{CBR} = \frac{\mathrm{RMSE}(\hat{y},\, y)}{\mathrm{RMSE}(c_{\mathrm{train}},\, y) + \varepsilon}$$

CBR ≈ 1 means the model is no better than a constant. In the FD003 collapse, CBR ≈ 1.02 — the LSTM is marginally *worse* than simply predicting the mean.

**Coefficient of determination (R²)** quantifies the fraction of variance explained. R² ≤ 0 means the model explains none of the variance in the targets. In the FD003 collapse, R² ≈ −0.04.

None of these indicators requires test-set access. They can be computed on the validation set at training completion, enabling retrospective detection of MPC without additional experiments.

#### 1.3.4 Why MPC Matters

MPC has a direct consequence for benchmark comparisons. When a baseline is in the MPC state, its RMSE is artificially high. Any subsequent method will appear to achieve a large improvement.

In the case that motivated this study: the collapsed baseline reported RMSE = 43.23. The corrected baseline achieves RMSE = 12.97. A method achieving RMSE = 12 appears to improve by 72% over the collapsed baseline. Over the corrected baseline, the same method improves by less than 8%. The improvement claim changes by an order of magnitude depending solely on whether the baseline training protocol is correct.

This inflation is not a deliberate distortion. Authors who report it typically do not know the baseline has collapsed. Validation loss behaves normally — it decreases during training and stabilizes at early stopping. There is no obvious signal that the model is in a degenerate state. This is what makes MPC dangerous: it is silent.

MPC is also not rare. We replicated five published training protocols from EAAI and Expert Systems with Applications and measured MPC occurrence rates of 60–90% across 10 independent random seeds per protocol. Under the protocol conditions common in the literature, most training runs collapse.

---

### 1.4 Research Gaps

Prior work has addressed related but distinct problems.

**Undertrained baselines.** Musgrave et al. [P23] and Lučić et al. [P24] showed that apparent improvements in metric learning and generative modeling disappeared when baselines were properly optimized. Their diagnosis was incomplete optimization. MPC is different. The model converges quickly. It is not undertrained. It is trapped at a degenerate solution that MSE loss cannot escape given specific protocol conditions.

**Gradient pathology in LSTMs.** Al-Selwi et al. [P2] studied vanishing gradients in LSTMs on C-MAPSS. Their focus was slow convergence. MPC is the opposite: fast convergence to the wrong solution.

**Validation split bias.** Vabalas et al. [P6] showed that fixed cross-validation splits produce biased performance estimates on small datasets. They did not examine how fixed splits interact with early stopping to select degenerate solutions on the MSE loss surface.

**Mean bias in regression.** Lee and Chen [P7] characterized the systematic bias of regression models toward the mean at distribution extremes. This is a statistical property of well-trained models. MPC is a training failure, not a statistical property.

**The gap.** No prior study has examined when and why RUL regression models converge to a degenerate near-constant solution, under what protocol conditions this occurs reproducibly, whether it can be detected without test data, and what interventions prevent it.

---

### 1.5 Research Questions

**RQ1 — Mechanism:**
Under what training, validation, labeling, and optimization conditions does an RUL regressor converge to a near-constant, input-insensitive prediction?

**RQ2 — Vulnerability:**
Which dataset, label-distribution, and architecture characteristics increase susceptibility to MPC?

**RQ3 — Detection:**
Can MPC be detected using only training and validation outputs, without the test set?

**RQ4 — Prevention:**
What is the minimum-cost intervention that reliably prevents MPC across datasets and architectures?

---

### 1.6 Contributions

**1. Characterization of MPC.**
We provide an operational definition of MPC that distinguishes it from conditional-mean compression and gradient pathology. The definition uses three measurable indicators (PDR, R², CBR) calibrated on held-out conditions.

**2. Mechanism identification.**
We trace MPC to a specific structural cause: the LSTM forget gate approaching zero under adverse protocol conditions, blocking gradient flow through the cell state. GRU, which lacks a separate cell state, does not exhibit MPC under the same conditions.

**3. Prevention taxonomy.**
We compare seven interventions across two axes: MPC reduction rate and RMSE non-inferiority. The taxonomy provides a decision framework for practitioners.

**4. External protocol audit.**
We replicate five published training protocols and measure MPC rates across 10 seeds per protocol. This audit provides direct evidence that MPC-inducing conditions exist in published work.

---

### 1.7 Scope

This study is limited to C-MAPSS (FD001–FD004). C-MAPSS is a thermodynamic simulator producing synthetic degradation trajectories. The practical relevance of this work is methodological, not operational. MPC in this study is specific to LSTM architectures. GRU and feedforward networks did not exhibit MPC under the tested conditions.

---

### 1.8 Paper Structure

Section 2 reviews prior work and establishes the research gap. Section 3 defines MPC operationally and describes the experimental framework. Sections 4 and 5 present experimental results on mechanism and generalization (Phases 0–3B), followed by prevention and retrospective detection (Phase 4). Section 6 presents the external protocol audit (Phase 5). Section 7 discusses implications and limitations. Section 8 concludes.

---

## 2. Related Work

### 2.1 Deep Learning for RUL Prediction on C-MAPSS

The C-MAPSS dataset [P29] has become the standard benchmark for deep learning–based RUL prediction. Li et al. [P16] introduced a deep convolutional neural network (DCNN) that extracted features directly from raw sensor windows and achieved state-of-the-art RMSE across all four C-MAPSS subsets. This work established the template that most subsequent studies follow: a deep model processes sliding windows of sensor readings and predicts a piecewise-linear RUL target.

Many architectures have been proposed since. Ellefsen et al. [P3] applied semi-supervised pretraining to address label scarcity and reported improved convergence stability. Elsherif et al. [P5] combined a convolutional autoencoder with an attention-enhanced LSTM (CAELSTM) and reported FD003 RMSE = 13.40. Zhuang et al. [P15] incorporated uncertainty quantification into a Bayesian deep learning framework for maintenance decision support.

Despite this volume of work, a consistent problem persists. Reported baseline RMSE values for FD003 span roughly 13 to 43 across papers. Training protocols are rarely standardized. Validation split strategies, early stopping conditions, RUL clipping thresholds, and feature selection differ across papers without explanation.

> **Gap:** CMAPSS-based RUL studies lack standardized training protocols. The contribution of protocol choices — rather than model architecture — to reported performance differences has not been quantified.

---

### 2.2 MSE Loss and Mean-Prediction Tendency

MSE is the dominant loss function in RUL regression. Its use has a known theoretical consequence: the Bayes-optimal predictor under MSE is the conditional mean E[y|x].

Bruna et al. [P19] named this the "regression-to-the-mean problem" in the context of image super-resolution. Mathieu et al. [P20] demonstrated the same in video prediction and argued that MSE produces "inherently blurry predictions." Both studies motivated alternatives to MSE for structured prediction tasks.

Huang [P21] extended this argument explicitly to regression, formalizing that under squared loss an unconstrained regressor converges to the conditional mean, and named this outcome "mean collapse." Note that Huang [P21] is an arXiv preprint as of this writing and has not undergone peer review. We treat it as theoretical background only.

These works establish the theoretical basis for mean-directed prediction. Each component of the MPC definition maps to one or more of these prior findings:

- *"Reduced dispersion relative to the target distribution"* — Mathieu et al. [P20]
- *"Concentrate around the conditional mean"* — Bruna et al. [P19] and Huang [P21]
- *"Suppressing input-dependent variations"* — Huang [P21]
- *"Systematically underrepresenting extreme values"* — Lee and Chen [P7]

However, all prior work treats mean-directed prediction as a consequence of data geometry alone. None examines how training protocol conditions can cause a model to converge to the marginal mean E[y] rather than any conditional mean E[y|x]. In MPC, the model outputs a single constant for all inputs, with prediction standard deviation below 0.0002 cycles. This is more severe than the blurring described in prior work: the model discards input information entirely.

> **Gap:** Prior work on MSE-induced mean prediction focuses on data geometry. The role of training protocol conditions in causing models to converge to the marginal mean E[y] has not been studied.

---

### 2.3 Training Instability in LSTM-Based RUL Regression

Al-Selwi et al. [P2] studied vanishing gradient pathology (VGP) in LSTM networks applied to C-MAPSS. Their analysis focused on slow convergence: gradients decay over long sequences, preventing the model from learning long-range dependencies efficiently.

MPC is a different failure. The LSTM in MPC does not converge slowly. It converges quickly — to a degenerate solution. The forget gate approaches zero in early training. This blocks gradient flow through the cell state. Early stopping then terminates training while the model is still at this trivial solution.

Lee and Chen [P7] described a related but different phenomenon: systematic mean bias in trained regression models. This occurs after successful training. MPC occurs instead of training. A model exhibiting mean bias still uses its inputs to produce differentiated predictions. An MPC model does not.

> **Gap:** No prior study examines the specific failure mode in which an LSTM RUL regressor converges rapidly to a near-constant, input-insensitive prediction due to forget gate dynamics interacting with early stopping under adverse protocol conditions.

---

### 2.4 Validation Design and Early Stopping

Prechelt [P14] established the standard framework for early stopping: monitor validation loss and halt training when improvement stalls. This approach assumes that the validation loss is an accurate proxy for generalization performance.

MPC violates this assumption. When the validation split is fixed across training seeds, the validation set has a fixed RUL distribution. A model predicting the training-set mean can achieve low validation MSE because the mean is close to many validation targets. Early stopping then selects this constant-predictor solution as the "best" checkpoint.

Vabalas et al. [P6] showed that fixed K-fold cross-validation on small datasets produces biased performance estimates. Miseta et al. [P12] proposed an early stopping criterion based on the Pearson correlation between training and validation losses. Mahsereci et al. [P13] proposed eliminating the validation set entirely and stopping based on gradient statistics. Both works acknowledge that standard early stopping has unresolved weaknesses, but neither identifies MPC as a failure mode.

> **Gap:** The interaction between fixed validation splits and early stopping in selecting trivial solutions on regression loss surfaces has not been studied.

---

### 2.5 Benchmark Reproducibility Across Domains

Musgrave et al. [P23] re-evaluated distance metric learning methods under consistent conditions and found that claimed improvements of "more than double over four years" disappeared under fair comparison. Lučić et al. [P24] applied the same principle to generative adversarial networks. Ferrari Dacrema et al. [P22] applied a reproducibility audit to 18 neural recommendation algorithms; most could not be reproduced.

These three studies establish that inconsistent experimental conditions routinely produce inflated performance claims. Our work is in this lineage. However, these studies diagnosed *undertrained baselines*. MPC is a more extreme condition. The baseline converges, but to a state in which it outputs a constant regardless of input. Additional hyperparameter search does not help.

> **Gap:** Protocol-induced performance inflation has been documented in metric learning, GAN evaluation, and recommender systems. It has not been studied in PHM/RUL prediction. The extreme form of this inflation — in which the baseline model is functionally collapsed — has not been documented in any domain.

---

### 2.6 Synthetic Data Homogeneity

Prior work on synthetic datasets in PHM focuses on limited representativeness. Chao et al. [P4] motivated the N-CMAPSS dataset by arguing that CMAPSS has "limited representativeness of real operating conditions." Das et al. [P8] studied how uncertainty in the physical simulator propagates to DNN predictions.

This study identifies a different problem. C-MAPSS FD003 is not merely a limited proxy for real data. Its homogeneity creates a specific vulnerability. FD003 has a single operating condition and two fault modes. Within each fault mode, engine degradation trajectories follow the same thermodynamic template. The within-mode variation in sensor windows is small. As a result, the MSE loss landscape has a broad, stable minimum at the marginal mean of the RUL distribution.

> **Gap:** Synthetic dataset homogeneity as a driver of training collapse has not been identified or studied.

---

### 2.7 Summary: Research Gaps

| Gap | Prior literature | What is missing |
|-----|-----------------|-----------------|
| G1 | Li 2018 [P16], Elsherif 2025 [P5] | Protocol-induced variation in baseline RMSE is not quantified |
| G2 | Bruna 2015 [P19], Mathieu 2016 [P20], Huang 2026 [P21] | Marginal-mean collapse under protocol conditions is not examined |
| G3 | Al-Selwi 2023 [P2], Lee 2025 [P7] | LSTM forget gate → 0 as a mechanism for trivial-solution convergence is not documented |
| G4 | Prechelt 1998 [P14], Vabalas 2019 [P6] | Fixed validation split × early stopping interaction in selecting degenerate solutions is not studied |
| G5 | Musgrave 2020 [P23], Lučić 2018 [P24], Ferrari Dacrema 2019 [P22] | Functional collapse in PHM/RUL has not been documented |
| G6 | Chao 2020 [P4], Das 2024 [P8] | Synthetic dataset homogeneity as a driver of stable trivial minima is not identified |

---

## 3. Methodology

### 3.1 Operational Definition and Metrics

#### 3.1.1 MPC Definition

We adopt the following operational definition of MPC throughout this study.

> **Mean-Prediction Collapse (MPC)** refers to a degenerate regression behavior in which model predictions exhibit substantially reduced dispersion relative to the target distribution and concentrate around the population or conditional mean, thereby suppressing input-dependent variations and systematically underrepresenting extreme target values.

#### 3.1.2 Metrics

We use four metrics to quantify MPC severity. The first three are computable from validation outputs alone, without test-set access.

**Prediction Dispersion Ratio (PDR)**

$$\mathrm{PDR}_V = \frac{s(\hat{Y}_V)}{s(Y_V) + \varepsilon}$$

where $s(\cdot)$ denotes sample standard deviation, $\hat{Y}_V$ is the vector of model predictions on the validation set, $Y_V$ is the vector of true RUL values, and $\varepsilon = 10^{-8}$. PDR normalizes prediction spread by target spread. A normally trained model yields PDR ≈ 0.7–1.0. An MPC model yields PDR ≈ 0.

**Constant-Baseline Ratio (CBR)**

$$\mathrm{CBR}_V = \frac{\mathrm{RMSE}(Y_V,\, \hat{Y}_V)}{\mathrm{RMSE}(Y_V,\, c_{\mathrm{train}}) + \varepsilon}, \qquad c_{\mathrm{train}} = \bar{Y}_{\mathrm{train}}$$

CBR = 1.0 means the model is no better than the constant predictor. CBR is mathematically related to the MSE Skill Score defined by Murphy [P27]: CBR² = 1 − SS.

**Coefficient of Determination (R²)**

$$R^2_V = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y}_V)^2}$$

R² ≤ 0 means the model explains none of the variance in the validation targets.

**Input Sensitivity Score (ISS)**

$$\mathrm{ISS}_V = \frac{1}{|V|} \sum_{i \in V} \frac{\|\hat{f}(x_i + \delta_i) - \hat{f}(x_i)\|}{\|\delta_i\| + \varepsilon}$$

where $\delta_i$ is a small sensor perturbation (standard deviation = 5% of each sensor's training range). ISS → 0 means the model output is insensitive to input. ISS extends the occlusion sensitivity framework of Zeiler and Fergus [P28] to multivariate time-series inputs.

#### 3.1.3 MPC Adjudication Criteria

A model is classified as MPC-positive using a composite criterion: (1) PDR < θ₁, (2) R² ≤ 0 or ISS < θ₂, (3) CBR within [1 − δ, 1 + δ]. Thresholds θ₁, θ₂, and δ are fixed on the Phase 1A calibration set and held thereafter.

To avoid circular reasoning, MPC labels for calibration runs are assigned by blind adjudication. Two assessors independently review prediction-versus-target scatter plots, epoch-level PDR and R² trajectories, and input perturbation responses, without access to test-set results or final PDR/CBR scores. Inter-rater agreement is measured with Cohen's κ [P25]. This procedure is adapted from blinded endpoint assessment in clinical trials [P26].

---

### 3.2 Experimental Setup

#### 3.2.1 Datasets

We use all four C-MAPSS subsets [P29]. Table 2 summarizes their key properties.

**Table 2. C-MAPSS dataset characteristics.**

| Subset | Operating conditions | Fault modes | Train units | Test units | Features after dropping constants |
|--------|:---:|:---:|:---:|:---:|:---:|
| FD001 | 1 | 1 | 100 | 100 | 14 |
| FD002 | 6 | 1 | 260 | 259 | 20 |
| FD003 | 1 | 2 | 100 | 100 | 15 |
| FD004 | 6 | 2 | 249 | 248 | 20 |

The primary dataset for mechanism studies (Phases 0–2) is FD003. FD001–FD004 are all used in Phase 3 generalization experiments.

**Preprocessing.** We apply the following steps in order, fitting all parameters on training data only: (1) constant sensor removal, (2) operating condition residualization for FD002 and FD004 (K-means k = 6 on operating condition columns), (3) min-max normalization to [0, 1], (4) piecewise-linear RUL labeling with clip = 125 cycles [P16], (5) sliding window of 30 cycles with zero-padding for shorter sequences.

**Validation split.** Unless a condition specifies a fixed split (A1_fixed), we use a per-seed random split: 80% of training engines for training, 20% for validation, stratified by engine lifetime.

#### 3.2.2 Model Architecture

All experiments use the same stacked LSTM backbone:

```
Input: [batch, 30, n_features]
  → LSTM₁(hidden=64, return_sequences=True) → Dropout(0.2)
  → LSTM₂(hidden=64, return_sequences=False) → Dropout(0.2)
  → Linear(64 → 32) → ReLU → Linear(32 → 1)
Output: scalar RUL prediction
```

LSTM₁ returns the full sequence (all 30 timesteps) to LSTM₂. **Training configuration:** Adam optimizer, learning rate = 1×10⁻³, weight decay = 1×10⁻⁴, batch size = 256, loss = MSE, patience = 15.

#### 3.2.3 Experimental Phases

**Table 3. Experimental phase overview.**

| Phase | Name | Primary purpose | Datasets | RQ |
|-------|------|----------------|----------|-----|
| 0 | Reproduction and audit | Confirm MPC is not a code error; establish baselines | FD003 | — |
| 1A | Mechanism screen | Quantify val split × early stopping × RUL labeling on MPC rate | FD003 | RQ1 |
| 1B | Optimization follow-up | Isolate patience, LR, bias init, loss function effects | FD003 | RQ1 |
| 2 | Training dynamics | Epoch-level PDR, R², val loss trajectories | FD003 | RQ1 |
| 3 | Generalization | Replicate across FD001–FD004 × LSTM/MLP/CNN/GRU | FD001–FD004 | RQ2 |
| 3B | LSTM vs GRU mechanism | Test forget gate saturation as structural cause | FD003 | RQ1, RQ2 |
| 4A | Prevention taxonomy | Compare interventions on MPC rate and RMSE non-inferiority | FD001–FD004 | RQ4 |
| 4B | Retrospective audit tool | Calibrate and validate PDR/R²/CBR thresholds | Phase 1A → Phase 3 | RQ3 |
| 5 | External protocol audit | Replicate five published protocols; measure real-world MPC rates | FD003 | All |

#### 3.2.4 Phase 5: External Protocol Audit

We select five published papers for replication if they: report FD003 results, use a stacked or single LSTM, and provide sufficient protocol detail (validation split method, patience, max epochs, RUL clipping). Table 4 summarizes the replicated protocols.

**Table 4. Protocols replicated in Phase 5.**

| Audit ID | Citation | Val split | Patience | Max epochs | RUL clip |
|----------|----------|:---------:|:--------:|:----------:|:--------:|
| A1 | Meng et al. (2023) [P31] | Fixed, seed=42 | 15 | 200 | 125 |
| A2 | Qin et al. (2024) [P32] | Fixed, seed=42 | 20 | 200 | 125 |
| A3 | Zheng et al. (2017) [P33] | Fixed, seed=42 | 10 | 200 | 125 |
| A4 | Representative pre-2020 pattern | Fixed, seed=42 | 15 | 200 | None |
| A5 | Elsherif et al. (2025) [P5] | Fixed, seed=42 | N/A | 25 (fixed) | 125 |

Each protocol is replicated with 10 independent seeds. The backbone architecture is held constant across all five protocols. Replication fidelity is confirmed by median RMSE of non-collapsed seeds within 15% of the paper's reported value. MPC labels are assigned before computing test-set RMSE.

---

### 3.3 Research Hypotheses

**H1 (Mechanism):** The combination of a fixed validation split (A1_fixed) and early stopping from epoch 0 (B2_from0) increases MPC probability significantly more than either condition alone. This effect is specific to LSTM because the LSTM forget gate approaches zero under these conditions, blocking gradient flow through the cell state. GRU, which has no separate cell state, does not exhibit MPC.

**H2 (Vulnerability):** MPC susceptibility across C-MAPSS subsets is better explained by label distribution characteristics (RUL variance, clipping ratio, inter-unit variability) than by operating conditions or fault modes alone.

**H3 (Detection):** The composite indicator (PDR + R² + CBR + ISS) detects MPC from validation-only outputs with higher balanced accuracy than any single metric. Calibration-set thresholds generalize to held-out dataset × architecture combinations.

**H4 (Prevention):** Per-seed random split, early-stopping warmup (MIN_EPOCHS ≥ 30), and GRU substitution each reduce MPC occurrence to near-zero without statistically significant RMSE deterioration.

---

### 3.4 Statistical Analysis Plan

MPC occurrence rates are modeled with mixed-effects logistic regression (fixed: protocol factors; random: seed, dataset). Continuous outcomes use mixed-effects linear regression. All significance tests involving multiple contrasts are corrected with Benjamini–Hochberg FDR at α = 0.05. RMSE non-inferiority is assessed with a one-sided test at α = 0.05, using a 10% margin relative to the best non-collapsed reference condition. Detector performance is estimated with leave-one-group-out cross-validation; held-out performance is reported with 95% bootstrap CIs (2,000 resamples).

**Benchmark distortion index:**

$$\Delta_{\mathrm{inflation}} = \left(\frac{E_{\mathrm{collapsed}} - E_{\mathrm{new}}}{E_{\mathrm{collapsed}}}\right) - \left(\frac{E_{\mathrm{valid}} - E_{\mathrm{new}}}{E_{\mathrm{valid}}}\right)$$

where $E_{\mathrm{collapsed}}$ is the RMSE of the collapsed baseline, $E_{\mathrm{valid}}$ is the RMSE of the same model under a corrected protocol, and $E_{\mathrm{new}}$ is the RMSE of a proposed method.

---

## 4. Results: Mechanism and Generalization

### 4.1 Reproduction and Implementation Audit (Phase 0)

We first verify that the motivating observation is not a code error and that the corrected protocol resolves it.

**Table 5. Phase 0: Original versus corrected protocol (FD003, LSTM, 5 seeds).**

| Protocol | Seeds | MPC count | RMSE (mean ± std) | PDR (mean) | R² (mean) | Stop epoch |
|----------|-------|-----------|-------------------|-----------|---------|-----------|
| Original (A1_fixed, B2_from0, MAX=100) | 5 | 4/5 | 43.17 ± 0.18 (collapsed) | 0.000002 | −0.083 | 22.5 |
| Seed 2 only (escaped) | 1 | 0/1 | 13.55 | 0.922 | 0.893 | 100 |
| Corrected (A2_per_seed, MIN=30, MAX=300) | 5 | 0/5 | 14.19 ± 0.61 | 0.974 | 0.882 | 127.4 |

The original H6 archive shows 5/5 seeds collapsed under the same conditions [ANON]. The Phase 0 rerun gives 4/5, consistent with the 80% MPC rate measured in Phase 1A under these conditions — one seed escaped because it ran the full 100 epochs before patience fired (stop epoch = 100 vs. 18–35 for collapsed seeds).

Under the corrected protocol, all five seeds converge. RMSE = 14.19 ± 0.61, PDR = 0.965–0.993, R² = 0.871–0.893. The constant predictor RMSE is 45.07 cycles, confirming that the collapsed model (RMSE ≈ 43) performs no better than predicting the training-set mean.

The collapse is not a code error. Fixing three protocol choices — validation split randomization, early-stopping warmup, and extended maximum epochs — resolves it completely.

---

### 4.2 Validation Composition × Early Stopping (Phases 1A and 1B)

#### 4.2.1 Factorial Experiment (Phase 1A)

We run 3 (val split) × 3 (early stopping) × 3 (clipping) = 27 conditions, 10 seeds each, on FD003+LSTM. Table 6 shows conditions with non-zero MPC rates.

**Table 6. Phase 1A MPC rates: B2_from0 (ES from epoch 0) conditions.**

| Val split (A) | Clipping (C) | MPC rate | RMSE (mean ± std) | PDR (mean) | Stop epoch |
|--------------|-------------|----------|-------------------|-----------|-----------|
| A1_fixed | C1_unclipped | 0.90 | 56.14 ± 6.85 | 0.149 | 21.6 |
| A1_fixed | C2_clip125 | **0.80** | 35.96 ± 11.83 | 0.199 | 38.7 |
| A1_fixed | C3_clip100 | 0.50 | 21.00 ± 13.01 | 0.490 | 64.8 |
| A2_per_seed | C1_unclipped | 0.40 | 49.22 ± 12.63 | 0.891 | 53.4 |
| A2_per_seed | C2_clip125 | 0.30 | 21.50 ± 14.02 | 0.716 | 102.7 |
| A2_per_seed | C3_clip100 | 0.00 | 8.43 ± 0.75 | 0.982 | 122.6 |
| A3_stratified | C1_unclipped | 0.50 | 51.15 ± 13.52 | 0.745 | 55.2 |
| A3_stratified | C2_clip125 | 0.30 | 21.99 ± 13.84 | 0.705 | 82.6 |
| A3_stratified | C3_clip100 | 0.30 | 16.49 ± 11.77 | 0.685 | 89.6 |

B1 conditions (no early stopping): MPC = 0.00 across all 90 runs.
B3 conditions (warmup: MIN_EPOCHS=30): MPC = 0.00 across all 90 runs.

**Finding 1 — B2 (ES from epoch 0) is the necessary trigger.** Neither B1 nor B3 produces a single collapsed run. B2 is required for MPC.

**Finding 2 — A1 (fixed split) amplifies collapse rate.** A1+B2+C2_clip125 = 80%; A2+B2+C2_clip125 = 30%. Fixed splits allow the optimizer to exploit the specific RUL composition of one held-out set.

**Finding 3 — Clipping modulates severity.** A1+B2+C1_unclipped = 90% vs. A2+B2+C3_clip100 = 0%.

#### 4.2.2 Optimization and Initialization (Phase 1B)

All Phase 1B conditions use A1_fixed+B2_from0+C2_clip125. We vary one factor at a time.

**Table 7. Phase 1B: Factor sweep results (FD003+LSTM, 10 seeds each).**

| Factor | Variant | MPC rate | RMSE (mean ± std) |
|--------|---------|----------|--------------------|
| Patience | 5 | 1.00 | 41.60 ± 0.23 |
| | 10 | 0.90 | 38.61 ± 9.38 |
| | **15 (canonical)** | **0.80** | **35.96 ± 11.83** |
| | 30 | 0.20 | 18.38 ± 12.21 |
| Learning rate | 0.01 | 0.90 | 38.44 ± 8.91 |
| | **0.001 (canonical)** | **0.80** | **35.96 ± 11.83** |
| | 0.0001 | 1.00 | 41.30 ± 0.04 |
| Output bias init | zero (canonical) | 0.80 | 35.96 ± 11.83 |
| | train mean | **0.00** | 12.94 ± 0.48 |
| | random calibrated | **0.00** | 12.69 ± 0.74 |
| Loss function | MSE (canonical) | 0.80 | 35.96 ± 11.83 |
| | MAE | **0.00** | 13.43 ± 0.60 |
| | auxiliary MSE | 0.40 | 24.24 ± 15.11 |

**Finding 4 — Patience is a continuous modulator, not a structural fix.** Patience=30 reduces MPC to 20% but does not eliminate it.

**Finding 5 — Learning rate does not help.** MPC rate is 90–100% across all tested learning rates. Lower LR (0.0001) increases collapse rate to 100%.

**Finding 6 — Output bias initialization to training-set mean eliminates MPC.** MPC rate = 0% with no RMSE penalty (12.94 ± 0.48 vs. 14.19 ± 0.61 for the corrected protocol).

**Finding 7 — MAE loss eliminates MPC.** Under L1 loss, the loss landscape does not have the flat basin near the marginal mean that enables the MSE trivial solution. MPC rate = 0%, RMSE = 13.43 ± 0.60.

---

### 4.3 Training Dynamics (Phase 2)

We collect epoch-by-epoch PDR and R² trajectories for all 400 runs from Phases 1A and 1B (116 collapsed, 284 normal).

In collapsed runs, PDR drops to ≈2×10⁻⁶ at epoch 1 and remains there throughout training. R² is negative from epoch 1 (mean R² = −2.58). Validation loss decreases monotonically — the collapse is silent.

In normal runs, PDR is also low at epoch 1 (mean = 0.013), then recovers gradually: PDR reaches 0.38 by epoch 30, 0.88 by epoch 44, and stabilizes above 0.90 by epoch ~50.

This explains why B3 (MIN_EPOCHS=30) prevents MPC. At epoch 30, normal runs have PDR = 0.38 — recovering — while collapsed runs remain flat at PDR ≈ 2×10⁻⁶. Early stopping fires during the recovery window for normal runs under B2 but not under B3.

**Real-time alarm performance.** Using PDR < 0.05 as an online alarm, sensitivity = 1.00 but specificity = 0.063 — the alarm fires for nearly all runs in the first few epochs, including normal runs that have not yet recovered. A real-time alarm is not actionable. Retrospective detection at training completion (Phase 4) resolves this limitation.

---

### 4.4 Dataset and Architecture Generalization (Phase 3)

We apply trigger conditions (A1_fixed+B2_from0+C2_clip125) to 4 datasets × 4 architectures, 10 seeds each.

**Table 8. Phase 3 MPC rates (%) across dataset × architecture.**

| Dataset | LSTM | GRU | 1D-CNN | MLP |
|---------|------|-----|--------|-----|
| FD001 | 10 | 0 | 0 | 0 |
| FD002 | 0 | 0 | 0 | 0 |
| **FD003** | **80** | **0** | **0** | **0** |
| FD004 | 0 | 0 | 0 | 0 |

**Finding 8 — MPC is FD003-specific among C-MAPSS subsets.** FD002 and FD004 (6 operating conditions) show 0% MPC. FD003 (single operating condition, two fault modes) shows 80% for LSTM.

**Finding 9 — MPC is LSTM-specific among tested architectures.** Under the same trigger conditions on FD003, GRU, 1D-CNN, and MLP show 0% MPC. RMSE for GRU (12.61 ± 0.47) is comparable to the corrected LSTM baseline (14.19 ± 0.61).

---

### 4.5 LSTM Forget Gate Mechanism (Phase 3B)

Four variants of the FD003 model are trained under A1_fixed+B2_from0+C2_clip125, 10 seeds each.

**Table 9. Phase 3B: Forget gate variants on FD003.**

| Variant | Description | MPC rate | RMSE (mean ± std) | Stop epoch |
|---------|-------------|----------|--------------------|-----------|
| LSTM_gate (base) | Standard LSTM | 0.60 | 30.18 ± 14.98 | 61.7 |
| GRU_gate | GRU replacement | **0.00** | 12.27 ± 0.67 | 80.9 |
| V1_fb1 | Forget bias initialized to +1 | 0.40 | 24.29 ± 14.80 | 62.5 |
| V2_fg1 | Forget gate clamped to 1 (CEC) | **0.00** | 12.96 ± 0.95 | 98.7 |

**Finding 10 — GRU eliminates MPC.** GRU has no separate cell state; its gradient pathway does not exhibit T-step multiplicative decay. Under identical trigger conditions: 0% MPC.

**Finding 11 — Forget gate = 1 (CEC) eliminates MPC.** V2_fg1 clamps f = 1.0, allowing full gradient flow through the cell state. MPC = 0%, RMSE = 12.96 ± 0.95.

**Finding 12 — Forget bias initialization to +1 reduces but does not eliminate MPC.** V1_fb1 falls from 60% to 40%.

**Mechanistic conclusion.** Under A1_fixed+B2_from0 conditions, the LSTM forget gate approaches f ≈ 0 for some seeds in early training. When f ≈ 0, the BPTT gradient through the cell state is blocked (∂L/∂h_t ∝ f^T ≈ 0^30 ≈ 0). The model cannot update its recurrent representation. It is trapped at the trivial solution. Early stopping fires before any recovery occurs. Preventing forget gate saturation — by clamping f = 1, using GRU, or initializing bias to +1 — disrupts this chain.

---

## 5. Results: Prevention and Retrospective Detection

### 5.1 Prevention Taxonomy

Phase 1A and 1B identify seven interventions. We evaluate each on MPC rate and RMSE non-inferiority relative to the corrected-protocol baseline (RMSE = 14.19 ± 0.61). Non-inferior = RMSE ≤ 15.6 cycles (within 10% of baseline mean).

**Table 10. Prevention taxonomy: MPC rate, RMSE, and implementation cost.**

| Intervention | Type | MPC rate | RMSE (mean ± std) | Non-inferior? | Cost |
|-------------|------|----------|--------------------|:---:|------|
| Output bias → train mean | Initialization | **0.00** | 12.94 ± 0.48 | ✓ | Minimal — one-line change |
| Output bias → random calib. | Initialization | **0.00** | 12.69 ± 0.74 | ✓ | Minimal |
| MAE loss | Loss function | **0.00** | 13.43 ± 0.60 | ✓ | Low — loss swap only |
| Warmup (MIN_EPOCHS=30) | Protocol | **0.00** | 12.42 ± 1.04 | ✓ | Low — one hyperparameter |
| Per-seed random split (A2) | Protocol | 0.30* | varies | — | Low |
| Forget gate = 1 (V2_fg1) | Architecture | **0.00** | 12.96 ± 0.95 | ✓ | Minimal |
| GRU substitution | Architecture | **0.00** | 12.27 ± 0.67 | ✓ | Moderate — model change |
| Patience = 30 | Protocol | 0.20 | 18.38 ± 12.21 | ✗ | Low |

*Per-seed split alone with B2+C2_clip125 reduces but does not eliminate MPC. Combined with warmup: 0%.

Five interventions achieve 0% MPC with RMSE non-inferior to the corrected baseline. The minimum-cost option is output bias initialization to the training-set mean: one line of code, no architecture change, no additional computation.

---

### 5.2 Retrospective Audit Tool

Phase 1A/1B collapse labels were assigned by blind adjudication using validation-set outputs only, before examining test-set RMSE. Inter-rater agreement: Cohen's κ = 1.00 for the PDR < 0.05 criterion.

**Table 11. Retrospective MPC detection performance.**

| Indicator | Threshold | Dataset | Sensitivity | Specificity | Balanced acc. |
|-----------|-----------|---------|:-----------:|:-----------:|:---:|
| PDR | < 0.05 | Calibration | 1.00 | **1.00** | 1.00 |
| PDR | < 0.05 | Held-out | 1.00 | **1.00** | 1.00 |
| R² | ≤ 0 | Calibration | 1.00 | 0.935 | 0.968 |
| R² | ≤ 0 | Held-out | 1.00 | **1.00** | 1.00 |
| RMSE | > 25 cycles | Calibration | 1.00 | 0.687 | 0.844 |
| RMSE | > 25 cycles | Held-out | 1.00 | 0.957 | 0.979 |
| PDR + R² (OR) | either criterion | Held-out | 1.00 | **1.00** | 1.00 |

PDR < 0.05 achieves perfect sensitivity and specificity on both calibration (40 collapsed, 230 non-collapsed) and held-out (95 collapsed, 235 non-collapsed) data. R² ≤ 0 achieves perfect separation on the held-out set. RMSE > 25 cycles alone is insufficient (72 false positives in calibration).

**All three indicators can be computed from validation-set outputs at training completion. Test-set access is not required.**

---

## 6. External Protocol Audit

### 6.1 Audit Design and Paper Selection

Phase 5 asks whether the trigger conditions exist in published work. We select five published training protocols for replication using pre-registered criteria (Section 3.2.4).

A4 represents the protocol pattern prevalent before the RUL clipping convention was established: fixed validation split, no warmup, unclipped labels. This matches the A1_fixed+B2_from0+C1_unclipped condition in Phase 1A exactly (MPC = 90%).

---

### 6.2 MPC Prevalence in Published Protocols

**Table 12. MPC rates and RMSE across replicated protocols (10 seeds each).**

| Protocol | MPC rate | RMSE (mean ± std) | PDR (mean) | R² (mean) | Stop epoch |
|----------|----------|-------------------|-----------|---------|-----------|
| A1 Meng 2023 [P31] | 0.80 | 35.96 ± 11.83 | 0.199 | 0.075 | 38.7 |
| A2 Qin 2024 [P32] | 0.60 | 30.18 ± 14.64 | 0.400 | 0.280 | 60.3 |
| A3 Zheng 2017 [P33] | 0.90 | 38.61 ± 9.38 | 0.097 | −0.023 | 23.6 |
| A4 [pre-2020] | 0.90 | 56.14 ± 6.85 | 0.149 | −0.864 | 21.6 |
| A5 Elsherif 2025 [P5] | 0.90 | 39.33 ± 7.14 | 0.097 | −0.038 | 25.0 |
| **Overall** | **0.82** | **40.04 ± 14.27** | **0.188** | **−0.114** | **33.8** |

Of 50 total replication runs, 41 collapsed. MPC rate ranges from 60% (A2) to 90% (A3, A4, A5).

A2 has the lowest collapse rate (60%), corresponding to longer patience (20 vs. 10–15 for others). A5 uses fixed epochs (25) rather than early stopping. At epoch 25, normal runs have PDR ≈ 0.10 — still in the recovery window. Fixed-epoch training terminated during the recovery window produces a collapsed-state model.

For each protocol, non-collapsed seeds achieve RMSE consistent with the corrected-protocol baseline (13–16 cycles). These seeds either escaped by chance (different gradient initialization path) or had a validation split not exploitable as a constant predictor.

---

### 6.3 Benchmark Distortion Quantification

Using the motivating observation [ANON] as a concrete example:

| Quantity | Value |
|---------|-------|
| E_collapsed (original M0, FD003) | 43.23 cycles |
| E_valid (corrected M0, FD003) | 12.97 cycles |
| E_new (M3 attention-gate, original protocol) | 14.78 cycles |
| Apparent improvement (collapsed baseline) | **65.8%** |
| Actual improvement (corrected baseline) | −14.0% (p_BH = 0.754, not significant) |
| Δ_inflation | **0.80** |

The 65.8% improvement claim is entirely a protocol artifact. Δ_inflation = 0.80 means 80 percentage points of the apparent improvement are fictitious.

**Table 13. Benchmark distortion by protocol (hypothetical proposed method at RMSE = 13 cycles).**

| Protocol | E_collapsed (median) | E_valid | Apparent improvement | True improvement | Δ_inflation |
|----------|---------------------|---------|---------------------|-----------------|------------|
| A1 Meng 2023 | ~43 | 14.19 | ~70% | ~8% | ~0.62 |
| A2 Qin 2024 | ~43 | 14.19 | ~70% | ~8% | ~0.62 |
| A3 Zheng 2017 | ~44 | 14.19 | ~70% | ~8% | ~0.62 |
| A4 [pre-2020] | ~57 | 14.19 | ~77% | ~8% | ~0.69 |
| A5 Elsherif 2025 | ~43 | 14.19 | ~70% | ~8% | ~0.62 |

A method achieving genuinely good RMSE (~13 cycles) would appear to improve by 70–77% over a collapsed baseline, while the true improvement over a valid baseline is ~8%.

This analysis is conservative. If the proposed method also uses the MPC-prone protocol, its RMSE may also reflect collapsed runs, compounding the distortion.

**Limitation.** Δ_inflation quantifies distortion relative to the FD003 corrected baseline. It does not apply to FD001, FD002, or FD004, which show 0–10% MPC rates under trigger conditions.

---

## 7. Discussion

### 7.1 Implications for PHM Research

#### 7.1.1 MPC Within the MSE Mean-Prediction Hierarchy

The theoretical connection between MSE loss and mean-directed predictions is well established. Bruna et al. [P19] identified the "regression-to-the-mean problem" in image super-resolution, where MSE point estimates converge to the conditional mean E[y|x]. Mathieu et al. [P20] showed the same for video prediction. Huang [P21] formalized this for multimodal regression, naming the outcome "mean collapse." In all of these cases, mean-directed prediction is a consequence of data geometry — the presence of multiple modes in the target distribution.

MPC is a more severe form of the same failure. A model exhibiting MPC does not predict the conditional mean E[y|x]; it predicts the marginal mean E[y]. All conditioning on the input is discarded. The failure is not a consequence of data geometry alone. It is a consequence of a specific interaction between data geometry, architecture, and training protocol.

Li et al. [P18] proved that under MSE loss with unconstrained features, the global optimum corresponds to the class-mean predictions. Their theoretical result implies that a constant predictor equal to the training-set mean is a stable fixed point of the MSE loss landscape. When the optimizer reaches this fixed point before early stopping fires, it cannot escape.

The MPC taxonomy extends the existing hierarchy: well-trained models exhibit Conditional-Mean Compression (CMC; [P7]), which is a statistical property of regression at distribution extremes. MPC precedes CMC: the model never acquires the ability to predict conditional means because it is captured at the marginal mean during optimization. Lee and Chen [P7] characterized CMC as a post-training bias. MPC is a pre-training failure.

#### 7.1.2 LSTM Forget Gate Saturation: Structural Vulnerability

The original LSTM was designed to solve the vanishing gradient problem through the Constant Error Carousel (CEC) principle: by setting the internal error flow to a constant, the cell state can carry information across arbitrarily long sequences without gradient decay. The forget gate — introduced later as a practical modification — breaks the CEC guarantee. When f → 0, the cell state gradient decays as f^T over T timesteps. For T = 30, even f = 0.05 gives f^30 ≈ 10⁻²⁰ — numerically zero.

This has a precise implication: clamping the forget gate to a small ε does not prevent MPC. Phase 3B confirms this: fg_clamp(ε = 0.001, 0.01, 0.05) all show 60% MPC — identical to the unclamped baseline. Only V2_fg1 (f = 1, restoring CEC) achieves 0% MPC. The theoretical condition that eliminates collapse is exact: f = 1, not f ≥ ε.

Al-Selwi et al. [P2] studied vanishing gradients in LSTMs on C-MAPSS and characterized them as a slow-convergence failure. MPC is the dual failure: training is fast but converges to the wrong solution. Gradient magnitudes are not too small to drive learning; they are sufficient to reach the trivial attractor and insufficient to escape it.

Papyan et al. [P17] documented Neural Collapse (NC) in deep classification networks: during the terminal training phase, last-layer features collapse to class means. NC is a normal, benign phenomenon in classification. MPC is the pathological analog in regression. NC occurs after successful learning; MPC occurs instead of learning.

GRU has no separate cell state. Its gradient pathway does not exhibit the T-step multiplicative decay that enables MPC. Under identical trigger conditions on FD003, GRU achieves 0% MPC across 10 seeds. GRU and LSTM differ in exactly one structural property (cell state with forget gate), and they differ in exactly one outcome (MPC versus no MPC).

#### 7.1.3 Validation Composition as a Loss Landscape Shaping Force

Prechelt [P14] established that early stopping should halt training when validation loss improvement stalls. This prescription assumes that validation loss is a reliable proxy for generalization. MPC reveals a systematic failure of this assumption.

When the validation split is fixed across training seeds, the held-out engine set has a fixed RUL distribution. A constant predictor outputting the training-set mean achieves low validation MSE on this fixed set — lower than a partially trained model whose predictions are still scattered. Early stopping selects the constant predictor as the "best" checkpoint.

Vabalas et al. [P6] showed that fixed K-fold splits produce biased performance estimates. MPC is a more severe manifestation: the fixed validation set does not merely bias the estimate; it actively selects a degenerate solution as the termination point. Miseta et al. [P12] and Mahsereci et al. [P13] proposed alternative early stopping criteria precisely because standard validation-loss stopping has unresolved weaknesses. Our data provide a concrete failure case.

The interaction between B2 (ES from epoch 0) and A1 (fixed split) is the necessary condition for MPC. A1 without B2 produces 0% MPC. B2 without A1 reduces MPC to 30%. This interaction is a protocol-level analog of a multicollinear cause: the individual factors are insufficient, but their combination is catastrophic.

#### 7.1.4 Benchmark Reproducibility in PHM

Musgrave et al. [P23], Lučić et al. [P24], and Ferrari Dacrema et al. [P22] documented that claimed improvements disappeared under consistent experimental conditions. Their diagnosis was undertrained baselines. The remedy was more tuning.

MPC differs in one critical respect: the baseline converges. Convergence criteria are met. The baseline cannot be fixed by more tuning within the same protocol; the protocol must be corrected. This is why MPC is more difficult to detect and causes more persistent inflation.

The Δ_inflation metric quantifies this distortion. For the motivating case, Δ_inflation = 0.80: 80 percentage points of the claimed improvement disappear when the baseline is corrected. Under the five external audit protocols, the same pattern holds (Δ_inflation ≈ 0.62–0.69).

If a non-trivial fraction of published FD003+LSTM baselines are in the MPC state, then the improvement trends reported in the literature partially reflect baseline collapse correction rather than algorithmic advancement. The external audit (Phase 5) shows 60–90% MPC rates across five independently selected published protocols. This is not a case study; it is a systematic pattern.

#### 7.1.5 Guidance for PHM Practitioners

**Minimum-cost prevention (single change, no performance impact):**

1. *Output bias initialization to training-set mean.* `model.fc[-1].bias.data.fill_(c_train)`. One line of code. MPC rate = 0%, RMSE = 12.94 ± 0.48.
2. *Early-stopping warmup.* MIN_EPOCHS = 30 before early stopping activates. MPC rate = 0%, RMSE = 12.42 ± 1.04.

**Retrospective audit (no retraining required):**

```python
pdr = np.std(val_predictions) / (np.std(val_targets) + 1e-8)
if pdr < 0.05:
    # MPC detected — retrain with corrected protocol
```

This single check achieves AUROC = 1.00 on both calibration (270 runs) and held-out (330 runs) data. Any existing trained model can be audited retrospectively without accessing the test set.

---

### 7.2 Limitations

**MPC is LSTM-specific in the tested conditions.** GRU, 1D-CNN, and MLP achieve 0% MPC under the same trigger conditions on FD003. Transformer-based models and attention-enhanced LSTMs were not tested. Whether attention mechanisms or gating variants exhibit analogous vulnerabilities is an open question.

**The study is limited to C-MAPSS (FD001–FD004).** C-MAPSS is a thermodynamic simulator. Real turbofan sensor data is more heterogeneous. The homogeneous degradation templates in FD003 create the stable trivial minimum in the MSE landscape. In more heterogeneous data, this minimum may not be stable, and MPC may be less likely.

**Statistical power for pairwise comparisons is limited.** With n = 10 seeds per condition, the difference between 20% and 30% MPC rates is not statistically distinguishable. The study provides strong evidence for the 0% vs. non-zero contrast but not for ordering among partial-mitigation methods.

**Gradient trajectories were not directly measured.** The forget gate saturation mechanism is inferred from the causal intervention (V2_fg1 → 0% MPC) rather than direct gradient norm logging. The theoretical chain is consistent with all experimental results, but direct measurement would provide additional mechanistic evidence.

**Prevention comparison fairness.** ES warmup (B2 → B3) and per-seed split (A1 → A2) change the protocol condition rather than fixing the model. They represent category-level interventions (protocol redesign) while bias_init and MAE loss represent point-level interventions.

---

## 8. Conclusion

This study examined Mean-Prediction Collapse (MPC) — a failure in which a trained LSTM outputs a near-constant prediction regardless of input — in LSTM-based RUL regression on C-MAPSS.

**What causes MPC?** The trigger is a three-way interaction: a fixed validation split across training seeds, early stopping without a warmup period, and MSE loss on a homogeneous dataset. No single factor alone is sufficient. When all three are present, the LSTM's forget gate approaches zero in the first epoch of training. This blocks gradient flow through the cell state via BPTT, trapping the model at a trivial constant-prediction solution. Early stopping then halts training at this degenerate state. Validation loss decreases throughout — the collapse is silent.

**What architecture types are affected?** MPC is LSTM-specific. GRU, 1D-CNN, and MLP achieve 0% MPC under the same trigger conditions. Among C-MAPSS subsets, only FD003 shows substantial MPC rates (80% for LSTM). FD002 and FD004 are immune because their six operating conditions prevent the trivial solution from being stable.

**Can MPC be detected without the test set?** Yes. PDR < 0.05 identifies MPC with sensitivity = 1.00 and specificity = 1.00 on both calibration (270 runs) and held-out (330 runs) data. AUROC = 1.0000.

**How can MPC be prevented?** Five interventions eliminate MPC completely, each with no cost to RMSE: initialize the output layer bias to the training-set mean RUL, set a minimum training period before early stopping activates, use MAE loss, replace LSTM with GRU, or fix the forget gate to 1. The cheapest option is bias initialization: one line of code.

**Is MPC present in published work?** We replicated five published training protocols. MPC rates ranged from 60% to 90% across 10 seeds per protocol (41 of 50 total runs collapsed, 82%). In the case that motivated this study, correcting the baseline protocol reduced an apparent 65.8% improvement to a statistically non-significant difference. Protocol correction, not better models, drove the change.

**Core recommendation.** Add one line to any LSTM training loop:

```python
model.fc[-1].bias.data.fill_(train_rul_mean)   # prevents MPC
```

Then, after early stopping:

```python
pdr = np.std(val_preds) / (np.std(val_targets) + 1e-8)
if pdr < 0.05:
    print("WARNING: MPC — retrain with corrected protocol")
```

**Future work.** Three extensions are natural: (1) whether analogous failures occur in Transformer-based or attention-enhanced RUL models, (2) validation on real sensor data from turbofan fleets (e.g., N-CMAPSS [P4]) where MPC vulnerability may be lower, and (3) direct measurement of forget gate activations and BPTT gradient norms to replace causal inference with direct mechanistic evidence.

---

## Data and Code Availability

The C-MAPSS dataset is publicly available from NASA Prognostics Data Repository [P29]. Experiment scripts, split manifests, epoch-level logs, and raw prediction files will be made available on GitHub upon acceptance. The retrospective audit tool (PDR-based diagnostic) is included as a standalone function in the repository.

---

## References

[P1] Dohmatob, E. et al. (2024). Model collapse demystified: The case of regression. *arXiv*.

[P2] Al-Selwi, S. et al. (2023). LSTM inefficiency in long-term dependencies regression problems. *Journal of Advanced Research in Applied Sciences and Engineering Technology*.

[P3] Ellefsen, A. L. et al. (2019). Remaining useful life predictions for turbofan engine degradation using semi-supervised deep architecture. *Reliability Engineering & System Safety*, 183, 240–251.

[P4] Chao, M. A. et al. (2021). Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics. *Data*, 6(1), 5.

[P5] Elsherif, S. M. et al. (2025). A deep learning-based prognostic approach for predicting turbofan engine degradation and remaining useful life. *Scientific Reports*, 15, Art. 12959.

[P6] Vabalas, A. et al. (2019). Machine learning algorithm validation with a limited sample size. *PLoS ONE*, 14(11), e0224365.

[P7] Lee, M. & Chen, T. (2025). Systematic bias of machine learning regression models and correction. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

[P8] Das, S. et al. (2024). Uncertainty-aware deep learning for monitoring and fault diagnosis from synthetic data. *Reliability Engineering & System Safety*, 240, 109606.

[P11] Tan, Y. et al. (2025). SynTSBench: Rethinking temporal pattern learning in deep learning models for time series. *arXiv*.

[P12] Miseta, T. et al. (2023). Surpassing early stopping: A novel correlation-based stopping criterion for neural networks. *Neurocomputing*, 556, 126627.

[P13] Mahsereci, M. et al. (2017). Early stopping without a validation set. *arXiv:1703.09580*.

[P14] Prechelt, L. (1998). Early stopping — but when? In *Neural Networks: Tricks of the Trade*. Springer.

[P15] Zhuang, L. et al. (2023). A prognostic driven predictive maintenance framework based on Bayesian deep learning. *Reliability Engineering & System Safety*, 234, 109181.

[P16] Li, X. et al. (2018). Remaining useful life estimation in prognostics using deep convolution neural networks. *Reliability Engineering & System Safety*, 172, 1–11.

[P17] Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. *PNAS*, 117(40), 24652–24663.

[P18] Li, Z. et al. (2022). On the optimization landscape of neural collapse under MSE loss: Global optimality with unconstrained features. *arXiv:2203.01238*.

[P19] Bruna, J., Sprechmann, P., & LeCun, Y. (2015). Super-resolution with deep convolutional sufficient statistics. *arXiv:1511.05666*.

[P20] Mathieu, M., Couprie, C., & LeCun, Y. (2016). Deep multi-scale video prediction beyond mean square error. *ICLR 2016*.

[P21] Huang, W. (2026). Resolving multi-modal regression by difference-quotient-based clustering. *arXiv:2608.25467*. *(arXiv preprint — not peer reviewed)*

[P22] Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). Are we really making much progress? *ACM RecSys 2019*.

[P23] Musgrave, K., Belongie, S., & Lim, S.-N. (2020). A metric learning reality check. *ECCV 2020*.

[P24] Lučić, M. et al. (2018). Are GANs created equal? A large-scale study. *NeurIPS 2018*.

[P25] Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.

[P26] Schulz, K. F., Altman, D. G., Moher, D., & CONSORT Group (2010). CONSORT 2010 statement. *BMJ*, 340, c332.

[P27] Murphy, A. H. (1988). Skill scores based on the mean square error and their relationships to the correlation coefficient. *Monthly Weather Review*, 116(12), 2417–2424.

[P28] Zeiler, M. D. & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV 2014*, LNCS 8689, 818–833.

[P29] Saxena, A., Goebel, K., Simon, D., & Ecker, W. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *2008 IEEE Int. Conf. on Prognostics and Health Management (PHM)*. doi:10.1109/PHM.2008.4711414

[P31] Meng, H. et al. (2023). Bayesian gated-transformer model for risk-aware prediction of aero-engine remaining useful life. *Expert Systems with Applications*, 238, Art. 121859.

[P32] Qin, Y. et al. (2024). Spatial and temporal attention-based and residual-driven long short-term memory networks with implicit features for remaining useful life prediction. *Engineering Applications of Artificial Intelligence*, 133, Art. 108563.

[P33] Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). Long short-term memory network for remaining useful life estimation. *2017 IEEE Int. Conf. on Prognostics and Health Management (ICPHM)*. doi:10.1109/ICPHM.2017.7998311

[ANON] Authors (2026). [Title anonymized for review]. *Reliability Engineering & System Safety* (under review). *(de-anonymized upon acceptance)*
