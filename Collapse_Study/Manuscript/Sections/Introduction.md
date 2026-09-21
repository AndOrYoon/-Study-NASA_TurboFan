# Introduction
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression

**작성일:** 2026-09-21  
**버전:** Draft v0.1  
**관련 파일:** `Research_Plan.md` (RQ·기여 정의), `Literature_Review.md` (선행 연구 상세), `Decision_log.md`

---

> **작성 방침:** 단순 문장, 능동태, 짧은 단락. 연구 갭과 연구 질문이 섹션 4–5에서 명확히 보이도록 구성.

---

## 1. Context

Remaining useful life (RUL) prediction enables condition-based maintenance. When a system's remaining life can be estimated accurately, maintenance can be scheduled before failure rather than after. This reduces downtime, lowers cost, and prevents hazardous failures.

The C-MAPSS turbofan engine dataset [P29] has become the standard benchmark for deep learning–based RUL prediction. Since its release, hundreds of studies have used C-MAPSS to report performance improvements. Most follow the same pattern: propose a new model, compare it to a baseline, and report a percentage improvement in RMSE or NASA score.

This study does not propose a new model. It questions whether the baselines in these comparisons are valid.

---

## 2. Motivating Observation

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

## 3. Mean-Prediction Collapse: Definition and Importance

We refer to this failure as **Mean-Prediction Collapse (MPC)**. This section defines the concept precisely, distinguishes it from related phenomena, and explains why it matters for benchmark-based research.

### 3.1 Formal Definition

> **Mean-Prediction Collapse (MPC)** refers to a degenerate regression behavior in which model predictions exhibit substantially reduced dispersion relative to the target distribution and concentrate around the population or conditional mean, thereby suppressing input-dependent variations and systematically underrepresenting extreme target values.

This definition has four components. Each is grounded in prior literature.

| Component | What it means in RUL regression | Prior evidence |
|-----------|----------------------------------|----------------|
| *Reduced dispersion relative to the target distribution* | The spread of predictions is negligible compared to the spread of true RUL values | Mathieu et al. [P20]: "inherently blurry predictions" under MSE |
| *Concentrate around the population or conditional mean* | All predictions cluster near a single value (~87 cycles in FD003), regardless of which engine or cycle is queried | Bruna et al. [P19]: "regression-to-the-mean problem"; Huang [P21]: squared-loss regression converges to conditional mean |
| *Suppressing input-dependent variations* | The partial derivative of the output with respect to the input approaches zero — the model is functionally constant | Huang [P21]: unconstrained regressor under squared loss ignores multi-modal structure |
| *Systematically underrepresenting extreme target values* | Engines near end-of-life (low RUL) and engines early in their life (high RUL) receive the same near-mean prediction | Lee & Chen [P7]: systematic mean bias at distribution extremes |

The "degenerate" characterization is not merely qualitative. An MPC model performs no better — and often worse — than a trivial constant predictor that always outputs the training-set mean RUL. Mathematically, its coefficient of determination R² ≤ 0.

### 3.2 Distinction from Related Phenomena

MPC must be distinguished from two phenomena it superficially resembles.

**Conditional-Mean Compression (CMC)** is a statistical property of well-trained regression models. A model exhibiting CMC still uses its inputs to produce differentiated predictions, but its output range is compressed toward the mean relative to the true target range. It systematically underestimates high RUL and overestimates low RUL. This is the phenomenon characterized by Lee and Chen [P7]. CMC occurs *after* successful training; the model has learned from the data. MPC occurs *instead of* training; the model has not learned from the data.

**Undertrained baselines** occur when a model receives insufficient optimization — too few epochs, an inadequate learning rate search, or a suboptimal architecture choice. An undertrained model still uses its inputs; it simply has not converged to its best solution yet. Musgrave et al. [P23] and Lučić et al. [P24] documented this extensively across metric learning and GAN evaluation. MPC is not an undertrained model. The LSTM in MPC converges quickly and stably — to the wrong solution. Additional tuning does not fix it unless the protocol is corrected.

The key distinction is functional: an MPC model has zero prognostic discriminability. Table 1 summarizes these distinctions.

**Table 1. MPC versus related phenomena.**

| Phenomenon | Uses inputs? | Occurs when? | Fix |
|------------|:---:|---|---|
| MPC (this study) | ✗ | During training; protocol-induced | Correct the protocol |
| Conditional-Mean Compression [P7] | ✓ | After training; statistical | Bias correction or loss modification |
| Undertrained baseline [P23, P24] | ✓ | During training; insufficient optimization | More tuning |
| Vanishing gradient (LSTM) [P2] | ✓ | During training; architecture | Gradient clipping, architecture change |

### 3.3 Measurable Indicators

MPC severity is quantified by three indicators, defined formally in Section 4. Here we introduce them briefly.

**Prediction Dispersion Ratio (PDR)** normalizes prediction spread by target spread:

$$\mathrm{PDR} = \frac{\mathrm{std}(\hat{y})}{\mathrm{std}(y) + \varepsilon}$$

PDR → 0 means predictions are nearly constant. In the FD003 collapse case, PDR < 0.0001. In a normally trained model, PDR ≈ 0.7–1.0.

**Constant-Baseline Ratio (CBR)** compares the model's RMSE to that of a trivial constant predictor equal to the training-set mean:

$$\mathrm{CBR} = \frac{\mathrm{RMSE}(\hat{y},\, y)}{\mathrm{RMSE}(c_{\mathrm{train}},\, y) + \varepsilon}$$

CBR ≈ 1 means the model is no better than a constant. In the FD003 collapse, CBR ≈ 1.02 — the LSTM is marginally *worse* than simply predicting the mean.

**Coefficient of determination (R²)** quantifies the fraction of variance explained. R² ≤ 0 means the model explains none of the variance in the targets — the constant predictor would be at least as good. In the FD003 collapse, R² ≈ −0.04.

None of these indicators requires test-set access. They can be computed on the validation set at training completion, enabling retrospective detection of MPC without additional experiments.

### 3.4 Why MPC Matters

MPC has a direct consequence for benchmark comparisons. When a baseline is in the MPC state, its RMSE is artificially high. Any subsequent method — regardless of whether it is genuinely better — will appear to achieve a large improvement.

In the case that motivated this study: the collapsed baseline reported RMSE = 43.23. The corrected baseline achieves RMSE = 12.97. A method achieving RMSE = 12 appears to improve by 72% over the collapsed baseline. Over the corrected baseline, the same method improves by less than 8%. The improvement claim changes by an order of magnitude depending solely on whether the baseline training protocol is correct.

This inflation is not a deliberate distortion. Authors who report it typically do not know the baseline has collapsed. Validation loss behaves normally — it decreases during training and stabilizes at early stopping. There is no obvious signal that the model is in a degenerate state. This is what makes MPC dangerous: it is silent.

MPC is also not rare. We replicated five published training protocols from EAAI and Expert Systems with Applications and measured MPC occurrence rates of 60–90% across 10 independent random seeds per protocol. Under the protocol conditions common in the literature, most training runs collapse.

---

## 4. Research Gaps

Prior work has addressed related but distinct problems.

**Undertrained baselines.** Musgrave et al. [P23] and Lučić et al. [P24] showed that apparent improvements in metric learning and generative modeling disappeared when baselines were properly optimized. Their diagnosis was incomplete optimization — baselines that needed more tuning. MPC is different. The model converges quickly. It is not undertrained. It is trapped at a degenerate solution that MSE loss cannot escape given specific protocol conditions.

**Gradient pathology in LSTMs.** Al-Selwi et al. [P2] studied vanishing gradients in LSTMs on C-MAPSS. Their focus was slow convergence — a training that eventually progresses but too slowly. MPC is the opposite: fast convergence to the wrong solution.

**Validation split bias.** Vabalas et al. [P6] showed that fixed cross-validation splits produce biased performance estimates on small datasets. They did not examine how fixed splits interact with early stopping to select degenerate solutions on the MSE loss surface.

**Mean bias in regression.** Lee and Chen [P7] characterized the systematic bias of regression models toward the mean at distribution extremes. This is a statistical property of well-trained models. MPC is a training failure, not a statistical property. The model does not predict conditional means; it predicts a near-constant marginal mean across all inputs.

**The gap.** No prior study has examined when and why RUL regression models converge to a degenerate near-constant solution, under what protocol conditions this occurs reproducibly, whether it can be detected without test data, and what interventions prevent it. This is the gap this study addresses.

---

## 5. Research Questions

We investigate four research questions.

**RQ1 — Mechanism:**  
Under what training, validation, labeling, and optimization conditions does an RUL regressor converge to a near-constant, input-insensitive prediction?

*We hypothesize that the combination of a fixed validation split and early stopping without a warmup period selects a degenerate solution. This selection is specific to LSTM architectures because of a gradient pathology in the LSTM forget gate.*

**RQ2 — Vulnerability:**  
Which dataset, label-distribution, and architecture characteristics increase susceptibility to MPC?

*We examine all four C-MAPSS subsets and three architecture families (MLP, 1D-CNN, LSTM/GRU). We do not assume in advance which subsets or architectures are most vulnerable.*

**RQ3 — Detection:**  
Can MPC be detected using only training and validation outputs, without the test set?

*We propose a composite indicator — Prediction Dispersion Ratio (PDR), coefficient of determination (R²), and Constant-Baseline Ratio (CBR) — and test whether it identifies MPC without test-set access.*

**RQ4 — Prevention:**  
What is the minimum-cost intervention that reliably prevents MPC across datasets and architectures?

*We compare protocol changes (per-seed split, early-stopping warmup), architecture changes (GRU substitution, forget gate clamping), and loss function changes (MAE, auxiliary losses), evaluated on both MPC rate and RMSE non-inferiority.*

---

## 6. Contributions

This paper makes four contributions.

**1. Characterization of MPC.**  
We provide an operational definition of MPC that distinguishes it from conditional-mean compression and gradient pathology. The definition uses three measurable indicators (PDR, R², CBR) and is calibrated on held-out conditions to avoid circular reasoning.

**2. Mechanism identification.**  
We trace MPC to a specific structural cause: the LSTM forget gate approaching zero under adverse protocol conditions, blocking gradient flow through the cell state. GRU, which lacks a separate cell state, does not exhibit MPC under the same conditions.

**3. Prevention taxonomy.**  
We compare seven interventions across two axes: MPC reduction rate and RMSE non-inferiority. The taxonomy provides a decision framework for practitioners selecting a prevention strategy.

**4. External protocol audit.**  
We replicate five published training protocols from EAAI and Expert Systems with Applications and measure MPC rates across 10 seeds per protocol. This audit provides direct evidence that MPC-inducing conditions exist in published work, not only in controlled experiments.

---

## 7. Scope and Limitations

This study is limited to C-MAPSS (FD001–FD004). C-MAPSS is a thermodynamic simulator. It produces synthetic degradation trajectories with controlled fault modes and operating conditions. Real turbofan sensor data is more heterogeneous and MPC is less likely to occur in real-world deployments.

The practical relevance of this work is methodological, not operational. The findings concern how RUL research is evaluated and compared, not how deployed systems behave.

MPC in this study is specific to LSTM architectures. GRU and feedforward networks did not exhibit MPC under the tested conditions. We do not claim that MPC is a universal risk in deep learning regression.

---

## 8. Paper Structure

Section 2 reviews prior work and establishes the research gap in detail. Section 3 defines MPC operationally and describes the experimental framework. Sections 4 and 5 present experimental results: mechanism and generalization (Phases 0–3B), followed by prevention and retrospective detection (Phase 4). Section 6 presents the external protocol audit (Phase 5). Section 7 discusses practical implications and limitations. Section 8 concludes.

---

## Reference Placeholders

- [P1] Dohmatob et al. (2024) — model collapse in generative models
- [P2] Al-Selwi et al. (2023) — LSTM vanishing gradient on C-MAPSS
- [P6] Vabalas et al. (2019) — validation split bias
- [P7] Lee & Chen (2025) — systematic mean bias in regression
- [P14] Prechelt (1998) — early stopping
- [P19] Bruna et al. (2015) — regression-to-the-mean
- [P20] Mathieu et al. (2016) — blurry predictions under MSE
- [P21] Huang (2026) — mean collapse in multimodal regression
- [P23] Musgrave et al. (2020) — metric learning reality check
- [P24] Lučić et al. (2018) — GANs created equal
- [C-MAPSS] Saxena & Goebel (2008) — C-MAPSS dataset
- [BMAD] [prior work, anonymized for review]
