# Related Work
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression

**작성일:** 2026-09-21  
**버전:** Draft v0.1  
**관련 파일:** `Literature_Review.md` (상세 선행 연구), `Research_Plan.md` §2, `Introduction.md`

---

> **작성 방침:** 단순 문장, 능동태. 각 주제 단락 말미에 연구 갭을 명시. Related Work는 Introduction §4의 갭을 증거 기반으로 구체화하는 역할.

---

## 2. Related Work

### 2.1 Deep Learning for RUL Prediction on C-MAPSS

The C-MAPSS dataset [P29] has become the standard benchmark for deep learning–based RUL prediction. Li et al. [P16] introduced a deep convolutional neural network (DCNN) that extracted features directly from raw sensor windows and achieved state-of-the-art RMSE across all four C-MAPSS subsets. This work established the template that most subsequent studies follow: a deep model processes sliding windows of sensor readings and predicts a piecewise-linear RUL target.

Many architectures have been proposed since. Ellefsen et al. [P3] applied semi-supervised pretraining to address label scarcity and reported improved convergence stability. Elsherif et al. [P5] combined a convolutional autoencoder with an attention-enhanced LSTM (CAELSTM) and reported FD003 RMSE = 13.40. Zhuang et al. [P15] incorporated uncertainty quantification into a Bayesian deep learning framework for maintenance decision support.

Despite this volume of work, a consistent problem persists. Reported baseline RMSE values for FD003 span roughly 13 to 43 across papers. Training protocols are rarely standardized. Validation split strategies, early stopping conditions, RUL clipping thresholds, and feature selection differ across papers without explanation. This variation makes fair comparison impossible.

> **Gap:** CMAPSS-based RUL studies lack standardized training protocols. The contribution of protocol choices — rather than model architecture — to reported performance differences has not been quantified.

---

### 2.2 MSE Loss and Mean-Prediction Tendency

MSE is the dominant loss function in RUL regression. Its use has a known theoretical consequence: the Bayes-optimal predictor under MSE is the conditional mean E[y|x]. When the conditional distribution of y given x is multimodal, this conditional mean lies between modes. No single mode is predicted well.

Bruna et al. [P19] named this the "regression-to-the-mean problem" in the context of image super-resolution. They showed that MSE-based point estimates converge to the conditional mean, producing blurry outputs when the true conditional distribution is multimodal. Mathieu et al. [P20] demonstrated the same in video prediction and argued that MSE produces "inherently blurry predictions." Both studies motivated alternatives to MSE — adversarial losses, gradient difference losses — for structured prediction tasks.

Huang [P21] extended this argument explicitly to regression. He formalized that under squared loss, an unconstrained regressor converges to the conditional mean, and when K > 1 modes exist, this mean is distant from all modes. He named this outcome "mean collapse" and used it as a worst-case baseline in experiments (MSE = 1.33 versus oracle = 0.09). Note that Huang [P21] is an arXiv preprint as of this writing and has not undergone peer review. We treat it as theoretical background only.

These works establish the theoretical basis for mean-directed prediction. Each component of the MPC definition (Section 3.1) maps to one or more of these prior findings:

- *"Reduced dispersion relative to the target distribution"* — Mathieu et al. [P20]: blurry predictions under MSE reduce the effective range of outputs.
- *"Concentrate around the conditional mean"* — Bruna et al. [P19] and Huang [P21]: squared-loss optimization converges to E[y|x].
- *"Suppressing input-dependent variations"* — Huang [P21]: multi-modal inputs lead the optimizer to a mean that is equidistant from all modes and thus insensitive to which mode is active.
- *"Systematically underrepresenting extreme values"* — Lee and Chen [P7]: trained regression models exhibit systematic bias toward the mean at distribution extremes, even after successful convergence.

However, all prior work treats mean-directed prediction as a consequence of data geometry alone — the presence of multiple modes or extreme values in the target distribution. None examines how training protocol conditions can cause an optimizer to converge to the marginal mean E[y] rather than any conditional mean E[y|x]. In MPC as documented in this study, the model outputs a single constant for all inputs, with prediction standard deviation below 0.0002 cycles. This is more severe than the blurring described in prior work: the model discards input information entirely, rather than blending modes.

> **Gap:** Prior work on MSE-induced mean prediction focuses on data geometry. The role of training protocol conditions — fixed validation splits, early stopping without warmup — in causing models to converge to the marginal mean E[y] rather than any conditional mean E[y|x] has not been studied.

---

### 2.3 Training Instability in LSTM-Based RUL Regression

Al-Selwi et al. [P2] studied vanishing gradient pathology (VGP) in LSTM networks applied to C-MAPSS. Their analysis focused on slow convergence: gradients decay over long sequences, preventing the model from learning long-range dependencies efficiently.

MPC is a different failure. The LSTM in MPC does not converge slowly. It converges quickly — to a degenerate solution. The forget gate approaches zero in early training. This blocks gradient flow through the cell state. Early stopping then terminates training while the model is still at this trivial solution. The result is a model that predicts a near-constant value.

Prior work on LSTM instability characterizes gradients that are too small to drive learning. MPC involves gradients that are sufficient to drive the model toward a local minimum, but the wrong one. These are distinct failure modes.

Lee and Chen [P7] described a related but different phenomenon: systematic mean bias in trained regression models. They showed that well-trained regression models tend to underestimate high values and overestimate low values at distribution extremes — a statistical regression-to-the-mean effect. This occurs after successful training. MPC occurs instead of training. A model exhibiting mean bias still uses its inputs to produce differentiated predictions. An MPC model does not.

Dohmatob et al. [P1] studied "model collapse" in generative models trained iteratively on their own outputs. This is a different mechanism and a different domain. It does not apply to single-run discriminative training.

> **Gap:** No prior study examines the specific failure mode in which an LSTM RUL regressor converges rapidly to a near-constant, input-insensitive prediction due to forget gate dynamics interacting with early stopping under adverse protocol conditions.

---

### 2.4 Validation Design and Early Stopping

Prechelt [P14] established the standard framework for early stopping: monitor validation loss and halt training when improvement stalls. This approach assumes that the validation loss is an accurate proxy for generalization performance.

MPC violates this assumption. When the validation split is fixed across training seeds, the validation set has a fixed RUL distribution. A model predicting the training-set mean can achieve low validation MSE because the mean is close to many validation targets. Early stopping then selects this constant-predictor solution as the "best" checkpoint. The validation loss decreases, but the model has learned nothing.

Vabalas et al. [P6] showed that fixed K-fold cross-validation on small datasets produces biased performance estimates. They recommended held-out test sets over fixed splits for small-sample settings. Their work focused on classification and did not examine how fixed validation splits interact with early stopping to select degenerate solutions on a regression loss surface.

Miseta et al. [P12] proposed an early stopping criterion based on the Pearson correlation between training and validation losses, improving on standard patience-based stopping. Mahsereci et al. [P13] proposed eliminating the validation set entirely and stopping based on gradient statistics. Both works acknowledge that standard early stopping has unresolved weaknesses, but neither identifies MPC as a failure mode or examines the interaction between validation composition and loss landscape geometry.

> **Gap:** The interaction between fixed validation splits and early stopping in selecting trivial solutions on regression loss surfaces has not been studied. The conditions under which this interaction causes functional collapse — rather than merely biased estimates — are unknown.

---

### 2.5 Benchmark Reproducibility Across Domains

The concern about collapsed or undertrained baselines appears across machine learning domains. Three papers are directly relevant to this study.

Musgrave et al. [P23] re-evaluated distance metric learning methods under consistent conditions — the same backbone, the same data splits, the same augmentation policy. They found that claimed improvements of "more than double over four years" disappeared under fair comparison. The gains were artifacts of inconsistent experimental conditions, not algorithmic advances.

Lučić et al. [P24] applied the same principle to generative adversarial networks. Under sufficient hyperparameter search, the performance differences between GAN variants were not statistically significant. They concluded that "the measurement protocol determines the conclusion."

Ferrari Dacrema et al. [P22] applied a reproducibility audit to 18 neural recommendation algorithms. Most could not be reproduced. Of those that could, the majority did not outperform simple non-neural baselines.

These three studies establish that inconsistent experimental conditions routinely produce inflated performance claims. Our work is in this lineage. However, these studies diagnosed *undertrained baselines* — baselines that needed more hyperparameter search or computational budget. MPC is a more extreme condition. The baseline converges, but to a state in which it outputs a constant regardless of input. Additional hyperparameter search does not help. The protocol itself must be corrected.

No study in this lineage has been conducted in PHM or RUL prediction. No study has examined whether a specific combination of protocol conditions — fixed validation split, early stopping without warmup, MSE loss — can produce functional collapse in a regression model.

> **Gap:** Protocol-induced performance inflation has been documented in metric learning, GAN evaluation, and recommender systems. It has not been studied in PHM/RUL prediction. More importantly, the extreme form of this inflation — in which the baseline model is functionally collapsed rather than merely undertrained — has not been documented or analyzed in any domain.

---

### 2.6 Synthetic Data Homogeneity

Prior work on synthetic datasets in PHM focuses on limited representativeness. Chao et al. [P4] motivated the N-CMAPSS dataset by arguing that CMAPSS has "limited representativeness of real operating conditions." Das et al. [P8] studied how uncertainty in the physical simulator propagates to DNN predictions. Both works treat synthetic data as a proxy for real data, with the concern that the proxy is imperfect.

This study identifies a different problem. C-MAPSS FD003 is not merely a limited proxy for real data. Its homogeneity creates a specific vulnerability. FD003 has a single operating condition and two fault modes. Within each fault mode, engine degradation trajectories follow the same thermodynamic template. The within-mode variation in sensor windows is small. As a result, the MSE loss landscape has a broad, stable minimum at the marginal mean of the RUL distribution. This minimum is easily accessible under adverse training conditions.

The vulnerability is paradoxical. More homogeneous data makes it easier for a model to collapse, not harder.

Tan et al. [P11] found that synthetic benchmark data may cause deep learning models to learn artifacts of the generation process rather than genuine temporal patterns. This supports our finding that the structure of synthetic data — rather than its lack of realism — can shape how optimization behaves.

> **Gap:** Synthetic dataset homogeneity as a driver of training collapse has not been identified or studied. The mechanism by which homogeneous within-mode trajectories create a stable degenerate minimum in the MSE landscape is not documented in prior work.

---

### 2.7 Summary: Research Gaps

The following table maps the six areas above to specific gaps that this study addresses.

| Gap | Prior literature | What is missing |
|-----|-----------------|-----------------|
| G1 | Li 2018 [P16], Elsherif 2025 [P5] | Protocol-induced variation in baseline RMSE is not quantified |
| G2 | Bruna 2015 [P19], Mathieu 2016 [P20], Huang 2026 [P21] | Marginal-mean collapse (as opposed to conditional-mean blurring) under protocol conditions is not examined |
| G3 | Al-Selwi 2023 [P2], Lee 2025 [P7] | LSTM forget gate → 0 as a mechanism for rapid trivial-solution convergence is not documented |
| G4 | Prechelt 1998 [P14], Vabalas 2019 [P6] | Fixed validation split × early stopping interaction in selecting degenerate solutions is not studied |
| G5 | Musgrave 2020 [P23], Lučić 2018 [P24], Ferrari Dacrema 2019 [P22] | Functional collapse (as opposed to undertrained baselines) in PHM/RUL has not been documented |
| G6 | Chao 2020 [P4], Das 2024 [P8], Tan 2025 [P11] | Synthetic dataset homogeneity as a driver of stable trivial minima is not identified |

Together, these gaps define a phenomenon that is theoretically motivated, practically consequential, and unexamined: a trained RUL regression model that outputs a near-constant prediction because its training protocol selected a degenerate solution, inflating all subsequent comparisons. This paper investigates the conditions, mechanisms, detection, and prevention of this failure.

---

## Reference Placeholders

*(번호 체계는 Literature_Review.md와 동일)*

- [P1] Dohmatob et al. (2024) — model collapse in generative models
- [P2] Al-Selwi et al. (2023) — LSTM vanishing gradient on C-MAPSS
- [P3] Ellefsen et al. (2019) — semi-supervised RUL prediction
- [P4] Chao et al. (2020/2021) — N-CMAPSS, physics-data fusion
- [P5] Elsherif et al. (2025) — CAELSTM, FD003 RMSE=13.40
- [P6] Vabalas et al. (2019) — validation split bias
- [P7] Lee & Chen (2025) — systematic mean bias in regression
- [P8] Das et al. (2024) — uncertainty quantification from synthetic data
- [P11] Tan et al. (2025) — SynTSBench, synthetic data artifacts
- [P12] Miseta et al. (2023) — correlation-based early stopping
- [P13] Mahsereci et al. (2017) — early stopping without validation set
- [P14] Prechelt (1998) — early stopping fundamentals
- [P15] Zhuang et al. (2023) — Bayesian deep learning for PHM
- [P16] Li et al. (2018) — DCNN for RUL, C-MAPSS baseline
- [P19] Bruna et al. (2015) — regression-to-the-mean problem
- [P20] Mathieu et al. (2016) — blurry predictions under MSE
- [P21] Huang (2026) — multimodal regression, mean collapse (arXiv preprint)
- [P22] Ferrari Dacrema et al. (2019) — reproducibility in recommender systems
- [P23] Musgrave et al. (2020) — metric learning reality check
- [P24] Lučić et al. (2018) — GANs created equal
