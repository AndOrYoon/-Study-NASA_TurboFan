# Discussion
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression

**작성일:** 2026-09-21  
**버전:** Draft v0.1  
**관련 파일:** `Research_Plan.md` §8, `Essence_Research.md` §7–8, `Results_Mechanism.md`, `Results_Prevention.md`, `External_Audit.md`

---

> **작성 방침:** 단순 문장, 능동태. 이론적 함의는 선행 연구 레퍼런스로 탄탄하게. 실용적 권고는 수치 기반으로.

---


#### §8.1.1 MSE 평균 예측 위계 내 MPC 위치 — Bruna [P19], Mathieu [P20], Huang [P21]의 MSE 이론 위에, Li et al. [P18]의 "MSE landscape에서 marginal mean이 global minimizer"라는 증명으로 MPC 안정성의 이론적 근거 제시. CMC(Lee & Chen [P7]) → conditional-mean blurring → MPC(marginal mean)로 이어지는 실패 위계 도입.
#### §8.1.2 LSTM forget gate 포화: 구조적 취약성 — CEC 원리(Hochreiter 원래 설계)와 forget gate 추가(Gers 등) 관계. fg_clamp 실패(ε^T ≈ 0)와 V2_fg1 성공(1^T = 1)의 대비가 이론을 직접 확증. Papyan et al. [P17]의 Neural Collapse(분류)와 MPC(회귀)의 병렬 구조로 이론적 깊이 추가.
#### §8.1.3 Validation Composition과 Loss Landscape — Prechelt [P14]의 early stopping 전제 위반 사례로 MPC 제시. Vabalas [P6]의 고정 split 편향과 MPC의 관계. Miseta [P12], Mahsereci [P13]의 대안 early stopping 제안 맥락 위에 PDR 소급 진단 위치.
#### §8.1.4 PHM 벤치마크 재현성 — Musgrave [P23], Lučić [P24], Ferrari Dacrema [P22] 계보 확장. "잘못 최적화된 baseline"과 "수렴했지만 degenerate solution인 baseline"의 차이가 핵심 이론적 기여.
#### §8.1.5 실무 권고 — 1줄 코드(bias_init=train_mean) + PDR 소급 진단 루틴 구체 제시.

## 8. Discussion

### 8.1 Implications for PHM Research

#### 8.1.1 MPC Within the MSE Mean-Prediction Hierarchy

The theoretical connection between MSE loss and mean-directed predictions is well established. Bruna et al. [P19] identified the "regression-to-the-mean problem" in image super-resolution, where MSE point estimates converge to the conditional mean E[y|x]. Mathieu et al. [P20] showed the same for video prediction. Huang [P21] formalized this for multimodal regression, naming the outcome "mean collapse." In all of these cases, mean-directed prediction is a consequence of data geometry — the presence of multiple modes in the target distribution.

MPC is a more severe form of the same failure. A model exhibiting MPC does not predict the conditional mean E[y|x]; it predicts the marginal mean E[y]. All conditioning on the input is discarded. The failure is not a consequence of data geometry alone. It is a consequence of a specific interaction between data geometry, architecture, and training protocol.

Li et al. [P18] proved that under MSE loss with unconstrained features, the global optimum corresponds to the class-mean predictions. Their theoretical result implies that a constant predictor equal to the training-set mean is a stable fixed point of the MSE loss landscape. When the optimizer reaches this fixed point before early stopping fires, it cannot escape. This is the theoretical basis for MPC stability: the trivial solution is not a random attractor but the analytically provable minimizer of MSE on a homogeneous dataset.

The MPC taxonomy thus extends the existing hierarchy: well-trained models exhibit Conditional-Mean Compression (CMC; [P7]), which is a statistical property of regression at distribution extremes. MPC precedes CMC: the model never acquires the ability to predict conditional means because it is captured at the marginal mean during optimization.

Lee and Chen [P7] characterized CMC as a post-training bias. MPC is a pre-training failure — the model was never successfully trained. These are related but distinct points in the failure taxonomy of regression models.

#### 8.1.2 LSTM Forget Gate Saturation: Structural Vulnerability

The original LSTM was designed to solve the vanishing gradient problem through the Constant Error Carousel (CEC) principle: by setting the internal error flow to a constant, the cell state can carry information across arbitrarily long sequences without gradient decay. The forget gate — introduced later as a practical modification — breaks the CEC guarantee. When f → 0, the cell state gradient decays as f^T over T timesteps. For T = 30 (the window used here), even f = 0.05 gives f^30 ≈ 10⁻^20 — numerically zero.

This finding has a precise implication: clamping the forget gate to a small ε does not prevent MPC. The Phase 3B experiment confirms this: fg_clamp(ε = 0.001, 0.01, 0.05) all show 60% MPC — identical to the unclamped baseline. Only V2_fg1 (f = 1, restoring CEC) achieves 0% MPC. The theoretical condition that eliminates collapse is exact: f = 1, not f ≥ ε.

Al-Selwi et al. [P2] studied vanishing gradients in LSTMs on C-MAPSS and characterized them as a slow-convergence failure — training progresses but too slowly. MPC is the dual failure: training is fast but converges to the wrong solution. Gradient magnitudes are not too small to drive learning; they are sufficient to reach the trivial attractor and insufficient to escape it. These failures share the vanishing BPTT pathway but differ in which phase of training they manifest.

Papyan et al. [P17] documented Neural Collapse (NC) in deep classification networks: during the terminal training phase, last-layer features collapse to class means. NC is a normal, benign phenomenon in classification. MPC is the pathological analog in regression. NC occurs after successful learning; MPC occurs instead of learning. In both, the model's representations collapse to a low-dimensional subspace — in MPC, a zero-dimensional one.

GRU has no separate cell state. Its gradient pathway does not exhibit the T-step multiplicative decay that enables MPC. Under identical trigger conditions (A1+B2+C2_clip125 on FD003), GRU achieves 0% MPC across 10 seeds. This architectural difference provides a clean experimental dissociation: GRU and LSTM differ in exactly one structural property (cell state with forget gate), and they differ in exactly one outcome (MPC versus no MPC).

#### 8.1.3 Validation Composition as a Loss Landscape Shaping Force

Prechelt [P14] established that early stopping should halt training when validation loss improvement stalls. This prescription assumes that validation loss is a reliable proxy for generalization. MPC reveals a systematic failure of this assumption.

When the validation split is fixed across training seeds, the held-out engine set has a fixed RUL distribution. For FD003, the test RUL mean is approximately 87 cycles. A constant predictor outputting 87 cycles achieves low validation MSE on this fixed set — lower than a partially trained model whose predictions are scattered around the mean. Early stopping selects the constant predictor as the "best" checkpoint.

Vabalas et al. [P6] showed that fixed K-fold splits produce biased performance estimates on small datasets due to unrepresentative fold composition. MPC is a more severe manifestation of the same problem. The fixed validation set does not merely bias the estimate; it actively selects a degenerate solution as the termination point.

This has implications for early stopping theory. Miseta et al. [P12] and Mahsereci et al. [P13] proposed alternative early stopping criteria (correlation-based and gradient-based, respectively) precisely because standard validation-loss stopping has unresolved weaknesses. Our data provide a concrete failure case: standard early stopping under fixed validation on homogeneous data selects the constant predictor. The PDR diagnostic (Phase 4) provides a post-hoc check that catches this failure after it occurs.

The interaction between B2 (ES from epoch 0) and A1 (fixed split) is the necessary condition for MPC, not either factor alone. A1 without B2 produces 0% MPC (the model trains long enough to recover). B2 without A1 reduces MPC to 30% (per-seed splits reduce the exploitability of the fixed validation set). This interaction is a protocol-level analog of a multicollinear cause: the individual factors are insufficient, but their combination is catastrophic.

#### 8.1.4 Benchmark Reproducibility in PHM

Musgrave et al. [P23], Lučić et al. [P24], and Ferrari Dacrema et al. [P22] documented that claimed improvements in metric learning, GAN evaluation, and recommendation systems disappeared under consistent experimental conditions. Their diagnosis was undertrained baselines — models that needed more hyperparameter optimization. The remedy in each case was more tuning.

MPC differs from undertrained baselines in one critical respect: the baseline converges. Convergence criteria (early stopping, loss stabilization) are met. The baseline cannot be "fixed" by more tuning within the same protocol; the protocol must be corrected. This is why MPC is more difficult to detect and why it causes more persistent inflation of performance claims.

The Δ_inflation metric quantifies this distortion. For the motivating case, Δ_inflation = 0.80: 80 percentage points of the claimed improvement disappear when the baseline is corrected. Under the five external audit protocols (A1–A5), a hypothetical method achieving RMSE = 13 cycles on FD003 would appear to improve by 70–77% over a collapsed baseline but only by ~8% over a valid one.

This magnitude of distortion has implications for progress assessment in FD003 RUL research. If a non-trivial fraction of published FD003+LSTM baselines are in the MPC state, then the improvement trends reported in the literature partially reflect baseline collapse correction rather than algorithmic advancement. The external audit (Phase 5) shows 60–90% MPC rates across 5 independently selected published protocols. This is not a case study; it is a systematic pattern.

#### 8.1.5 Guidance for PHM Practitioners

Five interventions eliminate MPC with no cost to normal-seed RMSE and negligible implementation overhead. We order them by implementation effort.

**Minimum-cost options (single change, no performance impact):**

1. *Output bias initialization to training-set mean.* Setting the final linear layer's bias to the training-set mean RUL eliminates MPC (0%) with RMSE = 12.94 ± 0.48. Implementation: `model.fc[-1].bias.data.fill_(c_train)`. One line of code.

2. *Early-stopping warmup.* Adding MIN_EPOCHS = 30 before early stopping fires eliminates MPC (0%) with RMSE = 12.42 ± 1.04. Implementation: one additional hyperparameter.

**Retrospective audit (no retraining required):**

At any training completion checkpoint, compute:
```python
pdr = np.std(val_predictions) / (np.std(val_targets) + 1e-8)
if pdr < 0.05:
    # MPC detected — retrain with corrected protocol
```

This single check achieves AUROC = 1.00 on both calibration (270 runs) and held-out (330 runs) data. Any existing trained model can be audited retrospectively without accessing the test set.

**For published results:** FD003+LSTM results in the literature should be evaluated against these indicators. A reported FD003 LSTM RMSE above 25 cycles warrants examination of the training protocol and, where possible, the raw validation-set prediction trajectory. The retrospective audit tool requires only the predictions at the early-stopping checkpoint.

---

### 8.2 Limitations

**MPC is LSTM-specific in the tested conditions.** GRU, 1D-CNN, and MLP achieve 0% MPC under the same trigger conditions on FD003. Transformer-based models and attention-enhanced LSTMs were not tested. The BPTT gradient decay mechanism requires a multiplicative cell state pathway; architectures without this structure (GRU, feedforward) are immune. Whether attention mechanisms or gating variants beyond GRU exhibit analogous vulnerabilities is an open question.

**The study is limited to C-MAPSS (FD001–FD004).** C-MAPSS is a thermodynamic simulator producing synthetic degradation trajectories. Real turbofan sensor data is more heterogeneous: multiple operating regimes, sensor noise, fleet diversity. The homogeneous degradation templates in C-MAPSS FD003 create the stable trivial minimum in the MSE landscape. In more heterogeneous data, this minimum may not be stable, and MPC may be less likely. Phase 3 results support this: FD002 and FD004, with 6 operating conditions each, show 0% MPC under trigger conditions that produce 80% MPC on FD003.

**Statistical power for pairwise comparisons is limited.** With n = 10 seeds per condition, the difference between 20% and 30% MPC rates (e.g., between patience=30 and A2_per_seed in Phase 1B) is not statistically distinguishable (Fisher's exact test, p = 0.48). The study provides strong evidence for the 0% vs. non-zero contrast (complete elimination vs. no elimination) but not for ordering among partial-mitigation methods.

**Gradient trajectories were not directly measured.** The forget gate saturation mechanism is inferred from the causal intervention (V2_fg1 → 0% MPC) rather than direct gradient norm logging. The theoretical chain — f → 0 → BPTT decay → trivial solution capture — is consistent with all experimental results, but direct measurement of forget gate activations and gradient norms across collapsed and normal runs would provide additional mechanistic evidence. This is deferred to future work.

**Prevention comparison fairness.** ES warmup (B2 → B3) and per-seed split (A1 → A2) change the protocol condition rather than fixing the model. Table 8 includes them for completeness, but they represent category-level interventions (protocol redesign) while bias_init and MAE loss represent point-level interventions (single parameter or loss change). Practitioners selecting an intervention should consider which category is feasible in their workflow.

---

## Reference Placeholders

*(번호 체계는 References.md와 동일)*

- [P2] Al-Selwi et al. (2023) — LSTM VGP on C-MAPSS
- [P6] Vabalas et al. (2019) — validation split bias
- [P7] Lee & Chen (2025) — CMC, systematic mean bias
- [P12] Miseta et al. (2023) — correlation-based early stopping
- [P13] Mahsereci et al. (2017) — early stopping without validation
- [P14] Prechelt (1998) — early stopping fundamentals
- [P17] Papyan et al. (2020) — Neural Collapse in classification
- [P18] Li et al. (2022) — MSE landscape, mean prediction optimality
- [P19] Bruna et al. (2015) — regression-to-the-mean
- [P20] Mathieu et al. (2016) — blurry predictions under MSE
- [P21] Huang (2026) — mean collapse in multimodal regression (arXiv preprint)
- [P22] Ferrari Dacrema et al. (2019) — recommender systems reproducibility
- [P23] Musgrave et al. (2020) — metric learning reproducibility
- [P24] Lučić et al. (2018) — GANs created equal
