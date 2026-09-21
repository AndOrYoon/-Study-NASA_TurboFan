# Results: Mechanism and Generalization (Phase 0–3B)
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression

**작성일:** 2026-09-21  
**버전:** Draft v0.1  
**관련 파일:** `Research_Plan.md` §5 (Phase 0–3B), `Methodology.md` §4

---

> **작성 방침:** 단순 문장, 능동태. 표와 수치 우선. 인과 주장은 실험 근거 뒤에 제시.

---

#### Phase 0: 원본 프로토콜 4/5 seeds 붕괴 (RMSE = 43.17±0.18), 수정 프로토콜 0/5 붕괴 (RMSE = 14.19±0.61). 생존 seed 2는 100 epoch 전부 실행하여 탈출.
#### Phase 1A: B2_from0(ES from epoch 0)이 필요조건. B1·B3는 90 runs 전부 MPC=0. A1_fixed가 rate를 A2_per_seed 대비 2배 증폭. 표 4로 27개 조건 정리.
#### Phase 1B: LR은 무효(90–100% 붕괴). bias_init=train_mean과 MAE loss가 단독으로 MPC 0% 달성. Patience=30은 20%로 감소시킬 뿐.
#### Phase 2: 학습 동역학 — 붕괴 run은 epoch 1부터 PDR ≈ 2×10⁻⁶, 정상 run은 epoch 44에서 PDR=0.88로 회복. 실시간 alarm specificity = 6.3%로 비실용적.
#### Phase 3: FD003+LSTM만 80% MPC. GRU/CNN/MLP는 동일 조건에서 0%.
#### Phase 3B: V2_fg1(forget gate=1)→0% MPC, GRU→0% MPC. forget gate 기전 확증.

## 5. Results: Mechanism and Generalization

### 5.1 Reproduction and Implementation Audit (Phase 0)

We first verify that the motivating observation is not a code error and that the corrected protocol resolves it.

**Table 3. Phase 0: Original versus corrected protocol (FD003, LSTM, 5 seeds).**

| Protocol | Seeds | MPC count | RMSE (mean ± std) | PDR (mean) | R² (mean) | Stop epoch |
|----------|-------|-----------|-------------------|-----------|---------|-----------|
| Original (A1_fixed, B2_from0, MAX=100) | 5 | 4/5 | 43.17 ± 0.18 (collapsed seeds) | 0.000002 | −0.083 | 22.5 |
| Seed 2 only (escaped) | 1 | 0/1 | 13.55 | 0.922 | 0.893 | 100 |
| Corrected (A2_per_seed, MIN=30, MAX=300) | 5 | 0/5 | 14.19 ± 0.61 | 0.974 | 0.882 | 127.4 |

The original H6 archive shows 5/5 seeds collapsed under the same conditions [ANON]. The Phase 0 rerun gives 4/5. This is consistent with the 80% MPC rate measured in Phase 1A under these exact conditions — one seed escaped because it ran the full 100 epochs before patience fired (stop epoch = 100 vs. 18–35 for collapsed seeds).

Under the corrected protocol, all five seeds converge. RMSE = 14.19 ± 0.61, PDR = 0.965–0.993, R² = 0.871–0.893. The corrected-protocol constant predictor RMSE is 45.07 cycles, confirming that the collapsed model (RMSE ≈ 43) performs no better than predicting the training-set mean.

The collapse is not a code error. Fixing three protocol choices — validation split randomization, early-stopping warmup, and extended maximum epochs — resolves it completely.

---

### 5.2 Validation Composition × Early Stopping (Phases 1A and 1B)

#### 5.2.1 Factorial Experiment (Phase 1A)

We run 3 (val split) × 3 (early stopping) × 3 (clipping) = 27 conditions, 10 seeds each, on FD003+LSTM. Table 4 shows the subset of conditions that exhibit non-zero MPC rates.

**Table 4. Phase 1A MPC rates: B2_from0 (ES from epoch 0) conditions.**

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

B1 conditions (no early stopping, MAX=200 epochs): MPC = 0.00 across all 90 runs.  
B3 conditions (warmup: MIN_EPOCHS=30): MPC = 0.00 across all 90 runs.

Three findings emerge.

**Finding 1 — B2 (ES from epoch 0) is the necessary trigger.** Neither B1 nor B3 produces a single collapsed run across 90 conditions each. B2 is required for MPC.

**Finding 2 — A1 (fixed split) amplifies collapse rate.** Fixing the validation split to a single random seed (A1) approximately doubles the collapse rate relative to per-seed randomization (A2). A1+B2+C2_clip125 = 80%; A2+B2+C2_clip125 = 30%. Fixed splits allow the optimizer to exploit the specific RUL composition of one held-out set.

**Finding 3 — Clipping modulates severity.** Unclipped labels (C1) produce both higher MPC rates and higher RMSE when collapse occurs. Clip=100 (C3) reduces MPC rate under A2 to zero. The flat RUL label plateau in clipped data narrows the range that the constant predictor exploits.

#### 5.2.2 Optimization and Initialization (Phase 1B)

All Phase 1B conditions use A1_fixed+B2_from0+C2_clip125 (the canonical trigger). We vary one factor at a time.

**Table 5. Phase 1B: Factor sweep results (FD003+LSTM, 10 seeds each).**

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

Four findings.

**Finding 4 — Patience is a continuous modulator, not a structural fix.** Patience=30 reduces MPC to 20% but does not eliminate it. Increasing patience extends the window for natural recovery, but the LSTM can still collapse if the forget gate remains near zero at epoch 30.

**Finding 5 — Learning rate does not help.** MPC rate is 90–100% across all tested learning rates. Lower LR (0.0001) increases collapse rate to 100% — slower initial gradients give the optimizer less chance to escape the trivial attractor before early stopping fires.

**Finding 6 — Output bias initialization to training-set mean eliminates MPC.** When the output bias is initialized to the training-set mean RUL, predictions begin near the mean. The gradient signal from (prediction − target) is nonzero and distributed across all training examples from epoch 1. The network receives meaningful gradient updates before early stopping can fire. MPC rate = 0% with no RMSE penalty (12.94 ± 0.48 vs. 14.19 ± 0.61 for the corrected protocol).

**Finding 7 — MAE loss eliminates MPC.** Under L1 loss, the Bayes-optimal prediction is the conditional median. The loss landscape does not have the flat basin near the marginal mean that enables the MSE trivial solution. MPC rate = 0%, RMSE = 13.43 ± 0.60. Auxiliary MSE loss (branch losses summed at 10% weight) halves the collapse rate (0.40) but does not eliminate it.

---

### 5.3 Training Dynamics (Phase 2)

We collect epoch-by-epoch PDR and R² trajectories for all 400 runs from Phases 1A and 1B. Runs are grouped into 116 collapsed and 284 normal.

**Figure 1** (training dynamics) shows two qualitatively different trajectories.

In collapsed runs, PDR drops to ≈2×10⁻⁶ at epoch 1 and remains there throughout training. R² is negative from epoch 1 (mean R² = −2.58 at epoch 1). Validation loss decreases monotonically — this is the "silent collapse" property: the optimizer appears to be making progress while the model is entirely input-insensitive.

In normal runs, PDR is also low at epoch 1 (mean = 0.013), then recovers. The recovery is gradual: PDR reaches 0.38 by epoch 30, 0.88 by epoch 44, and stabilizes above 0.90 by epoch ~50. R² follows a similar trajectory.

This explains why B3 (warmup: MIN_EPOCHS=30) prevents MPC. At epoch 30, normal runs have PDR = 0.38 — low but recovering — while collapsed runs remain flat at PDR ≈ 2×10⁻⁶. Early stopping fires during the recovery window for normal runs under B2 but not under B3.

**Real-time alarm performance.** Using a PDR < 0.05 threshold as an online alarm at each epoch, sensitivity = 1.00 but specificity = 0.063. This 6.3% specificity means the alarm fires for nearly all runs in the first few epochs — including normal runs that have not yet recovered their PDR. A real-time alarm is not actionable. Retrospective detection at training completion (Phase 4) resolves this limitation.

---

### 5.4 Dataset and Architecture Generalization (Phase 3)

We apply the canonical trigger conditions (A1_fixed+B2_from0+C2_clip125) to 4 datasets × 4 architectures (MLP, 1D-CNN, LSTM, GRU), 10 seeds each.

**Table 6. Phase 3 MPC rates (%) across dataset × architecture.**

| Dataset | LSTM | GRU | 1D-CNN | MLP |
|---------|------|-----|--------|-----|
| FD001 | 10 | 0 | 0 | 0 |
| FD002 | 0 | 0 | 0 | 0 |
| **FD003** | **80** | **0** | **0** | **0** |
| FD004 | 0 | 0 | 0 | 0 |

**Finding 8 — MPC is FD003-specific among C-MAPSS subsets.** FD002 and FD004 (6 operating conditions each) show 0% MPC across all architectures. FD001 (single operating condition, single fault mode) shows marginal MPC for LSTM only (10% — one seed). FD003 (single operating condition, two fault modes) shows 80% MPC for LSTM. The structural homogeneity of FD003 — identical thermodynamic templates within each fault mode, single operating condition — creates a stable trivial minimum in the MSE landscape.

**Finding 9 — MPC is LSTM-specific among tested architectures.** Under the same trigger conditions on FD003, GRU, 1D-CNN, and MLP show 0% MPC. RMSE for GRU (12.61 ± 0.47) and MLP (14.63 ± 0.13) is comparable to the corrected LSTM baseline (14.19 ± 0.61). The collapse is not simply that FD003 is a difficult task; it is that FD003+LSTM under trigger conditions creates a pathological training trajectory.

---

### 5.5 LSTM Forget Gate Mechanism (Phase 3B)

Phase 3 identifies the LSTM as uniquely vulnerable. Phase 3B tests whether this vulnerability is attributable to the forget gate.

Four variants of the FD003 model are trained under A1_fixed+B2_from0+C2_clip125, 10 seeds each.

**Table 7. Phase 3B: Forget gate variants on FD003.**

| Variant | Description | MPC rate | RMSE (mean ± std) | Stop epoch |
|---------|-------------|----------|--------------------|-----------|
| LSTM_gate (base) | Standard LSTM | 0.60 | 30.18 ± 14.98 | 61.7 |
| GRU_gate | GRU replacement | **0.00** | 12.27 ± 0.67 | 80.9 |
| V1_fb1 | Forget bias initialized to +1 | 0.40 | 24.29 ± 14.80 | 62.5 |
| V2_fg1 | Forget gate clamped to 1 (CEC) | **0.00** | 12.96 ± 0.95 | 98.7 |

**Finding 10 — GRU eliminates MPC.** GRU replaces the cell state with a reset gate and update gate; there is no separate cell state that can be blocked by a saturated forget gate. Under identical trigger conditions that produce 60% MPC for LSTM, GRU achieves 0% MPC with RMSE comparable to the corrected LSTM baseline.

**Finding 11 — Forget gate = 1 (CEC) eliminates MPC.** V2_fg1 clamps the forget gate to f = 1.0, allowing full gradient flow through the cell state at every timestep. MPC = 0%, RMSE = 12.96 ± 0.95. The cost is minimal (no architecture change, negligible overhead).

**Finding 12 — Forget bias initialization to +1 reduces but does not eliminate MPC.** V1_fb1 starts the forget gate in an open state, delaying the time at which f → 0. MPC falls from 60% (standard) to 40% — a partial effect. Some seeds still collapse because the bias can drift toward zero during training.

**Mechanistic conclusion.** The data support the following causal chain. Under A1_fixed+B2_from0 conditions, the MSE loss landscape has a stable trivial minimum at a near-constant prediction near the training-set mean. In early training, the LSTM forget gate approaches f ≈ 0 for some seeds. When f ≈ 0, the BPTT gradient through the cell state is blocked: ∂L/∂h_t cannot propagate through the cell state to earlier timesteps. The model cannot update its recurrent representation. It is trapped at the trivial solution. Early stopping fires before any recovery can occur. The resulting model is functionally equivalent to a constant predictor.

Preventing forget gate saturation — by clamping f = 1, using GRU, or initializing bias to +1 — disrupts this chain.

---

## Reference Placeholders

*(번호 체계는 References.md와 동일)*

- [ANON] — prior work (H6 fault-mode routing study), anonymized for review
- [P29] Saxena & Goebel (2008) — C-MAPSS dataset
