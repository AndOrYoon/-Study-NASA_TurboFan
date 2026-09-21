# Results: Prevention and Retrospective Detection (Phase 4)
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression

**작성일:** 2026-09-21  
**버전:** Draft v0.1  
**관련 파일:** `Research_Plan.md` §6 (Phase 4), `Results_Mechanism.md` (Phase 1A/1B source data)

---

> **작성 방침:** 단순 문장, 능동태. 표로 비교 우선. Prevention taxonomy에서 비용 축이 중요.

---
#### Prevention taxonomy: 5가지 zero-MPC 개입 — bias_init=train_mean, MAE loss, ES warmup(MIN=30), V2_fg1, GRU. 비용 모두 최소.
#### Retrospective detection: PDR < 0.05가 calibration·held-out 전부 sensitivity=1.00, specificity=1.00. Test set 접근 불필요.


## 6. Results: Prevention and Retrospective Detection

### 6.1 Prevention Taxonomy

Phase 1A and 1B identify seven interventions that reduce or eliminate MPC. We evaluate each on two axes: MPC rate and RMSE non-inferiority relative to the corrected-protocol baseline (RMSE = 14.19 ± 0.61 under A2_per_seed+B3_warmup).

A non-inferior result is defined as RMSE within 10% of the corrected-protocol baseline mean (i.e., RMSE ≤ 15.6 cycles on FD003).

**Table 8. Prevention taxonomy: MPC rate, RMSE, and implementation cost.**

| Intervention | Type | MPC rate | RMSE (mean ± std) | Non-inferior? | Cost |
|-------------|------|----------|--------------------|:---:|------|
| Output bias → train mean | Initialization | **0.00** | 12.94 ± 0.48 | ✓ | Minimal — one-line change |
| Output bias → random calib. | Initialization | **0.00** | 12.69 ± 0.74 | ✓ | Minimal |
| MAE loss | Loss function | **0.00** | 13.43 ± 0.60 | ✓ | Low — loss swap only |
| Warmup (MIN_EPOCHS=30) | Protocol | **0.00** | 12.42 ± 1.04 | ✓ | Low — one hyperparameter |
| Per-seed random split (A2) | Protocol | 0.30* | varies | — | Low |
| Forget gate = 1 (V2_fg1) | Architecture | **0.00** | 12.96 ± 0.95 | ✓ | Minimal — clamp one parameter |
| GRU substitution | Architecture | **0.00** | 12.27 ± 0.67 | ✓ | Moderate — model change |
| Patience = 30 | Protocol | 0.20 | 18.38 ± 12.21 | ✗ | Low (high RMSE std) |

*Per-seed split alone with B2+C2_clip125 reduces but does not eliminate MPC. Combined with warmup: 0%.

Five interventions achieve 0% MPC with RMSE non-inferior to the corrected baseline. Patience extension to 30 reduces collapse rate to 20% but does not eliminate it, and the surviving collapsed runs inflate the mean RMSE (18.38) and std (12.21) beyond the non-inferiority threshold.

**Recommendation.** The minimum-cost, zero-overhead intervention is output bias initialization to the training-set mean. This requires changing one value in the model constructor. Early-stopping warmup (MIN_EPOCHS=30) is equally effective and does not alter the model architecture. Both should be adopted as default practice.

No intervention requires additional data, longer total training, or a new architecture. The prevention cost is near zero.

---

### 6.2 Retrospective Audit Tool

Phase 1A/1B collapse labels were assigned by blind adjudication using validation-set outputs only, before examining test-set RMSE. Inter-rater agreement: Cohen's κ = 1.00 for the PDR < 0.05 criterion. We then calibrate and validate a composite detection rule on held-out Phase 3 and 3B conditions.

**Table 9. Retrospective MPC detection performance.**

| Indicator | Threshold | Dataset | Sensitivity | Specificity | Balanced acc. |
|-----------|-----------|---------|:-----------:|:-----------:|:---:|
| PDR | < 0.05 | Calibration | 1.00 | **1.00** | 1.00 |
| PDR | < 0.05 | Held-out | 1.00 | **1.00** | 1.00 |
| R² | ≤ 0 | Calibration | 1.00 | 0.935 | 0.968 |
| R² | ≤ 0 | Held-out | 1.00 | **1.00** | 1.00 |
| RMSE | > 25 cycles | Calibration | 1.00 | 0.687 | 0.844 |
| RMSE | > 25 cycles | Held-out | 1.00 | 0.957 | 0.979 |
| PDR + R² (OR) | either criterion | Calibration | 1.00 | 0.935 | 0.968 |
| PDR + R² (OR) | either criterion | Held-out | 1.00 | **1.00** | 1.00 |

The PDR < 0.05 threshold achieves perfect sensitivity and specificity on both calibration and held-out data. In calibration, 40 collapsed runs were correctly identified with 0 false negatives, 230 non-collapsed runs were correctly cleared with 0 false positives.

R² ≤ 0 is nearly as reliable: 15 false positives in calibration (models that performed similarly to the constant predictor but were not fully collapsed), but 0 false positives on the held-out validation set.

RMSE > 25 cycles alone is not sufficient: 72 false positives in calibration (unclipped runs with legitimately high RMSE).

**All three indicators can be computed from validation-set outputs at training completion. Test-set access is not required.**

The composite rule PDR < 0.05 OR R² ≤ 0 achieves perfect held-out performance and provides redundancy if PDR is not logged. In practice, PDR alone is sufficient.

**Computational overhead.** Computing PDR and R² requires storing one vector of N_val predictions at each early-stopping checkpoint — the same storage already needed for early stopping. The additional computation is negligible (two scalar operations per checkpoint evaluation).

---

## Reference Placeholders

*(번호 체계는 References.md와 동일)*

- [P25] Cohen (1960) — Cohen's κ inter-rater agreement
- [P26] Schulz et al. (2010) — blind adjudication protocol
- [P27] Murphy (2012) — constant predictor as regression baseline (CBR definition)
