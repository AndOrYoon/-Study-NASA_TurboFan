# External Protocol Audit (Phase 5)
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression

**작성일:** 2026-09-21  
**버전:** Draft v0.1  
**관련 파일:** `Research_Plan.md` §7, `Methodology.md` §4.2.4 (audit design), `References.md` (A1–A5)

---

> **작성 방침:** 단순 문장, 능동태. 외부 audit이므로 선정 기준·재현 절차 투명하게 기술. Δ_inflation은 FD003 특이적임을 명시.

---

#### A1–A5 (5편 논문 재현): MPC rate 60–90%, 전체 41/50 runs 붕괴(82%).
#### Δ_inflation 공식으로 정량화: 동기 사례([ANON]) Δ_inflation = 0.80 — 65.8% 개선 주장의 80 percentage point가 허구.
#### 외부 5개 프로토콜에서 같은 수준의 왜곡 예상 (Δ_inflation ≈ 0.62–0.69).


## 7. External Protocol Audit

### 7.1 Audit Design and Paper Selection

Phase 1–3B establishes MPC under controlled conditions. Phase 5 asks whether the trigger conditions exist in published work.

We select five published training protocols for replication. Selection criteria: (1) FD003 results reported, (2) LSTM or stacked RNN backbone, (3) training hyperparameters sufficiently specified for replication (patience, learning rate, epoch limit), (4) published in a peer-reviewed venue or widely cited preprint. Criteria were pre-registered before audit data collection.

**Table 10. Audit paper selection (A1–A5).**

| ID | Citation | Venue | Patience | Epoch limit | RUL clipping |
|----|----------|-------|----------|------------|--------------|
| A1 | Meng et al. (2023) [P31] | ESWA | 15 | 200 | clip=125 |
| A2 | Qin et al. (2024) [P32] | EAAI | 20 | 200 | clip=125 |
| A3 | Zheng et al. (2017) [P33] | ICPHM | 10 | 200 | none |
| A4 | [pre-2020 representative pattern] | — | 15 | 200 | none |
| A5 | Elsherif et al. (2025) [P5] | Sci. Rep. | — | 25 fixed | clip=125 |

A4 represents the protocol pattern prevalent before the RUL clipping convention was established: fixed validation split, no warmup, unclipped labels. This matches the A1_fixed+B2_from0+C1_unclipped condition in Phase 1A exactly (MPC = 90%).

Each protocol is replicated with 10 independent random seeds. All other implementation choices (backbone architecture, window size, preprocessing) follow the Phase 0–4 standard to isolate the protocol factor. Replication fidelity is confirmed by RMSE ±15% of the paper's reported value for non-collapsed seeds.

MPC labeling uses the calibrated PDR < 0.05 threshold from Phase 4, applied at training completion. Labels are assigned before computing test-set RMSE.

---

### 7.2 MPC Prevalence in Published Protocols

**Table 11. MPC rates and RMSE across replicated protocols (10 seeds each).**

| Protocol | MPC rate | RMSE (mean ± std) | PDR (mean) | R² (mean) | Stop epoch |
|----------|----------|-------------------|-----------|---------|-----------|
| A1 Meng 2023 [P31] | 0.80 | 35.96 ± 11.83 | 0.199 | 0.075 | 38.7 |
| A2 Qin 2024 [P32] | 0.60 | 30.18 ± 14.64 | 0.400 | 0.280 | 60.3 |
| A3 Zheng 2017 [P33] | 0.90 | 38.61 ± 9.38 | 0.097 | −0.023 | 23.6 |
| A4 [pre-2020] | 0.90 | 56.14 ± 6.85 | 0.149 | −0.864 | 21.6 |
| A5 Elsherif 2025 [P5] | 0.90 | 39.33 ± 7.14 | 0.097 | −0.038 | 25.0 |
| **Overall** | **0.82** | **40.04 ± 14.27** | **0.188** | **−0.114** | **33.8** |

Of 50 total replication runs, 41 collapsed. MPC rate ranges from 60% (A2) to 90% (A3, A4, A5). The aggregate MPC rate is 82%.

A2 has the lowest collapse rate (60%), corresponding to longer patience (20 vs. 10–15 for others). This is consistent with Phase 1B Finding 4: higher patience reduces but does not eliminate MPC.

A5 uses fixed epochs (25) rather than early stopping. Despite the absence of early stopping, 9/10 seeds collapse. At epoch 25, most seeds are still in the early-training phase where the LSTM has not recovered PDR (Phase 2 data shows PDR ≈ 0.10 at epoch 25 for normal runs). Fixed-epoch training terminated during the recovery window produces a collapsed-state model.

For each protocol, the non-collapsed seeds achieve RMSE consistent with the corrected-protocol baseline (13–16 cycles on FD003). These are the seeds that escape MPC — either by chance (different gradient initialization path) or because the random split for that seed happens to yield a validation set that is not exploitable as a constant predictor.

---

### 7.3 Benchmark Distortion Quantification

MPC in the baseline produces inflated performance claims. We quantify this inflation for FD003 using:

$$\Delta_{\text{inflation}} = \left(\frac{E_{\text{collapsed}} - E_{\text{new}}}{E_{\text{collapsed}}}\right) - \left(\frac{E_{\text{valid}} - E_{\text{new}}}{E_{\text{valid}}}\right)$$

where E_collapsed is the baseline RMSE under a collapsed protocol, E_valid is the baseline RMSE under the corrected protocol, and E_new is the proposed method's RMSE under the same protocol as E_collapsed.

Using the motivating observation from [ANON] as a concrete example:

| Quantity | Value |
|---------|-------|
| E_collapsed (original M0, FD003) | 43.23 cycles |
| E_valid (corrected M0, FD003) | 12.97 cycles |
| E_new (M3 attention-gate, original protocol) | 14.78 cycles |
| Apparent improvement (collapsed baseline) | **65.8%** |
| Actual improvement (corrected baseline) | −14.0% (not significant, p_BH = 0.754) |
| Δ_inflation | **0.80** |

The 65.8% improvement claim is entirely a protocol artifact. After correcting the baseline, the proposed method is not statistically different from the corrected baseline. Δ_inflation = 0.80 means 80 percentage points of the apparent improvement are fictitious.

For the five external audit protocols, we compute Δ_inflation using the median collapsed seed RMSE as E_collapsed and the corrected-protocol RMSE (14.19 ± 0.61) as E_valid. A hypothetical proposed method achieving RMSE = 13 cycles would yield the following apparent improvements:

**Table 12. Benchmark distortion by protocol (hypothetical proposed method at RMSE = 13 cycles).**

| Protocol | E_collapsed (median) | E_valid | Apparent improvement | True improvement | Δ_inflation |
|----------|---------------------|---------|---------------------|-----------------|------------|
| A1 Meng 2023 | ~43 | 14.19 | ~70% | ~8% | ~0.62 |
| A2 Qin 2024 | ~43 | 14.19 | ~70% | ~8% | ~0.62 |
| A3 Zheng 2017 | ~44 | 14.19 | ~70% | ~8% | ~0.62 |
| A4 [pre-2020] | ~57 | 14.19 | ~77% | ~8% | ~0.69 |
| A5 Elsherif 2025 | ~43 | 14.19 | ~70% | ~8% | ~0.62 |

The inflation is consistent across protocols: a method achieving genuinely good RMSE (~13 cycles) would appear to improve by 70–77% over a collapsed baseline, while the true improvement over a valid baseline is ~8%.

This analysis is conservative. If the proposed method also uses the MPC-prone protocol, its RMSE may also reflect collapsed runs in some seeds, compounding the distortion in ways that are not captured by Δ_inflation.

**Limitation.** Δ_inflation as defined here quantifies distortion relative to the FD003 corrected baseline. It does not apply to FD001, FD002, or FD004, which show 0–10% MPC rates and where baseline RMSE values are reliable. Protocol distortion in this study is specific to FD003+LSTM under the tested trigger conditions.

---

## Reference Placeholders

*(번호 체계는 References.md와 동일)*

- [P5] Elsherif et al. (2025) — CAELSTM, FD003 RMSE=13.40
- [P31] Meng et al. (2023) — ESWA audit paper (A1)
- [P32] Qin et al. (2024) — EAAI audit paper (A2)
- [P33] Zheng et al. (2017) — ICPHM audit paper (A3)
- [ANON] — prior work (H6 fault-mode routing), anonymized for review
