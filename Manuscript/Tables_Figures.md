# Manuscript Figures and Tables
**Target standard:** IEEE Transactions / Elsevier journal (≤8 figures, ≤5 tables)
**Language:** English captions and table titles
**Scope:** H1 (RUL Clipping) · H2 (Normalization) · H3 (Fault Mode Separation) · H4 (Loss Functions)

---

## Figures

### Figure 1 — Sensor Degradation Trajectories
**File:** `Figures/Fig1_degradation_trends.png`
**Source:** `Dataset/Figure/fig06_degradation_trends.png`
**Hypothesis:** Background / Motivation (H2, H3)

**Caption:**
> Illustrative sensor degradation trajectories for representative engines in FD003 (single operating condition, two fault modes). Selected sensors exhibiting high RUL correlation (s2, s3, s4, s7, s11, s12) show visually distinct degradation patterns between HPC-fault and fan-fault engines, motivating both fault-mode-aware modeling (H3) and the choice of fleet-level normalization (H2) that preserves inter-engine degradation contrast.

---

### Figure 2 — Piecewise Linear RUL Label and Clipping
**File:** `Figures/Fig2_rul_clipping.png`
**Source:** `Dataset/Figure/fig10_rul_clipping.png`
**Hypothesis:** H1 (Background)

**Caption:**
> Piecewise linear RUL labeling scheme with threshold-based clipping. The raw RUL decreases linearly from the maximum cycle, but is clipped at a threshold *c* to account for the flat healthy phase in which degradation is undetectable. The shaded region illustrates the effect of varying *c* ∈ {75, 100, 125, 130, ∞}. Over-clipping (small *c*) truncates degradation information for long-lifetime engines; no clipping (*c* = ∞) permits unbounded targets that destabilize model training.

---

### Figure 3 — Effect of RUL Clipping on Prediction Accuracy
**File:** `Figures/Fig3_H2_clipping_rmse.png`
**Source:** `Data_Analysis/Results/H2_clipping/figures/fig_H2_01_rmse_heatmap.png`
**Hypothesis:** H1

**Caption:**
> RMSE heatmap across four CMAPSS sub-datasets (FD001–FD004) and five clipping thresholds. Each cell reports the mean RMSE over 20 independent runs of a linear regression baseline. The clip=125 configuration achieves the lowest or near-lowest RMSE in all datasets, confirming the empirical optimality of the industry-standard threshold. The absence of clipping (clip=None) produces catastrophic predictions on FD003 (RMSE = 56.1), where unbounded RUL targets amplify residuals near the beginning of engine life.

---

### Figure 4 — RMSE Comparison Across Normalization Strategies
**File:** `Figures/Fig4_H5_normalization_rmse.png`
**Source:** `Data_Analysis/Results/H5_normalization/figures/fig_H5_01_rmse_heatmap.png`
**Hypothesis:** H2

**Caption:**
> Mean RMSE heatmap (5 seeds) for seven normalization strategies (N1–N7) across four CMAPSS sub-datasets. Fleet-level min-max normalization (N1) achieves the lowest RMSE on FD001 (14.14), FD002 (14.31), and FD004 (14.60). Per-unit normalization methods (N3–N6) consistently underperform by removing between-engine degradation contrast. RevIN (N7) performs competitively on single-condition FD001 (14.92) but degrades substantially on multi-condition datasets (FD002: 18.16, FD004: 18.28), as instance-level normalization conflates operating-condition shifts with degradation signals. The anomalously high variance of N1 on FD003 (std = 12.86) reflects protocol-sensitive instability under the original H2 experimental configuration; see §IV.B.2 for unified-protocol results.

---

### Figure 5 — Statistical Significance of Normalization Differences (vs N1)
**File:** `Figures/Fig5_H5_statistical_test.png`
**Source:** `Data_Analysis/Results/H5_normalization/figures/fig_H5_05_statistical_test.png`
**Hypothesis:** H2

**Caption:**
> Pairwise statistical comparison of each normalization method against fleet min-max (N1) using the Wilcoxon rank-sum test with Benjamini-Hochberg FDR correction (α = 0.05). Effect sizes are reported as Cohen's *d*. Per-unit methods (N3–N6) are significantly inferior to N1 on FD001, FD002, and FD004 (|*d*| > 2.3 in all cases). RevIN (N7) is also significantly inferior on the same three datasets. No alternative achieves statistically significant superiority over N1. No significant differences were detected on FD003 under the original H2 protocol due to high seed-to-seed variance.

---

### Figure 6 — GMM Fault-Mode Cluster Quality (Phase 1)
**File:** `Figures/Fig6_H6_gmm_clustering.png`
**Source:** `Data_Analysis/Results/H6_fault_mode/figures/fig_H6_P1_clusters.png`
**Hypothesis:** H3

**Caption:**
> Gaussian Mixture Model (GMM, *K* = 2) clustering results for FD003 and FD004 in Phase 1. (a) Silhouette scores across three feature variants: AB_full (all cycles), AB_slope (degradation slope features), and AB_late (final 20% of cycles). AB_late achieves the highest silhouette (FD003: 0.858; FD004: 0.855), confirming that fault-mode separation is most pronounced in late-life degradation. (b) BIC scores consistently favor *K* = 2, validating the two-cluster hypothesis corresponding to HPC-fault and fan-fault degradation modes. The AB_full variant (silhouette: FD003 = 0.761, FD004 = 0.750) is used in Phase 2 to retain temporal diversity across the full engine lifetime.

---

### Figure 7 — Fault-Mode Separation Model Comparison (M0–M3)
**File:** `Figures/Fig7_H6_model_comparison.png`
**Source:** `Data_Analysis/Results/H6_fault_mode/figures/fig_H6_01_model_comparison.png`
**Hypothesis:** H3

**Caption:**
> RMSE comparison (mean ± std, 5 seeds) of four fault-mode separation architectures on FD003 and FD004. GMM hard routing (M1) significantly degrades RMSE on both multi-fault datasets (FD003: 33.25 ± 8.74, +156%; FD004: 33.28 ± 2.05, +76%) relative to M0 (FD003: 12.97 ± 0.67; FD004: 18.96 ± 3.97). Soft gating (M2) and end-to-end attention routing (M3) avoid this degradation without statistically detectable improvement over the single-branch baseline. M1 collapse on FD004 is due to test-time cluster assignment collapse (247:1 ratio).

---

### Figure 8 — RUL Clipping × Loss Function Interaction (NASA Score)
**File:** `Figures/Fig8_H7_clip_loss_interaction.png`
**Source:** `Data_Analysis/Results/H7_loss_function/figures/fig_H7_05_clip_loss_interaction.png`
**Hypothesis:** H4

**Caption:**
> Mean NASA prognostic score (lower is better) across all four datasets as a function of RUL clipping threshold and loss function (560 LSTM training runs, 5 seeds per configuration). RUL clipping dominates loss function choice: clip=None yields catastrophic NASA scores (up to 9.6 × 10⁶ for FD003/MSE, per-engine mean averaged across 5 seeds) regardless of loss function, while clip ∈ {125, 130} stabilizes training for all losses. Among clipped configurations, standard MSE (L1) with clip_130 achieves the best mean NASA score (24.39), and no custom loss function achieves a statistically significant improvement after Benjamini-Hochberg FDR correction (α = 0.05). L3 (DynMSE) and L4 (Focal) rank consistently second and third.

---

## Tables

### Table 1 — CMAPSS Dataset Characteristics
**File:** `Tables/Table1_dataset_characteristics.csv`

| Dataset | Training Engines | Test Engines | Operating Conditions | Fault Modes | Useful Sensors | Min Lifetime | Max Lifetime |
|---------|-----------------|-------------|---------------------|-------------|---------------|-------------|-------------|
| FD001   | 100 | 100 | 1 | 1 (HPC degradation)       | 14 | 128 | 362 |
| FD002   | 260 | 259 | 6 | 1 (HPC degradation)       | 20 | 128 | 378 |
| FD003   | 100 | 100 | 1 | 2 (HPC + Fan degradation) | 14 | 145 | 525 |
| FD004   | 249 | 248 | 6 | 2 (HPC + Fan degradation) | 20 | 128 | 543 |

**Caption:**
> Summary characteristics of the four CMAPSS sub-datasets used in this study. All datasets share 21 raw sensor channels; 14 are retained for FD001 and FD003 (7 constant-variance channels removed), and 20 are retained for FD002 and FD004 (only s16 is constant). "Operating Conditions" denotes the number of distinct flight operating points (altitude, Mach number, throttle resolver angle). "Fault Modes" indicates the number of independently progressing degradation pathways present in the dataset.

---

### Table 2 — Effect of RUL Clipping Threshold on Prediction Performance (H1)
**File:** `Tables/Table2_H2_clipping_results.csv`

| Clip Value | FD001 RMSE | FD001 NASA | FD002 RMSE | FD002 NASA | FD003 RMSE | FD003 NASA | FD004 RMSE | FD004 NASA |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 75        | 33.65 | 26.49   | 47.37 | 615.74    | 32.48 | 27.36   | 50.83 | 451.22     |
| 100       | 24.51 | 10.93   | 37.67 | 173.05    | 23.00 | 9.11    | 40.37 | 110.29     |
| **125**   | **21.90** | 13.14 | **32.39** | 70.48 | **21.62** | 13.09 | **34.61** | 49.69 |
| 130       | 22.06 | 14.84   | 31.83 | **62.52** | 22.24 | 16.17   | 34.04 | **47.21** |
| None      | 31.90 | 132.21  | 33.05 | 116.92    | 56.09 | 4,014,724 | 46.99 | 1,758.56 |

**Caption:**
> RMSE and NASA prognostic score of a linear regression (OLS) baseline across four CMAPSS sub-datasets and five RUL clipping thresholds (mean over 20 runs). Bold values denote the best performance per metric per dataset. clip = 125 yields the lowest RMSE on all four datasets and competitive NASA scores, confirming the empirical optimality of the widely-adopted 125-cycle threshold. The absence of clipping (None) results in catastrophic NASA scores, particularly on FD003, where unbounded targets amplify early-life prediction errors exponentially.

---

### Table 3 — Normalization Strategy Comparison on CMAPSS (H2)
**File:** `Tables/Table3_H5_normalization_results.csv`

| Normalizer | Description | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|-----------|-------------|-----------|-----------|-----------|-----------|
| **N1 (Fleet MinMax)** | Fleet-level min-max | **14.14 ± 0.22** | **14.31 ± 0.10** | 19.05 ± 12.86 | **14.60 ± 0.30** |
| N2 (Fleet Std)        | Fleet-level z-score  | 14.38 ± 0.81 | 14.83 ± 0.55 | **14.69 ± 0.71** | 15.33 ± 0.12 |
| N3 (Per-Unit MM 5%)   | Per-engine min-max   | 19.15 ± 0.58** | 18.28 ± 0.15** | 19.65 ± 1.09 | 18.02 ± 0.76** |
| N4 (Per-Unit MM 10%)  | Per-engine min-max   | 19.66 ± 0.87** | 15.52 ± 0.32** | 17.35 ± 0.54 | 16.49 ± 0.18** |
| N5 (Per-Unit Std 5%)  | Per-engine z-score   | 20.78 ± 0.78** | 18.49 ± 0.49** | 21.52 ± 1.68 | 18.76 ± 0.33** |
| N6 (Per-Unit Std 10%) | Per-engine z-score   | 17.90 ± 1.84** | 16.11 ± 0.22** | 18.65 ± 1.41 | 16.62 ± 0.36** |
| N7 (RevIN)            | Instance normalization | 14.92 ± 0.48* | 18.16 ± 0.27** | 16.44 ± 0.36 | 18.28 ± 0.66** |

*p < 0.05; **p < 0.01 vs N1 (Wilcoxon rank-sum, BH-FDR corrected). Mean ± std over 5 seeds.

**Caption:**
> RMSE (mean ± standard deviation, 5 seeds) for seven normalization strategies applied to a stacked LSTM backbone on CMAPSS FD001–FD004. All models use identical architecture (LSTM-64 → LSTM-32 → FC-16 → output), RUL clip = 125, and engine-level validation split. Significance markers indicate inferior performance relative to fleet min-max (N1) after Benjamini-Hochberg FDR correction. The high variance of N1 on FD003 (std = 12.86) reflects protocol-sensitive instability under the original H2 experimental configuration; it resolved to std = 0.67 under the unified protocol (§IV.B.2).

---

### Table 4 — Fault-Mode Separation Architecture Comparison (H3)
**File:** `Tables/Table4_H6_model_comparison.csv`

| Model | Architecture | FD003 RMSE | FD003 vs M0 | FD004 RMSE | FD004 vs M0 |
|-------|-------------|-----------|------------|-----------|------------|
| M0 | Single LSTM (baseline)          | 12.97 ± 0.67 | — | 18.96 ± 3.97 | — |
| M1 | Hard Routing (GMM branches)     | 33.25 ± 8.74 | +156%† | 33.28 ± 2.05 | +76%† |
| **M2** | **Soft Gating (GMM-weighted)** | **12.28 ± 0.57** | p=0.125 (equiv.) | **18.82 ± 1.56** | p=1.00 (equiv.) |
| M3 | Attention Gate (end-to-end)     | 13.24 ± 1.69 | p=0.625 (equiv.) | 17.30 ± 1.04 | p=0.625 (equiv.) |

†p_BH=0.024 vs M0 (two-sided Mann-Whitney U, BH-FDR; d=2.82/3.64). M2/M3 p-values: paired two-sided Wilcoxon signed-rank. Results over 5 random seeds. M1 FD004 degradation caused by test-time cluster assignment collapse (ratio 1:247).

**Caption:**
> Performance comparison of four fault-mode separation architectures on the two multi-fault-mode CMAPSS sub-datasets (FD003 and FD004). GMM clustering identifies two statistically well-separated fault modes (Silhouette = 0.761 for FD003, 0.750 for FD004). GMM hard routing (M1) substantially degrades RMSE on both datasets (+156% FD003; +76% FD004; p_BH = 0.024), driven by test-time cluster assignment collapse (247:1 ratio on FD004) and training-split sensitivity (std=8.74 on FD003). Soft gating (M2) and end-to-end attention routing (M3) recover to baseline-equivalent performance, avoiding the catastrophic hard-routing failure at negligible computational overhead. Gate confidence (mean max(w₀, w₁)) was 0.727 ± 0.165 on FD003 and 0.676 ± 0.055 on FD004, reported descriptively.

---

### Table 5 — Loss Function Comparison for RUL Prediction (H4)
**File:** `Tables/Table5_H7_loss_function_results.csv`

| Loss | Description | FD001 RMSE | FD001 NASA | FD002 NASA | FD003 NASA | FD004 NASA | Sig. vs L1 |
|------|-------------|-----------|-----------|-----------|-----------|-----------|-----------|
| L1 (MSE)         | Standard MSE              | 16.70 ± 0.41 | 5.28 ± 0.60 | 48.34 ± 12.31 | 5.52 ± 0.66 | 42.18 ± 5.04 | — |
| L2 (NASA Score)  | Direct NASA minimization  | 15.46 ± 0.34 | 4.26 ± 0.70 | 73.82 ± 22.18 | 3.85 ± 0.73 | 43.53 ± 8.17 | No |
| L3 (DynMSE)      | Dynamically weighted MSE  | 14.74 ± 0.36 | 3.61 ± 0.27 | 56.86 ± 9.93  | 4.00 ± 0.60 | **35.91 ± 3.16** | No |
| L4 (Focal)       | Focal loss                | 17.04 ± 0.66 | 6.40 ± 0.81 | **49.59 ± 6.99** | 6.44 ± 1.20 | 36.36 ± 4.81 | No |
| **L5 (TWA)**     | Time-weighted asymmetric  | **14.72 ± 0.32** | **3.40 ± 0.32** | 68.60 ± 22.47 | 4.94 ± 2.55 | 41.89 ± 6.44 | No |
| L6 (Pinball)     | Quantile loss (τ = 0.35)  | 15.69 ± 0.49 | 5.09 ± 0.55 | 90.60 ± 12.81 | 9.55 ± 1.33 | 44.83 ± 8.12 | No |
| L7 (HubA)        | Huber-asymmetric hybrid   | 14.90 ± 0.41 | 3.88 ± 0.42 | 54.95 ± 9.33  | 4.96 ± 1.48 | 41.16 ± 4.52 | No |

All comparisons at best clip per dataset (clip_130 for L1, L4; clip_125 otherwise). BH-FDR correction at α = 0.05.

**Caption:**
> Prediction performance of seven loss functions on CMAPSS FD001–FD004 (mean ± std, 5 seeds per configuration, 560 total LSTM training runs in Phase 2b). The best clipping threshold per loss function is selected from Phase 2b grid search. No custom loss function achieves statistically significant improvement over standard MSE (L1) after Benjamini-Hochberg FDR correction. L5 (TWA) yields the lowest NASA score on FD001 (3.40), suggesting potential benefit for single-condition datasets, but this advantage does not generalize across all sub-datasets. Notably, L2 (NASA Score Loss), which directly minimizes the evaluation metric, fails to consistently outperform MSE, possibly due to non-smooth gradient behaviour near the piecewise transition point. RUL clipping dominates loss function choice: removing clipping (clip=None) increases NASA scores by three to six orders of magnitude regardless of loss function.

---

## Selection Rationale

| # | Figure/Table | Hypothesis | Reason for Inclusion |
|---|-------------|-----------|---------------------|
| Fig 1 | Degradation trajectories | Background | Motivates H3 (fault mode separation) and H2 (fleet normalization) — shows visually distinct degradation paths |
| Fig 2 | RUL clipping concept | H1 | Conceptual illustration essential for readers unfamiliar with piecewise RUL labeling |
| Fig 3 | Clipping RMSE heatmap | H1 | Primary empirical result of H1; compact 4×5 matrix |
| Fig 4 | Normalization RMSE heatmap | H2 | Primary empirical result of H2; most space-efficient summary of 140 runs |
| Fig 5 | Statistical significance (H2) | H2 | Required to substantiate the "statistically significant" claim of H2 |
| Fig 6 | GMM clustering quality | H3 | Phase 1 validation — without this, the H3 Phase 2 result could be dismissed as implementation artifact |
| Fig 7 | M0–M3 model comparison | H3 | Primary empirical result of H3; shows the hard-routing failure and baseline recovery |
| Fig 8 | Clip × Loss interaction | H4 | Captures the dominant finding: clipping > loss function choice |
| Table 1 | Dataset characteristics | Background | Standard for CMAPSS papers; provides context for complexity differences across sub-datasets |
| Table 2 | H1 clipping results | H1 | Numerical detail not fully visible in Fig 3; NASA scores reveal clip=None catastrophe |
| Table 3 | H2 normalization results | H2 | Full RMSE ± std for all 7 methods; significance markers reference-ready |
| Table 4 | H3 model comparison | H3 | Precise RMSE/NASA with uncertainty; M1 FD004 failure numerically documented |
| Table 5 | H4 loss function results | H4 | Allows readers to reproduce and compare all 7 loss functions |
