# IV. Results

> **Draft status:** v1.6 — 2026-08-31 (Synced with manuscript_full_text_clean.md: 검정명 수정, "substantially" 통일, gate confidence descriptive-only, M3 "did not differ detectably", H4 power-limited 표현 수정)
> **Models:** H1 = LinearRegression (deterministic, 20 runs); H2/H3/H4 = Stacked LSTM (5 seeds, mean ± std)
> **Baseline:** clip = 125 cycles, Fleet min-max (N1), single-branch LSTM (M0), MSE (L1) throughout unless noted

---

## A. Effect of RUL Clipping (H1)

**clip = 125 cycles yielded the lowest or tied-lowest RMSE on all four sub-datasets.** The full matrix of results is presented in Table IV. On FD001 and FD003 (single operating condition), clip = 125 was the clear optimum (RMSE = 21.90 and 21.62, respectively). On FD002 and FD004, clip = 130 produced a marginally lower RMSE (Δ = −0.57 cycles on both), but the difference did not approach statistical significance (Wilcoxon rank-sum, two-sided, p = 0.97 on each; BH-FDR applied). The non-significant result (p = 0.97) was directionally consistent with clip=130 being marginally worse than clip=125, not better. No tested alternative threshold achieved a statistically significant improvement over clip = 125, confirming its status as the practically validated standard [4, 5]. Aggressive clipping at clip = 75 was significantly inferior on all four datasets (Δ RMSE = +10.5 to +16.2 cycles; p ≤ 0.028), indicating that excessive truncation discards degradation signal in the upper RUL range. These findings were consistent with earlier single-dataset reports that converged empirically on 125 cycles [4, 5] and extend them to a rigorous four sub-dataset evaluation.

**Removing the RUL ceiling entirely was catastrophic on FD003.** At clip = None, the mean NASA prognostic penalty on FD003 reached 4,014,724 — a 306,000-fold increase over clip = 125 (13.09) — while FD004 reached 1,759 (35.4×). The mechanism is the asymmetric exponential NASA metric: late predictions are penalised by exp(d/10), which grows unboundedly as the model, trained on uncapped labels exceeding 500 cycles, systematically over-predicts RUL throughout the test trajectory.

RMSE was also significantly elevated on FD001 (31.90 vs 21.90; p = 0.006) and FD004 (46.99 vs 34.61; p = 0.002), though less extreme than the NASA Score collapse. FD002 was substantially more robust (RMSE = 33.05; p = 0.054), likely because its 260 training engines and six operating conditions provided sufficient distributional support for the model to learn a conservative bias without an explicit ceiling. FD003's disproportionate RMSE increase — 159% versus FD001's 46%, despite both having a single operating condition — may in part reflect FD003's bimodal fault-mode structure (Section C), where unbounded targets spanning two distinct degradation-lifetime populations could compound the over-prediction bias.

This dataset-dependent sensitivity confirmed that clip = 125 cannot be treated as universally optimal, but it remains the safest default across the full benchmark.

---

## B. Effect of Normalization Strategy (H2)

**Normalization outcomes differ sharply between FD003 and the remaining three sub-datasets.** FD003 serves as the sole exception to an otherwise consistent pattern and is therefore addressed first.

On FD001, FD002, and FD004, fleet min-max (N1) achieved the lowest RMSE: 14.14 ± 0.22, 14.31 ± 0.10, and 14.60 ± 0.30 cycles, respectively. All per-unit strategies (N3–N6) were significantly inferior on FD001 (ΔRMSE = +3.8 to +6.6; p_BH = 0.011; d = 2.3–7.6) and FD002 (ΔRMSE = +1.2 to +4.2; p_BH = 0.011; d = 4.2–22.4). On CMAPSS, inter-engine sensor variation is modest relative to degradation magnitude; fleet-level statistics therefore preserve more discriminative information than engine-local references.

RevIN (N7) was likewise significantly inferior on FD001 (ΔRMSE = +0.78; p_BH = 0.011; d = 1.66), FD002 (+3.85; d = 20.0), and FD004 (+3.68; d = 6.19), despite its learnable affine parameters (Table V). RevIN [20] excels at multi-step forecasting under temporal distribution shift [24], but on CMAPSS the dominant challenge is extracting a shared degradation signature across engines — a context in which per-window adaptation is counterproductive. Fleet z-score (N2) showed no significant difference from N1 on FD001/FD002/FD003, indicating that the fleet-vs-per-unit distinction matters more than the specific scaling transform. Figures 4 and 5 visualise the full RMSE matrix and the pairwise statistical comparison against N1.

**FD003 was the sole dataset where no normalization strategy separated from N1.** N1 achieved RMSE = 19.05 ± 12.86 on FD003 — the inter-seed standard deviation of 12.86 anomalously high relative to ≤ 1.84 on all other dataset-normaliser combinations — and all competitors showed p_BH ≥ 0.14. (Note: H2 N1/FD003 RMSE = 19.05 ± 12.86 and H3 M0/FD003 RMSE = 12.97 ± 0.67 reflect genuinely different experimental conditions — backbone architecture, input features, validation split method, and test-prediction clipping all differ between the two hypotheses (Table III). Each baseline is interpreted only against its own controlled conditions.)

Notably, however, this elevated variance was confined to N1 alone: the RevIN-style forward normalisation (N7) achieved 16.44 ± 0.36 and N4 achieved 17.35 ± 0.54, both with narrow distributions. N1's per-seed RMSE spanned approximately 7 to 43 cycles — a profile inconsistent with normalisation failure and indicative of protocol-sensitive instability.

The variance spike was confined to N1 under the original H2 protocol and did not persist under the unified protocol (§IV.B.2). It is therefore interpreted as protocol-sensitive instability. FD003's heterogeneous fault structure may contribute to this sensitivity, but the present experiments do not isolate its causal contribution.

---

## B.2 Protocol-Unified N1–N3 Robustness Analysis

To verify that N1's advantage over per-unit normalisation was not an artefact of the original H2 protocol, N1 and N3 were directly compared under a unified backbone-matched protocol (full-capacity LSTM, sensors-only features, random 20% engine validation split, 30-epoch minimum warm-up, prediction clipping to [0, 125]; 5 seeds per strategy per dataset; 40 runs total: 2 strategies × 4 datasets × 5 seeds).

Under the unified protocol, N1 retained its advantage over N3 on FD001 (RMSE 13.59 vs 18.61; p_BH = 0.009; d = 13.15) and FD002 (15.84 vs 17.98; p_BH = 0.009; d = 3.53). On FD003, the fleet advantage was also detectable under the unified protocol (12.97 vs 21.34; p_BH = 0.009; d = 8.34), whereas it was not detectable under the original H2 protocol — confirming that the FD003 anomaly was a protocol interaction rather than a genuine absence of fleet-level advantage. No significant difference was found on FD004 (17.92 vs 18.51; p_BH = 0.46; d = 0.26), indicating that the normalization ranking on multi-condition datasets is sensitive to whether explicit operating-condition columns are retained. Crucially, the N1/FD003 standard deviation collapsed from 12.86 under the original H2 protocol to 0.67 under the unified protocol, confirming that the anomalous inter-seed variance was a protocol-configuration effect rather than a property of fleet normalization or fault-mode heterogeneity per se.

**Table V-B: N1 (fleet min-max) vs N3 (per-unit min-max) RMSE under unified backbone-matched protocol (5 seeds; Wilcoxon rank-sum, BH-FDR corrected)**

| Dataset | N1 RMSE (mean ± SD) | N3 RMSE (mean ± SD) | ΔRMSE | Cohen's *d* | *p*_BH | Sig. |
|---------|---------------------|---------------------|-------|-------------|--------|------|
| FD001 | 13.59 ± 0.17 | 18.61 ± 0.51 | +5.02 | 13.15 | 0.009 | ✓ |
| FD002 | 15.84 ± 0.40 | 17.98 ± 0.76 | +2.15 | 3.53 | 0.009 | ✓ |
| FD003 | 12.97 ± 0.67 | 21.34 ± 1.25 | +8.36 | 8.34 | 0.009 | ✓ |
| FD004 | 17.92 ± 2.50 | 18.51 ± 2.05 | +0.59 | 0.26 | 0.459 | — |

ΔRMSE = N3 − N1 (positive = N3 worse). Sig. ✓: p_BH < 0.05 and |d| ≥ 0.3.

---

## C. Fault-Mode Architectures (H3)

### C.1  Unsupervised Cluster Quality

Before evaluating branch architectures, the presence of two distinct fault modes in FD003 and FD004 was verified. GMM clustering (k = 2) on combined late-cycle means and degradation slopes of seven discriminant sensors yielded Silhouette scores of 0.761 (FD003, AB_full variant) and 0.750 (FD004, AB_full), both well above the conventional quality threshold of 0.50 [31]. Slope-only clustering (AB_slope), which controls for possible life-length confounding, yielded Silhouette = 0.702 (FD003) and 0.683 (FD004), confirming that the two clusters reflect genuine sensor-trajectory differences rather than an artefact of unequal engine lifetimes. Sensor s15 (bypass pressure ratio) exhibited the largest inter-cluster z-score (|Δz| = 31.3), followed by s20 (HPT bleed, 16.0) and s21 (LPT bleed, 15.2), consistent with the known distinction between HPC-dominated and fan-dominated degradation pathways in CMAPSS FD003/FD004 [32].

### C.2  FD003: Hard Routing Failure and Fault-Mode Architecture Results

**GMM hard-routing (M1) substantially degraded FD003 RMSE relative to the single-branch baseline.** Detailed results appear in Table VI. Under a corrected experimental protocol — using the per-run training seed for validation split, a 30-epoch minimum warmup, and evaluation clipping to [0, 125] cycles — M0 yielded RMSE = 12.97 ± 0.67 cycles and mean NASA Penalty = 3.47 ± 0.49. GMM hard-routing (M1) regressed substantially to RMSE = 33.25 ± 8.74 cycles (+156%; p_BH = 0.024, d = 2.82). The large standard deviation (8.74 cycles) indicates that hard-routing quality was highly sensitive to the random training split: GMM labels derived from trajectory-level statistics produced markedly different branch assignments across seeds, leading to unstable per-seed RMSE outcomes.

GMM soft-gating (M2) recovered to RMSE = 12.28 ± 0.57 cycles, numerically lower than M0 (12.97; d = −1.06), but the difference was not statistically detectable with five seeds (paired Wilcoxon, p = 0.125). M3, which replaced the GMM entirely with a lightweight GatingNet trained end-to-end on the first K = 10 observed cycles, achieved RMSE = 13.24 ± 1.69 cycles (mean NASA Penalty = 3.86 ± 2.17). M3 did not differ detectably from M0 (paired Wilcoxon, p = 0.625; d = 0.24); the 0.27-cycle mean difference is practically negligible. The inter-seed spread of M3 (std = 1.69) was slightly wider than M0 (std = 0.67), reflecting seed-to-seed variability in GatingNet random initialisation.

A post-hoc sensitivity analysis varying K ∈ {5, 10, 15, 20, 30} revealed that M3 RMSE was largely insensitive to the number of initial cycles used for routing. K = 5 already achieved RMSE = 14.23 ± 0.34, indicating limited sensitivity of prognostic performance to prefix length within these CMAPSS conditions.

Mean maximum gate weight (mean max(w₀, w₁)) on FD003 was 0.727 ± 0.165. Because ground-truth routing labels are unavailable, this value is reported descriptively and is not interpreted as routing accuracy.

**A fair-baseline re-evaluation (M1_kprefix) confirms that M1 failure is not attributable solely to train–test feature mismatch.** A variant (M1_kprefix) was implemented using identical K = 10 prefix features for both training GMM assignment and test-time routing, placing it on the same information horizon as M3's GatingNet. M1_kprefix yielded RMSE = 42.54 ± 0.37 on FD003 and RMSE = 35.43 ± 7.04 on FD004 — substantially worse than M0 on both datasets and no better than original M1. On FD003, the K-prefix GMM captures less fault-mode signal than full-trajectory features (early-cycle means are less discriminative than late-cycle degradation slopes), resulting in weaker cluster separation and worse routing. On FD004, both variants fail for the same structural reason: operating-condition-driven sensor variability overwhelms the fault-mode signal regardless of which trajectory segment is used for routing. M1_kprefix confirms that hard GMM partitioning is unreliable under realistic deployment conditions, irrespective of feature-construction alignment between training and test.

### C.3  FD004: Cluster Collapse Under Hard Routing

**Hard routing (M1) degraded below the baseline on FD004.** M0 achieved RMSE = 18.96 ± 3.97 on FD004; M1 regressed to 33.28 ± 2.05, a 75.5% increase (p_BH = 0.024, d = 3.64). Post-hoc inspection of test-time cluster assignments revealed the root cause: 247 of 248 test engines were assigned to the same branch by the GMM argmax, compared with an approximately balanced assignment during training. The root mechanism is a feature distribution shift at test time: the GMM was fitted on complete run-to-failure trajectories (computing late-cycle means and degradation slopes across each training engine's full life), but test-engine routing had to use only the last 30-cycle observation window. On FD004, which spans six operating conditions, the sensor statistics of a 30-cycle window vary substantially depending on which operating condition the engine encountered in its final cycles — a variation unrelated to fault mode. This operating-condition-driven variability overwhelmed the fault-mode signal in the GMM features, causing the decision boundary — learned on trajectory-level statistics — to assign virtually all test engines to a single branch. Soft-gating (M2) recovered at RMSE = 18.82 ± 1.56, comparable to M0 (paired Wilcoxon, p = 1.00; d = −0.06), confirming that soft weighting avoids the collapse without improving over the baseline.

M3 achieved RMSE = 17.30 ± 1.04 on FD004, and did not differ detectably from M0 (paired Wilcoxon, p = 0.625; d = −0.48). Because M3's GatingNet reads only the first 10 cycles of each engine — information that is equally available in training and test settings — it was immune to the test-time distribution collapse that undermined M1. On both FD003 and FD004, M2 and M3 avoided the hard-routing failure but provided no statistically detectable improvement over the single-branch baseline. **M0 is the recommended default architecture.** M2 and M3 are alternatives for practitioners who require an explicit fault-mode routing structure, as they avoided M1's catastrophic degradation at modest additional complexity (approximately 2× parameters; see Table II).

Mean maximum gate weight on FD004 was 0.676 ± 0.055. This value is reported descriptively; ground-truth routing labels are unavailable.

---

## D. Loss Function Comparison (H4)

**With only five seeds per condition and a large multiple-comparison family, H4 had limited power to detect anything other than large and consistent effects.** Non-significant results are therefore interpreted as an absence of detectable improvement rather than evidence of equivalence.

**No statistically detectable improvement from any custom loss function was found.** Across all 96 pairwise comparisons (6 loss functions × 4 clipping values × 4 datasets), zero reached p_BH < 0.05; the minimum corrected p-value was 0.176.

Some individual combinations showed nominally lower scores: L5 (TWA) at clip = 125 on FD001 achieved Mean NASA Penalty = 3.40 ± 0.32 vs 5.66 ± 1.12 for MSE (d = −1.54); L7 (HubA) reduced RMSE to 14.90 ± 0.33 vs 16.34 ± 0.40 but worsened Mean NASA Penalty (d = +2.00), because HubA's symmetric Huber penalty does not sufficiently penalise late predictions. Neither survived BH-FDR correction (p_BH = 1.0). Positive d denotes a worse NASA Score than MSE; negative d denotes improvement. The null result diverged from single-dataset gains reported by Rengasamy et al. [33, 35] because cross-dataset BH-FDR substantially raises the significance threshold. Clipping threshold findings are independent of backbone architecture; H1 used linear regression (OLS) specifically to isolate label-engineering effects from model capacity.

The dominant driver of NASA Score variance across this study was RUL clipping, not loss function choice. Table VII illustrates this: at clip = 125, all seven loss functions converged to within a factor of 1.5× of one another in NASA Score on FD001 (range: 3.40–6.14). Removing the clip inflated FD003 NASA Score to between 3,124 (L6, Pinball) and 9,613,539 (L1, MSE), a span of three orders of magnitude that dwarfs any inter-loss difference at a fixed clip value. Even the loss functions designed to suppress late predictions (L5 TWA, L6 Pinball, L7 HubA) reduced but did not eliminate the clip = None catastrophe on FD003: L6 Pinball achieved NASA = 3,124 vs L1 MSE's 9,614,000 — a 3,000× improvement within the clip-free condition, yet still 630× worse than the worst result at clip = 125. This finding implies that loss-function engineering is a second-order design choice: it cannot substitute for proper label engineering (RUL clipping) as a mechanism for controlling the tail behaviour of the NASA penalisation function. Practitioners should fix the clipping threshold before considering custom loss functions.
