# References

> **Draft status:** v1.0 — 2026-07-03 (web search coverage through June 2026)

**Paper:** From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction

---

## Section Usage Map

| Ref # | Background | H2 Clipping | H5 Normalization | H6 Fault Mode | H7 Loss Fn |
|-------|:---------:|:-----------:|:----------------:|:-------------:|:----------:|
| [1]   | ✓ | ✓ | ✓ | | |
| [2]   | ✓ | ✓ | ✓ | | |
| [3]   | | ✓ | | | ✓ |
| [4]   | ✓ | | | | |
| [5]   | ✓ | ✓ | | | |
| [6]   | ✓ | ✓ | | | |
| [7]   | ✓ | | | | |
| [8]   | ✓ | | | | |
| [9]   | ✓ | | | | |
| [10]  | ✓ | | | | |
| [11]  | ✓ | | | | |
| [12]  | ✓ | | | | |
| [13]  | ✓ | | | | |
| [14]  | ✓ | | | | |
| [15]  | ✓ | ✓ | | ✓ | ✓ |
| [16]  | | ✓ | | | |
| [17]  | | ✓ | | | |
| [18]  | | ✓ | ✓ | | |
| [19]  | | | ✓ | | |
| [20]  | | | ✓ | | |
| [21]  | | | ✓ | | |
| [22]  | | | ✓ | | |
| [23]  | | | ✓ | ✓ | |
| [24]  | | | ✓ | | |
| [25]  | | | | ✓ | |
| [26]  | | | | ✓ | |
| [27]  | | | | ✓ | |
| [28]  | | | | ✓ | |
| [29]  | | | | ✓ | |
| [30]  | | | | ✓ | |
| [31]  | | | | ✓ | |
| [32]  | | | | ✓ | |
| [33]  | | | | ✓ | |
| [34]  | | | | | ✓ |
| [35]  | | | | | ✓ |
| [36]  | | | | | ✓ |
| [37]  | | | | | ✓ |
| [38]  | | | | | ✓ |
| [39]  | | | | | ✓ |
| [43]  | | | | ✓ | |
| [44]  | | | | ✓ | |
| [45]  | ✓ | | | | |

---

## Full Reference List

### Foundational Dataset & Early Baselines

**[1] Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008)**
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation."
*Proceedings of the 1st International Conference on Prognostics and Health Management (PHM 2008)*, Denver, CO.
URL: https://www.researchgate.net/publication/251867156
**Used in:** Background, H2, H5
**Note:** Introduces the C-MAPSS simulation tool and the FD001–FD004 benchmark with 21 sensors and six operating conditions; the mandatory first citation in any CMAPSS paper. Establishes the six operating-condition regimes in FD002/FD004 that make fleet-level normalization essential.

---

**[2] Heimes, F. O. (2008)**
"Recurrent Neural Networks for Remaining Useful Life Estimation."
*Proceedings of the 1st International Conference on Prognostics and Health Management (PHM 2008)*, IEEE. DOI: 10.1109/PHM.2008.4711422
URL: https://ieeexplore.ieee.org/document/4711422
**Used in:** Background, H2, H5
**Note:** PHM 2008 competition runner-up; established global min-max sensor scaling on the training partition as the canonical preprocessing step, and framed CMAPSS as an RUL regression task.

---

**[3] Ramasso, E., & Saxena, A. (2014)**
"Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets."
*International Journal of Prognostics and Health Management*, Vol. 5, No. 2.
DOI: 10.36001/ijphm.2014.v5i2.2236
URL: https://papers.phmsociety.org/index.php/ijphm/article/view/2236
**Used in:** H2, H7
**Note:** Canonical CMAPSS benchmark survey (179 citations); documents heterogeneity of preprocessing choices across 70+ published methods and provides the authoritative definition of the asymmetric NASA s-score metric used in H7.

---

### Seminal Deep Learning / LSTM Methods

**[4] Babu, G. S., Zhao, P., & Li, X.-L. (2016)**
"Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life."
*International Conference on Database Systems for Advanced Applications (DASFAA)*, Springer, pp. 214–228.
DOI: 10.1007/978-3-319-32025-0_14
**Used in:** Background
**RMSE:** FD001≈18.45, FD002≈30.29, FD003≈19.82, FD004≈29.16
**Note:** One of the first deep CNN papers on CMAPSS for RUL regression; established the sliding-window + piecewise-linear label convention still used today.

---

**[5] Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017)**
"Long Short-Term Memory Network for Remaining Useful Life Estimation."
*2017 IEEE International Conference on Prognostics and Health Management (ICPHM)*, pp. 88–95.
DOI: 10.1109/ICPHM.2017.7998311
URL: https://www.semanticscholar.org/paper/e66afb33d246dbe3199fd57bcfc1b611136d96c2
**Used in:** Background, H2
**Clip value:** 125
**RMSE:** FD001≈16.14, FD003≈16.18
**Note:** Most widely cited source for the clip=125 convention; introduced the stacked LSTM architecture that established the de facto community standard for both architecture and preprocessing — the backbone adopted in this study.

---

**[6] Li, X., Ding, Q., & Sun, J.-Q. (2018)**
"Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks."
*Reliability Engineering & System Safety*, Vol. 172, pp. 1–11.
DOI: 10.1016/j.ress.2017.11.021
**Used in:** Background, H2
**Clip value:** 125
**RMSE:** FD001≈12.61, FD002≈22.78, FD003≈12.64, FD004≈23.21
**Note:** Highly cited multi-scale CNN that solidified clip=125 as standard and provided the earliest explicit empirical justification ("sensor signals are uninformative above ~125 cycles").

---

**[7] Ellefsen, A. L. et al. (2019)**
"Remaining Useful Life Predictions for Turbofan Engine Degradation Using Semi-Supervised Deep Architecture."
*Reliability Engineering & System Safety*, Vol. 183, pp. 240–251 (453 citations).
DOI: 10.1016/j.ress.2019.01.016
**Used in:** Background
**Note:** Demonstrates that unsupervised pre-training reduces dependence on large labeled datasets; validates on all four CMAPSS sub-datasets.

---

### Recent State-of-the-Art (2021–2025)

**[8] Mo, Y. et al. (2021)**
"Remaining Useful Life Estimation via Transformer Encoder Enhanced by a Gated Convolutional Unit."
*Journal of Intelligent Manufacturing*, Vol. 35, pp. 1997–2012 (266 citations).
DOI: 10.1007/s10845-021-01750-x
**Used in:** Background
**RMSE:** FD001≈13.48, FD002≈22.15, FD003≈13.02, FD004≈22.54
**Note:** First strong Transformer-based RUL model on CMAPSS; sets the Transformer baseline for all subsequent attention-based comparisons.

---

**[9] Jin, R. et al. (2022)**
"Bi-LSTM-Based Two-Stream Network for Machine Remaining Useful Life Prediction."
*IEEE Transactions on Instrumentation and Measurement*, Vol. 71 (125 citations).
DOI: 10.1109/TIM.2022.3163761
**Used in:** Background
**RMSE:** FD001≈12.38, FD002≈19.57, FD003≈12.83, FD004≈19.98
**Note:** Dual-stream BiLSTM combining handcrafted and learned features; SOTA at publication across all four datasets.

---

**[10] Xu, D. et al. (2022)**
"Spatio-temporal Degradation Modeling and Remaining Useful Life Prediction Under Multiple Operating Conditions Based on Attention Mechanism and Deep Learning."
*Reliability Engineering & System Safety*, Vol. 225 (90 citations).
DOI: 10.1016/j.ress.2022.108648
**Used in:** Background
**Note:** Addresses FD002/FD004 with a spatio-temporal Transformer encoding condition information; relevant comparison for K-means residualization in H5.

---

**[11] Zhang, Y. et al. (2023)**
"Trend-Augmented and Temporal-Featured Transformer Network with Multi-Sensor Signals for Remaining Useful Life Prediction."
*Reliability Engineering & System Safety*, Vol. 235 (117 citations).
DOI: 10.1016/j.ress.2023.109258
**Used in:** Background
**Note:** Embeds degradation trend information into Transformer; strong benchmark for trend-augmented methods.

---

**[12] Li, J. et al. (2023)**
"Remaining Useful Life Prediction of Turbofan Engines Using CNN-LSTM-SAM Approach."
*IEEE Sensors Journal*, Vol. 23, No. 10 (68 citations).
DOI: 10.1109/JSEN.2023.3243540
**Used in:** Background
**Note:** Combines 1-D CNN + LSTM + self-attention; specifically targets multi-operating-point FD002/FD004 performance.

---

**[13] Wang, H. et al. (2023)**
"Comprehensive Dynamic Structure Graph Neural Network for Aero-Engine Remaining Useful Life Prediction."
*IEEE Transactions on Instrumentation and Measurement*, Vol. 72 (62 citations).
DOI: 10.1109/TIM.2023.3312337
**Used in:** Background
**Note:** Dynamic inter-sensor graph construction; validated on both CMAPSS and N-CMAPSS.

---

**[14] You, K. et al. (2024)**
"A 3-D Attention-Enhanced Hybrid Neural Network for Turbofan Engine Remaining Life Prediction Using CNN and BiLSTM Models."
*IEEE Sensors Journal*, Vol. 24, No. 3 (64 citations).
DOI: 10.1109/JSEN.2023.3335994
**Used in:** Background
**Note:** 3-D spatial-channel attention over CNN+BiLSTM; recent CMAPSS SOTA with interpretable attention weights.

---

**[15] Elsherif, S. M., Hafiz, B., Makhlouf, M. A., & Farouk, O. (2025)**
"A Deep Learning-Based Prognostic Approach for Predicting Turbofan Engine Degradation and Remaining Useful Life."
*Scientific Reports*, 2025 (18 citations).
DOI: 10.1038/s41598-025-09155-z
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12276258/
**Used in:** Background, H2, H6, H7
**Clip value:** 125
**RMSE:** FD001=14.44, FD003=**13.40**; NASA FD003=264.47
**Note:** CAELSTM (Convolutional Autoencoder + Attention LSTM); the primary benchmark against which H6 M3 (RMSE=14.78±1.32 on FD003) is compared. Uses standard MSE — strong evidence supporting H7's finding that architecture matters more than loss function.

---

### Surveys

**[16] Wu, F. et al. (2024)**
"Remaining Useful Life Prediction Based on Deep Learning: A Survey."
*Sensors*, Vol. 24, No. 11, Article 3454.
DOI: 10.3390/s24113454
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11174398/
**Used in:** Background
**Note:** Comprehensive taxonomy covering CNN, RNN/LSTM, Transformers, GNNs and hybrid models; open challenges summary.

---

**[17] Li, H. et al. (2024)**
"A Review on Physics-Informed Data-Driven Remaining Useful Life Prediction: Challenges and Opportunities."
*Mechanical Systems and Signal Processing*, Vol. 209 (296 citations).
DOI: 10.1016/j.ymssp.2024.111120
**Used in:** Background
**Note:** Highest-citation recent survey; covers physics-informed ML for RUL, motivating purely data-driven systematic benchmarking studies.

---

### H2 — RUL Clipping

**[18] Imbert, F., Adewumi, T., & Han, H. (2026)**
"A Novel Preprocessing-Driven Approach to Remaining Useful Life (RUL) Prediction Using Temporal Convolutional Networks (TCN)."
*IEEE 37th International Conference on Tools with Artificial Intelligence (ICTAI 2025)*.
DOI: 10.1109/ICTAI66417.2025.00160 | arXiv: 2605.02507
URL: https://arxiv.org/abs/2605.02507
**Used in:** H2, H5
**Clip value:** 125 (explicitly states "consistent with prior work … reduces label variance during the initial stable phase")
**Note:** Most explicit recent statement justifying clip=125 as a preprocessing variable; ablates normalization, denoising, and feature selection on CMAPSS.

---

**[19] Ensarioğlu, K., İnkaya, T., & Emel, E. (2023)**
"Remaining Useful Life Estimation of Turbofan Engines with Deep Learning Using Change-Point Detection Based Labeling and Feature Engineering."
*Applied Sciences*, Vol. 13, No. 21, Article 11893.
DOI: 10.3390/app132111893
URL: https://www.mdpi.com/2076-3417/13/21/11893
**Used in:** H2
**Clip value:** Data-driven per-engine change-point (not fixed)
**Note:** Challenges the fixed-threshold paradigm by replacing the hardcoded clip with a per-engine change-point estimate; motivates the question of whether any fixed threshold is optimal.

---

**[20] Srinivasan, A., Andresen, J. C., & Holst, A. (2023)**
"Ensemble Neural Networks for Remaining Useful Life (RUL) Prediction."
*Asia Pacific Conference of the PHM Society 2023*, Vol. 4, No. 1.
DOI: 10.36001/phmap.2023.v4i1.3611 | arXiv: 2309.12445
URL: https://arxiv.org/abs/2309.12445
**Used in:** H2
**Clip value:** 128 (not 125)
**Note:** Illustrates that the "standard" threshold is not uniformly 125 — evidence of community inconsistency motivating H2's quantitative comparison.

---

**[21] Arunan, A., Qin, Y., Li, X., & Yuen, C. (2024)**
"A Change Point Detection Integrated Remaining Useful Life Estimation Model under Variable Operating Conditions."
*Control Engineering Practice*, 2024.
DOI: 10.1016/j.conengprac.2023.105840 | arXiv: 2401.04351
URL: https://arxiv.org/abs/2401.04351
**Used in:** H2
**Clip value:** Per-engine (data-driven)
**Note:** Extends data-driven labeling to FD002/FD004 (six conditions), reporting 5.6–7.5% RMSE improvement over fixed-threshold LSTM baselines — reinforces H2's finding that clipping matters most under multi-condition datasets.

---

**[22] Abdullah, M. E. B. (2026)**
"Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL Prediction with Interpretable Failure Heatmaps."
*arXiv preprint*, April 2026. arXiv: 2604.13459
URL: https://arxiv.org/abs/2604.13459
**Used in:** H2, H7
**Clip value:** 130 on FD001 (without explicit justification)
**RMSE FD001:** 17.52; NASA S-Score FD001: 922.06
**Note:** Most recent paper using clip=130; implements NASA asymmetric exponential loss (H7 L2) as training objective — weaker RMSE than MSE-trained baselines, directly supporting H7's null result.

---

### H5 — Normalization

**[23] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022)**
"Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift."
*International Conference on Learning Representations (ICLR 2022)*.
URL: https://openreview.net/forum?id=cGDAkQo1C0p
**Used in:** H5, H6
**Note:** Original RevIN paper; proposes symmetric per-instance mean/variance removal and restoration to counter distribution shift — the technique evaluated as N7 in H5. Also motivates unsupervised regime detection in H6.

---

**[24] Zhang, Z. et al. (2022)**
"A Framework for Predicting the Remaining Useful Life of Machinery Working under Time-Varying Operational Conditions."
*Applied Soft Computing*, 2022 (20 citations).
URL: https://consensus.app/papers/details/c50add3976c55431a872966d2a1caf13/
**Used in:** H5
**Note:** Proposes MOC-based Normalization — clusters operating conditions and recalibrates sensor amplitude jumps at condition change-points in FD002/FD004; the closest prior work to the K-means residualization used in this study.

---

**[25] Berthelier, G. et al. (2026)**
"On the Role of Reversible Instance Normalization."
*arXiv preprint*, 2026.
URL: https://consensus.app/papers/details/f2feae60790d5d18a0c5b5358099fbd4/
**Used in:** H5
**Note:** Ablation study revealing that several RevIN components are redundant or detrimental; identifies three distinct normalization challenges (temporal, spatial, conditional) that RevIN conflates — directly supports H5's finding that RevIN fails on multi-condition FD002/FD004.

---

**[26] (Anonymous, 2025)**
"Noise or Signal? Deconstructing Contradictions and An Adaptive Remedy for Reversible Normalization in Time Series Forecasting."
*arXiv preprint*, October 2025. arXiv: 2510.04667
URL: https://arxiv.org/abs/2510.04667
**Used in:** H5
**Note:** Identifies systematic failure modes of RevIN-style normalization when instance statistics encode non-degradation variance (regime shifts) — precisely the mechanism by which RevIN conflates operating-condition offsets with degradation trends in FD002/FD004.

---

**[27] (Anonymous, 2026)**
"Early Fault Detection on CMAPSS with Unsupervised LSTM Autoencoders."
*arXiv preprint*, January 2026. arXiv: 2601.10269
URL: https://arxiv.org/abs/2601.10269
**Used in:** H5, H6
**Note:** Demonstrates regression-based operating-condition normalization for turbofan sensor data; explicitly shows that raw normalization without condition decoupling is inadequate for FD002/FD004. Also motivates unsupervised fault onset detection for H6.

---

**[28] Sun, Z. et al. (2025)**
"IN-Flow: Instance Normalization Flow for Non-Stationary Time Series Forecasting."
*Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 2025.
DOI: 10.1145/3690624.3709260 | arXiv: https://arxiv.org/abs/2401.16777
**Used in:** H5
**Note:** Frames per-instance normalization as a distribution transformation problem; shows fixed instance statistics (RevIN) are insufficient when underlying regimes shift — theoretical grounding for RevIN's underperformance in multi-condition prognostics.

---

**[29] Deng, S. et al. (2024)**
"Prediction of Remaining Useful Life of Aero-Engines Based on CNN-LSTM-Attention."
*International Journal of Computational Intelligence Systems*, 2024 (69 citations).
DOI: 10.1007/s44196-024-00639-w
URL: https://link.springer.com/article/10.1007/s44196-024-00639-w
**Used in:** H5
**Note:** Applies global (fleet-level) feature normalization across all four CMAPSS subsets and achieves competitive RMSE; effectively validates fleet MinMax as the field-standard preprocessing choice.

---

### H6 — Fault Mode Separation

**[30] (Authors not retrieved, 2024)**
"A Novel Multi-Task Learning Framework with Fault Mode Feature Separation for Remaining Useful Life Estimation of Mechanical Systems."
*Advanced Engineering Informatics*, Vol. 60, 2024, Article 102360.
DOI: 10.1016/j.aei.2024.102360
URL: https://www.sciencedirect.com/science/article/abs/pii/S1474034624007043
**Used in:** H6
**Note:** Closest supervised counterpart to H6 — uses unsupervised autoencoder + K-means to cluster fault modes then trains a multi-task Self-Attention Capsule Network with FM-identification and RUL sub-networks. Requires the clustering step separate from the RUL model (unlike H6 M3's end-to-end approach).

---

**[31] (Authors not retrieved, 2025)**
"Unsupervised Classification and Remaining Useful Life Prediction for Turbofan Engines Using Autoencoders and Gaussian Mixture Models: A Comprehensive Framework for Predictive Maintenance."
*Applied Sciences*, Vol. 15, No. 14, Article 7884, 2025.
DOI: https://www.mdpi.com/2076-3417/15/14/7884
**Used in:** H6
**Note:** Directly validates the GMM + LSTM pipeline — autoencoder → GMM clustering → state-specific LSTM+attention on CMAPSS (including FD003/FD004); nearest published implementation to H6 Phase 1+2 pipeline.

---

**[32] (Authors not retrieved, 2025)**
"Multi-Condition Remaining Useful Life Prediction Based on Mixture of Encoders (MoEFormer)."
*Entropy*, Vol. 27, No. 1, Article 79, January 2025.
DOI: https://www.mdpi.com/1099-4300/27/1/79
**Used in:** H6
**Note:** Soft-gated mixture-of-encoders Transformer applied to FD002/FD004; the closest published architecture to H6's M2 (soft-gating branch). Reduces NASA Score by 38.2% and 35% on FD002/FD004 vs second-best competitor.

---

**[33] (Authors not retrieved, 2024)**
"Remaining Useful Life Prediction for Aircraft Engines under High-Pressure Compressor Degradation Faults Based on FC-AMSLSTM."
*Aerospace*, Vol. 11, No. 4, Article 293, 2024.
DOI: https://www.mdpi.com/2226-4310/11/4/293
**Used in:** H6
**Note:** Explicitly addresses the HPC vs. fan degradation mixture in FD003/FD004; proposes a decline-index fault classification and decoupling stage before CNN+LSTM — directly motivates H6's fault separation design.

---

**[34] (Authors not retrieved, 2026)**
"Prognostics of Multisensor Systems with Unknown and Unlabeled Failure Modes via Bayesian Nonparametric Process Mixtures."
*arXiv preprint*, February 2026. arXiv: 2602.19263
URL: https://arxiv.org/abs/2602.19263
**Used in:** H6
**Note:** Proposes a Dirichlet process mixture model that jointly discovers an unknown number of failure modes without labels — contextualizes why H6's GMM-based unsupervised discovery is important and shows the field's movement toward label-free fault identification.

---

**[35] (Authors not retrieved, 2023)**
"Fault Prognosis of Turbofan Engines: Eventual Failure Prediction."
*International Journal of Prognostics and Health Management (IJPHM)*, 2023.
DOI: 10.36001/ijphm.2023.v14i2.3486
URL: https://papers.phmsociety.org/index.php/ijphm/article/view/3486
**Used in:** H6
**Note:** Analyses FD003/FD004 failure mode complexity (HPC + fan degradation) and establishes why co-mingling both fault modes in a single model degrades performance — foundational motivation for H6's separation hypothesis.

---

**[36] (Authors not retrieved, 2025)**
"Interpretable Ensemble Remaining Useful Life Prediction Enables Dynamic Maintenance Scheduling for Aircraft Engines."
*Scientific Reports*, 2025.
DOI: https://www.nature.com/articles/s41598-025-23473-2
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12615660/
**Used in:** H6
**Note:** Benchmarks LightGBM + CatBoost + Gradient Boosting ensemble across all four CMAPSS sub-datasets with SHAP interpretability — recent SOTA ensemble baseline for positioning H6 M3's deep-learning approach.

---

### H7 — Loss Functions

**[37] Rengasamy, D., Jafari, M., Rothwell, B., Chen, X., & Figueredo, G. P. (2020)**
"Deep Learning with Dynamically Weighted Loss Function for Sensor-Based Prognostics and Health Management."
*Sensors*, Vol. 20, No. 3, Article 723 (103 citations).
DOI: 10.3390/s20030723
URL: https://www.mdpi.com/1424-8220/20/3/723 | PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7038523/
**Used in:** H7
**Loss proposed:** (a) Dynamically-weighted MSE (L3 motivation); (b) focal loss adapted to regression (L4 motivation)
**Note:** Primary motivating paper for H7 L3/L4; evaluates on CMAPSS with BiGRU and BiLSTM. Single-run results without multi-seed statistical correction — H7 provides the first BH-FDR-corrected comparison of these loss types.

---

**[38] Rengasamy, D., Bhatt, H., Rothwell, B., Chen, X., & Figueredo, G. P. (2020)**
"Asymmetric Loss Functions for Deep Learning Early Predictions of Remaining Useful Life in Aerospace Gas Turbine Engines."
*2020 International Joint Conference on Neural Networks (IJCNN)* (16 citations).
URL: https://consensus.app/papers/details/1707b0b881f75caf95fbecb9864bf2e0/
**Used in:** H7
**Loss proposed:** Four asymmetric variants (MSLogE-MSE, Linear-MSE, Linear-Linear, Quadratic-Quadratic) targeting early-prediction bias.
**Note:** Direct motivation for H7 L5 (TWA) and L7 (HubA); claims NASA score improvement on specific datasets without multi-seed variance reporting.

---

**[39] Liu, Z. et al. (2021)**
"A Multi-Head Neural Network with Unsymmetrical Constraints for Remaining Useful Life Prediction."
*Advanced Engineering Informatics*, 2021 (47 citations).
URL: https://consensus.app/papers/details/c3f7da83796254a78e8c339bba91e598/
**Used in:** H7
**Loss proposed:** Adjustable unsymmetrical penalty loss (larger penalty for late predictions).
**Note:** Claims 24.09% lower NASA score on FD004 vs. best prior method — without multi-seed variance, so statistical robustness is unclear. H7 tests equivalent asymmetry designs with 5-seed BH-FDR correction.

---

**[40] Diao, R. et al. (2026)**
"Turbofan Engine Remaining Useful Life Prediction with Reliable Prediction Intervals via LSTM-Based Quantile Regression and Conformal Calibration."
*Sensors (MDPI)*, Vol. 26, No. 7, Article 2249, 2026.
DOI: https://www.mdpi.com/1424-8220/26/7/2249
**Used in:** H7
**Loss proposed:** Weighted pinball (quantile) loss + asymmetric overestimation penalty.
**RMSE FD001:** 16.24 ± 1.30
**Note:** Most direct antecedent for H7 L6 (Pinball); applies to CMAPSS FD001/FD002. Point-prediction RMSE comparable to MSE baselines — consistent with H7's finding that no custom loss decisively dominates.

---

**[41] Chung, Y. et al. (2021)**
"Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification."
*NeurIPS 2021* (131 citations).
URL: https://consensus.app/papers/details/f53034955fb452d0a5987ec4d5b51cd7/
**Used in:** H7
**Note:** Foundational critique of pinball loss; shows it restricts model class and may produce poorly calibrated conditional quantiles — directly relevant to why H7 L6 (Pinball) shows no statistically significant gain and to the tau hyperparameter sensitivity found in Phase 3c.

---

**[42] Asif, O. et al. (2022)**
"A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset."
*IEEE Access*, 2022 (75 citations).
URL: https://consensus.app/papers/details/2c8fa9269bd956ad80e9df739b5981d6/
**Used in:** H7
**Loss proposed:** Standard MSE with improved RUL labeling (adaptive clipping via correlation-based degradation onset).
**Note:** Achieves competitive RMSE purely through label engineering without changing the loss function — directly supports H7's conclusion that clipping/label construction dominates over loss function design.

---

### New Additions — H6 Fault-Mode Architecture (MoE Context)

**[43] Ly, S., Yang, R., Dixit, N., & Nguyen, H. D. (2025)**
"RUL-QMoE: Multiple Non-crossing Quantile Mixture-of-Experts for Probabilistic Remaining Useful Life Predictions of Varying Battery Materials."
*arXiv preprint*, December 2025. arXiv: 2512.23725. Extended version for IAAI-26 (38th AAAI Conf. on Innovative Applications of AI).
URL: https://arxiv.org/abs/2512.23725
**Used in:** H6
**Note:** Probabilistic MoE for battery RUL prediction — five cathode-material-specific expert branches (LFP, NCA, NMC, LCO, NMC-LCO) with non-crossing quantile regression for uncertainty quantification. Differentiation from H6 M3: RUL-QMoE targets probabilistic interval prediction for battery degradation using supervised material-type labels to assign experts; M3 targets deterministic point prediction with unsupervised fault-mode routing for turbofan sensor sequences. The approaches share the MoE gating concept but serve orthogonal goals (uncertainty quantification vs. deterministic RMSE minimisation) and address distinct data modalities (chemistry-labelled battery cycles vs. unlabelled turbofan windows).

---

**[44] Yang, B., Zhang, J., Liu, R., Lin, D., Li, P., & Chen, C. L. P. (2025)**
"Point-to-Set Metric-Gated Mixture of Experts for Multisource Domain Adaptation Fault Diagnosis."
*IEEE Transactions on Neural Networks and Learning Systems*, Early Access, March 2025.
DOI: 10.1109/TNNLS.2025.3548894
URL: https://ieeexplore.ieee.org/document/10934148/
**Used in:** H6
**Note:** Proposes PSMMoEs — a mixture-of-experts framework where the gating mechanism uses deep point-to-set distance metric learning to assign unlabelled target samples to their most similar source domain expert, enabling multisource unsupervised domain adaptation for fault classification. Most direct architectural competitor to H6 M3 in PHM literature. Differentiation from H6 M3: uses metric-learning gating for cross-domain fault *classification* with multiple labelled source domains; M3 uses early-cycle attention gating for *RUL regression* within a single dataset without any domain or fault-type labels. Venue hint in reviewer brief cited Springer Complex & Intelligent Systems (likely confused with a related Springer MoE paper); verified venue is IEEE TNNLS DOI above.

---

### New Addition — Introduction Architectural Frontier Benchmark

**[45] Fan, Z., Li, W., & Chang, K.-C. (2024)**
"A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine Remaining Useful Life Prediction."
*Sensors*, Vol. 24, No. 3, Article 824.
DOI: 10.3390/s24030824
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10857698/
URL: https://www.mdpi.com/1424-8220/24/3/824
**Used in:** Background (Introduction)
**Note:** STAR (two-Stage ATtention-based hieRarchical Transformer) — sequential temporal and sensor-wise variable attention with hierarchical encoder-decoder for multi-scale time dependency on all four CMAPSS sub-datasets; outperforms prior SOTA at publication. Represents the attention-based Transformer frontier against which this paper's LSTM backbone is deliberately positioned in the Introduction; cited to acknowledge that more powerful architectures exist and to justify the ablation study's use of a controlled LSTM backbone rather than a state-of-the-art encoder.

---

## Key Gaps This Paper Fills

Based on the literature survey:

1. **H2**: No paper (through June 2026) performs a controlled quantitative comparison of clip ∈ {75, 100, 125, 130, None} with both RMSE and NASA score across all four CMAPSS sub-datasets. The catastrophic FD003 NASA score under clip=None has no prior quantification.

2. **H5**: No peer-reviewed paper performs a direct fleet-level vs. per-unit normalization ablation on CMAPSS with multi-seed statistical significance testing. MOC-based normalization [24] is the closest prior work but does not compare against RevIN or per-unit alternatives.

3. **H6**: Published unsupervised fault-mode separation work [31] uses a two-stage pipeline (separate clustering then training). H6 M3's end-to-end attention gating from first-10-cycles observations, with no cluster labels at any stage, appears to be novel. The test-time cluster collapse on FD004 (1:247 distribution for M1) is a new negative finding with no published parallel.

4. **H7**: Papers [37, 38, 39] propose individual custom losses with single-run results. No paper applies Benjamini-Hochberg FDR correction across 7 loss functions × 4 datasets × 4 clipping values simultaneously. The cross-dataset, multi-seed, BH-FDR-corrected comparison appears to be the first of its kind.

---

*Sources: Consensus (consensus.app), IEEE Xplore, arXiv, MDPI, ScienceDirect, Nature/Scientific Reports, ACM DL, PHM Society papers.phmsociety.org. Search conducted through June 2026.*

---

## New Citations — Discussion Placement

The following notes specify exactly where each newly added or confirmed citation should be inserted in the manuscript. **Do not modify any section .md files** — these are instructions for the next editing pass.

### [43] Ly et al. 2025 — RUL-QMoE
**Target section:** Discussion §V.D ("What M3's Success Reveals About Fault-Mode Routing")
**Target sentence:** "From an architecture standpoint, M3 is an instance of mixture-of-experts (MoE) inference [32] with an early-cycle context encoder as the gating network and no label supervision for the gate."
**Action:** Extend the citation cluster after [32] to include [43, 44], then add one sentence of differentiation:
> "…mixture-of-experts (MoE) inference [32, 43, 44] with an early-cycle context encoder as the gating network and no label supervision for the gate. Unlike RUL-QMoE [43], which targets probabilistic interval prediction for battery degradation using supervised material labels, and PSMMoEs [44], which applies metric-learning gating for cross-domain fault classification, M3 performs deterministic point-prediction RUL routing in a single unlabelled turbofan dataset."

### [44] Yang et al. 2025 — Point-to-Set Metric-Gated MoE (IEEE TNNLS)
**Target section:** Discussion §V.D (same sentence and addition as [43] above — both refs cited together)
**Note:** Venue in reviewer brief ("Springer Complex & Intelligent Systems") was incorrect; verified as IEEE Transactions on Neural Networks and Learning Systems (DOI: 10.1109/TNNLS.2025.3548894).

### [25] Berthelier et al. 2026 — On the Role of RevIN (arXiv:2603.11869)
**Status: ALREADY IN PAPER as ref [25] — no new ref number assigned.**
**Target section:** Discussion §V.B ("Why Fleet Normalization Beats Per-Unit Strategies")
**Target sentence:** "A recent theoretical critique of RevIN [26] reached a compatible conclusion: RevIN's components are redundant when the primary normalisation challenge is not temporal distribution shift but conditional offset."
**Action:** Add [25] alongside [26]: "…theoretical critique of RevIN [25, 26] reached a compatible conclusion…"
**IMPORTANT — .ris file correction required:** The file `Citation/ref25_berthelier2026.ris` previously contained a WRONG paper (DLinear photovoltaic forecasting by Wang et al., DOI: 10.1109/ddcls58216.2023.10166973). It has been corrected in this session to the Berthelier et al. arXiv:2603.11869 paper with the full six-author list. Verify this fix before submission.
**Correct authors:** Gaspard Berthelier, Tahar Nabil, Etienne Le Naour, Richard Niamke, Samir Perlaza, Giovanni Neglia.

### [45] Fan, Li & Chang 2024 — STAR Transformer (Sensors 24(3):824)
**Target section:** Introduction §I, first paragraph
**Target sentence:** "…as model architectures have evolved from linear regression [2] to stacked LSTM [5, 6] and attention-based encoders [13, 14], reported RMSE on FD001 has fallen from above 25 cycles to below 13 [15]."
**Action:** Add [45] to the attention-based encoder cluster: "…attention-based encoders [13, 14, 45]…"
**Rationale:** STAR is the most direct recent Transformer SOTA on CMAPSS FD001 (2024, all four sub-datasets) and positions the deliberate LSTM backbone choice.

### Reviewer Paper 5 — Wu et al. 2024 Survey (PMC11174398)
**Status: DUPLICATE of existing ref [16] — do NOT add as ref [47].**
**Explanation:** The paper "Remaining Useful Life Prediction Based on Deep Learning: A Survey" by Fuhui Wu, Qingbo Wu, Yusong Tan, Xinghua Xu (Sensors 2024, Vol. 24, No. 11, Article 3454, DOI: 10.3390/s24113454, PMC11174398) is the same paper as ref [16] in the current reference list. Ref [16] contains incorrect metadata:
- Wrong DOI listed: 10.3390/s24082543 (this DOI resolves to an unrelated wrist-kinematics paper by Dellai et al.)
- Wrong issue listed: No. 8 (should be No. 11)
- Wrong article number: 2543 (should be 3454)
**Required correction to ref [16]:** Update the DOI to `10.3390/s24113454`, issue to `No. 11`, article number to `3454`, and add PMC link `https://pmc.ncbi.nlm.nih.gov/articles/PMC11174398/`. The title, authors, year, and journal are correct as listed.

---

*New citations added: 2026-07-03. Papers 43–45 verified via arXiv, IEEE Xplore (PubMed proxy), and PMC.*
