# Cover Letter — IEEE Transactions on Industrial Informatics

> **Status:** Draft v3.0 — 2026-07-07 (TII-specific rewrite; addresses all Stage 4 checklist items)
> **Target:** IEEE Transactions on Industrial Informatics (TII)
> **Action required before submission:** Fill in `[Author Name]` and `[Position/Title]`; replace Gmail with ETRI institutional email

---

[Author Name]
Electronics and Telecommunications Research Institute (ETRI)
Daejeon, Republic of Korea
[ETRI institutional email — required; Gmail not accepted]

July 2026

The Editor-in-Chief
IEEE Transactions on Industrial Informatics

---

Dear Editor-in-Chief,

We submit for your consideration the manuscript entitled **"From Fleet Normalization to Fault-Mode Gating: A Cross-Dataset Ablation Study of Turbofan Remaining Useful Life Prediction"** for publication as a Regular Paper in *IEEE Transactions on Industrial Informatics*.

**Why IEEE TII.** This work is submitted to TII rather than to sister IEEE IES venues because its central contribution is industrial-informatics in nature, not electronic hardware or control theory. IEEE Transactions on Industrial Electronics (TIE) requires experimental verification on physical hardware rigs; the NASA CMAPSS benchmark used here is the accepted gold standard for data-driven turbofan PHM research and does not involve hardware beyond sensors already embedded in simulated engines. IEEE Transactions on Control Systems Technology (TCST) focuses on control-system design and stability; no new controller is proposed here. The paper's contribution — a cross-dataset, multi-factor ablation that delivers actionable design guidance for practitioners deploying data-driven condition-monitoring systems — falls squarely within TII's stated scope of AI-enhanced industrial automation, industrial cyber-physical systems, and data-driven predictive maintenance.

**Outstanding and original contribution.** TII's acceptance criteria require work that is outstanding and original, not merely technically correct. We submit that this manuscript satisfies that standard on three grounds. First, it is the first study to conduct a controlled, statistically rigorous ablation of four interdependent pipeline design factors — RUL label engineering, sensor normalization, fault-mode architecture, and training loss function — simultaneously across all four NASA CMAPSS sub-datasets, totalling over 800 LSTM training runs with all comparisons corrected by Benjamini-Hochberg FDR. Prior work evaluates these factors in isolation and predominantly on a single sub-dataset, making cross-factor attribution impossible. Second, the attention-gate architecture (M3) achieves a 65.8% RMSE reduction on FD003 (14.78 ± 1.32 vs. 43.23 ± 0.18) without any fault-mode supervision, and does so from as few as five initial flight cycles — a result that resolves a long-standing high-variance anomaly in the normalization literature and advances the state of the art on a widely cited benchmark. Third, the paper establishes a three-tier design hierarchy — label engineering, fault-mode architecture, loss function — that is novel in its framing and practically significant: it demonstrates that clipping threshold errors inflate the NASA prognostic score by up to six orders of magnitude regardless of all other design choices, a finding with direct operational consequences for industrial maintenance systems that has not previously been quantified across datasets.

**Summary of findings.** The controlled ablation yields four principal results: (i) clip = 125 cycles is confirmed as the cross-dataset optimal RUL ceiling, and its omission is catastrophic on multi-fault datasets (FD003 NASA Score: 4,014,724 vs. 13.09 at clip = 125); (ii) fleet-level min-max normalization significantly outperforms all per-unit and RevIN strategies on three of four sub-datasets, and the anomalous inter-seed RMSE variance on FD003 (std = 12.86 vs. ≤ 1.84 elsewhere) is a diagnostic signature of latent fault-mode heterogeneity, not a normalization deficiency; (iii) the M3 attention-gate model routes engines to fault-specific branches from early-cycle observations alone, achieving a 65.8% RMSE improvement on FD003 while remaining statistically equivalent to the baseline on multi-condition FD004 — immune to the test-time cluster-distribution collapse that degrades GMM hard-routing by 75.4% on FD004; and (iv) no custom loss function achieves a statistically detectable improvement over MSE after BH-FDR correction across 96 comparisons, establishing loss function selection as a second-order design choice once clipping is correctly applied.

**Industrial informatics significance.** The industrial informatics contribution of this work is not incidental to its machine-learning content — it is the primary deliverable. Turbofan RUL predictors are deployed in safety-critical industrial maintenance pipelines where label engineering errors can propagate into maintenance scheduling decisions with direct cost and safety consequences. The three-tier hierarchy provides a prioritised, evidence-based decision framework that PHM system engineers can apply when configuring or auditing industrial prognostics pipelines, directing development effort toward the factors that demonstrably determine end-to-end reliability. M3's commissioning-time fault-mode routing — operative within five flight cycles and requiring no fault-mode labels — is directly deployable in fleet management systems that must initialise maintenance models before degradation history is available. We believe these results will be of immediate practical value to the segment of the TII readership working on industrial condition monitoring, data-driven maintenance, and cyber-physical systems for industrial asset management.

**Originality and ethical compliance.** This manuscript is original work, has not been previously published, and is not currently under review at any other journal or conference. All authors have approved the submission. The study is conducted entirely on publicly available benchmark data (NASA CMAPSS) and involves no human subjects, no identifiable operational records, and no conflicts of interest.

Thank you for your time and consideration. We look forward to the possibility of contributing to IEEE Transactions on Industrial Informatics.

Sincerely,

[Author Name]
[Position/Title]
Electronics and Telecommunications Research Institute (ETRI)
[ETRI institutional email]

---

> **Pre-submission checklist for this letter:**
> - [ ] Replace `[Author Name]` and `[Position/Title]`
> - [ ] Replace Gmail with ETRI institutional email (mandatory; see Searching_Target_Journal.md item A)
> - [ ] Confirm EiC name at [ieee-ies.org/pubs/transactions-on-industrial-informatics](https://www.ieee-ies.org/pubs/transactions-on-industrial-informatics) — if named EiC preferred, replace "Dear Editor-in-Chief"
> - [ ] Add co-author names and affiliations if applicable
> - [ ] Keep to one printed page when converting to PDF (trim "Summary of findings" paragraph if over length)
