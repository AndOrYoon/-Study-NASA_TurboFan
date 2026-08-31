# Cover Letter — Reliability Engineering & System Safety

> **Status:** Draft v2.0 — 2026-08-31 (FFN revision 반영: M3 결과 업데이트, three-tier→attribution checklist, Related Work §II 추가, gate confidence 수정)
> **Target:** Reliability Engineering & System Safety (Elsevier)
> **Action required before submission:** 날짜 교체; 연구비 번호 No. XXXX 확인

---

Young Seog Yoon, Ph.D.
Principal Researcher
Behavioral Intelligence for Autonomous Manufacturing Research Section
Electronics and Telecommunications Research Institute (ETRI)
Daejeon, Republic of Korea
isay@etri.re.kr

August 2026

Prof. Marko Čepin
Editor-in-Chief
Reliability Engineering & System Safety

---

Dear Professor Čepin,

We respectfully submit for your consideration the manuscript entitled **"From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction"** for publication as a full-length article in *Reliability Engineering & System Safety*. This work is original, has not been previously published, and is not under concurrent review at any other journal or conference. The manuscript has been prepared in accordance with the journal's single-blind review policy.

Remaining Useful Life (RUL) prediction for turbofan engines is a foundational task in Prognostic and Health Management (PHM) and bears directly on safety-critical maintenance scheduling decisions in civil and military aviation. The NASA prognostic score reflects the underlying risk asymmetry explicitly: late predictions carry an exponentially higher penalty than early ones, because an undetected engine failure costs orders of magnitude more than a precautionary shop visit. Despite a decade of rapid architectural advances — from stacked LSTM to Transformer encoders and hybrid attention systems — the relative contribution of individual pipeline design decisions to predictive reliability has not been rigorously isolated. The prevailing evaluation pattern bundles preprocessing, architecture, and loss function into a single proposed system, making it impossible to attribute observed performance gains to any specific component. This attribution problem motivated the present study: a controlled, statistically rigorous ablation of four pipeline design factors — RUL label engineering, sensor normalisation, fault-mode architecture, and training loss function — across all four NASA CMAPSS sub-datasets, covering more than 800 LSTM training runs with all comparisons corrected by Benjamini-Hochberg FDR. Section II of the manuscript provides a structured review of how each factor has been treated in prior work, identifying four specific attribution gaps that motivate the controlled ablation design.

This manuscript reports four principal contributions. First, we present the first multi-threshold RUL label ablation across all four CMAPSS sub-datasets using a capacity-limited OLS baseline, demonstrating that clip = 125 cycles is cross-dataset optimal or statistically tied-optimal, and that omitting the RUL ceiling entirely inflates FD003 NASA Score by 306,000-fold (4,014,724 vs. 13.09) — establishing label engineering as a safety-relevant prerequisite design decision. Second, we report the first statistically rigorous comparison of seven sensor normalisation strategies across all four CMAPSS sub-datasets under Benjamini-Hochberg FDR correction, demonstrating that fleet-level min-max normalisation significantly outperforms all per-unit and RevIN strategies on FD001 and FD002 (ΔRMSE = 5.02 and 2.15 cycles respectively; p_BH = 0.009), and a further N1–N3 unified protocol comparison confirms this advantage on FD003 (ΔRMSE = 8.36; p_BH = 0.009) with no detectable difference on FD004 (p_BH = 0.458). The original FD003 anomalous inter-seed RMSE variance (std = 12.86) resolved to std = 0.67 under the unified protocol, identifying it as a protocol-configuration effect rather than a normalisation deficiency. Third, we report a controlled comparison of four fault-mode architectures (M0–M3) on the multi-fault CMAPSS sub-datasets. GMM-based hard routing (M1) substantially degraded prognostic accuracy on both FD003 (+156% RMSE: 33.25 ± 8.74 vs. M0 baseline of 12.97 ± 0.67) and FD004 (+76%: 33.28 ± 2.05 vs. M0 of 18.96 ± 3.97), driven by training-split sensitivity of trajectory-level GMM labels on FD003 and test-time cluster-distribution collapse on FD004. The M3 Attention Gate — a 4,900-parameter GatingNet reading only the first ten flight cycles — avoided both failure modes and remained statistically equivalent to the single-branch baseline (M0) on both sub-datasets (FD003: 13.24 ± 1.69, p = 0.625; FD004: 17.30 ± 1.04, p = 0.625), as did soft-gating (M2). M0 is the recommended default architecture; M3 and M2 offer alternatives for practitioners who require explicit fault-mode routing structure without incurring hard-routing reliability risk. Mean gate weight (max(w₀, w₁)) was 0.727 ± 0.165 on FD003 and 0.676 ± 0.055 on FD004, reported descriptively as a routing decisiveness indicator. Fourth, we find that no custom loss function achieved statistically detectable improvement over MSE across 96 pairwise comparisons (6 losses × 4 clipping values × 4 datasets) under BH-FDR correction. The dominant driver of NASA Score variance was RUL clipping, not loss design: removing the RUL ceiling increased FD003 NASA Score by three to six orders of magnitude regardless of loss choice, confirming that loss-function engineering is a second-order design decision once label engineering is correctly applied. Together, these four results support an attribution-oriented sequential design checklist — verify label engineering first, evaluate normalisation strategy second, assess fault-mode routing architecture third (with M0 as default and hard GMM routing actively avoided), and treat loss-function selection as a later-stage refinement — derived from CMAPSS single-factor evidence and intended as a structured starting point for PHM pipeline configuration.

We are confident that this manuscript falls within the core scope of *Reliability Engineering & System Safety*. The journal has previously published foundational studies on data-driven turbofan RUL prediction that this work directly extends: Li et al. (2018), Ellefsen et al. (2019), Xu et al. (2022), and Zhang et al. (2023) all appear in our reference list. Beyond that lineage, the paper's reliability significance is substantive: a single label-engineering misconfiguration inflated the NASA prognostic score by up to 306,000-fold on multi-fault datasets — equivalent in practice to a prognostics system that chronically over-predicts remaining life and delays maintenance intervention. The identification of GMM hard-routing as an avoidable reliability risk (+156% RMSE on FD003) constitutes a directly actionable safety finding for PHM practitioners. The sequential design checklist directly supports the kind of system-level engineering guidance central to RESS's editorial mission.

The study is conducted entirely on publicly available benchmark data (NASA CMAPSS, provided by NASA Glenn Research Center) and requires no proprietary datasets or physical hardware. All experimental code and result logs are available for sharing upon editorial request. The authors declare no conflicts of interest. All authors have read and approved the final manuscript.

Thank you for your time and consideration. We look forward to the possibility of contributing to *Reliability Engineering & System Safety*.

Sincerely, on behalf of all co-authors,

Young Seog Yoon, Ph.D.
Principal Researcher (Corresponding Author)
Behavioral Intelligence for Autonomous Manufacturing Research Section
Electronics and Telecommunications Research Institute (ETRI)
isay@etri.re.kr

---

> **Pre-submission checklist for this letter:**
> - [x] Author name and position confirmed — Young Seog Yoon, Principal Researcher
> - [x] ETRI institutional email — isay@etri.re.kr
> - [x] Editor-in-Chief confirmed — Prof. Marko Čepin (RESS journal homepage)
> - [x] RESS reference citations confirmed (Li 2018, Ellefsen 2019, Xu 2022, Zhang 2023)
> - [x] Review policy — single-blind confirmed
> - [x] Co-author signature — "on behalf of all co-authors" (Eun Seo Lee, Hyeontae Kim, Ji Yeon Son — all ETRI)
>
> **⚠️ 투고 당일 최종 확인 필수:**
> - [ ] **날짜 교체** — `August 2026` → 실제 투고일 (예: `31 August 2026`)
> - [ ] **Funding 과제 번호** — 원고 Funding 섹션의 `No. XXXX` → 실제 NIPA 과제 번호로 교체
> - [x] **AI 사용 선언** — `declaration_ai_use.txt` 확인 완료
> - [ ] **단어 수** — 최종 원고 반영 후 재확인
> - [x] **Figures 폴더** — `Manuscript/Submission/RESS/Figures/`에 Fig1–Fig8.png 확인 완료
> - [x] **elsarticle.cls** — `Manuscript/Submission/RESS/`에 확인 완료
> - [ ] **추천 리뷰어** — 2–3인 준비 (PHM/RESS 게재 경험자)
