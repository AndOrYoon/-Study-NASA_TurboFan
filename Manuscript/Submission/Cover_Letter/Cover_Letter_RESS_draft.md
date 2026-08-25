# Cover Letter — Reliability Engineering & System Safety

> **Status:** Draft v1.2 — 2026-07-30 (저자 정보 반영: Young Seog Yoon, isay@etri.re.kr)
> **Target:** Reliability Engineering & System Safety (Elsevier)
> **Action required before submission:** Editor-in-Chief 성함 확인 (RESS 저널 페이지); 공동저자 있을 경우 서명란 추가

---

Young Seog Yoon, Ph.D.
Principal Researcher
Behavioral Intelligence for Autonomous Manufacturing Research Section
Electronics and Telecommunications Research Institute (ETRI)
Daejeon, Republic of Korea
isay@etri.re.kr

August 2026

Prof. Marko Cepin
Editor-in-Chief
Reliability Engineering & System Safety

---

Dear Professor Cepin,

We respectfully submit for your consideration the manuscript entitled **"From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction"** for publication as a full-length article in *Reliability Engineering & System Safety*. This work is original, has not been previously published, and is not under concurrent review at any other journal or conference. The manuscript has been prepared in accordance with the journal's single-blind review policy.

Remaining Useful Life (RUL) prediction for turbofan engines is a foundational task in Prognostic and Health Management (PHM) and bears directly on safety-critical maintenance scheduling decisions in civil and military aviation. Despite a decade of rapid progress in data-driven predictors, the reliability of deployed PHM systems remains constrained by an unresolved practical question: which pipeline design decisions — among RUL label engineering, sensor normalization strategy, degradation-mode architecture, and training loss function — are the primary drivers of predictive reliability, and in what priority order should they be addressed? Answering this question rigorously is essential for engineers who must configure, validate, and certify RUL predictors in high-consequence environments where miscalibration carries direct safety and economic consequences.

This manuscript reports four principal contributions. First, we present the first statistically rigorous ablation of seven sensor normalization strategies across all four NASA CMAPSS sub-datasets, demonstrating that fleet-level Min-Max normalization significantly outperforms per-unit and instance-normalization (RevIN) approaches under Benjamini-Hochberg FDR correction; we further show that anomalous inter-seed RMSE variance on the multi-fault sub-dataset FD003 is a diagnostic signal of latent fault-mode heterogeneity rather than a normalization artifact. Second, we introduce the M3 Attention Gate — a 4,900-parameter GatingNet trained on the first ten cycles — which achieves a 65.8% RMSE reduction on FD003 (14.78 ± 1.32 vs. 43.23 ± 0.18 cycles) without fault-mode supervision and without modifying the shared LSTM backbone. A systematic false-routing sensitivity analysis further examines gate confidence — mean max(w₀, w₁) over training engines — as a post-training indicator of routing decisiveness: on FD003 (gate confidence = 0.840), forced gate inversion raises RMSE by 48% and NASA Score 12-fold, while on FD004 (confidence = 0.677) the same perturbation yields only +1.6% — suggesting that gate confidence may serve as a useful pre-deployment diagnostic, though these two data points are insufficient to establish a calibrated threshold. Third, we find evidence of a three-tier priority ordering — label engineering, fault-mode architecture, loss function — across more than 800 LSTM training runs, offering a structured starting point for reliability engineers configuring RUL systems. Fourth, we confirm that no custom loss function achieves statistically detectable improvement over MSE after BH-FDR correction, and that RUL clipping misconfiguration inflates NASA prognostic scores by up to six orders of magnitude — establishing that label engineering is a safety-relevant design decision that should precede all other pipeline configuration choices.

We are confident that this manuscript falls within the core scope of *Reliability Engineering & System Safety*. The journal has previously published foundational studies on data-driven turbofan RUL prediction that this work directly extends: Li et al. (2018), Ellefsen et al. (2019), Xu et al. (2022), and Zhang et al. (2023) all appear in our reference list. Beyond that lineage, the paper's reliability significance is substantive: omitting RUL label clipping raises the NASA prognostic score by up to six orders of magnitude on multi-fault datasets (FD003 unclipped: 4,014,724 vs. clipped: 13.09), illustrating how a single preprocessing misconfiguration can constitute the most severe performance degradation observed across all experiment conditions. The three-tier design hierarchy directly supports the kind of actionable, system-level engineering guidance that is central to RESS's editorial mission.

The study is conducted entirely on publicly available benchmark data (NASA CMAPSS, provided by NASA Glenn Research Center) and requires no proprietary datasets or physical hardware. All experimental code and result logs are available for sharing upon editorial request. The authors declare no conflicts of interest. No external funding sources with competing interests were involved. All authors have read and approved the final manuscript.

Thank you for your time and consideration. We look forward to the possibility of contributing to *Reliability Engineering & System Safety*.

Sincerely, on behalf of all co-authors,

Young Seog Yoon, Ph.D.
Principal Researcher (Corresponding Author)
Behavioral Intelligence for Autonomous Manufacturing Research Section
Electronics and Telecommunications Research Institute (ETRI)
isay@etri.re.kr

---

> **Pre-submission checklist for this letter:**
> - [x] Replace `[Author Name]` and `[Position/Title]` → Young Seog Yoon, Principal Researcher
> - [x] Replace Gmail with ETRI institutional email → isay@etri.re.kr
> - [x] Confirm current Editor-in-Chief name → Prof. Marko Čepin (confirmed from RESS journal homepage, 2026-08-25)
> - [x] Verify all four RESS reference citations (Li 2018 line 672, Ellefsen 2019 line 674, Xu 2022 line 680, Zhang 2023 line 682) — all confirmed in main.tex
> - [x] Confirm review policy → single-blind confirmed; paragraph 1 already correct
> - [x] Add co-author names and affiliations to signature block → "on behalf of all co-authors" 방식 적용 (공저자 3명: Eun Seo Lee, Hyeontae Kim, Ji Yeon Son — 모두 ETRI)
>
> **⚠️ 투고 당일 최종 확인 필수:**
> - [ ] **날짜 교체** — `August 2026` → 실제 투고일 (예: `28 August 2026`)
> - [ ] **Funding 과제 번호** — `main.tex` line 747의 `No. XXXX` → 실제 NIPA 과제 번호로 교체 (`manuscript_full_text.md` Funding 섹션도 동일하게)
> - [x] **AI 사용 선언** — `declaration_ai_use.txt` 내용 확인 완료 ("drafting" 삭제, 2026-08-25); Editorial Manager 업로드는 투고 시
> - [ ] **단어 수** — texcount 기준 현재 12,312 / 13,000 (688 여유); 최종 원고 반영 후 재확인
> - [x] **Figures 폴더** — `Manuscript/Submission/RESS/Figures/`에 Fig1–Fig8.png 확인 완료 (2026-08-25)
> - [x] **elsarticle.cls** — `Manuscript/Submission/RESS/`에 확인 완료 (2026-08-25)
> - [ ] **추천 리뷰어** — 2–3인 준비 (PHM/RESS 게재 경험자)
