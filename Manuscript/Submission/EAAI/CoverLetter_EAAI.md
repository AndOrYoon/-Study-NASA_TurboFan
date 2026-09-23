# Cover Letter — Engineering Applications of Artificial Intelligence

> **Status:** Draft v1.0 — 2026-09-23
> **Target:** Engineering Applications of Artificial Intelligence (EAAI, Elsevier/IFAC)
> **Action required before submission:** EAAI Editor-in-Chief 성함 확인 (저널 홈페이지); Elsevier Editorial Manager 제출 시 "Previously submitted elsewhere?" 질문에 RESS 데스크 리젝션 사실 기재 필요 (scope 불일치로 심사 미진입 — 내용 수정 없음)

---

Young Seog Yoon, Ph.D.
Principal Researcher
Behavioral Intelligence for Autonomous Manufacturing Research Section
Electronics and Telecommunications Research Institute (ETRI)
Daejeon, Republic of Korea
isay@etri.re.kr

23 September 2026

Prof. Patrick Siarry, Ph.D.
Editor-in-Chief
Engineering Applications of Artificial Intelligence

---

Dear Prof. Siarry,

We respectfully submit for your consideration the manuscript entitled **"From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction"** for publication as a full-length article in *Engineering Applications of Artificial Intelligence*. This work is original, has not been previously published, and is not under concurrent review at any other journal or conference. The manuscript has been prepared in accordance with the journal's double anonymized review policy.

Deploying data-driven Remaining Useful Life (RUL) predictors for turbofan engines requires a sequence of interdependent design decisions — RUL label engineering, sensor normalization strategy, fault-mode routing architecture, and training loss function — whose individual contributions to predictive accuracy have not been systematically isolated. Without a controlled attribution study, engineers configuring PHM pipelines cannot determine where to direct development effort, and reported improvements in the literature remain bundled and difficult to reproduce. This attribution problem is both an AI methodology challenge — requiring a rigorous ablation design and multi-hypothesis statistical framework — and a direct engineering concern, as we demonstrate that a single label-engineering misconfiguration inflated the NASA prognostic score by up to 306,000-fold on multi-fault datasets.

This manuscript reports four principal contributions. **As an AI contribution**, we develop a controlled stacked Long Short-Term Memory (LSTM) ablation framework evaluating five clipping thresholds, seven normalization strategies, four fault-mode routing architectures, and seven training loss functions across all four C-MAPSS sub-datasets (FD001–FD004), with all hypothesis comparisons corrected using the Benjamini-Hochberg False Discovery Rate (FDR) procedure across over 1,000 training runs. **Applied as an engineering tool for turbofan health management**, the framework yields the following findings: (i) fleet-level min-max normalization significantly outperforms per-unit and instance-normalization (RevIN) alternatives on three of four sub-datasets (p_BH = 0.009 on FD003), with the advantage being cross-protocol robust on FD001 and FD002; (ii) GMM-based hard fault-mode routing is statistically significantly worse than the single-model baseline on both multi-fault datasets (p_BH = 0.0045), while soft and end-to-end attention routing recover to baseline performance — the M3 Attention Gate's auxiliary branch loss structure provides training robustness by avoiding mean-prediction collapse under adverse validation splits, with FD004 inter-seed RMSE standard deviation reduced from 3.97 to 1.04; (iii) no custom loss function achieves statistically detectable improvement over MSE after FDR correction across 96 pairwise comparisons; and (iv) RUL clipping misconfiguration is the dominant source of prognostic score variance, inflating the NASA metric by up to six orders of magnitude and establishing label engineering as the highest-priority design decision. Together, these findings yield an evidence-based design priority ordering — label engineering, normalization, fault-mode architecture, loss function — that offers a structured starting point for PHM pipeline configuration on CMAPSS-class degradation data.

We are confident that this manuscript falls within the core scope of *Engineering Applications of Artificial Intelligence*. The journal explicitly covers "Intelligent fault detection, fault analysis, diagnostics and monitoring" and "Deep learning and real-world engineering applications," both of which are central to this work. Our controlled ablation methodology — combining stacked LSTM architectures, multi-strategy normalization, GMM-based and attention-gated routing, and asymmetric loss functions — represents a direct contribution to AI methodology as applied to an industrial engineering benchmark. The NASA C-MAPSS benchmark has been used in multiple prior EAAI publications, and our study directly addresses the attribution gap that makes prior bundled-system results difficult to interpret or transfer. The experimental code and result files are publicly archived at Zenodo (DOI: 10.5281/zenodo.22910912), supporting full reproducibility.

This study is conducted entirely on publicly available benchmark data (NASA C-MAPSS, provided by NASA Glenn Research Center) and requires no proprietary datasets or physical hardware. The authors declare no conflicts of interest. This work was supported by the National IT Industry Promotion Agency (NIPA) grant funded by the Korea government (MSIT): "Development of Physical Data Quality Management Technologies for Physical AI," No. RS-2026-25621690. All authors have read and approved the final manuscript.

Thank you for your time and consideration. We look forward to the possibility of contributing to *Engineering Applications of Artificial Intelligence*.

Sincerely, on behalf of all co-authors,

Young Seog Yoon, Ph.D.
Principal Researcher (Corresponding Author)
Behavioral Intelligence for Autonomous Manufacturing Research Section
Electronics and Telecommunications Research Institute (ETRI)
isay@etri.re.kr

---

## Pre-submission Checklist

| 항목 | 상태 |
|------|------|
| Editor-in-Chief 성함 확인 (EAAI 저널 홈페이지) | ⬜ 제출 직전 확인 필요 |
| 날짜 교체 (제출 당일로 수정) | ⬜ |
| Elsevier EM: "Previously submitted?" → RESS 데스크 리젝션 기재 | ⬜ (scope 불일치, 심사 미진입, 내용 무수정) |
| Zenodo DOI 최종 확인 (10.5281/zenodo.22910912) | ✅ |
| 기관 이메일 사용 (isay@etri.re.kr) | ✅ |
| Funding 과제번호 (RS-2026-25621690) | ✅ |
| 추천 리뷰어 2–3인 준비 (선택) | ⬜ |
| main_EAAI_anon.tex — 저자 정보 완전 제거 확인 | ✅ |
| highlights.txt 첨부 파일 준비 | ✅ |
| titlepage_EAAI.tex (저자 정보 별도 파일) 준비 | ✅ |
