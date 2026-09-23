# Abstract — EAAI Submission

> **Draft status:** v2.0 — 2026-09-23 (EAAI 재작성: DR-1 AI/공학 구분 명시, DR-2 RUL·C-MAPSS·FDR 약어 정의, 단어 수 236단어)

**Title:**
From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction

---

## Abstract

Remaining Useful Life (RUL) prediction for turbofan engines involves interdependent design decisions — label clipping, sensor normalization, fault-mode routing architecture, and loss function — whose individual contributions to predictive accuracy are rarely isolated under equal experimental conditions.

As an AI contribution, this study develops a stacked Long Short-Term Memory (LSTM) ablation framework evaluating five clipping thresholds, seven normalization strategies, four fault-mode architectures, and seven loss functions across all C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) sub-datasets (FD001–FD004), with all comparisons corrected via Benjamini-Hochberg False Discovery Rate (FDR) across over 1,000 training runs.

Applied to turbofan engine health management, the framework yields three concrete findings. Fleet min-max normalization significantly outperforms per-unit and instance-level alternatives on three of four datasets (p_BH = 0.009 on FD003). GMM-based hard fault-mode partitioning substantially degrades RMSE by 76–156% on multi-fault datasets; soft and attention-gated routing recover to baseline without statistically detectable improvement. No custom loss function outperforms mean squared error after FDR correction; removing RUL clipping inflates prognostic scores up to 306,000-fold regardless of loss design.

These results provide an actionable design priority ordering for industrial RUL pipelines: label engineering and fleet normalization exert the largest effects; hard fault-mode routing is a deployment risk to manage, not a performance lever; and loss customization provides no measurable benefit under the evaluated conditions.

---

## Metadata

| Item | Detail |
|------|--------|
| Word count | ~236 words (EAAI limit: 250) |
| Target venue | Engineering Applications of Artificial Intelligence (EAAI) |
| DR-1 | AI 기여: stacked LSTM controlled ablation, BH-FDR correction (para 2) |
| DR-1 | 공학 응용: 터보팬 엔진 PHM 설계 인자 우선순위 (para 3–4) |
| DR-2 | RUL, C-MAPSS, FDR 첫 등장 시 풀어쓰기 적용 |
| Primary claim | Fleet normalization beats per-unit; hard routing is a reliability risk |
| Key number | M1: +156% RMSE on FD003, +76% on FD004; M2/M3 recover to baseline |

---

## Draft Notes

- "over 1,000 training runs" = H2 (140) + H5 (280) + H6 (40) + H7 (560+150+100+45) ≈ 1,055 runs total.
- N1–N3 표기는 abstract에서 제거 (EAAI DR-2: 미정의 약어). 해당 내용은 "fleet min-max normalization" 등 풀어쓰기로 처리.
- "p_BH = 0.009" 수치는 ad-hoc 통합 프로토콜 결과 (N1 vs N3, FD003).
- RevIN은 "instance-level alternatives"에 포함 — 리뷰어 질의 시 명시 가능.
- RESS v1.8 → EAAI v2.0 주요 변경: (1) 약어 정의 추가, (2) AI contribution para 신설, (3) engineering application 명시적 framing, (4) N1–N3 제거, (5) 단어 수 200→236.
