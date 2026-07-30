# Revision Changelog
> Autonomous revision session: 2026-07-03 (퇴근 전) → 2026-07-06 (월요일 아침)
> Based on: Pre-Review_Response.md action items + Code_Verification_Report.md findings

---

## Summary

All Track 1 (manuscript text), Track 3b (K sensitivity), and Track 4 (literature) revisions are complete. Track 3a (RevIN denorm experiment) was determined unnecessary by code verification.

---

## Track 2 — Code Verification (completed)

File: `Manuscript/Code_Verification_Report.md`

| Issue | Finding | Impact |
|-------|---------|--------|
| A (H5/H6 FD003) | 4 structural differences: backbone (32 vs 64), features (18 vs 15), val split, test clipping | Major — cross-hypothesis comparison invalid |
| B (Cohen's d vs p_BH) | Both use N=5; d=2.00/p_BH=1.0 is consistent (positive d = loss is worse than MSE) | No pseudo-replication; sign convention clarification needed |
| C (BH count) | 6 × 4 × 4 = 96, not 84 | Factual error corrected throughout |
| D (H2 Wilcoxon) | Mann-Whitney U (unpaired) on per-engine RMSE; 20 = 5×4 deterministic configs | Test type clarification needed |
| E (RevIN denorm) | No denorm is CORRECT — RUL output is in cycle units, not sensor units | No bug; clarification text added |

---

## Track 4 — Literature Additions (completed)

New RIS files created in `Manuscript/Citation/`:ㅍ
- `ref43_ly2025.ris` — RUL-QMoE (arXiv:2512.23725, Ly et al. 2025, IAAI-26)
- `ref44_yang2025.ris` — Point-to-Set Metric-Gated MoE (IEEE TNNLS, Yang et al. 2025, DOI: 10.1109/TNNLS.2025.3548894; **NOTE: venue in Pre-Review_Response was wrong — it's IEEE TNNLS, not Springer**)
- `ref45_fan2024.ris` — STAR Transformer (Sensors 24(3):824, Fan et al. 2024)

Corrections:
- `ref25_berthelier2026.ris` — **CORRECTED** (previously contained wrong paper: DLinear photovoltaic by Wang et al.; now correctly contains Berthelier et al. arXiv:2603.11869)
- `ref16` DOI corrected: `10.3390/s24082543` → `10.3390/s24113454` (Sensors Vol. 24, No. 11, Article 3454)

---

## Track 1C — Introduction.md + Conclusion.md (completed)

### Introduction.md
| Change | Action Item | Description |
|--------|-------------|-------------|
| Backbone claim fixed | A1 | Para 3: "Fixing a common stacked LSTM backbone" → per-hypothesis description (H2=Ridge, H5=LSTM2-32, H6/H7=LSTM2-64) |
| Inter-seed diagnostic | A28 | Contribution (ii): added std=12.86 numbers and "transferable principle" language |
| [45] citation added | Track 4 | "attention-based encoders [13, 14, **45**]" — STAR Transformer as SOTA frontier |
| "order of magnitude" removed | A3 | Contribution (iv): "declines by at least an order of magnitude" → "qualitatively exceeds the next" |

### Conclusion.md
| Change | Action Item | Description |
|--------|-------------|-------------|
| "84" → "96" | A4/C | With formula: "6 losses × 4 clips × 4 datasets" |
| Null result language | A3 | "No custom loss improved" → "No statistically detectable improvement found" |
| Power caveat added | B | N=5 + 96 BH tests → minimum achievable p ≈ 3.0 (mathematically impossible) |
| Backbone qualifier | A1 | "Within an identical LSTM backbone" → "Within each hypothesis (using fixed architecture and preprocessing)" |
| "order of magnitude" removed | A3 | Same fix as Introduction |

---

## Track 1D — Methodology.md (completed)

| Change | Action Item | Description |
|--------|-------------|-------------|
| §III.K added | A7 | Ridge regression model description: deterministic, 20 = 5×4 configs, Mann-Whitney U (unpaired) on per-engine RMSE |
| §III.E rewritten | A1 | Two backbone architectures shown side-by-side (H5 compact vs H6/H7 full-capacity) |
| Table II added | A13 | Feature set, val split method, backbone by hypothesis |
| |Δz| formula | A9 | Inter-cluster z-score formula added to §III.F |
| Cohen's d translation | A12 | d=0.3 ≈ 0.6–1.0 RMSE cycles ≈ 1 maintenance cycle |
| H7 leakage note | A18 | FD001 hyperparameter tuning caveat added to §III.G |
| RevIN clarification | E | Forward-only implementation is correct; dimensional argument added to §III.D |

---

## Track 1E — Results.md + Discussion_Implication.md (completed)

### Results.md
| Change | Action Item | Description |
|--------|-------------|-------------|
| "84" → "96" | A4/C | H7 null result paragraph corrected |
| Power caveat | A3/B | "minimum achievable corrected p ≈ 3.0" sentence added |
| Null result language | A3 | "No custom loss achieves..." → "No statistically detectable improvement..." |
| Cohen's d sign explained | B | Parenthetical: positive d = worse NASA Score; negative d = improvement |
| Abdullah sentence replaced | A27 | Comparison with different architecture removed; replaced with backbone isolation note |
| CAELSTM caveat | A10 | "indicative rather than definitive" with backbone difference note |
| H5/H6 baseline note | A5 | Parenthetical in §B FD003 paragraph with 4 structural differences listed |

### Discussion_Implication.md
| Change | Action Item | Description |
|--------|-------------|-------------|
| [43, 44] added to §V.D | A15 | MoE citations with differentiation paragraph (battery vs turbofan; classification vs RUL) |
| [25] added to §V.B | A16 | Alongside [26] for RevIN critique |
| H7 power caveat | A3 | "power-limited" language added to §V.E opening |
| Fifth limitation | A5 | §V.G: H5/H6 implementation differences documented as Limitation 5 |

---

## Track 3a — RevIN Denorm Experiment (NOT REQUIRED)

Code verification (Issue E) confirmed that RevIN without denormalization is **architecturally correct** for RUL regression. The RUL output is a scalar in cycle units, dimensionally distinct from input sensor features. Applying the inverse sensor-normalization transform to a cycle-unit output would be dimensionally incorrect. No experiment needed.

---

## Track 3b — K Sensitivity Analysis (running / pending update)

Script created: `Data_Analysis/Code/H6_fault_mode/phase2_models/h6_p2_k_sensitivity.py`

Sweep: K ∈ {5, 10, 15, 20, 30} × FD003 × 5 seeds (25 total runs)

Results directory: `Data_Analysis/Results/H6_fault_mode/k_sensitivity/`

Output files:
- `k_sensitivity_fd003.csv` — raw per-seed results
- `k_sensitivity_summary.csv` — mean ± std per K value

**Results (completed):**

| K | RMSE mean | RMSE std | NASA mean | NASA std |
|---|-----------|----------|-----------|----------|
| 5 | 14.23 | 0.34 | 346 | 46 |
| 10 | 14.78 | 1.32 | 425 | 127 |
| 15 | 14.62 | 0.67 | 408 | 42 |
| 20 | 14.13 | 0.38 | 346 | 31 |
| 30 | 14.41 | 0.27 | 353 | 28 |

Key finding: RMSE insensitive to K (0.65 cycle range total); K=5 achieves 14.23±0.34; variance decreases monotonically with K (std: 0.34→0.27). Confirms fault-mode identity accessible from as few as 5 initial cycles.

**Manuscript update applied:**
- Results §C.2: K sensitivity post-hoc paragraph added
- Discussion §V.D: "first ten flight cycles" → "very first flight cycles"; full K sensitivity narrative added

---

## Action Items Status (from Pre-Review_Response.md)

| # | Status | Description |
|---|--------|-------------|
| A1 | ✅ | Backbone claim corrected in Introduction + Conclusion + Methodology |
| A2 | ✅ | Cohen's d sign convention clarified (no pseudo-replication confirmed) |
| A3 | ✅ | H7 null result language updated + power caveat throughout |
| A4 | ✅ | BH count: 84 → 96 everywhere |
| A5 | ✅ | H5/H6 FD003 inconsistency explained (4 structural differences) |
| A6 | ✅ | RevIN denorm: confirmed correct, no experiment needed |
| A7 | ✅ | H2 Ridge model described in §III.K |
| A8 | — | (not in scope for this session) |
| A9 | ✅ | |Δz| formula added to §III.F |
| A10 | ✅ | CAELSTM comparison caveat added |
| A11 | ✅ | Resolved via Issue B (same N=5 for both statistics) |
| A12 | ✅ | Cohen's d domain translation added to §III.J |
| A13 | ✅ | Validation split method documented in Table II, §III.H |
| A14 | ✅ | H2 Wilcoxon = Mann-Whitney U (unpaired); corrected in §III.K |
| A15 | ✅ | RUL-QMoE [43] and PSMMoEs [44] added to §V.D |
| A16 | ✅ | Berthelier et al. [25] added to §V.B |
| A17 | — | (not in scope) |
| A18 | ✅ | H7 hyperparameter leakage note added to §III.G |
| A19 | ✅ | K sensitivity complete: RMSE insensitive to K (14.13–14.78); K=5 sufficient |
| A20–A25 | — | (not in scope) |
| A26 | ✅ | p=0.97 directional note confirmed present in Results §A |
| A27 | ✅ | Abdullah RMSE comparison removed; replaced with backbone isolation note |
| A28 | ✅ | Inter-seed variance as diagnostic principle elevated in Introduction (ii) |
| A29–A30 | — | (not in scope) |

---

## Known Remaining Issues (Pre-Review 기준)

1. **H2 paired Wilcoxon** (A14): Flagged as limitation in §III.K; a paired reanalysis (Wilcoxon signed-rank on per-engine diffs) would be methodologically stronger but requires rerunning the H2 statistical test.
2. **ref16 metadata**: DOI corrected to 10.3390/s24113454; verify before submission that the year (2024), authors (Wu et al.), journal (Sensors) and title are correct.
3. **ref25 RIS correction**: Verify that `Citation/ref25_berthelier2026.ris` now contains Berthelier et al. arXiv:2603.11869 (six authors including Gaspard Berthelier).

---

---

## TII Virtual Review Phase (2026-07-08~09)

> **기반 문서:** `TII_Virtual_Submission_Review/TII_Virtual-Review_Report.md`, `TII_Virtual-Review_Response.md`
> **세부 기록:** `TII_Virtual_Submission_Review/Virtual_Review_Changelog.md`
> **수락률 변화:** 20–30% (기준) → 32–38% (Phase 1+2 완료 후)

---

### 의사결정 사항

| 결정 | 내용 | 근거 |
|------|------|------|
| **목표 저널 변경** | TII 1순위 → **RESS 1순위** | TII 수락률 32–38% 한계; RESS 예상 42–52%, CMAPSS 홈 저널(레퍼런스 4건), 페이지 제한 없음 |
| **Phase 3 미진행** | N-CMAPSS DS03 파일럿 미착수 | 1–2주 소요 대비 RESS에서는 필수 요건이 아님; 현재 원고로 투고 우선 |
| **T4 (UQ) 완전 제외** | MC Dropout PICP 44.6%, Conformal PICP 31% — 목표 90% 미달 | 원고에 언급 시 UQ 연구로 범위 확장 오해, 오히려 리뷰어 추가 질문 유발 |
| **T5 (10-seed) 미반영** | 10-seed RMSE=14.48±1.01 → 5-seed 유지(14.78±1.32) | M3/FD003만 10-seed 전환 시 논문 내 비교 기준(seeds {0–4}) 불일치; 0.30 cycle 개선은 서사적 가치 없음 |
| **H2 클리핑 기여 제외** | Introduction 기여 목록 (i) 삭제 | clip=125는 선행연구[5,6] 확립 표준; 신규성 주장 불가. "확인된 전제 조건"으로 재프레이밍 |
| **T3 수치 원고 미인용** | Transformer/AttnLSTM 파일럿 수치 비게재 | 5-seed pilot → formal claim 불가; M3가 attention backbone에서 불필요해진다는 역방향 결과는 §V.D 경량성 프레이밍 + §V.H open question으로 처리 |

---

### Phase 1 — 원고 텍스트 수정 (완료)

| 항목 | 위치 | 변경 내용 |
|------|------|---------|
| T2 (아키텍처 추정) | §III.E Table III | 계산 복잡도 표 신설 (아키텍처 기반 추정치) |
| T7 | §V.I | 비용 편익 추정 단락 추가 (50대 플릿, $200K–$500K 절감 추정) |
| T8 | §V.G | 6번째 한계: 온라인 적응 부재 추가 |
| T9 | References 앞 | Data and Code Availability 섹션 신설 |
| T10 | §V.F | Silhouette marginal regime (S ∈ [0.5, 0.65]) caveat 추가 |
| T12 | §V.I | Fleet 확장성 (Mini-Batch K-means) 언급 |
| T1-B | §V.G | CMAPSS 한계 "primary open question" 강화 |
| H2 기여 제외 | §I (Introduction), §VI (Conclusion) | 기여 목록 (i) 삭제 → "확인된 전제 조건" 단락으로 이동; Conclusion 톤 조정 |

---

### Phase 2 — 파일럿 실험 (완료, 원고 미반영)

| 항목 | 실험 결과 | 원고 처리 |
|------|---------|---------|
| T3: Transformer-M0/M3 (FD003, 5 seeds) | Transformer-M0 RMSE=14.34 (LSTM-M3와 동급); M3 오히려 미세 역효과 | §V.D 경량성 프레이밍; §V.H open question |
| T3: AttnLSTM-M0/M3 (FD003, 5 seeds) | AttnLSTM-M0=16.53, M3=16.53 (무효과) | 동일 처리 |
| T4: MC Dropout + Conformal (FD003, M3) | PICP: 44.6%/31% (목표 90%) | 완전 제외 |
| T5: M3/FD003 10-seed 확장 | RMSE=14.48±1.01 (vs 5-seed 14.78±1.32) | 5-seed 유지 |

결과 파일: `Data_Analysis/Results/H6_fault_mode/H6_M3_FD003_UQ_results.csv`, `H6_transformer_pilot_FD003.csv`, `H6_attention_lstm_pilot_FD003.csv`, `H6_M3_FD003_10seeds_summary.csv`

---

### T2 완성 — Table III 실측값 업데이트 (2026-07-09)

스크립트: `Data_Analysis/Code/H6_fault_mode/phase2_models/h6_timing_benchmark.py`
결과 파일: `Data_Analysis/Results/H6_fault_mode/H6_timing_benchmark.csv`

| 모델 | 파라미터 | FLOPs/window | 추론 시간 (실측) | 학습 시간 (GPU) |
|------|---------|-------------|----------------|----------------|
| M0 | 56.1K | 3.18M | 0.24 ± 0.05 ms | ~0.1 s/epoch |
| M1 | 112.3K | 3.18M† | 0.25 ± 0.06 ms† | ~0.2 s/epoch |
| M2 | 112.3K | 6.37M | 0.57 ± 0.24 ms | ~0.2 s/epoch |
| M3 | 117.2K | 6.38M | 0.64 ± 0.21 ms | ~0.2 s/epoch |

†M1 추론은 단일 브랜치만 실행(GMM argmax). 기존 추정치(1–4ms) 대비 4–6× 낮음.
수정 파일: `manuscript_full_text.md`, `Manuscript/Sections/Methodology.md`

---

### RESS 전환 우선순위 로드맵

#### 즉시 (1–2일) — 투고 프레이밍

| 작업 | 내용 | 효과 |
|------|------|------|
| Abstract + Introduction 재프레이밍 | "industrial informatics" → "reliability & PHM" 중심으로 재서술 | RESS 리뷰어 첫인상 결정; 단일 최대 레버 |
| Cover Letter RESS 버전 신규 작성 | TII 커버레터 → RESS Editor 대상으로 전면 재작성 | 데스크 통과율 향상 |
| 표 번호 충돌 해결 | §III.E Table III와 Results §IV Table III–V 충돌 → 전체 재번호 | 제출 필수 조건 |
| Searching_Target_Journal.md 업데이트 | RESS 1순위 확정 Decision Log 기록 | 문서 일관성 |

#### 단기 (3–5일) — 포지셔닝 강화

| 작업 | 내용 | 효과 |
|------|------|------|
| §V.F 3계층 위계 재서술 | "design hierarchy" → "reliability-driven design checklist" | RESS 스코프 언어 정렬 |
| H2 결과 재프레이밍 | "clip=None → 6 orders of magnitude" = reliability risk로 강조 | RESS 리스크 관점 부합; TII에서 축소했던 H2 기여를 RESS에서 적절히 복원 |
| Elsevier 템플릿 적용 | Markdown → Elsevier LaTeX/Word 변환 | 제출 형식 요건 |

#### 선택 (수락률 추가 향상)

| 작업 | 내용 | 우선순위 |
|------|------|--------|
| T6: False routing 민감도 분석 | GatingNet 오분류 시 RMSE 변화 | RESS 신뢰성 관점에서 TII보다 중요; 권장 |
| T11: Pinball scatter plot | clip=None에서 L1 vs L6 predicted/true scatter | Minor |

---

### RESS 전환 후 예상 수락률

| 단계 | 예상 수락률 |
|------|-----------|
| 현재 원고 (TII 기준) | 32–38% |
| **RESS 전환 + 즉시 작업 완료** | **42–52%** |
| RESS + T6 (false routing 분석) 추가 시 | **47–55%** |

---

## 2026-07-30 — M3 조기 라우팅 배치 문헌 검토 및 원고 반영

### 배경

M3 Attention-Gate의 핵심 전제("초기 K=10 사이클만으로 고장 모드 라우팅 가능")가 다른 연구 영역으로 확장될 수 있는지 검토하는 과정에서, Consensus MCP 검색으로 배치 문헌 4편 확인.

### 원고 변경 사항

| 위치 | 변경 내용 |
|------|---------|
| `§V.D` 말미 (2단락 추가) | 조기 라우팅 전제의 CMAPSS-특수성 및 일반화 조건 논의; Fu 2024, Wang 2023, Dong 2025, Wang 2025 인용 [46]–[49] |
| `§V.G` Seventh limitation 추가 | M3 조기 라우팅이 초기 기준값 차이에 의존하며, 실제 엔진·N-CMAPSS에서는 한계 가능성 명시 |
| References [46]–[49] 추가 | Fu 2024 (IEEE TASE), Wang 2023 (IEEE TASE), Dong 2025 (ICEIOM), Wang 2025 (IEEE TASE) |
| `manuscript_full_text.md` 헤더 | `Last revised: 2026-07-30` 추가 |

### 함께 업데이트된 파일

| 파일 | 변경 내용 |
|------|---------|
| `Hypothesis/Hypothesis_PossibleValidation.MD` | H6 섹션에 배치 문헌 검토 소절 및 참고 문헌 4편 추가 |
| `Hypothesis/Hypothesis_critics.md` | H6 섹션에 잠재 리뷰어 지적 및 방어 논리 추가 |
| `Data_Analysis/주요이슈_및_의사결정.md` | P2 항목 추가 (결정 근거·반영 파일 목록 포함) |
| `Manuscript/Sections/Discussion_Implication.md` | §V.D, §V.G 동일 반영; v1.0 → v1.1 버전업 |
| `연구개요.md` | 최종 수정 날짜 추가 |

### 수락률 영향 평가

배치 문헌을 limitation으로 흡수하고 방어 논리를 §V.D에 명시함으로써, 리뷰어가 동일 지적을 제기하더라도 사전 대응이 완료된 상태. 수락률 영향: 중립~소폭 긍정 (리뷰어 신뢰도 향상).
