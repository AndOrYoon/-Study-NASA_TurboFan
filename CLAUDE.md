# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Environment

**Python:** 3.13 (TensorFlow not supported; use PyTorch)
**Virtual environment:** `C:\BMAD_PY313\NASA_TurboFan\`

```powershell
# Activate
NASA_TurboFan\Scripts\activate

# Run any experiment script
python Data_Analysis\Code\H7_loss_function\run_all_h7.py

# Run a single hypothesis stage
python Data_Analysis\Code\H5_normalization\04_run_experiments.py

# EDA re-run
python eda_turbofan.py
```

---

## Project Structure

```
C:\BMAD_PY313\
├── Dataset/                     ← Raw CMAPSS .txt files + EDA output
│   ├── train_FD00{1-4}.txt
│   ├── test_FD00{1-4}.txt
│   ├── RUL_FD00{1-4}.txt       ← Ground-truth RUL for test engines
│   └── Figure/                  ← EDA plots (fig01–fig12)
├── Hypothesis/                  ← Literature review, research-gap analysis
│   ├── Hypothesis_PossibleValidation.MD
│   └── Hypothesis_critics.md   ← Reviewer critique checklist
├── Data_Analysis/
│   ├── Analysis_Plan.md        ← Canonical experiment design for H2/H5/H6/H7
│   ├── 주요이슈_및_의사결정.md   ← All resolved design decisions (read before changing anything)
│   ├── Ad-hoc_Analysis/        ← Post-hoc analysis reports and revision planning
│   ├── Code/
│   │   ├── shared/
│   │   │   └── op_condition_utils.py  ← K-means residualization (used by H5 & H6)
│   │   ├── H2_clipping/        ← 01–05 scripts (data → run → eval → stats → viz)
│   │   ├── H5_normalization/   ← 01–06 scripts
│   │   ├── H6_fault_mode/
│   │   │   ├── phase1_clustering/
│   │   │   ├── phase2_models/  ← h6_p2_model_utils.py (shared backbone + utils)
│   │   │   └── phase3_evaluation/
│   │   ├── H7_loss_function/   ← 00–08 scripts + run_all_h7.py
│   │   └── Ad-hoc_Analysis/    ← Unified protocol & corrected H6 scripts
│   │       ├── 01_unified_data_loader.py
│   │       ├── 02_run_unified.py
│   │       ├── 03_analyze_results.py
│   │       ├── 04_run_h6_corrected.py  ← C-Full corrected H6 (M0/M1/M2/M3, 40 runs)
│   │       └── 05_run_fd4_unified_op.py ← FD004 op-condition consistency check
│   └── Results/
│       ├── H{2,5,6,7}_*/       ← Original experiment CSVs, figures, checkpoints
│       ├── Ad-hoc_Analysis/    ← unified_results.csv, statistical_tests.csv, figures/
│       └── H6_corrected/       ← h6_corrected_results.csv (corrected protocol, 40 runs)
└── Manuscript/
    ├── Full-Text_Manuscript/
    │   ├── manuscript_full_text.md       ← Primary compiled manuscript (single source of truth)
    │   └── Revision/
    │       ├── manuscript_full_text_reviewed(FFFN).md  ← Revision draft with corrected results
    │       └── manuscript_full_text_FFN_reviewed_comments.md  ← Inline review comments (2026-08-31)
    ├── Sections/                ← Source section files (sync with manuscript_full_text.md)
    │   ├── Methodology.md
    │   └── Discussion_Implication.md
    ├── Figures/                 ← Fig1–Fig8 (final manuscript figures)
    ├── Tables/                  ← Table1–Table5 CSV
    ├── Tables_Figures.md        ← English figure/table captions
    ├── Submission/
    │   └── RESS/               ← RESS (Elsevier) LaTeX submission package
    │       ├── main.tex        ← Original submission LaTeX
    │       └── Revision/
    │           └── main_revision.tex  ← Revised LaTeX with Option A+ applied
    └── Pre-Review/
        ├── Revision_Changelog.md
        └── TII_Virtual_Submission_Review/
            ├── TII_Virtual-Review_Report.md   ← Virtual reviewer critique
            ├── TII_Virtual-Review_Response.md ← Response plan (Phase 1–3)
            └── Virtual_Review_Changelog.md    ← All Phase 1/2 decisions & rationale
```

---

## Data & Labels

- **Datasets:** FD001–FD004; 21 raw sensors, space-separated `.txt` files, no header
- **Column order:** `unit cycle op1 op2 op3 s1…s21`
- **Constant sensors to drop** (zero-variance, per dataset):
  - FD001: `s1 s5 s6 s10 s16 s18 s19` → 14 features remaining
  - FD003: `s1 s5 s10 s16 s18 s19` → 15 features remaining (s6 is NOT constant in FD003)
  - FD002: `s16` → 20 features; FD004: `s16` → 20 features
- **RUL labeling:** piecewise-linear, `clip=125` is the validated standard. `clip=None` causes catastrophic NASA scores on FD003.
- **Train/val split:** engine-level (hold out 20% of complete engines, not cycles). Never split by cycle — it causes RUL distribution mismatch.
- **Test sequences:** last-window (window=30) per engine, zero-pad from the front if `len < 30`.

---

## Operating Condition Residualization (FD002 / FD004)

FD002 and FD004 have 6 operating conditions that shift sensor absolute values by tens to hundreds of units. **All experiments on FD002/FD004 must apply operating-condition residualization before any normalization or model training.**

Canonical implementation: `Data_Analysis/Code/shared/op_condition_utils.py`

```python
# Train time — fit on train data only
scaler, km = fit_op_condition_kmeans(train_df)          # K-means k=6 on op1/op2/op3
cluster_means = compute_cluster_means(train_df, scaler, km, sensor_cols)

# Both train and test
train_df = apply_op_residual(train_df, scaler, km, cluster_means, sensor_cols)
test_df  = apply_op_residual(test_df,  scaler, km, cluster_means, sensor_cols)
```

**H6 variant** (`h6_p2_model_utils.py`): `fit_op_residual_fd004(train_df, sensor_cols)` — fits fresh K-means each run (does not load Phase 1 artifacts). Use `apply_op_residual_fd004()` for transform. Never call the old `load_op_artifacts_fd004()` in Phase 2 model scripts.

---

## Shared LSTM Backbone Architecture

All hypotheses use the same stacked LSTM:
- **LSTM1(64)** → full sequence output (all timesteps) → Dropout(0.2)
- **LSTM2(64)** → last timestep → Dropout(0.2) → FC(64→32→ReLU→1)
- **Window:** 30 cycles, **Batch:** 256, **LR:** 1e-3 (Adam), **WD:** 1e-4, **Patience:** 15 epochs

LSTM1 must pass the **full sequence** (all timesteps) to LSTM2, not just the final hidden state. This is the stacked-LSTM bug that caused incorrect results — always use `out, _ = self.lstm1(x)` and pass `out` to `lstm2`.

### H6 Model Variants (FD003 / FD004 only)

| Model | Architecture | Notes |
|-------|-------------|-------|
| M0 | Single LSTM (baseline) | Standard backbone above |
| M1 | Hard Routing (GMM argmax) | Fails FD004 at test time — cluster distribution collapses to [1:247] |
| M2 | Soft Gating (GMM probabilities) | Two branches weighted by GMM soft probs; auxiliary loss ×0.1 per branch |
| M3 | Attention Gate (end-to-end) | `GatingNet` reads first K=10 cycles → softmax gate → two branches; immune to test-time cluster collapse |

M3 training loss: `MSE(final) + 0.05*MSE(branch0) + 0.05*MSE(branch1)`
M2 training loss: `MSE(final) + 0.1*MSE(branch0) + 0.1*MSE(branch1)`

---

## Metrics

```python
# RMSE — primary performance metric
rmse = np.sqrt(np.mean((pred - true) ** 2))

# NASA Score — official competition metric, lower is better
# Penalises late predictions (d>0) more than early (d<0)
d = pred - true
s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
nasa_score = float(np.mean(s))   # per-engine mean used for cross-dataset comparison
```

Report as `mean ± std` over 5 seeds: `[0, 1, 2, 3, 4]`.

> ⚠️ **NASA Score 재계산 주의:** `nasa_score()` in `h6_p2_model_utils.py`는 `pen.sum()`을 사용한다. CSV에 저장된 per-engine mean을 재조합할 때 `np.mean(pen)`이 아닌 `pen.sum()`을 써야 한다. FD003은 100개 테스트 엔진이므로 잘못 쓰면 결과가 100× 작게 나온다.

---

## Statistical Testing

All hypothesis tests use:
1. **Wilcoxon rank-sum** (one-sided, α=0.05) — treatment vs. baseline
2. **Benjamini-Hochberg FDR** correction when comparing multiple methods simultaneously
3. **Cohen's d** (pooled SD) for effect size
4. Significance threshold: `p_BH < 0.05` AND `|d| ≥ 0.3`

---

## H7 Loss Functions

Defined in `Data_Analysis/Code/H7_loss_function/02_loss_functions.py`. All share the signature `loss_fn(pred, true, life_ratio=None, clip_value=None)`.

| ID | Name | Key parameter |
|----|------|--------------|
| L1 | MSE (baseline) | — |
| L2 | NASA Score Loss | differentiable approx |
| L3 | DynMSE | `lambda_dyn=1.0` |
| L4 | Focal-RUL | `gamma=2.0` |
| L5 | TWA (Time-Weighted Asymmetric) | `lambda_t=10.0, lambda_a=1.0` (Phase 3a best) |
| L6 | Pinball | `tau=0.35` default; `tau=0.25` best FD001/FD003 |
| L7 | HubA (Huber-Asymmetric) | `delta=20, lambda_a=3.0` (Phase 3d best) |

`life_ratio` is computed from **training data only** (current cycle / max train cycle). It is `None` at test time — all loss functions must gracefully fall back to unweighted MSE when `life_ratio is None`.

**Key finding:** RUL clipping dominates loss function choice. No custom loss achieves statistically significant improvement over L1 MSE after BH-FDR correction.

---

## Experiment Results Summary (completed)

| Hypothesis | Verdict | Key result |
|-----------|---------|-----------|
| H2 (clipping) | Partially accepted | clip=125 minimises RMSE; clip=130 minimises NASA on FD002/FD004 |
| H5 (normalization) | Rejected | Fleet MinMax (N1) is best; per-unit and RevIN are significantly worse |
| H6 (fault mode) | Partially revised — see note | M1 hard routing significantly worse (p_BH=0.0045); M2/M3 statistically equivalent to M0. Original 65.8% claim was from a collapsed M0 baseline. |
| H7 (loss functions) | Rejected | No loss beats MSE after BH-FDR; clip dominates loss choice |

Results CSVs: `Data_Analysis/Results/H{2,5,6,7}_*/`
Corrected H6 results: `Data_Analysis/Results/H6_corrected/h6_corrected_results.csv`
Unified N1 vs N3 results: `Data_Analysis/Results/Ad-hoc_Analysis/`
Manuscript-ready figures/tables: `Manuscript/Figures/`, `Manuscript/Tables/`

### H6 Corrected Results (C-Full protocol, 5 seeds, 40 runs — 2026-08-27)

| Model | FD003 RMSE±std | FD004 RMSE±std | vs M0 (FD003) | Significant? |
|-------|---------------|---------------|--------------|-------------|
| M0 (baseline) | 12.97 ± 0.67 | 18.96 ± 3.97 | — | — |
| M1 (hard routing) | 33.25 ± 8.74 | 33.28 ± 2.05 | +20.28 worse | ✅ p_BH=0.0045 |
| M2 (soft gating) | 12.28 ± 0.57 | 18.82 ± 1.56 | −0.69 | ✗ p_BH=0.352 |
| M3 (attention gate) | 13.24 ± 1.69 | 17.30 ± 1.04 | +0.27 worse | ✗ p_BH=0.754 |

> ⚠️ **Why original 65.8% was wrong:** H6 M0 FD003 (original RMSE=43.23) showed mean-prediction collapse — all 5 seeds predicted a constant ~87 for all 100 test engines (std<0.0002). Root causes: (1) fixed val split seed=42, (2) no MIN_EPOCHS warmup, (3) single MSE loss with weak gradient near trivial solution. M3 survived because auxiliary branch losses provided extra gradient paths. After correcting the protocol, M0 converges normally (12.97±0.67) and M3 shows no statistically significant advantage.

---

## Key Design Decisions (already resolved — do not re-open)

See `Data_Analysis/주요이슈_및_의사결정.md` for full rationale. Summary:

- **H2 clip candidates:** {75, 100, 125, 130, None} — **not 150** (no academic precedent)
- **H5 FD002/FD004:** K-means residualization applied *before* any normalizer (shared util)
- **H5 RevIN (N7):** implemented as `LSTMWithRevIN` model subclass, not preprocessing
- **H5 boundary group:** engines with lifetime < 150 cycles (not < 50, which is always empty)
- **H6 M3 gating:** uses first K=10 cycles only — avoids train/test late-cycle mismatch
- **H2 runs:** 20 (deterministic Ridge; seeds are meaningless for `LinearRegression`)
- **H2 = prerequisite, not contribution:** clip=125 is confirmed established standard [5,6]. Do not reframe H2 as a novel contribution in the manuscript — it belongs in setup/methods context only.
- **M3 is LSTM-specific:** Pilot experiments (T3) showed that adding any attention mechanism (Transformer or self-attention LSTM) to the backbone makes M3 routing redundant — attention already captures early-cycle fault patterns implicitly. M3's value is its lightweight overhead (4.9K params, <5%) on top of a standard sequential LSTM backbone.
- **H6 M3 framing (post ad-hoc):** Do NOT claim "65.8% RMSE reduction." Corrected framing: M3's auxiliary branch loss structure provides **training robustness** — it avoids mean-prediction collapse under adverse validation splits. FD004 std improvement (3.97→1.04) is the concrete evidence. Fault-mode routing performance claim (vs properly trained M0) is not statistically supported.
- **H6 Three-tier Tier 2 reframing:** Tier 2 (architecture) is a **design boundary**, not a performance lever. M1 hard routing is statistically significantly worse (p_BH=0.0045); M2/M3 recover to baseline. The value is avoiding M1-type collapse, not exceeding baseline.
- **FD004 op-condition utils equivalence:** `op_condition_utils.py` and `fit_op_residual_fd004()` are mathematically identical (bit-for-bit same output). Residual differences between unified-experiment and corrected-H6 FD004 results come from pipeline-level differences (column naming → RNG state), not from the op-condition step itself.

---

## Manuscript Status (as of 2026-09-10)

**Target journal:** RESS (Reliability Engineering & System Safety, Elsevier). **Expected acceptance rate:** 45–60%.  
> TII was deprioritized (2026-07-10): CMAPSS synthetic-data limitation is borderline for TII's "outstanding and original" bar. Do not reference TII workflows (ScholarOne, IEEE format) going forward.

**🔴 RESS 투고 취소 (2026-09-23) — Scope 불일치. 다른 저널 재투고 준비 중.**

**Submitted via:** Elsevier Editorial Manager  
**Submission package:** `Manuscript/Submission/RESS/` (LaTeX, elsarticle.cls)  
**Primary manuscript (original):** `Manuscript/Full-Text_Manuscript/manuscript_full_text.md`  
**Funding:** "Development of Physical Data Quality Management Technologies for Physical AI", No. RS-2026-25621690, Korea government (MSIT).

### Next Steps

RESS scope 불일치 확인 → 대안 저널 탐색 및 재투고 준비.

### Core Contributions (revised framing — post ad-hoc analysis)

1. First controlled normalisation ablation — Fleet MinMax (N1) significantly outperforms per-unit and RevIN on FD001–FD003 (p_BH=0.009); FD004 N1 advantage disappears when op columns are removed (protocol interaction).
2. M3 Attention Gate — **training robustness** via auxiliary branch loss: avoids mean-prediction collapse under adverse validation splits. FD004 std: 3.97→1.04. No statistically significant RMSE improvement over a properly trained M0 baseline.
3. Architecture as design boundary — M1 hard routing is statistically significantly worse than baseline (p_BH=0.0045, both datasets); M2/M3 recover to M0 level. Architecture choice protects against collapse, not lifts performance.
4. Practical deployment framework — ordered design decisions: label engineering > normalisation ≈ architecture > loss function.

### Phase 2 Pilot Results (excluded from manuscript)

| Task | Result | Decision |
|------|--------|---------|
| T3: Transformer + AttnLSTM backbone | M3 routing redundant when backbone has attention | §V.D efficiency framing + §V.H open question only |
| T4: MC Dropout + Conformal UQ | PICP 44.6%/31%, far below 90% target | Completely excluded |
| T5: 10-seed M3/FD003 expansion | RMSE=14.48±1.01 (vs 14.78±1.32) | 5-seed result maintained for table consistency |

Phase 3 (N-CMAPSS pilot): not proceeding.

---

## Windows Encoding

Scripts that print Unicode characters (em dash `—`, etc.) may fail on Korean Windows locale (cp949). Add at the top of any script:

```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
```
