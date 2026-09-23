# Turbofan Engine RUL Prediction: A Cross-Dataset Ablation Study

**Paper:** From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction  
**Target journal:** Engineering Applications of Artificial Intelligence (EAAI, Elsevier/IFAC, IF ~7.8, Q1)  
**Submission branch:** [`submission/eaai`](../../tree/submission/eaai)  
**Status:** EAAI submission preparation complete (September 2026)

---

## Overview

Controlled ablation study isolating four interdependent design decisions in turbofan engine Remaining Useful Life (RUL) prediction, evaluated on the NASA C-MAPSS benchmark (FD001–FD004) with a shared stacked-LSTM backbone.

| Hypothesis | Factor | Levels | Verdict |
|-----------|--------|--------|---------|
| H2 | RUL label clipping threshold | 5 (75, 100, 125, 130, None) | Partially accepted |
| H5 | Sensor normalization strategy | 7 (N1–N7) | Rejected (N1 wins) |
| H6 | Fault-mode routing architecture | 4 (M0–M3) | Partially revised |
| H7 | Training loss function | 7 (L1–L7) | Rejected (MSE wins) |

All comparisons: Wilcoxon rank-sum + Benjamini-Hochberg FDR correction (α = 0.05).  
**Total training runs: 1,000+** across all hypotheses and datasets.

---

## Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)

| Sub-dataset | Train engines | Test engines | Op. conditions | Fault modes |
|-------------|--------------|-------------|---------------|-------------|
| FD001 | 100 | 100 | 1 | HPC degradation |
| FD002 | 260 | 259 | 6 | HPC degradation |
| FD003 | 100 | 100 | 1 | HPC + Fan degradation |
| FD004 | 249 | 248 | 6 | HPC + Fan degradation |

Column order: `unit cycle op1 op2 op3 s1…s21`  
RUL label: piecewise-linear, clip = 125 cycles.

---

## Shared Backbone

Stacked LSTM used identically across all hypotheses:

```
Input(30 × F) → LSTM1(64, return_sequences=True) → Dropout(0.2)
             → LSTM2(64, return_sequences=False) → Dropout(0.2)
             → FC(64) → ReLU → FC(32) → ReLU → FC(1)
```

Window: 30 cycles | Batch: 256 | LR: 1e-3 (Adam) | WD: 1e-4 | Early-stop patience: 15  
Seeds: [0, 1, 2, 3, 4] — report mean ± std over 5 runs.

---

## Key Results

### H2 — RUL Clipping

clip = 125 minimises RMSE across all datasets. clip = None causes catastrophic NASA Score inflation on FD003 (306,000× vs. clip = 125).

| Clip | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|------|-----------|-----------|-----------|-----------|
| 75 | 33.65 | 47.37 | 32.48 | 50.83 |
| 100 | 24.51 | 37.67 | 23.00 | 40.37 |
| **125** | **21.90** | **32.39** | **21.62** | **34.61** |
| 130 | 22.06 | 31.83 | 22.24 | 34.04 |
| None | 31.90 | 33.05 | 56.09 | 46.99 |

---

### H5 — Sensor Normalization

Fleet min-max (N1) significantly outperforms all per-unit and instance-level alternatives on FD001, FD002, and FD003 (p_BH = 0.009). No statistically detectable difference on FD004 under the unified protocol (p_BH = 0.46 — protocol-sensitive multi-condition dataset).

| Strategy | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|----------|-----------|-----------|-----------|-----------|
| **N1 Fleet MinMax** | **14.06** | **20.62** | **14.57** | **18.96** |
| N3 Per-unit MinMax | 16.37 | 24.41 | 15.24 | 19.09 |
| N7 RevIN | 14.92 | 28.18 | 15.01 | 21.36 |

RevIN is competitive on FD001 but degrades substantially on FD002/FD004 (multi-condition datasets where instance statistics encode operating-condition offsets, not degradation).

---

### H6 — Fault-Mode Architecture

> ⚠️ **Corrected results** — original protocol had M0 mean-prediction collapse on FD003 (all 5 seeds predicted constant ~87; RMSE = 43.23). After correcting the protocol (C-Full: randomised val split, MIN_EPOCHS warmup, 5 seeds × 8 conditions = 40 runs):

| Model | FD003 RMSE ± std | FD004 RMSE ± std | vs M0 (FD003) | p_BH |
|-------|-----------------|-----------------|--------------|------|
| M0 — Single LSTM (baseline) | 12.97 ± 0.67 | 18.96 ± 3.97 | — | — |
| M1 — Hard Routing (GMM argmax) | 33.25 ± 8.74 | 33.28 ± 2.05 | +20.28 worse | **0.0045** ✅ |
| M2 — Soft Gating (GMM probs) | 12.28 ± 0.57 | 18.82 ± 1.56 | −0.69 | 0.352 |
| M3 — Attention Gate (K=10 cycles) | 13.24 ± 1.69 | 17.30 ± 1.04 | +0.27 | 0.754 |

**Interpretation:**
- M1 hard routing is statistically significantly worse than M0 (p_BH = 0.0045, both datasets) due to test-time cluster collapse (247:1 assignment ratio on FD004).
- M2 and M3 recover to M0-level RMSE — no statistically detectable improvement.
- M3's auxiliary branch loss provides **training robustness**: it avoids mean-prediction collapse under adverse validation splits. FD004 inter-seed std drops from 3.97 (M0) to 1.04 (M3).
- Architecture choice is a **reliability safeguard**, not a performance lever.

---

### H7 — Training Loss Function

No custom loss achieves statistically significant improvement over MSE after BH-FDR correction across 96 pairwise comparisons (minimum p_BH = 0.176). RUL clipping is the dominant source of NASA Score variance, not the loss function.

| Loss | Key parameter | Verdict |
|------|-------------|---------|
| L1 MSE (baseline) | — | Reference |
| L2 NASA Score Loss | differentiable approx | p_BH > 0.05, not significant |
| L5 TWA | λ_t=10, λ_a=1 | p_BH > 0.05, not significant |
| L7 HubA | δ=20, λ_a=3 | p_BH > 0.05, not significant |

---

## Design Priority Ordering

```
UPSTREAM ──────────────────────────────────── DOWNSTREAM

[1] Label Engineering     [2] Normalization     [3] Architecture     [4] Loss Function
    clip = 125                Fleet MinMax           Avoid M1              MSE default
    306,000× NASA Score       p_BH = 0.009           p_BH = 0.0045         No gain after
    if miscalibrated          on FD003               significantly          BH-FDR
                                                      worse
```

Upstream choices produce larger and more consistent effects than downstream choices. Resolve each tier before investing in the next.

---

## Repository Structure

```
C:\BMAD_PY313\
├── Dataset/                            ← Raw CMAPSS .txt files + EDA outputs
│   ├── train/test_FD00{1-4}.txt
│   ├── RUL_FD00{1-4}.txt
│   └── Figure/                         ← EDA plots (fig01–fig12)
├── Hypothesis/                         ← Literature review, research gap analysis
├── Data_Analysis/
│   ├── Analysis_Plan.md
│   ├── Code/
│   │   ├── shared/
│   │   │   └── op_condition_utils.py   ← K-means residualization (FD002/FD004)
│   │   ├── H2_clipping/                ← Scripts 01–05
│   │   ├── H5_normalization/           ← Scripts 01–06
│   │   ├── H6_fault_mode/              ← phase1 / phase2 / phase3
│   │   ├── H7_loss_function/           ← Scripts 00–08 + run_all_h7.py
│   │   └── Ad-hoc_Analysis/            ← Unified N1/N3 protocol + corrected H6
│   └── Results/
│       ├── H{2,5,6,7}_*/               ← Experiment CSVs, figures, checkpoints
│       ├── Ad-hoc_Analysis/            ← unified_results.csv, statistical_tests.csv
│       └── H6_corrected/               ← h6_corrected_results.csv (C-Full, 40 runs)
└── Manuscript/
    ├── Full-Text_Manuscript/
    │   └── manuscript_full_text.md     ← Primary compiled manuscript
    ├── Sections/                       ← Source section files
    ├── Figures/                        ← Final figures (Fig1–Fig8)
    ├── Tables/                         ← Tables 1–5 CSV
    └── Submission/
        ├── RESS/
        │   └── Revision/
        │       └── main_revision.tex   ← Final RESS revision (Option A+; withdrawn 2026-09-23)
        └── EAAI/                       ← Active submission workspace (this branch)
            ├── main_EAAI.tex           ← Full manuscript (preprint 10pt, ~48p)
            ├── main_EAAI_anon.tex      ← Double-blind anonymous version
            ├── titlepage_EAAI.tex      ← Author info (separate for blind review)
            ├── highlights.txt          ← 5 highlights (≤85 chars each)
            ├── Abstract_EAAI.md        ← v2.0 (AI/engineering distinction)
            ├── References_EAAI.md      ← v1.1 (54 refs, Section Usage Map)
            └── Preparation_for_Submitting_EAAI.md  ← 7-task checklist
```

---

## Submission History

| Date | Event |
|------|-------|
| 2026-09-09 | Submitted to RESS (Reliability Engineering & System Safety) |
| 2026-09-23 | RESS desk rejection — scope mismatch ("signal processing / fault diagnosis no longer in scope") |
| 2026-09-23 | Retargeted to **EAAI**; `submission/eaai` branch created |
| 2026-09-23 | EAAI preparation complete (Tasks 1–7); Zenodo DOI pending |

---

## EAAI Submission Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Abstract rewrite (AI/engineering distinction + acronym definitions) | ✅ |
| 2 | Keywords reduced to 6 | ✅ |
| 3 | Journal name updated in LaTeX | ✅ |
| 4 | Highlights (5 bullets, ≤85 chars each) | ✅ |
| 5 | Double-blind files separated | ✅ |
| 6 | Data Availability Statement inserted | ✅ (Zenodo DOI pending) |
| 7 | Generative AI declaration + CRediT contributions | ✅ |

---

## Environment

```powershell
# Activate virtual environment
NASA_TurboFan\Scripts\activate

# Run all H7 loss function experiments
python Data_Analysis\Code\H7_loss_function\run_all_h7.py

# Run corrected H6 (C-Full protocol, 40 runs)
python Data_Analysis\Code\Ad-hoc_Analysis\04_run_h6_corrected.py

# Run single hypothesis stage
python Data_Analysis\Code\H5_normalization\04_run_experiments.py
```

**Python:** 3.13 (TensorFlow not supported — PyTorch throughout)  
**Virtual environment:** `C:\BMAD_PY313\NASA_TurboFan\`

---

## Funding

This work was supported by the National IT Industry Promotion Agency (NIPA) grant funded by the Korea government (MSIT): "Development of Physical Data Quality Management Technologies for Physical AI," No. RS-2026-25621690.

---

## Citation (preprint)

> Yoon, Y. S., Lee, E. S., Kim, H., & Son, J. Y. (2026). *From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction.* Manuscript submitted to Engineering Applications of Artificial Intelligence.
