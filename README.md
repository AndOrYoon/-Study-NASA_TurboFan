# Turbofan Engine RUL Prediction: A Cross-Dataset Ablation Study

**Paper title:** From Fleet Normalization to Fault-Mode Gating: A Systematic Ablation Study of Turbofan Remaining Useful Life Prediction  
**Target journal:** Reliability Engineering & System Safety (Elsevier, RESS)  
**Status:** Submission ready (September 2026)

---

## Overview

This repository contains the full experimental pipeline and manuscript for a controlled ablation study on turbofan engine Remaining Useful Life (RUL) prediction using the NASA CMAPSS benchmark (FD001–FD004).

Four interdependent design decisions are systematically isolated and tested:

| Hypothesis | Factor | Levels |
|-----------|--------|--------|
| H2 | RUL label clipping threshold | 5 (75, 100, 125, 130, None) |
| H5 | Sensor normalization strategy | 7 (N1–N7) |
| H6 | Fault-mode architecture | 4 (M0–M3) |
| H7 | Training loss function | 7 (L1–L7) |

All comparisons are Wilcoxon rank-sum tested with Benjamini-Hochberg FDR correction (α = 0.05).  
**Total training runs: 800+** (5 seeds × conditions × 4 datasets)

---

## Dataset

**NASA CMAPSS** (C-MAPSS: Commercial Modular Aero-Propulsion System Simulation)

| Sub-dataset | Train engines | Test engines | Op. conditions | Fault modes |
|-------------|--------------|-------------|---------------|-------------|
| FD001 | 100 | 100 | 1 | HPC degradation only |
| FD002 | 260 | 259 | 6 | HPC degradation only |
| FD003 | 100 | 100 | 1 | HPC + Fan degradation |
| FD004 | 249 | 248 | 6 | HPC + Fan degradation |

- 21 raw sensors, space-separated `.txt` format, no header
- Column order: `unit cycle op1 op2 op3 s1…s21`

---

## Shared Backbone

Stacked LSTM used across all hypotheses:

```
LSTM1(64) → full sequence → Dropout(0.2)
LSTM2(64) → last timestep → Dropout(0.2) → FC(64→32→ReLU→1)
```

Window: 30 cycles | Batch: 256 | LR: 1e-3 (Adam) | WD: 1e-4 | Patience: 15

---

## Key Results

### H2 — RUL Clipping (Ridge regression, 20 runs)

| Clip | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|------|-----------|-----------|-----------|-----------|
| 75 | 33.65 | 47.37 | 32.48 | 50.83 |
| 100 | 24.51 | 37.67 | 23.00 | 40.37 |
| **125** | **21.90** | **32.39** | **21.62** | **34.61** |
| 130 | 22.06 | 31.83 | 22.24 | 34.04 |
| None | 31.90 | 33.05 | 56.09 | 46.99 |

**Verdict (partially accepted):** clip=125 is optimal or statistically tied-optimal across all datasets. clip=None yields catastrophic NASA Score on FD003 (4,014,724 vs. 13.09 at clip=125, a 306,000-fold increase).

---

### H5 — Sensor Normalization (LSTM, 5 seeds × 7 strategies × 4 datasets = 140 runs)

- **Fleet MinMax (N1):** Best on FD001/FD002/FD004 (RMSE ≈ 14.1–14.6); significantly outperforms all alternatives under BH-FDR correction
- Per-unit strategies (N3–N6): significantly inferior on FD001, FD002, FD004
- RevIN (N7): competitive on FD001 (14.92), inferior on FD002/FD004 (18.2+)
- FD003 N1 std = 12.86 (vs. ≤1.84 elsewhere): anomalous variance is a **diagnostic signal of latent fault-mode heterogeneity**, not a normalization failure

**Verdict (rejected):** Fleet MinMax is the clear winner. Per-unit adaptation is counterproductive on homogeneous fleets.

---

### H6 — Fault-Mode Architecture (LSTM, 5 seeds, FD003/FD004)

| Model | FD003 RMSE±std | FD003 NASA | FD004 RMSE±std | FD004 NASA |
|-------|---------------|-----------|---------------|-----------|
| M0 — Single LSTM (baseline) | 43.23±0.18 | 34,339 | 28.05±1.74 | 10,586 |
| M1 — Hard Routing (GMM) | 32.45±11.37 | 27,098 | 49.20±7.17 | 132,003 |
| M2 — Soft Gating (GMM) | 26.16±14.81 | 20,881 | 30.71±0.73 | 50,715 |
| **M3 — Attention Gate (K=10)** | **14.78±1.32** | **425** | **28.33±1.03** | **13,229** |

**Verdict (accepted — M3):** M3 achieves −65.8% RMSE and −98.8% NASA Score on FD003 vs. M0. Routes engines from first 10 cycles only (applicable at commissioning time). M1 collapses on FD004 due to test-time cluster assignment collapse (247:1 ratio).

---

### H7 — Loss Function (LSTM, 5 seeds × 7 losses × 4 clips × 4 datasets = 560 runs)

- No custom loss achieves statistically significant improvement over MSE (L1) after BH-FDR correction across 96 comparisons (minimum p_BH = 0.176)
- clip=None is the dominant source of NASA Score variance, not the loss function choice

**Verdict (rejected):** Custom loss functions provide no statistically detectable benefit. RUL clipping dominates loss function selection.

---

## Three-Tier Design Hierarchy (Core Contribution)

```
TIER 1 [Critical] — Label Engineering    →  clip = 125
                                              Up to 306,000× NASA Score difference
        ↓
TIER 2 [Important] — Fault-Mode Architecture  →  M3 Attention Gate (FD003/FD004)
                                                   −65.8% RMSE on FD003
        ↓
TIER 3 [Secondary] — Loss Function        →  MSE (L1) default
                                              No statistically detectable gain
```

Each tier produces qualitatively larger effects than the tier below it.

---

## Repository Structure

```
C:\BMAD_PY313\
├── Dataset/                        ← Raw CMAPSS .txt files + EDA outputs
│   ├── train/test_FD00{1-4}.txt
│   ├── RUL_FD00{1-4}.txt
│   └── Figure/                     ← EDA plots (fig01–fig12)
├── Hypothesis/                     ← Literature review, research gap analysis
├── Data_Analysis/
│   ├── Analysis_Plan.md            ← Canonical experiment design
│   ├── Code/
│   │   ├── shared/
│   │   │   └── op_condition_utils.py   ← K-means residualization (H5/H6)
│   │   ├── H2_clipping/            ← Scripts 01–05
│   │   ├── H5_normalization/       ← Scripts 01–06
│   │   ├── H6_fault_mode/          ← phase1/phase2/phase3
│   │   └── H7_loss_function/       ← Scripts 00–08 + run_all_h7.py
│   └── Results/                    ← Output CSVs, figures, model checkpoints
└── Manuscript/
    ├── Full-Text_Manuscript/
    │   └── manuscript_full_text.md     ← Primary compiled manuscript
    ├── Sections/                   ← Source section files
    ├── Figures/                    ← Final manuscript figures (Fig1–Fig8)
    ├── Tables/                     ← Tables 1–5 CSV
    └── Submission/
        ├── RESS/                   ← Submission package (main.tex + figures)
        │   ├── main.tex
        │   ├── main.pdf
        │   ├── Fig1–Fig8.png
        │   ├── highlights.txt
        │   ├── declaration_ai_use.txt
        │   ├── conflict_of_interest.txt
        │   └── credit_author_statement.txt
        └── Cover_Letter/
            └── Cover_Letter_RESS_draft.md
```

---

## Submission Status

| Item | Status |
|------|--------|
| All experiments (H2/H5/H6/H7) | ✅ Complete |
| Manuscript (full text) | ✅ Complete |
| LaTeX conversion (elsarticle) | ✅ Complete |
| Figures (Fig1–Fig8) | ✅ Complete |
| Cover Letter | ✅ Complete (10 September 2026) |
| Funding acknowledgement | ✅ NIPA No.RS-2026-25621690 |
| AI use declaration | ✅ Confirmed |
| Abstract word count | ✅ ≤250 words |
| Keywords | ✅ 5 keywords finalized |
| PDF compilation | ✅ 67 pages |
| **RESS submission** | 🔲 In progress |

---

## Environment

```powershell
# Activate virtual environment
NASA_TurboFan\Scripts\activate

# Run full H7 experiment
python Data_Analysis\Code\H7_loss_function\run_all_h7.py

# Run single hypothesis stage
python Data_Analysis\Code\H5_normalization\04_run_experiments.py
```

- **Python:** 3.13 (TensorFlow not supported — PyTorch used)
- **Virtual environment:** `C:\BMAD_PY313\NASA_TurboFan\`

---

## Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," in *Proc. 1st Int. Conf. Prognostics and Health Management (PHM08)*, Denver, CO, 2008.
