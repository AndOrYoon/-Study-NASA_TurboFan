# Abstract
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression

**작성일:** 2026-09-21  
**버전:** Draft v0.1  
**목표 단어 수:** 200–230 words

---

## Abstract

Deep learning models for remaining useful life (RUL) prediction are routinely evaluated against LSTM baselines on the C-MAPSS benchmark. We show that a subset of these baselines exhibits Mean-Prediction Collapse (MPC): the model outputs a near-constant prediction equal to the training-set mean, regardless of input, achieving no better RMSE than a trivial constant predictor. MPC is silent — validation loss decreases normally, and the failure is invisible without inspecting prediction dispersion.

We conduct a controlled experiment on C-MAPSS FD001–FD004 to identify when MPC occurs, why, and how to prevent it. Under the trigger conditions common in published work — a fixed validation split combined with early stopping from epoch zero — FD003+LSTM collapses at a rate of 80% across independent training seeds. GRU, 1D-CNN, and MLP are unaffected under identical conditions. The structural cause is LSTM forget gate saturation: as the gate approaches zero, backpropagation through the cell state is blocked, and the model is trapped at a trivial solution. Clamping the gate at ε ≤ 0.05 does not help; fixing it at 1 (restoring the original constant error carousel design) eliminates MPC entirely.

Five independent interventions achieve zero MPC with no RMSE penalty. The lowest-cost option is initializing the output layer bias to the training-set mean RUL — a single line of code. For retrospective diagnosis, the Prediction Dispersion Ratio (PDR = std(ŷ)/std(y)) computed at training completion identifies MPC with AUROC = 1.0000 across 600 runs, without test-set access.

We replicate five published training protocols and find MPC rates of 60–90% per protocol (41 of 50 total runs collapsed). In the case that motivated this study, correcting the baseline protocol reduced an apparent 65.8% improvement to a statistically non-significant difference. Protocol correction, not better models, drove the change. The mechanism explanation, zero-cost prevention options, and test-free diagnostic introduced here give PHM researchers the tools to verify that LSTM baselines are valid before drawing comparative conclusions.

---

**Keywords:** remaining useful life, LSTM, mean-prediction collapse, early stopping, benchmark reliability, prognostics and health management
