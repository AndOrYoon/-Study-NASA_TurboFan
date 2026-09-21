# Conclusion
## Mean-Prediction Collapse in Deep Remaining Useful Life Regression

**작성일:** 2026-09-21  
**버전:** Draft v0.1  
**관련 파일:** `Research_Plan.md` §9, `Essence_Research.md` §7, §11

---

> **작성 방침:** 명료하고 쉽게. RQ1–4 결론 → 핵심 권고 → Future work. 새로운 내용 금지.

---

## 9. Conclusion

This study examined Mean-Prediction Collapse (MPC) — a failure in which a trained LSTM outputs a near-constant prediction regardless of input — in LSTM-based RUL regression on C-MAPSS.

**What causes MPC?** The trigger is a three-way interaction: a fixed validation split across training seeds, early stopping without a warmup period, and MSE loss on a homogeneous dataset. No single factor alone is sufficient. When all three are present, the LSTM's forget gate approaches zero in the first epoch of training. This blocks gradient flow through the cell state via BPTT, trapping the model at a trivial constant-prediction solution. Early stopping then halts training at this degenerate state. Validation loss decreases throughout — the collapse is silent.

**What architecture types are affected?** MPC is LSTM-specific. GRU, 1D-CNN, and MLP achieve 0% MPC under the same trigger conditions. Among C-MAPSS subsets, only FD003 shows substantial MPC rates (80% for LSTM). FD002 and FD004 are immune because their six operating conditions prevent the trivial solution from being stable.

**Can MPC be detected without the test set?** Yes. At training completion, computing the Prediction Dispersion Ratio (PDR = std(ŷ) / std(y)) requires only the validation-set predictions. PDR < 0.05 identifies MPC with sensitivity = 1.00 and specificity = 1.00 on both calibration (270 runs) and held-out (330 runs) data. AUROC = 1.0000.

**How can MPC be prevented?** Five interventions eliminate MPC completely, each with no cost to RMSE:

- Initialize the output layer bias to the training-set mean RUL. One line of code.
- Set a minimum training period (e.g., MIN_EPOCHS = 30) before early stopping activates.
- Use MAE loss instead of MSE.
- Replace LSTM with GRU.
- Fix the forget gate to 1 (restoring the original CEC design).

The cheapest option is bias initialization. It requires no architectural change, no additional computation, and no hyperparameter tuning.

**Is MPC present in published work?** We replicated five published training protocols from EAAI and Expert Systems with Applications. MPC rates ranged from 60% to 90% across 10 independent seeds per protocol. Of 50 total replication runs, 41 collapsed (82%). These protocols are representative of common practice in FD003+LSTM RUL research.

The central finding is this: published RMSE values on FD003 that were compared against LSTM baselines trained under these protocols may reflect a 70–77% apparent improvement driven entirely by baseline collapse, not algorithmic progress. The same method compared against a correctly trained baseline shows approximately 8% improvement.

**Core recommendation.** Add one line to any LSTM training loop:

```python
model.fc[-1].bias.data.fill_(train_rul_mean)   # prevents MPC
```

Then, after early stopping:

```python
pdr = np.std(val_preds) / (np.std(val_targets) + 1e-8)
if pdr < 0.05:
    print("WARNING: MPC — retrain with corrected protocol")
```

These two additions cost nothing and eliminate both the failure and its silent propagation into benchmarks.

**Future work.** This study is limited to C-MAPSS and LSTM architectures. Three extensions are natural. First, whether analogous failures occur in Transformer-based or attention-enhanced RUL models under similar protocol conditions remains an open question. Attention mechanisms may create different but related trivial attractors. Second, real sensor data from turbofan fleets (e.g., N-CMAPSS [P4]) has more heterogeneity than C-MAPSS; MPC vulnerability may be lower, but the detection and prevention tools developed here apply regardless. Third, direct measurement of forget gate activations and BPTT gradient norms during collapsed and normal training runs would replace the current causal inference (from intervention) with direct mechanistic evidence.

---

## Reference Placeholders

- [P4] Chao et al. (2020/2021) — N-CMAPSS dataset
