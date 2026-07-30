# Code Verification Report
> Generated: 2026-07-03

---

## Issue A: H5 vs H6 FD003 Baseline Inconsistency

### Finding

H5 N1/FD003 (RMSE = 19.05 ± 12.86) and H6 M0/FD003 (RMSE = 43.23 ± 0.18) are NOT the same experiment. Four concrete differences were found:

**Difference 1 — LSTM Architecture**

H5 uses `LSTMBase` (file: `03_lstm_model.py`, lines 68–92):
- LSTM1(n_features → 64) → Dropout(0.2)
- LSTM2(64 → **32**) → Dropout(0.2) → FC(**32** → 16 → ReLU → 1)

H6 uses `LSTMBranch` (file: `h6_p2_model_utils.py`, lines 238–257):
- LSTM1(n_features → 64) → Dropout(0.2)
- LSTM2(64 → **64**) → Dropout(0.2) → FC(**64** → 32 → ReLU → 1)

The hidden dimension of LSTM2 differs (32 vs 64), as does the final FC head.

**Difference 2 — Feature Set (Input Dimension)**

H5 `get_feature_cols` (`01_data_loader.py`, lines 51–54) returns everything that is not `unit`, `cycle`, or `rul`. For FD003 this includes `op1, op2, op3` plus 15 sensors = **18 features**.

H6 `get_sensor_cols` (`h6_p2_model_utils.py`, lines 90–92) returns only sensor columns (s1–s21 minus constants). For FD003 this is **15 features** (op columns excluded entirely).

**Difference 3 — Train/Val Split Method**

H5 `val_split` (`04_run_experiments.py`, lines 125–134): always selects the engines with the highest unit IDs. For FD003 (100 engines), this is always engines 81–100. The split is fully deterministic and seed-independent.

H6 `split_engines` (`h6_p2_model_utils.py`, lines 188–196): uses `np.random.RandomState(seed=42)` to randomly choose 20 engines. The split is also deterministic across experiment seeds (hardcoded `seed=42` at `h6_p2_baseline_lstm.py` line 51), but selects a DIFFERENT 20 engines than H5.

**Difference 4 — Test Prediction and Ground Truth Clipping**

H5 (`04_run_experiments.py`, lines 251–252):
```python
true_rul = load_true_rul(dataset, clip=RUL_CLIP)   # clips ground truth to 125
pred_rul = np.clip(pred_rul, 0, RUL_CLIP)           # clips predictions to [0, 125]
```

H6 `load_test_rul` (`h6_p2_model_utils.py`, lines 73–78): returns raw RUL values with no clipping. H6 `predict_sequences` does not clip predictions. RMSE is computed on unclipped values.

### Evidence

```python
# H5 val_split — 04_run_experiments.py, lines 125-134
def val_split(train_df: pd.DataFrame, val_frac: float = VAL_FRAC):
    units = np.array(sorted(train_df["unit"].unique()))
    n_val = max(1, int(len(units) * val_frac))
    val_units = set(units[-n_val:])   # always last N engines by ID
    mask = train_df["unit"].isin(val_units)
    return (train_df[~mask].reset_index(drop=True),
            train_df[mask].reset_index(drop=True))

# H6 split_engines — h6_p2_model_utils.py, lines 188-196
def split_engines(units: np.ndarray, val_frac: float = 0.2, seed: int = 42):
    unique_units = np.unique(units)
    rng  = np.random.RandomState(seed)
    n_val = max(1, int(len(unique_units) * val_frac))
    val_u = set(rng.choice(unique_units, n_val, replace=False).tolist())
    tr_m  = np.array([u not in val_u for u in units])
    va_m  = ~tr_m
    return tr_m, va_m

# H6 baseline script — h6_p2_baseline_lstm.py, line 51
tr_m, va_m = split_engines(units, val_frac=0.2, seed=42)  # hardcoded seed

# H5 architecture — 03_lstm_model.py, lines 76-92
self.lstm2 = nn.LSTM(64, 32, batch_first=True)   # hidden=32
self.fc    = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

# H6 architecture — h6_p2_model_utils.py, lines 241-250
self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)  # hidden=64
self.fc    = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

# H5 feature cols — 01_data_loader.py, lines 51-54
def get_feature_cols(df):
    exclude = {"unit", "cycle", "rul"}
    return [c for c in df.columns if c not in exclude]  # includes op1/op2/op3

# H6 feature cols — h6_p2_model_utils.py, lines 90-92
def get_sensor_cols(dataset):
    const = CONST_SENSORS.get(dataset, set())
    return [f"s{i}" for i in range(1, 22) if f"s{i}" not in const]  # sensors only
```

### Conclusion

The discrepancy is fully explained by the four differences above. These are not the same experiment. The primary drivers of the RMSE gap (19.05 vs 43.23) are likely the architecture difference (LSTM2 hidden=32 vs 64, different FC heads) and the different training data (different 20-engine val splits meaning different 80-engine training sets). The difference in variance (±12.86 vs ±0.18) is consistent with H5's smaller model being more sensitive to random weight initialization, while H6's larger, fixed-split training converges to the same solution across seeds.

The manuscript should NOT present these two rows as equivalent baselines. If an apples-to-apples comparison of N1 vs M0 on FD003 is needed, the architectures, feature sets, val split method, and test clipping must be unified first.

---

## Issue B: Cohen's d vs Wilcoxon N

### Finding

Both Cohen's d and the Wilcoxon test in `07_evaluation.py` use **identical arrays** with N=5 (one per-seed NASA score per condition). There is no pseudo-replication; the sample size is the same for both statistics. The combination "d = 2.00, p_BH = 1.0" is mathematically self-consistent.

The d sign convention in the code defines `cohens_d(alt, bl) = (alt.mean() - bl.mean()) / pooled_std`. A **positive** d means the alternative loss has higher (worse) NASA Score than the baseline. The one-sided Wilcoxon test uses `alternative="less"`, which tests whether `alt < bl` (i.e., improvement). When d is large and positive (loss is much worse), the test correctly returns p ≈ 1.0 — the null hypothesis of no improvement cannot be rejected in the direction of improvement.

There is a separate power issue: with N=5, the minimum achievable one-sided Wilcoxon p-value is 0.03125 (1/2^5 = 1/32). After BH-FDR correction over 96 tests (see Issue C), this minimum p scales to 0.03125 × 96 / 1 ≈ 3.0, which is capped at 1.0. So even a loss function that is unambiguously better across all 5 seeds cannot pass BH correction given only 5 seeds and this family size.

### Evidence

```python
# 07_evaluation.py, lines 54-84
for clip in clip_vals:
    for ds in DATASETS:
        bl = df[(df["clip"] == clip) & (df["loss_fn"] == baseline) &
                (df["dataset"] == ds)]["nasa_score"].values   # shape: (5,)

        for loss in LOSS_NAMES:
            if loss == baseline:
                continue
            alt = df[(df["clip"] == clip) & (df["loss_fn"] == loss) &
                     (df["dataset"] == ds)]["nasa_score"].values   # shape: (5,)
            if len(alt) == 0 or len(alt) != len(bl):
                continue

            stat, p = stats.wilcoxon(alt, bl, alternative="less")   # N=5, paired
            d = cohens_d(alt, bl)   # comment says "negative d → loss beats baseline"
                                    # i.e., POSITIVE d means loss is WORSE

# cohens_d definition — 07_evaluation.py, lines 43-46
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return (a.mean() - b.mean()) / pooled_std  # positive → a worse than b
```

### Conclusion

There is **no pseudo-replication**. Both statistics use the same N=5 seed-level arrays. The d = 2.00, p_BH = 1.0 combination indicates that the loss function is substantially worse than MSE baseline (d positive = alt is higher NASA Score = worse), and the BH-corrected test correctly gives p = 1.0 (no evidence of improvement). The manuscript should clarify the sign convention: positive d means the loss is worse. Additionally, the manuscript should note that N=5 gives insufficient power to survive BH-FDR correction over a ~96-test family; the minimum achievable post-BH p is approximately 0.03125 × 96 ≈ 3.0 (clipped to 1.0), meaning no comparison can be statistically significant regardless of effect size.

---

## Issue C: BH Family Size

### Finding

The code in `07_evaluation.py` constructs the comparison family by looping over `clip_vals × DATASETS × (LOSS_NAMES minus baseline)`. From `config_h7.py`:
- 4 clips: `clip_100`, `clip_125`, `clip_130`, `clip_none`
- 4 datasets: FD001, FD002, FD003, FD004
- 7 losses total, 1 is baseline (L1_MSE), so **6 non-baseline losses**

Total = 4 × 4 × 6 = **96 comparisons**.

The manuscript states "84 comparisons (6 loss functions × 4 clips × 4 datasets)". The listed formula evaluates to 6 × 4 × 4 = **96**, not 84. The number "84" is arithmetically inconsistent with the manuscript's own description.

### Evidence

```python
# config_h7.py, lines 18-27
CLIPS = {
    "clip_100":  100,
    "clip_125":  125,
    "clip_130":  130,
    "clip_none": None,
}
DATASETS   = ["FD001", "FD002", "FD003", "FD004"]
LOSS_NAMES = ["L1_MSE", "L2_NASA", "L3_DynMSE", "L4_Focal",
              "L5_TWA", "L6_Pinball", "L7_HubA"]

# 07_evaluation.py, lines 52-84 — the comparison loop
clip_vals = df["clip"].unique()   # dynamically read from data; matches config (4 values)

for clip in clip_vals:            # 4 iterations
    for ds in DATASETS:           # 4 iterations
        ...
        for loss in LOSS_NAMES:   # 7 iterations
            if loss == baseline:
                continue          # skip L1_MSE → 6 per (clip, dataset) pair
            ...
            test_rows.append({...})

test_df = pd.DataFrame(test_rows)
test_df["p_BH"] = bh_fdr(test_df["p_raw"].values)   # len(test_rows) = 96
```

### Conclusion

The correct BH family size is **96** (4 clips × 4 datasets × 6 non-baseline losses). The manuscript's stated "84" is a factual error. The formula "6 loss functions × 4 clips × 4 datasets" evaluates to 96, not 84. The manuscript must correct this number. Note that this error also affects the power analysis in Issue B: the correct family size (96, not 84) makes the minimum achievable BH-adjusted p even larger (0.03125 × 96 ≈ 3.0 vs 0.03125 × 84 ≈ 2.6 — both capped at 1.0), so the qualitative conclusion does not change.

---

## Issue D: H2 Wilcoxon Unit of Observation

### Finding

The unit of observation is **per-engine RMSE** — one value per test engine. The raw prediction CSV files contain one row per test engine with a column `sq_error` (squared error). The code takes `np.sqrt(sq_error)` for each engine individually. There are no "20 runs" in the sense of independent repetitions.

H2 uses `sklearn.linear_model.LinearRegression`, which is fully deterministic (no random state, no seeds). Each (dataset, clip) combination is run **exactly once**. The "20 runs" in CLAUDE.md means 5 clip values × 4 datasets = 20 unique configurations, each deterministic and non-repeated.

**Critical statistical concern:** The test uses `scipy.stats.ranksums` (Mann-Whitney U, unpaired/independent samples), treating the two conditions as independent groups. However, the SAME test engines are evaluated under both clip conditions, so the errors are correlated (the same engine appears in both groups). The appropriate test for paired samples would be Wilcoxon signed-rank (`scipy.stats.wilcoxon`). Using an unpaired test on paired data reduces statistical efficiency.

### Evidence

```python
# 04_statistical_test.py, lines 48-78
base_df   = pd.read_csv(base_path)
base_rmse = np.sqrt(base_df['sq_error'].values)   # per-engine RMSE, shape: (N_engines,)

pred_df   = pd.read_csv(pred_path)
pred_rmse = np.sqrt(pred_df['sq_error'].values)   # same engines, different clip

stat, p_val = stats.ranksums(pred_rmse, base_rmse)   # unpaired Mann-Whitney U

# 02_run_experiments.py, lines 135-153 — single deterministic run per (clip, dataset)
def run_experiment(fd_id, clip_value):
    ...
    model = LinearRegression()
    model.fit(X_train_sc, y_train)   # no random state, fully deterministic
    ...
    return { 'rmse': ..., 'nasa_score': ... }   # one result per (clip, dataset)

# main loop — no seeds, no repeated runs
for fd_id in FD_IDS:
    for clip_val in CLIP_VALUES:
        result = run_experiment(fd_id, clip_val)   # called once each
```

### Conclusion

The Wilcoxon test in H2 is valid in the sense that it operates on real, non-fabricated data (per-engine errors, N ≈ 100–259 per dataset). However, it uses an **unpaired** Mann-Whitney U test (`ranksums`) on data that is inherently **paired** (same test engines in both groups). The manuscript should: (1) correct the description — the test statistic is Mann-Whitney U, not Wilcoxon signed-rank; (2) acknowledge that the "20 runs" are 20 distinct deterministic configurations, not independent replications; (3) note that a paired Wilcoxon signed-rank test would be more statistically appropriate and should be reported in a revised analysis.

---

## Issue E: RevIN Denormalization

### Finding

The `RevIN.forward()` method computes `mean` and `std` as local variables, normalizes the input tensor, and returns the normalized tensor. The statistics are **not stored** as instance attributes and there is no reverse/denormalization method. The `LSTMWithRevIN.forward()` applies `self.revin(x)` to normalize input features, then passes the normalized sequence through two LSTM layers and a fully-connected head, returning a raw RUL scalar. No denormalization step is applied to the model output.

This is the **correct behavior** for RUL regression: the model output is a scalar in units of engine cycles (0–125 after clipping), which is not in the same unit-space as the input sensor features. Applying the inverse RevIN transformation to the output would be dimensionally incorrect. The comparison with N1–N6 is fair: all normalizers transform only the input features; none modifies the RUL target or the model output.

There is, however, a limitation worth noting in the manuscript: the original RevIN paper (Kim et al., ICLR 2022) was designed for time series forecasting where the output is future values in the same unit as the input. The denormalization step in the original paper is meaningless for RUL regression and cannot be replicated. The implementation uses only the "normalize inputs" half of RevIN, which is equivalent to per-sample (instance) normalization with learnable affine parameters.

### Evidence

```python
# 03_lstm_model.py, lines 114-121 — RevIN forward (no stored state, no reverse)
def forward(self, x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True).detach()   # local variable only
    std  = x.std(dim=1, keepdim=True, unbiased=False).detach() + self.eps
    x    = (x - mean) / std
    if self.affine:
        x = x * self.gamma + self.beta
    return x    # normalized features; mean/std NOT saved as self.mean_/self.std_

# 03_lstm_model.py, lines 147-153 — LSTMWithRevIN forward
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x       = self.revin(x)              # normalize input features only
    out, _ = self.lstm1(x)
    out     = self.drop1(out)
    out, _ = self.lstm2(out)
    out     = self.drop2(out[:, -1, :])
    return self.fc(out).squeeze(-1)      # raw RUL in cycles — no denorm

# 04_run_experiments.py, lines 251-254 — test evaluation (same for all normalizers)
true_rul = load_true_rul(dataset, clip=RUL_CLIP)   # ground truth in cycles
pred_rul = predict_all(model, X_test, device)       # raw model output in cycles
pred_rul = np.clip(pred_rul, 0, RUL_CLIP)           # clip to valid range
rmse = float(np.sqrt(np.mean((pred_rul - true_rul) ** 2)))
```

### Conclusion

No bug exists. The `RevIN` implementation correctly normalizes input features per sample without applying denormalization to the RUL output. This is architecturally appropriate for RUL regression. The manuscript should clarify that N7 applies only the forward (normalization) pass of RevIN to input features, not the reverse (denormalization) pass to outputs, and explain why this is semantically correct (RUL is not in sensor units). No additional experiment is strictly required, but the manuscript should distinguish this usage from the original time-series forecasting context of RevIN.

---

## Summary: Manuscript Corrections Required

| Issue | Section requiring correction |
|-------|------------------------------|
| **A** | The claim that H6 M0/FD003 serves as a comparable baseline for H5 N1/FD003 must be removed or qualified; the two experiments differ in architecture (LSTM hidden 32 vs 64), feature set (18 vs 15 inputs), train/val engine selection, and test RUL clipping. |
| **B** | The framing of "d=2.00 but p_BH=1.0" as paradoxical must be corrected: positive d means the loss is worse than baseline (not better), and the p=1.0 is expected; additionally, the methods section should note that N=5 combined with a 96-test BH family makes statistical significance mathematically impossible (minimum adjusted p ≈ 3.0). |
| **C** | "84 comparisons" must be corrected to **96 comparisons** (6 losses × 4 clips × 4 datasets = 96); the arithmetic in the manuscript is wrong. |
| **D** | The H2 statistical test description must be corrected to "Mann-Whitney U (unpaired, two-sided)" rather than "Wilcoxon"; the "20 runs" must be clarified as "20 deterministic configurations (5 clips × 4 datasets), each run once"; and the limitations section should note that a paired test on per-engine errors would be statistically more appropriate. |
| **E** | The RevIN (N7) description must clarify that only the input-normalization pass is implemented (no output denormalization), and explain that this is correct for RUL regression because the model output is in cycle units, not in sensor units. |
