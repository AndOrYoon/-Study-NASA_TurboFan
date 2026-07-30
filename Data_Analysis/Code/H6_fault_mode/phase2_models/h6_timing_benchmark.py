# -*- coding: utf-8 -*-
"""
T2 — H6 Architecture Timing Benchmark
Measures actual parameter counts, FLOPs, and inference latency for M0-M3.

Outputs:
  Data_Analysis/Results/H6_fault_mode/H6_timing_benchmark.csv
"""

import sys, os, time
import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from h6_p2_model_utils import LSTMBranch, SoftGatingModel, AttentionGateModel

RESULTS_DIR = r"C:\BMAD_PY313\Data_Analysis\Results\H6_fault_mode"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
F        = 15       # FD003 features
W        = 30       # window size
K        = 10       # GatingNet initial cycles
WARMUP   = 200      # GPU warmup runs
REPEATS  = 2000     # timed runs
TRAIN_BS = 256      # training batch size
TRAIN_N  = 100      # dummy batches per epoch timing run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"F={F}, W={W}, K={K}, warmup={WARMUP}, repeats={REPEATS}\n")


# ── Parameter counter ─────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ── FLOPs calculator (LSTM analytical formula) ───────────────────────────────
def lstm_flops(input_size, hidden_size, seq_len):
    """8 × (input_size + hidden_size) × hidden_size per timestep (4 gates × 2 matmuls)."""
    return 8 * (input_size + hidden_size) * hidden_size * seq_len

def fc_flops(in_f, out_f):
    return 2 * in_f * out_f  # multiply-add

def branch_flops(n_feat, hidden=64, seq_len=30):
    return (
        lstm_flops(n_feat, hidden, seq_len)   # LSTM1
        + lstm_flops(hidden, hidden, seq_len)  # LSTM2
        + fc_flops(hidden, 32)                 # FC1
        + fc_flops(32, 1)                      # FC2
    )

def gating_flops(K, n_feat, hidden=32):
    flat = K * n_feat  # flatten
    return fc_flops(flat, hidden) + fc_flops(hidden, 2)


# ── Single-engine inference timer ─────────────────────────────────────────────
def time_inference(fn, warmup=WARMUP, repeats=REPEATS):
    """Returns (mean_ms, std_ms) over REPEATS runs after warmup."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    return float(np.mean(times)), float(np.std(times))


# ── Training epoch timer ──────────────────────────────────────────────────────
def time_train_epoch(model, forward_fn, n_batches=TRAIN_N, bs=TRAIN_BS):
    """Dummy training epoch: forward + backward + optimizer step."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # warmup
    for _ in range(5):
        forward_fn(model, optimizer, criterion, dummy_batch=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_batches):
        forward_fn(model, optimizer, criterion, dummy_batch=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0)

    # Scale to full epoch (FD003 train ~1700 seqs → ~7 batches @ bs=256)
    batches_per_epoch = 1700 / bs
    return elapsed / n_batches * batches_per_epoch  # seconds


# ── Build models ──────────────────────────────────────────────────────────────
m0 = LSTMBranch(F).to(device).eval()
m1_b0 = LSTMBranch(F).to(device).eval()   # M1: two independent branches (one used at inference)
m1_b1 = LSTMBranch(F).to(device).eval()
m2 = SoftGatingModel(F).to(device).eval()
m3 = AttentionGateModel(F, K=K).to(device).eval()

# ── Dummy inputs ──────────────────────────────────────────────────────────────
x_full  = torch.randn(1, W, F, device=device)   # single engine, full window
x_init  = torch.randn(1, K, F, device=device)   # M3 initial K cycles
w0_dummy = torch.tensor([0.6], device=device)
w1_dummy = torch.tensor([0.4], device=device)

x_train_full  = torch.randn(TRAIN_BS, W, F, device=device)
x_train_init  = torch.randn(TRAIN_BS, K, F, device=device)
y_train       = torch.randn(TRAIN_BS, 1, device=device)
w0_train      = torch.rand(TRAIN_BS, device=device)
w1_train      = 1 - w0_train

# ── FLOPs (analytical) ───────────────────────────────────────────────────────
flops_m0 = branch_flops(F)                                # 1 branch
flops_m1 = branch_flops(F)                                # 1 branch at inference (argmax routing)
flops_m2 = 2 * branch_flops(F)                           # 2 branches
flops_m3 = gating_flops(K, F) + 2 * branch_flops(F)     # GatingNet + 2 branches

# ── Measure inference latency (batch=1) ───────────────────────────────────────
print("Measuring inference latency (batch=1, single engine)...")

with torch.no_grad():
    m0_mean, m0_std = time_inference(lambda: m0(x_full))
    print(f"  M0: {m0_mean:.3f} ± {m0_std:.3f} ms")

    # M1 inference: argmax → one branch
    m1_mean, m1_std = time_inference(lambda: m1_b0(x_full))
    print(f"  M1: {m1_mean:.3f} ± {m1_std:.3f} ms (single branch, argmax routing)")

    m2_mean, m2_std = time_inference(lambda: m2(x_full, w0_dummy, w1_dummy))
    print(f"  M2: {m2_mean:.3f} ± {m2_std:.3f} ms")

    m3_mean, m3_std = time_inference(lambda: m3(x_full, x_init))
    print(f"  M3: {m3_mean:.3f} ± {m3_std:.3f} ms")

# ── Measure training epoch time ───────────────────────────────────────────────
print("\nMeasuring training time (scaled to FD003 epoch, batch=256)...")

def train_step_m0(model, opt, crit, dummy_batch=True):
    model.train()
    opt.zero_grad()
    y = model(x_train_full)
    loss = crit(y, y_train)
    loss.backward()
    opt.step()
    model.eval()

def train_step_m2(model, opt, crit, dummy_batch=True):
    model.train()
    opt.zero_grad()
    y_f, y0, y1 = model(x_train_full, w0_train, w1_train)
    loss = crit(y_f, y_train) + 0.1 * crit(y0, y_train) + 0.1 * crit(y1, y_train)
    loss.backward()
    opt.step()
    model.eval()

def train_step_m3(model, opt, crit, dummy_batch=True):
    model.train()
    opt.zero_grad()
    y_f, y0, y1 = model(x_train_full, x_train_init)
    loss = crit(y_f, y_train) + 0.05 * crit(y0, y_train) + 0.05 * crit(y1, y_train)
    loss.backward()
    opt.step()
    model.eval()

m0_train = time_train_epoch(LSTMBranch(F).to(device), train_step_m0)
print(f"  M0: {m0_train:.2f} s/epoch")

m1_train = m0_train  # M1 trains each branch independently, similar cost per branch
# Total M1 training ≈ 2× M0 (two branches trained separately)
m1_train_total = 2 * m0_train
print(f"  M1: {m1_train_total:.2f} s/epoch (2 branches × M0 cost)")

m2_train = time_train_epoch(SoftGatingModel(F).to(device), train_step_m2)
print(f"  M2: {m2_train:.2f} s/epoch")

m3_train = time_train_epoch(AttentionGateModel(F, K=K).to(device), train_step_m3)
print(f"  M3: {m3_train:.2f} s/epoch")

# ── Parameter counts ──────────────────────────────────────────────────────────
params_m0 = count_params(m0)
params_m1 = count_params(m1_b0) + count_params(m1_b1)   # both branches stored
params_m2 = count_params(m2)
params_m3 = count_params(m3)

print(f"\nParameter counts:")
print(f"  M0: {params_m0:,} ({params_m0/1e3:.1f}K)")
print(f"  M1: {params_m1:,} ({params_m1/1e3:.1f}K)")
print(f"  M2: {params_m2:,} ({params_m2/1e3:.1f}K)")
print(f"  M3: {params_m3:,} ({params_m3/1e3:.1f}K)")
print(f"     GatingNet only: {count_params(m3.gating):,} ({count_params(m3.gating)/1e3:.1f}K)")

print(f"\nFLOPs per inference window (analytical):")
print(f"  M0: {flops_m0/1e6:.2f}M")
print(f"  M1: {flops_m1/1e6:.2f}M (single branch, argmax)")
print(f"  M2: {flops_m2/1e6:.2f}M")
print(f"  M3: {flops_m3/1e6:.2f}M")

# ── Save results ──────────────────────────────────────────────────────────────
import csv
out_path = os.path.join(RESULTS_DIR, "H6_timing_benchmark.csv")
rows = [
    {
        "model": "M0", "label": "single branch",
        "params": params_m0, "params_K": round(params_m0/1e3, 1),
        "flops_M": round(flops_m0/1e6, 2),
        "infer_mean_ms": round(m0_mean, 3), "infer_std_ms": round(m0_std, 3),
        "train_s_epoch": round(m0_train, 2),
    },
    {
        "model": "M1", "label": "hard routing",
        "params": params_m1, "params_K": round(params_m1/1e3, 1),
        "flops_M": round(flops_m1/1e6, 2),
        "infer_mean_ms": round(m1_mean, 3), "infer_std_ms": round(m1_std, 3),
        "train_s_epoch": round(m1_train_total, 2),
    },
    {
        "model": "M2", "label": "soft gating",
        "params": params_m2, "params_K": round(params_m2/1e3, 1),
        "flops_M": round(flops_m2/1e6, 2),
        "infer_mean_ms": round(m2_mean, 3), "infer_std_ms": round(m2_std, 3),
        "train_s_epoch": round(m2_train, 2),
    },
    {
        "model": "M3", "label": "attention gate K=10",
        "params": params_m3, "params_K": round(params_m3/1e3, 1),
        "flops_M": round(flops_m3/1e6, 2),
        "infer_mean_ms": round(m3_mean, 3), "infer_std_ms": round(m3_std, 3),
        "train_s_epoch": round(m3_train, 2),
    },
]
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nResults saved: {out_path}")
print("\n=== Table III values (measured) ===")
print(f"{'Model':<28} {'Params':>8} {'FLOPs':>8} {'Infer (ms)':>14} {'Train (s/ep)':>14}")
print("-" * 78)
for r in rows:
    print(f"  {r['model']} ({r['label']:<22}) "
          f"{r['params_K']:>5.1f}K  "
          f"{r['flops_M']:>5.2f}M  "
          f"{r['infer_mean_ms']:>6.3f}±{r['infer_std_ms']:.3f}ms  "
          f"{r['train_s_epoch']:>8.2f}s")
