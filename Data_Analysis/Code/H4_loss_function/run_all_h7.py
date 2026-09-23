"""
H7 Master Runner — executes all phases in sequence.
Run this script to execute the full H7 experiment pipeline.

Usage:
    python run_all_h7.py [--skip-phase3]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable

def run(script: str, label: str):
    path = os.path.join(SCRIPT_DIR, script)
    print(f"\n{'='*60}")
    print(f"RUNNING: {label}")
    print(f"  {path}")
    print(f"{'='*60}")
    t0  = time.time()
    ret = subprocess.run([PYTHON, path], check=False)
    elapsed = time.time() - t0
    status  = "OK" if ret.returncode == 0 else f"FAILED (code {ret.returncode})"
    print(f"\n[{status}] {label}  ({elapsed/60:.1f} min)")
    return ret.returncode == 0


parser = argparse.ArgumentParser()
parser.add_argument("--skip-phase3", action="store_true",
                    help="Skip Phase 3 hyperparameter sweeps")
parser.add_argument("--phase2b-only", action="store_true",
                    help="Run only Phase 2b (assumes Phase 1 done)")
args = parser.parse_args()

wall_start = time.time()

if not args.phase2b_only:
    ok = run("00_pipeline_check.py",           "Phase 0: Pipeline Check")
    if not ok:
        print("Pipeline check failed — aborting.")
        sys.exit(1)

    run("04_phase1_linear_screening.py",        "Phase 1: Linear Screening (80 runs)")
    run("05_phase2a_lstm_pilot.py",             "Phase 2a: LSTM Pilot (25 runs)")

run("05_phase2b_lstm_full.py",                  "Phase 2b: Full LSTM (560 runs)")

if not args.skip_phase3:
    run("06_phase3a_lambda_grid.py",            "Phase 3a: L5 Lambda Grid (150 runs)")
    run("06_phase3c_pinball_tau.py",            "Phase 3c: Pinball τ Sweep (100 runs)")
    run("06_phase3d_huba_grid.py",              "Phase 3d: HubA Grid (45 runs)")

run("07_evaluation.py",                         "Phase 7: Statistical Evaluation")
run("08_visualization.py",                      "Phase 8: Visualization")

total = time.time() - wall_start
print(f"\n{'='*60}")
print(f"H7 COMPLETE — total wall time: {total/3600:.2f} hours")
print(f"{'='*60}")
