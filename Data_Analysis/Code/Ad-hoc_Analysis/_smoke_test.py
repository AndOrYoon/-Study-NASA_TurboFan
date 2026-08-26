import sys, importlib
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, "Data_Analysis/Code/H5_normalization")
sys.path.insert(0, "Data_Analysis/Code/H6_fault_mode/phase2_models")
sys.path.insert(0, "Data_Analysis/Code/shared")
sys.path.insert(0, "Data_Analysis/Code/Ad-hoc_Analysis")

runner = importlib.import_module("02_run_unified")
row = runner.run_one("A", "N1", "M0", "FD001", 0)
print(f"RMSE={row['rmse']:.2f}  NASA={row['nasa_score']:.1f}  elapsed={row['elapsed_s']:.0f}s")

# M3 smoke test
row3 = runner.run_one("C", "N1", "M3", "FD003", 0)
print(f"M3 FD003: RMSE={row3['rmse']:.2f}  conf={row3['gate_conf']:.3f}  elapsed={row3['elapsed_s']:.0f}s")
