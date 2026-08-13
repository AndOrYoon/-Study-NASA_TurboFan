"""
RESS submission figure processor (Elsevier-compliant).

Elsevier/RESS standard: figure titles appear ONLY in the LaTeX caption,
NOT embedded inside the figure image.

This script:
  - For EDA-source figures (Fig1, Fig2): crops the matplotlib suptitle from
    the top of the image so no "Fig N —" label appears in the PNG.
  - For hypothesis-source figures (Fig3–Fig8): copies directly — the
    visualization scripts no longer add suptitles.

Destinations:
  Manuscript/Figures/<name>.png
  Manuscript/Submission/RESS/Fig<N>.png
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image
import os, shutil

BASE = r"C:\BMAD_PY313"

FIGURES = [
    # ── EDA figures: crop suptitle from the original PNG ──────────────────
    {
        "src": os.path.join(BASE, "Dataset", "Figure", "fig06_degradation_trends.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig1_degradation_trends.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig1.png"),
        ],
        "crop_top_px": 60,   # pixels occupied by the matplotlib suptitle band
    },
    {
        "src": os.path.join(BASE, "Dataset", "Figure", "fig10_rul_clipping.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig2_rul_clipping.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig2.png"),
        ],
        "crop_top_px": 0,
    },
    # ── Hypothesis figures: copy directly (no embedded titles in source) ───
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H2_clipping",
                            "figures", "fig_H2_01_rmse_heatmap.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig3_H1_clipping_rmse.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig3.png"),
        ],
        "crop_top_px": 0,
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H5_normalization",
                            "figures", "fig_H5_01_rmse_heatmap.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig4_H2_normalization_rmse.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig4.png"),
        ],
        "crop_top_px": 0,
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H5_normalization",
                            "figures", "fig_H5_05_statistical_test.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig5_H2_statistical_test.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig5.png"),
        ],
        "crop_top_px": 0,
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H6_fault_mode",
                            "figures", "fig_H6_02_ablation_silhouette.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig6_H3_gmm_clustering.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig6.png"),
        ],
        "crop_top_px": 0,
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H6_fault_mode",
                            "figures", "fig_H6_01_model_comparison.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig7_H3_model_comparison.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig7.png"),
        ],
        "crop_top_px": 0,
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H7_loss_function",
                            "figures", "fig_H7_05_clip_loss_interaction.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig8_H4_clip_loss_interaction.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig8.png"),
        ],
        "crop_top_px": 0,
    },
]


def process(cfg):
    src = cfg["src"]
    if not os.path.exists(src):
        print(f"  [SKIP] source not found: {src}")
        return

    crop_px = cfg.get("crop_top_px", 0)

    if crop_px > 0:
        img = Image.open(src).convert("RGB")
        w, h = img.size
        cropped = img.crop((0, crop_px, w, h))
        for dest in cfg["dests"]:
            cropped.save(dest, "PNG")
            print(f"  Saved (cropped {crop_px}px) → {dest}")
    else:
        for dest in cfg["dests"]:
            shutil.copy2(src, dest)
            print(f"  Copied → {dest}")


for cfg in FIGURES:
    print(f"\n{os.path.basename(cfg['src'])}")
    process(cfg)

print("\nAll done.")
