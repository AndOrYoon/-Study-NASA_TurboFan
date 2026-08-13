"""
v2: Start from original source PNGs.
- Add new white band at TOP (expand canvas upward)
- Write "Fig. X — Title" in the new band
- White out the OLD internal title in the original portion (now shifted down)
- No existing plot content is touched
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image, ImageDraw, ImageFont
import os

BASE = r"C:\BMAD_PY313"

FIGURES = [
    {
        "src": os.path.join(BASE, "Dataset", "Figure", "fig06_degradation_trends.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig1_degradation_trends.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig1.png"),
        ],
        "title": "Fig. 1 — Top-6 Sensor Degradation Trends (FD003, 10 engines + fleet mean)",
        "old_title_px": 58,   # approx height of "Fig 6 —" suptitle in original
    },
    {
        "src": os.path.join(BASE, "Dataset", "Figure", "fig10_rul_clipping.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig2_rul_clipping.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig2.png"),
        ],
        "title": "Fig. 2 — Effect of RUL Clipping Strategy (FD001)",
        "old_title_px": 52,
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H2_clipping", "figures", "fig_H2_01_rmse_heatmap.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig3_H1_clipping_rmse.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig3.png"),
        ],
        "title": "Fig. 3 — RMSE by Dataset and Clipping Threshold (H1: LinearRegression)",
        "old_title_px": 72,   # 2-line title
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H5_normalization", "figures", "fig_H5_01_rmse_heatmap.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig4_H2_normalization_rmse.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig4.png"),
        ],
        "title": "Fig. 4 — RMSE by Normalizer × Dataset (H2: mean over 5 seeds)",
        "old_title_px": 48,
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H5_normalization", "figures", "fig_H5_05_statistical_test.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig5_H2_statistical_test.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig5.png"),
        ],
        "title": "Fig. 5 — BH-FDR Adjusted p-values vs N1 Baseline (H2)",
        "old_title_px": 42,
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H6_fault_mode", "figures", "fig_H6_02_ablation_silhouette.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig6_H3_gmm_clustering.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig6.png"),
        ],
        "title": "Fig. 6 — Clustering Silhouette by Feature Variant (H3, GMM k=2)",
        "old_title_px": 44,
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H6_fault_mode", "figures", "fig_H6_01_model_comparison.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig7_H3_model_comparison.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig7.png"),
        ],
        "title": "Fig. 7 — Multi-Branch LSTM RMSE by Model (H3)",
        "old_title_px": 44,
    },
    {
        "src": os.path.join(BASE, "Data_Analysis", "Results", "H7_loss_function", "figures", "fig_H7_05_clip_loss_interaction.png"),
        "dests": [
            os.path.join(BASE, "Manuscript", "Figures", "Fig8_H4_clip_loss_interaction.png"),
            os.path.join(BASE, "Manuscript", "Submission", "RESS", "Fig8.png"),
        ],
        "title": "Fig. 8 — Clip × Loss Interaction Effect on NASA Score (H4)",
        "old_title_px": 40,
    },
]

BAND_HEIGHT = 62   # new white space added at top for Fig. X label

def load_bold_font(size):
    for path in [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\verdanab.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def process(cfg):
    src = cfg["src"]
    if not os.path.exists(src):
        print(f"  [SKIP] source not found: {src}")
        return

    orig = Image.open(src).convert("RGB")
    ow, oh = orig.size

    # New canvas: BAND_HEIGHT taller at top
    new_h = oh + BAND_HEIGHT
    canvas = Image.new("RGB", (ow, new_h), (255, 255, 255))

    # Paste original below the new band
    canvas.paste(orig, (0, BAND_HEIGHT))

    draw = ImageDraw.Draw(canvas)

    # Erase old internal title in the shifted original portion
    old_end = BAND_HEIGHT + cfg["old_title_px"]
    draw.rectangle([0, BAND_HEIGHT, ow, old_end], fill=(255, 255, 255))

    # Choose font size to fill ~65% of BAND_HEIGHT
    font_size = max(14, int(BAND_HEIGHT * 0.60))
    font = load_bold_font(font_size)

    title = cfg["title"]
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Reduce font size if title is too wide
    while tw > ow * 0.96 and font_size > 11:
        font_size -= 1
        font = load_bold_font(font_size)
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

    # Center text in the new BAND
    x = (ow - tw) / 2
    y = (BAND_HEIGHT - th) / 2 - bbox[1]
    draw.text((x, y), title, fill=(0, 0, 0), font=font)

    for dest in cfg["dests"]:
        canvas.save(dest, "PNG")
        print(f"  Saved → {dest}")

for cfg in FIGURES:
    print(f"\n{os.path.basename(cfg['src'])}")
    process(cfg)

print("\nAll done.")
