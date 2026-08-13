"""
Fix figure titles for RESS submission.
- Fig1/Fig2: replace wrong "Fig 6" / "Fig 10" labels with correct ones
- Fig3-Fig8: white out internal H1/H2/H3/H4 titles and add "Fig. X —" suptitle style
Updates both Manuscript/Figures/ and Manuscript/Submission/RESS/
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image, ImageDraw, ImageFont
import os

BASE = r"C:\BMAD_PY313"
FIGS_DIR = os.path.join(BASE, "Manuscript", "Figures")
RESS_DIR = os.path.join(BASE, "Manuscript", "Submission", "RESS")

# (src_filename, ress_filename, new_title)
FIGURES = [
    (
        "Fig1_degradation_trends.png", "Fig1.png",
        "Fig. 1 — Top-6 Sensor Degradation Trends (FD003, 10 engines + fleet mean)"
    ),
    (
        "Fig2_rul_clipping.png", "Fig2.png",
        "Fig. 2 — Effect of RUL Clipping Strategy (FD001)"
    ),
    (
        "Fig3_H1_clipping_rmse.png", "Fig3.png",
        "Fig. 3 — RMSE by Dataset and Clipping Threshold (H1: LinearRegression)"
    ),
    (
        "Fig4_H2_normalization_rmse.png", "Fig4.png",
        "Fig. 4 — RMSE by Normalizer × Dataset (H2: mean over 5 seeds)"
    ),
    (
        "Fig5_H2_statistical_test.png", "Fig5.png",
        "Fig. 5 — BH-FDR Adjusted p-values vs N1 Baseline (H2)"
    ),
    (
        "Fig6_H3_gmm_clustering.png", "Fig6.png",
        "Fig. 6 — Clustering Silhouette by Feature Variant (H3, GMM k=2)"
    ),
    (
        "Fig7_H3_model_comparison.png", "Fig7.png",
        "Fig. 7 — Multi-Branch LSTM RMSE by Model (H3)"
    ),
    (
        "Fig8_H4_clip_loss_interaction.png", "Fig8.png",
        "Fig. 8 — Clip × Loss Interaction Effect on NASA Score (H4)"
    ),
]

def load_font(size):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\Arial Bold.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\verdanab.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def process_figure(src_path, dest_paths, new_title):
    img = Image.open(src_path).convert("RGBA")
    width, height = img.size

    # Title band height: match matplotlib suptitle look (~6% of height, min 48px)
    band_h = max(48, int(height * 0.065))

    # Create white band
    overlay = Image.new("RGBA", (width, band_h), (255, 255, 255, 255))
    img.paste(overlay, (0, 0))

    # Choose font size to fill ~65% of band height
    font_size = max(14, int(band_h * 0.62))
    font = load_font(font_size)

    draw = ImageDraw.Draw(img)

    # Measure and center the text
    bbox = draw.textbbox((0, 0), new_title, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # If title is too wide, reduce font size iteratively
    while text_w > width * 0.96 and font_size > 10:
        font_size -= 1
        font = load_font(font_size)
        bbox = draw.textbbox((0, 0), new_title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    x = (width - text_w) / 2
    y = (band_h - text_h) / 2 - bbox[1]  # compensate for ascender offset

    draw.text((x, y), new_title, fill=(0, 0, 0, 255), font=font)

    # Convert back to RGB for PNG save
    result = img.convert("RGB")

    for dest in dest_paths:
        result.save(dest, "PNG")
        print(f"  Saved: {dest}")

for src_name, ress_name, title in FIGURES:
    src = os.path.join(FIGS_DIR, src_name)
    dests = [
        src,  # update in-place
        os.path.join(RESS_DIR, ress_name),
    ]
    print(f"\nProcessing {src_name} ...")
    print(f"  Title: {title}")
    process_figure(src, dests, title)

print("\nDone.")
