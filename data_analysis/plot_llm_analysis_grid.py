"""Kendall tau decomposition grid by model.

Usage:
    python data_analysis/plot_llm_analysis_grid.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

matplotlib.use("Agg")

BASE_DIR = "results/llm_student_eval/dsa_hk231"

MODELS = [
    ("Opus 4.6", "analysis_opus"),
    ("Qwen3-14B", "analysis_qwen"),
    ("Gemma-4-31B", "analysis_gemma"),
]


def main():
    paths = [os.path.join(BASE_DIR, subdir, "decomposition_test.png")
             for _, subdir in MODELS]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"ERROR: missing input images for grid: {missing}")
    imgs = [mpimg.imread(p) for p in paths]

    # Crop whitespace from each image
    cropped = []
    for img in imgs:
        gray = np.mean(img[:, :, :3], axis=2) if img.ndim == 3 else img
        mask = gray < 0.99
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped.append(img[rmin:rmax+1, cmin:cmax+1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                              gridspec_kw={"width_ratios": [c.shape[1] for c in cropped],
                                           "wspace": 0.01})

    for col, (name, _) in enumerate(MODELS):
        axes[col].imshow(cropped[col])
        axes[col].set_title(name, fontsize=12, fontweight="normal")
        axes[col].axis("off")
    out_path = os.path.join(BASE_DIR, "llm_analysis_grid.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")

    overleaf_path = os.path.join("overleaf", "figures", "llm_analysis_grid.png")
    shutil.copy2(out_path, overleaf_path)
    print(f"Copied to: {overleaf_path}")


if __name__ == "__main__":
    main()
