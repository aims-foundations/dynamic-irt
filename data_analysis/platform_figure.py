"""Annotated platform figure for the paper.

Takes the original platform screenshot and adds a border and a large
legible header over each of the three sections (problem + editor,
failing-attempt feedback, passing attempt). The screenshot pixels are
kept at native resolution; headers are rendered with LaTeX Times to
match the paper font.

Usage:
    python data_analysis/platform_figure.py
"""

import io
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREENSHOT = os.path.join(
    REPO_ROOT, "overleaf", "figures", "codeinsight", "codeinsights_platform.png"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schematic_outputs")

BORDER_COLOR = "#44505c"
HEADER_BAND = 470
TITLES = [
    "Problem statement & code editor",
    "Feedback on failing attempts",
    "Submission passing all test cases",
]


def _panel_segments(img):
    """x-ranges of the three panels, found via the white gaps between them."""
    arr = np.asarray(img.convert("L"))
    white_col = (arr > 240).mean(axis=0) > 0.99

    runs, start = [], None
    for i, w in enumerate(white_col):
        if w and start is None:
            start = i
        elif not w and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(white_col)))
    margin = 100
    gaps = [
        r for r in runs
        if r[1] - r[0] >= 8 and r[0] > margin and r[1] < arr.shape[1] - margin
    ]
    if len(gaps) != 2:
        widths = sorted(runs, key=lambda r: r[0] - r[1])[:10]
        raise RuntimeError(
            f"Expected 2 white gaps between panels, found {len(gaps)}: "
            f"runs={widths}"
        )
    (g1s, g1e), (g2s, g2e) = gaps
    return [(0, g1s), (g1e, g2s), (g2e, arr.shape[1])]


def _load_font(size):
    path = font_manager.findfont(
        font_manager.FontProperties(family="Times New Roman", weight="bold")
    )
    return ImageFont.truetype(path, size)


def _fitted_font_size(titles, segments, max_size=230):
    """Largest common font size at which every title fits its panel."""
    size = max_size
    while size > 60:
        font = _load_font(size)
        if all(
            font.getbbox(t)[2] <= (x1 - x0) - 200
            for t, (x0, x1) in zip(titles, segments)
        ):
            return size
        size -= 10
    return size


def _render_header_band(w_px, band_px, titles, segments, size_px):
    """Render the header titles with LaTeX Times, the font used by the
    paper and all other figures, onto a transparent strip."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{times}",
        }
    )
    dpi = 600
    fig = plt.figure(figsize=(w_px / dpi, band_px / dpi), dpi=dpi)
    for (x0, x1), title in zip(segments, titles):
        fig.text(
            (x0 + x1) / 2 / w_px,
            0.45,
            title.replace("&", r"\&"),
            ha="center",
            va="center",
            fontsize=size_px / dpi * 72,
            fontweight="bold",
            color="black",
        )
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, transparent=True, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def annotate_screenshot(out_path):
    img = Image.open(SCREENSHOT).convert("RGB")
    segments = _panel_segments(img)
    w, h = img.size

    canvas = Image.new("RGB", (w, h + HEADER_BAND), "white")
    canvas.paste(img, (0, HEADER_BAND))
    draw = ImageDraw.Draw(canvas)
    for (x0, x1) in segments:
        draw.rounded_rectangle(
            [x0 + 4, HEADER_BAND + 4, x1 - 4, HEADER_BAND + h - 6],
            radius=36,
            outline=BORDER_COLOR,
            width=12,
        )

    size_px = _fitted_font_size(TITLES, segments)
    band = _render_header_band(w, HEADER_BAND, TITLES, segments, size_px)
    canvas.paste(band, (0, 0), band)

    canvas.save(out_path)
    print(f"Saved {out_path} ({canvas.size[0]}x{canvas.size[1]})")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    annotate_screenshot(
        os.path.join(OUT_DIR, "codeinsights_platform_annotated.png")
    )


if __name__ == "__main__":
    main()
