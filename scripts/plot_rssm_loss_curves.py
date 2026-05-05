"""Plot training loss curves: Ablated RSSM vs Full RSSM.

One row per course, columns for each horizon (W).
Overlays both models on the same axes.

Usage:
    python scripts/plot_rssm_loss_curves.py
"""

import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = "results/temporal_eval"
OUTPUT_DIR = "results/rssm_comparison"
COURSES = ["dsa_hk231", "dsa_hk221", "pf_hk232", "pf_hk222"]
COURSE_LABELS = {
    "dsa_hk231": "DSA 231", "dsa_hk221": "DSA 221",
    "pf_hk232": "PF 232", "pf_hk222": "PF 222",
}
MODELS = ["RSSM", "RSSMFull"]
MODEL_LABELS = {"RSSM": "Ablated RSSM", "RSSMFull": "Full RSSM"}
COLORS = {"RSSM": "#4477aa", "RSSMFull": "#ee6677"}


def load_loss_curves():
    curves = {}
    for course in COURSES:
        for model in MODELS:
            pred_path = os.path.join(RESULTS_DIR, course, f"{model}_predictions.pkl")
            if not os.path.exists(pred_path):
                continue
            with open(pred_path, "rb") as f:
                preds = pickle.load(f)
            for horizon, pred in preds.items():
                if pred.losses and "train" in pred.losses:
                    curves[(course, model, horizon)] = pred.losses["train"]
    return curves


def main():
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    curves = load_loss_curves()

    if not curves:
        print("No loss curves found.")
        return

    # Find all horizons per course
    course_horizons = {}
    for (course, model, horizon) in curves:
        course_horizons.setdefault(course, set()).add(horizon)

    courses_with_data = [c for c in COURSES if c in course_horizons]
    if not courses_with_data:
        print("No courses with loss data for both models.")
        return

    max_horizons = max(len(course_horizons[c]) for c in courses_with_data)
    n_rows = len(courses_with_data)
    n_cols = max_horizons

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )

    for row, course in enumerate(courses_with_data):
        horizons = sorted(course_horizons[course])
        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(horizons):
                ax.set_visible(False)
                continue

            horizon = horizons[col]
            for model in MODELS:
                key = (course, model, horizon)
                if key not in curves:
                    continue
                losses = curves[key]
                ax.plot(
                    losses, color=COLORS[model],
                    label=MODEL_LABELS[model],
                    linewidth=0.8, alpha=0.85,
                )

            ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel(f"{COURSE_LABELS.get(course, course)}\nLoss")
            if row == 0:
                ax.set_title(f"W={horizon}", fontsize=9)
            ax.grid(True, alpha=0.2)

            if row == 0 and col == 0:
                ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "loss_curves_grid.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Also make a zoomed version showing just the last 300 epochs
    fig2, axes2 = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )

    for row, course in enumerate(courses_with_data):
        horizons = sorted(course_horizons[course])
        for col in range(n_cols):
            ax = axes2[row][col]
            if col >= len(horizons):
                ax.set_visible(False)
                continue

            horizon = horizons[col]
            for model in MODELS:
                key = (course, model, horizon)
                if key not in curves:
                    continue
                losses = curves[key]
                start = max(0, len(losses) - 300)
                epochs = list(range(start, len(losses)))
                ax.plot(
                    epochs, losses[start:],
                    color=COLORS[model],
                    label=MODEL_LABELS[model],
                    linewidth=0.8, alpha=0.85,
                )

            ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel(f"{COURSE_LABELS.get(course, course)}\nLoss")
            if row == 0:
                ax.set_title(f"W={horizon}", fontsize=9)
            ax.grid(True, alpha=0.2)

            if row == 0 and col == 0:
                ax.legend(fontsize=6, loc="upper right")

    fig2.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "loss_curves_grid_zoomed.pdf")
    fig2.savefig(path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
