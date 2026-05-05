"""Compare ablated RSSM vs full RSSM across all courses and horizons.

Generates:
    results/rssm_comparison/comparison_table.csv
    results/rssm_comparison/auc_by_horizon.pdf
    results/rssm_comparison/metrics_heatmap.pdf
    results/rssm_comparison/loss_curves.pdf

Usage:
    python scripts/compare_rssm_models.py
"""

import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = "results/temporal_eval"
OUTPUT_DIR = "results/rssm_comparison"
COURSES = ["dsa_hk231", "dsa_hk221", "pf_hk232", "pf_hk222"]
MODELS = ["RSSM", "RSSMFull"]
MODEL_LABELS = {"RSSM": "Ablated RSSM", "RSSMFull": "Full RSSM"}
COLORS = {"RSSM": "#4477aa", "RSSMFull": "#ee6677"}


def load_all_results():
    rows = []
    for course in COURSES:
        for model in MODELS:
            path = os.path.join(RESULTS_DIR, course, f"{model}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df["course"] = course
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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


def make_comparison_table(df):
    auc = df[df["metric"] == "auc"].copy()
    auc["W"] = auc["horizon"].astype(int)
    pivot = auc.pivot_table(index=["course", "W"], columns="model", values="value")
    if "RSSM" in pivot.columns and "RSSMFull" in pivot.columns:
        pivot["delta"] = pivot["RSSMFull"] - pivot["RSSM"]
        pivot["winner"] = pivot.apply(
            lambda r: "Full" if r["delta"] > 0.005
            else ("Ablated" if r["delta"] < -0.005 else "Tie"),
            axis=1,
        )
    return pivot


def plot_auc_by_horizon(df):
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    auc = df[df["metric"] == "auc"].copy()
    courses_present = [c for c in COURSES if c in auc["course"].values]
    n = len(courses_present)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, course in zip(axes, courses_present):
        for model in MODELS:
            subset = auc[(auc["course"] == course) & (auc["model"] == model)]
            subset = subset.sort_values("horizon")
            if len(subset) == 0:
                continue
            ax.plot(
                subset["horizon"], subset["value"],
                "o-", color=COLORS[model], label=MODEL_LABELS[model],
                linewidth=1.5, markersize=5,
            )
        ax.set_xlabel("Train cutoff week (W)")
        if ax == axes[0]:
            ax.set_ylabel("AUC")
        course_label = course.replace("_", " ").upper()
        ax.set_title(course_label, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "auc_by_horizon.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_metrics_heatmap(df):
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    metrics_to_show = ["auc", "accuracy", "f1", "log_likelihood", "rmse"]
    auc_df = df[df["metric"].isin(metrics_to_show)].copy()

    # Average across horizons per course-model
    avg = auc_df.groupby(["course", "model", "metric"])["value"].mean().reset_index()
    avg = avg.dropna(subset=["value"])

    # Compute delta (Full - Ablated) per course-metric
    pivot = avg.pivot_table(index=["course", "metric"], columns="model", values="value")
    if "RSSM" not in pivot.columns or "RSSMFull" not in pivot.columns:
        return
    pivot["delta"] = pivot["RSSMFull"] - pivot["RSSM"]
    delta_pivot = pivot["delta"].reset_index().pivot(index="course", columns="metric", values="delta")

    # For RMSE and LL, negative delta means Full is better (lower is better for RMSE)
    # Actually for LL, higher is better; for RMSE, lower is better
    # Let's just show raw deltas with annotation

    fig, ax = plt.subplots(figsize=(6, 3))
    import seaborn as sns

    # Reorder
    metric_order = [m for m in metrics_to_show if m in delta_pivot.columns]
    course_order = [c for c in COURSES if c in delta_pivot.index]
    data = delta_pivot.loc[course_order, metric_order]

    sns.heatmap(
        data, annot=True, fmt=".3f", center=0,
        cmap="RdBu", ax=ax, linewidths=0.5,
        cbar_kws={"label": "Full - Ablated"},
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Metric differences (Full RSSM - Ablated RSSM)\nPositive = Full is higher")

    path = os.path.join(OUTPUT_DIR, "metrics_heatmap.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_curves(curves):
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    # Pick W=1 for each course to compare training dynamics
    courses_with_data = []
    for course in COURSES:
        if (course, "RSSM", 1) in curves and (course, "RSSMFull", 1) in curves:
            courses_with_data.append(course)

    if not courses_with_data:
        return

    n = len(courses_with_data)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, course in zip(axes, courses_with_data):
        for model in MODELS:
            key = (course, model, 1)
            if key not in curves:
                continue
            losses = curves[key]
            ax.plot(
                losses, color=COLORS[model],
                label=MODEL_LABELS[model], linewidth=1, alpha=0.8,
            )
        ax.set_xlabel("Epoch")
        if ax == axes[0]:
            ax.set_ylabel("Training loss")
        course_label = course.replace("_", " ").upper()
        ax.set_title(f"{course_label} (W=1)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "loss_curves.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(table):
    print("\n" + "=" * 70)
    print("RSSM COMPARISON: Ablated vs Full (AUC)")
    print("=" * 70)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    if "winner" in table.columns:
        wins = table["winner"].value_counts()
        print(f"\nWins:  Full={wins.get('Full', 0)}  "
              f"Ablated={wins.get('Ablated', 0)}  "
              f"Tie={wins.get('Tie', 0)}")

        valid = table["delta"].dropna()
        print(f"Mean delta (Full - Ablated): {valid.mean():.4f}")
        print(f"Median delta: {valid.median():.4f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_all_results()
    if len(df) == 0:
        print("No results found.")
        return

    table = make_comparison_table(df)
    print_summary(table)

    csv_path = os.path.join(OUTPUT_DIR, "comparison_table.csv")
    table.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    plot_auc_by_horizon(df)
    plot_metrics_heatmap(df)

    curves = load_loss_curves()
    plot_loss_curves(curves)

    print(f"\nAll outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
