"""Visualization of temporal evaluation results."""

import os
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .base_adapter import PredictionResult
from .data_loader import UnifiedData

matplotlib.use("Agg")

COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44", "#66ccee"]
METRIC_LABELS = {
    "auc": "AUC",
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "log_likelihood": "Log-Likelihood",
    "rmse": "RMSE",
}


def plot_metrics_vs_horizon(results_df: pd.DataFrame, output_dir: str):
    """Line plots: metric (y) vs cutoff week (x), one line per model.

    If results contain multiple courses, plots mean +/- std error bands.
    """
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    multi_course = "course" in results_df.columns and results_df["course"].nunique() > 1

    metrics = [m for m in ["log_likelihood"]
               if m in results_df["metric"].values]

    if not metrics:
        print("No metrics to plot.")
        return

    models = sorted(results_df["model"].unique())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 3.5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(models):
            subset = results_df[
                (results_df["model"] == model) & (results_df["metric"] == metric)
            ]

            if len(subset) == 0:
                continue

            color = COLORS[i % len(COLORS)]

            if multi_course:
                grouped = subset.groupby("horizon")["value"]
                means = grouped.mean().sort_index()
                sems = grouped.sem().sort_index()
                ax.plot(means.index, means.values, "o-", color=color,
                        label=model, linewidth=1.5, markersize=4)
            else:
                subset = subset.sort_values("horizon")
                ax.plot(subset["horizon"], subset["value"], "o-", color=color,
                        label=model, linewidth=1.5, markersize=4)

        ax.set_xlabel("Train Cutoff Week")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "temporal_eval_metrics.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


def plot_summary_table(results_df: pd.DataFrame, output_dir: str):
    """Save a pivot table as CSV for easy inspection."""
    if len(results_df) == 0:
        return

    pivot = results_df.pivot_table(
        index=["model", "horizon"],
        columns="metric",
        values="value",
    ).reset_index()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "temporal_eval_summary.csv")
    pivot.to_csv(save_path, index=False)
    print(f"Summary table saved: {save_path}")


# ---------------------------------------------------------------------------
# Student trajectory plots (ported from compare_growth.py)
# ---------------------------------------------------------------------------

def _aggregate_by_question(y_values, student_indices, item_indices, student_id):
    """Aggregate test-case-level values to question-level means for one student.

    Returns arrays of (question_indices_sorted, mean_values) ordered by question index.
    """
    mask = student_indices == student_id
    items = item_indices[mask]
    values = y_values[mask]
    unique_items = np.unique(items)
    q_means = np.array([values[items == q].mean() for q in unique_items])
    order = np.argsort(unique_items)
    return unique_items[order], q_means[order]


def plot_student_trajectories(
    predictions: Dict[Tuple[str, int], PredictionResult],
    data: UnifiedData,
    output_dir: str,
    n_students: int = 5,
):
    """Plot predicted vs actual trajectories per student, aggregated to question level.

    Uses the last (largest) horizon. For each model, groups predictions by
    (student, question) and plots the mean P(correct) per question.
    """
    try:
        from tueplots import bundles, figsizes
        plt.rcParams.update(bundles.neurips2024())
        fig_width = figsizes.neurips2024()["figure.figsize"][0]
    except ImportError:
        fig_width = 6.0

    if not predictions:
        print("No predictions available for trajectory plots.")
        return

    max_horizon = max(h for _, h in predictions.keys())
    model_preds = {}
    for (model, horizon), pred in predictions.items():
        if horizon == max_horizon and pred.student_indices is not None and pred.item_indices is not None:
            model_preds[model] = pred

    if not model_preds:
        print("No models with student/item indices at last horizon — skipping trajectories.")
        return

    ref_model = max(model_preds, key=lambda m: len(model_preds[m].y_true))
    ref_pred = model_preds[ref_model]

    # Count unique questions per student (not raw test cases)
    unique_students = np.unique(ref_pred.student_indices)
    student_question_counts = {}
    for s in unique_students:
        n_questions = len(np.unique(ref_pred.item_indices[ref_pred.student_indices == s]))
        student_question_counts[s] = n_questions

    # Pick students with a good spread of question counts (not just the max)
    sorted_students = sorted(student_question_counts, key=student_question_counts.get, reverse=True)
    # Take from different quartiles for diversity
    n_available = len(sorted_students)
    if n_available <= n_students:
        selected = sorted_students
    else:
        indices = np.linspace(0, n_available - 1, n_students, dtype=int)
        selected = [sorted_students[i] for i in indices]

    if not selected:
        print("No students found for trajectory plot.")
        return

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, 2.2 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for ax, sidx in zip(axes, selected):
        # Aggregate actual outcomes to question level
        q_ids, q_actual = _aggregate_by_question(
            ref_pred.y_true.astype(np.float32),
            ref_pred.student_indices, ref_pred.item_indices, sidx
        )
        x_axis = np.arange(len(q_ids))

        if len(q_actual) < 3:
            continue

        # Actual question-level scores as dots
        ax.scatter(x_axis, q_actual, color="black", alpha=0.3, s=10,
                   zorder=1, label="Actual (per question)")

        # Plot each model's question-level predictions
        for i, (model_name, pred) in enumerate(sorted(model_preds.items())):
            if pred.item_indices is None:
                continue
            mq_ids, mq_preds = _aggregate_by_question(
                pred.y_pred_prob, pred.student_indices, pred.item_indices, sidx
            )
            # Align to reference question ordering
            mx = np.arange(len(mq_ids))
            color = COLORS[i % len(COLORS)]
            ax.plot(mx, mq_preds, color=color, linewidth=1.2, alpha=0.8,
                    label=model_name, zorder=3)

        if sidx < len(data.student_ids):
            student_label = data.student_ids[sidx]
        else:
            student_label = sidx
        n_q = len(q_ids)
        ax.set_ylabel(r"$P(\mathrm{correct})$", fontsize=7)
        ax.set_title(f"Student {student_label} ({n_q} questions)", fontsize=7, pad=2)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=5, loc="lower right", ncol=2)
    axes[-1].set_xlabel("Question index (ordered)")
    fig.tight_layout(h_pad=0.8)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "student_trajectories.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    predictions: Dict[Tuple[str, int], PredictionResult],
    output_dir: str,
):
    """Plot training loss curves for each model (using the last horizon)."""
    try:
        from tueplots import bundles, figsizes
        plt.rcParams.update(bundles.neurips2024())
        fig_width = figsizes.neurips2024()["figure.figsize"][0]
    except ImportError:
        fig_width = 6.0

    max_horizon = max(h for _, h in predictions.keys())
    models_with_losses = {}
    for (model, horizon), pred in predictions.items():
        if horizon == max_horizon and pred.losses:
            models_with_losses[model] = pred.losses

    if not models_with_losses:
        return

    fig, ax = plt.subplots(figsize=(fig_width, 3.5))
    for i, (model, losses) in enumerate(sorted(models_with_losses.items())):
        color = COLORS[i % len(COLORS)]
        train = losses.get("train", [])
        if train:
            ax.plot(train, color=color, label=model, linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "loss_curves.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Parameter distribution histograms
# ---------------------------------------------------------------------------

def plot_param_distributions(
    predictions: Dict[Tuple[str, int], PredictionResult],
    output_dir: str,
):
    """Plot histogram + KDE for each learned parameter, one figure per model."""
    try:
        from tueplots import bundles, figsizes
        plt.rcParams.update(bundles.neurips2024())
        half_width = figsizes.neurips2024()["figure.figsize"][0] / 2
    except ImportError:
        half_width = 3.0

    max_horizon = max(h for _, h in predictions.keys())
    os.makedirs(output_dir, exist_ok=True)

    for (model, horizon), pred in sorted(predictions.items()):
        if horizon != max_horizon:
            continue

        all_params = {}
        if pred.student_params:
            all_params.update(pred.student_params)
        if pred.item_params:
            all_params.update(pred.item_params)

        if not all_params:
            continue

        n_params = len(all_params)
        fig, axes = plt.subplots(1, n_params, figsize=(half_width * n_params, 3.0))
        if n_params == 1:
            axes = [axes]

        for ax, (param_name, values) in zip(axes, all_params.items()):
            lo, hi = np.percentile(values, (1, 99))
            clipped = values[(values >= lo) & (values <= hi)]
            ax.hist(clipped, bins=30, density=True, alpha=0.3, color=COLORS[0])
            sns.kdeplot(clipped, color=COLORS[0], linewidth=1.5, bw_adjust=0.5, ax=ax)
            ax.set_xlabel(param_name, fontsize=7)
            ax.set_ylabel("Density", fontsize=7)
            ax.tick_params(labelsize=6)

        fig.suptitle(f"{model} Parameter Distributions", fontsize=9)
        fig.tight_layout()
        safe_name = model.lower().replace(" ", "_")
        save_path = os.path.join(output_dir, f"{safe_name}_params.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Concept pair scatter (ported from concept_pair_scatter.py)
# ---------------------------------------------------------------------------

# Hardcoded item pairs for DSA course (question_unittest_id pairs)
CONCEPT_PAIRS = [
    # Near-identical (same problem, different names)
    (190, 245, "Near-identical problem", "Bubble sort v2/exam"),
    (164, 292, "Near-identical problem", "Interpolation search"),
    (55, 10, "Near-identical problem", "Khoảng trắng/v2"),
    (16, 39, "Near-identical problem", "Pirate Alliance/Easy"),
    (127, 36, "Near-identical problem", "isSublist & isEqual"),
    (54, 364, "Near-identical problem", "Tìm từ đầu tiên"),
    (22, 68, "Near-identical problem", "Danh sách reaction"),
    (153, 317, "Near-identical problem", "Insert AVL/+complexity"),
    (42, 381, "Near-identical problem", "Palindrome"),
    # Same algorithm, different data structure
    (139, 140, "Same algorithm, diff DS", "Merge Sort: Array vs LL"),
    (138, 140, "Same algorithm, diff DS", "Quick Sort vs Merge Sort LL"),
    # Same DS, different operations
    (147, 148, "Same DS, diff operation", "LL: Add vs Get"),
    (148, 149, "Same DS, diff operation", "LL: Get vs Remove"),
    (147, 149, "Same DS, diff operation", "LL: Add vs Remove"),
    (150, 151, "Same DS, diff operation", "LL: Reverse vs Rotate"),
    (193, 194, "Same DS, diff operation", "DLL: Add vs Get"),
    (194, 196, "Same DS, diff operation", "DLL: Get vs Remove"),
    (152, 153, "Same DS, diff operation", "AVL: Delete vs Insert"),
    (153, 154, "Same DS, diff operation", "AVL: Insert vs Search"),
    (249, 250, "Same DS, diff operation", "BST: kth-smallest vs RangeCount"),
    (250, 251, "Same DS, diff operation", "BST: RangeCount vs SubtreeRange"),
    (169, 171, "Same DS, diff operation", "Heap: Peek vs Push"),
    (171, 173, "Same DS, diff operation", "Heap: Push vs Reheap"),
    # Same concept family
    (190, 191, "Same concept family", "Bubble vs Selection Sort"),
    (191, 192, "Same concept family", "Selection vs Shell Sort"),
    (138, 139, "Same concept family", "Quick vs Merge Sort (Arr)"),
    (141, 139, "Same concept family", "Tim vs Merge Sort (Arr)"),
    (163, 164, "Same concept family", "Binary vs Interpolation"),
    (163, 165, "Same concept family", "Binary vs Jump"),
    (164, 165, "Same concept family", "Interpolation vs Jump"),
    (209, 211, "Same concept family", "BFS vs DFS"),
    (210, 214, "Same concept family", "Connected vs isCyclic"),
    (156, 162, "Same concept family", "Stack: Impl vs ValidParens"),
    (157, 161, "Same concept family", "Stack: NextGreater vs StockSpan"),
    # Different concept (negative control)
    (163, 152, "Different concept (control)", "BinarySearch vs AVL Delete"),
    (190, 209, "Different concept (control)", "BubbleSort vs BFS"),
    (156, 139, "Different concept (control)", "Stack Impl vs MergeSort"),
    (132, 153, "Different concept (control)", "Queue Impl vs AVL Insert"),
    (147, 42, "Different concept (control)", "LL Add vs Palindrome"),
    (211, 192, "Different concept (control)", "DFS vs ShellSort"),
]

CONCEPT_CATEGORIES = [
    ("Near-identical problem", "#4477aa"),
    ("Same algorithm, diff DS", "#228833"),
    ("Same DS, diff operation", "#ee6677"),
    ("Same concept family", "#ccbb44"),
    ("Different concept (control)", "#aaaaaa"),
]


def plot_concept_pair_scatter(
    predictions: Dict[Tuple[str, int], PredictionResult],
    output_dir: str,
    model_name: str = None,
):
    """Scatter plot of estimated item difficulties for concept-similar pairs.

    Computes per-item difficulty as 1 - mean(y_pred_prob) from a model's
    predictions. Items that test related concepts should have similar difficulty
    (dots near the diagonal).

    Args:
        predictions: Dict mapping (model_name, horizon) to PredictionResult.
        output_dir: Where to save the plot.
        model_name: Which model to use. If None, uses the first model at the
            last horizon that has item_indices.
    """
    try:
        from tueplots import bundles, figsizes
        plt.rcParams.update(bundles.neurips2024())
        fig_size = figsizes.neurips2024()["figure.figsize"]
    except ImportError:
        fig_size = (3.5, 3.5)

    if not predictions:
        print("No predictions available for concept pair scatter.")
        return

    # Find prediction to use
    max_horizon = max(h for _, h in predictions.keys())
    pred = None
    used_model = None
    for (m, h), p in predictions.items():
        if h == max_horizon and p.item_indices is not None:
            if model_name is None or m == model_name:
                pred = p
                used_model = m
                break

    if pred is None:
        print("No model with item_indices at last horizon — skipping concept scatter.")
        return

    # Compute per-item difficulty: 1 - mean(y_pred_prob)
    item_difficulty = {}
    unique_items = np.unique(pred.item_indices)
    for item_id in unique_items:
        mask = pred.item_indices == item_id
        item_difficulty[item_id] = 1.0 - np.mean(pred.y_pred_prob[mask])

    # Build per-category data
    cat_data = {name: ([], []) for name, _ in CONCEPT_CATEGORIES}
    skipped = []
    for id_a, id_b, cat, label in CONCEPT_PAIRS:
        ba = item_difficulty.get(id_a)
        bb = item_difficulty.get(id_b)
        if ba is None or bb is None:
            skipped.append((id_a, id_b, label))
            continue
        cat_data[cat][0].append(ba)
        cat_data[cat][1].append(bb)

    if skipped:
        print(f"  Concept scatter: skipped {len(skipped)} pairs (not in test set)")

    # Compute overall Pearson correlation
    all_xs, all_ys = [], []
    for xs, ys in cat_data.values():
        all_xs.extend(xs)
        all_ys.extend(ys)

    if len(all_xs) < 3:
        print("  Too few concept pairs in test set — skipping scatter.")
        return

    r, _ = stats.pearsonr(all_xs, all_ys)

    # Axis range
    lo = min(all_xs + all_ys) - 0.15
    hi = max(all_xs + all_ys) + 0.15
    lo = np.floor(lo * 2) / 2
    hi = np.ceil(hi * 2) / 2
    ticks = np.arange(lo, hi + 0.01, 0.5)

    # Plot
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[0]))
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.75", lw=0.8, zorder=0)

    for cat_name, color in CONCEPT_CATEGORIES:
        xs, ys = cat_data[cat_name]
        if not xs:
            continue
        ax.scatter(xs, ys, c=color, s=28, alpha=0.85, zorder=2,
                   label=f"{cat_name} ({len(xs)})",
                   edgecolors="white", linewidths=0.3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Difficulty $b_i$")
    ax.set_ylabel(r"Difficulty $b_j$")
    ax.set_title(f"{used_model} Item Difficulty ($r = {r:.2f}$)")
    ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "concept_pair_scatter.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")

    # Print summary table
    print(f"\n  {'Category':<30} {'Pairs':>5} {'RMSE':>7} {'Mean|Δb|':>9}")
    print(f"  {'-' * 55}")
    for cat_name, _ in CONCEPT_CATEGORIES:
        xs, ys = cat_data[cat_name]
        if not xs:
            continue
        diffs = np.abs(np.array(xs) - np.array(ys))
        rmse = np.sqrt(np.mean(diffs ** 2))
        print(f"  {cat_name:<30} {len(xs):>5} {rmse:>7.3f} {np.mean(diffs):>9.3f}")
