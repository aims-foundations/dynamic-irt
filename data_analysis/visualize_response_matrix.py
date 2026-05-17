"""Visualize the raw correctness matrix as a heatmap.

Rows = students (sorted by pass rate), columns = unit tests (grouped by question).
Colors: blue=pass, red=fail, white=no data.

Usage:
    python data_analysis/visualize_response_matrix.py --course dsa_hk231
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

from dynamic_models.temporal_eval.data_filter import DEFAULT_FILTER, DataFilterConfig, apply_filter
from dynamic_models.temporal_eval.data_loader import load_student_split_data

matplotlib.use("Agg")


def load_data(course):
    cache_dir = os.path.join(
        os.path.dirname(__file__), "..", ".cache", "matrices", course
    )
    cache_dir = os.path.abspath(cache_dir)

    corr = torch.load(f"{cache_dir}/correctness_matrix.pt", map_location="cpu")
    qi = pd.read_csv(f"{cache_dir}/question_infos.csv")
    si = pd.read_csv(f"{cache_dir}/student_info.csv")
    return corr, qi, si


def collapse_last_attempt(corr, max_attempt=None):
    """Take last valid value per (student, item), optionally capped at max_attempt.

    If max_attempt is set, only considers the first max_attempt valid observations.
    """
    n_s, n_i, n_a = corr.shape
    result = torch.full((n_s, n_i), -1, dtype=corr.dtype)
    if max_attempt is not None:
        count = torch.zeros((n_s, n_i), dtype=torch.long)
        for a in range(n_a):
            valid = corr[:, :, a] != -1
            under_cap = count < max_attempt
            use = valid & under_cap
            result[use] = corr[:, :, a][use]
            count[valid] += 1
    else:
        for a in range(n_a):
            valid = corr[:, :, a] != -1
            result[valid] = corr[:, :, a][valid]
    return result.numpy().astype(float)



def _add_question_separators(ax, qi, col_indices, selected_qs):
    """Add vertical separators between question groups."""
    qi_sub = qi.iloc[col_indices].reset_index(drop=True)
    pos = 0
    for qidx in selected_qs:
        n_tc = (qi_sub["qidx"] == qidx).sum()
        if n_tc == 0:
            continue
        if pos > 0:
            ax.axvline(pos - 0.5, color="gray", linewidth=0.5, alpha=0.5)
        pos += n_tc
    ax.set_xticks([])


def get_attempt_snapshots(corr, max_attempt=5):
    """Get results at each attempt number (0-indexed), carrying forward last known value.

    If a student has fewer than max_attempt attempts on an item,
    later snapshots repeat their last valid result.
    """
    n_s, n_i, n_a = corr.shape
    arr = corr.numpy()
    valid_mask = arr != -1
    cum_valid = np.cumsum(valid_mask, axis=2)

    prev = np.full((n_s, n_i), -1.0)
    snapshots = []
    for a in range(max_attempt):
        result = prev.copy()
        target = a + 1
        hits = cum_valid == target
        has_hit = hits.any(axis=2)
        first_idx = np.argmax(hits, axis=2)
        s_idx, i_idx = np.where(has_hit)
        result[s_idx, i_idx] = arr[s_idx, i_idx, first_idx[s_idx, i_idx]]
        snapshots.append(result)
        prev = result
    return snapshots


def plot_avg_improvement(corr, row_indices, col_indices, output_path):
    """Line plot of average score across attempts: filtered vs all data."""
    snapshots = get_attempt_snapshots(corr, max_attempt=10)
    attempts = list(range(1, 11))

    all_students = np.arange(corr.shape[0])
    all_items = np.arange(corr.shape[1])

    filtered_scores = []
    all_scores = []
    for snap in snapshots:
        sub_f = snap[np.ix_(row_indices, col_indices)]
        valid_f = sub_f[sub_f != -1]
        filtered_scores.append(valid_f.mean() if len(valid_f) > 0 else 0.0)

        sub_a = snap[np.ix_(all_students, all_items)]
        valid_a = sub_a[sub_a != -1]
        all_scores.append(valid_a.mean() if len(valid_a) > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(attempts, filtered_scores, marker="o", linewidth=2, color="#1f77b4",
            label=f"Filtered ({len(row_indices)} students, {len(col_indices)} items)")
    ax.plot(attempts, all_scores, marker="s", linewidth=2, color="#d62728", linestyle="--",
            label=f"All data ({corr.shape[0]} students, {corr.shape[1]} items)")

    ax.set_xlabel("Attempt Number")
    ax.set_ylabel("Average Score (pass rate)")
    ax.set_title("Average Score by Attempt")
    ax.set_xticks(attempts)
    all_vals = filtered_scores + all_scores
    y_min = min(all_vals) - 0.05
    y_max = max(all_vals) + 0.05
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend()

    for i, (sf, sa) in enumerate(zip(filtered_scores, all_scores)):
        ax.annotate(f"{sf:.3f}", (attempts[i], sf), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7, color="#1f77b4")
        ax.annotate(f"{sa:.3f}", (attempts[i], sa), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=7, color="#d62728")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def plot_attempt_progression_compact(corr, matrix, qi, si, row_indices, col_indices, selected_qs, output_path):
    """All attempts 1-10 in a single image with shared axis labels."""
    attempt_nums = list(range(1, 11))

    sub_last = matrix[np.ix_(row_indices, col_indices)]
    pass_rates = np.array([
        np.mean(sub_last[i][sub_last[i] != -1]) if np.any(sub_last[i] != -1) else 0.5
        for i in range(len(sub_last))
    ])
    row_order = np.argsort(pass_rates)[::-1]

    snapshots = get_attempt_snapshots(corr, max_attempt=10)

    cmap = ListedColormap(["#d63030", "#b8b8b8", "#2080cc"])
    bounds = [-0.1, 0.25, 0.75, 1.1]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    n_rows, n_cols = sub_last.shape
    n_panel_rows, n_panel_cols = 2, 5
    panel_w = max(4, min(n_cols * 0.02, 12))
    panel_h = max(5, min(n_rows * 0.02, 30))
    fig, axes = plt.subplots(n_panel_rows, n_panel_cols, figsize=(panel_w * n_panel_cols, panel_h * n_panel_rows), sharey=True, sharex=True)

    for idx, a in enumerate(attempt_nums):
        r, c = divmod(idx, n_panel_cols)
        ax = axes[r, c]
        sub = snapshots[a - 1][np.ix_(row_indices, col_indices)][row_order]
        plot_data = sub.copy()
        plot_data[plot_data == -1] = 0.5
        ax.imshow(plot_data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        _add_question_separators(ax, qi, col_indices, selected_qs)
        ax.set_title(f"Attempt {a}", fontsize=10)

    fig.supxlabel("Unit tests (chronological)", fontsize=12)
    fig.supylabel("Students", fontsize=12)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#2080cc", label="Pass"),
        Patch(facecolor="#d63030", label="Fail"),
        Patch(facecolor="#b8b8b8", label="No data"),
    ], loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0.02, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize raw response matrix")
    parser.add_argument("--course", type=str, default="dsa_hk231")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["temporal", "student"], default="student")
    args = parser.parse_args()

    output_dir = args.output_dir or f"results/student_eval/{args.course}/figures"

    corr, qi, si = load_data(args.course)
    print(f"Loaded: {corr.shape[0]} students, {corr.shape[1]} items, {corr.shape[2]} max attempts")

    if args.mode == "student":
        data, split = load_student_split_data(args.course)

        # Test students, weeks 1-3 items
        row_indices = split.test_student_indices
        col_indices = split.train_item_indices
        qi = data.question_infos
        corr = data.correctness_matrix
        selected_qs = sorted(set(qi.iloc[col_indices]["qidx"].values))

        print(f"Test students: {len(row_indices)}, weeks 1-3 items: {len(col_indices)}")
    else:
        row_indices, col_indices, selected_qs = apply_filter(corr, qi)

    matrix = collapse_last_attempt(corr, max_attempt=DEFAULT_FILTER.max_attempts)

    compact_path = os.path.join(output_dir, "attempt_progression_compact.png")
    plot_attempt_progression_compact(corr, matrix, qi, si, row_indices, col_indices, selected_qs, compact_path)

    improvement_path = os.path.join(output_dir, "avg_improvement_by_attempt.png")
    plot_avg_improvement(corr, row_indices, col_indices, improvement_path)


if __name__ == "__main__":
    main()
