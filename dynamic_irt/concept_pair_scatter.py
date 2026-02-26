"""
Scatter plot of Elo-estimated item difficulties for concept-similar pairs.

Pairs of items that test related concepts should receive similar difficulty
estimates if the model is well-calibrated — i.e. dots should cluster near
the diagonal.

Usage:
    cd CodeInsights && python -m dynamic_irt.concept_pair_scatter
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tueplots import bundles, figsizes

matplotlib.use("Agg")
plt.rcParams.update(bundles.aaai2024())

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Concept-similar item pairs
# Format: (item_id_a, item_id_b, category, short_label)
# ---------------------------------------------------------------------------

PAIRS = [
    # --- Near-identical (unmerged, different names but same problem) ---
    (190, 245, "Near-identical problem", "Bubble sort v2/exam"),
    (164, 292, "Near-identical problem", "Interpolation search"),
    (55, 10, "Near-identical problem", "Khoảng trắng/v2"),
    (16, 39, "Near-identical problem", "Pirate Alliance/Easy"),
    (127, 36, "Near-identical problem", "isSublist & isEqual"),
    (54, 364, "Near-identical problem", "Tìm từ đầu tiên"),
    (22, 68, "Near-identical problem", "Danh sách reaction"),
    (153, 317, "Near-identical problem", "Insert AVL/+complexity"),
    (42, 381, "Near-identical problem", "Palindrome"),

    # --- Same algorithm, different data structure ---
    (139, 140, "Same algorithm, diff DS", "Merge Sort: Array vs LL"),
    (138, 140, "Same algorithm, diff DS", "Quick Sort vs Merge Sort LL"),

    # --- Same DS, different operations ---
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

    # --- Same concept family ---
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

    # --- Different concept (negative control) ---
    (163, 152, "Different concept (control)", "BinarySearch vs AVL Delete"),
    (190, 209, "Different concept (control)", "BubbleSort vs BFS"),
    (156, 139, "Different concept (control)", "Stack Impl vs MergeSort"),
    (132, 153, "Different concept (control)", "Queue Impl vs AVL Insert"),
    (147, 42, "Different concept (control)", "LL Add vs Palindrome"),
    (211, 192, "Different concept (control)", "DFS vs ShellSort"),
]

# Category display order and colors (Paul Tol qualitative)
CATEGORIES = [
    ("Near-identical problem",      "#4477aa"),
    ("Same algorithm, diff DS",     "#228833"),
    ("Same DS, diff operation",     "#ee6677"),
    ("Same concept family",         "#ccbb44"),
    ("Different concept (control)", "#aaaaaa"),
]


def main():
    diff_path = os.path.join(REPO_ROOT, "results", "elo", "elo_difficulty.csv")
    diff_df = pd.read_csv(diff_path)
    diff_map = dict(zip(diff_df["ItemID_SF"], diff_df["average_difficulty"]))

    # Build per-category data
    cat_data = {name: ([], []) for name, _ in CATEGORIES}
    skipped = []
    for id_a, id_b, cat, label in PAIRS:
        ba = diff_map.get(id_a)
        bb = diff_map.get(id_b)
        if ba is None or bb is None:
            skipped.append((id_a, id_b, label))
            continue
        cat_data[cat][0].append(ba)
        cat_data[cat][1].append(bb)

    if skipped:
        print(f"Skipped {len(skipped)} pairs (missing difficulty):")
        for a, b, lbl in skipped:
            print(f"  {a} / {b}: {lbl}")

    # Compute overall Pearson correlation
    all_xs, all_ys = [], []
    for xs, ys in cat_data.values():
        all_xs.extend(xs)
        all_ys.extend(ys)
    r, p = stats.pearsonr(all_xs, all_ys)

    # Axis range — symmetric, same ticks on both axes
    lo = min(all_xs + all_ys) - 0.15
    hi = max(all_xs + all_ys) + 0.15
    lo = np.floor(lo * 2) / 2   # round down to nearest 0.5
    hi = np.ceil(hi * 2) / 2    # round up to nearest 0.5
    ticks = np.arange(lo, hi + 0.01, 0.5)

    # Plot
    fig, ax = plt.subplots(figsize=figsizes.aaai2024_half()["figure.figsize"])

    # Diagonal reference
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.75", lw=0.8, zorder=0)

    for cat_name, color in CATEGORIES:
        xs, ys = cat_data[cat_name]
        if not xs:
            continue
        ax.scatter(
            xs, ys,
            c=color, s=28, alpha=0.85, zorder=2,
            label=f"{cat_name} ({len(xs)})",
            edgecolors="white", linewidths=0.3,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Difficulty $b_i$")
    ax.set_ylabel(r"Difficulty $b_j$")
    ax.set_title(f"Item Difficulty ($r = {r:.2f}$)")
    ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False)

    save_path = os.path.join(REPO_ROOT, "results", "elo", "concept_pair_difficulty_scatter.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Print summary table
    print(f"\n{'Category':<30} {'Pairs':>5} {'RMSE':>7} {'Mean|Δb|':>9}")
    print("-" * 55)
    for cat_name, _ in CATEGORIES:
        xs, ys = cat_data[cat_name]
        if not xs:
            continue
        diffs = np.abs(np.array(xs) - np.array(ys))
        rmse = np.sqrt(np.mean(diffs ** 2))
        print(f"{cat_name:<30} {len(xs):>5} {rmse:>7.3f} {np.mean(diffs):>9.3f}")


if __name__ == "__main__":
    main()
