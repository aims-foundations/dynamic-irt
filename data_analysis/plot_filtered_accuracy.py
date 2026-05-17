"""Plot per-attempt accuracy for models on student-split evaluation.

Each point has a confidence interval from two-way cluster bootstrap
(resampling both students and questions).

Usage:
    python data_analysis/plot_filtered_accuracy.py
    python data_analysis/plot_filtered_accuracy.py --courses dsa_hk231
    python data_analysis/plot_filtered_accuracy.py --mode temporal --courses dsa_hk231
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dynamic_models.temporal_eval.data_filter import DEFAULT_FILTER

matplotlib.use("Agg")

ALL_COURSES = ["dsa_hk231", "dsa_hk221", "pf_hk232", "pf_hk222"]

MODEL_STYLES = {
    "IRT": {"color": "#2080cc", "marker": "o"},
    "CIRT-Decay": {"color": "#d63030", "marker": "s"},
    "DynamicIRT": {"color": "#e68a00", "marker": "p"},
    "BKT": {"color": "#2e8b57", "marker": "D"},
    "DKT": {"color": "#7b2d8e", "marker": "^"},
}


def twoway_cluster_bootstrap_ci(student_ids, item_ids, matches, n_boot=2000, confidence=0.95):
    """Two-way cluster bootstrap: resample both students and questions.

    Uses a sparse matrix approach for speed: build a (student x item) mean
    matrix, then resample rows and columns.
    """
    rng = np.random.default_rng(42)
    alpha = (1 - confidence) / 2

    students = np.unique(student_ids)
    items = np.unique(item_ids)
    s_map = {s: i for i, s in enumerate(students)}
    q_map = {q: i for i, q in enumerate(items)}

    n_s, n_q = len(students), len(items)
    sum_mat = np.zeros((n_s, n_q))
    count_mat = np.zeros((n_s, n_q))
    for s, q, m in zip(student_ids, item_ids, matches):
        si, qi = s_map[s], q_map[q]
        sum_mat[si, qi] += m
        count_mat[si, qi] += 1

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        s_idx = rng.integers(0, n_s, size=n_s)
        q_idx = rng.integers(0, n_q, size=n_q)
        sub_sum = sum_mat[np.ix_(s_idx, q_idx)]
        sub_cnt = count_mat[np.ix_(s_idx, q_idx)]
        total_cnt = sub_cnt.sum()
        if total_cnt > 0:
            boot_means[b] = sub_sum.sum() / total_cnt
        else:
            boot_means[b] = np.nan

    boot_means = boot_means[~np.isnan(boot_means)]
    return np.quantile(boot_means, alpha), np.quantile(boot_means, 1 - alpha)


def load_prediction_result(model_name, output_dir, mode="student"):
    """Load a saved PredictionResult pickle."""
    if mode == "student":
        pkl_path = os.path.join(output_dir, f"{model_name}_student_pred.pkl")
    else:
        cutoff = DEFAULT_FILTER.max_week
        pkl_path = os.path.join(output_dir, f"{model_name}_W{cutoff}_pred.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_per_attempt_accuracy(prediction, max_attempts=10):
    """Compute accuracy and CI at each attempt number."""
    y_true = prediction.y_true
    y_pred = prediction.y_pred_prob
    s_idx = prediction.student_indices
    i_idx = prediction.item_indices

    if prediction.attempt_indices is not None:
        attempt_nums = prediction.attempt_indices
    else:
        attempt_nums = np.zeros(len(y_true), dtype=int)
        prev_pair = None
        attempt = 0
        for i in range(len(y_true)):
            pair = (s_idx[i], i_idx[i])
            if pair != prev_pair:
                attempt = 0
                prev_pair = pair
            attempt_nums[i] = attempt
            attempt += 1

    accs = []
    ci_los = []
    ci_his = []
    counts = []

    for a in range(max_attempts):
        mask = attempt_nums == a
        if mask.sum() < 10:
            accs.append(np.nan)
            ci_los.append(np.nan)
            ci_his.append(np.nan)
            counts.append(int(mask.sum()))
            continue

        pred_binary = (y_pred[mask] >= 0.5).astype(int)
        actual = y_true[mask].astype(int)
        matches = (pred_binary == actual).astype(int)
        acc = matches.mean()
        accs.append(acc)
        counts.append(int(mask.sum()))

        lo, hi = twoway_cluster_bootstrap_ci(
            s_idx[mask], i_idx[mask], matches
        )
        ci_los.append(lo)
        ci_his.append(hi)

    return np.array(accs), np.array(ci_los), np.array(ci_his), np.array(counts)


def plot_course(course, models, max_attempts, output_dir, mode="student"):
    predictions = {}
    missing = []
    for m in models:
        pred = load_prediction_result(m, output_dir, mode)
        if pred is not None:
            predictions[m] = pred
            print(f"  Loaded saved predictions for {m}")
        else:
            missing.append(m)

    if missing:
        print(f"  WARNING: Missing predictions for {missing}. "
              f"Run run_student_eval.py first.")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, max_attempts + 1)

    for model_name in models:
        pred = predictions.get(model_name)
        if pred is None:
            continue

        accs, ci_los, ci_his, counts = compute_per_attempt_accuracy(pred, max_attempts)
        style = MODEL_STYLES.get(model_name, {"color": "gray", "marker": "o"})

        err_lo = accs - ci_los
        err_hi = ci_his - accs
        valid = ~np.isnan(accs)

        ax.errorbar(
            x[valid], accs[valid],
            yerr=[err_lo[valid], err_hi[valid]],
            fmt=f"{style['marker']}-",
            color=style["color"],
            linewidth=2, markersize=7, capsize=4, capthick=1.5,
            label=model_name,
        )

    ax.set_xlabel("Attempt Number")
    ax.set_ylabel("Accuracy")
    split_label = "Student Split" if mode == "student" else "Temporal Split"
    ax.set_title(f"Per-Attempt Accuracy — {split_label} ({course})")
    ax.set_xticks(x)
    ax.set_xlim(0.5, max_attempts + 0.5)
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "accuracy_vs_attempt.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-attempt accuracy")
    parser.add_argument("--courses", nargs="+", default=["dsa_hk231"])
    parser.add_argument("--models", nargs="+", default=["IRT", "CIRT-Decay", "DynamicIRT", "BKT", "DKT"])
    parser.add_argument("--max_attempts", type=int, default=10)
    parser.add_argument("--mode", choices=["student", "temporal"], default="student")
    args = parser.parse_args()

    for course in args.courses:
        print(f"\n{'=' * 60}")
        print(f"Course: {course}")
        print(f"{'=' * 60}")
        if args.mode == "student":
            output_dir = f"results/student_eval/{course}"
        else:
            output_dir = f"results/filtered_eval/{course}"
        plot_course(course, args.models, args.max_attempts, output_dir, args.mode)


if __name__ == "__main__":
    main()
