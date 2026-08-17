"""Plot per-attempt balanced accuracy for models on student-split evaluation.

Each point has a confidence interval from an i.i.d. bootstrap over
observations at that attempt.

Usage:
    python data_analysis/plot_filtered_accuracy.py
    python data_analysis/plot_filtered_accuracy.py --courses dsa_hk231
"""

import argparse
import os
import pickle
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from data_analysis.llm_eval_common import (
    COURSE_TITLES,
    MODEL_DISPLAY_NAMES as DISPLAY_NAMES,
    MODEL_STYLES,
    compute_llm_balanced_accuracy,
)

class _DictPrediction:
    """Wrap a dict result (from Modal) to look like PredictionResult."""
    def __init__(self, d, ref_attempts=None):
        self.y_true = np.array(d["y_true"])
        self.y_pred_prob = np.array(d["y_pred_prob"])
        self.attempt_indices = np.array(d["attempt_indices"]) if "attempt_indices" in d else ref_attempts
        self.student_indices = np.array(d["student_indices"]) if "student_indices" in d else np.zeros(len(self.y_true), dtype=int)
        self.item_indices = np.array(d["item_indices"]) if "item_indices" in d else np.zeros(len(self.y_true), dtype=int)
        self.synthetic_indices = "student_indices" not in d or "item_indices" not in d
        self.losses = d.get("losses")


def _load_ref_attempts(output_dir, n_obs):
    """Borrow attempt_indices from another model with the same split."""
    for fname in os.listdir(output_dir):
        if not fname.endswith("_student_pred.pkl"):
            continue
        with open(os.path.join(output_dir, fname), "rb") as f:
            pred = pickle.load(f)
        if hasattr(pred, "attempt_indices") and pred.attempt_indices is not None:
            if len(pred.attempt_indices) == n_obs:
                return np.array(pred.attempt_indices)
    return None


def load_prediction_result(model_name, output_dir):
    """Load a saved PredictionResult pickle."""
    pkl_path = os.path.join(output_dir, f"{model_name}_student_pred.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        pred = pickle.load(f)
    if isinstance(pred, dict):
        ref = _load_ref_attempts(output_dir, len(pred["y_true"]))
        return _DictPrediction(pred, ref)
    return pred


def _balanced_accuracy(actual, pred_binary):
    pos = actual == 1
    neg = actual == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    tpr = (pred_binary[pos] == 1).mean()
    tnr = (pred_binary[neg] == 0).mean()
    return (tpr + tnr) / 2


def compute_per_attempt_accuracy(prediction, max_attempts=10):
    """Compute balanced accuracy and CI at each attempt number."""
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
        bal_acc = _balanced_accuracy(actual, pred_binary)
        accs.append(bal_acc)
        counts.append(int(mask.sum()))
        if np.isnan(bal_acc):
            ci_los.append(np.nan)
            ci_his.append(np.nan)
            continue

        rng = np.random.default_rng(42)
        boot_vals = []
        for _ in range(2000):
            idx = rng.integers(0, len(actual), size=len(actual))
            b = _balanced_accuracy(actual[idx], pred_binary[idx])
            if not np.isnan(b):
                boot_vals.append(b)
        boot_vals = np.array(boot_vals)
        ci_los.append(np.quantile(boot_vals, 0.025) if len(boot_vals) > 10 else np.nan)
        ci_his.append(np.quantile(boot_vals, 0.975) if len(boot_vals) > 10 else np.nan)

    return np.array(accs), np.array(ci_los), np.array(ci_his), np.array(counts)


LLM_MODELS = {
    "LLM": {
        "jsonl": "qwen_server_attempts10.jsonl",
        "color": "#883300",
        "marker": "P",
    },
}


def compute_llm_per_attempt_accuracy(llm_jsonl_path, course, max_attempts=10):
    accs, ci_los, ci_his, counts, _ = compute_llm_balanced_accuracy(
        llm_jsonl_path, course, max_attempts,
    )
    return accs, ci_los, ci_his, counts


def plot_course(course, models, max_attempts, output_dir, output_suffix=""):
    predictions = {}
    missing = []
    for m in models:
        pred = load_prediction_result(m, output_dir)
        if pred is not None:
            predictions[m] = pred
            print(f"  Loaded saved predictions for {m}")
        else:
            missing.append(m)

    if missing:
        print(f"  WARNING: Missing predictions for {missing}. "
              f"Run run_student_eval.py first.")

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(1, max_attempts + 1)

    for model_name in models:
        pred = predictions.get(model_name)
        if pred is None:
            continue

        accs, ci_los, ci_his, _ = compute_per_attempt_accuracy(pred, max_attempts)
        style = MODEL_STYLES.get(model_name, {"color": "gray", "marker": "o"})
        valid = ~np.isnan(accs)

        display = DISPLAY_NAMES.get(model_name, model_name)
        ax.plot(
            x[valid], accs[valid],
            marker=style["marker"], linestyle="-",
            color=style["color"],
            linewidth=2.5, markersize=9,
            label=display,
        )
        ci_valid = valid & ~np.isnan(ci_los) & ~np.isnan(ci_his)
        if ci_valid.any():
            yerr = np.array([accs[ci_valid] - ci_los[ci_valid],
                             ci_his[ci_valid] - accs[ci_valid]])
            ax.errorbar(
                x[ci_valid], accs[ci_valid], yerr=yerr,
                fmt="none", ecolor=style["color"], alpha=0.4,
                capsize=3, capthick=1,
            )

    if "LLM" in models:
        llm_dir = output_dir.replace("student_eval", "llm_student_eval")
        for llm_name, llm_cfg in LLM_MODELS.items():
            llm_path = os.path.join(llm_dir, llm_cfg["jsonl"])
            if not os.path.exists(llm_path):
                continue
            llm_accs, llm_ci_los, llm_ci_his, _ = compute_llm_per_attempt_accuracy(
                llm_path, course, max_attempts,
            )
            v = ~np.isnan(llm_accs)
            ax.plot(
                x[v], llm_accs[v],
                marker=llm_cfg["marker"], linestyle="--",
                color=llm_cfg["color"],
                linewidth=2.5, markersize=9,
                label=llm_name,
            )
            ci_v = v & ~np.isnan(llm_ci_los) & ~np.isnan(llm_ci_his)
            if ci_v.any():
                yerr = np.array([llm_accs[ci_v] - llm_ci_los[ci_v],
                                 llm_ci_his[ci_v] - llm_accs[ci_v]])
                ax.errorbar(
                    x[ci_v], llm_accs[ci_v], yerr=yerr,
                    fmt="none", ecolor=llm_cfg["color"], alpha=0.4,
                    capsize=3, capthick=1,
                )

    ax.set_xlabel("Attempt Number", fontsize=16)
    ax.set_ylabel("Balanced Accuracy", fontsize=16)
    ax.set_title(COURSE_TITLES.get(course, course), fontsize=18)
    ax.set_xticks(x)
    ax.tick_params(labelsize=14)
    ax.set_xlim(0.5, max_attempts + 0.5)
    ax.autoscale(axis="y")
    y_lo, y_hi = ax.get_ylim()
    pad = 0.02 * (y_hi - y_lo)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"accuracy_vs_attempt{output_suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-attempt accuracy")
    parser.add_argument("--courses", nargs="+", default=["dsa_hk231"])
    parser.add_argument("--models", nargs="+", default=["IRT", "CIRT", "BKT", "DKT", "CodeDKT", "RSSM", "LLM"])
    parser.add_argument("--max_attempts", type=int, default=10)
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    for course in args.courses:
        print(f"\n{'=' * 60}")
        print(f"Course: {course}")
        print(f"{'=' * 60}")
        output_dir = f"results/student_eval/{course}"
        plot_course(course, args.models, args.max_attempts, output_dir, args.output_suffix)

    # Combined plot if multiple courses
    if len(args.courses) > 1:
        plot_combined(args.courses, args.models, args.max_attempts)


def plot_combined(courses, models, max_attempts):
    fig, axes = plt.subplots(len(courses), 1, figsize=(10, 7 * len(courses)))
    x = np.arange(1, max_attempts + 1)

    for idx, course in enumerate(courses):
        ax = axes[idx]
        output_dir = f"results/student_eval/{course}"

        for model_name in models:
            if model_name == "LLM":
                continue
            pred = load_prediction_result(model_name, output_dir)
            if pred is None:
                continue
            accs, ci_los, ci_his, _ = compute_per_attempt_accuracy(pred, max_attempts)
            style = MODEL_STYLES.get(model_name, {"color": "gray", "marker": "o"})
            valid = ~np.isnan(accs)
            display = DISPLAY_NAMES.get(model_name, model_name)
            ax.plot(x[valid], accs[valid], marker=style["marker"], linestyle="-",
                    color=style["color"], linewidth=2.5, markersize=9, label=display)
            ci_valid = valid & ~np.isnan(ci_los) & ~np.isnan(ci_his)
            if ci_valid.any():
                yerr = np.array([accs[ci_valid] - ci_los[ci_valid],
                                 ci_his[ci_valid] - accs[ci_valid]])
                ax.errorbar(x[ci_valid], accs[ci_valid], yerr=yerr,
                            fmt="none", ecolor=style["color"], alpha=0.4,
                            capsize=3, capthick=1)

        if "LLM" in models:
            llm_dir = output_dir.replace("student_eval", "llm_student_eval")
            for llm_name, llm_cfg in LLM_MODELS.items():
                llm_path = os.path.join(llm_dir, llm_cfg["jsonl"])
                if not os.path.exists(llm_path):
                    continue
                llm_accs, llm_ci_los, llm_ci_his, _ = compute_llm_per_attempt_accuracy(
                    llm_path, course, max_attempts)
                v = ~np.isnan(llm_accs)
                ax.plot(x[v], llm_accs[v], marker=llm_cfg["marker"], linestyle="--",
                        color=llm_cfg["color"], linewidth=2.5, markersize=9, label=llm_name)
                ci_v = v & ~np.isnan(llm_ci_los) & ~np.isnan(llm_ci_his)
                if ci_v.any():
                    yerr = np.array([llm_accs[ci_v] - llm_ci_los[ci_v],
                                     llm_ci_his[ci_v] - llm_accs[ci_v]])
                    ax.errorbar(x[ci_v], llm_accs[ci_v], yerr=yerr,
                                fmt="none", ecolor=llm_cfg["color"], alpha=0.4,
                                capsize=3, capthick=1)

        ax.set_xlabel("Attempt Number", fontsize=16)
        ax.set_ylabel("Balanced Accuracy", fontsize=16)
        ax.set_title(COURSE_TITLES.get(course, course), fontsize=18)
        ax.set_xticks(x)
        ax.tick_params(labelsize=14)
        ax.set_xlim(0.5, max_attempts + 0.5)
        ax.autoscale(axis="y")
        y_lo, y_hi = ax.get_ylim()
        pad = 0.02 * (y_hi - y_lo)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.grid(True, alpha=0.3)

    # One shared horizontal legend below the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_dir = os.path.join(REPO_ROOT, "overleaf", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "accuracy_vs_attempt_combined.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined plot saved: {out_path}")


if __name__ == "__main__":
    main()
