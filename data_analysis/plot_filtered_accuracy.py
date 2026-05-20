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

DISPLAY_NAMES = {
    "CIRT-Decay": "CIRT",
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


def _resolve_llm_jsonl(output_dir):
    path = os.path.join(
        output_dir.replace("student_eval", "llm_student_eval"),
        "claude_attempts10.jsonl",
    )
    return path if os.path.exists(path) else None


def compute_llm_per_attempt_accuracy(llm_jsonl_path, course, max_attempts=10):
    import json
    from collections import defaultdict
    from dynamic_models.temporal_eval.data_loader import load_student_split_data
    from huggingface_hub import snapshot_download
    import pandas as pd

    if not os.path.exists(llm_jsonl_path):
        return None, None, None, None

    data, split = load_student_split_data(course)
    qi = data.question_infos
    test_item_set = set(split.test_item_indices.tolist())

    hf_dir = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        repo_type="dataset", local_files_only=True,
    )
    hf_qi = pd.read_csv(f"{hf_dir}/question_infos.csv")
    qid_to_name = {
        str(int(row["question_id"])): row["question_name"]
        for _, row in hf_qi.iterrows()
    }

    with open(llm_jsonl_path) as f:
        rows = [json.loads(l) for l in f]

    llm_by_pair = defaultdict(list)
    for r in rows:
        llm_by_pair[(str(r["student_id"]), str(r["question_unittest_id"]))].append(r)

    accs = []
    ci_los = []
    ci_his = []
    counts = []

    for a in range(max_attempts):
        actuals_list = []
        preds_list = []
        student_ids = []
        item_ids = []

        for (sid, qid), attempts in llm_by_pair.items():
            s_indices = [i for i in split.test_student_indices
                         if str(data.student_ids[i]) == sid]
            if not s_indices:
                continue
            s_idx = s_indices[0]

            qname = qid_to_name.get(qid, "")
            q_items = [i for i in qi[qi["qname"] == qname].index
                       if i in test_item_set]
            if not q_items:
                continue

            real_results = []
            for qidx in q_items:
                obs = data.correctness_matrix[s_idx, qidx, :].numpy()
                valid = obs[obs != -1]
                if a < len(valid):
                    real_results.append(int(valid[a]))
                elif len(valid) > 0:
                    real_results.append(int(valid[-1]))

            if a < len(attempts):
                llm_pp = attempts[a]["pass"]
            else:
                llm_pp = attempts[-1]["pass"]

            if not real_results:
                continue

            llm_binary = [int(c) for c in str(llm_pp) if c in "01"]
            if not llm_binary:
                continue

            n = min(len(real_results), len(llm_binary))
            for ti in range(n):
                actuals_list.append(real_results[ti])
                preds_list.append(llm_binary[ti])
                student_ids.append(sid)
                item_ids.append(qid)

        if len(actuals_list) >= 10:
            actuals_arr = np.array(actuals_list)
            preds_arr = np.array(preds_list)
            pos = actuals_arr == 1
            neg = actuals_arr == 0
            tpr = (preds_arr[pos] == 1).mean() if pos.sum() > 0 else np.nan
            tnr = (preds_arr[neg] == 0).mean() if neg.sum() > 0 else np.nan
            bal_acc = (tpr + tnr) / 2 if not (np.isnan(tpr) or np.isnan(tnr)) else np.nan
            accs.append(bal_acc)

            rng = np.random.default_rng(42)
            boot_vals = []
            for _ in range(500):
                idx = rng.integers(0, len(actuals_arr), size=len(actuals_arr))
                ba, bp = actuals_arr[idx], preds_arr[idx]
                p = ba == 1; n = ba == 0
                t = (bp[p] == 1).mean() if p.sum() > 0 else np.nan
                tn = (bp[n] == 0).mean() if n.sum() > 0 else np.nan
                if not (np.isnan(t) or np.isnan(tn)):
                    boot_vals.append((t + tn) / 2)
            boot_vals = np.array(boot_vals)
            ci_los.append(np.quantile(boot_vals, 0.025) if len(boot_vals) > 10 else np.nan)
            ci_his.append(np.quantile(boot_vals, 0.975) if len(boot_vals) > 10 else np.nan)
        else:
            accs.append(np.nan)
            ci_los.append(np.nan)
            ci_his.append(np.nan)
        counts.append(len(actuals_list))

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

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(1, max_attempts + 1)

    for model_name in models:
        pred = predictions.get(model_name)
        if pred is None:
            continue

        accs, ci_los, ci_his, counts = compute_per_attempt_accuracy(pred, max_attempts)
        style = MODEL_STYLES.get(model_name, {"color": "gray", "marker": "o"})
        valid = ~np.isnan(accs)

        display = DISPLAY_NAMES.get(model_name, model_name)
        ax.plot(
            x[valid], accs[valid],
            f"{style['marker']}-",
            color=style["color"],
            linewidth=2.5, markersize=9,
            label=display,
        )

    llm_path = _resolve_llm_jsonl(output_dir)
    if llm_path:
        llm_accs, llm_ci_los, llm_ci_his, llm_counts = compute_llm_per_attempt_accuracy(
            llm_path, course, max_attempts,
        )
        if llm_accs is not None:
            llm_style = {"color": "#e68a00", "marker": "P"}
            v = ~np.isnan(llm_accs)
            ax.plot(
                x[v], llm_accs[v],
                f"{llm_style['marker']}--",
                color=llm_style["color"],
                linewidth=2.5, markersize=9,
                label="LLM",
            )
            print(f"  LLM Balanced accuracy: {llm_accs}")


    ax.set_xlabel("Attempt Number", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_title("DSA 231", fontsize=18)
    ax.set_xticks(x)
    ax.tick_params(labelsize=14)
    ax.set_xlim(0.5, max_attempts + 0.5)
    ax.set_ylim(0.5, 0.9)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "accuracy_vs_attempt.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
