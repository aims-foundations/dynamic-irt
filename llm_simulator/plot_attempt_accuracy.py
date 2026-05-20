"""Plot per-attempt accuracy: LLM simulation vs RSSM vs random baseline.

Compares unit-test level accuracy at each attempt number between
LLM-simulated students and RSSM predictions on DSA 2023.

Usage:
    python -m llm_simulator.plot_attempt_accuracy \
        --sim_path results/llm_eval/opus_dsa_50_iterative/claude_n5_attempts10.jsonl \
        --rssm_path results/temporal_eval/dsa_hk231/RSSM_predictions.pkl \
        --course dsa_hk231 \
        --output_dir results/llm_eval/opus_dsa_50_iterative
"""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--CodeInsightTeam--code_insights_csv/"
    "snapshots/a88c99da850ddd26e2f4612b5147eb9efead9aa9"
)

COURSE_ID_MAP = {
    "dsa_hk231": 1,
    "dsa_hk221": 3,
    "pf_hk232": 0,
    "pf_hk222": 2,
}


def pass_to_binary(s):
    s = str(s).replace(".", "").strip()
    if not s or s == "nan":
        return None
    return [int(c) for c in s if c in "01"]


def twoway_cluster_bootstrap_ci(records, n_boot=5000, confidence=0.95):
    """Two-way cluster bootstrap: resample both students and questions.

    Each record is a dict with student_id, question_unittest_id, match (0/1).
    Returns (ci_lo, ci_hi).
    """
    from collections import defaultdict
    rng = np.random.default_rng(42)
    alpha = (1 - confidence) / 2

    students = np.unique([r["student_id"] for r in records])
    questions = np.unique([r["question_unittest_id"] for r in records])

    lookup = defaultdict(list)
    for r in records:
        lookup[(r["student_id"], r["question_unittest_id"])].append(r["match"])

    boot_means = []
    for _ in range(n_boot):
        s_sample = rng.choice(students, size=len(students), replace=True)
        q_sample = rng.choice(questions, size=len(questions), replace=True)
        vals = []
        for s in s_sample:
            for q in q_sample:
                vals.extend(lookup.get((s, q), []))
        if vals:
            boot_means.append(np.mean(vals))

    boot_means = np.array(boot_means)
    return np.quantile(boot_means, alpha), np.quantile(boot_means, 1 - alpha)


def load_llm_per_attempt(sim_path, real_df, max_attempts=10):
    with open(sim_path) as f:
        sim = pd.DataFrame([json.loads(l) for l in f])
    sim["attempt_num"] = pd.to_numeric(sim["attempt_id"]) + 1
    sim["student_id"] = sim["student_id"].astype(str)
    sim["question_unittest_id"] = sim["question_unittest_id"].astype(str)

    sim_pairs = set(zip(sim["student_id"], sim["question_unittest_id"]))
    real_filtered = real_df[
        real_df.apply(lambda r: (r["student_id"], r["question_unittest_id"]) in sim_pairs, axis=1)
    ]

    # For each (student, question) pair, find the last sim attempt and last real attempt
    # to use for padding after they stop appearing
    last_sim = {}
    for _, sr in sim.iterrows():
        key = (sr["student_id"], sr["question_unittest_id"])
        last_sim[key] = sr
    last_real = {}
    for _, rr in real_filtered.iterrows():
        key = (rr["student_id"], rr["question_unittest_id"])
        last_real[key] = rr

    all_pairs = set(zip(sim["student_id"], sim["question_unittest_id"])) & \
                set(zip(real_filtered["student_id"], real_filtered["question_unittest_id"]))

    accs, counts, per_attempt_records = [], [], []
    for attempt in range(1, max_attempts + 1):
        sim_at = sim[sim["attempt_num"] == attempt]
        real_at = real_filtered[real_filtered["attempt_num"] == attempt]
        sim_at_lookup = {(r["student_id"], r["question_unittest_id"]): r for _, r in sim_at.iterrows()}
        real_at_lookup = {(r["student_id"], r["question_unittest_id"]): r for _, r in real_at.iterrows()}

        all_pred, all_actual, records = [], [], []
        for sid, qid in all_pairs:
            # Use current attempt if available, otherwise pad with last known
            sr = sim_at_lookup.get((sid, qid), last_sim.get((sid, qid)))
            rr = real_at_lookup.get((sid, qid), last_real.get((sid, qid)))
            if sr is None or rr is None:
                continue

            sb = pass_to_binary(sr["pass"])
            rb = pass_to_binary(rr["pass"])
            if sb is None or rb is None:
                continue
            n = min(len(sb), len(rb))
            for ti in range(n):
                match = int(sb[ti] == rb[ti])
                all_pred.append(sb[ti])
                all_actual.append(rb[ti])
                records.append({"student_id": sid, "question_unittest_id": qid, "match": match, "actual": rb[ti]})

            # Update last known for padding future attempts
            if (sid, qid) in sim_at_lookup:
                last_sim[(sid, qid)] = sr
            if (sid, qid) in real_at_lookup:
                last_real[(sid, qid)] = rr

        if all_pred:
            accs.append(accuracy_score(all_actual, all_pred))
            counts.append(len(all_pred))
        else:
            accs.append(np.nan)
            counts.append(0)
        per_attempt_records.append(records)
    return accs, counts, per_attempt_records


def load_rssm_per_attempt(course, cutoff_week=1, max_attempts=10, output_dir="results/temporal_eval",
                          model_name="RSSM"):
    """Load saved model predictions for a given cutoff week and compute per-attempt accuracy.

    Uses the same saved predictions from the temporal eval harness.
    Works for any model: RSSM, CIRT, Elo, DynamicIRT, etc.
    """
    from dynamic_models.temporal_eval.harness import load_saved_results

    _, predictions = load_saved_results(course, output_dir, models=[model_name])
    result = predictions.get((model_name, cutoff_week))

    if result is None:
        logger.warning("No %s predictions found for cutoff_week=%d. Available: %s",
                        model_name, cutoff_week, [k for k in predictions.keys()])
        return [np.nan] * max_attempts, [0] * max_attempts

    logger.info("%s W%d: %d predictions.", model_name, cutoff_week, len(result.y_true))

    # Filter out items with extreme pass rates (<5% or >95%)
    unique_items = np.unique(result.item_indices)
    item_pass_rates = {}
    for item in unique_items:
        m = result.item_indices == item
        item_pass_rates[item] = result.y_true[m].mean()
    keep_mask = np.array([0.05 <= item_pass_rates[item] <= 0.95 for item in result.item_indices])
    n_before = len(result.y_true)
    y_true = result.y_true[keep_mask]
    y_pred = result.y_pred_prob[keep_mask]
    s_idx = result.student_indices[keep_mask]
    i_idx = result.item_indices[keep_mask]
    logger.info("  Filtered extreme items: %d -> %d predictions", n_before, len(y_true))

    if result.attempt_indices is not None:
        attempt_nums = result.attempt_indices[keep_mask]
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

    accs, counts = [], []
    for a in range(max_attempts):
        mask = attempt_nums == a
        if mask.sum() > 0:
            pred = (y_pred[mask] >= 0.5).astype(int)
            actual = y_true[mask].astype(int)
            accs.append(accuracy_score(actual, pred))
            counts.append(int(mask.sum()))
        else:
            accs.append(np.nan)
            counts.append(0)
    return accs, counts


def main():
    parser = argparse.ArgumentParser(description="Plot per-attempt accuracy: LLM vs RSSM")
    parser.add_argument("--sim_path", type=str, required=True, help="Path to LLM simulation JSONL")
    parser.add_argument("--course", type=str, default="dsa_hk231", help="Course name")
    parser.add_argument("--max_attempts", type=int, default=10, help="Max attempts to plot")
    parser.add_argument("--cutoff_week", type=int, default=1, help="Training cutoff week (predict on weeks after this)")
    parser.add_argument("--models", nargs="+", default=["RSSM"],
                        help="Temporal eval models to plot (default: RSSM)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for plot")
    args = parser.parse_args()

    course_id = COURSE_ID_MAP.get(args.course)
    if course_id is None:
        raise ValueError(f"Unknown course: {args.course}")

    logger.info("Loading real student data...")
    real = pd.read_csv(
        f"{CACHE_PATH}/main_data.csv", dtype={"pass": str},
        low_memory=False, on_bad_lines="skip",
    )
    real = real[real["course_id"] == course_id]
    real = real[real["response_type"].isin(["Submit", "Prechecked"])].dropna(subset=["pass"])
    real["timestamp_dt"] = pd.to_datetime(real["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce")
    real = real.sort_values("timestamp_dt")
    real["attempt_num"] = real.groupby(["student_id", "question_unittest_id"]).cumcount() + 1
    real["student_id"] = real["student_id"].astype(str)
    real["question_unittest_id"] = real["question_unittest_id"].astype(str)

    sim_path = args.sim_path

    logger.info("Computing LLM per-attempt accuracy...")
    llm_accs, llm_counts, llm_records = load_llm_per_attempt(sim_path, real, args.max_attempts)

    logger.info("Computing LLM bootstrap CIs...")
    llm_ci_los, llm_ci_his = [], []
    for i, records in enumerate(llm_records):
        if records:
            lo, hi = twoway_cluster_bootstrap_ci(records)
            llm_ci_los.append(lo)
            llm_ci_his.append(hi)
            logger.info("  Attempt %d: acc=%.3f, CI=[%.3f, %.3f]", i + 1, llm_accs[i], lo, hi)
        else:
            llm_ci_los.append(np.nan)
            llm_ci_his.append(np.nan)

    model_results = {}
    for model_name in args.models:
        logger.info("Computing %s per-attempt accuracy (W%d)...", model_name, args.cutoff_week)
        accs, counts = load_rssm_per_attempt(
            args.course, cutoff_week=args.cutoff_week, max_attempts=args.max_attempts,
            model_name=model_name,
        )
        model_results[model_name] = (accs, counts)
        logger.info("  %s: %s", model_name, list(zip(range(1, args.max_attempts+1), accs, counts)))

    x = list(range(1, args.max_attempts + 1))
    llm_accs = np.array(llm_accs)
    llm_ci_los = np.array(llm_ci_los)
    llm_ci_his = np.array(llm_ci_his)
    llm_err_lo = llm_accs - llm_ci_los
    llm_err_hi = llm_ci_his - llm_accs

    majority_baselines = []
    for records in llm_records:
        if records:
            actual_pass_rate = np.mean([r["actual"] for r in records])
            majority_baselines.append(max(actual_pass_rate, 1 - actual_pass_rate))
        else:
            majority_baselines.append(0.5)

    model_colors = {
        "RSSM": "green", "RSSMFull": "darkgreen",
        "CIRT": "purple", "CIRT-Decay": "mediumorchid",
        "DynamicIRT": "orange", "Elo": "brown",
        "IRT": "teal",
    }
    model_markers = {
        "RSSM": "s", "RSSMFull": "D",
        "CIRT": "^", "CIRT-Decay": "v",
        "DynamicIRT": "P", "Elo": "X",
        "IRT": "d",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, llm_accs, yerr=[llm_err_lo, llm_err_hi], fmt="o-",
                color="steelblue", linewidth=2.5, markersize=8, capsize=5,
                capthick=2, label="LLM Simulated")

    for model_name, (accs, counts) in model_results.items():
        color = model_colors.get(model_name, "gray")
        marker = model_markers.get(model_name, "s")
        ax.plot(x, accs, f"{marker}-", color=color, linewidth=2, markersize=7, label=model_name)

    ax.plot(x, majority_baselines, "^--", color="red", linewidth=2, markersize=6, alpha=0.7, label="Majority Class")


    ax.set_xlabel("Attempt Number")
    ax.set_ylabel("Unit-test Accuracy")
    ax.set_title(f"Per-Attempt Accuracy: LLM vs RSSM ({args.course}, RSSM W{args.cutoff_week})")
    ax.set_xlim(0.5, args.max_attempts + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=11)
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "per_attempt_accuracy_vs_rssm.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved to %s", out_path)


if __name__ == "__main__":
    main()
