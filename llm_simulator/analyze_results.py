"""Compare LLM simulation predictions against actual student performance.

Loads simulation JSONL output, merges with real student data, and produces
a comparison dataset with predicted vs actual scores per (student, question).

Usage:
    python -m llm_simulator.analyze_results --results_path results/llm_eval/claude_n3_attempts50.jsonl
    python -m llm_simulator.analyze_results --results_dir results/llm_eval/

Output:
    results/llm_eval/analysis/  (CSV + summary stats)
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_REPO_ID = "CodeInsightTeam/code_insights_csv"


def _pass_fraction(s) -> float:
    s = str(s).strip()
    if not s or s == "nan":
        return np.nan
    return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan


def load_simulation(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["student_id"] = df["student_id"].astype(str)
    df["question_unittest_id"] = pd.to_numeric(df["question_unittest_id"], errors="coerce")
    df = df.dropna(subset=["question_unittest_id"])
    df["question_unittest_id"] = df["question_unittest_id"].astype(int)
    df["pass_fraction"] = df["pass"].apply(_pass_fraction)
    return df


def load_real_data() -> pd.DataFrame:
    hf_dir = snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset")
    df = pd.read_csv(
        os.path.join(hf_dir, "main_data.csv"),
        dtype={"pass": str},
        on_bad_lines="skip",
        low_memory=False,
    )
    df = df[df["response_type"] == "Submit"].copy()
    df = df.dropna(subset=["pass"])
    df["student_id"] = df["student_id"].astype(str)
    df["pass_fraction"] = df["pass"].apply(_pass_fraction)
    return df


def compute_comparison(sim_df: pd.DataFrame, real_df: pd.DataFrame, attempt_mode: str = "last_submit") -> pd.DataFrame:
    """Compare simulation vs real at the (student, question, test_index) level.

    Uses last submit for both simulated and real data.
    Explodes pass strings into per-unit-test rows.
    """
    sim_submits = sim_df[sim_df["response_type"] == "Submit"].copy()
    sim_submits["student_id"] = sim_submits["student_id"].astype(str)
    sim_submits["question_unittest_id"] = sim_submits["question_unittest_id"].astype(str)
    sim_submits["attempt_id"] = pd.to_numeric(sim_submits["attempt_id"], errors="coerce")
    real_df = real_df.copy()
    real_df["student_id"] = real_df["student_id"].astype(str)
    real_df["question_unittest_id"] = real_df["question_unittest_id"].astype(str)

    # Last submit per (student, question)
    sim_last = sim_submits.sort_values("attempt_id").groupby(
        ["student_id", "question_unittest_id"]
    ).last().reset_index()

    real_submits = real_df[real_df["response_type"] == "Submit"].copy()
    real_submits["timestamp_dt"] = pd.to_datetime(
        real_submits["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
    )
    real_submits["attempt_id"] = real_submits.groupby(
        ["student_id", "question_unittest_id"]
    ).cumcount()
    real_last = real_submits.sort_values("timestamp_dt").groupby(
        ["student_id", "question_unittest_id"]
    ).last().reset_index()

    # Keep pass strings and attempt numbers
    sim_last = sim_last[["student_id", "question_unittest_id", "pass", "attempt_id"]].copy()
    sim_last.rename(columns={"pass": "sim_pass", "attempt_id": "sim_attempt"}, inplace=True)
    real_last = real_last[["student_id", "question_unittest_id", "pass", "attempt_id"]].copy()
    real_last.rename(columns={"pass": "real_pass", "attempt_id": "real_attempt"}, inplace=True)

    merged = sim_last.merge(real_last, on=["student_id", "question_unittest_id"], how="inner")

    # Filter out rows with invalid pass strings
    merged = merged[
        merged["sim_pass"].apply(lambda s: bool(str(s).strip()) and str(s).strip() != "nan")
        & merged["real_pass"].apply(lambda s: bool(str(s).strip()) and str(s).strip() != "nan")
    ]

    # Explode into per-unit-test rows
    rows = []
    for _, r in merged.iterrows():
        sim_p = str(r["sim_pass"]).replace(".", "").strip()
        real_p = str(r["real_pass"]).replace(".", "").strip()
        n = min(len(sim_p), len(real_p))
        for ti in range(n):
            rows.append({
                "student_id": r["student_id"],
                "question_unittest_id": r["question_unittest_id"],
                "test_index": ti,
                "predicted": int(sim_p[ti]) if sim_p[ti] in "01" else 0,
                "actual": int(real_p[ti]) if real_p[ti] in "01" else 0,
                "sim_attempt": int(r["sim_attempt"]),
                "real_attempt": int(r["real_attempt"]),
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Also add question-level aggregates for convenience
    q_level = result.groupby(["student_id", "question_unittest_id"]).agg(
        predicted_score=("predicted", "mean"),
        actual_score=("actual", "mean"),
    ).reset_index()

    return result, q_level


def plot_binary_analysis(test_level: pd.DataFrame, q_level: pd.DataFrame, output_dir: str, label: str = ""):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, accuracy_score

    safe_label = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    pred = test_level["predicted"].values
    actual = test_level["actual"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Confusion matrix
    cm = confusion_matrix(actual, pred, labels=[0, 1])
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Fail", "Pass"])
    axes[0].set_yticklabels(["Fail", "Pass"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix (unit-test level)")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    # 2. Per-student accuracy
    student_acc = test_level.groupby("student_id").apply(
        lambda g: accuracy_score(g["actual"], g["predicted"]), include_groups=False
    ).sort_values()
    axes[1].barh(range(len(student_acc)), student_acc.values, color="steelblue", alpha=0.8)
    axes[1].set_yticks(range(len(student_acc)))
    axes[1].set_yticklabels(student_acc.index, fontsize=7)
    axes[1].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_title("Per-Student Accuracy")
    axes[1].legend()

    if label:
        fig.suptitle(label, fontsize=12, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{safe_label}_binary_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Binary analysis plot saved to %s", plot_path)


def plot_accuracy_over_time(test_level: pd.DataFrame, output_dir: str, label: str = ""):
    """Plot accuracy as a function of how many prior questions the student has attempted."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score

    safe_label = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")

    # For each (student, question), sim_attempt tells us the chronological position
    # Group unit tests back to question level with accuracy
    q_acc = test_level.groupby(["student_id", "question_unittest_id", "sim_attempt"]).apply(
        lambda g: pd.Series({
            "correct": (g["predicted"] == g["actual"]).mean(),
            "n_tests": len(g),
        }), include_groups=False
    ).reset_index()

    # sim_attempt is the attempt within this question. We need question order.
    # Use the question_unittest_id ordering per student as a proxy for chronological order
    q_acc["q_order"] = q_acc.groupby("student_id").cumcount()

    # Bin by question order and compute mean accuracy
    max_order = q_acc["q_order"].max()
    if max_order < 3:
        return

    n_bins = min(10, max_order + 1)
    q_acc["order_bin"] = pd.cut(q_acc["q_order"], bins=n_bins, labels=False)
    binned = q_acc.groupby("order_bin").agg(
        mean_accuracy=("correct", "mean"),
        count=("correct", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(binned["order_bin"], binned["mean_accuracy"], "o-", color="steelblue", linewidth=2, markersize=8)
    ax.fill_between(binned["order_bin"], binned["mean_accuracy"], alpha=0.15, color="steelblue")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Random")

    for _, row in binned.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row["order_bin"], row["mean_accuracy"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="gray")

    ax.set_xlabel("Question Order (chronological position per student)")
    ax.set_ylabel("Prediction Accuracy")
    ax.set_title(f"Accuracy Over Time — {label}")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{safe_label}_accuracy_over_time.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Accuracy over time plot saved to %s", plot_path)


def plot_comparison(merged: pd.DataFrame, output_path: str, label: str = ""):
    import matplotlib.pyplot as plt
    from scipy.stats import kendalltau

    # Per-student Kendall tau
    student_taus = []
    for sid, g in merged.groupby("student_id"):
        if len(g) >= 3:
            tau_s, _ = kendalltau(g["predicted_score"], g["actual_score"])
            if not np.isnan(tau_s):
                student_taus.append(tau_s)

    # Per-question Kendall tau
    question_taus = []
    for qid, g in merged.groupby("question_unittest_id"):
        if len(g) >= 3:
            tau_q, _ = kendalltau(g["predicted_score"], g["actual_score"])
            if not np.isnan(tau_q):
                question_taus.append(tau_q)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if student_taus:
        mean_s = np.mean(student_taus)
        axes[0].hist(student_taus, bins=20, color="olivedrab", alpha=0.8, edgecolor="white")
        axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label=r"$\tau = 0$")
        axes[0].axvline(mean_s, color="orange", linewidth=2, label=f"Mean = {mean_s:.3f}")
        axes[0].set_xlabel(r"Per-Student Kendall $\tau$")
        axes[0].set_ylabel("Number of Students")
        axes[0].set_title("Per-Student Rank Correlation")
        axes[0].legend()

    if question_taus:
        mean_q = np.mean(question_taus)
        axes[1].hist(question_taus, bins=20, color="olivedrab", alpha=0.8, edgecolor="white")
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label=r"$\tau = 0$")
        axes[1].axvline(mean_q, color="orange", linewidth=2, label=f"Mean = {mean_q:.3f}")
        axes[1].set_xlabel(r"Per-Question Kendall $\tau$")
        axes[1].set_ylabel("Number of Questions")
        axes[1].set_title("Per-Question Rank Correlation")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved to %s", output_path)

    # Scatter plot: predicted vs actual
    scatter_path = output_path.replace("_plot.png", "_scatter.png")
    tau_all, p_all = kendalltau(merged["predicted_score"], merged["actual_score"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["actual_score"], merged["predicted_score"], alpha=0.4, s=25, color="steelblue", edgecolors="white", linewidth=0.3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect prediction")
    ax.set_xlabel("Actual Pass Fraction")
    ax.set_ylabel("Predicted Pass Fraction")
    ax.set_title(f"Predicted vs Actual (Kendall $\\tau$ = {tau_all:.3f}, p = {p_all:.4f})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Scatter plot saved to %s", scatter_path)


def print_summary(test_level: pd.DataFrame, q_level: pd.DataFrame, label: str = "", output_dir: str = None):
    from scipy.stats import kendalltau
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    if len(test_level) == 0:
        logger.warning("No matched pairs found.")
        return

    pred = test_level["predicted"].values
    actual = test_level["actual"].values

    acc = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, zero_division=0)
    mcc = matthews_corrcoef(actual, pred)

    header = f"=== {label} ===" if label else "=== Summary ==="
    print(header)
    print(f"Unit-test pairs:     {len(test_level):,}")
    print(f"Question pairs:      {len(q_level):,}")
    print(f"Unique students:     {test_level['student_id'].nunique():,}")
    print(f"Unique questions:    {test_level['question_unittest_id'].nunique():,}")
    print(f"Mean predicted:      {pred.mean():.4f}")
    print(f"Mean actual:         {actual.mean():.4f}")
    print()
    print(f"Accuracy:            {acc:.4f}")
    print(f"F1:                  {f1:.4f}")
    print(f"MCC:                 {mcc:.4f}")
    print()

    if output_dir and len(q_level) > 0:
        safe_label = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        plot_path = os.path.join(output_dir, f"{safe_label}_plot.png")
        plot_comparison(q_level, plot_path, label)
        plot_binary_analysis(test_level, q_level, output_dir, label)
        plot_accuracy_over_time(test_level, output_dir, label)


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM simulation results vs actual student performance")
    parser.add_argument("--results_path", type=str, default=None, help="Path to a single JSONL results file")
    parser.add_argument("--results_dir", type=str, default="results/llm_eval", help="Directory containing JSONL results files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: {results_dir}/analysis/)")
    args = parser.parse_args()

    if args.results_path:
        jsonl_files = [args.results_path]
    else:
        jsonl_files = sorted(Path(args.results_dir).glob("*.jsonl"))
        if not jsonl_files:
            logger.error("No JSONL files found in %s", args.results_dir)
            return

    output_dir = args.output_dir or os.path.join(args.results_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading real student data...")
    real_df = load_real_data()
    logger.info("Real data: %d submit rows, %d students, %d questions",
                len(real_df), real_df["student_id"].nunique(),
                real_df["question_unittest_id"].nunique())

    for jsonl_path in jsonl_files:
        jsonl_path = str(jsonl_path)
        label = Path(jsonl_path).stem
        logger.info("Processing %s ...", jsonl_path)

        sim_df = load_simulation(jsonl_path)
        logger.info("  Simulation: %d rows, %d students, %d questions",
                     len(sim_df), sim_df["student_id"].nunique(),
                     sim_df["question_unittest_id"].nunique())

        result = compute_comparison(sim_df, real_df)
        if isinstance(result, tuple):
            test_level, q_level = result
        else:
            logger.warning("No matched pairs found for %s", label)
            continue

        test_csv = os.path.join(output_dir, f"{label}_unittest_comparison.csv")
        test_level.to_csv(test_csv, index=False)
        logger.info("  Saved %d unit-test pairs → %s", len(test_level), test_csv)

        q_csv = os.path.join(output_dir, f"{label}_question_comparison.csv")
        q_level.to_csv(q_csv, index=False)
        logger.info("  Saved %d question pairs → %s", len(q_level), q_csv)

        print_summary(test_level, q_level, label=label, output_dir=output_dir)

    logger.info("Analysis complete. Output in %s", output_dir)


if __name__ == "__main__":
    main()
