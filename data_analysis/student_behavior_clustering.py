"""Student Behavior Analysis for CodeInsight Dataset.

Plots raw student behavioral patterns across three features:
1. Average time between consecutive submissions per question (seconds)
2. Average edit distance between consecutive code submissions per question (Levenshtein)
3. Total number of submissions made

Usage:
    python student_behavior_clustering.py

Output:
    - clustering_outputs/student_metrics.csv
    - overleaf/figures/student_behavior_all_courses.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from tueplots import bundles

plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "clustering_outputs")
CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--CodeInsightTeam--code_insights_csv/"
    "snapshots/a88c99da850ddd26e2f4612b5147eb9efead9aa9"
)


def load_data():
    main_data = pd.read_csv(f"{CACHE_PATH}/main_data.csv", low_memory=False, on_bad_lines="skip")
    courses = pd.read_csv(f"{CACHE_PATH}/course_infos.csv")
    main_data = main_data.merge(courses, on="course_id", how="left")
    print(f"Total submissions: {len(main_data):,}")
    print(f"Total students: {main_data['student_id'].nunique():,}")
    return main_data


def extract_student_metrics(df, min_submissions=2):
    submit_df = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submit_df = submit_df.dropna(subset=["response"])
    submit_df["timestamp_dt"] = pd.to_datetime(
        submit_df["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
    )
    submit_df = submit_df.dropna(subset=["timestamp_dt"])
    submit_df = submit_df.sort_values(["student_id", "question_unittest_id", "timestamp_dt"])

    all_time_diffs = {}
    all_edit_dists = {}
    all_sub_counts = {}

    for (student_id, qid), g in submit_df.groupby(["student_id", "question_unittest_id"]):
        responses = g["response"].tolist()
        timestamps = g["timestamp_dt"].tolist()

        all_sub_counts[student_id] = all_sub_counts.get(student_id, 0) + len(responses)

        if len(responses) < 2:
            continue

        for i in range(1, len(responses)):
            td = (timestamps[i] - timestamps[i - 1]).total_seconds()
            ed = levenshtein_distance(str(responses[i]), str(responses[i - 1]))
            all_time_diffs.setdefault(student_id, []).append(td)
            all_edit_dists.setdefault(student_id, []).append(ed)

    student_metrics = []
    for student_id in all_sub_counts:
        if all_sub_counts[student_id] < min_submissions:
            continue
        student_metrics.append({
            "student_id": student_id,
            "average_time_diff": np.mean(all_time_diffs.get(student_id, [0])),
            "total_submissions": all_sub_counts[student_id],
            "average_edit_distance": np.mean(all_edit_dists.get(student_id, [0])),
        })

    print(f"Analyzed {len(student_metrics)} students with >= {min_submissions} submissions")
    return pd.DataFrame(student_metrics)


def plot_raw(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(
        df["average_time_diff"], df["average_edit_distance"],
        alpha=0.4, s=30, edgecolors="white", linewidth=0.3,
    )
    axes[0].set_xlabel("Average Time Between Submissions (s)")
    axes[0].set_ylabel("Average Edit Distance")
    axes[0].set_title("Time vs Edit Distance")
    axes[0].set_xscale("log")

    axes[1].scatter(
        df["average_time_diff"], df["total_submissions"],
        alpha=0.4, s=30, edgecolors="white", linewidth=0.3,
    )
    axes[1].set_xlabel("Average Time Between Submissions (s)")
    axes[1].set_ylabel("Total Submissions")
    axes[1].set_title("Time vs Total Submissions")
    axes[1].set_xscale("log")

    axes[2].scatter(
        df["average_edit_distance"], df["total_submissions"],
        alpha=0.4, s=30, edgecolors="white", linewidth=0.3,
    )
    axes[2].set_xlabel("Average Edit Distance")
    axes[2].set_ylabel("Total Submissions")
    axes[2].set_title("Edit Distance vs Total Submissions")

    fig_path = os.path.join(OUTPUT_DIR, "student_behavior_all_courses.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {fig_path}")

    metrics_path = os.path.join(OUTPUT_DIR, "student_metrics.csv")
    df.to_csv(metrics_path, index=False)
    print(f"Metrics saved: {metrics_path}")


def main():
    print("Loading data...")
    main_data = load_data()

    print("\nExtracting behavioral metrics...")
    df = extract_student_metrics(main_data, min_submissions=2)

    if len(df) == 0:
        print("No students found!")
        return

    print("\nPlotting...")
    plot_raw(df)

    print(f"\nTotal students analyzed: {len(df)}")


if __name__ == "__main__":
    main()
