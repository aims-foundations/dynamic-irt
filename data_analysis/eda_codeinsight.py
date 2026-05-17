"""
Exploratory Data Analysis for CodeInsight Dataset.

This script provides comprehensive visualizations of the CodeInsight dataset
to understand student learning patterns, response distributions, and data characteristics.

Usage:
    python -m data_analysis.eda_codeinsight
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import login, snapshot_download
from tueplots import bundles

# Set style for all plots with LaTeX fonts and tueplots
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
sns.set_palette("husl")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_outputs")


def beautify_course_name(course_name):
    """Convert course codes to readable names.

    Examples:
        pf_hk232 -> Programming Fundamentals (Spring 2023)
        dsa_hk231 -> Data Structures \& Algorithms (Fall 2023)
    """
    if pd.isna(course_name):
        return course_name

    course_name = str(course_name).lower()

    # Parse course type
    if course_name.startswith("pf"):
        course_type = "Programming Fundamentals"
    elif course_name.startswith("dsa"):
        course_type = "Data Structures \\& Algorithms"
    else:
        course_type = course_name.split("_")[0].upper()

    # Parse semester (HK format: HKXYZ where X=year, Y=semester, Z=unused)
    # HK231 = Fall 2023, HK232 = Spring 2023, HK221 = Fall 2021, HK222 = Spring 2022
    if "_hk" in course_name:
        semester_code = course_name.split("_hk")[1][:3]
        year = "20" + semester_code[0:2]
        semester = "Fall" if semester_code[2] == "1" else "Spring"
        return f"{course_type} ({semester} {year})"

    return course_type


def load_codeinsight_data():
    """Load CodeInsight data from HuggingFace or local cache."""
    # Try local cache first
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--CodeInsightTeam--code_insights_csv/"
        "snapshots/99d53fe7c11f6302fb28b82fab5ebd77c00e5d12"
    )
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        path = cache_path
    else:
        # Fall back to HuggingFace download
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        path = snapshot_download(
            repo_id="CodeInsightTeam/code_insights_csv", repo_type="dataset"
        )

    main_data = pd.read_csv(f"{path}/main_data.csv", low_memory=False)
    questions = pd.read_csv(f"{path}/question_infos.csv")
    courses = pd.read_csv(f"{path}/course_infos.csv")

    # Merge course names into main data
    main_data = main_data.merge(courses, on="course_id", how="left")

    # Beautify course names for display
    main_data["course_name_display"] = main_data["course_name"].apply(beautify_course_name)

    return main_data, questions


def print_dataset_summary(df, questions):
    """Print basic dataset statistics."""
    print("=" * 60)
    print("CODEINSIGHT DATASET SUMMARY")
    print("=" * 60)

    n_students = df["student_id"].nunique()
    n_questions = df["question_unittest_id"].nunique()
    n_submissions = len(df)

    print(f"\nBasic Counts:")
    print(f"  - Total submissions: {n_submissions:,}")
    print(f"  - Unique students: {n_students:,}")
    print(f"  - Unique questions (with test cases): {n_questions:,}")
    print(f"  - Unique question templates: {questions['question_id'].nunique():,}")

    # Response type breakdown
    print(f"\nResponse Types:")
    response_counts = df["response_type"].value_counts()
    for rtype, count in response_counts.items():
        print(f"  - {rtype}: {count:,} ({100*count/n_submissions:.1f}%)")

    # Course breakdown
    print(f"\nCourses:")
    course_counts = df["course_name_display"].value_counts()
    for course, count in course_counts.items():
        print(f"  - {course}: {count:,} submissions")

    # Sparsity
    density = n_submissions / (n_students * n_questions)
    print(f"\nData Density:")
    print(f"  - Student x Question matrix density: {density:.4f}")
    print(f"  - Average submissions per student: {n_submissions/n_students:.1f}")
    print(f"  - Average submissions per question: {n_submissions/n_questions:.1f}")


def plot_pass_rate_distribution(df, save=True):
    """Plot distribution of pass rates across submissions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Filter to submissions with pass data
    submit_df = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submit_df = submit_df.dropna(subset=["pass"])

    # Calculate pass rate per submission (proportion of test cases passed)
    def calc_pass_rate(pass_str):
        try:
            s = str(pass_str).strip()
            if s == "" or s == "nan":
                return np.nan
            # Handle float format like "111.0"
            if "." in s:
                s = str(int(float(s)))
            return sum(c == "1" for c in s) / len(s)
        except:
            return np.nan

    submit_df["pass_rate"] = submit_df["pass"].apply(calc_pass_rate)
    submit_df = submit_df.dropna(subset=["pass_rate"])

    # Left: Histogram of pass rates
    axes[0].hist(submit_df["pass_rate"], bins=20, edgecolor="black", alpha=0.7)
    axes[0].axvline(submit_df["pass_rate"].mean(), color="red", linestyle="--",
                    label=f"Mean: {submit_df['pass_rate'].mean():.2f}")
    axes[0].set_xlabel("Pass Rate (proportion of test cases passed)")
    axes[0].set_ylabel("Number of Submissions")
    axes[0].set_title("Distribution of Pass Rates per Submission")
    axes[0].legend()

    # Right: Binary pass/fail (all tests passed vs not)
    binary_pass = (submit_df["pass_rate"] == 1.0).value_counts()
    labels = ["Partial/Failed", "All Tests Passed"]
    colors = ["#ff6b6b", "#4ecdc4"]
    axes[1].pie([binary_pass.get(False, 0), binary_pass.get(True, 0)],
                labels=labels, autopct="%1.1f%%", colors=colors,
                explode=(0, 0.05), shadow=True)
    axes[1].set_title("Submission Outcomes")

    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/pass_rate_distribution.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/pass_rate_distribution.png")
    plt.show()

    return submit_df


def plot_attempts_distribution(df, save=True):
    """Plot distribution of attempts per student and per question."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter to actual submissions
    submit_df = df[df["response_type"].isin(["Submit", "Prechecked"])]

    # Attempts per student
    attempts_per_student = submit_df.groupby("student_id").size()
    axes[0, 0].hist(attempts_per_student, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(attempts_per_student.mean(), color="red", linestyle="--",
                       label=f"Mean: {attempts_per_student.mean():.1f}")
    axes[0, 0].axvline(attempts_per_student.median(), color="orange", linestyle="--",
                       label=f"Median: {attempts_per_student.median():.1f}")
    axes[0, 0].set_xlabel("Number of Submissions")
    axes[0, 0].set_ylabel("Number of Students")
    axes[0, 0].set_title("Submissions per Student")
    axes[0, 0].legend()

    # Attempts per question
    attempts_per_question = submit_df.groupby("question_unittest_id").size()
    axes[0, 1].hist(attempts_per_question, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(attempts_per_question.mean(), color="red", linestyle="--",
                       label=f"Mean: {attempts_per_question.mean():.1f}")
    axes[0, 1].set_xlabel("Number of Submissions")
    axes[0, 1].set_ylabel("Number of Questions")
    axes[0, 1].set_title("Submissions per Question")
    axes[0, 1].legend()

    # Unique questions attempted per student
    questions_per_student = submit_df.groupby("student_id")["question_unittest_id"].nunique()
    axes[1, 0].hist(questions_per_student, bins=30, edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(questions_per_student.mean(), color="red", linestyle="--",
                       label=f"Mean: {questions_per_student.mean():.1f}")
    axes[1, 0].set_xlabel("Number of Unique Questions")
    axes[1, 0].set_ylabel("Number of Students")
    axes[1, 0].set_title("Unique Questions Attempted per Student")
    axes[1, 0].legend()

    # Attempts per student-question pair (retries)
    attempts_per_pair = submit_df.groupby(["student_id", "question_unittest_id"]).size()
    axes[1, 1].hist(attempts_per_pair, bins=range(1, min(21, attempts_per_pair.max()+2)),
                    edgecolor="black", alpha=0.7, align="left")
    axes[1, 1].axvline(attempts_per_pair.mean(), color="red", linestyle="--",
                       label=f"Mean: {attempts_per_pair.mean():.1f}")
    axes[1, 1].set_xlabel("Number of Attempts")
    axes[1, 1].set_ylabel("Number of (Student, Question) Pairs")
    axes[1, 1].set_title("Attempts per Student-Question Pair")
    axes[1, 1].legend()

    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/attempts_distribution.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/attempts_distribution.png")
    plt.show()


def plot_temporal_patterns(df, save=True):
    """Plot temporal patterns in submissions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Parse timestamps
    df_time = df.copy()
    df_time["timestamp"] = pd.to_datetime(df_time["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce")
    df_time = df_time.dropna(subset=["timestamp"])

    # Hour of day distribution
    df_time["hour"] = df_time["timestamp"].dt.hour
    hour_counts = df_time["hour"].value_counts().sort_index()
    axes[0, 0].bar(hour_counts.index, hour_counts.values, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Hour of Day")
    axes[0, 0].set_ylabel("Number of Submissions")
    axes[0, 0].set_title("Submissions by Hour of Day")
    axes[0, 0].set_xticks(range(0, 24, 2))

    # Day of week distribution
    df_time["dayofweek"] = df_time["timestamp"].dt.dayofweek
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_counts = df_time["dayofweek"].value_counts().sort_index()
    axes[0, 1].bar([day_names[i] for i in day_counts.index], day_counts.values,
                   edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Day of Week")
    axes[0, 1].set_ylabel("Number of Submissions")
    axes[0, 1].set_title("Submissions by Day of Week")

    # Submissions over time (by week)
    df_time["week"] = df_time["timestamp"].dt.to_period("W")
    weekly_counts = df_time.groupby("week").size()
    axes[1, 0].plot(range(len(weekly_counts)), weekly_counts.values, marker="o", markersize=3)
    axes[1, 0].set_xlabel("Week Number")
    axes[1, 0].set_ylabel("Number of Submissions")
    axes[1, 0].set_title("Submissions Over Time (Weekly)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Time between consecutive submissions (same student)
    submit_df = df_time[df_time["response_type"].isin(["Submit", "Prechecked"])].copy()
    submit_df = submit_df.sort_values(["student_id", "timestamp"])
    submit_df["time_diff"] = submit_df.groupby("student_id")["timestamp"].diff()
    submit_df["time_diff_minutes"] = submit_df["time_diff"].dt.total_seconds() / 60

    # Filter to reasonable range (< 1 day)
    time_diffs = submit_df["time_diff_minutes"].dropna()
    time_diffs = time_diffs[(time_diffs > 0) & (time_diffs < 60 * 24)]

    axes[1, 1].hist(time_diffs, bins=50, edgecolor="black", alpha=0.7)
    axes[1, 1].axvline(time_diffs.median(), color="red", linestyle="--",
                       label=f"Median: {time_diffs.median():.1f} min")
    axes[1, 1].set_xlabel("Time Between Submissions (minutes)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Time Between Consecutive Submissions (Same Student)")
    axes[1, 1].legend()

    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/temporal_patterns.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/temporal_patterns.png")
    plt.show()


def plot_question_difficulty(df, save=True):
    """Plot question difficulty based on pass rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Filter to submissions with pass data
    submit_df = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submit_df = submit_df.dropna(subset=["pass"])

    # Calculate if fully passed
    def is_fully_passed(pass_str):
        try:
            s = str(pass_str).strip()
            if s == "" or s == "nan":
                return np.nan
            if "." in s:
                s = str(int(float(s)))
            return all(c == "1" for c in s)
        except:
            return np.nan

    submit_df["fully_passed"] = submit_df["pass"].apply(is_fully_passed)
    submit_df = submit_df.dropna(subset=["fully_passed"])

    # Question difficulty = 1 - success rate
    question_stats = submit_df.groupby("question_unittest_id").agg(
        success_rate=("fully_passed", "mean"),
        n_attempts=("fully_passed", "count"),
        n_students=("student_id", "nunique")
    ).reset_index()

    # Filter to questions with enough data
    question_stats = question_stats[question_stats["n_students"] >= 5]
    question_stats["difficulty"] = 1 - question_stats["success_rate"].astype(float)
    question_stats["n_attempts"] = question_stats["n_attempts"].astype(float)

    # Left: Difficulty distribution
    axes[0].hist(question_stats["difficulty"].astype(float), bins=20, edgecolor="black", alpha=0.7)
    axes[0].axvline(question_stats["difficulty"].mean(), color="red", linestyle="--",
                    label=f"Mean: {question_stats['difficulty'].mean():.2f}")
    axes[0].set_xlabel("Question Difficulty (1 - success rate)")
    axes[0].set_ylabel("Number of Questions")
    axes[0].set_title("Distribution of Question Difficulty")
    axes[0].legend()

    # Right: Difficulty vs number of attempts (scatter)
    axes[1].scatter(question_stats["n_attempts"], question_stats["difficulty"],
                    alpha=0.5, s=20)
    axes[1].set_xlabel("Number of Total Attempts")
    axes[1].set_ylabel("Difficulty")
    axes[1].set_title("Question Difficulty vs Total Attempts")

    # Add trend line
    z = np.polyfit(question_stats["n_attempts"], question_stats["difficulty"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(question_stats["n_attempts"].min(), question_stats["n_attempts"].max(), 100)
    axes[1].plot(x_line, p(x_line), "r--", alpha=0.8, label="Trend")
    axes[1].legend()

    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/question_difficulty.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/question_difficulty.png")
    plt.show()

    return question_stats


def plot_learning_curves(df, save=True):
    """Plot learning curves showing improvement over attempts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Filter to submissions
    submit_df = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submit_df = submit_df.dropna(subset=["pass"])

    # Parse timestamp and sort
    submit_df["timestamp"] = pd.to_datetime(submit_df["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce")
    submit_df = submit_df.dropna(subset=["timestamp"])
    submit_df = submit_df.sort_values(["student_id", "question_unittest_id", "timestamp"])

    # Calculate pass rate
    def calc_pass_rate(pass_str):
        try:
            s = str(pass_str).strip()
            if s == "" or s == "nan":
                return np.nan
            if "." in s:
                s = str(int(float(s)))
            return sum(c == "1" for c in s) / len(s)
        except:
            return np.nan

    submit_df["pass_rate"] = submit_df["pass"].apply(calc_pass_rate)
    submit_df = submit_df.dropna(subset=["pass_rate"])

    # Add attempt number within each student-question pair
    submit_df["attempt_num"] = submit_df.groupby(["student_id", "question_unittest_id"]).cumcount() + 1

    # Left: Average pass rate by attempt number
    attempt_stats = submit_df.groupby("attempt_num").agg(
        mean_pass_rate=("pass_rate", "mean"),
        count=("pass_rate", "count")
    ).reset_index()

    # Filter to attempts with enough data
    attempt_stats = attempt_stats[attempt_stats["count"] >= 50]
    attempt_stats = attempt_stats[attempt_stats["attempt_num"] <= 15]

    axes[0].plot(attempt_stats["attempt_num"], attempt_stats["mean_pass_rate"],
                 marker="o", linewidth=2, markersize=8)
    axes[0].fill_between(attempt_stats["attempt_num"],
                          attempt_stats["mean_pass_rate"] - 0.02,
                          attempt_stats["mean_pass_rate"] + 0.02,
                          alpha=0.2)
    axes[0].set_xlabel("Attempt Number")
    axes[0].set_ylabel("Average Pass Rate")
    axes[0].set_title("Learning Curve: Pass Rate by Attempt Number")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # Right: Success probability by global attempt order (per student)
    submit_df["global_attempt"] = submit_df.groupby("student_id").cumcount() + 1

    # Bin into deciles of global attempt
    submit_df["attempt_bin"] = pd.qcut(submit_df["global_attempt"], q=10, labels=False, duplicates="drop")

    bin_stats = submit_df.groupby("attempt_bin").agg(
        mean_pass_rate=("pass_rate", "mean"),
        count=("pass_rate", "count")
    ).reset_index()

    axes[1].bar(bin_stats["attempt_bin"], bin_stats["mean_pass_rate"],
                edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Attempt Progress Decile (1=Early, 10=Late)")
    axes[1].set_ylabel("Average Pass Rate")
    axes[1].set_title("Pass Rate by Student Progress")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/learning_curves.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/learning_curves.png")
    plt.show()


def plot_code_length_distribution(df, save=True):
    """Plot distribution of code lengths (tokens) across submissions and students."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter to submissions with code
    code_df = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    code_df = code_df.dropna(subset=["response"])

    # Count tokens (simple whitespace split)
    def count_tokens(code):
        try:
            return len(str(code).split())
        except:
            return 0

    code_df["code_length"] = code_df["response"].apply(count_tokens)
    code_df = code_df[code_df["code_length"] > 0]

    # Top-left: Distribution of code length per submission
    # Filter outliers for better visualization (remove top 1%)
    q99 = code_df["code_length"].quantile(0.99)
    code_lengths_filtered = code_df[code_df["code_length"] <= q99]["code_length"]

    axes[0, 0].hist(code_lengths_filtered, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(code_df["code_length"].mean(), color="red", linestyle="--",
                       label=f"Mean: {code_df['code_length'].mean():.0f}")
    axes[0, 0].axvline(code_df["code_length"].median(), color="orange", linestyle="--",
                       label=f"Median: {code_df['code_length'].median():.0f}")
    axes[0, 0].set_xlabel("Code Length (tokens)")
    axes[0, 0].set_ylabel("Number of Submissions")
    axes[0, 0].set_title("Code Length per Submission (99th percentile)")
    axes[0, 0].legend()

    # Top-right: Total code volume per student
    student_code_volume = code_df.groupby("student_id").agg(
        total_tokens=("code_length", "sum"),
        n_submissions=("code_length", "count")
    ).reset_index()

    # Filter outliers
    q99_student = student_code_volume["total_tokens"].quantile(0.99)
    student_filtered = student_code_volume[student_code_volume["total_tokens"] <= q99_student]

    axes[0, 1].hist(student_filtered["total_tokens"], bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(student_code_volume["total_tokens"].mean(), color="red", linestyle="--",
                       label=f"Mean: {student_code_volume['total_tokens'].mean():.0f}")
    axes[0, 1].set_xlabel("Total Code Volume (tokens)")
    axes[0, 1].set_ylabel("Number of Students")
    axes[0, 1].set_title("Total Code Volume per Student (99th percentile)")
    axes[0, 1].legend()

    # Bottom-left: Code length vs pass rate
    # Calculate pass rate per submission
    def calc_pass_rate(pass_str):
        try:
            s = str(pass_str).strip()
            if s == "" or s == "nan":
                return np.nan
            if "." in s:
                s = str(int(float(s)))
            return sum(c == "1" for c in s) / len(s)
        except:
            return np.nan

    code_df["pass_rate"] = code_df["pass"].apply(calc_pass_rate)
    code_with_pass = code_df.dropna(subset=["pass_rate"]).copy()

    # Bin code lengths for better visualization
    code_with_pass["length_bin"] = pd.qcut(code_with_pass["code_length"], q=10,
                                            labels=False, duplicates="drop")

    length_pass_stats = code_with_pass.groupby("length_bin").agg(
        mean_pass_rate=("pass_rate", "mean"),
        mean_length=("code_length", "mean"),
        count=("pass_rate", "count")
    ).reset_index()

    axes[1, 0].scatter(length_pass_stats["mean_length"], length_pass_stats["mean_pass_rate"],
                       s=length_pass_stats["count"] / 100, alpha=0.6)
    axes[1, 0].set_xlabel("Average Code Length (tokens)")
    axes[1, 0].set_ylabel("Average Pass Rate")
    axes[1, 0].set_title("Code Length vs Pass Rate")
    axes[1, 0].set_ylim(0, 1)

    # Add trend line
    z = np.polyfit(length_pass_stats["mean_length"], length_pass_stats["mean_pass_rate"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(length_pass_stats["mean_length"].min(),
                         length_pass_stats["mean_length"].max(), 100)
    axes[1, 0].plot(x_line, p(x_line), "r--", alpha=0.8, label="Trend")
    axes[1, 0].legend()

    # Bottom-right: Code length distribution by course
    courses = code_df["course_name_display"].unique()
    data_for_box = [code_df[code_df["course_name_display"] == c]["code_length"].dropna()
                    for c in courses]

    bp = axes[1, 1].boxplot(data_for_box, tick_labels=courses, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    axes[1, 1].set_ylabel("Code Length (tokens)")
    axes[1, 1].set_title("Code Length Distribution by Course")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/code_length_distribution.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/code_length_distribution.png")
    plt.show()

    return student_code_volume


def plot_course_comparison(df, save=True):
    """Compare statistics across different courses."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter to submissions
    submit_df = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()

    # Calculate pass rate
    def calc_pass_rate(pass_str):
        try:
            s = str(pass_str).strip()
            if s == "" or s == "nan":
                return np.nan
            if "." in s:
                s = str(int(float(s)))
            return sum(c == "1" for c in s) / len(s)
        except:
            return np.nan

    submit_df["pass_rate"] = submit_df["pass"].apply(calc_pass_rate)

    # Course stats
    course_stats = submit_df.groupby("course_name_display").agg(
        n_students=("student_id", "nunique"),
        n_questions=("question_unittest_id", "nunique"),
        n_submissions=("student_id", "count"),
        avg_pass_rate=("pass_rate", "mean")
    ).reset_index()

    # Top-left: Students per course
    axes[0, 0].barh(course_stats["course_name_display"], course_stats["n_students"],
                    edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Number of Students")
    axes[0, 0].set_title("Students per Course")

    # Top-right: Submissions per course
    axes[0, 1].barh(course_stats["course_name_display"], course_stats["n_submissions"],
                    edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Number of Submissions")
    axes[0, 1].set_title("Submissions per Course")

    # Bottom-left: Average pass rate by course
    course_stats_sorted = course_stats.sort_values("avg_pass_rate")
    axes[1, 0].barh(course_stats_sorted["course_name_display"], course_stats_sorted["avg_pass_rate"],
                    edgecolor="black", alpha=0.7)
    axes[1, 0].set_xlabel("Average Pass Rate")
    axes[1, 0].set_title("Average Pass Rate by Course")
    axes[1, 0].set_xlim(0, 1)

    # Bottom-right: Pass rate distribution by course (boxplot)
    courses = submit_df["course_name_display"].unique()
    data_for_box = [submit_df[submit_df["course_name_display"] == c]["pass_rate"].dropna() for c in courses]
    bp = axes[1, 1].boxplot(data_for_box, tick_labels=courses, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    axes[1, 1].set_ylabel("Pass Rate")
    axes[1, 1].set_title("Pass Rate Distribution by Course")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/course_comparison.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/course_comparison.png")
    plt.show()


def main():
    """Run all EDA analyses."""
    print("Loading CodeInsight data...")
    main_data, questions = load_codeinsight_data()

    # Summary statistics
    print_dataset_summary(main_data, questions)

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Generate all plots
    print("\n1. Pass Rate Distribution...")
    plot_pass_rate_distribution(main_data)

    print("\n2. Attempts Distribution...")
    plot_attempts_distribution(main_data)

    print("\n3. Temporal Patterns...")
    plot_temporal_patterns(main_data)

    print("\n4. Question Difficulty...")
    plot_question_difficulty(main_data)

    print("\n5. Learning Curves...")
    plot_learning_curves(main_data)

    print("\n6. Course Comparison...")
    plot_course_comparison(main_data)

    print("\n7. Code Length Distribution...")
    plot_code_length_distribution(main_data)

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
