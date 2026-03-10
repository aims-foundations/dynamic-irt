"""
Sawtooth Analysis: Visualize Learning vs Memorization in Student Trajectories.

Shows the "sawtooth" pattern: within each question, marks improve over attempts
(memorization / item-specific overfitting), but when switching to a new question,
marks crash back down (only transferable learning persists).

Usage:
    cd CodeInsights
    python -m data_analysis.sawtooth_analysis
    python -m data_analysis.sawtooth_analysis --course_name dsa_hk231
    python -m data_analysis.sawtooth_analysis --all_courses

Output:
    Files saved to data_analysis/sawtooth_outputs/ directory.
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from huggingface_hub import login, snapshot_download
from tueplots import bundles

# Publication-quality styling
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sawtooth_outputs")


# ---------------------------------------------------------------------------
# Data loading (reused pattern from pace_analysis.py)
# ---------------------------------------------------------------------------

def sanitize_latex(text):
    """Escape LaTeX special characters."""
    if pd.isna(text):
        return text
    text = str(text)
    for old, new in {'_': '\\_', '&': '\\&', '%': '\\%', '$': '\\$',
                      '#': '\\#', '{': '\\{', '}': '\\}'}.items():
        text = text.replace(old, new)
    return text


def beautify_course_name(course_name):
    """pf_hk232 -> Programming Fundamentals (Spring 2023)."""
    if pd.isna(course_name):
        return course_name
    course_name = str(course_name).lower()
    if course_name.startswith("pf"):
        course_type = "Programming Fundamentals"
    elif course_name.startswith("dsa"):
        course_type = "Data Structures \\& Algorithms"
    else:
        course_type = course_name.split("_")[0].upper()
    if "_hk" in course_name:
        semester_code = course_name.split("_hk")[1][:3]
        year = "20" + semester_code[0:2]
        semester = "Fall" if semester_code[2] == "1" else "Spring"
        return f"{course_type} ({semester} {year})"
    return course_type


def compute_marks_from_pass(pass_str):
    """Binary pass string (e.g. '111011') -> fraction passed (0.833)."""
    if pd.isna(pass_str) or pass_str == '':
        return np.nan
    try:
        s = str(pass_str).strip()
        if '.' in s:
            s = str(int(float(s)))
        if len(s) == 0:
            return np.nan
        passed = sum(1 for c in s if c == '1')
        return passed / len(s) if len(s) > 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def load_data(course_name=None):
    """Load from HuggingFace stair-lab/code_insights_csv."""
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--stair-lab--code_insights_csv/"
        "snapshots/99d53fe7c11f6302fb28b82fab5ebd77c00e5d12"
    )
    if os.path.exists(cache_path):
        path = cache_path
    else:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        path = snapshot_download(
            repo_id="stair-lab/code_insights_csv", repo_type="dataset"
        )

    main_data = pd.read_csv(f"{path}/main_data.csv", low_memory=False)
    question_infos = pd.read_csv(f"{path}/question_infos.csv")
    course_infos = pd.read_csv(f"{path}/course_infos.csv")

    # Filter to actual submissions
    main_data = main_data[
        main_data["response_type"].isin(["Submit", "Prechecked"])
    ].copy()
    main_data = main_data.dropna(subset=["pass"])

    # Merge course names
    main_data = main_data.merge(course_infos, on="course_id", how="left")

    if course_name:
        main_data = main_data[main_data["course_name"] == course_name].copy()

    # Compute marks
    main_data["marks"] = main_data["pass"].apply(compute_marks_from_pass)

    print(f"Loaded {len(main_data):,} submissions"
          f" ({main_data['student_id'].nunique()} students)")

    return main_data, question_infos, course_infos


# ---------------------------------------------------------------------------
# Core data transformation
# ---------------------------------------------------------------------------

def build_student_trajectories(main_data, question_infos):
    """Build chronological per-student trajectories with question boundaries.

    Returns DataFrame with columns:
        student_id, question_unittest_id, question_name, week,
        marks, global_sub_idx, within_q_attempt, is_first_on_q, question_order
    """
    # Merge question metadata
    df = main_data.merge(
        question_infos[["question_id", "question_name", "week"]],
        left_on="question_unittest_id",
        right_on="question_id",
        how="left",
    )

    # Parse timestamps for chronological ordering
    df["ts"] = pd.to_datetime(df["timestamp"], format="%d/%m/%y, %H:%M:%S",
                              errors="coerce")

    # Sort chronologically per student
    df = df.sort_values(["student_id", "ts"]).reset_index(drop=True)

    # Sequential submission index per student
    df["global_sub_idx"] = df.groupby("student_id").cumcount()

    # Within-question attempt index
    df["within_q_attempt"] = df.groupby(
        ["student_id", "question_unittest_id"]
    ).cumcount()

    # Detect question boundaries: first time each question appears per student
    df["is_first_on_q"] = df["within_q_attempt"] == 0

    # Question order: which new-question number is this for the student
    first_attempts = df[df["is_first_on_q"]].copy()
    first_attempts["question_order"] = first_attempts.groupby(
        "student_id"
    ).cumcount() + 1
    df = df.merge(
        first_attempts[["student_id", "question_unittest_id", "question_order"]],
        on=["student_id", "question_unittest_id"],
        how="left",
    )

    return df


# ---------------------------------------------------------------------------
# Student selection
# ---------------------------------------------------------------------------

def select_representative_students(df, n_students=2):
    """Select students that clearly show the sawtooth pattern.

    Picks students with many questions, multiple attempts per question,
    and one with positive transferable learning slope, one with weaker slope.
    """
    stats = df.groupby("student_id").agg(
        n_questions=("question_unittest_id", "nunique"),
        n_submissions=("global_sub_idx", "count"),
    )
    stats["avg_attempts"] = stats["n_submissions"] / stats["n_questions"]

    # Need enough questions to show pattern, but not so many submissions
    # that the plot becomes unreadable (cap at ~200 submissions)
    candidates = stats[
        (stats["n_questions"] >= 8)
        & (stats["avg_attempts"] >= 3)
        & (stats["n_submissions"] <= 200)
    ].index

    if len(candidates) < n_students:
        # Relax criteria
        candidates = stats[
            (stats["n_questions"] >= 5)
            & (stats["avg_attempts"] >= 2)
            & (stats["n_submissions"] <= 300)
        ].index

    if len(candidates) == 0:
        candidates = stats.nlargest(10, "n_questions").index

    first = df[df["is_first_on_q"] & df["student_id"].isin(candidates)].copy()

    # Compute per-student slope of first-attempt marks over question_order
    def _slope(g):
        if len(g) < 3:
            return np.nan
        return np.polyfit(g["question_order"], g["marks"], 1)[0]

    slopes = first.groupby("student_id").apply(
        _slope, include_groups=False
    ).dropna()
    if len(slopes) == 0:
        return list(candidates[:n_students])

    # Pick students with clear sawtooth: high avg memorization gain
    q_stats = df[df["student_id"].isin(candidates)].groupby(
        ["student_id", "question_unittest_id"]
    )["marks"].agg(["first", "last"])
    q_stats["gain"] = q_stats["last"] - q_stats["first"]
    mean_gain = q_stats.groupby("student_id")["gain"].mean()

    # Score candidates: prefer high memorization gain + moderate slope
    valid = slopes.index.intersection(mean_gain.index)
    score = mean_gain.reindex(valid).fillna(0)
    # Pick the one with highest memorization gain (clearest sawtooth)
    best = score.nlargest(min(5, len(score)))
    # From top candidates, pick one with positive slope and one with lower slope
    best_slopes = slopes.reindex(best.index)
    strong = best_slopes.idxmax()
    weak = best_slopes.idxmin()
    if weak == strong and len(best) > 1:
        weak = best.index[1]

    selected = [strong, weak][:n_students]
    print(f"Selected students: {selected}")
    for sid in selected:
        s = stats.loc[sid]
        print(f"  {sid}: {s['n_questions']} questions, {s['n_submissions']} submissions, "
              f"slope={slopes.get(sid, float('nan')):.4f}")
    return selected


# ---------------------------------------------------------------------------
# Figure 1: Individual Student Sawtooth
# ---------------------------------------------------------------------------

def plot_individual_sawtooth(df, student_ids, output_dir, course_name):
    """Plot sawtooth trajectories for individual students."""
    n = len(student_ids)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.8 * n), squeeze=False)

    cmap = plt.colormaps["tab10"]

    for row, sid in enumerate(student_ids):
        ax = axes[row, 0]
        sdf = df[df["student_id"] == sid].copy()

        # Get unique questions in chronological order
        q_order = sdf.drop_duplicates("question_unittest_id")[
            ["question_unittest_id", "question_name", "question_order"]
        ].sort_values("question_order")

        # Plot each question segment in a different color
        for i, (_, qrow) in enumerate(q_order.iterrows()):
            qid = qrow["question_unittest_id"]
            qdf = sdf[sdf["question_unittest_id"] == qid]
            color = cmap(i % 10)
            ax.plot(qdf["global_sub_idx"], qdf["marks"],
                    color=color, alpha=0.6, linewidth=1.2)

        # Vertical dashed lines at question boundaries
        boundaries = sdf[sdf["is_first_on_q"]]["global_sub_idx"].values
        for b in boundaries:
            ax.axvline(b, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

        # Bold line connecting first attempts (the "transferable ability floor")
        first = sdf[sdf["is_first_on_q"]].sort_values("global_sub_idx")
        ax.plot(first["global_sub_idx"], first["marks"],
                color="black", linewidth=2.0, zorder=5, label="First attempt")
        ax.scatter(first["global_sub_idx"], first["marks"],
                   color="black", s=25, zorder=6)

        # Annotate question names at boundaries (top of plot, rotated)
        for _, frow in first.iterrows():
            qname = frow.get("question_name", "")
            if pd.notna(qname):
                # Shorten long names
                short = str(qname)[:15]
                ax.annotate(
                    sanitize_latex(short),
                    xy=(frow["global_sub_idx"], ax.get_ylim()[1]),
                    xytext=(0, 2), textcoords="offset points",
                    fontsize=5, rotation=45, ha="left", va="bottom",
                    color="gray",
                )

        ax.set_ylabel("Marks")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title(f"Student {sanitize_latex(str(sid))}", fontsize=9)

    axes[-1, 0].set_xlabel("Submission Index (chronological)")
    fig.suptitle(
        f"Sawtooth Pattern: {beautify_course_name(course_name)}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(
            f"{output_dir}/sawtooth_individual_{course_name}.{ext}",
            bbox_inches="tight", dpi=200,
        )
    plt.close(fig)
    print(f"Saved individual sawtooth plot to {output_dir}/")


# ---------------------------------------------------------------------------
# Figure 2: Aggregate Sawtooth
# ---------------------------------------------------------------------------

def plot_aggregate_sawtooth(df, output_dir, course_name):
    """Aggregate within-question improvement and first-attempt trend."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    # --- Left: Within-question improvement (normalized) ---
    # For each (student, question) pair, compute marks improvement relative to
    # first attempt. This avoids survivorship bias where high attempt indices
    # are dominated by harder questions / weaker students.
    pairs = df.groupby(["student_id", "question_unittest_id"])
    # Only include pairs with ≥2 attempts
    multi_attempt = pairs.filter(lambda g: len(g) >= 2).copy()
    first_marks = multi_attempt.groupby(
        ["student_id", "question_unittest_id"]
    )["marks"].transform("first")
    multi_attempt["marks_improvement"] = multi_attempt["marks"] - first_marks

    # Aggregate improvement by within-question attempt index
    within = multi_attempt.groupby("within_q_attempt")["marks_improvement"].agg(
        ["mean", "std", "count"]
    )
    within = within[within["count"] >= 50]
    within = within[within.index <= 20]

    ax1.plot(within.index, within["mean"], color="#4477aa", linewidth=1.5)
    se = within["std"] / np.sqrt(within["count"])
    ax1.fill_between(within.index, within["mean"] - se, within["mean"] + se,
                     alpha=0.2, color="#4477aa")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Attempt Index (within question)")
    ax1.set_ylabel("Marks Improvement from 1st Attempt")
    ax1.set_title("Within-Question Improvement", fontsize=9)

    # --- Right: First-attempt marks over question order ---
    first = df[df["is_first_on_q"]].copy()
    trend = first.groupby("question_order")["marks"].agg(["mean", "std", "count"])
    trend = trend[trend["count"] >= 30]
    trend = trend[trend.index <= 30]

    ax2.plot(trend.index, trend["mean"], color="#ee6677", linewidth=1.5)
    se2 = trend["std"] / np.sqrt(trend["count"])
    ax2.fill_between(trend.index, trend["mean"] - se2, trend["mean"] + se2,
                     alpha=0.2, color="#ee6677")
    ax2.set_xlabel("Question Order (chronological)")
    ax2.set_ylabel("Mean First-Attempt Marks")
    ax2.set_title("Transferable Learning Trend", fontsize=9)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(
        f"Aggregate Sawtooth: {beautify_course_name(course_name)}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(
            f"{output_dir}/sawtooth_aggregate_{course_name}.{ext}",
            bbox_inches="tight", dpi=200,
        )
    plt.close(fig)
    print(f"Saved aggregate sawtooth plot to {output_dir}/")


# ---------------------------------------------------------------------------
# Figure 3: Learning vs Memorization Decomposition
# ---------------------------------------------------------------------------

def plot_learning_vs_memorization(df, output_dir, course_name):
    """Scatter: memorization gain vs transferable learning slope per student."""
    # Per-student, per-question: compute memorization gain (last - first attempt marks)
    q_stats = df.groupby(["student_id", "question_unittest_id"]).agg(
        first_mark=("marks", "first"),
        last_mark=("marks", "last"),
        n_attempts=("marks", "count"),
    )
    q_stats["mem_gain"] = q_stats["last_mark"] - q_stats["first_mark"]

    # Per-student averages
    student_stats = q_stats.groupby("student_id").agg(
        mean_mem_gain=("mem_gain", "mean"),
        n_questions=("mem_gain", "count"),
    )

    # Compute transfer slope from first-attempt marks
    first = df[df["is_first_on_q"]].copy()

    def _slope(g):
        if len(g) < 3:
            return np.nan
        return np.polyfit(g["question_order"], g["marks"], 1)[0]

    slopes = first.groupby("student_id").apply(
        _slope, include_groups=False
    ).rename("transfer_slope")
    student_stats = student_stats.join(slopes)
    student_stats = student_stats.dropna(subset=["transfer_slope"])

    # Filter to students with enough data
    student_stats = student_stats[student_stats["n_questions"] >= 3]

    fig, ax = plt.subplots(figsize=(4, 3.5), layout="constrained")
    scatter = ax.scatter(
        student_stats["mean_mem_gain"],
        student_stats["transfer_slope"],
        c=student_stats["n_questions"],
        cmap="viridis",
        alpha=0.5,
        s=15,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Questions Attempted", fontsize=8)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Mean Memorization Gain\n(last $-$ first attempt marks)")
    ax.set_ylabel("Transferable Learning Slope\n(first-attempt marks trend)")
    ax.set_title(
        f"Learning vs Memorization: {beautify_course_name(course_name)}",
        fontsize=9,
    )

    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(
            f"{output_dir}/sawtooth_decomposition_{course_name}.{ext}",
            bbox_inches="tight", dpi=200,
        )
    plt.close(fig)
    print(f"Saved decomposition plot to {output_dir}/")

    # Print summary
    print(f"\n--- Decomposition Summary ({course_name}) ---")
    print(f"Students analyzed: {len(student_stats)}")
    print(f"Mean memorization gain: {student_stats['mean_mem_gain'].mean():.3f} "
          f"(std={student_stats['mean_mem_gain'].std():.3f})")
    print(f"Mean transfer slope: {student_stats['transfer_slope'].mean():.4f} "
          f"(std={student_stats['transfer_slope'].std():.4f})")
    corr = student_stats[["mean_mem_gain", "transfer_slope"]].corr().iloc[0, 1]
    print(f"Correlation (mem_gain vs transfer_slope): {corr:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sawtooth analysis: learning vs memorization in student trajectories"
    )
    parser.add_argument(
        "--course_name", type=str, default="dsa_hk231",
        help="Course name (default: dsa_hk231)",
    )
    parser.add_argument(
        "--all_courses", action="store_true",
        help="Analyze all courses",
    )
    parser.add_argument(
        "--n_students", type=int, default=2,
        help="Number of representative students for individual plot",
    )
    parser.add_argument(
        "--student_ids", type=str, nargs="+", default=None,
        help="Specific student IDs (overrides automatic selection)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: sawtooth_outputs/)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if args.all_courses:
        courses = [None]  # None = all courses combined
    else:
        courses = [args.course_name]

    for course in courses:
        course_label = course or "all"
        print(f"\n{'='*60}")
        print(f"Analyzing: {course_label}")
        print(f"{'='*60}")

        main_data, question_infos, _ = load_data(course)
        df = build_student_trajectories(main_data, question_infos)

        # Figure 1: Individual sawtooth
        if args.student_ids:
            sids = args.student_ids
        else:
            sids = select_representative_students(df, args.n_students)
        plot_individual_sawtooth(df, sids, output_dir, course_label)

        # Figure 2: Aggregate
        plot_aggregate_sawtooth(df, output_dir, course_label)

        # Figure 3: Decomposition
        plot_learning_vs_memorization(df, output_dir, course_label)

    print("\nDone.")


if __name__ == "__main__":
    main()
