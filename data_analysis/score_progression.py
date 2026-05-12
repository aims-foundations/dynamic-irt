"""Student score progression across problems within a course.

Three views:
1. Raw score progression: each student's n-th problem score, averaged across
   students at each position.
2. Per-student normalized: subtract each student's own mean, showing whether
   individual students trend up or down over their problem sequence.
3. Slope distribution: per-student linear regression slope of score vs problem
   sequence, showing what fraction of students improve vs decline.

Output:
    results/score_progression/
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "score_progression")
CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--CodeInsightTeam--code_insights_csv/"
    "snapshots/a88c99da850ddd26e2f4612b5147eb9efead9aa9"
)

YEAR_COLORS = {"2022": "#e74c3c", "2023": "#3498db"}
COURSE_GROUPS = {
    "DSA": {"2022": "dsa_hk221", "2023": "dsa_hk231"},
    "PF": {"2022": "pf_hk222", "2023": "pf_hk232"},
}
METRICS = ["last_submit"]
METRIC_LABELS = {
    "last_submit": "Last Submit",
}
MAX_PROBLEMS = 80
MIN_PROBLEMS_PER_STUDENT = 5
ROLLING_WINDOW = 5


def compute_marks(pass_str):
    if pd.isna(pass_str) or not isinstance(pass_str, str) or len(pass_str) == 0:
        return np.nan
    return pass_str.count("1") / len(pass_str) * 10


def load_data():
    print("Loading data...")
    df = pd.read_csv(f"{CACHE_PATH}/main_data.csv", low_memory=False, on_bad_lines="skip")
    sections = pd.read_csv(f"{CACHE_PATH}/section_infos.csv")
    courses = pd.read_csv(f"{CACHE_PATH}/course_infos.csv")
    sections = sections.merge(courses, on="course_id")
    df = df.merge(sections[["section_id", "course_name"]], on="section_id", how="inner")

    submits = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submits["timestamp_dt"] = pd.to_datetime(
        submits["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
    )
    submits = submits.dropna(subset=["timestamp_dt"])
    submits = submits.sort_values(["student_id", "question_unittest_id", "timestamp_dt"])
    submits["marks"] = submits["pass"].apply(compute_marks)
    submits = submits.dropna(subset=["marks"])
    return submits


def compute_student_problem_scores(course_data):
    """For each student-problem pair, compute first/last/best and first-attempt timestamp."""
    records = []
    for (sid, qid), g in course_data.groupby(["student_id", "question_unittest_id"]):
        marks = g.sort_values("timestamp_dt")["marks"].tolist()
        if not marks:
            continue
        records.append({
            "student_id": sid,
            "question_unittest_id": qid,
            "first_attempt_time": g["timestamp_dt"].min(),
            "last_submit": marks[-1],
        })
    return pd.DataFrame(records)


def build_per_student_sequences(scores_df, metric):
    """Order problems per student chronologically. Returns list of (student_id, sequence) tuples."""
    sequences = []
    for sid, sdata in scores_df.groupby("student_id"):
        sdata = sdata.sort_values("first_attempt_time")
        seq = sdata[metric].values[:MAX_PROBLEMS]
        if len(seq) < MIN_PROBLEMS_PER_STUDENT:
            continue
        sequences.append((sid, seq))
    return sequences


def compute_position_means(sequences):
    """Compute mean score at each position across variable-length sequences."""
    max_len = min(MAX_PROBLEMS, max(len(s) for _, s in sequences))
    sums = np.zeros(max_len)
    counts = np.zeros(max_len)
    for _, seq in sequences:
        for i, v in enumerate(seq):
            sums[i] += v
            counts[i] += 1
    mask = counts >= 10
    means = np.full(max_len, np.nan)
    means[mask] = sums[mask] / counts[mask]
    return means, counts, mask


def compute_per_student_slopes(sequences):
    """Compute OLS slope of score vs problem index for each student."""
    slopes = []
    for sid, seq in sequences:
        x = np.arange(len(seq))
        slope, _, _, _, _ = sp_stats.linregress(x, seq)
        slopes.append(slope)
    return np.array(slopes)


def plot_progression():
    submits = load_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Precompute scores for all courses
    course_scores = {}
    for group_name, year_courses in COURSE_GROUPS.items():
        for year, course_name in year_courses.items():
            course_data = submits[submits["course_name"] == course_name]
            n_students = course_data["student_id"].nunique()
            scores_df = compute_student_problem_scores(course_data)
            course_scores[(group_name, year, course_name)] = (scores_df, n_students)

    # === Score progression with individual student trajectories ===
    for metric in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))

        for idx, (group_name, year_courses) in enumerate(COURSE_GROUPS.items()):
            ax = axes[idx]

            for year, course_name in year_courses.items():
                scores_df, n_students = course_scores[(group_name, year, course_name)]
                if scores_df.empty:
                    continue

                sequences = build_per_student_sequences(scores_df, metric)
                means, counts, mask = compute_position_means(sequences)

                x = np.arange(1, len(means) + 1)
                smoothed = pd.Series(means).rolling(
                    ROLLING_WINDOW, min_periods=1, center=True
                ).mean().values

                np.random.seed(42)
                sample_idx = np.random.choice(len(sequences), min(50, len(sequences)), replace=False)
                for i in sample_idx:
                    _, seq = sequences[i]
                    sx = np.arange(1, len(seq) + 1)
                    ax.plot(sx, seq, color=YEAR_COLORS[year], alpha=0.05, linewidth=0.3)

                ax.plot(
                    x[mask], smoothed[mask],
                    color=YEAR_COLORS[year], linewidth=2.5, linestyle="--",
                    label=f"{year} (n={len(sequences)})", zorder=9,
                )

            ax.set_xlabel("Student's $n$-th Problem (chronological)")
            ax.set_ylabel("Marks")
            ax.set_title(f"{group_name} -- {METRIC_LABELS[metric]}", fontsize=10)
            ax.set_ylim(0, 10)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(OUTPUT_DIR, f"progression_{metric}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def main():
    plot_progression()


if __name__ == "__main__":
    main()
