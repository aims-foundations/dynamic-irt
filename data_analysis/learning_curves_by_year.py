"""Learning trajectories compared across years (2022 vs 2023).

1x2 figure: left = DSA courses, right = PF courses.
Each subplot shows background student trajectories and aggregate curves colored by year.

Output:
    clustering_outputs/learning_curves_by_year.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

YEAR_COLORS = {"2022": "#e74c3c", "2023": "#3498db"}
MAX_ATTEMPTS = 30

COURSE_GROUPS = {
    "DSA": {"2022": "dsa_hk221", "2023": "dsa_hk231"},
    "PF": {"2022": "pf_hk222", "2023": "pf_hk232"},
}


def compute_marks(pass_str):
    if pd.isna(pass_str) or not isinstance(pass_str, str) or len(pass_str) == 0:
        return np.nan
    return pass_str.count("1") / len(pass_str) * 10


def main():
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
    submits["attempt_num"] = submits.groupby(
        ["student_id", "question_unittest_id"]
    ).cumcount() + 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(1, MAX_ATTEMPTS + 1)

    for idx, (group_name, year_courses) in enumerate(COURSE_GROUPS.items()):
        ax = axes[idx]

        for year, course_name in year_courses.items():
            course_data = submits[submits["course_name"] == course_name]
            n_students = course_data["student_id"].nunique()

            padded = []
            for (sid, qid), g in course_data.groupby(["student_id", "question_unittest_id"]):
                marks = g.sort_values("attempt_num")["marks"].tolist()
                if not marks:
                    continue
                best = max(marks)
                p = marks[:MAX_ATTEMPTS]
                p += [best] * (MAX_ATTEMPTS - len(p))
                padded.append(p)

            np.random.seed(42)
            sample_idx = np.random.choice(len(padded), size=min(200, len(padded)), replace=False)
            for i in sample_idx:
                ax.plot(x, padded[i], color=YEAR_COLORS[year], alpha=0.15, linewidth=0.5)

            arr = np.array(padded)
            means = np.mean(arr, axis=0)
            ax.plot(x, means, color=YEAR_COLORS[year], linewidth=2.5, linestyle="--",
                    label=f"{year} (n={n_students})", zorder=9)

            print(f"  {course_name}: {n_students} students, {len(padded):,} trajectories, "
                  f"attempt 1 mean={means[0]:.2f}, attempt 30 mean={means[-1]:.2f}")

        ax.set_xlabel("Attempts")
        ax.set_ylabel("Marks")
        ax.set_title(f"{group_name} Courses")
        ax.set_xlim(1, MAX_ATTEMPTS)
        ax.set_ylim(0, 10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUTPUT_DIR, "learning_curves_by_year.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
