"""Generate a LaTeX summary table for the CodeInsight dataset."""

import os

import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "dataset_summary")

CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--CodeInsightTeam--code_insights_csv/"
    "snapshots/a88c99da850ddd26e2f4612b5147eb9efead9aa9"
)

COURSE_DISPLAY = {
    "dsa_hk231": "DSA Fall '23",
    "dsa_hk221": "DSA Fall '22",
    "pf_hk232": "PF Spring '23",
    "pf_hk222": "PF Spring '22",
}


def compute_stats(main_data, questions):
    questions["n_tests"] = questions["question_unittests"].str.count(r"Unittest \d+:")

    rows = []
    for course_name, display in COURSE_DISPLAY.items():
        cd = main_data[main_data["course_name"] == course_name]
        cq = questions[questions["course_name"] == course_name]
        rows.append({
            "Course": display,
            "Students": cd["student_id"].nunique(),
            "Problems": len(cq),
            "Unit Tests": int(cq["n_tests"].sum()),
            "Submissions": len(cd),
        })

    total = {
        "Students": main_data["student_id"].nunique(),
        "Problems": len(questions),
        "Unit Tests": int(questions["n_tests"].sum()),
        "Submissions": len(main_data),
    }
    return pd.DataFrame(rows), total


def to_latex(df, total):
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Summary statistics of the CodeInsight dataset, broken down by course."
        r" Students may appear in multiple courses, so per-course counts do not sum to the unique total.}",
        r"  \label{tab:dataset_summary}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Course & Students & Problems & Unit Tests & Submissions \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"    {row['Course']} & {row['Students']:,} & {row['Problems']:,} & "
            f"{row['Unit Tests']:,} & {row['Submissions']:,} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"    \\[-0.5em]",
        f"    \\textbf{{Unique Total}} & {total['Students']:,} & {total['Problems']:,} & "
        f"{total['Unit Tests']:,} & {total['Submissions']:,} \\\\",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    print("Loading CodeInsight data...")
    main_data = pd.read_csv(f"{CACHE_PATH}/main_data.csv", low_memory=False, on_bad_lines="skip")
    questions = pd.read_csv(f"{CACHE_PATH}/question_infos.csv")
    courses = pd.read_csv(f"{CACHE_PATH}/course_infos.csv")
    main_data = main_data.merge(courses, on="course_id", how="left")
    questions = questions.merge(courses, on="course_id", how="left")

    print("Computing summary statistics...")
    df, total = compute_stats(main_data, questions)

    print("\nDataset Summary:")
    print(df.to_string(index=False))
    print(f"\nUnique Total: {total}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tex_path = os.path.join(OUTPUT_DIR, "dataset_summary.tex")
    tex = to_latex(df, total)
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"\nLaTeX table written to {tex_path}")


if __name__ == "__main__":
    main()
