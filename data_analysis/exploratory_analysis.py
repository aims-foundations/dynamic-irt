"""Unified exploratory analysis script for CodeInsight Dataset.

Combines all exploratory analyses that are not directly used in the paper
into a single CLI-switchable script.

Usage:
    python data_analysis/exploratory_analysis.py <analysis>

Available analyses:
    year-comparison      - Metric gaps on shared questions between 2022 and 2023 cohorts
    ai-detection         - Trajectory-level feature extraction for AI usage signals
    ai-impact            - 7-pronged investigation of DSA score jump causes
    cheating             - Suspicious consecutive submission pair detection
    edit-distance        - Edit distance distribution comparison (2022 vs 2023)
    active-time          - Active time vs time between submissions correlation
    section-metrics      - Kruskal-Wallis tests on behavioral metrics by section type
    section-curves       - Learning trajectories by section type (L, CC, DT)
    section-curves-course - Section learning curves split by individual course
    all                  - Run all analyses sequentially
"""

import argparse
import difflib
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from scipy import stats
from tueplots import bundles

# ---------------------------------------------------------------------------
# Shared constants and utilities
# ---------------------------------------------------------------------------

CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--CodeInsightTeam--code_insights_csv/"
    "snapshots/a88c99da850ddd26e2f4612b5147eb9efead9aa9"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
CLUSTERING_DIR = os.path.join(BASE_DIR, "clustering_outputs")

TIMESTAMP_FMT = "%d/%m/%y, %H:%M:%S"

YEAR_PAIRS = [
    ("dsa_hk231", "dsa_hk221", "DSA"),
    ("pf_hk232", "pf_hk222", "PF"),
]

COURSE_COLORS = {"dsa_hk221": "#e74c3c", "dsa_hk231": "#3498db",
                 "pf_hk222": "#e74c3c", "pf_hk232": "#3498db"}
COURSE_LABELS = {"dsa_hk221": "DSA 2022", "dsa_hk231": "DSA 2023",
                 "pf_hk222": "PF 2022", "pf_hk232": "PF 2023"}
YEAR_COLORS = {"2022": "#e74c3c", "2023": "#3498db"}

SECTION_GROUPS = ["L", "CC", "DT"]
SECTION_COLORS = {"L": "#3498db", "CC": "#2ecc71", "DT": "#9b59b6", "ALL": "#555555"}
SECTION_LABELS = {
    "L": "L (Regular)",
    "CC": "CC (Credit-Constrained)",
    "DT": "DT (Deferred/Repeat)",
    "ALL": "All Students",
}

VIET_PATTERN = re.compile(r"[\u00C0-\u00FF\u0100-\u024F\u1E00-\u1EFF]")
COMMENT_LINE = re.compile(r"^\s*(//|/\*|\*)")


def setup_plotting():
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


def pass_string_to_score(val):
    if not isinstance(val, str) or len(val) == 0:
        return np.nan
    return val.count("1") / len(val)


def compute_marks(pass_str):
    if pd.isna(pass_str) or not isinstance(pass_str, str) or len(pass_str) == 0:
        return np.nan
    return pass_str.count("1") / len(pass_str) * 10


def normalize_question_name(name):
    name = re.sub(r"\[.*?\]", "", name)
    name = name.strip().strip("-").strip().lower()
    name = re.sub(r"^p\d+-\d+-", "", name)
    name = re.sub(r"^application\s*-\s*", "", name)
    name = re.sub(r"\s*-\s*v\d+$", "", name)
    name = re.sub(r"\s*\(.*?\)\s*$", "", name)
    return name.strip()


def load_main_with_courses():
    df = pd.read_csv(f"{CACHE_PATH}/main_data.csv", low_memory=False, on_bad_lines="skip")
    sections = pd.read_csv(f"{CACHE_PATH}/section_infos.csv")
    courses = pd.read_csv(f"{CACHE_PATH}/course_infos.csv")
    sections = sections.merge(courses, on="course_id")
    df = df.merge(sections[["section_id", "course_name"]], on="section_id", how="inner")
    return df, sections, courses


def load_data_with_questions():
    main = pd.read_csv(f"{CACHE_PATH}/main_data.csv", on_bad_lines="skip", low_memory=False)
    questions = pd.read_csv(f"{CACHE_PATH}/question_infos.csv")
    courses = pd.read_csv(f"{CACHE_PATH}/course_infos.csv")
    questions = questions.merge(courses, on="course_id")
    questions["norm_name"] = questions["question_name"].apply(normalize_question_name)
    main = main.merge(
        questions[["question_id", "course_id", "question_name", "norm_name", "course_name"]],
        left_on=["question_unittest_id", "course_id"],
        right_on=["question_id", "course_id"],
        how="inner",
    )
    main["score"] = main["pass"].apply(pass_string_to_score)
    main["timestamp_dt"] = pd.to_datetime(main["timestamp"], format=TIMESTAMP_FMT, errors="coerce")
    return main, questions


def find_shared_questions(questions):
    shared = []
    for course_23, course_22, subject in YEAR_PAIRS:
        names_23 = set(questions[questions.course_name == course_23]["norm_name"])
        names_22 = set(questions[questions.course_name == course_22]["norm_name"])
        for name in names_23 & names_22:
            shared.append({"norm_name": name, "subject": subject,
                           "course_23": course_23, "course_22": course_22})
    return pd.DataFrame(shared)


def prepare_submits(df):
    submits = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submits["timestamp_dt"] = pd.to_datetime(
        submits["timestamp"], format=TIMESTAMP_FMT, errors="coerce"
    )
    submits = submits.dropna(subset=["timestamp_dt"])
    submits = submits.sort_values(["student_id", "question_unittest_id", "timestamp_dt"])
    submits["marks"] = submits["pass"].apply(compute_marks)
    submits["attempt_num"] = submits.groupby(
        ["student_id", "question_unittest_id"]
    ).cumcount() + 1
    return submits


# ---------------------------------------------------------------------------
# Analysis: year-comparison
# ---------------------------------------------------------------------------

def run_year_comparison():
    output_dir = os.path.join(RESULTS_DIR, "year_comparison")
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    print("Loading data...")
    data, questions = load_data_with_questions()
    shared = find_shared_questions(questions)
    print(f"Found {len(shared)} shared questions across year pairs")

    def compute_edit_distances(group):
        group = group.sort_values("timestamp_dt")
        responses = group["response"].astype(str).tolist()
        dists = [np.nan]
        for i in range(1, len(responses)):
            dists.append(levenshtein_distance(responses[i - 1], responses[i]))
        group = group.copy()
        group["edit_distance_from_prev"] = dists
        group["attempt_number"] = range(1, len(group) + 1)
        return group

    def compute_metrics_for_year(df):
        df = df.sort_values("timestamp_dt")
        final = df.groupby(["student_id", "question_unittest_id"]).last().reset_index()
        avg_final_score = final["score"].mean()
        attempts = df.groupby(["student_id", "question_unittest_id"]).size()
        avg_attempts = attempts.mean()
        first = df.groupby(["student_id", "question_unittest_id"]).first().reset_index()
        first_pass_rate = (first["score"] == 1.0).mean()
        with_ed = df.groupby(
            ["student_id", "question_unittest_id"], group_keys=False
        ).apply(compute_edit_distances, include_groups=False)
        avg_edit_dist = with_ed["edit_distance_from_prev"].mean()
        return {
            "avg_final_score": avg_final_score,
            "avg_attempts": avg_attempts,
            "first_attempt_pass_rate": first_pass_rate,
            "avg_edit_distance": avg_edit_dist,
        }

    results = []
    for _, row in shared.iterrows():
        norm_name = row["norm_name"]
        subject = row["subject"]
        course_23, course_22 = row["course_23"], row["course_22"]
        df_23 = data[(data.course_name == course_23) & (data.norm_name == norm_name)]
        df_22 = data[(data.course_name == course_22) & (data.norm_name == norm_name)]
        if df_23.empty or df_22.empty:
            continue
        m23 = compute_metrics_for_year(df_23)
        m22 = compute_metrics_for_year(df_22)
        result = {
            "question_name": norm_name, "subject": subject,
            "n_students_2022": df_22["student_id"].nunique(),
            "n_students_2023": df_23["student_id"].nunique(),
            "n_submissions_2022": len(df_22), "n_submissions_2023": len(df_23),
        }
        for metric in ["avg_final_score", "avg_attempts", "first_attempt_pass_rate", "avg_edit_distance"]:
            result[f"{metric}_2022"] = m22[metric]
            result[f"{metric}_2023"] = m23[metric]
            result[f"{metric}_gap"] = m23[metric] - m22[metric]
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("avg_final_score_gap", ascending=False)
    results_df.to_csv(os.path.join(output_dir, "question_gaps.csv"), index=False)
    print(f"\nSaved question gaps to {output_dir}/question_gaps.csv")

    print("\n=== Top 10 questions by score improvement (2023 - 2022) ===")
    for _, r in results_df.head(10).iterrows():
        print(
            f"  [{r['subject']}] {r['question_name']}: "
            f"score gap = {r['avg_final_score_gap']:+.3f} "
            f"({r['avg_final_score_2022']:.3f} -> {r['avg_final_score_2023']:.3f})"
        )

    # Save trajectories for top 5
    for _, row in results_df.head(5).iterrows():
        norm_name = row["question_name"]
        subject = row["subject"]
        course_23 = YEAR_PAIRS[0][0] if subject == "DSA" else YEAR_PAIRS[1][0]
        course_22 = YEAR_PAIRS[0][1] if subject == "DSA" else YEAR_PAIRS[1][1]
        frames = []
        for course, year in [(course_22, 2022), (course_23, 2023)]:
            df_year = data[(data.course_name == course) & (data.norm_name == norm_name)].copy()
            if df_year.empty:
                continue
            df_year = df_year.groupby(
                ["student_id", "question_unittest_id"], group_keys=False
            ).apply(compute_edit_distances, include_groups=False)
            df_year["year"] = year
            frames.append(df_year[["student_id", "year", "attempt_number", "response",
                                   "pass", "timestamp", "edit_distance_from_prev"]])
        if frames:
            traj = pd.concat(frames, ignore_index=True)
            safe_name = re.sub(r"[^\w\s-]", "", norm_name).strip().replace(" ", "_")[:80]
            path = os.path.join(traj_dir, f"{safe_name}.csv")
            traj.to_csv(path, index=False)
            print(f"  Saved {path} ({len(traj)} rows)")


# ---------------------------------------------------------------------------
# Analysis: ai-detection
# ---------------------------------------------------------------------------

def run_ai_detection():
    output_dir = os.path.join(RESULTS_DIR, "ai_detection")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    data, questions = load_data_with_questions()
    shared = find_shared_questions(questions)
    print(f"Found {len(shared)} shared questions")

    def compute_student_question_features(group):
        group = group.sort_values("timestamp_dt")
        responses = group["response"].astype(str).tolist()
        scores = group["score"].tolist()
        n = len(responses)
        if n < 1:
            return None

        edit_dists, norm_edit_dists = [], []
        empty_to_complete = 0
        for i in range(1, n):
            prev, curr = responses[i - 1], responses[i]
            ed = levenshtein_distance(prev, curr)
            max_len = max(len(prev), len(curr), 1)
            edit_dists.append(ed)
            norm_edit_dists.append(ed / max_len)
            if len(prev) < 50 and len(curr) > 200:
                empty_to_complete += 1

        fail_to_pass = pass_to_fail = oscillations = 0
        for i in range(1, n):
            prev_pass = scores[i - 1] > 0.5 if not np.isnan(scores[i - 1]) else False
            curr_pass = scores[i] > 0.5 if not np.isnan(scores[i]) else False
            if prev_pass != curr_pass:
                oscillations += 1
            if not prev_pass and curr_pass:
                fail_to_pass += 1
            if prev_pass and not curr_pass:
                pass_to_fail += 1

        n_subs_with_viet = 0
        total_viet_comment_lines = 0
        for resp in responses:
            has_viet = False
            for line in resp.splitlines():
                if COMMENT_LINE.match(line) and VIET_PATTERN.search(line):
                    total_viet_comment_lines += 1
                    has_viet = True
            if has_viet:
                n_subs_with_viet += 1

        total_comment_lines = total_code_lines = 0
        for resp in responses:
            lines = resp.splitlines()
            total_code_lines += len(lines)
            total_comment_lines += sum(1 for l in lines if COMMENT_LINE.match(l))

        first_score = np.nan
        for s in scores:
            if not np.isnan(s):
                first_score = s
                break

        return {
            "max_edit_distance": max(edit_dists) if edit_dists else 0,
            "avg_edit_distance": np.mean(edit_dists) if edit_dists else 0,
            "max_normalized_edit_distance": max(norm_edit_dists) if norm_edit_dists else 0,
            "empty_to_complete_jumps": empty_to_complete,
            "pass_fail_oscillations": oscillations,
            "fail_to_pass": fail_to_pass,
            "pass_to_fail": pass_to_fail,
            "first_attempt_score": first_score,
            "final_score": scores[-1] if not np.isnan(scores[-1]) else 0,
            "n_attempts": n,
            "n_subs_with_vietnamese": n_subs_with_viet,
            "total_vietnamese_comment_lines": total_viet_comment_lines,
            "pct_subs_with_vietnamese": n_subs_with_viet / n,
            "comment_ratio": total_comment_lines / max(total_code_lines, 1),
            "max_code_length": max(len(r) for r in responses),
        }

    print("Extracting trajectory-level features...")
    rows = []
    for _, sq in shared.iterrows():
        norm_name = sq["norm_name"]
        subject = sq["subject"]
        for course, year in [(sq["course_22"], 2022), (sq["course_23"], 2023)]:
            df_year = data[(data.course_name == course) & (data.norm_name == norm_name)]
            if df_year.empty:
                continue
            for (sid, qid), g in df_year.groupby(["student_id", "question_unittest_id"]):
                feats = compute_student_question_features(g)
                if feats is None:
                    continue
                feats.update({"student_id": sid, "norm_name": norm_name,
                              "subject": subject, "year": year})
                rows.append(feats)

    features = pd.DataFrame(rows)
    features.to_csv(os.path.join(output_dir, "student_features.csv"), index=False)
    print(f"Saved {len(features)} rows to {output_dir}/student_features.csv")

    print("\nYear-over-year feature comparison:")
    for col in ["avg_edit_distance", "max_edit_distance", "empty_to_complete_jumps",
                "fail_to_pass", "pass_to_fail", "pass_fail_oscillations",
                "n_subs_with_vietnamese", "pct_subs_with_vietnamese",
                "comment_ratio", "first_attempt_score", "n_attempts", "max_code_length"]:
        v22 = features[features.year == 2022][col].dropna()
        v23 = features[features.year == 2023][col].dropna()
        _, p = stats.mannwhitneyu(v22, v23, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {col:35s}  2022={v22.mean():8.3f}  2023={v23.mean():8.3f}  p={p:.2e} {sig}")

    # Plot histograms
    setup_plotting()
    metrics = [
        ("avg_edit_distance", "Avg Edit Distance"),
        ("max_edit_distance", "Max Edit Distance"),
        ("empty_to_complete_jumps", "Empty-to-Complete Jumps"),
        ("fail_to_pass", "Fail $\\to$ Pass Transitions"),
        ("pass_to_fail", "Pass $\\to$ Fail Transitions"),
        ("pass_fail_oscillations", "Total Pass/Fail Oscillations"),
        ("n_subs_with_vietnamese", "Submissions with Vietnamese Comments"),
        ("pct_subs_with_vietnamese", "\\% Submissions with Vietnamese Comments"),
        ("comment_ratio", "Comment-to-Code Ratio"),
        ("first_attempt_score", "First Attempt Score"),
        ("n_attempts", "Number of Attempts"),
        ("max_code_length", "Max Code Length (chars)"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    axes = axes.flatten()
    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        v22 = features[features.year == 2022][col].dropna()
        v23 = features[features.year == 2023][col].dropna()
        _, p_val = stats.mannwhitneyu(v22, v23, alternative="two-sided")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        all_vals = pd.concat([v22, v23])
        lo, hi = all_vals.quantile(0.0), all_vals.quantile(0.95)
        if hi == lo:
            hi = lo + 1
        bins = np.arange(lo - 0.5, hi + 1.5, 1) if all_vals.nunique() <= 15 else np.linspace(lo, hi, 40)
        ax.hist(v22, bins=bins, color=YEAR_COLORS["2022"], alpha=0.5, density=True, label="2022")
        ax.hist(v23, bins=bins, color=YEAR_COLORS["2023"], alpha=0.5, density=True, label="2023")
        ax.axvline(v22.mean(), color=YEAR_COLORS["2022"], linestyle="--", linewidth=1, alpha=0.8)
        ax.axvline(v23.mean(), color=YEAR_COLORS["2023"], linestyle="--", linewidth=1, alpha=0.8)
        ax.set_title(f"{label} ({sig})", fontsize=8)
        ax.set_ylabel("Density")
        if idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "trajectory_histograms.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Analysis: ai-impact
# ---------------------------------------------------------------------------

def run_ai_impact():
    output_dir = os.path.join(RESULTS_DIR, "ai_impact")
    os.makedirs(output_dir, exist_ok=True)

    setup_plotting()
    plt.rcParams.update({"legend.fontsize": 7})

    print("Loading data...")
    df, sections, courses = load_main_with_courses()
    submits = prepare_submits(df)

    report = []
    report.append("=" * 70)
    report.append("AI IMPACT ANALYSIS: DSA 2022 vs 2023 SCORE JUMP")
    report.append("=" * 70)
    report.append("")

    # Analysis 1: First-attempt quality
    report.append("\n" + "=" * 70)
    report.append("1. FIRST-ATTEMPT QUALITY")
    report.append("=" * 70)
    first = submits[submits["attempt_num"] == 1].dropna(subset=["marks"])
    for prefix in ["dsa", "pf"]:
        c_names = sorted(c for c in first["course_name"].unique() if c.startswith(prefix))
        report.append(f"\n  {prefix.upper()} courses:")
        for course in c_names:
            m = first[first["course_name"] == course]["marks"]
            report.append(f"    {COURSE_LABELS.get(course, course)}:")
            report.append(f"      Mean first-attempt marks: {m.mean():.2f}")
            report.append(f"      Perfect on first try: {(m == 10).mean() * 100:.1f}%")
            report.append(f"      N submissions: {len(m):,}")

    # Analysis 2: Sudden perfection
    report.append("\n" + "=" * 70)
    report.append("2. SUDDEN PERFECTION: 0 -> 10 IN ONE SUBMISSION")
    report.append("=" * 70)
    scored = submits.dropna(subset=["marks"]).copy()
    for course in ["dsa_hk221", "dsa_hk231", "pf_hk222", "pf_hk232"]:
        c = scored[scored["course_name"] == course]
        sudden_count = total_transitions = 0
        for (sid, qid), g in c.groupby(["student_id", "question_unittest_id"]):
            marks = g.sort_values("attempt_num")["marks"].tolist()
            for i in range(1, len(marks)):
                if marks[i - 1] == 0:
                    total_transitions += 1
                    if marks[i] == 10:
                        sudden_count += 1
        rate = sudden_count / max(total_transitions, 1) * 100
        report.append(f"  {COURSE_LABELS[course]}: {sudden_count:,}/{total_transitions:,} ({rate:.2f}%)")

    # Analysis 3: Code length
    report.append("\n" + "=" * 70)
    report.append("3. CODE LENGTH PATTERNS")
    report.append("=" * 70)
    with_response = submits.dropna(subset=["response"]).copy()
    with_response["code_len"] = with_response["response"].str.len()
    code_lengths = {}
    for course in ["dsa_hk221", "dsa_hk231", "pf_hk222", "pf_hk232"]:
        c = with_response[with_response["course_name"] == course]["code_len"]
        code_lengths[course] = c
        report.append(f"  {COURSE_LABELS[course]}: mean={c.mean():.0f}, median={c.median():.0f}")

    # Analysis 4: Large jumps
    report.append("\n" + "=" * 70)
    report.append("4. LARGE EDIT JUMPS AND RESULTING SCORES")
    report.append("=" * 70)
    jump_results = {}
    for course in ["dsa_hk221", "dsa_hk231", "pf_hk222", "pf_hk232"]:
        c = scored[scored["course_name"] == course].dropna(subset=["response"])
        large_marks, small_marks = [], []
        for (sid, qid), g in c.groupby(["student_id", "question_unittest_id"]):
            responses = g["response"].tolist()
            marks = g["marks"].tolist()
            for i in range(1, len(responses)):
                ed = levenshtein_distance(str(responses[i]), str(responses[i - 1]))
                if ed > 500:
                    large_marks.append(marks[i])
                elif 1 <= ed <= 50:
                    small_marks.append(marks[i])
        large, small = np.array(large_marks), np.array(small_marks)
        jump_results[course] = (large, small)
        report.append(f"  {COURSE_LABELS[course]}:")
        if len(large) > 0:
            report.append(f"    Large jumps (>500): {len(large):,}, perfect rate: {(large==10).mean()*100:.1f}%")
        if len(small) > 0:
            report.append(f"    Small edits (1-50): {len(small):,}, perfect rate: {(small==10).mean()*100:.1f}%")

    # Analysis 5: Inter-student similarity
    report.append("\n" + "=" * 70)
    report.append("5. INTER-STUDENT CODE SIMILARITY")
    report.append("=" * 70)
    perfect = scored[scored["marks"] == 10].dropna(subset=["response"])
    dsa_shared_q = set(
        perfect[perfect["course_name"] == "dsa_hk221"]["question_unittest_id"].unique()
    ) & set(
        perfect[perfect["course_name"] == "dsa_hk231"]["question_unittest_id"].unique()
    )
    np.random.seed(42)
    similarity_results = {}
    for course in ["dsa_hk221", "dsa_hk231", "pf_hk222", "pf_hk232"]:
        c = perfect[perfect["course_name"] == course]
        if course.startswith("dsa"):
            c = c[c["question_unittest_id"].isin(dsa_shared_q)]
        similarities = []
        for qid, qgroup in c.groupby("question_unittest_id"):
            students = qgroup.groupby("student_id")["response"].first().tolist()
            if len(students) < 2:
                continue
            n_pairs = min(20, len(students) * (len(students) - 1) // 2)
            for _ in range(n_pairs):
                i, j = np.random.choice(len(students), 2, replace=False)
                s1, s2 = str(students[i]), str(students[j])
                max_len = max(len(s1), len(s2))
                if max_len == 0:
                    continue
                similarities.append(1 - levenshtein_distance(s1, s2) / max_len)
        sims = np.array(similarities)
        similarity_results[course] = sims
        if len(sims) > 0:
            report.append(f"  {COURSE_LABELS[course]}: mean={sims.mean():.3f}, >0.9: {(sims>0.9).mean()*100:.1f}%")

    # Analysis 6: Time of day
    report.append("\n" + "=" * 70)
    report.append("6. TIME-OF-DAY SUBMISSION PATTERNS")
    report.append("=" * 70)
    time_results = {}
    for course in ["dsa_hk221", "dsa_hk231", "pf_hk222", "pf_hk232"]:
        hours = submits[submits["course_name"] == course]["timestamp_dt"].dt.hour
        time_results[course] = hours
        night = ((hours >= 22) | (hours < 6)).mean() * 100
        report.append(f"  {COURSE_LABELS[course]}: night (10pm-6am): {night:.1f}%")

    # Analysis 7: Weekly breakdown
    report.append("\n" + "=" * 70)
    report.append("7. WEEKLY SCORE PROGRESSION")
    report.append("=" * 70)
    scored["week"] = scored["timestamp_dt"].dt.isocalendar().week.astype(int)
    for course in ["dsa_hk221", "dsa_hk231"]:
        c = scored[scored["course_name"] == course]
        weekly = c.groupby("week")["marks"].mean().sort_index()
        report.append(f"  {COURSE_LABELS[course]}:")
        for w, m in weekly.items():
            report.append(f"    Week {w}: {m:.2f}")

    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "ai_impact_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report saved: {report_path}")

    # Generate figures
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    for course in ["dsa_hk221", "dsa_hk231"]:
        fa = first[first["course_name"] == course]["marks"].dropna()
        ax.hist(fa, bins=np.linspace(0, 10, 21), color=COURSE_COLORS[course],
                alpha=0.5, density=True, label=COURSE_LABELS[course])
    ax.set_xlabel("Marks")
    ax.set_ylabel("Density")
    ax.set_title("First-Attempt Score Distribution (DSA)")
    ax.legend()

    ax = axes[0, 1]
    categories = ["Small edit\n(1-50)", "Large jump\n($>$500)"]
    x_pos = np.arange(len(categories))
    bar_w = 0.35
    for i, course in enumerate(["dsa_hk221", "dsa_hk231"]):
        large, small = jump_results[course]
        perfect_rates = [
            (small == 10).mean() * 100 if len(small) > 0 else 0,
            (large == 10).mean() * 100 if len(large) > 0 else 0,
        ]
        ax.bar(x_pos + i * bar_w - bar_w / 2, perfect_rates, bar_w,
               color=COURSE_COLORS[course], alpha=0.8, label=COURSE_LABELS[course])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Perfect Score Rate (\\%)")
    ax.set_title("Perfect Rate After Edit Type (DSA)")
    ax.legend()

    ax = axes[0, 2]
    for course in ["dsa_hk221", "dsa_hk231"]:
        sims = similarity_results[course]
        if len(sims) > 0:
            ax.hist(sims, bins=np.linspace(0, 1, 40), color=COURSE_COLORS[course],
                    alpha=0.5, density=True, label=COURSE_LABELS[course])
    ax.set_xlabel("Pairwise Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Correct Solution Similarity (DSA)")
    ax.legend()

    ax = axes[1, 0]
    for course in ["dsa_hk221", "dsa_hk231"]:
        cl = code_lengths[course]
        ax.hist(cl[cl <= 2000], bins=np.linspace(0, 2000, 60), color=COURSE_COLORS[course],
                alpha=0.5, density=True, label=COURSE_LABELS[course])
    ax.set_xlabel("Code Length (chars)")
    ax.set_ylabel("Density")
    ax.set_title("Code Length Distribution (DSA)")
    ax.legend()

    ax = axes[1, 1]
    for course in ["dsa_hk221", "dsa_hk231"]:
        hours = time_results[course]
        counts, _ = np.histogram(hours, bins=np.arange(25))
        counts = counts / counts.sum()
        ax.plot(np.arange(24), counts, color=COURSE_COLORS[course], linewidth=1.5,
                label=COURSE_LABELS[course])
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Proportion")
    ax.set_title("Submission Time of Day (DSA)")
    ax.legend()

    ax = axes[1, 2]
    for course in ["pf_hk222", "pf_hk232"]:
        fa = first[first["course_name"] == course]["marks"].dropna()
        ax.hist(fa, bins=np.linspace(0, 10, 21), color=COURSE_COLORS[course],
                alpha=0.5, density=True, label=COURSE_LABELS[course])
    ax.set_xlabel("Marks")
    ax.set_ylabel("Density")
    ax.set_title("First-Attempt Score Distribution (PF)")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(output_dir, "ai_impact_figures.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figures saved: {out}")


# ---------------------------------------------------------------------------
# Analysis: cheating
# ---------------------------------------------------------------------------

def run_cheating():
    output_dir = os.path.join(RESULTS_DIR, "cheating_detection")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(f"{CACHE_PATH}/main_data.csv", low_memory=False, on_bad_lines="skip")
    courses = pd.read_csv(f"{CACHE_PATH}/course_infos.csv")
    df = df.merge(courses, on="course_id", how="left")

    submit_df = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submit_df = submit_df.dropna(subset=["response"])
    submit_df["timestamp_dt"] = pd.to_datetime(
        submit_df["timestamp"], format=TIMESTAMP_FMT, errors="coerce"
    )
    submit_df = submit_df.dropna(subset=["timestamp_dt"])
    submit_df = submit_df.sort_values(["student_id", "question_unittest_id", "timestamp_dt"])

    print("Computing consecutive submission pairs...")
    pairs = []
    for (student_id, qid), g in submit_df.groupby(["student_id", "question_unittest_id"]):
        responses = g["response"].tolist()
        timestamps = g["timestamp_dt"].tolist()
        pass_vals = g["pass"].tolist()
        for i in range(1, len(responses)):
            prev_r, curr_r = str(responses[i - 1]), str(responses[i])
            td = (timestamps[i] - timestamps[i - 1]).total_seconds()
            ed = levenshtein_distance(prev_r, curr_r)
            max_len = max(len(prev_r), len(curr_r), 1)
            pairs.append({
                "student_id": student_id, "question_unittest_id": qid,
                "time_diff_seconds": td, "edit_distance": ed,
                "normalized_edit_distance": ed / max_len,
                "prev_len": len(prev_r), "curr_len": len(curr_r),
                "prev_pass": pass_vals[i - 1], "curr_pass": pass_vals[i],
                "prev_response": prev_r, "curr_response": curr_r,
            })

    pairs_df = pd.DataFrame(pairs)
    print(f"Total pairs: {len(pairs_df):,}")

    metrics = pairs_df.drop(columns=["prev_response", "curr_response"])
    metrics.to_csv(os.path.join(output_dir, "pair_metrics.csv"), index=False)

    edit_p90 = pairs_df["normalized_edit_distance"].quantile(0.90)
    time_p25 = pairs_df["time_diff_seconds"].quantile(0.25)

    high_edit_mask = pairs_df["normalized_edit_distance"] >= edit_p90
    low_time_low_edit_mask = (pairs_df["time_diff_seconds"] <= time_p25) & (pairs_df["normalized_edit_distance"] <= 0.05)
    near_zero_mask = pairs_df["normalized_edit_distance"] < 0.02

    def sample_region(mask, name, n=30):
        region = pairs_df[mask]
        if len(region) == 0:
            return pd.DataFrame()
        sample = region.sample(n=min(n, len(region)), random_state=42).copy()
        sample["diff"] = sample.apply(
            lambda r: "".join(difflib.unified_diff(
                r["prev_response"].splitlines(keepends=True),
                r["curr_response"].splitlines(keepends=True), lineterm="")[:50]),
            axis=1)
        sample["region"] = name
        return sample

    cols = ["region", "student_id", "question_unittest_id", "time_diff_seconds",
            "edit_distance", "normalized_edit_distance", "prev_len", "curr_len",
            "prev_pass", "curr_pass", "prev_response", "curr_response", "diff"]

    for name, mask in [("high_edit", high_edit_mask),
                       ("low_time_low_edit", low_time_low_edit_mask),
                       ("near_zero_edit", near_zero_mask)]:
        sample = sample_region(mask, name)
        if len(sample) > 0:
            path = os.path.join(output_dir, f"samples_{name}.csv")
            sample[cols].to_csv(path, index=False)
            print(f"Saved {len(sample)} samples: {path}")

    summary = "\n".join([
        "=== CHEATING DETECTION SUMMARY ===",
        f"Total pairs: {len(pairs_df):,}",
        f"High edit (norm >= {edit_p90:.3f}): {high_edit_mask.sum():,} ({100*high_edit_mask.mean():.1f}%)",
        f"Low time + low edit: {low_time_low_edit_mask.sum():,} ({100*low_time_low_edit_mask.mean():.1f}%)",
        f"Near-zero edit (norm < 0.02): {near_zero_mask.sum():,} ({100*near_zero_mask.mean():.1f}%)",
    ])
    print(f"\n{summary}")
    with open(os.path.join(output_dir, "quadrant_summary.txt"), "w") as f:
        f.write(summary)


# ---------------------------------------------------------------------------
# Analysis: edit-distance
# ---------------------------------------------------------------------------

def run_edit_distance():
    output_dir = CLUSTERING_DIR
    os.makedirs(output_dir, exist_ok=True)
    setup_plotting()

    print("Loading data...")
    df, _, _ = load_main_with_courses()
    dsa = df[df["course_name"].str.startswith("dsa")]
    submits = dsa[dsa["response_type"].isin(["Submit", "Prechecked"])].copy()
    submits = submits.dropna(subset=["response"])
    submits["timestamp_dt"] = pd.to_datetime(submits["timestamp"], format=TIMESTAMP_FMT, errors="coerce")
    submits = submits.dropna(subset=["timestamp_dt"])
    submits = submits.sort_values(["student_id", "question_unittest_id", "timestamp_dt"])

    print("Computing edit distances...")
    dists_by_course = {}
    for course in ["dsa_hk221", "dsa_hk231"]:
        c = submits[submits["course_name"] == course]
        dists = []
        for (sid, qid), g in c.groupby(["student_id", "question_unittest_id"]):
            responses = g["response"].tolist()
            for i in range(1, len(responses)):
                dists.append(levenshtein_distance(str(responses[i]), str(responses[i - 1])))
        dists_by_course[course] = np.array(dists)
        print(f"  {COURSE_LABELS[course]}: {len(dists):,} pairs")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for course in ["dsa_hk221", "dsa_hk231"]:
        d = np.sort(dists_by_course[course])
        ecdf = np.arange(1, len(d) + 1) / len(d)
        step = max(1, len(d) // 5000)
        axes[0].plot(d[::step], ecdf[::step], color=COURSE_COLORS[course],
                     label=COURSE_LABELS[course], linewidth=1.5)
    axes[0].set_xlabel("Edit Distance")
    axes[0].set_ylabel("Cumulative Proportion")
    axes[0].set_xscale("log")
    axes[0].set_title("Cumulative Distribution (log scale)")
    axes[0].legend(loc="lower right")

    bin_edges = [0, 1, 10, 50, 100, 500, np.inf]
    bin_labels = ["0\n(identical)", "1--10", "11--50", "51--100", "101--500", "$>$500"]
    x_pos = np.arange(len(bin_labels))
    bar_width = 0.35
    for i, course in enumerate(["dsa_hk221", "dsa_hk231"]):
        d = dists_by_course[course]
        proportions = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            if lo == 0:
                proportions.append((d == 0).mean())
            else:
                proportions.append(((d > lo) & (d <= hi)).mean() if hi != np.inf else (d > lo).mean())
        offset = -bar_width / 2 + i * bar_width
        axes[1].bar(x_pos + offset, proportions, bar_width,
                    color=COURSE_COLORS[course], alpha=0.8, label=COURSE_LABELS[course])
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(bin_labels)
    axes[1].set_xlabel("Edit Distance Range")
    axes[1].set_ylabel("Proportion of Pairs")
    axes[1].set_title("Edit Distance Breakdown")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(output_dir, "edit_distance_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out}")


# ---------------------------------------------------------------------------
# Analysis: active-time
# ---------------------------------------------------------------------------

def run_active_time():
    output_dir = CLUSTERING_DIR
    os.makedirs(output_dir, exist_ok=True)
    setup_plotting()

    print("Loading data...")
    df = pd.read_csv(f"{CACHE_PATH}/main_data.csv", low_memory=False, on_bad_lines="skip")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FMT, errors="coerce")
    df = df.dropna(subset=["timestamp_dt"])
    submits = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submits = submits.sort_values("timestamp_dt")

    records = []
    for (sid, qid), group in submits.groupby(["student_id", "question_unittest_id"]):
        n = len(group)
        if n < 2:
            continue
        timestamps = group["timestamp_dt"].tolist()
        active_time_min = (timestamps[-1] - timestamps[0]).total_seconds() / 60.0
        if active_time_min <= 0:
            continue
        diffs = [(timestamps[i] - timestamps[i - 1]).total_seconds() / 60.0 for i in range(1, len(timestamps))]
        records.append({
            "student_id": sid, "question_unittest_id": qid,
            "active_time_min": active_time_min,
            "total_submissions": n,
            "mean_time_between_min": np.mean(diffs),
        })

    pairs = pd.DataFrame(records)
    print(f"(student, question) pairs with >= 2 submissions: {len(pairs):,}")

    student = pairs.groupby("student_id").agg(
        median_active_time=("active_time_min", "median"),
        median_time_between=("mean_time_between_min", "median"),
    )

    pairs["active_time_hr"] = pairs["active_time_min"] / 60
    pairs["mean_time_between_hr"] = pairs["mean_time_between_min"] / 60
    student["median_active_hr"] = student["median_active_time"] / 60
    student["median_between_hr"] = student["median_time_between"] / 60

    pairs_clip = pairs[(pairs["active_time_hr"] <= 24) & (pairs["mean_time_between_hr"] <= 24)]
    student_clip = student[(student["median_active_hr"] <= 24) & (student["median_between_hr"] <= 24)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sample = pairs_clip.sample(n=min(8000, len(pairs_clip)), random_state=42)
    axes[0].scatter(sample["active_time_hr"], sample["mean_time_between_hr"],
                    alpha=0.25, s=8, color="#2ecc71", edgecolors="none")
    axes[0].set_xlabel("Active Time (hrs)")
    axes[0].set_ylabel("Mean Time Between Submissions (hrs)")
    axes[0].set_title("Per Question")

    axes[1].scatter(student_clip["median_active_hr"], student_clip["median_between_hr"],
                    alpha=0.4, s=15, color="#2ecc71", edgecolors="none")
    axes[1].set_xlabel("Median Active Time (hrs)")
    axes[1].set_ylabel("Median Time Between Submissions (hrs)")
    axes[1].set_title("Per Student")

    plt.suptitle("Active Time vs Time Between Submissions", fontsize=13, y=1.0)
    plt.tight_layout()
    out = os.path.join(output_dir, "active_time_vs_time_between.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out}")


# ---------------------------------------------------------------------------
# Analysis: section-metrics
# ---------------------------------------------------------------------------

def run_section_metrics():
    output_dir = CLUSTERING_DIR
    os.makedirs(output_dir, exist_ok=True)
    setup_plotting()

    print("Loading data...")
    df = pd.read_csv(f"{CACHE_PATH}/main_data.csv", low_memory=False, on_bad_lines="skip")
    sections = pd.read_csv(f"{CACHE_PATH}/section_infos.csv")
    courses = pd.read_csv(f"{CACHE_PATH}/course_infos.csv")
    sections = sections.merge(courses, on="course_id")
    dsa_sections = sections[sections["course_name"].str.startswith("dsa")].copy()
    dsa_sections["section_prefix"] = dsa_sections["section_name"].str.extract(r"^([A-Z]+)")
    dsa_sections["section_prefix"] = dsa_sections["section_prefix"].replace("CN", "CC")
    df = df.merge(dsa_sections[["section_id", "section_prefix"]], on="section_id", how="inner")

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FMT, errors="coerce")
    df = df.dropna(subset=["timestamp_dt"])
    submits = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submits = submits.sort_values(["student_id", "question_unittest_id", "timestamp_dt"])
    submits["marks"] = submits["pass"].apply(compute_marks)

    student_section = df.groupby("student_id")["section_prefix"].first()
    records = []
    for sid, sdata in submits.groupby("student_id"):
        prefix = student_section.get(sid)
        if prefix not in SECTION_GROUPS:
            continue
        valid_marks = sdata["marks"].dropna()
        questions_attempted = sdata["question_unittest_id"].nunique()
        total_subs = len(sdata)
        subs_per_q = total_subs / max(questions_attempted, 1)

        if len(valid_marks) > 0:
            perfect_rate = (valid_marks == 10).mean()
            mean_marks = valid_marks.mean()
            first_attempt_marks = []
            for qid, qgroup in sdata.groupby("question_unittest_id"):
                m = qgroup["marks"].dropna()
                if len(m) > 0:
                    first_attempt_marks.append(m.iloc[0])
            first_attempt_avg = np.mean(first_attempt_marks) if first_attempt_marks else np.nan
        else:
            perfect_rate = mean_marks = first_attempt_avg = np.nan

        all_actions = df[df["student_id"] == sid]
        n_prechecks = (all_actions["response_type"] == "Prechecked").sum()
        n_submits_total = all_actions["response_type"].isin(["Submit", "Prechecked"]).sum()
        precheck_rate = n_prechecks / max(n_submits_total, 1)

        timestamps = sdata["timestamp_dt"].tolist()
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 60.0
                          for i in range(1, len(timestamps))]
            median_time_between = np.median(time_diffs)
        else:
            median_time_between = np.nan

        improvements = []
        for qid, qgroup in sdata.groupby("question_unittest_id"):
            m = qgroup["marks"].dropna()
            if len(m) >= 2:
                improvements.append(m.iloc[-1] - m.iloc[0])
        mean_improvement = np.mean(improvements) if improvements else np.nan

        solved = sum(1 for _, qg in sdata.groupby("question_unittest_id")
                     if qg["marks"].dropna().max() == 10) if len(valid_marks) > 0 else 0
        solve_rate = solved / max(questions_attempted, 1)

        attempts_to_solve = []
        for qid, qgroup in sdata.groupby("question_unittest_id"):
            m = qgroup["marks"].dropna().tolist()
            for i, mark in enumerate(m):
                if mark == 10:
                    attempts_to_solve.append(i + 1)
                    break
        mean_attempts_to_solve = np.mean(attempts_to_solve) if attempts_to_solve else np.nan

        records.append({
            "student_id": sid, "section_prefix": prefix,
            "total_submissions": total_subs, "questions_attempted": questions_attempted,
            "subs_per_question": subs_per_q, "perfect_rate": perfect_rate,
            "mean_marks": mean_marks, "first_attempt_avg": first_attempt_avg,
            "precheck_rate": precheck_rate, "median_time_between_min": median_time_between,
            "mean_improvement": mean_improvement, "solve_rate": solve_rate,
            "mean_attempts_to_solve": mean_attempts_to_solve,
        })

    students = pd.DataFrame(records)
    print(f"Students: {len(students)}")

    metrics = [
        ("total_submissions", "Total Submissions"),
        ("questions_attempted", "Questions Attempted"),
        ("subs_per_question", "Submissions per Question"),
        ("perfect_rate", "Perfect Score Rate"),
        ("mean_marks", "Mean Marks"),
        ("first_attempt_avg", "First Attempt Avg Marks"),
        ("precheck_rate", "Precheck Usage Rate"),
        ("median_time_between_min", "Median Time Between (min)"),
        ("mean_improvement", "Mean Improvement (last - first)"),
        ("solve_rate", "Solve Rate"),
        ("mean_attempts_to_solve", "Mean Attempts to Solve"),
    ]

    print("\n=== Kruskal-Wallis H test (L vs CC vs DT) ===")
    ranked = []
    for col, label in metrics:
        groups = [students[students["section_prefix"] == p][col].dropna() for p in SECTION_GROUPS]
        if all(len(g) > 5 for g in groups):
            h, p = stats.kruskal(*groups)
            medians = [g.median() for g in groups]
            print(f"  {label:<35} H={h:>8.2f}  p={p:.2e}  L={medians[0]:.2f} CC={medians[1]:.2f} DT={medians[2]:.2f}")
            ranked.append((col, label, h, p))
    ranked.sort(key=lambda x: x[2], reverse=True)

    top_metrics = ranked[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for idx, (col, label, h, p) in enumerate(top_metrics):
        ax = axes[idx]
        data = [students[students["section_prefix"] == pf][col].dropna() for pf in SECTION_GROUPS]
        bp = ax.boxplot(data, tick_labels=[SECTION_LABELS[p] for p in SECTION_GROUPS],
                        patch_artist=True, widths=0.6)
        for patch, pf in zip(bp["boxes"], SECTION_GROUPS):
            patch.set_facecolor(SECTION_COLORS[pf])
            patch.set_alpha(0.7)
        ax.set_title(f"{label}\n(H={h:.1f}, p={p:.1e})")
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Top Distinguishing Metrics Across Section Types", fontsize=13, y=1.0)
    plt.tight_layout()
    out = os.path.join(output_dir, "section_metric_exploration.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Box plots saved: {out}")

    # Cross-metric scatters
    students["median_time_between_min"] = students["median_time_between_min"].clip(lower=0)
    cross_pairs = [
        ("total_submissions", "first_attempt_avg"),
        ("subs_per_question", "solve_rate"),
        ("total_submissions", "solve_rate"),
        ("first_attempt_avg", "mean_attempts_to_solve"),
        ("median_time_between_min", "first_attempt_avg"),
        ("questions_attempted", "mean_improvement"),
        ("precheck_rate", "solve_rate"),
        ("subs_per_question", "mean_improvement"),
        ("total_submissions", "mean_marks"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for idx, (col_x, col_y) in enumerate(cross_pairs):
        ax = axes.flatten()[idx]
        for prefix in SECTION_GROUPS:
            sd = students[students["section_prefix"] == prefix]
            ax.scatter(sd[col_x], sd[col_y], color=SECTION_COLORS[prefix],
                       alpha=0.5, s=20, edgecolors="none", label=SECTION_LABELS[prefix])
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        if idx == 0:
            ax.legend(loc="best", fontsize=6)

    plt.suptitle("Cross-Metric Scatter: Engagement vs Performance by Section", fontsize=13, y=1.0)
    plt.tight_layout()
    out2 = os.path.join(output_dir, "section_metric_cross_scatters.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Cross-metric scatter saved: {out2}")


# ---------------------------------------------------------------------------
# Analysis: section-curves
# ---------------------------------------------------------------------------

def _build_padded_trajectories(submits, group_col, groups, max_attempts=30):
    padded = {g: [] for g in groups}
    for (sid, qid), group in submits.groupby(["student_id", "question_unittest_id"]):
        g = group[group_col].iloc[0]
        if g not in padded:
            continue
        marks = group.sort_values("attempt_num")["marks"].tolist()
        if not marks:
            continue
        best = max(marks)
        p = marks[:max_attempts]
        p += [best] * (max_attempts - len(p))
        padded[g].append(p)
    return padded


def _load_dsa_section_data():
    df = pd.read_csv(f"{CACHE_PATH}/main_data.csv", low_memory=False, on_bad_lines="skip")
    sections = pd.read_csv(f"{CACHE_PATH}/section_infos.csv")
    courses = pd.read_csv(f"{CACHE_PATH}/course_infos.csv")
    sections = sections.merge(courses, on="course_id")
    sections["section_prefix"] = sections["section_name"].str.extract(r"^([A-Z]+)")
    sections["section_prefix"] = sections["section_prefix"].replace("CN", "CC")
    df = df.merge(sections[["section_id", "section_prefix", "course_name"]], on="section_id", how="inner")

    submits = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submits["timestamp_dt"] = pd.to_datetime(submits["timestamp"], format=TIMESTAMP_FMT, errors="coerce")
    submits = submits.dropna(subset=["timestamp_dt"])
    submits = submits.sort_values(["student_id", "question_unittest_id", "timestamp_dt"])
    submits["marks"] = submits["pass"].apply(compute_marks)
    submits = submits.dropna(subset=["marks"])
    submits["attempt_num"] = submits.groupby(["student_id", "question_unittest_id"]).cumcount() + 1
    return submits


def run_section_curves():
    output_dir = CLUSTERING_DIR
    os.makedirs(output_dir, exist_ok=True)
    setup_plotting()

    print("Loading data...")
    submits = _load_dsa_section_data()
    dsa_data = submits[submits["course_name"].str.startswith("dsa")]
    print(f"Total scored submissions: {len(dsa_data):,}")

    max_attempts = 30
    padded_by_section = _build_padded_trajectories(dsa_data, "section_prefix", SECTION_GROUPS, max_attempts)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    x = np.arange(1, max_attempts + 1)

    np.random.seed(42)
    for prefix in SECTION_GROUPS:
        all_padded = padded_by_section[prefix]
        if not all_padded:
            continue
        sample_idx = np.random.choice(len(all_padded), size=min(150, len(all_padded)), replace=False)
        for i in sample_idx:
            ax.plot(x, all_padded[i], color=SECTION_COLORS[prefix], alpha=0.1, linewidth=0.6)

    for prefix in SECTION_GROUPS:
        arr = np.array(padded_by_section[prefix])
        if len(arr) == 0:
            continue
        means = np.mean(arr, axis=0)
        n = dsa_data[dsa_data["section_prefix"] == prefix]["student_id"].nunique()
        ax.plot(x, means, color=SECTION_COLORS[prefix], linewidth=2.5,
                linestyle="--", label=f"{SECTION_LABELS[prefix]} (n={n})", zorder=9)

    ax.set_xlabel("Attempts")
    ax.set_ylabel("Marks")
    ax.set_title("Student Learning Trajectories by Section Type (DSA)")
    ax.legend(loc="lower right")
    ax.set_xlim(1, max_attempts)
    ax.set_ylim(0, 10)

    out = os.path.join(output_dir, "section_learning_curves.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out}")


# ---------------------------------------------------------------------------
# Analysis: section-curves-course
# ---------------------------------------------------------------------------

def run_section_curves_course():
    output_dir = CLUSTERING_DIR
    os.makedirs(output_dir, exist_ok=True)
    setup_plotting()
    plt.rcParams.update({"legend.fontsize": 7})

    print("Loading data...")
    submits = _load_dsa_section_data()
    dsa_data = submits[submits["course_name"].str.startswith("dsa")]
    print(f"Total scored submissions: {len(dsa_data):,}")

    max_attempts = 30
    x = np.arange(1, max_attempts + 1)

    course_titles = {"dsa_hk231": "DSA Fall 2023", "dsa_hk221": "DSA Fall 2022"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Combined DSA
    groups = [g for g in SECTION_GROUPS if g in dsa_data["section_prefix"].values]
    padded = _build_padded_trajectories(dsa_data, "section_prefix", groups, max_attempts)
    np.random.seed(42)
    for g in groups:
        if not padded[g]:
            continue
        sample_idx = np.random.choice(len(padded[g]), size=min(150, len(padded[g])), replace=False)
        for i in sample_idx:
            axes[0].plot(x, padded[g][i], color=SECTION_COLORS[g], alpha=0.1, linewidth=0.5)
    for g in groups:
        arr = np.array(padded[g])
        if len(arr) == 0:
            continue
        n = dsa_data[dsa_data["section_prefix"] == g]["student_id"].nunique()
        axes[0].plot(x, np.mean(arr, axis=0), color=SECTION_COLORS[g], linewidth=2.5,
                     linestyle="--", label=f"{SECTION_LABELS[g]} (n={n})", zorder=9)
    axes[0].set_title("DSA Combined")
    axes[0].set_xlabel("Attempts")
    axes[0].set_ylabel("Marks")
    axes[0].legend(loc="lower right")
    axes[0].set_xlim(1, max_attempts)
    axes[0].set_ylim(0, 10)

    for idx, course_name in enumerate(["dsa_hk231", "dsa_hk221"]):
        ax = axes[idx + 1]
        course_data = dsa_data[dsa_data["course_name"] == course_name]
        cgroups = [g for g in SECTION_GROUPS if g in course_data["section_prefix"].values]
        padded = _build_padded_trajectories(course_data, "section_prefix", cgroups, max_attempts)
        np.random.seed(42)
        for g in cgroups:
            if not padded[g]:
                continue
            sample_idx = np.random.choice(len(padded[g]), size=min(150, len(padded[g])), replace=False)
            for i in sample_idx:
                ax.plot(x, padded[g][i], color=SECTION_COLORS[g], alpha=0.1, linewidth=0.5)
        for g in cgroups:
            arr = np.array(padded[g])
            if len(arr) == 0:
                continue
            n = course_data[course_data["section_prefix"] == g]["student_id"].nunique()
            ax.plot(x, np.mean(arr, axis=0), color=SECTION_COLORS[g], linewidth=2.5,
                    linestyle="--", label=f"{SECTION_LABELS[g]} (n={n})", zorder=9)
        ax.set_title(course_titles[course_name])
        ax.set_xlabel("Attempts")
        ax.set_ylabel("Marks")
        ax.legend(loc="lower right")
        ax.set_xlim(1, max_attempts)
        ax.set_ylim(0, 10)

    plt.tight_layout()
    out = os.path.join(output_dir, "section_learning_curves_by_course.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out}")


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------

ANALYSES = {
    "year-comparison": ("Metric gaps on shared questions (2022 vs 2023)", run_year_comparison),
    "ai-detection": ("Trajectory-level AI usage signal features", run_ai_detection),
    "ai-impact": ("7-pronged investigation of DSA score jump", run_ai_impact),
    "cheating": ("Suspicious submission pair detection", run_cheating),
    "edit-distance": ("Edit distance distribution comparison", run_edit_distance),
    "active-time": ("Active time vs time between submissions", run_active_time),
    "section-metrics": ("Kruskal-Wallis behavioral metrics by section type", run_section_metrics),
    "section-curves": ("Learning trajectories by section type", run_section_curves),
    "section-curves-course": ("Section learning curves split by course", run_section_curves_course),
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified exploratory analysis for CodeInsight dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python data_analysis/exploratory_analysis.py ai-impact\n"
               "  python data_analysis/exploratory_analysis.py all\n"
               "  python data_analysis/exploratory_analysis.py --list",
    )
    parser.add_argument(
        "analysis", nargs="?", default=None,
        choices=list(ANALYSES.keys()) + ["all"],
        help="Which analysis to run (or 'all' to run everything)",
    )
    parser.add_argument("--list", action="store_true", help="List available analyses")
    args = parser.parse_args()

    if args.list or args.analysis is None:
        print("Available analyses:\n")
        for name, (desc, _) in ANALYSES.items():
            print(f"  {name:<25} {desc}")
        print(f"\n  {'all':<25} Run all analyses sequentially")
        return

    if args.analysis == "all":
        for name, (desc, func) in ANALYSES.items():
            print(f"\n{'='*70}")
            print(f"Running: {name} - {desc}")
            print(f"{'='*70}\n")
            func()
    else:
        _, func = ANALYSES[args.analysis]
        func()


if __name__ == "__main__":
    main()
