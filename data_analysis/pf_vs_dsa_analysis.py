"""Compare PF vs DSA courses to show why PF is not suitable for
studying iterative problem-solving models."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

plt.rcParams.update(bundles.icml2022())
plt.rcParams["text.usetex"] = True

from dynamic_models.temporal_eval.data_loader import load_student_split_data

COLORS = {"pf_hk232": "#aa3377", "pf_hk222": "#cc6699", "dsa_hk231": "#4477aa", "dsa_hk221": "#ee6677"}
LABELS = {"pf_hk232": "PF HK232", "pf_hk222": "PF HK222", "dsa_hk231": "DSA HK231", "dsa_hk221": "DSA HK221"}

OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "rssm_analysis", "report")


def load_course(course):
    data, _ = load_student_split_data(course, max_attempts=10, test_frac=0.3, seed=42)
    return data.correctness_matrix.numpy()


def savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    courses = ["pf_hk232", "pf_hk222", "dsa_hk231", "dsa_hk221"]

    all_data = {}
    for course in courses:
        corr = load_course(course)
        n_students, n_items, _ = corr.shape

        # Student pass rates
        student_prs = []
        for s in range(n_students):
            sv = corr[s][corr[s] != -1]
            if len(sv) > 0:
                student_prs.append(sv.mean())

        # Question pass rates
        q_prs = []
        for q in range(n_items):
            v = corr[:, q, :] != -1
            if v.sum() > 0:
                q_prs.append(corr[:, q, :][v].mean())

        # Pass rate by attempt
        pass_by_att = {}
        for a in range(10):
            v = corr[:, :, a] != -1
            if v.sum() > 0:
                pass_by_att[a + 1] = corr[:, :, a][v].mean()

        # Avg attempts per student-item
        avg_attempts = []
        for s in range(n_students):
            per_item = []
            for q in range(n_items):
                nv = (corr[s, q] != -1).sum()
                if nv > 0:
                    per_item.append(nv)
            if per_item:
                avg_attempts.append(np.mean(per_item))

        # Floor effect: always-fail trajectories
        always_fail = 0
        always_pass = 0
        mixed = 0
        for s in range(n_students):
            for q in range(n_items):
                vals = corr[s, q]
                vals = vals[vals != -1]
                if len(vals) >= 2:
                    if vals.sum() == 0:
                        always_fail += 1
                    elif vals.mean() == 1.0:
                        always_pass += 1
                    else:
                        mixed += 1

        # Improvement trajectories
        improving = 0
        declining = 0
        flat = 0
        total_traj = 0
        for s in range(n_students):
            for q in range(n_items):
                vals = corr[s, q]
                vals = vals[vals != -1]
                if len(vals) >= 3:
                    total_traj += 1
                    mid = len(vals) // 2
                    diff = vals[mid:].mean() - vals[:mid].mean()
                    if diff > 0.05:
                        improving += 1
                    elif diff < -0.05:
                        declining += 1
                    else:
                        flat += 1

        all_data[course] = {
            "n_students": n_students,
            "n_items": n_items,
            "student_prs": np.array(student_prs),
            "q_prs": np.array(q_prs),
            "pass_by_att": pass_by_att,
            "avg_attempts": np.array(avg_attempts),
            "always_fail": always_fail,
            "always_pass": always_pass,
            "mixed": mixed,
            "improving": improving,
            "declining": declining,
            "flat": flat,
            "total_traj": total_traj,
        }

    # ============================================================
    # Figure 1: Student pass rate distributions
    # ============================================================
    print("Figure: Student pass rate distributions")
    fig, ax = plt.subplots(figsize=(4, 2.8))
    bins = np.arange(0, 0.85, 0.025)
    for course in courses:
        d = all_data[course]
        ax.hist(d["student_prs"], bins=bins, alpha=0.5, color=COLORS[course],
                edgecolor="white", linewidth=0.3, label=LABELS[course], density=True)
        ax.axvline(d["student_prs"].mean(), color=COLORS[course], linestyle="--", linewidth=1)
    ax.set_xlabel("Student Pass Rate")
    ax.set_ylabel("Density")
    ax.set_title("Student Pass Rate Distribution")
    ax.legend(fontsize=5)
    savefig(fig, "pf_student_passrate_dist.png")

    # ============================================================
    # Figure 2: Question difficulty distributions
    # ============================================================
    print("Figure: Question difficulty distributions")
    fig, ax = plt.subplots(figsize=(4, 2.8))
    bins = np.arange(0, 1.05, 0.05)
    for course in courses:
        d = all_data[course]
        ax.hist(d["q_prs"], bins=bins, alpha=0.5, color=COLORS[course],
                edgecolor="white", linewidth=0.3, label=LABELS[course], density=True)
    ax.set_xlabel("Problem Pass Rate")
    ax.set_ylabel("Density")
    ax.set_title("Problem Difficulty Distribution")
    ax.legend(fontsize=5)
    savefig(fig, "pf_question_difficulty_dist.png")

    # ============================================================
    # Figure 3: Pass rate by attempt
    # ============================================================
    print("Figure: Pass rate by attempt")
    fig, ax = plt.subplots(figsize=(4, 2.8))
    for course in courses:
        d = all_data[course]
        atts = sorted(d["pass_by_att"].keys())
        prs = [d["pass_by_att"][a] for a in atts]
        ax.plot(atts, prs, color=COLORS[course], marker="o", markersize=3,
                linewidth=1.2, label=LABELS[course])
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Pass Rate")
    ax.set_title("Pass Rate by Attempt")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=5)
    savefig(fig, "pf_passrate_by_attempt.png")

    # ============================================================
    # Figure 4: Floor effect — always-fail trajectories
    # ============================================================
    print("Figure: Trajectory outcome breakdown")
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    x = np.arange(len(courses))
    w = 0.25
    for i, course in enumerate(courses):
        d = all_data[course]
        total = d["always_fail"] + d["always_pass"] + d["mixed"]
        vals = [d["always_fail"] / total, d["mixed"] / total, d["always_pass"] / total]
        bottom = 0
        cat_colors = ["#ee6677", "#ccbb44", "#228833"]
        cat_labels = ["Always Fail", "Mixed", "Always Pass"]
        for v, cc, cl in zip(vals, cat_colors, cat_labels):
            ax.bar(i, v, w * 3, bottom=bottom, color=cc, alpha=0.8,
                   label=cl if i == 0 else None)
            if v > 0.05:
                ax.text(i, bottom + v / 2, f"{v:.0%}", ha="center", fontsize=5.5, va="center")
            bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in courses], fontsize=7)
    ax.set_ylabel("Fraction of Trajectories")
    ax.set_title("Trajectory Outcomes (2+ Attempts)")
    ax.legend(fontsize=5, loc="upper right")
    savefig(fig, "pf_trajectory_outcomes.png")

    # ============================================================
    # Figure 5: Improvement pattern breakdown
    # ============================================================
    print("Figure: Improvement patterns")
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    x = np.arange(len(courses))
    for i, course in enumerate(courses):
        d = all_data[course]
        t = d["total_traj"]
        vals = [d["improving"] / t, d["flat"] / t, d["declining"] / t]
        bottom = 0
        cat_colors = ["#228833", "#ccbb44", "#ee6677"]
        cat_labels = ["Improving", "Flat", "Declining"]
        for v, cc, cl in zip(vals, cat_colors, cat_labels):
            ax.bar(i, v, w * 3, bottom=bottom, color=cc, alpha=0.8,
                   label=cl if i == 0 else None)
            if v > 0.05:
                ax.text(i, bottom + v / 2, f"{v:.0%}", ha="center", fontsize=5.5, va="center")
            bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in courses], fontsize=7)
    ax.set_ylabel("Fraction of Trajectories")
    ax.set_title("Improvement Patterns (3+ Attempts)")
    ax.legend(fontsize=5, loc="upper right")
    savefig(fig, "pf_improvement_patterns.png")

    # ============================================================
    # Figure 6: Attempt intensity
    # ============================================================
    print("Figure: Attempt intensity")
    fig, ax = plt.subplots(figsize=(4, 2.8))
    bins = np.arange(1, 8.5, 0.5)
    for course in courses:
        d = all_data[course]
        ax.hist(d["avg_attempts"], bins=bins, alpha=0.5, color=COLORS[course],
                edgecolor="white", linewidth=0.3,
                label=f"{LABELS[course]} ($\\mu$={d['avg_attempts'].mean():.2f})", density=True)
    ax.set_xlabel("Avg. Attempts per Problem (per Student)")
    ax.set_ylabel("Density")
    ax.set_title("Student Attempt Intensity")
    ax.legend(fontsize=5)
    savefig(fig, "pf_attempt_intensity.png")

    # ============================================================
    # Figure 7: Comprehensive summary panel
    # ============================================================
    print("Figure: Summary comparison")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))

    # (0,0) Student pass rate
    ax = axes[0, 0]
    bins = np.arange(0, 0.85, 0.025)
    for course in courses:
        d = all_data[course]
        ax.hist(d["student_prs"], bins=bins, alpha=0.5, color=COLORS[course],
                edgecolor="white", linewidth=0.3, label=LABELS[course], density=True)
    ax.set_xlabel("Student Pass Rate")
    ax.set_ylabel("Density")
    ax.set_title("Student Ability Distribution")
    ax.legend(fontsize=4.5)

    # (0,1) Question difficulty
    ax = axes[0, 1]
    bins = np.arange(0, 1.05, 0.05)
    for course in courses:
        d = all_data[course]
        ax.hist(d["q_prs"], bins=bins, alpha=0.5, color=COLORS[course],
                edgecolor="white", linewidth=0.3, label=LABELS[course], density=True)
    ax.set_xlabel("Problem Pass Rate")
    ax.set_title("Problem Difficulty")

    # (0,2) Pass rate by attempt
    ax = axes[0, 2]
    for course in courses:
        d = all_data[course]
        atts = sorted(d["pass_by_att"].keys())
        prs = [d["pass_by_att"][a] for a in atts]
        ax.plot(atts, prs, color=COLORS[course], marker="o", markersize=3,
                linewidth=1.2, label=LABELS[course])
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Pass Rate")
    ax.set_title("Learning Curve")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=4.5)

    # (1,0) Trajectory outcomes
    ax = axes[1, 0]
    x = np.arange(len(courses))
    w = 0.25
    for i, course in enumerate(courses):
        d = all_data[course]
        total = d["always_fail"] + d["always_pass"] + d["mixed"]
        vals = [d["always_fail"] / total, d["mixed"] / total, d["always_pass"] / total]
        bottom = 0
        cat_colors = ["#ee6677", "#ccbb44", "#228833"]
        cat_labels = ["Always Fail", "Mixed", "Always Pass"]
        for v, cc, cl in zip(vals, cat_colors, cat_labels):
            ax.bar(i, v, w * 3, bottom=bottom, color=cc, alpha=0.8,
                   label=cl if i == 0 else None)
            if v > 0.05:
                ax.text(i, bottom + v / 2, f"{v:.0%}", ha="center", fontsize=5, va="center")
            bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in courses], fontsize=6)
    ax.set_title("Trajectory Outcomes")
    ax.legend(fontsize=4.5)

    # (1,1) Improvement patterns
    ax = axes[1, 1]
    for i, course in enumerate(courses):
        d = all_data[course]
        t = d["total_traj"]
        vals = [d["improving"] / t, d["flat"] / t, d["declining"] / t]
        bottom = 0
        cat_colors = ["#228833", "#ccbb44", "#ee6677"]
        cat_labels = ["Improving", "Flat", "Declining"]
        for v, cc, cl in zip(vals, cat_colors, cat_labels):
            ax.bar(i, v, w * 3, bottom=bottom, color=cc, alpha=0.8,
                   label=cl if i == 0 else None)
            if v > 0.05:
                ax.text(i, bottom + v / 2, f"{v:.0%}", ha="center", fontsize=5, va="center")
            bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in courses], fontsize=6)
    ax.set_title("Improvement Patterns")
    ax.legend(fontsize=4.5)

    # (1,2) Summary stats
    ax = axes[1, 2]
    stats = ["Student\nPass Rate", "Student\nAbility Std", "Frac\nAlways Fail", "Frac\nImproving"]
    x_s = np.arange(len(stats))
    n_c = len(courses)
    w_s = 0.8 / n_c
    for i, course in enumerate(courses):
        d = all_data[course]
        total_multi = d["always_fail"] + d["always_pass"] + d["mixed"]
        vals = [
            d["student_prs"].mean(),
            d["student_prs"].std(),
            d["always_fail"] / total_multi,
            d["improving"] / d["total_traj"],
        ]
        ax.bar(x_s + (i - n_c/2 + 0.5) * w_s, vals, w_s, color=COLORS[course], alpha=0.8, label=LABELS[course])
    ax.set_xticks(x_s)
    ax.set_xticklabels(stats, fontsize=5.5)
    ax.set_title("Summary Statistics")
    ax.legend(fontsize=4.5)

    fig.suptitle("PF vs DSA: Course Characteristics for Problem-Solving Analysis", y=1.02, fontsize=8)
    fig.tight_layout()
    savefig(fig, "pf_vs_dsa_summary.png")

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Metric':<35} {'PF HK232':>12} {'PF HK222':>12} {'DSA HK231':>12} {'DSA HK221':>12}")
    print("-" * 87)

    rows = [
        ("Students", [all_data[c]["n_students"] for c in courses]),
        ("Items (testcases)", [all_data[c]["n_items"] for c in courses]),
        ("Mean student pass rate", [all_data[c]["student_prs"].mean() for c in courses]),
        ("Student ability std", [all_data[c]["student_prs"].std() for c in courses]),
        ("Student ability IQR width", [np.percentile(all_data[c]["student_prs"], 75) - np.percentile(all_data[c]["student_prs"], 25) for c in courses]),
        ("Mean question pass rate", [all_data[c]["q_prs"].mean() for c in courses]),
        ("Questions <10% pass rate", [f"{(all_data[c]['q_prs'] < 0.10).mean()*100:.1f}%" for c in courses]),
        ("Avg attempts per item", [all_data[c]["avg_attempts"].mean() for c in courses]),
        ("Always-fail trajectories", [f"{all_data[c]['always_fail']/(all_data[c]['always_fail']+all_data[c]['always_pass']+all_data[c]['mixed'])*100:.1f}%" for c in courses]),
        ("Improving trajectories", [f"{all_data[c]['improving']/all_data[c]['total_traj']*100:.1f}%" for c in courses]),
        ("Flat trajectories", [f"{all_data[c]['flat']/all_data[c]['total_traj']*100:.1f}%" for c in courses]),
        ("Pass rate change (att 1→10)", [f"{list(all_data[c]['pass_by_att'].values())[-1] - list(all_data[c]['pass_by_att'].values())[0]:.4f}" for c in courses]),
    ]
    for label, vals in rows:
        parts = [f"{label:<35}"]
        for v in vals:
            if isinstance(v, float):
                parts.append(f"{v:>12.4f}")
            else:
                parts.append(f"{str(v):>12}")
        print("".join(parts))


if __name__ == "__main__":
    main()
