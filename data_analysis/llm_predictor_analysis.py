"""Deep-dive analysis of LLM prediction data.

5 investigations into where the LLM predictor succeeds, fails, and why.
Produces detailed markdown files for each investigation plus figures.
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from scipy.stats import kendalltau, pearsonr, spearmanr


def _pass_fraction(s):
    s = str(s).strip()
    if not s or s == "nan":
        return np.nan
    return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan


def load_data():
    sim_dir = snapshot_download(
        repo_id="CodeInsightTeam/simulation_output",
        repo_type="dataset", local_files_only=True,
    )
    csv_dir = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        repo_type="dataset", local_files_only=True,
    )
    merged_path = os.path.join(sim_dir, "v4_profile_mindiff", "glm_v4_merged.jsonl")
    keep = ["student_id", "question_unittest_id", "attempt_id", "response_type", "pass"]
    rows = []
    with open(merged_path) as f:
        for line in f:
            rec = json.loads(line)
            rows.append({k: rec.get(k) for k in keep})
    sim_df = pd.DataFrame(rows)
    sim_df["student_id"] = sim_df["student_id"].astype(str)
    sim_df["question_unittest_id"] = pd.to_numeric(sim_df["question_unittest_id"], errors="coerce")
    sim_df["attempt_id"] = pd.to_numeric(sim_df["attempt_id"], errors="coerce")
    sim_df = sim_df.dropna(subset=["question_unittest_id"])
    sim_df["question_unittest_id"] = sim_df["question_unittest_id"].astype(int)
    sim_df["score"] = sim_df["pass"].apply(_pass_fraction)

    real_df = pd.read_csv(
        os.path.join(csv_dir, "main_data.csv"), low_memory=False, on_bad_lines="skip",
    )
    real_df = real_df[real_df["response_type"] == "Submit"].copy()
    real_df = real_df.dropna(subset=["pass"])
    real_df["student_id"] = real_df["student_id"].astype(str)
    real_df["question_unittest_id"] = pd.to_numeric(real_df["question_unittest_id"], errors="coerce")
    real_df = real_df.dropna(subset=["question_unittest_id"])
    real_df["question_unittest_id"] = real_df["question_unittest_id"].astype(int)
    real_df["score"] = real_df["pass"].apply(_pass_fraction)
    real_df = real_df.dropna(subset=["score"])
    return sim_df, real_df


def compute_merged(sim_df, real_df):
    sim_sub = sim_df[sim_df["response_type"] == "Submit"].sort_values("attempt_id")
    sim_last = sim_sub.groupby(["student_id", "question_unittest_id"]).last().reset_index()
    sim_last["y_pred"] = sim_last["score"]
    sim_last = sim_last.dropna(subset=["y_pred"])

    real_sub = real_df.copy()
    real_sub["attempt_id"] = pd.to_numeric(real_sub.get("attempt_id", 0), errors="coerce").fillna(0)
    real_sub = real_sub.sort_values("attempt_id")
    real_last = real_sub.groupby(["student_id", "question_unittest_id"]).last().reset_index()
    real_last["y_true"] = real_last["score"]

    merged = sim_last[["student_id", "question_unittest_id", "y_pred"]].merge(
        real_last[["student_id", "question_unittest_id", "y_true"]],
        on=["student_id", "question_unittest_id"], how="inner",
    )
    return merged


def compute_question_level(sim_df, real_df):
    sim_sub = sim_df[sim_df["response_type"] == "Submit"].dropna(subset=["score"])
    sim_q = sim_sub.groupby("question_unittest_id").agg(
        sim_pass_rate=("score", lambda x: (x >= 1.0).mean()),
        sim_mean_score=("score", "mean"),
        sim_n=("score", "count"),
    ).reset_index()

    real_q = real_df.dropna(subset=["score"]).groupby("question_unittest_id").agg(
        real_pass_rate=("score", lambda x: (x >= 1.0).mean()),
        real_mean_score=("score", "mean"),
        real_n=("score", "count"),
    ).reset_index()

    q_merged = sim_q.merge(real_q, on="question_unittest_id", how="inner")
    q_merged = q_merged[(q_merged["sim_n"] >= 5) & (q_merged["real_n"] >= 5)]
    return q_merged

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.facecolor": "white",
})

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "llm_predictor"
)
os.makedirs(OUT_DIR, exist_ok=True)



def load_question_infos():
    csv_dir = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        repo_type="dataset", local_files_only=True,
    )
    qi = pd.read_csv(os.path.join(csv_dir, "question_infos.csv"))
    qi = qi.rename(columns={"question_id": "question_unittest_id"})
    qi["n_testcases"] = qi["question_unittests"].fillna("").apply(
        lambda s: s.count("Unittest ")
    )
    return qi


def enrich_merged(merged, real_df, q_info):
    course_map = real_df[["student_id", "question_unittest_id", "course_id"]].drop_duplicates()
    merged = merged.merge(course_map, on=["student_id", "question_unittest_id"], how="left")
    q_cols = ["question_unittest_id", "course_id", "week", "topic", "question_name", "n_testcases"]
    q_sub = q_info[q_cols].drop_duplicates(subset=["question_unittest_id"])
    merged = merged.merge(
        q_sub.rename(columns={"course_id": "q_course_id"}),
        on="question_unittest_id", how="left",
    )
    merged["course_id"] = merged["course_id"].fillna(merged["q_course_id"])
    merged = merged.drop(columns=["q_course_id"])
    return merged


# ── Investigation 1: Best/Worst Predicted Questions ──────────────────────

def inv1_best_worst_questions(merged, q_info):
    q_stats = []
    for qid, grp in merged.groupby("question_unittest_id"):
        if len(grp) < 10:
            continue
        if grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        r, _ = pearsonr(grp["y_pred"], grp["y_true"])
        q_stats.append({
            "question_unittest_id": qid,
            "pearson_r": r,
            "n_students": len(grp),
            "mean_y_pred": grp["y_pred"].mean(),
            "mean_y_true": grp["y_true"].mean(),
        })
    qs = pd.DataFrame(q_stats).sort_values("pearson_r", ascending=False)

    q_cols = ["question_unittest_id", "question_name", "topic", "week",
              "course_id", "n_testcases", "question_text"]
    q_sub = q_info[q_cols].drop_duplicates(subset=["question_unittest_id"])
    qs = qs.merge(q_sub, on="question_unittest_id", how="left")

    print(f"\n  Questions with per-question Pearson r: {len(qs)}")
    print(f"  Mean r: {qs['pearson_r'].mean():.4f}, Median r: {qs['pearson_r'].median():.4f}")
    print(f"\n  TOP 15 (best predicted):")
    top = qs.head(15)
    for _, row in top.iterrows():
        print(f"    r={row['pearson_r']:+.3f}  n={int(row['n_students']):>3d}  "
              f"topic={str(row['topic']):<20s}  week={row['week']}  "
              f"name={row['question_name']}")
    print(f"\n  BOTTOM 15 (worst predicted):")
    bot = qs.tail(15)
    for _, row in bot.iterrows():
        print(f"    r={row['pearson_r']:+.3f}  n={int(row['n_students']):>3d}  "
              f"topic={str(row['topic']):<20s}  week={row['week']}  "
              f"name={row['question_name']}")

    # Summary by topic
    print("\n  Mean Pearson r by topic:")
    by_topic = qs.groupby("topic").agg(
        mean_r=("pearson_r", "mean"), n_questions=("pearson_r", "count")
    ).sort_values("mean_r", ascending=False)
    for topic, row in by_topic.iterrows():
        print(f"    {str(topic):<25s}  r={row['mean_r']:+.3f}  (n={int(row['n_questions'])})")

    # Summary by course
    print("\n  Mean Pearson r by course:")
    by_course = qs.groupby("course_id").agg(
        mean_r=("pearson_r", "mean"), n_questions=("pearson_r", "count")
    ).sort_values("mean_r", ascending=False)
    for cid, row in by_course.iterrows():
        print(f"    course {int(cid)}:  r={row['mean_r']:+.3f}  (n={int(row['n_questions'])})")

    # Write markdown
    md = ["# Investigation 1: Per-Question Prediction Quality\n"]
    md.append(f"Questions analyzed: {len(qs)} (>=10 students, non-zero variance on both sides)\n")
    md.append(f"Mean Pearson r: {qs['pearson_r'].mean():.4f}, Median: {qs['pearson_r'].median():.4f}\n")
    md.append("## All Questions (sorted by Pearson r, best first)\n")
    for _, row in qs.iterrows():
        md.append(f"### Q{int(row['question_unittest_id'])} — {row['question_name']}\n")
        md.append(f"- **Pearson r**: {row['pearson_r']:.4f}")
        md.append(f"- **Topic**: {row['topic']}")
        md.append(f"- **Week**: {row['week']}")
        md.append(f"- **Course**: {int(row['course_id']) if pd.notna(row['course_id']) else 'N/A'}")
        md.append(f"- **N students**: {int(row['n_students'])}")
        md.append(f"- **Mean y_pred**: {row['mean_y_pred']:.3f}")
        md.append(f"- **Mean y_true**: {row['mean_y_true']:.3f}")
        md.append(f"- **N testcases**: {int(row['n_testcases']) if pd.notna(row['n_testcases']) else 'N/A'}")
        qtext = str(row.get("question_text", ""))[:2000]
        if qtext and qtext != "nan":
            md.append(f"\n<details><summary>Question Text</summary>\n\n```\n{qtext}\n```\n</details>\n")
        md.append("")

    with open(os.path.join(OUT_DIR, "per_question_prediction_quality.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\n  Saved per_question_prediction_quality.md")

    # Figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(qs["pearson_r"], bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="r = 0")
    ax.axvline(qs["pearson_r"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean = {qs['pearson_r'].mean():.3f}")
    ax.set_xlabel("Per-Question Pearson r")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Distribution of Per-Question Prediction Quality")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "per_question_pearson_r.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return qs


# ── Investigation 3: Real=0, LLM>0 (Overconfident) ──────────────────────

def inv3_overconfident(q_merged, q_info):
    qm = q_merged.merge(
        q_info[["question_unittest_id", "question_name", "topic", "week",
                "course_id", "n_testcases", "question_text",
                "question_template", "question_unittests"]].drop_duplicates(subset=["question_unittest_id"]),
        on="question_unittest_id", how="left",
    )

    overconf = qm[(qm["real_mean_score"] <= 0.05) & (qm["sim_mean_score"] > 0.1)].copy()
    underconf = qm[(qm["sim_mean_score"] <= 0.05) & (qm["real_mean_score"] > 0.1)].copy()

    print(f"\n  Overconfident (real<=0.05, LLM>0.1): {len(overconf)} questions")
    print(f"  Underconfident (LLM<=0.05, real>0.1): {len(underconf)} questions")

    if len(overconf) > 0:
        print(f"\n  Overconfident questions:")
        for _, row in overconf.sort_values("sim_mean_score", ascending=False).iterrows():
            print(f"    Q{int(row['question_unittest_id'])}: sim={row['sim_mean_score']:.3f} "
                  f"real={row['real_mean_score']:.3f}  topic={row['topic']}  "
                  f"week={row['week']}  name={row['question_name']}")

    if len(underconf) > 0:
        print(f"\n  Underconfident questions:")
        for _, row in underconf.sort_values("real_mean_score", ascending=False).iterrows():
            print(f"    Q{int(row['question_unittest_id'])}: sim={row['sim_mean_score']:.3f} "
                  f"real={row['real_mean_score']:.3f}  topic={row['topic']}  "
                  f"week={row['week']}  name={row['question_name']}")

    # Write markdown
    md = ["# Investigation 3: Overconfident & Underconfident Questions\n"]
    md.append(f"Overconfident (real<=0.05, LLM>0.1): {len(overconf)} questions")
    md.append(f"Underconfident (LLM<=0.05, real>0.1): {len(underconf)} questions\n")

    md.append("## Overconfident Questions (LLM solves, students don't)\n")
    for _, row in overconf.sort_values("sim_mean_score", ascending=False).iterrows():
        md.append(f"### Q{int(row['question_unittest_id'])} — {row['question_name']}\n")
        md.append(f"- **LLM mean score**: {row['sim_mean_score']:.3f} (n={int(row['sim_n'])})")
        md.append(f"- **Real mean score**: {row['real_mean_score']:.3f} (n={int(row['real_n'])})")
        md.append(f"- **Topic**: {row['topic']}")
        md.append(f"- **Week**: {row['week']}")
        md.append(f"- **Course**: {int(row['course_id']) if pd.notna(row['course_id']) else 'N/A'}")
        md.append(f"- **N testcases**: {int(row['n_testcases']) if pd.notna(row['n_testcases']) else 'N/A'}")
        qtext = str(row.get("question_text", ""))
        if qtext and qtext != "nan":
            md.append(f"\n<details><summary>Question Text</summary>\n\n```\n{qtext}\n```\n</details>")
        qtempl = str(row.get("question_template", ""))
        if qtempl and qtempl != "nan":
            md.append(f"\n<details><summary>Template</summary>\n\n```cpp\n{qtempl}\n```\n</details>")
        qtests = str(row.get("question_unittests", ""))
        if qtests and qtests != "nan":
            md.append(f"\n<details><summary>Unit Tests</summary>\n\n```\n{qtests[:3000]}\n```\n</details>")
        md.append("")

    md.append("## Underconfident Questions (students solve, LLM doesn't)\n")
    for _, row in underconf.sort_values("real_mean_score", ascending=False).iterrows():
        md.append(f"### Q{int(row['question_unittest_id'])} — {row['question_name']}\n")
        md.append(f"- **LLM mean score**: {row['sim_mean_score']:.3f} (n={int(row['sim_n'])})")
        md.append(f"- **Real mean score**: {row['real_mean_score']:.3f} (n={int(row['real_n'])})")
        md.append(f"- **Topic**: {row['topic']}")
        md.append(f"- **Week**: {row['week']}")
        md.append(f"- **Course**: {int(row['course_id']) if pd.notna(row['course_id']) else 'N/A'}")
        qtext = str(row.get("question_text", ""))
        if qtext and qtext != "nan":
            md.append(f"\n<details><summary>Question Text</summary>\n\n```\n{qtext}\n```\n</details>")
        qtempl = str(row.get("question_template", ""))
        if qtempl and qtempl != "nan":
            md.append(f"\n<details><summary>Template</summary>\n\n```cpp\n{qtempl}\n```\n</details>")
        qtests = str(row.get("question_unittests", ""))
        if qtests and qtests != "nan":
            md.append(f"\n<details><summary>Unit Tests</summary>\n\n```\n{qtests[:3000]}\n```\n</details>")
        md.append("")

    with open(os.path.join(OUT_DIR, "overconfident_questions.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\n  Saved overconfident_questions.md")

    return overconf, underconf


# ── Investigation 4: LLM=0 but Real>0 (LLM Blind Spots) ─────────────────

def inv4_llm_blind_spots(q_merged, q_info):
    qm = q_merged.merge(
        q_info[["question_unittest_id", "question_name", "topic", "week",
                "course_id", "n_testcases", "question_text",
                "question_template", "question_unittests"]].drop_duplicates(subset=["question_unittest_id"]),
        on="question_unittest_id", how="left",
    )

    blind = qm[(qm["sim_mean_score"] <= 0.05) & (qm["real_mean_score"] > 0.05)].copy()
    blind = blind.sort_values("real_mean_score", ascending=False)

    # Compare to questions where LLM does score > 0
    llm_active = qm[qm["sim_mean_score"] > 0.05]

    print(f"\n  LLM blind spots (sim<=0.05, real>0.05): {len(blind)} questions")
    print(f"  Questions where LLM scores >0.05: {len(llm_active)}")
    print(f"\n  Blind spot questions (sorted by real score):")
    for _, row in blind.iterrows():
        print(f"    Q{int(row['question_unittest_id'])}: sim={row['sim_mean_score']:.3f} "
              f"real={row['real_mean_score']:.3f}  topic={row['topic']}  "
              f"week={row['week']}  name={row['question_name']}")

    # Compare distributions
    print(f"\n  Comparison: blind spots vs LLM-active questions:")
    print(f"  {'Metric':<25s} {'Blind (LLM=0)':>15s} {'LLM active':>15s}")
    print(f"  {'─'*25} {'─'*15} {'─'*15}")
    print(f"  {'Mean real score':<25s} {blind['real_mean_score'].mean():>15.3f} {llm_active['real_mean_score'].mean():>15.3f}")
    print(f"  {'Mean n_testcases':<25s} {blind['n_testcases'].mean():>15.1f} {llm_active['n_testcases'].mean():>15.1f}")
    print(f"  {'Mean real_n (students)':<25s} {blind['real_n'].mean():>15.1f} {llm_active['real_n'].mean():>15.1f}")
    print(f"  {'Mean week':<25s} {blind['week'].mean():>15.1f} {llm_active['week'].mean():>15.1f}")

    # Topic breakdown
    print(f"\n  Topic distribution:")
    blind_topics = blind["topic"].value_counts()
    active_topics = llm_active["topic"].value_counts()
    all_topics = sorted(set(blind_topics.index) | set(active_topics.index))
    for topic in sorted(blind_topics.index, key=lambda t: -blind_topics.get(t, 0)):
        b = blind_topics.get(topic, 0)
        a = active_topics.get(topic, 0)
        print(f"    {str(topic):<30s}  blind={b:>3d}  active={a:>3d}")

    # Course breakdown
    print(f"\n  Course distribution:")
    for cid in sorted(qm["course_id"].dropna().unique()):
        b = len(blind[blind["course_id"] == cid])
        a = len(llm_active[llm_active["course_id"] == cid])
        total = len(qm[qm["course_id"] == cid])
        print(f"    Course {int(cid)}: blind={b}/{total} ({100*b/max(total,1):.1f}%), "
              f"active={a}/{total} ({100*a/max(total,1):.1f}%)")

    # Write markdown
    md = ["# Investigation 4: LLM Blind Spots (LLM=0, Real>0)\n"]
    md.append(f"These are questions where the LLM scores near zero but real students ")
    md.append(f"actually pass some or all test cases. The LLM completely fails to solve ")
    md.append(f"problems that students can handle.\n")
    md.append(f"LLM blind spots (sim<=0.05, real>0.05): **{len(blind)} questions**")
    md.append(f"Questions where LLM scores >0.05: {len(llm_active)}\n")

    md.append("## Summary Comparison\n")
    md.append("| Metric | Blind (LLM=0) | LLM Active |")
    md.append("|--------|---------------|------------|")
    md.append(f"| Count | {len(blind)} | {len(llm_active)} |")
    md.append(f"| Mean real score | {blind['real_mean_score'].mean():.3f} | {llm_active['real_mean_score'].mean():.3f} |")
    md.append(f"| Mean n_testcases | {blind['n_testcases'].mean():.1f} | {llm_active['n_testcases'].mean():.1f} |")
    md.append(f"| Mean week | {blind['week'].mean():.1f} | {llm_active['week'].mean():.1f} |\n")

    md.append("## Topic Breakdown\n")
    md.append("| Topic | Blind | Active |")
    md.append("|-------|-------|--------|")
    for topic in sorted(blind_topics.index, key=lambda t: -blind_topics.get(t, 0)):
        b = blind_topics.get(topic, 0)
        a = active_topics.get(topic, 0)
        md.append(f"| {topic} | {b} | {a} |")
    md.append("")

    md.append("## All Blind Spot Questions (sorted by real score, highest first)\n")
    for _, row in blind.iterrows():
        md.append(f"### Q{int(row['question_unittest_id'])} — {row['question_name']}\n")
        md.append(f"- **LLM mean score**: {row['sim_mean_score']:.3f} (n={int(row['sim_n'])})")
        md.append(f"- **Real mean score**: {row['real_mean_score']:.3f} (n={int(row['real_n'])})")
        md.append(f"- **LLM pass rate**: {row['sim_pass_rate']:.3f}")
        md.append(f"- **Real pass rate**: {row['real_pass_rate']:.3f}")
        md.append(f"- **Topic**: {row['topic']}")
        md.append(f"- **Week**: {row['week']}")
        md.append(f"- **Course**: {int(row['course_id']) if pd.notna(row['course_id']) else 'N/A'}")
        md.append(f"- **N testcases**: {int(row['n_testcases']) if pd.notna(row['n_testcases']) else 'N/A'}")
        qtext = str(row.get("question_text", ""))
        if qtext and qtext != "nan":
            md.append(f"\n<details><summary>Question Text</summary>\n\n```\n{qtext}\n```\n</details>")
        qtempl = str(row.get("question_template", ""))
        if qtempl and qtempl != "nan":
            md.append(f"\n<details><summary>Template</summary>\n\n```cpp\n{qtempl}\n```\n</details>")
        qtests = str(row.get("question_unittests", ""))
        if qtests and qtests != "nan":
            md.append(f"\n<details><summary>Unit Tests</summary>\n\n```\n{qtests[:3000]}\n```\n</details>")
        md.append("")

    with open(os.path.join(OUT_DIR, "blind_spot_questions.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\n  Saved blind_spot_questions.md")
    return blind


# ── Investigation 5: Broad Patterns ─────────────────────────────────────

def inv5_broad_patterns(merged_enriched, q_merged, q_info):
    m = merged_enriched.copy()

    # Add question difficulty tier from q_merged
    q_diff = q_merged[["question_unittest_id", "real_pass_rate", "real_mean_score"]].copy()
    q_diff["difficulty"] = pd.cut(
        q_diff["real_pass_rate"],
        bins=[-0.01, 0.2, 0.6, 1.01],
        labels=["hard", "medium", "easy"],
    )
    m = m.merge(q_diff[["question_unittest_id", "difficulty", "real_pass_rate"]],
                on="question_unittest_id", how="left")

    # Student ability tier
    student_ability = m.groupby("student_id")["y_true"].mean().reset_index()
    student_ability.columns = ["student_id", "student_mean_score"]
    terciles = student_ability["student_mean_score"].quantile([0.33, 0.66])
    student_ability["ability"] = pd.cut(
        student_ability["student_mean_score"],
        bins=[-0.01, terciles.iloc[0], terciles.iloc[1], 1.01],
        labels=["low", "mid", "high"],
    )
    m = m.merge(student_ability[["student_id", "ability", "student_mean_score"]],
                on="student_id", how="left")

    # n_testcases tier
    m["tc_tier"] = pd.cut(
        m["n_testcases"],
        bins=[-1, 5, 10, 100],
        labels=["1-5", "6-10", "11+"],
    )

    segments = {
        "difficulty": "difficulty",
        "course": "course_id",
        "topic": "topic",
        "student_ability": "ability",
        "n_testcases": "tc_tier",
    }

    results = []
    for seg_name, seg_col in segments.items():
        for val, grp in m.groupby(seg_col, observed=True):
            if len(grp) < 20:
                continue
            if grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
                r_val = 0.0
            else:
                r_val, _ = pearsonr(grp["y_pred"], grp["y_true"])
            mae = (grp["y_pred"] - grp["y_true"]).abs().mean()
            results.append({
                "segment": seg_name,
                "value": str(val),
                "pearson_r": r_val,
                "mae": mae,
                "n_pairs": len(grp),
                "mean_y_pred": grp["y_pred"].mean(),
                "mean_y_true": grp["y_true"].mean(),
            })

    res = pd.DataFrame(results)

    print(f"\n  Segment analysis results:")
    for seg_name in segments:
        print(f"\n  By {seg_name}:")
        sub = res[res["segment"] == seg_name].sort_values("pearson_r", ascending=False)
        for _, row in sub.iterrows():
            print(f"    {str(row['value']):<20s}  r={row['pearson_r']:+.4f}  "
                  f"MAE={row['mae']:.3f}  n={int(row['n_pairs']):>6d}  "
                  f"pred={row['mean_y_pred']:.3f}  true={row['mean_y_true']:.3f}")

    # Write markdown
    md = ["# Investigation 5: Broad Patterns of LLM Prediction\n"]
    md.append("## Segment Summary\n")
    for seg_name in segments:
        md.append(f"### By {seg_name}\n")
        md.append("| Segment | Pearson r | MAE | N pairs | Mean y_pred | Mean y_true |")
        md.append("|---------|-----------|-----|---------|-------------|-------------|")
        sub = res[res["segment"] == seg_name].sort_values("pearson_r", ascending=False)
        for _, row in sub.iterrows():
            md.append(f"| {row['value']} | {row['pearson_r']:.4f} | {row['mae']:.3f} "
                      f"| {int(row['n_pairs'])} | {row['mean_y_pred']:.3f} | {row['mean_y_true']:.3f} |")
        md.append("")

    # Per-segment question lists
    md.append("## Per-Segment Question Breakdown\n")
    q_name_map = q_info.set_index("question_unittest_id")["question_name"].to_dict()
    q_topic_map = q_info.set_index("question_unittest_id")["topic"].to_dict()

    for seg_name, seg_col in segments.items():
        if seg_name not in ["difficulty", "course"]:
            continue
        md.append(f"### Questions by {seg_name}\n")
        for val, grp in m.groupby(seg_col, observed=True):
            if len(grp) < 20:
                continue
            md.append(f"#### {seg_name} = {val}\n")
            q_in_seg = []
            for qid, qgrp in grp.groupby("question_unittest_id"):
                if len(qgrp) < 5:
                    continue
                if qgrp["y_pred"].std() == 0 or qgrp["y_true"].std() == 0:
                    qr = 0.0
                else:
                    qr, _ = pearsonr(qgrp["y_pred"], qgrp["y_true"])
                q_in_seg.append({
                    "qid": int(qid),
                    "name": q_name_map.get(qid, "?"),
                    "topic": q_topic_map.get(qid, "?"),
                    "r": qr,
                    "n": len(qgrp),
                })
            q_in_seg.sort(key=lambda x: x["r"], reverse=True)
            md.append("| QID | Name | Topic | Pearson r | N |")
            md.append("|-----|------|-------|-----------|---|")
            for q in q_in_seg:
                md.append(f"| {q['qid']} | {q['name']} | {q['topic']} | {q['r']:.3f} | {q['n']} |")
            md.append("")

    # Summary section
    best_seg = res.loc[res["pearson_r"].idxmax()]
    worst_seg = res.loc[res["pearson_r"].idxmin()]
    md.append("## Where Does the LLM Work?\n")
    md.append(f"- **Best segment**: {best_seg['segment']}={best_seg['value']} "
              f"(r={best_seg['pearson_r']:.4f}, n={int(best_seg['n_pairs'])})")
    md.append(f"- **Worst segment**: {worst_seg['segment']}={worst_seg['value']} "
              f"(r={worst_seg['pearson_r']:.4f}, n={int(worst_seg['n_pairs'])})")
    md.append(f"- Overall Pearson r: {pearsonr(m['y_pred'], m['y_true'])[0]:.4f}")
    md.append(f"\nEven the best segment has very weak prediction quality.")

    with open(os.path.join(OUT_DIR, "broad_patterns_by_segment.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\n  Saved broad_patterns_by_segment.md")

    # Figure: topic-only vertical bars (exclude near-zero correlations)
    topic_res = res[res["segment"] == "topic"].sort_values("pearson_r", ascending=False)
    topic_res = topic_res[topic_res["pearson_r"].abs() > 0.05]
    fig, ax = plt.subplots(figsize=(max(10, len(topic_res) * 0.4), 5))
    colors = plt.cm.RdYlGn((topic_res["pearson_r"] - topic_res["pearson_r"].min()) /
                             max(topic_res["pearson_r"].max() - topic_res["pearson_r"].min(), 0.001))
    x_pos = np.arange(len(topic_res))
    ax.bar(x_pos, topic_res["pearson_r"], color=colors, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(topic_res["value"], fontsize=8, rotation=60, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Prediction Quality by Topic")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    for i, (_, row) in enumerate(topic_res.iterrows()):
        offset = 0.01 if row["pearson_r"] >= 0 else -0.03
        ax.text(i, row["pearson_r"] + offset, f"{row['pearson_r']:.3f}",
                ha="center", va="bottom" if row["pearson_r"] >= 0 else "top", fontsize=7)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "prediction_quality_by_topic.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return res


def fig_aggregate_vs_pairwise(merged, q_merged):
    """Scatter: question-level aggregate (Spearman) + student-question pair (Pearson)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: question-level aggregate
    ax = axes[0]
    ax.scatter(q_merged["sim_mean_score"], q_merged["real_mean_score"],
               alpha=0.5, s=20, c="#4C72B0", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LLM Mean Score (per question)")
    ax.set_ylabel("Real Student Mean Score (per question)")
    rho, _ = spearmanr(q_merged["sim_mean_score"], q_merged["real_mean_score"])
    ax.set_title(f"Aggregate (Question-Level)\nSpearman r = {rho:.3f}")

    # Right: student-level aggregate
    ax = axes[1]
    student_agg = merged.groupby("student_id").agg(
        mean_pred=("y_pred", "mean"), mean_true=("y_true", "mean")
    ).reset_index()
    ax.scatter(student_agg["mean_pred"], student_agg["mean_true"],
               alpha=0.3, s=15, c="#C44E52", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LLM Mean Score (per student)")
    ax.set_ylabel("Real Student Mean Score (per student)")
    rho_s, _ = spearmanr(student_agg["mean_pred"], student_agg["mean_true"])
    ax.set_title(f"Aggregate (Student-Level)\nSpearman r = {rho_s:.3f}")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "aggregate_question_and_student.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def fig_correlation_histograms(merged):
    """Combined figure: (a) per-student Pearson r histogram, (b) per-question Pearson r histogram."""
    # Per-student r
    student_stats = []
    for sid, grp in merged.groupby("student_id"):
        if len(grp) < 5:
            continue
        if grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        r, _ = pearsonr(grp["y_pred"], grp["y_true"])
        student_stats.append({"student_id": sid, "r": r})
    sa = pd.DataFrame(student_stats)

    # Per-question r
    question_stats = []
    for qid, grp in merged.groupby("question_unittest_id"):
        if len(grp) < 10:
            continue
        if grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        r, _ = pearsonr(grp["y_pred"], grp["y_true"])
        question_stats.append({"question_unittest_id": qid, "r": r})
    qa = pd.DataFrame(question_stats)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(sa["r"], bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="r = 0")
    ax.axvline(sa["r"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean = {sa['r'].mean():.3f}")
    ax.set_xlabel("Per-Student Pearson r")
    ax.set_ylabel("Number of Students")
    ax.set_title("Per-Student Correlation")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(qa["r"], bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="r = 0")
    ax.axvline(qa["r"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean = {qa['r'].mean():.3f}")
    ax.set_xlabel("Per-Question Pearson r")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Per-Question Correlation")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "per_student_and_question_correlation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Kendall Tau Figures ─────────────────────────────────────────────────

def fig_kendall_tau(merged, q_merged):
    """Kendall Tau rank correlation at question-aggregate and student-aggregate levels."""
    # Per-student tau
    student_taus = []
    for sid, grp in merged.groupby("student_id"):
        if len(grp) < 5:
            continue
        if grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        tau, p = kendalltau(grp["y_pred"], grp["y_true"])
        student_taus.append({"student_id": sid, "tau": tau, "p": p, "n": len(grp)})
    st = pd.DataFrame(student_taus)

    # Per-question tau
    question_taus = []
    for qid, grp in merged.groupby("question_unittest_id"):
        if len(grp) < 10:
            continue
        if grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        tau, p = kendalltau(grp["y_pred"], grp["y_true"])
        question_taus.append({"question_unittest_id": qid, "tau": tau, "p": p, "n": len(grp)})
    qt = pd.DataFrame(question_taus)

    # Aggregates
    q_tau, _ = kendalltau(q_merged["sim_mean_score"], q_merged["real_mean_score"])
    student_agg = merged.groupby("student_id").agg(
        mean_pred=("y_pred", "mean"), mean_true=("y_true", "mean")
    ).reset_index()
    s_tau, _ = kendalltau(student_agg["mean_pred"], student_agg["mean_true"])

    # --- Figure 1: Histograms of per-student and per-question tau ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(st["tau"], bins=40, color="#6A9B59", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label=r"$\tau$ = 0")
    ax.axvline(st["tau"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=fr"Mean = {st['tau'].mean():.3f}")
    ax.set_xlabel(r"Per-Student Kendall $\tau$")
    ax.set_ylabel("Number of Students")
    ax.set_title("Per-Student Rank Correlation")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(qt["tau"], bins=30, color="#6A9B59", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label=r"$\tau$ = 0")
    ax.axvline(qt["tau"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=fr"Mean = {qt['tau'].mean():.3f}")
    ax.set_xlabel(r"Per-Question Kendall $\tau$")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Per-Question Rank Correlation")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kendall_tau_histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Figure 2: Aggregate scatter with tau annotations ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.scatter(q_merged["sim_mean_score"], q_merged["real_mean_score"],
               alpha=0.5, s=20, c="#6A9B59", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LLM Mean Score")
    ax.set_ylabel("Real Mean Score")
    ax.set_title(fr"Question-Level Aggregate ($\tau$ = {q_tau:.3f})")
    ax.set_aspect("equal")

    ax = axes[1]
    ax.scatter(student_agg["mean_pred"], student_agg["mean_true"],
               alpha=0.3, s=15, c="#C44E52", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LLM Mean Score")
    ax.set_ylabel("Real Mean Score")
    ax.set_title(fr"Student-Level Aggregate ($\tau$ = {s_tau:.3f})")
    ax.set_aspect("equal")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kendall_tau_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Print summary
    print(f"  Per-student tau: mean={st['tau'].mean():.4f}, median={st['tau'].median():.4f}, "
          f"tau>0: {(st['tau'] > 0).mean():.1%}")
    print(f"  Per-question tau: mean={qt['tau'].mean():.4f}, median={qt['tau'].median():.4f}, "
          f"tau>0: {(qt['tau'] > 0).mean():.1%}")
    print(f"  Question-aggregate tau: {q_tau:.4f}")
    print(f"  Student-aggregate tau: {s_tau:.4f}")

    return st, qt


def fig_kendall_tau_decomposition(merged):
    """Decompose per-student Kendall Tau into question-difficulty vs student-specific signal.

    Two analyses:
    1. Variance decomposition: compare how much LLM predictions vary across questions
       (per student) vs across students (per question).
    2. Centered tau: subtract per-question mean prediction, then recompute per-student tau.
       If tau drops to zero, the original correlation was entirely from question difficulty.
    """
    # ── Variance decomposition ──
    var_across_questions = merged.groupby("student_id")["y_pred"].var().dropna()
    var_across_students = merged.groupby("question_unittest_id")["y_pred"].var().dropna()
    var_q_true = merged.groupby("student_id")["y_true"].var().dropna()
    var_s_true = merged.groupby("question_unittest_id")["y_true"].var().dropna()

    # ── Centered tau ──
    q_mean_pred = merged.groupby("question_unittest_id")["y_pred"].transform("mean")
    q_mean_true = merged.groupby("question_unittest_id")["y_true"].transform("mean")
    m = merged.copy()
    m["y_pred_c"] = m["y_pred"] - q_mean_pred
    m["y_true_c"] = m["y_true"] - q_mean_true

    orig_taus, centered_taus = [], []
    for _, grp in m.groupby("student_id"):
        if len(grp) < 5 or grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        tau_o, _ = kendalltau(grp["y_pred"], grp["y_true"])
        orig_taus.append(tau_o)
        if grp["y_pred_c"].std() > 0 and grp["y_true_c"].std() > 0:
            tau_c, _ = kendalltau(grp["y_pred_c"], grp["y_true_c"])
            centered_taus.append(tau_c)

    orig_taus = np.array(orig_taus)
    centered_taus = np.array(centered_taus)

    # ── Figure: 1x2 ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── Centered tau per question (remove student ability signal) ──
    s_mean_pred = merged.groupby("student_id")["y_pred"].transform("mean")
    s_mean_true = merged.groupby("student_id")["y_true"].transform("mean")
    m["y_pred_cs"] = m["y_pred"] - s_mean_pred
    m["y_true_cs"] = m["y_true"] - s_mean_true

    orig_q_taus, centered_q_taus = [], []
    for _, grp in m.groupby("question_unittest_id"):
        if len(grp) < 10 or grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        tau_o, _ = kendalltau(grp["y_pred"], grp["y_true"])
        if not np.isnan(tau_o):
            orig_q_taus.append(tau_o)
        if grp["y_pred_cs"].std() > 0 and grp["y_true_cs"].std() > 0:
            tau_c, _ = kendalltau(grp["y_pred_cs"], grp["y_true_cs"])
            if not np.isnan(tau_c):
                centered_q_taus.append(tau_c)

    orig_q_taus = np.array(orig_q_taus)
    centered_q_taus = np.array(centered_q_taus)

    ax = axes[0]
    ax.hist(orig_taus, bins=40, alpha=0.6, color="#6A9B59", edgecolor="white",
            label=fr"Original (mean={orig_taus.mean():.3f})")
    ax.hist(centered_taus, bins=40, alpha=0.6, color="#C44E52", edgecolor="white",
            label=fr"Centered (mean={centered_taus.mean():.3f})")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"Per-Student Kendall $\tau$")
    ax.set_ylabel("Number of Students")
    ax.set_title("Remove Question Difficulty")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(orig_q_taus, bins=30, alpha=0.6, color="#6A9B59", edgecolor="white",
            label=fr"Original (mean={orig_q_taus.mean():.3f})")
    ax.hist(centered_q_taus, bins=30, alpha=0.6, color="#C44E52", edgecolor="white",
            label=fr"Centered (mean={centered_q_taus.mean():.3f})")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"Per-Question Kendall $\tau$")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Remove Student Ability")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kendall_tau_decomposition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Print summary ──
    ratio = var_across_questions.mean() / var_across_students.mean()
    drop_pct = (orig_taus.mean() - centered_taus.mean()) / orig_taus.mean() * 100
    print(f"  Variance ratio (question-axis / student-axis): {ratio:.1f}x")
    print(f"  Per-student tau:  original={orig_taus.mean():.4f}, centered={centered_taus.mean():.4f}, drop={drop_pct:.0f}%")
    print(f"  Per-question tau: original={orig_q_taus.mean():.4f}, centered={centered_q_taus.mean():.4f}")


def fig_kendall_tau_raw(merged):
    """Raw (student, question) pair scatter with global Kendall Tau."""
    tau, _ = kendalltau(merged["y_pred"], merged["y_true"])

    fig, ax = plt.subplots(figsize=(5, 5))
    rng = np.random.default_rng(42)
    jx = merged["y_pred"].values + rng.normal(0, 0.02, len(merged))
    jy = merged["y_true"].values + rng.normal(0, 0.02, len(merged))
    ax.scatter(jx, jy, alpha=0.15, s=8, c="#C44E52", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("LLM Predicted Score")
    ax.set_ylabel("Real Student Score")
    ax.set_title(fr"All Student-Question Pairs ($\tau$ = {tau:.3f}, n = {len(merged):,})")
    ax.set_aspect("equal")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kendall_tau_raw.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    print(f"  Global Kendall tau: {tau:.4f}")


# ── Difficulty Comparison Across All Models ─────────────────────────────

def fig_difficulty_comparison(merged):
    """Compare question difficulty estimation: LLM vs temporal eval models (CIRT, DynamicIRT, Elo, RSSM)."""
    from dynamic_models.temporal_eval.harness import load_saved_results, ALL_COURSES

    # ── LLM difficulty: 1 - mean(y_pred) per question ──
    llm_q = merged.groupby("question_unittest_id").agg(
        pred_diff=("y_pred", lambda x: 1.0 - x.mean()),
        actual_diff=("y_true", lambda x: 1.0 - x.mean()),
        n=("y_true", "count"),
    ).reset_index()
    llm_q = llm_q[llm_q["n"] >= 5]

    tau_llm, _ = kendalltau(llm_q["pred_diff"], llm_q["actual_diff"])
    r_llm = np.corrcoef(llm_q["pred_diff"], llm_q["actual_diff"])[0, 1]

    # ── Temporal eval models: load per-course, aggregate test cases to questions ──
    from dynamic_models.temporal_eval.data_loader import load_unified_data
    te_dir = os.path.join(os.path.dirname(__file__), "..", "results", "temporal_eval")
    model_taus = {}
    model_rs = {}
    model_diffs = {}

    # Load each course's data and predictions once
    course_cache = {}
    for course in ALL_COURSES:
        _, preds = load_saved_results(course, te_dir)
        if not preds:
            continue
        data = load_unified_data(course)
        tc_to_qidx = data.question_infos["qidx"].values
        n_questions = data.question_infos["qidx"].nunique()
        course_cache[course] = (preds, tc_to_qidx, n_questions)

    for model_name in ["CIRT", "DynamicIRT", "Elo", "RSSM"]:
        all_actual, all_predicted = [], []
        for course, (preds, tc_to_qidx, n_questions) in course_cache.items():
            horizons = [h for m, h in preds if m == model_name]
            if not horizons:
                continue
            max_h = max(horizons)
            pred = preds[(model_name, max_h)]
            if pred.item_indices is None:
                continue

            items = pred.item_indices
            if items.max() < len(tc_to_qidx) and len(np.unique(items)) > n_questions:
                items = tc_to_qidx[items]

            for q in np.unique(items):
                mask = items == q
                if mask.sum() < 5:
                    continue
                all_actual.append(1.0 - pred.y_true[mask].mean())
                all_predicted.append(1.0 - pred.y_pred_prob[mask].mean())

        if len(all_actual) > 10:
            actual = np.array(all_actual)
            predicted = np.array(all_predicted)
            tau, _ = kendalltau(predicted, actual)
            r = np.corrcoef(predicted, actual)[0, 1]
            model_taus[model_name] = tau
            model_rs[model_name] = r
            model_diffs[model_name] = (actual, predicted)

    model_taus["LLM"] = tau_llm
    model_rs["LLM"] = r_llm
    model_diffs["LLM"] = (llm_q["actual_diff"].values, llm_q["pred_diff"].values)

    # ── Figure: one scatter per model ──
    MODEL_COLORS = {"CIRT": "#4477aa", "DynamicIRT": "#ee6677",
                    "Elo": "#228833", "RSSM": "#aa3377", "LLM": "#ccbb44"}
    models = ["CIRT", "DynamicIRT", "Elo", "RSSM", "LLM"]
    models = [m for m in models if m in model_taus]

    fig, axes = plt.subplots(1, len(models), figsize=(3 * len(models), 3))
    if len(models) == 1:
        axes = [axes]

    for ax, m in zip(axes, models):
        actual, predicted = model_diffs[m]
        ax.scatter(actual, predicted, s=12, alpha=0.4,
                   color=MODEL_COLORS[m], edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Actual Difficulty")
        if ax == axes[0]:
            ax.set_ylabel("Predicted Difficulty")
        ax.set_title(fr"{m} ($\tau$={model_taus[m]:.3f})", fontsize=9)
        ax.set_aspect("equal")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "difficulty_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    for m in models:
        print(f"  {m}: tau={model_taus[m]:.4f}, r={model_rs[m]:.4f}")


# ── Student-level example plots ──────────────────────────────────────────

def _plot_student_scatter(ax, grp, title, color="#4C72B0"):
    """Plot one student's y_pred vs y_true, one dot per question."""
    ax.scatter(grp["y_pred"], grp["y_true"], s=50, c=color, edgecolors="black",
               linewidths=0.5, zorder=5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LLM Predicted Score")
    ax.set_ylabel("Real Student Score")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def fig_inv1_students(merged, q_info):
    """Pick students from the best and worst predicted questions."""
    # Find students with many questions and compute their r
    student_stats = []
    for sid, grp in merged.groupby("student_id"):
        if len(grp) < 8:
            continue
        if grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        r, _ = pearsonr(grp["y_pred"], grp["y_true"])
        student_stats.append({"student_id": sid, "r": r, "n": len(grp),
                              "mean_true": grp["y_true"].mean()})
    sa = pd.DataFrame(student_stats).sort_values("r", ascending=False)

    # Pick: best r, median r, worst r, and one with high n
    picks = []
    picks.append(("Best correlated", sa.iloc[0]["student_id"]))
    mid = len(sa) // 2
    picks.append(("Median correlated", sa.iloc[mid]["student_id"]))
    picks.append(("Worst correlated", sa.iloc[-1]["student_id"]))
    high_n = sa[sa["n"] >= 15].iloc[len(sa[sa["n"] >= 15]) // 2]
    picks.append(("Typical (15+ questions)", high_n["student_id"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, (label, sid) in zip(axes.flat, picks):
        grp = merged[merged["student_id"] == sid]
        r_val = sa[sa["student_id"] == sid]["r"].values[0]
        _plot_student_scatter(ax, grp,
                              f"{label}\nStudent {sid} (r={r_val:.3f}, n={len(grp)})")

    fig.suptitle("Investigation 1: Example Students at Different Prediction Levels",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, "student_scatter_examples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Central Tendency Bias Figure (for paper) ─────────────────────────────

def fig_central_tendency(merged):
    m = merged.copy()

    student_ability = m.groupby("student_id")["y_true"].mean()
    terciles = student_ability.quantile([0.33, 0.66])
    tier_map = pd.cut(
        student_ability,
        bins=[-0.01, terciles.iloc[0], terciles.iloc[1], 1.01],
        labels=["Low", "Mid", "High"],
    )
    m["ability"] = m["student_id"].map(tier_map)

    q_difficulty = m.groupby("question_unittest_id")["y_true"].mean()
    q_terciles = q_difficulty.quantile([0.33, 0.66])
    q_tier_map = pd.cut(
        q_difficulty,
        bins=[-0.01, q_terciles.iloc[0], q_terciles.iloc[1], 1.01],
        labels=["Hard", "Medium", "Easy"],
    )
    m["difficulty"] = m["question_unittest_id"].map(q_tier_map)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    w = 0.35

    tiers = ["Low", "Mid", "High"]
    pred_means = [m[m["ability"] == t]["y_pred"].mean() for t in tiers]
    true_means = [m[m["ability"] == t]["y_true"].mean() for t in tiers]
    counts = [len(m[m["ability"] == t]) for t in tiers]
    x = np.arange(len(tiers))
    bars1 = axes[0].bar(x - w/2, true_means, w, label="Real student", color="#4878CF", edgecolor="white")
    bars2 = axes[0].bar(x + w/2, pred_means, w, label="LLM predicted", color="#D65F5F", edgecolor="white")
    axes[0].set_xlabel("Student ability tier")
    axes[0].set_ylabel("Mean score")
    axes[0].set_title("By student ability")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{t}\n(n={c:,})" for t, c in zip(tiers, counts)])
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0, 0.85)
    for bar, val in zip(bars1, true_means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, pred_means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    diff_tiers = ["Hard", "Medium", "Easy"]
    pred_d = [m[m["difficulty"] == t]["y_pred"].mean() for t in diff_tiers]
    true_d = [m[m["difficulty"] == t]["y_true"].mean() for t in diff_tiers]
    counts_d = [len(m[m["difficulty"] == t]) for t in diff_tiers]
    x2 = np.arange(len(diff_tiers))
    bars3 = axes[1].bar(x2 - w/2, true_d, w, label="Real student", color="#4878CF", edgecolor="white")
    bars4 = axes[1].bar(x2 + w/2, pred_d, w, label="LLM predicted", color="#D65F5F", edgecolor="white")
    axes[1].set_xlabel("Question difficulty tier")
    axes[1].set_ylabel("Mean score")
    axes[1].set_title("By question difficulty")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([f"{t}\n(n={c:,})" for t, c in zip(diff_tiers, counts_d)])
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0, 0.95)
    for bar, val in zip(bars3, true_d):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars4, pred_d):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "overconfidence_by_ability_and_difficulty.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

    for t in tiers:
        sub = m[m["ability"] == t]
        print(f"  {t} ability: pred={sub['y_pred'].mean():.3f}, true={sub['y_true'].mean():.3f}, n={len(sub)}")


# ── Main ─────────────────────────────────────────────────────────────────

def filter_blind_spots(merged, q_merged):
    """Remove structurally broken questions: LLM mean <= 0.05, real mean > 0.05.

    These 64 questions fail because of platform-specific template issues
    (hidden APIs, source-code guards, Vietnamese specs), not because the LLM
    can't model student behavior. Manual review confirmed ~83% have clear
    structural causes.
    """
    blind_qids = set(
        q_merged[(q_merged["sim_mean_score"] <= 0.05) & (q_merged["real_mean_score"] > 0.05)]
        ["question_unittest_id"]
    )
    m_filt = merged[~merged["question_unittest_id"].isin(blind_qids)].copy()
    q_filt = q_merged[~q_merged["question_unittest_id"].isin(blind_qids)].copy()

    n_removed = len(blind_qids)
    n_pairs_removed = len(merged) - len(m_filt)
    print(f"\n  FILTERING: Removed {n_removed} structurally broken questions "
          f"({n_pairs_removed:,} pairs)")
    print(f"  Remaining: {len(q_filt)} questions, {len(m_filt):,} pairs")
    return m_filt, q_filt, blind_qids


def write_summary_metrics(merged, q_merged, label):
    n_students = merged["student_id"].nunique()
    n_questions = merged["question_unittest_id"].nunique()
    n_pairs = len(merged)

    overall_r, _ = pearsonr(merged["y_pred"], merged["y_true"])
    overall_rho, _ = spearmanr(merged["y_pred"], merged["y_true"])
    overall_mae = (merged["y_pred"] - merged["y_true"]).abs().mean()

    q_rho, _ = spearmanr(q_merged["sim_mean_score"], q_merged["real_mean_score"])

    student_agg = merged.groupby("student_id").agg(
        mean_pred=("y_pred", "mean"), mean_true=("y_true", "mean")
    )
    student_rho, _ = spearmanr(student_agg["mean_pred"], student_agg["mean_true"])

    per_student_r = []
    for _, grp in merged.groupby("student_id"):
        if len(grp) < 5 or grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        r, _ = pearsonr(grp["y_pred"], grp["y_true"])
        per_student_r.append(r)
    per_student_r = pd.Series(per_student_r)

    per_question_r = []
    for _, grp in merged.groupby("question_unittest_id"):
        if len(grp) < 10 or grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        r, _ = pearsonr(grp["y_pred"], grp["y_true"])
        per_question_r.append(r)
    per_question_r = pd.Series(per_question_r)

    md = [f"# Summary Metrics ({label})\n"]
    md.append("## Dataset\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Students | {n_students:,} |")
    md.append(f"| Questions | {n_questions:,} |")
    md.append(f"| Student-question pairs | {n_pairs:,} |")
    md.append(f"| Mean LLM score | {merged['y_pred'].mean():.3f} |")
    md.append(f"| Mean real score | {merged['y_true'].mean():.3f} |")
    md.append(f"| LLM full-pass rate | {(merged['y_pred'] >= 1.0).mean():.3f} |")
    md.append(f"| Real full-pass rate | {(merged['y_true'] >= 1.0).mean():.3f} |")

    md.append("\n## Correlation\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Pair-level Pearson r | {overall_r:.4f} |")
    md.append(f"| Pair-level Spearman r | {overall_rho:.4f} |")
    md.append(f"| Pair-level MAE | {overall_mae:.4f} |")
    md.append(f"| Question-aggregate Spearman r | {q_rho:.4f} |")
    md.append(f"| Student-aggregate Spearman r | {student_rho:.4f} |")
    md.append(f"| Per-student Pearson r mean | {per_student_r.mean():.4f} |")
    md.append(f"| Per-student Pearson r median | {per_student_r.median():.4f} |")
    md.append(f"| Per-student r > 0 | {(per_student_r > 0).mean():.1%} |")
    md.append(f"| Per-student r computed (n) | {len(per_student_r)} |")
    md.append(f"| Per-question Pearson r mean | {per_question_r.mean():.4f} |")
    md.append(f"| Per-question Pearson r median | {per_question_r.median():.4f} |")
    md.append(f"| Per-question r > 0 | {(per_question_r > 0).mean():.1%} |")
    md.append(f"| Per-question r computed (n) | {len(per_question_r)} |")

    path = os.path.join(OUT_DIR, "summary_metrics.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    print(f"  Saved {path}")

    for line in md[3:]:
        if line.startswith("|") and not line.startswith("|--"):
            print(f"  {line}")


def run_analyses(merged, q_merged, real_df, q_info, label):
    global OUT_DIR
    base_dir = os.path.join(os.path.dirname(__file__), "..", "results", "llm_predictor")
    OUT_DIR = os.path.join(base_dir, label)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"[{label}] SUMMARY METRICS")
    print("=" * 70)
    write_summary_metrics(merged, q_merged, label)

    merged_enriched = enrich_merged(merged, real_df, q_info)

    print("\n" + "=" * 70)
    print(f"[{label}] INVESTIGATION 1: Best/Worst Predicted Questions")
    print("=" * 70)
    inv1_best_worst_questions(merged, q_info)
    fig_inv1_students(merged, q_info)

    print("\n" + "=" * 70)
    print(f"[{label}] INVESTIGATION 3: Real=0, LLM>0 (Overconfident)")
    print("=" * 70)
    inv3_overconfident(q_merged, q_info)

    print("\n" + "=" * 70)
    print(f"[{label}] INVESTIGATION 5: Broad Patterns by Segment")
    print("=" * 70)
    inv5_res = inv5_broad_patterns(merged_enriched, q_merged, q_info)

    print("\n" + "=" * 70)
    print(f"[{label}] AGGREGATE VS PAIRWISE")
    print("=" * 70)
    fig_aggregate_vs_pairwise(merged, q_merged)

    print("\n" + "=" * 70)
    print(f"[{label}] CORRELATION HISTOGRAMS")
    print("=" * 70)
    fig_correlation_histograms(merged)

    print("\n" + "=" * 70)
    print(f"[{label}] KENDALL TAU")
    print("=" * 70)
    fig_kendall_tau(merged, q_merged)
    fig_kendall_tau_raw(merged)
    fig_kendall_tau_decomposition(merged)
    fig_difficulty_comparison(merged)

    print("\n" + "=" * 70)
    print(f"[{label}] CENTRAL TENDENCY BIAS")
    print("=" * 70)
    fig_central_tendency(merged)

    return inv5_res


def main():
    global OUT_DIR
    print("Loading data...")
    sim_df, real_df = load_data()
    merged_raw = compute_merged(sim_df, real_df)
    q_merged_raw = compute_question_level(sim_df, real_df)
    q_info = load_question_infos()

    merged, q_merged, blind_qids = filter_blind_spots(merged_raw, q_merged_raw)

    # Run on unfiltered data
    print("\n" + "#" * 70)
    print("# UNFILTERED DATASET")
    print("#" * 70)
    run_analyses(merged_raw, q_merged_raw, real_df, q_info, "unfiltered")

    # Run on filtered data
    print("\n" + "#" * 70)
    print("# FILTERED DATASET (blind spots removed)")
    print("#" * 70)
    run_analyses(merged, q_merged, real_df, q_info, "filtered")

    # Investigation 4 (blind spots) only makes sense on unfiltered data
    base_dir = os.path.join(os.path.dirname(__file__), "..", "results", "llm_predictor")
    OUT_DIR = os.path.join(base_dir, "unfiltered")
    print("\n" + "=" * 70)
    print("INVESTIGATION 4: LLM Blind Spots (unfiltered)")
    print("=" * 70)
    inv4_llm_blind_spots(q_merged_raw, q_info)

    print("\n" + "=" * 70)
    print("DONE — all outputs saved to:")
    print(f"  {base_dir}/unfiltered/")
    print(f"  {base_dir}/filtered/")
    print("=" * 70)


if __name__ == "__main__":
    main()
