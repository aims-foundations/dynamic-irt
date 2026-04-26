"""Generate report: LLM simulation predictions vs real student outcomes.

Produces figures and prints statistics for the report showing that
aggregate-level correlation does not translate to student-level prediction.
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "temporal_eval")
os.makedirs(OUT_DIR, exist_ok=True)


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
    sim_first = sim_sub.groupby(["student_id", "question_unittest_id"]).first().reset_index()
    sim_first["y_pred"] = sim_first["score"]
    sim_first = sim_first.dropna(subset=["y_pred"])

    real_sub = real_df.copy()
    real_sub["attempt_id"] = pd.to_numeric(real_sub.get("attempt_id", 0), errors="coerce").fillna(0)
    real_sub = real_sub.sort_values("attempt_id")
    real_first = real_sub.groupby(["student_id", "question_unittest_id"]).first().reset_index()
    real_first["y_true"] = real_first["score"]

    merged = sim_first[["student_id", "question_unittest_id", "y_pred"]].merge(
        real_first[["student_id", "question_unittest_id", "y_true"]],
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


# ── Figure 1: Aggregate question-level correlation ──────────────────────
def fig1_question_level(q_merged):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pass rate
    ax = axes[0]
    ax.scatter(q_merged["sim_pass_rate"], q_merged["real_pass_rate"],
               alpha=0.4, s=20, c="#4C72B0", edgecolors="none")
    r_s, _ = spearmanr(q_merged["sim_pass_rate"], q_merged["real_pass_rate"])
    r_p, _ = pearsonr(q_merged["sim_pass_rate"], q_merged["real_pass_rate"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("LLM Simulated Pass Rate")
    ax.set_ylabel("Real Student Pass Rate")
    ax.set_title(f"Question-Level Pass Rates\n(Spearman r = {r_s:.3f}, n = {len(q_merged)})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    # Mean score
    ax = axes[1]
    ax.scatter(q_merged["sim_mean_score"], q_merged["real_mean_score"],
               alpha=0.4, s=20, c="#DD8452", edgecolors="none")
    r_s2, _ = spearmanr(q_merged["sim_mean_score"], q_merged["real_mean_score"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("LLM Simulated Mean Score")
    ax.set_ylabel("Real Student Mean Score")
    ax.set_title(f"Question-Level Mean Scores\n(Spearman r = {r_s2:.3f}, n = {len(q_merged)})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "report_fig1_question_level.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return r_s, r_s2


# ── Figure 2: Per-student correlation distribution ──────────────────────
def fig2_per_student_correlation(merged):
    student_stats = []
    for sid, grp in merged.groupby("student_id"):
        if len(grp) < 5:
            continue
        if grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        r, _ = pearsonr(grp["y_pred"], grp["y_true"])
        student_stats.append({
            "student_id": sid, "r": r,
            "n": len(grp), "mean_score": grp["y_true"].mean(),
        })
    sa = pd.DataFrame(student_stats)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(sa["r"], bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No correlation (r = 0)")
    ax.axvline(sa["r"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean = {sa['r'].mean():.3f}")
    ax.set_xlabel("Per-Student Pearson r (LLM score vs real score)")
    ax.set_ylabel("Number of Students")
    ax.set_title(f"Distribution of Per-Student Correlation\n(n = {len(sa)} students with 5+ questions)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "report_fig2_per_student_auc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return sa


# ── Figure 3: Calibration — predicted vs actual ────────────────────────
def fig3_calibration(merged):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Binned calibration
    ax = axes[0]
    merged["pred_bin"] = pd.cut(merged["y_pred"],
                                 bins=[0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01],
                                 labels=["0", "(0,0.1)", "[0.1,0.3)", "[0.3,0.5)",
                                         "[0.5,0.7)", "[0.7,0.9)", "[0.9,1)", "1"],
                                 include_lowest=True)
    cal = merged.groupby("pred_bin", observed=True).agg(
        actual=("y_true", "mean"), n=("y_true", "count"),
        pred_mean=("y_pred", "mean"),
    ).reset_index()
    cal = cal[cal["n"] >= 10]

    bars = ax.bar(range(len(cal)), cal["actual"], color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(cal)))
    ax.set_xticklabels(cal["pred_bin"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Real Student Mean Score")
    ax.set_xlabel("LLM Predicted Score Bin")
    ax.set_title("Calibration: LLM Score vs Real Score")
    ax.axhline(merged["y_true"].mean(), color="red", linestyle="--", alpha=0.6,
               label=f"Overall mean score = {merged['y_true'].mean():.3f}")
    ax.legend(fontsize=9)
    for bar, (_, row) in zip(bars, cal.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={int(row['n'])}", ha="center", va="bottom", fontsize=7)

    # Scatter: y_pred vs y_true (jittered)
    ax = axes[1]
    jitter_y = merged["y_true"] + np.random.normal(0, 0.02, len(merged))
    jitter_x = merged["y_pred"] + np.random.normal(0, 0.01, len(merged))
    ax.scatter(jitter_x, jitter_y, alpha=0.02, s=3, c="#4C72B0", rasterized=True)
    r_overall, _ = pearsonr(merged["y_pred"], merged["y_true"])
    rho_overall, _ = spearmanr(merged["y_pred"], merged["y_true"])
    ax.set_xlabel("LLM Predicted Score (pass fraction)")
    ax.set_ylabel("Real Student Score (pass fraction, jittered)")
    ax.set_title(f"Student-Level: LLM Score vs Real Score\n(Pearson r = {r_overall:.3f}, n = {len(merged):,})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.15)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "report_fig3_calibration.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return r_overall, rho_overall, cal


# ── Figure 4: Per-question Kendall tau (student ranking) ────────────────
def fig4_per_question_ranking(merged):
    q_taus = []
    for qid, grp in merged.groupby("question_unittest_id"):
        if len(grp) < 10:
            continue
        if grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        tau, _ = kendalltau(grp["y_pred"], grp["y_true"])
        q_taus.append({"qid": qid, "tau": tau, "n": len(grp)})
    qt = pd.DataFrame(q_taus)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(qt["tau"], bins=30, color="#55A868", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No ranking ability (tau = 0)")
    ax.axvline(qt["tau"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean = {qt['tau'].mean():.3f}")
    ax.set_xlabel("Kendall Tau (LLM ranking vs real ranking of students)")
    ax.set_ylabel("Number of Questions")
    ax.set_title(f"Can the LLM Rank Students Within a Question?\n"
                 f"(n = {len(qt)} questions with 10+ students)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "report_fig4_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return qt


# ── Figure 5: Aggregate vs pairwise — the core contrast ────────────────
def fig5_aggregate_vs_pairwise(q_merged, merged):
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # Left: question-level (aggregate) — looks decent
    ax = fig.add_subplot(gs[0])
    ax.scatter(q_merged["sim_mean_score"], q_merged["real_mean_score"],
               alpha=0.5, s=25, c="#4C72B0", edgecolors="none")
    r_s, _ = spearmanr(q_merged["sim_mean_score"], q_merged["real_mean_score"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("LLM Mean Score (per question)")
    ax.set_ylabel("Real Student Mean Score (per question)")
    ax.set_title(f"Aggregate (Question-Level)\nSpearman r = {r_s:.3f}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    # Right: student-question pair level — random
    ax = fig.add_subplot(gs[1])
    sample = merged.sample(min(5000, len(merged)), random_state=42)
    jitter_y = sample["y_true"] + np.random.normal(0, 0.02, len(sample))
    jitter_x = sample["y_pred"] + np.random.normal(0, 0.01, len(sample))
    ax.scatter(jitter_x, jitter_y, alpha=0.08, s=5, c="#C44E52", rasterized=True)
    r_pair, _ = pearsonr(merged["y_pred"], merged["y_true"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("LLM Predicted Score (per student-question)")
    ax.set_ylabel("Real Student Score (per student-question)")
    ax.set_title(f"Per Student-Question Pair\nPearson r = {r_pair:.3f}")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.15)

    fig.suptitle("The Aggregation Illusion: Question-Level Signal Vanishes at the Student Level",
                 fontsize=14, fontweight="bold", y=1.02)
    path = os.path.join(OUT_DIR, "report_fig5_aggregate_vs_pairwise.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Print stats for report ──────────────────────────────────────────────
def print_report_stats(merged, q_merged, sa, qt, r_overall, rho_overall, cal):
    print("\n" + "=" * 60)
    print("REPORT STATISTICS")
    print("=" * 60)

    print(f"\nDataset:")
    print(f"  Matched (student, question) pairs: {len(merged):,}")
    print(f"  Unique students: {merged['student_id'].nunique():,}")
    print(f"  Unique questions: {merged['question_unittest_id'].nunique()}")
    print(f"  Overall real mean score: {merged['y_true'].mean():.3f}")
    print(f"  y_true == 0: {(merged['y_true'] == 0).mean()*100:.1f}%")
    print(f"  y_true == 1: {(merged['y_true'] >= 1.0).mean()*100:.1f}%")
    print(f"  y_true in (0,1): {((merged['y_true'] > 0) & (merged['y_true'] < 1)).mean()*100:.1f}%")

    print(f"\nQuestion-Level (Aggregate):")
    print(f"  Questions compared: {len(q_merged)}")
    r_pass, _ = spearmanr(q_merged["sim_pass_rate"], q_merged["real_pass_rate"])
    r_mean, _ = spearmanr(q_merged["sim_mean_score"], q_merged["real_mean_score"])
    print(f"  Pass rate Spearman: {r_pass:.3f}")
    print(f"  Mean score Spearman: {r_mean:.3f}")

    print(f"\nStudent-Level (Per Pair):")
    print(f"  Overall Pearson r: {r_overall:.4f}")
    print(f"  Overall Spearman rho: {rho_overall:.4f}")
    rmse = np.sqrt(((merged["y_pred"] - merged["y_true"])**2).mean())
    mae = (merged["y_pred"] - merged["y_true"]).abs().mean()
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    print(f"\n  Per-student correlation (n={len(sa)} students):")
    print(f"    Mean r: {sa['r'].mean():.3f}, Median r: {sa['r'].median():.3f}")
    print(f"    % with r > 0: {(sa['r'] > 0).mean()*100:.1f}%")
    print(f"    % with r > 0.2: {(sa['r'] > 0.2).mean()*100:.1f}%")

    print(f"\n  Per-question student ranking (n={len(qt)} questions):")
    print(f"    Mean Kendall tau: {qt['tau'].mean():.4f}")
    print(f"    % with positive tau: {(qt['tau'] > 0).mean()*100:.1f}%")

    print(f"\n  Calibration:")
    for _, row in cal.iterrows():
        print(f"    LLM score {row['pred_bin']:>10s}: real mean score = {row['actual']:.3f} (n={int(row['n'])})")


if __name__ == "__main__":
    print("Loading data...")
    sim_df, real_df = load_data()

    print("Computing merged pairs...")
    merged = compute_merged(sim_df, real_df)

    print("Computing question-level stats...")
    q_merged = compute_question_level(sim_df, real_df)

    print("\nGenerating figures...")
    r_s_pass, r_s_mean = fig1_question_level(q_merged)
    sa = fig2_per_student_correlation(merged)
    r_overall, rho_overall, cal = fig3_calibration(merged)
    qt = fig4_per_question_ranking(merged)
    fig5_aggregate_vs_pairwise(q_merged, merged)

    print_report_stats(merged, q_merged, sa, qt, r_overall, rho_overall, cal)
