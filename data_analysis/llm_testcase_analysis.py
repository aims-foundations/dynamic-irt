"""Test-case level analysis of LLM predictions vs real student outcomes.

Investigates whether the LLM captures the same sub-question failure modes
as real students by comparing pass/fail patterns at the individual test-case level.

Two analyses:
1. Test-case difficulty correlation: per question, do the LLM and real students
   find the same test cases hard?
2. Failure pattern overlap: when both get partial credit, do they fail the same
   test cases?

Usage:
    python data_analysis/llm_testcase_analysis.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from data_analysis.llm_predictor_analysis import load_data

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "llm_predictor", "testcase")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_pass_vectors(df):
    """Extract per-submission pass vectors from the 'pass' column."""
    df = df.copy()
    df["pass_str"] = df["pass"].astype(str).str.strip()
    df = df[df["pass_str"].str.len() > 0]
    df = df[df["pass_str"] != "nan"]
    return df


def compute_testcase_pass_rates(df):
    """For each question, compute per-test-case pass rate across all submissions.

    Returns dict: question_id -> np.array of pass rates (one per test case).
    """
    rates = {}
    for qid, grp in df.groupby("question_unittest_id"):
        vectors = grp["pass_str"].values
        n_tc = len(vectors[0])
        if not all(len(v) == n_tc for v in vectors):
            continue
        matrix = np.array([[int(c) for c in v] for v in vectors])
        rates[qid] = matrix.mean(axis=0)
    return rates


def testcase_difficulty_correlation(sim_df, real_df):
    """Analysis 1: Per-question correlation of test-case difficulty between LLM and real students."""
    sim_sub = parse_pass_vectors(sim_df[sim_df["response_type"] == "Submit"])
    real_sub = parse_pass_vectors(real_df)

    # Use ALL submissions (not just last) to maximize test-case level variance
    sim_last = sim_sub
    real_last = real_sub

    sim_rates = compute_testcase_pass_rates(sim_last)
    real_rates = compute_testcase_pass_rates(real_last)

    common_qs = set(sim_rates) & set(real_rates)
    # Filter to questions where pass string lengths match
    common_qs = {q for q in common_qs if len(sim_rates[q]) == len(real_rates[q])}
    # Need at least 3 test cases for meaningful correlation
    common_qs = {q for q in common_qs if len(sim_rates[q]) >= 3}

    results = []
    for qid in sorted(common_qs):
        sr = sim_rates[qid]
        rr = real_rates[qid]
        # Skip if either has zero variance
        if sr.std() == 0 or rr.std() == 0:
            results.append({"question_unittest_id": qid, "tau": np.nan,
                            "n_testcases": len(sr),
                            "sim_mean": sr.mean(), "real_mean": rr.mean()})
            continue
        tau, _ = kendalltau(sr, rr)
        results.append({"question_unittest_id": qid, "tau": tau,
                        "n_testcases": len(sr),
                        "sim_mean": sr.mean(), "real_mean": rr.mean()})

    df_results = pd.DataFrame(results)
    valid = df_results.dropna(subset=["tau"])

    print(f"\n{'='*60}")
    print("ANALYSIS 1: Test-Case Difficulty Correlation")
    print(f"{'='*60}")
    print(f"  Questions with matched test cases: {len(common_qs)}")
    print(f"  Questions with computable tau: {len(valid)}")
    print(f"  Mean tau: {valid['tau'].mean():.4f}")
    print(f"  Median tau: {valid['tau'].median():.4f}")
    print(f"  tau > 0: {(valid['tau'] > 0).mean():.1%}")

    # Figure: histogram of per-question test-case tau
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(valid["tau"], bins=30, color="#6A9B59", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label=r"$\tau$ = 0")
    ax.axvline(valid["tau"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=fr"Mean = {valid['tau'].mean():.3f}")
    ax.set_xlabel(r"Per-Question Test-Case Kendall $\tau$")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Test-Case Difficulty Agreement")
    ax.legend(fontsize=8)

    # Scatter: LLM mean pass rate vs real mean pass rate per test case (pooled)
    ax = axes[1]
    all_sim, all_real = [], []
    for qid in sorted(common_qs):
        sr = sim_rates[qid]
        rr = real_rates[qid]
        all_sim.extend(sr)
        all_real.extend(rr)
    all_sim = np.array(all_sim)
    all_real = np.array(all_real)
    global_tau, _ = kendalltau(all_sim, all_real)
    ax.scatter(all_real, all_sim, s=8, alpha=0.3, color="#4477aa", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Real Student Pass Rate")
    ax.set_ylabel("LLM Pass Rate")
    ax.set_title(fr"Per-Test-Case Pass Rates ($\tau$={global_tau:.3f}, n={len(all_sim):,})")
    ax.set_aspect("equal")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "testcase_difficulty_correlation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    return df_results, sim_rates, real_rates, common_qs


def failure_pattern_overlap(sim_df, real_df, common_qs, sim_rates, real_rates):
    """Analysis 2: When both get partial credit, do they fail the same test cases?"""
    sim_sub = parse_pass_vectors(sim_df[sim_df["response_type"] == "Submit"])
    real_sub = parse_pass_vectors(real_df)

    sim_last = sim_sub.sort_values("attempt_id").groupby(
        ["student_id", "question_unittest_id"]).last().reset_index()
    real_last = real_sub.groupby(
        ["student_id", "question_unittest_id"]).last().reset_index()

    # For each question, compute Jaccard similarity of failed test case sets
    # between partial-credit LLM submissions and partial-credit real submissions
    jaccard_per_question = []
    for qid in sorted(common_qs):
        n_tc = len(sim_rates[qid])

        # Get partial credit submissions (not all pass, not all fail)
        sim_q = sim_last[sim_last["question_unittest_id"] == qid].copy()
        real_q = real_last[real_last["question_unittest_id"] == qid].copy()

        def get_fail_sets(grp, n_tc):
            fail_sets = []
            for _, row in grp.iterrows():
                v = row["pass_str"]
                if len(v) != n_tc:
                    continue
                score = sum(int(c) for c in v) / n_tc
                if 0.01 < score < 0.99:  # partial credit only
                    failed = frozenset(i for i, c in enumerate(v) if c == "0")
                    fail_sets.append(failed)
            return fail_sets

        sim_fails = get_fail_sets(sim_q, n_tc)
        real_fails = get_fail_sets(real_q, n_tc)

        if not sim_fails or not real_fails:
            continue

        # Aggregate: which test cases fail most often?
        sim_fail_rate = np.zeros(n_tc)
        for fs in sim_fails:
            for i in fs:
                sim_fail_rate[i] += 1
        sim_fail_rate /= len(sim_fails)

        real_fail_rate = np.zeros(n_tc)
        for fs in real_fails:
            for i in fs:
                real_fail_rate[i] += 1
        real_fail_rate /= len(real_fails)

        # Jaccard between the "commonly failed" test cases (fail rate > 0.5)
        sim_common_fails = set(np.where(sim_fail_rate > 0.5)[0])
        real_common_fails = set(np.where(real_fail_rate > 0.5)[0])

        if sim_common_fails or real_common_fails:
            intersection = len(sim_common_fails & real_common_fails)
            union = len(sim_common_fails | real_common_fails)
            jaccard = intersection / union if union > 0 else 0
        else:
            jaccard = np.nan

        # Also compute correlation of failure rates
        if sim_fail_rate.std() > 0 and real_fail_rate.std() > 0:
            tau, _ = kendalltau(sim_fail_rate, real_fail_rate)
        else:
            tau = np.nan

        jaccard_per_question.append({
            "question_unittest_id": qid,
            "jaccard": jaccard,
            "failure_tau": tau,
            "n_sim_partial": len(sim_fails),
            "n_real_partial": len(real_fails),
            "n_testcases": n_tc,
        })

    df_j = pd.DataFrame(jaccard_per_question)
    valid_j = df_j.dropna(subset=["jaccard"])
    valid_t = df_j.dropna(subset=["failure_tau"])

    print(f"\n{'='*60}")
    print("ANALYSIS 2: Failure Pattern Overlap")
    print(f"{'='*60}")
    print(f"  Questions with partial credit in both: {len(df_j)}")
    print(f"  Mean Jaccard (common failed test cases): {valid_j['jaccard'].mean():.4f}")
    print(f"  Median Jaccard: {valid_j['jaccard'].median():.4f}")
    print(f"  Mean failure-rate tau: {valid_t['failure_tau'].mean():.4f}")
    print(f"  Median failure-rate tau: {valid_t['failure_tau'].median():.4f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(valid_j["jaccard"], bins=20, color="#C44E52", alpha=0.8, edgecolor="white")
    ax.axvline(valid_j["jaccard"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean = {valid_j['jaccard'].mean():.3f}")
    ax.set_xlabel("Jaccard Similarity (Failed Test Cases)")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Failure Pattern Overlap")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(valid_t["failure_tau"], bins=25, color="#4477aa", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label=r"$\tau$ = 0")
    ax.axvline(valid_t["failure_tau"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=fr"Mean = {valid_t['failure_tau'].mean():.3f}")
    ax.set_xlabel(r"Failure Rate Kendall $\tau$")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Failure Rate Correlation")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "failure_pattern_overlap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    return df_j


def main():
    print("Loading data...")
    sim_df, real_df = load_data()

    print("\nRunning test-case level analyses...")
    df_results, sim_rates, real_rates, common_qs = testcase_difficulty_correlation(sim_df, real_df)
    df_j = failure_pattern_overlap(sim_df, real_df, common_qs, sim_rates, real_rates)

    print(f"\nDone — outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
