import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse


def load_data(path):
    records = [json.loads(l) for l in open(path)]
    for r in records:
        r["all_pass"] = all(c == "1" for c in r["pass"])
    return records


def binomial_ci(n_success, n_total, confidence=0.95):
    """Wilson score interval - better than normal approx for proportions."""
    from scipy.stats import norm

    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return center - margin, center + margin


def bootstrap_ci(values, n_boot=10000, confidence=0.95, cluster_ids=None):
    """Bootstrap CI, optionally cluster-resampling by cluster_ids."""
    rng = np.random.default_rng(42)
    alpha = (1 - confidence) / 2

    if cluster_ids is not None:
        unique_clusters = np.unique(cluster_ids)
        cluster_means = []
        cluster_map = defaultdict(list)
        for v, c in zip(values, cluster_ids):
            cluster_map[c].append(v)

        boot_means = []
        for _ in range(n_boot):
            sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            sampled_vals = [v for c in sampled for v in cluster_map[c]]
            boot_means.append(np.mean(sampled_vals))
    else:
        values = np.array(values)
        boot_means = []
        for _ in range(n_boot):
            idx = rng.choice(len(values), size=len(values), replace=True)
            boot_means.append(np.mean(values[idx]))

    boot_means = np.array(boot_means)
    return np.quantile(boot_means, alpha), np.quantile(boot_means, 1 - alpha)


def effective_sample_size(values, cluster_ids):
    """Compute design effect and effective N due to clustering."""
    cluster_map = defaultdict(list)
    for v, c in zip(values, cluster_ids):
        cluster_map[c].append(v)

    n = len(values)
    k = len(cluster_map)
    cluster_sizes = [len(vs) for vs in cluster_map.values()]
    m_bar = np.mean(cluster_sizes)

    overall_mean = np.mean(values)
    ss_between = sum(len(vs) * (np.mean(vs) - overall_mean) ** 2 for vs in cluster_map.values())
    ss_within = sum(sum((v - np.mean(vs)) ** 2 for v in vs) for vs in cluster_map.values())

    ms_between = ss_between / (k - 1) if k > 1 else 0
    ms_within = ss_within / (n - k) if n > k else 0

    if ms_within == 0:
        icc = 0.0
    else:
        icc = max(0, (ms_between - ms_within) / (ms_between + (m_bar - 1) * ms_within))

    deff = 1 + (m_bar - 1) * icc
    n_eff = n / deff
    return icc, deff, n_eff


def margin_of_error_for_n(n_students, p=0.54, avg_records_per_student=38, confidence=0.95):
    """Predict margin of error for a given number of students, accounting for clustering."""
    from scipy.stats import norm

    z = norm.ppf(1 - (1 - confidence) / 2)
    n_total = n_students * avg_records_per_student
    naive_se = np.sqrt(p * (1 - p) / n_total)
    return z * naive_se


def main():
    parser = argparse.ArgumentParser(description="Power analysis for LLM simulation accuracy")
    parser.add_argument("--data", default="results/llm_eval/opus_dsa_50_iterative/claude_n5_attempts10.jsonl")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", default="results/llm_eval/power_analysis.txt")
    args = parser.parse_args()

    records = load_data(args.data)
    values = [int(r["all_pass"]) for r in records]
    student_ids = [r["student_id"] for r in records]
    question_ids = [r["question_unittest_id"] for r in records]

    n = len(values)
    p_hat = np.mean(values)
    n_students = len(set(student_ids))
    n_questions = len(set(question_ids))

    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 70)
    log("POWER ANALYSIS: LLM Simulation Accuracy")
    log("=" * 70)
    log(f"\nData: {args.data}")
    log(f"Total records: {n}")
    log(f"Unique students: {n_students}")
    log(f"Unique questions: {n_questions}")
    log(f"Overall accuracy (all tests pass): {p_hat:.4f}")
    log(f"Confidence level: {args.confidence}")

    # 1. Naive binomial CI (assumes independence)
    log(f"\n{'─' * 70}")
    log("1. NAIVE BINOMIAL CI (assumes independent observations)")
    log(f"{'─' * 70}")
    lo, hi = binomial_ci(sum(values), n, args.confidence)
    log(f"   Wilson interval: [{lo:.4f}, {hi:.4f}]")
    log(f"   Width: {hi - lo:.4f}")
    log(f"   ± {(hi - lo) / 2:.4f}")

    # 2. Cluster-adjusted CIs
    log(f"\n{'─' * 70}")
    log("2. CLUSTER-ADJUSTED ANALYSIS")
    log(f"{'─' * 70}")

    # By student
    icc_stu, deff_stu, neff_stu = effective_sample_size(values, student_ids)
    log(f"\n   Clustering by STUDENT:")
    log(f"   ICC (intraclass correlation): {icc_stu:.4f}")
    log(f"   Design effect: {deff_stu:.2f}")
    log(f"   Effective N: {neff_stu:.0f} (out of {n})")
    lo_stu, hi_stu = binomial_ci(int(p_hat * neff_stu), int(neff_stu), args.confidence)
    log(f"   Adjusted CI: [{lo_stu:.4f}, {hi_stu:.4f}]")
    log(f"   Width: {hi_stu - lo_stu:.4f}")
    log(f"   ± {(hi_stu - lo_stu) / 2:.4f}")

    # By question
    icc_q, deff_q, neff_q = effective_sample_size(values, question_ids)
    log(f"\n   Clustering by QUESTION:")
    log(f"   ICC (intraclass correlation): {icc_q:.4f}")
    log(f"   Design effect: {deff_q:.2f}")
    log(f"   Effective N: {neff_q:.0f} (out of {n})")
    lo_q, hi_q = binomial_ci(int(p_hat * neff_q), int(neff_q), args.confidence)
    log(f"   Adjusted CI: [{lo_q:.4f}, {hi_q:.4f}]")
    log(f"   Width: {hi_q - lo_q:.4f}")
    log(f"   ± {(hi_q - lo_q) / 2:.4f}")

    # 3. Bootstrap CIs
    log(f"\n{'─' * 70}")
    log("3. BOOTSTRAP CIs (10,000 resamples)")
    log(f"{'─' * 70}")

    lo_b, hi_b = bootstrap_ci(values, confidence=args.confidence)
    log(f"   Naive bootstrap:           [{lo_b:.4f}, {hi_b:.4f}]  width={hi_b - lo_b:.4f}")

    lo_bs, hi_bs = bootstrap_ci(values, confidence=args.confidence, cluster_ids=student_ids)
    log(f"   Cluster bootstrap (student):[{lo_bs:.4f}, {hi_bs:.4f}]  width={hi_bs - lo_bs:.4f}")

    lo_bq, hi_bq = bootstrap_ci(values, confidence=args.confidence, cluster_ids=question_ids)
    log(f"   Cluster bootstrap (question):[{lo_bq:.4f}, {hi_bq:.4f}]  width={hi_bq - lo_bq:.4f}")

    # 4. Per-attempt CIs
    log(f"\n{'─' * 70}")
    log("4. PER-ATTEMPT ACCURACY WITH CIs")
    log(f"{'─' * 70}")
    by_attempt = defaultdict(list)
    by_attempt_students = defaultdict(list)
    for r in records:
        by_attempt[int(r["attempt_id"])].append(int(r["all_pass"]))
        by_attempt_students[int(r["attempt_id"])].append(r["student_id"])

    for att in sorted(by_attempt.keys()):
        vals = by_attempt[att]
        sids = by_attempt_students[att]
        p_a = np.mean(vals)
        lo_a, hi_a = binomial_ci(sum(vals), len(vals), args.confidence)
        lo_ab, hi_ab = bootstrap_ci(vals, confidence=args.confidence, cluster_ids=sids)
        log(f"   Attempt {att}: {p_a:.3f}  n={len(vals):4d}  "
            f"Wilson=[{lo_a:.3f},{hi_a:.3f}]  "
            f"ClusterBoot=[{lo_ab:.3f},{hi_ab:.3f}]")

    # 5. Power analysis: how many students needed?
    log(f"\n{'─' * 70}")
    log("5. SAMPLE SIZE PROJECTIONS (students needed for target precision)")
    log(f"{'─' * 70}")
    log(f"   Assuming p≈{p_hat:.2f}, ~{n // n_students} records/student, "
        f"student ICC={icc_stu:.3f}")

    avg_per_student = n / n_students
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - args.confidence) / 2)

    for target_margin in [0.01, 0.02, 0.03, 0.05, 0.10]:
        naive_n = (z**2 * p_hat * (1 - p_hat)) / target_margin**2
        naive_students = naive_n / avg_per_student
        adjusted_students = naive_students * deff_stu
        log(f"   ±{target_margin:.2f}: ~{max(1, int(np.ceil(adjusted_students))):>5d} students needed "
            f"(naive: {max(1, int(np.ceil(naive_students))):>5d})")

    # 6. What precision do we currently have?
    log(f"\n{'─' * 70}")
    log("6. CURRENT PRECISION SUMMARY")
    log(f"{'─' * 70}")
    best_ci = (lo_bs, hi_bs)
    width = best_ci[1] - best_ci[0]
    log(f"   Best estimate (cluster bootstrap by student): [{best_ci[0]:.4f}, {best_ci[1]:.4f}]")
    log(f"   Width: {width:.4f}")
    log(f"   Margin of error: ±{width/2:.4f}")
    log(f"   With {n_students} students, we can distinguish accuracy differences of ~{width:.3f}")

    log("")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
