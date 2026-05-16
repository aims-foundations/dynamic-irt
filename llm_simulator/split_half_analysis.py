import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse
import csv


def load_data(path):
    records = [json.loads(l) for l in open(path)]
    for r in records:
        r["all_pass"] = all(c == "1" for c in r["pass"])
    return records


def student_accuracy(records, student_set):
    vals = [int(r["all_pass"]) for r in records if r["student_id"] in student_set]
    return np.mean(vals) if vals else 0.0


def question_accuracy(records, student_set):
    """Per-question accuracy for a subset of students."""
    by_q = defaultdict(list)
    for r in records:
        if r["student_id"] in student_set:
            by_q[r["question_unittest_id"]].append(int(r["all_pass"]))
    return {q: np.mean(vs) for q, vs in by_q.items()}


def main():
    parser = argparse.ArgumentParser(description="Split-half reliability analysis")
    parser.add_argument("--data", default="results/llm_eval/opus_dsa_50_iterative/claude_n5_attempts10.jsonl")
    parser.add_argument("--n_splits", type=int, default=1000)
    parser.add_argument("--output_dir", default="results/llm_eval/split_half")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_data(args.data)
    students = sorted(set(r["student_id"] for r in records))
    n = len(students)
    half = n // 2
    rng = np.random.default_rng(42)

    overall_acc = np.mean([int(r["all_pass"]) for r in records])

    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 70)
    log("SPLIT-HALF RELIABILITY ANALYSIS")
    log("=" * 70)
    log(f"\nData: {args.data}")
    log(f"Total records: {len(records)}")
    log(f"Students: {n}")
    log(f"Overall accuracy: {overall_acc:.4f}")
    log(f"Random splits: {args.n_splits}")

    # ─── 1. Random split-half: overall accuracy ───
    log(f"\n{'─' * 70}")
    log("1. SPLIT-HALF: OVERALL ACCURACY AGREEMENT")
    log(f"{'─' * 70}")

    acc_a_list, acc_b_list, diffs = [], [], []
    for i in range(args.n_splits):
        perm = rng.permutation(students)
        half_a = set(perm[:half])
        half_b = set(perm[half:])
        acc_a = student_accuracy(records, half_a)
        acc_b = student_accuracy(records, half_b)
        acc_a_list.append(acc_a)
        acc_b_list.append(acc_b)
        diffs.append(abs(acc_a - acc_b))

    acc_a_arr = np.array(acc_a_list)
    acc_b_arr = np.array(acc_b_list)
    diffs_arr = np.array(diffs)
    correlation = np.corrcoef(acc_a_arr, acc_b_arr)[0, 1]

    log(f"   Half A accuracy: mean={acc_a_arr.mean():.4f}, std={acc_a_arr.std():.4f}")
    log(f"   Half B accuracy: mean={acc_b_arr.mean():.4f}, std={acc_b_arr.std():.4f}")
    log(f"   |A - B| difference: mean={diffs_arr.mean():.4f}, median={np.median(diffs_arr):.4f}, max={diffs_arr.max():.4f}")
    log(f"   Correlation between halves: {correlation:.4f}")
    log(f"   Spearman-Brown corrected reliability: {2 * correlation / (1 + correlation):.4f}")
    log(f"   90% of splits agree within: ±{np.quantile(diffs_arr, 0.90):.4f}")
    log(f"   95% of splits agree within: ±{np.quantile(diffs_arr, 0.95):.4f}")

    # ─── 2. Per-question accuracy correlation across halves ───
    log(f"\n{'─' * 70}")
    log("2. SPLIT-HALF: PER-QUESTION ACCURACY CORRELATION")
    log(f"{'─' * 70}")

    q_corrs = []
    for i in range(min(args.n_splits, 500)):
        perm = rng.permutation(students)
        half_a = set(perm[:half])
        half_b = set(perm[half:])
        qa = question_accuracy(records, half_a)
        qb = question_accuracy(records, half_b)
        shared = sorted(set(qa) & set(qb))
        if len(shared) > 5:
            va = [qa[q] for q in shared]
            vb = [qb[q] for q in shared]
            q_corrs.append(np.corrcoef(va, vb)[0, 1])

    q_corrs = np.array(q_corrs)
    log(f"   Per-question accuracy correlation between halves:")
    log(f"   Mean r = {q_corrs.mean():.4f}, std = {q_corrs.std():.4f}")
    log(f"   This measures: if question X is hard for half A's students,")
    log(f"   is it also hard for half B's students?")

    # ─── 3. Increasing sample size simulation ───
    log(f"\n{'─' * 70}")
    log("3. CONVERGENCE: HOW ACCURACY STABILIZES WITH MORE STUDENTS")
    log(f"{'─' * 70}")

    sample_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    sample_sizes = [s for s in sample_sizes if s <= n]

    convergence_rows = []
    for size in sample_sizes:
        accs = []
        for _ in range(500):
            sampled = set(rng.choice(students, size=size, replace=False))
            accs.append(student_accuracy(records, sampled))
        accs = np.array(accs)
        ci_lo, ci_hi = np.quantile(accs, 0.025), np.quantile(accs, 0.975)
        log(f"   {size:3d} students: mean={accs.mean():.4f}, 95% range=[{ci_lo:.4f}, {ci_hi:.4f}], width={ci_hi - ci_lo:.4f}")
        convergence_rows.append({
            "n_students": size,
            "mean_acc": round(accs.mean(), 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "ci_width": round(ci_hi - ci_lo, 4),
        })

    # ─── 4. Fixed split: show one concrete example ───
    log(f"\n{'─' * 70}")
    log("4. ONE CONCRETE SPLIT (first 25 vs last 25 alphabetically)")
    log(f"{'─' * 70}")

    sorted_students = sorted(students)
    half_a = set(sorted_students[:half])
    half_b = set(sorted_students[half:])
    acc_a = student_accuracy(records, half_a)
    acc_b = student_accuracy(records, half_b)
    n_a = sum(1 for r in records if r["student_id"] in half_a)
    n_b = sum(1 for r in records if r["student_id"] in half_b)

    log(f"   Half A: {len(half_a)} students, {n_a} records, accuracy = {acc_a:.4f}")
    log(f"   Half B: {len(half_b)} students, {n_b} records, accuracy = {acc_b:.4f}")
    log(f"   Difference: {abs(acc_a - acc_b):.4f}")

    # ─── Summary ───
    log(f"\n{'─' * 70}")
    log("5. INTERPRETATION")
    log(f"{'─' * 70}")
    log(f"   Overall accuracy: {overall_acc:.4f}")
    log(f"   Split-half agreement (95th pctl): ±{np.quantile(diffs_arr, 0.95):.4f}")
    log(f"   Per-question ranking is {'stable' if q_corrs.mean() > 0.7 else 'moderately stable' if q_corrs.mean() > 0.5 else 'unstable'} across halves (r={q_corrs.mean():.3f})")

    width_at_50 = convergence_rows[-1]["ci_width"]
    width_at_25 = next(r for r in convergence_rows if r["n_students"] == 25)["ci_width"]
    log(f"   At 25 students, 95% range width = {width_at_25:.4f}")
    log(f"   At 50 students, 95% range width = {width_at_50:.4f}")

    if np.quantile(diffs_arr, 0.95) < 0.10:
        log(f"   → Two random halves of your data agree within ~{np.quantile(diffs_arr, 0.95):.1%} — results are reasonably stable.")
    else:
        log(f"   → Halves can disagree by {np.quantile(diffs_arr, 0.95):.1%} — consider adding more students.")

    log("")

    # Save outputs
    (out_dir / "split_half_report.txt").write_text("\n".join(lines))
    with open(out_dir / "convergence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=convergence_rows[0].keys())
        w.writeheader()
        w.writerows(convergence_rows)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
