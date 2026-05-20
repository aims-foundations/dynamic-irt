"""Error type flow: Re-grade LLM and student code to classify error types.

Produces the error_type_flow.png Sankey diagram showing how test-case
outcomes transition from real students to LLM predictor.

Usage:
    python data_analysis/llm_bug_comparison.py
"""

import json
import os
import shutil
import subprocess
import tempfile
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

JSONL_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "results", "llm_student_eval", "dsa_hk231", "claude_attempts10.jsonl",
)
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "llm_predictor", "bug_comparison",
)

PALETTE = {
    "correct":      "#6A9B59",
    "wrong_output": "#4C72B0",
    "runtime":      "#E8A838",
    "compile":      "#C44E52",
}
ERROR_ORDER = ["correct", "wrong_output", "runtime", "compile"]


def _pass_fraction(s):
    s = str(s).strip()
    if not s or s == "nan":
        return np.nan
    return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan


def _parse_test_cases(unittests_str):
    from llm_simulator.data_loader import parse_test_cases
    return parse_test_cases(unittests_str)


def _grade_detailed(template, testcases, code):
    """Grade code and return per-test-case error type + actual output."""
    formatted = []
    std_inputs = []
    for tc in testcases:
        formatted.append({"testcode": tc["input"], "expected_output": tc["output"]})
        std_inputs.append(tc.get("std_in", ""))

    code_with_answer = template.replace("{{ STUDENT_ANSWER }}", code)
    start_idx = code_with_answer.find("{% for TEST in TESTCASES %}")
    end_idx = code_with_answer.find("{% endfor %}") + len("{% endfor %}")
    sources = [code_with_answer[:start_idx] + tc["testcode"] + code_with_answer[end_idx:]
               for tc in formatted]

    temp_dir = tempfile.mkdtemp()
    results = []
    try:
        for i, src in enumerate(sources):
            cpp_file = os.path.join(temp_dir, f"tc_{i}.cpp")
            exe_file = os.path.join(temp_dir, f"tc_{i}.out")
            with open(cpp_file, "w") as f:
                f.write(src)

            comp = subprocess.run(
                ["g++", "-std=c++11", cpp_file, "-o", exe_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            if comp.returncode != 0:
                results.append({"error_type": "compile", "output": "", "expected": testcases[i]["output"].strip()})
                continue

            try:
                run = subprocess.run(
                    ["timeout", "10", exe_file], input=std_inputs[i],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                if run.returncode != 0:
                    results.append({"error_type": "runtime", "output": "", "expected": testcases[i]["output"].strip()})
                    continue
                actual = run.stdout.strip()
                expected = testcases[i]["output"].strip()
                if actual == expected:
                    results.append({"error_type": "correct", "output": actual, "expected": expected})
                else:
                    results.append({"error_type": "wrong_output", "output": actual, "expected": expected})
            except Exception:
                results.append({"error_type": "runtime", "output": "", "expected": testcases[i]["output"].strip()})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return results


def _grade_one_pair(args):
    qid, template, testcases, llm_code, real_code, sid, attempt_id = args
    llm_results = _grade_detailed(template, testcases, llm_code)
    real_results = _grade_detailed(template, testcases, real_code)
    return qid, sid, attempt_id, llm_results, real_results


def load_and_grade():
    with open(JSONL_PATH) as f:
        llm_rows = [json.loads(l) for l in f]
    llm_df = pd.DataFrame(llm_rows)
    llm_df = llm_df[llm_df["response_type"] == "Submit"].copy()
    llm_df["student_id"] = llm_df["student_id"].astype(str)
    llm_df["question_unittest_id"] = llm_df["question_unittest_id"].astype(str)
    llm_df["attempt_id"] = pd.to_numeric(llm_df["attempt_id"], errors="coerce")

    hf_dir = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        repo_type="dataset", local_files_only=True,
    )
    qi = pd.read_csv(os.path.join(hf_dir, "question_infos.csv"))
    q_info = {}
    for qid_int, grp in qi.groupby("question_id"):
        row = grp.iloc[0]
        template = str(row["question_template"])
        unittests = str(row["question_unittests"])
        tcs = _parse_test_cases(unittests)
        if tcs:
            q_info[str(qid_int)] = {"template": template, "testcases": tcs}

    real_df = pd.read_csv(
        os.path.join(hf_dir, "main_data.csv"),
        dtype={"pass": str}, low_memory=False, on_bad_lines="skip",
    )
    real_df = real_df[
        (real_df["response_type"] == "Submit") & real_df["response"].notna()
    ].copy()
    real_df["student_id"] = real_df["student_id"].astype(str)
    real_df["question_unittest_id"] = real_df["question_unittest_id"].astype(str)

    real_by_pair = defaultdict(list)
    for _, r in real_df.iterrows():
        p = str(r["pass"]).strip()
        if p and p != "nan":
            real_by_pair[(r["student_id"], r["question_unittest_id"])].append({
                "pass": p, "code": r["response"],
            })

    # Build grading jobs: partial-credit pattern matches
    jobs = []
    for _, r in llm_df.iterrows():
        sid, qid = r["student_id"], r["question_unittest_id"]
        llm_p = str(r["pass"]).strip()
        if not llm_p or llm_p == "nan":
            continue
        frac = _pass_fraction(llm_p)
        if frac == 1.0 or frac == 0.0:
            continue
        if qid not in q_info:
            continue

        real_attempts = real_by_pair.get((sid, qid), [])
        matching = [a for a in real_attempts if a["pass"] == llm_p]
        if not matching:
            # Also include non-matching partial-credit for baseline
            non_matching = [a for a in real_attempts if 0 < _pass_fraction(a["pass"]) < 1]
            if non_matching:
                jobs.append((
                    qid, q_info[qid]["template"], q_info[qid]["testcases"],
                    r["response"], non_matching[0]["code"],
                    sid, int(r["attempt_id"]),
                ))
            continue

        jobs.append((
            qid, q_info[qid]["template"], q_info[qid]["testcases"],
            r["response"], matching[0]["code"],
            sid, int(r["attempt_id"]),
        ))

    print(f"  Grading {len(jobs)} code pairs...")

    all_results = []
    n_workers = min(16, len(jobs))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_grade_one_pair, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                qid, sid, attempt_id, llm_res, real_res = future.result()
                n = min(len(llm_res), len(real_res))
                for ti in range(n):
                    all_results.append({
                        "student_id": sid,
                        "question_unittest_id": qid,
                        "attempt_id": attempt_id,
                        "test_index": ti,
                        "llm_error_type": llm_res[ti]["error_type"],
                        "real_error_type": real_res[ti]["error_type"],
                        "llm_output": llm_res[ti]["output"],
                        "real_output": real_res[ti]["output"],
                        "expected_output": llm_res[ti]["expected"],
                    })
            except Exception as e:
                print(f"  Grading error: {e}")

    df = pd.DataFrame(all_results)
    print(f"  Graded {len(df)} test-case pairs across {df['question_unittest_id'].nunique()} questions")
    return df


# ── Error type flow diagram ──────────────────────────────────────────────


def fig_error_flow(df):
    from matplotlib.patches import Polygon

    counts = Counter()
    for _, row in df.iterrows():
        counts[(row["real_error_type"], row["llm_error_type"])] += 1

    # Vertical layout: top = Real Student, bottom = LLM Predictor
    # Left-to-right ordering: correct, wrong_output, runtime, compile
    display_order = ERROR_ORDER
    display_labels = ["Correct", "Wrong\nOutput", "Runtime\nError", "Compile\nError"]

    total = sum(counts.values())
    gap = total * 0.04

    top_totals = {t: sum(v for (r, l), v in counts.items() if r == t) for t in display_order}
    bot_totals = {t: sum(v for (r, l), v in counts.items() if l == t) for t in display_order}

    # Use display widths with minimum for small categories (so labels fit)
    min_display = total * 0.12
    top_display = {t: max(top_totals[t], min_display) if top_totals[t] > 0 else 0 for t in display_order}
    bot_display = {t: max(bot_totals[t], min_display) if bot_totals[t] > 0 else 0 for t in display_order}

    # Compute x positions using display widths (for bars/labels)
    top_x0_display = {}
    x = 0
    for t in display_order:
        top_x0_display[t] = x
        x += top_display[t] + gap

    bot_x0_display = {}
    x = 0
    for t in display_order:
        bot_x0_display[t] = x
        x += bot_display[t] + gap

    # Compute x positions using real widths (for flow connection points)
    # Center real widths within display widths
    top_x0 = {}
    for t in display_order:
        offset = (top_display[t] - top_totals[t]) / 2
        top_x0[t] = top_x0_display[t] + offset

    bot_x0 = {}
    for t in display_order:
        offset = (bot_display[t] - bot_totals[t]) / 2
        bot_x0[t] = bot_x0_display[t] + offset

    y_top = 0.80
    y_bot = 0.20
    bar_h = 0.04

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw bars using display widths, colored portion = real width
    for t in display_order:
        w_real = top_totals[t]
        if w_real > 0:
            ax.bar(top_x0[t] + w_real/2, bar_h, width=w_real, bottom=y_top,
                   color=PALETTE[t], edgecolor="white", linewidth=1.5, zorder=3)
            label = display_labels[display_order.index(t)]
            cx = top_x0_display[t] + top_display[t] / 2
            ax.text(cx, y_top + bar_h + 0.02,
                    label, ha="center", va="bottom", fontsize=24,
                    color="#333333", fontweight="bold", linespacing=1.3)

        w_real = bot_totals[t]
        if w_real > 0:
            ax.bar(bot_x0[t] + w_real/2, bar_h, width=w_real, bottom=y_bot - bar_h,
                   color=PALETTE[t], edgecolor="white", linewidth=1.5, zorder=3)
            label = display_labels[display_order.index(t)]
            cx = bot_x0_display[t] + bot_display[t] / 2
            ax.text(cx, y_bot - bar_h - 0.02,
                    label, ha="center", va="top", fontsize=24,
                    color="#333333", fontweight="bold", linespacing=1.3)

    # Draw flows (vertical: top to bottom)
    top_cursor = {t: top_x0[t] for t in display_order}
    bot_cursor = {t: bot_x0[t] for t in display_order}

    flow_items = sorted(counts.items(), key=lambda x: (x[0][0] == x[0][1], x[1]))

    def _sigmoid_vert(x0, x1, y_top, y_bot, n=60):
        ys = np.linspace(y_top, y_bot, n)
        t = (ys - y_top) / (y_bot - y_top)
        s = 3 * t**2 - 2 * t**3
        xs = x0 + s * (x1 - x0)
        return xs, ys

    for (real_t, llm_t), count in flow_items:
        if count == 0:
            continue

        x0_left = top_cursor[real_t]
        x1_left = bot_cursor[llm_t]
        top_cursor[real_t] += count
        bot_cursor[llm_t] += count

        is_match = real_t == llm_t
        color = PALETTE[real_t]
        alpha = 0.45 if is_match else 0.18
        zorder = 2 if is_match else 1

        xs_l, ys_l = _sigmoid_vert(x0_left, x1_left, y_top, y_bot)
        xs_r, ys_r = _sigmoid_vert(x0_left + count, x1_left + count, y_top, y_bot)

        verts = list(zip(xs_l, ys_l)) + list(zip(xs_r[::-1], ys_r[::-1]))
        poly = Polygon(verts, closed=True, facecolor=color, alpha=alpha,
                       edgecolor=color, linewidth=0.3, zorder=zorder)
        ax.add_patch(poly)

    ax.text(-gap, y_top + bar_h / 2, "Real Student",
            ha="right", va="center", fontsize=26, fontweight="bold", color="#333333")
    ax.text(-gap, y_bot - bar_h / 2, "LLM Predictor",
            ha="right", va="center", fontsize=26, fontweight="bold", color="#333333")

    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    path = os.path.join(OUT_DIR, "error_type_flow.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache_path = os.path.join(OUT_DIR, "graded_pairs.csv")
    if os.path.exists(cache_path):
        print("Loading cached grading results...")
        df = pd.read_csv(cache_path)
    else:
        print("Loading data and grading code pairs...")
        df = load_and_grade()
        df.to_csv(cache_path, index=False)
        print(f"  Cached to {cache_path}")

    print(f"\n  Total test-case pairs: {len(df)}")

    print("\nError type flow diagram...")
    fig_error_flow(df)

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
