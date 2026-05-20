"""Behavioral comparison: How does LLM coding behavior differ from students?

Compares code characteristics, editing patterns, and convergence trajectories
between LLM-generated and real student submissions on the same C++ problems.

Usage:
    python data_analysis/llm_behavioral_comparison.py
"""

import difflib
import json
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from Levenshtein import distance as levenshtein_distance

from tueplots import bundles

warnings.filterwarnings("ignore")
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

JSONL_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "results", "llm_student_eval", "dsa_hk231", "claude_attempts10.jsonl",
)
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "llm_predictor", "behavioral",
)

COLORS = {"student": "#4C72B0", "llm": "#C44E52"}
MAX_ATTEMPTS = 10


def _pass_fraction(s):
    s = str(s).strip()
    if not s or s == "nan":
        return np.nan
    return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan


def _code_metrics(code):
    code = str(code)
    lines = code.splitlines()
    non_empty = [l for l in lines if l.strip()]
    return {
        "loc": len(non_empty),
        "max_depth": max((len(l) - len(l.lstrip())) for l in lines) // 4 if lines else 0,
        "n_functions": len(re.findall(r'\b\w+\s+\w+\s*\([^)]*\)\s*\{', code)),
        "n_loops": len(re.findall(r'\b(for|while)\s*\(', code)),
        "n_conditionals": len(re.findall(r'\b(if|else\s+if|switch)\s*[\(\{]', code)),
        "n_variables": len(set(re.findall(r'\b(int|float|double|bool|char|string|auto|Node\*?|T)\s+(\w+)', code))),
        "avg_identifier_len": np.mean([len(m) for m in re.findall(r'\b([a-zA-Z_]\w{0,30})\b', code)]) if re.findall(r'\b([a-zA-Z_]\w{0,30})\b', code) else 0,
        "has_comments": 1 if re.search(r'//|/\*', code) else 0,
    }


def _normalize_code(code):
    code = str(code)
    code = re.sub(r'//[^\n]*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code).strip()
    return code


def _code_similarity(code1, code2):
    return difflib.SequenceMatcher(None, _normalize_code(code1), _normalize_code(code2)).ratio()


def load_data():
    print("Loading data...")
    with open(JSONL_PATH) as f:
        llm_rows = [json.loads(l) for l in f]
    llm_df = pd.DataFrame(llm_rows)
    llm_df = llm_df[llm_df["response_type"] == "Submit"].copy()
    llm_df["student_id"] = llm_df["student_id"].astype(str)
    llm_df["question_unittest_id"] = llm_df["question_unittest_id"].astype(str)
    llm_df["attempt_id"] = pd.to_numeric(llm_df["attempt_id"], errors="coerce")
    llm_df = llm_df.sort_values(["student_id", "question_unittest_id", "attempt_id"])

    # Sequential attempt numbers (1-indexed), capped at MAX_ATTEMPTS
    llm_df["attempt_num"] = llm_df.groupby(
        ["student_id", "question_unittest_id"]
    ).cumcount() + 1
    llm_df = llm_df[llm_df["attempt_num"] <= MAX_ATTEMPTS].copy()
    llm_df["source"] = "llm"

    hf_dir = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        repo_type="dataset", local_files_only=True,
    )
    real_df = pd.read_csv(
        os.path.join(hf_dir, "main_data.csv"),
        dtype={"pass": str}, low_memory=False, on_bad_lines="skip",
    )
    real_df = real_df[
        (real_df["response_type"] == "Submit") & real_df["response"].notna()
    ].copy()
    real_df["student_id"] = real_df["student_id"].astype(str)
    real_df["question_unittest_id"] = real_df["question_unittest_id"].astype(str)

    llm_sids = set(llm_df["student_id"])
    llm_qids = set(llm_df["question_unittest_id"])
    real_df = real_df[
        real_df["student_id"].isin(llm_sids) &
        real_df["question_unittest_id"].isin(llm_qids)
    ].copy()

    real_df = real_df.sort_values(["student_id", "question_unittest_id", "timestamp"])
    real_df["attempt_num"] = real_df.groupby(
        ["student_id", "question_unittest_id"]
    ).cumcount() + 1
    real_df = real_df[real_df["attempt_num"] <= MAX_ATTEMPTS].copy()
    real_df["source"] = "student"

    print(f"  LLM submissions: {len(llm_df)}")
    print(f"  Real student submissions (matched): {len(real_df)}")
    return llm_df, real_df


def build_trajectories(llm_df, real_df):
    print("Building trajectory metrics...")
    keep_cols = ["student_id", "question_unittest_id", "attempt_num", "source", "response", "pass"]
    combined = pd.concat([
        llm_df[keep_cols].copy(),
        real_df[keep_cols].copy(),
    ], ignore_index=True)

    combined["code"] = combined["response"].fillna("").astype(str)
    combined["pass_fraction"] = combined["pass"].apply(_pass_fraction)
    combined["char_count"] = combined["code"].str.len()

    # Code metrics
    metrics_list = []
    for code in combined["code"]:
        metrics_list.append(_code_metrics(code))
    metrics_df = pd.DataFrame(metrics_list)
    for col in metrics_df.columns:
        combined[col] = metrics_df[col].values

    # Edit distance from previous attempt
    combined["edit_distance_from_prev"] = np.nan
    for (sid, qid, src), grp in combined.groupby(["student_id", "question_unittest_id", "source"]):
        idx = grp.sort_values("attempt_num").index
        for i in range(1, len(idx)):
            prev_code = combined.loc[idx[i - 1], "code"][:2000]
            curr_code = combined.loc[idx[i], "code"][:2000]
            combined.loc[idx[i], "edit_distance_from_prev"] = levenshtein_distance(prev_code, curr_code)

    # Similarity to final attempt
    combined["similarity_to_final"] = np.nan
    for (sid, qid, src), grp in combined.groupby(["student_id", "question_unittest_id", "source"]):
        idx = grp.sort_values("attempt_num").index
        final_code = combined.loc[idx[-1], "code"]
        for i in idx:
            combined.loc[i, "similarity_to_final"] = _code_similarity(
                combined.loc[i, "code"], final_code
            )

    combined = combined.drop(columns=["response", "code", "pass"])
    print(f"  Trajectory rows: {len(combined)}")
    return combined


def _plot_carry_forward(ax, traj, metric, source_label, color, label, n_sample=200):
    """Background trajectories (carry-forward) + dashed aggregate mean."""
    x = np.arange(1, MAX_ATTEMPTS + 1)
    sub = traj[traj["source"] == source_label].copy()

    all_trajs = []
    for (sid, qid), grp in sub.groupby(["student_id", "question_unittest_id"]):
        grp = grp.sort_values("attempt_num")
        vals = grp.set_index("attempt_num")[metric].reindex(x)
        vals = vals.ffill()
        if vals.notna().sum() > 1:
            all_trajs.append(vals.values)

    if not all_trajs:
        return

    np.random.seed(42)
    sample_idx = np.random.choice(len(all_trajs), size=min(n_sample, len(all_trajs)), replace=False)
    for i in sample_idx:
        ax.plot(x, all_trajs[i], color=color, alpha=0.15, linewidth=0.5)

    arr = np.array(all_trajs)
    means = np.nanmean(arr, axis=0)
    n_trajs = len(all_trajs)
    ax.plot(x, means, color=color, linewidth=2.5, linestyle="--",
            label=label, zorder=9)


def _plot_raw(ax, traj, metric, source_label, color, label, n_sample=200, min_attempt=1):
    """Background trajectories (no carry-forward) + dashed aggregate mean."""
    x = np.arange(min_attempt, MAX_ATTEMPTS + 1)
    sub = traj[(traj["source"] == source_label) & (traj["attempt_num"] >= min_attempt)].copy()

    trajs = []
    for (sid, qid), grp in sub.groupby(["student_id", "question_unittest_id"]):
        grp = grp.sort_values("attempt_num")
        vals = grp.set_index("attempt_num")[metric].reindex(x)
        if vals.notna().sum() > 1:
            trajs.append((x, vals.values))

    if not trajs:
        return

    np.random.seed(42)
    sample_idx = np.random.choice(len(trajs), size=min(n_sample, len(trajs)), replace=False)
    for i in sample_idx:
        tx, tv = trajs[i]
        valid = ~np.isnan(tv)
        ax.plot(tx[valid], tv[valid], color=color, alpha=0.06, linewidth=0.3)

    means = []
    for a in x:
        vals = sub[sub["attempt_num"] == a][metric].dropna()
        means.append(vals.mean() if len(vals) > 0 else np.nan)
    means = np.array(means)
    valid = ~np.isnan(means)
    ax.plot(x[valid], means[valid], color=color, linewidth=2.5, linestyle="--",
            label=label, zorder=9)


def fig_combined(traj):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # Left: Score trajectory (carry-forward)
    ax = axes[0]
    _plot_carry_forward(ax, traj, "pass_fraction", "student", COLORS["student"], "Student")
    _plot_carry_forward(ax, traj, "pass_fraction", "llm", COLORS["llm"], "LLM")
    ax.set_xlabel("Attempts")
    ax.set_ylabel("Score")
    ax.set_xlim(1, MAX_ATTEMPTS)
    ax.set_ylim(0.5, 1.0)

    # Right: Edit distance trajectory
    ax = axes[1]
    _plot_raw(ax, traj, "edit_distance_from_prev", "student", COLORS["student"], "Student", min_attempt=2)
    _plot_raw(ax, traj, "edit_distance_from_prev", "llm", COLORS["llm"], "LLM", min_attempt=2)
    ax.set_xlabel("Attempts")
    ax.set_ylabel("Edit Distance")
    ax.set_xlim(2, MAX_ATTEMPTS)
    ax.set_ylim(0, 1000)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUT_DIR, "behavioral_combined.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache_path = os.path.join(OUT_DIR, "trajectory_metrics.csv")
    if os.path.exists(cache_path):
        print(f"Loading cached trajectories from {cache_path}")
        traj = pd.read_csv(cache_path)
    else:
        llm_df, real_df = load_data()
        traj = build_trajectories(llm_df, real_df)
        traj.to_csv(cache_path, index=False)
        print(f"  Cached to {cache_path}")

    n_student = len(traj[traj["source"] == "student"])
    n_llm = len(traj[traj["source"] == "llm"])
    print(f"\nTrajectory data: {n_student} student rows, {n_llm} LLM rows")

    fig_combined(traj)

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
