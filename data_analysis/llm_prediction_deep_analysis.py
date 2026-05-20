"""Kendall tau decomposition: Does the LLM model question difficulty but not student ability?

Loads LLM predictions from local JSONL, aligns with ground truth via
load_student_split_data, and produces the decomposition test figure.

Usage:
    python data_analysis/llm_prediction_deep_analysis.py
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
from scipy.stats import kendalltau

from dynamic_models.temporal_eval.data_loader import load_student_split_data

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.facecolor": "white",
})

JSONL_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "results", "llm_student_eval", "dsa_hk231", "claude_attempts10.jsonl",
)
COURSE = "dsa_hk231"
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "llm_predictor", "student_split",
)


def _pass_fraction(s):
    s = str(s).strip()
    if not s or s == "nan":
        return np.nan
    return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan


def load_and_align():
    with open(JSONL_PATH) as f:
        rows = [json.loads(l) for l in f]

    llm_df = pd.DataFrame(rows)
    llm_df = llm_df[llm_df["response_type"] == "Submit"].copy()
    llm_df["attempt_id"] = pd.to_numeric(llm_df["attempt_id"], errors="coerce")
    llm_df = llm_df.sort_values("attempt_id")
    llm_last = llm_df.groupby(["student_id", "question_unittest_id"]).last().reset_index()

    data, split = load_student_split_data(COURSE)
    qi = data.question_infos
    test_item_set = set(split.test_item_indices.tolist())

    hf_dir = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        repo_type="dataset", local_files_only=True,
    )
    hf_qi = pd.read_csv(f"{hf_dir}/question_infos.csv")
    qid_to_name = {
        str(int(row["question_id"])): row["question_name"]
        for _, row in hf_qi.iterrows()
    }

    sid_to_idx = {}
    for idx in split.test_student_indices.tolist():
        sid_to_idx[str(data.student_ids[idx])] = idx

    merged_rows = []
    for _, row in llm_last.iterrows():
        sid = str(row["student_id"])
        qid = str(row["question_unittest_id"])

        if sid not in sid_to_idx:
            continue
        s_idx = sid_to_idx[sid]

        qname = qid_to_name.get(qid, "")
        q_items = [i for i in qi[qi["qname"] == qname].index if i in test_item_set]
        if not q_items:
            continue

        real_results = []
        for qidx in q_items:
            obs = data.correctness_matrix[s_idx, qidx, :].numpy()
            valid = obs[obs != -1]
            if len(valid) > 0:
                real_results.append(int(valid[-1]))

        if not real_results:
            continue

        llm_pass_str = str(row["pass"])
        llm_binary = [int(c) for c in llm_pass_str if c in "01"]

        n = min(len(real_results), len(llm_binary))
        if n == 0:
            continue

        merged_rows.append({
            "student_id": sid,
            "question_unittest_id": qid,
            "y_pred": sum(llm_binary[:n]) / n,
            "y_true": sum(real_results[:n]) / n,
        })

    merged = pd.DataFrame(merged_rows)
    print(f"  Aligned {len(merged)} (student, question) pairs")
    print(f"  {merged['student_id'].nunique()} students, "
          f"{merged['question_unittest_id'].nunique()} questions")
    return merged


def fig_decomposition_test(merged):
    m = merged.copy()

    # Center by question (remove difficulty signal)
    q_mean_pred = m.groupby("question_unittest_id")["y_pred"].transform("mean")
    q_mean_true = m.groupby("question_unittest_id")["y_true"].transform("mean")
    m["y_pred_cq"] = m["y_pred"] - q_mean_pred
    m["y_true_cq"] = m["y_true"] - q_mean_true

    # Center by student (remove ability signal)
    s_mean_pred = m.groupby("student_id")["y_pred"].transform("mean")
    s_mean_true = m.groupby("student_id")["y_true"].transform("mean")
    m["y_pred_cs"] = m["y_pred"] - s_mean_pred
    m["y_true_cs"] = m["y_true"] - s_mean_true

    # Per-student tau: original vs centered-by-question
    orig_s_taus, centered_s_taus = [], []
    for _, grp in m.groupby("student_id"):
        if len(grp) < 5 or grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        tau_o, _ = kendalltau(grp["y_pred"], grp["y_true"])
        if not np.isnan(tau_o):
            orig_s_taus.append(tau_o)
        if grp["y_pred_cq"].std() > 0 and grp["y_true_cq"].std() > 0:
            tau_c, _ = kendalltau(grp["y_pred_cq"], grp["y_true_cq"])
            if not np.isnan(tau_c):
                centered_s_taus.append(tau_c)

    # Per-question tau: original vs centered-by-student
    orig_q_taus, centered_q_taus = [], []
    for _, grp in m.groupby("question_unittest_id"):
        if len(grp) < 5 or grp["y_pred"].std() == 0 or grp["y_true"].std() == 0:
            continue
        tau_o, _ = kendalltau(grp["y_pred"], grp["y_true"])
        if not np.isnan(tau_o):
            orig_q_taus.append(tau_o)
        if grp["y_pred_cs"].std() > 0 and grp["y_true_cs"].std() > 0:
            tau_c, _ = kendalltau(grp["y_pred_cs"], grp["y_true_cs"])
            if not np.isnan(tau_c):
                centered_q_taus.append(tau_c)

    orig_s_taus = np.array(orig_s_taus)
    centered_s_taus = np.array(centered_s_taus)
    orig_q_taus = np.array(orig_q_taus)
    centered_q_taus = np.array(centered_q_taus)

    # Per-student tau figure (2x1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    ax = axes[0]
    ax.hist(orig_s_taus, bins=18, color="#6A9B59", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(orig_s_taus.mean(), color="#6A9B59", linestyle="--", linewidth=2)
    ax.set_xlabel(r"Per-Student Kendall $\tau$", fontsize=14)
    ax.set_ylabel("Number of Students", fontsize=14)
    ax.tick_params(labelsize=12)

    ax = axes[1]
    ax.hist(orig_s_taus, bins=18, alpha=0.6, color="#6A9B59", edgecolor="white",
            label="Original")
    ax.hist(centered_s_taus, bins=18, alpha=0.6, color="#C44E52", edgecolor="white",
            label="Centered")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"Per-Student Kendall $\tau$", fontsize=14)
    ax.set_ylabel("Number of Students", fontsize=14)
    ax.set_title("Remove Question Difficulty", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    all_s = np.concatenate([orig_s_taus, centered_s_taus])
    s_lim = (min(all_s.min(), -0.05) - 0.05, max(all_s.max(), 0.05) + 0.05)
    axes[0].set_xlim(s_lim)
    axes[1].set_xlim(s_lim)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "decomposition_test.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Per-question tau figure (2x1)
    fig2, axes2 = plt.subplots(2, 1, figsize=(8, 10))

    ax = axes2[0]
    ax.hist(orig_q_taus, bins=18, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(orig_q_taus.mean(), color="#4C72B0", linestyle="--", linewidth=2)
    ax.set_xlabel(r"Per-Question Kendall $\tau$", fontsize=14)
    ax.set_ylabel("Number of Questions", fontsize=14)
    ax.tick_params(labelsize=12)

    ax = axes2[1]
    ax.hist(orig_q_taus, bins=18, alpha=0.6, color="#4C72B0", edgecolor="white",
            label="Original")
    ax.hist(centered_q_taus, bins=18, alpha=0.6, color="#C44E52", edgecolor="white",
            label="Centered")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"Per-Question Kendall $\tau$", fontsize=14)
    ax.set_ylabel("Number of Questions", fontsize=14)
    ax.set_title("Remove Student Ability", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    all_q = np.concatenate([orig_q_taus, centered_q_taus])
    q_lim = (min(all_q.min(), -0.05) - 0.05, max(all_q.max(), 0.05) + 0.05)
    axes2[0].set_xlim(q_lim)
    axes2[1].set_xlim(q_lim)

    fig2.tight_layout()
    path2 = os.path.join(OUT_DIR, "decomposition_test_question.png")
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path2}")

    s_drop = (orig_s_taus.mean() - centered_s_taus.mean()) / max(abs(orig_s_taus.mean()), 1e-9) * 100
    q_drop = (orig_q_taus.mean() - centered_q_taus.mean()) / max(abs(orig_q_taus.mean()), 1e-9) * 100

    decomp = {
        "orig_student_tau_mean": orig_s_taus.mean(),
        "centered_student_tau_mean": centered_s_taus.mean(),
        "drop_pct_student": s_drop,
        "orig_question_tau_mean": orig_q_taus.mean(),
        "centered_question_tau_mean": centered_q_taus.mean(),
        "drop_pct_question": q_drop,
    }

    print(f"  Per-student tau: original={orig_s_taus.mean():.4f} -> "
          f"centered={centered_s_taus.mean():.4f} (drop={s_drop:.0f}%)")
    print(f"  Per-question tau: original={orig_q_taus.mean():.4f} -> "
          f"centered={centered_q_taus.mean():.4f} (drop={q_drop:.0f}%)")
    return decomp


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading and aligning LLM predictions with ground truth...")
    merged = load_and_align()

    print("\nDecomposition test...")
    decomp = fig_decomposition_test(merged)

    decomp_df = pd.DataFrame([decomp])
    path = os.path.join(OUT_DIR, "decomposition_summary.csv")
    decomp_df.to_csv(path, index=False)
    print(f"  Saved {path}")

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
