"""Test LLM adapter: evaluate on ALL matched pairs, no train/test split.

The LLM doesn't train on anything, so the temporal split is meaningless.
Just match every (student, question) pair with both sim and real data.
"""

import json
import logging
import os
from glob import glob

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO)

HF_SIM_REPO = "CodeInsightTeam/simulation_output"
HF_REAL_REPO = "CodeInsightTeam/code_insights_csv"


def _pass_fraction(s):
    s = str(s).strip()
    if not s or s == "nan":
        return np.nan
    return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan


# 1. Load simulation data
print("=" * 60)
print("STEP 1: Load simulation data")
print("=" * 60)
repo_dir = snapshot_download(
    repo_id=HF_SIM_REPO, repo_type="dataset", local_files_only=True,
)
merged_path = os.path.join(repo_dir, "v4_profile_mindiff", "glm_v4_merged.jsonl")
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

print(f"  Rows: {len(sim_df)}")
print(f"  Students: {sim_df['student_id'].nunique()}")
print(f"  Questions: {sim_df['question_unittest_id'].nunique()}")
print(f"  Response types: {sim_df['response_type'].value_counts().to_dict()}")
print(f"  Sample rows:")
print(sim_df.head(3).to_string(index=False))

# 2. Extract first-submit predictions (one per student-question pair)
print("\n" + "=" * 60)
print("STEP 2: First-submit predictions")
print("=" * 60)
submits = sim_df[sim_df["response_type"] == "Submit"].sort_values("attempt_id")
first_submit = submits.groupby(["student_id", "question_unittest_id"]).first().reset_index()
first_submit["y_pred"] = first_submit["pass"].apply(_pass_fraction)
first_submit = first_submit.dropna(subset=["y_pred"])

print(f"  Pairs: {len(first_submit)}")
print(f"  y_pred distribution:")
print(f"    mean={first_submit['y_pred'].mean():.3f}, median={first_submit['y_pred'].median():.3f}")
print(f"  y_pred value counts (top 10):")
print(first_submit["y_pred"].value_counts().head(10).to_string())
print(f"\n  Sample:")
print(first_submit[["student_id", "question_unittest_id", "pass", "y_pred"]].head(10).to_string(index=False))

# 3. Also extract best-submit predictions
print("\n" + "=" * 60)
print("STEP 3: Best-submit predictions")
print("=" * 60)
submits_scored = sim_df[sim_df["response_type"] == "Submit"].copy()
submits_scored["score"] = submits_scored["pass"].apply(_pass_fraction)
submits_scored = submits_scored.dropna(subset=["score"])
idx = submits_scored.groupby(["student_id", "question_unittest_id"])["score"].idxmax()
best_submit = submits_scored.loc[idx].reset_index(drop=True)
best_submit = best_submit.rename(columns={"score": "y_pred_best"})

print(f"  Pairs: {len(best_submit)}")
print(f"  y_pred_best distribution:")
print(f"    mean={best_submit['y_pred_best'].mean():.3f}, median={best_submit['y_pred_best'].median():.3f}")
print(f"  y_pred_best value counts (top 10):")
print(best_submit["y_pred_best"].value_counts().head(10).to_string())

# 4. Load real data
print("\n" + "=" * 60)
print("STEP 4: Load real student data")
print("=" * 60)
csv_dir = snapshot_download(
    repo_id=HF_REAL_REPO, repo_type="dataset", local_files_only=True,
)
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

# Real first submit per (student, question)
real_submits = real_df[real_df["response_type"] == "Submit"].copy()
real_submits["attempt_id"] = pd.to_numeric(real_submits.get("attempt_id", 0), errors="coerce").fillna(0)
real_submits = real_submits.sort_values("attempt_id")
real_first = real_submits.groupby(
    ["student_id", "question_unittest_id"]
).first().reset_index()
real_first["y_true_first"] = (real_first["score"] >= 1.0).astype(float)
real_first["y_true_first_frac"] = real_first["score"]

# Real best score per (student, question)
real_best = real_df.groupby(
    ["student_id", "question_unittest_id"]
)["score"].max().reset_index()
real_best["y_true_best"] = (real_best["score"] >= 1.0).astype(float)
real_best["y_true_best_frac"] = real_best["score"]

print(f"  Real submission rows: {len(real_df)}")
print(f"  Real unique pairs (first): {len(real_first)}")
print(f"  Real unique pairs (best):  {len(real_best)}")
print(f"  Real first-submit pass rate: {real_first['y_true_first'].mean():.3f}")
print(f"  Real best-submit pass rate:  {real_best['y_true_best'].mean():.3f}")
print(f"  Students: {real_best['student_id'].nunique()}")
print(f"  Questions: {real_best['question_unittest_id'].nunique()}")

# 5. Inner join — all matched pairs, like-for-like
print("\n" + "=" * 60)
print("STEP 5: Inner join — ALL matched pairs (like-for-like)")
print("=" * 60)

# Join sim first with real first
merged = first_submit[["student_id", "question_unittest_id", "y_pred"]].merge(
    real_first[["student_id", "question_unittest_id", "y_true_first", "y_true_first_frac"]],
    on=["student_id", "question_unittest_id"],
    how="inner",
)
# Join sim best with real best
merged = merged.merge(
    best_submit[["student_id", "question_unittest_id", "y_pred_best"]],
    on=["student_id", "question_unittest_id"],
    how="left",
)
merged = merged.merge(
    real_best[["student_id", "question_unittest_id", "y_true_best", "y_true_best_frac"]],
    on=["student_id", "question_unittest_id"],
    how="left",
)

print(f"  Matched pairs: {len(merged)}")
print(f"  (sim had {len(first_submit)}, real first had {len(real_first)}, real best had {len(real_best)})")
print(f"\n  Sim first submit y_pred:      mean={merged['y_pred'].mean():.3f}")
print(f"  Real first submit y_true:     mean={merged['y_true_first'].mean():.3f}")
print(f"  Sim best submit y_pred_best:  mean={merged['y_pred_best'].mean():.3f}")
print(f"  Real best submit y_true_best: mean={merged['y_true_best'].mean():.3f}")
print(f"\n  Sample matched rows:")
cols = ["student_id", "question_unittest_id", "y_pred", "y_true_first", "y_pred_best", "y_true_best"]
print(merged[cols].head(20).to_string(index=False))

# 6. Compute metrics — like-for-like comparisons
print("\n" + "=" * 60)
print("STEP 6: Metrics (like-for-like)")
print("=" * 60)

# First vs First (binary: full pass or not)
auc_ff = roc_auc_score(merged["y_true_first"], merged["y_pred"])
acc_ff = accuracy_score(merged["y_true_first"], (merged["y_pred"] >= 0.5).astype(int))
print(f"  Sim first  vs Real first  (binary):  AUC={auc_ff:.4f}, Acc={acc_ff:.4f}")

# Best vs Best (binary: full pass or not)
auc_bb = roc_auc_score(merged["y_true_best"], merged["y_pred_best"])
acc_bb = accuracy_score(merged["y_true_best"], (merged["y_pred_best"] >= 0.5).astype(int))
print(f"  Sim best   vs Real best   (binary):  AUC={auc_bb:.4f}, Acc={acc_bb:.4f}")

# Also: continuous score correlation (pass fraction vs pass fraction)
from scipy.stats import pearsonr, spearmanr
r_first, p_first = pearsonr(merged["y_pred"], merged["y_true_first_frac"])
r_best, p_best = pearsonr(merged["y_pred_best"], merged["y_true_best_frac"])
rho_first, _ = spearmanr(merged["y_pred"], merged["y_true_first_frac"])
rho_best, _ = spearmanr(merged["y_pred_best"], merged["y_true_best_frac"])
print(f"\n  Sim first  vs Real first  (continuous): Pearson={r_first:.4f} (p={p_first:.2e}), Spearman={rho_first:.4f}")
print(f"  Sim best   vs Real best   (continuous): Pearson={r_best:.4f} (p={p_best:.2e}), Spearman={rho_best:.4f}")

# Cross comparisons (the unfair ones we were doing before)
auc_fb = roc_auc_score(merged["y_true_best"], merged["y_pred"])
auc_bf = roc_auc_score(merged["y_true_first"], merged["y_pred_best"])
print(f"\n  Cross (unfair):")
print(f"  Sim first  vs Real best:   AUC={auc_fb:.4f}")
print(f"  Sim best   vs Real first:  AUC={auc_bf:.4f}")

# 7. Sanity check: y_pred conditioned on y_true
print("\n" + "=" * 60)
print("STEP 7: Sanity checks — mean y_pred by y_true group")
print("=" * 60)
for mode, pred_col, true_col in [
    ("first vs first", "y_pred", "y_true_first"),
    ("best vs best", "y_pred_best", "y_true_best"),
]:
    print(f"\n  {mode}:")
    for label, val in [(1, 1.0), (0, 0.0)]:
        subset = merged[merged[true_col] == val]
        print(f"    {true_col}={label}: n={len(subset):>6}, "
              f"y_pred mean={subset[pred_col].mean():.4f}, "
              f"median={subset[pred_col].median():.4f}")
    diff = merged[merged[true_col] == 1][pred_col].mean() - merged[merged[true_col] == 0][pred_col].mean()
    print(f"    Gap (positive = signal): {diff:+.4f}")
