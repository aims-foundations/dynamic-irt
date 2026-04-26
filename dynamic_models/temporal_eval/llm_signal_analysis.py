"""Multi-strategy signal extraction from LLM simulation data.

Tries 4 strategies to extract predictive signal from simulation data:
1. v1 vs v4 variant comparison (does prompting strategy matter?)
2. Iteration trajectory features + logistic regression
3. Calibrated relative predictions (z-scores)
4. Question-difficulty-only baseline

Usage:
    python -m dynamic_models.temporal_eval.llm_signal_analysis
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

HF_SIM_REPO = "CodeInsightTeam/simulation_output"
HF_REAL_REPO = "CodeInsightTeam/code_insights_csv"
OUTPUT_DIR = os.path.join("results", "temporal_eval")


def _pass_fraction(s) -> float:
    s = str(s).strip()
    if not s or s == "nan":
        return np.nan
    return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_v4_data() -> pd.DataFrame:
    repo_dir = snapshot_download(
        repo_id=HF_SIM_REPO, repo_type="dataset", local_files_only=True,
    )
    merged = os.path.join(
        repo_dir, "v4_profile_mindiff", "glm_v4_merged.jsonl"
    )
    if not os.path.exists(merged):
        raise FileNotFoundError(f"v4 merged JSONL not found at {merged}")

    keep = [
        "student_id", "question_unittest_id", "attempt_id",
        "response_type", "pass",
    ]
    rows = []
    with open(merged) as f:
        for line in f:
            rec = json.loads(line)
            rows.append({k: rec.get(k) for k in keep})
    df = pd.DataFrame(rows)
    df["student_id"] = df["student_id"].astype(str)
    df["question_unittest_id"] = pd.to_numeric(df["question_unittest_id"], errors="coerce")
    df["attempt_id"] = pd.to_numeric(df["attempt_id"], errors="coerce")
    df = df.dropna(subset=["question_unittest_id"])
    df["question_unittest_id"] = df["question_unittest_id"].astype(int)
    df["score"] = df["pass"].apply(_pass_fraction)
    print(f"Loaded v4: {len(df)} rows, {df['student_id'].nunique()} students, "
          f"{df['question_unittest_id'].nunique()} questions")
    return df


def load_v1_data() -> pd.DataFrame:
    repo_dir = snapshot_download(
        repo_id=HF_SIM_REPO, repo_type="dataset", local_files_only=True,
    )
    v1_dir = os.path.join(repo_dir, "v1_rewrite_prompt")
    if not os.path.isdir(v1_dir):
        # Try older snapshot
        snapshots_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub",
            "datasets--CodeInsightTeam--simulation_output", "snapshots",
        )
        for snap in os.listdir(snapshots_dir):
            candidate = os.path.join(snapshots_dir, snap, "v1_rewrite_prompt")
            if os.path.isdir(candidate):
                v1_dir = candidate
                break

    files = sorted(glob(os.path.join(v1_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No v1 JSONL files found in {v1_dir}")

    keep = [
        "student_id", "question_unittest_id", "attempt_id",
        "response_type", "pass",
    ]
    rows = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                rec = json.loads(line)
                rows.append({k: rec.get(k) for k in keep})
    df = pd.DataFrame(rows)
    df["student_id"] = df["student_id"].astype(str)
    df["question_unittest_id"] = pd.to_numeric(df["question_unittest_id"], errors="coerce")
    df["attempt_id"] = pd.to_numeric(df["attempt_id"], errors="coerce")
    df = df.dropna(subset=["question_unittest_id"])
    df["question_unittest_id"] = df["question_unittest_id"].astype(int)
    df["score"] = df["pass"].apply(_pass_fraction)
    # Deduplicate: keep all shards but remove rows that appear in both merged and shard
    df = df.drop_duplicates(subset=["student_id", "question_unittest_id", "attempt_id", "response_type"])
    print(f"Loaded v1: {len(df)} rows, {df['student_id'].nunique()} students, "
          f"{df['question_unittest_id'].nunique()} questions")
    return df


def load_real_data() -> Tuple[pd.DataFrame, Dict[int, int]]:
    repo_dir = snapshot_download(
        repo_id=HF_REAL_REPO, repo_type="dataset", local_files_only=True,
    )
    main_data = pd.read_csv(
        os.path.join(repo_dir, "main_data.csv"),
        low_memory=False, on_bad_lines="skip",
    )
    question_infos = pd.read_csv(os.path.join(repo_dir, "question_infos.csv"))

    main_data = main_data[
        main_data["response_type"].isin(["Submit", "Prechecked"])
    ].copy()
    main_data = main_data.dropna(subset=["pass"])
    main_data["student_id"] = main_data["student_id"].astype(str)
    main_data["question_unittest_id"] = pd.to_numeric(
        main_data["question_unittest_id"], errors="coerce"
    )
    main_data = main_data.dropna(subset=["question_unittest_id"])
    main_data["question_unittest_id"] = main_data["question_unittest_id"].astype(int)
    main_data["score"] = main_data["pass"].apply(_pass_fraction)
    main_data = main_data.dropna(subset=["score"])

    qid_to_week = dict(zip(
        question_infos["question_id"].astype(int),
        question_infos["week"].astype(int),
    ))
    main_data["week"] = main_data["question_unittest_id"].map(qid_to_week)

    print(f"Loaded real: {len(main_data)} rows, "
          f"{main_data['student_id'].nunique()} students")
    return main_data, qid_to_week


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_first_submit(sim_df: pd.DataFrame) -> pd.DataFrame:
    submits = sim_df[sim_df["response_type"] == "Submit"].copy()
    submits = submits.sort_values("attempt_id")
    first = submits.groupby(
        ["student_id", "question_unittest_id"]
    ).first().reset_index()
    return first[["student_id", "question_unittest_id", "score"]].rename(
        columns={"score": "first_submit_score"}
    )


def extract_best_submit(sim_df: pd.DataFrame) -> pd.DataFrame:
    submits = sim_df[sim_df["response_type"] == "Submit"].copy()
    submits = submits.dropna(subset=["score"])
    idx = submits.groupby(
        ["student_id", "question_unittest_id"]
    )["score"].idxmax()
    best = submits.loc[idx].reset_index(drop=True)
    return best[["student_id", "question_unittest_id", "score"]].rename(
        columns={"score": "best_submit_score"}
    )


def extract_trajectory_features(sim_df: pd.DataFrame) -> pd.DataFrame:
    submits = sim_df[sim_df["response_type"] == "Submit"].copy()
    submits = submits.dropna(subset=["score"])
    submits = submits.sort_values(["student_id", "question_unittest_id", "attempt_id"])

    prechecks = sim_df[sim_df["response_type"] == "Prechecked"].copy()
    precheck_counts = prechecks.groupby(
        ["student_id", "question_unittest_id"]
    ).size().reset_index(name="n_prechecks")

    features = []
    for (sid, qid), grp in submits.groupby(["student_id", "question_unittest_id"]):
        scores = grp["score"].values
        n = len(scores)
        first = scores[0]
        best = scores.max()
        final = scores[-1]
        best_idx = int(np.argmax(scores))

        diffs = np.diff(scores)
        stalls = int(np.sum(diffs == 0))
        improvements = int(np.sum(diffs > 0))

        features.append({
            "student_id": sid,
            "question_unittest_id": qid,
            "first_score": first,
            "best_score": best,
            "final_score": final,
            "n_attempts": n,
            "attempts_to_max": best_idx + 1,
            "improvement_rate": (best - first) / max(n, 1),
            "stall_count": stalls,
            "improvement_count": improvements,
            "score_variance": float(np.var(scores)) if n > 1 else 0.0,
            "any_full_pass": float(best >= 1.0),
        })

    feat_df = pd.DataFrame(features)
    feat_df = feat_df.merge(precheck_counts, on=["student_id", "question_unittest_id"], how="left")
    feat_df["n_prechecks"] = feat_df["n_prechecks"].fillna(0)
    feat_df["precheck_ratio"] = feat_df["n_prechecks"] / (feat_df["n_attempts"] + feat_df["n_prechecks"])
    return feat_df


def get_real_outcomes(real_df: pd.DataFrame, qid_to_week: Dict[int, int],
                      cutoff_week: int) -> pd.DataFrame:
    real_df = real_df.copy()
    real_df["week"] = real_df["question_unittest_id"].map(qid_to_week)
    test = real_df[real_df["week"] > cutoff_week].copy()
    best = test.groupby(
        ["student_id", "question_unittest_id"]
    )["score"].max().reset_index()
    best["y_true"] = (best["score"] >= 1.0).astype(float)
    return best[["student_id", "question_unittest_id", "y_true"]]


def get_student_base_rates(real_df: pd.DataFrame, qid_to_week: Dict[int, int],
                           cutoff_week: int) -> pd.Series:
    real_df = real_df.copy()
    real_df["week"] = real_df["question_unittest_id"].map(qid_to_week)
    train = real_df[real_df["week"] <= cutoff_week]
    train_best = train.groupby(
        ["student_id", "question_unittest_id"]
    )["score"].max().reset_index()
    train_best["full_pass"] = (train_best["score"] >= 1.0).astype(float)
    return train_best.groupby("student_id")["full_pass"].mean()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strategy_variant_comparison(
    v4_df: pd.DataFrame, v1_df: pd.DataFrame,
    real_outcomes: pd.DataFrame, qid_to_week: Dict[int, int],
    cutoff_week: int,
) -> Dict[str, Dict[str, float]]:
    results = {}

    for name, sim_df in [("v4_first", v4_df), ("v4_best", v4_df),
                          ("v1_first", v1_df), ("v1_best", v1_df)]:
        variant = name.split("_")[0]
        mode = name.split("_")[1]

        if mode == "first":
            preds = extract_first_submit(sim_df)
            pred_col = "first_submit_score"
        else:
            preds = extract_best_submit(sim_df)
            pred_col = "best_submit_score"

        # Filter to test weeks
        preds["week"] = preds["question_unittest_id"].map(qid_to_week)
        preds = preds[preds["week"] > cutoff_week].copy()

        merged = preds.merge(real_outcomes, on=["student_id", "question_unittest_id"])
        if len(merged) < 10:
            results[name] = {"auc": np.nan, "accuracy": np.nan, "n": len(merged)}
            continue

        y_true = merged["y_true"].values
        y_pred = merged[pred_col].values

        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        results[name] = {"auc": auc, "accuracy": acc, "n": len(merged)}

    return results


def strategy_trajectory_logreg(
    v4_df: pd.DataFrame, real_df: pd.DataFrame,
    qid_to_week: Dict[int, int], cutoff_week: int,
) -> Dict[str, float]:
    feat_df = extract_trajectory_features(v4_df)

    # Add week info
    feat_df["week"] = feat_df["question_unittest_id"].map(qid_to_week)
    feat_df = feat_df.dropna(subset=["week"])

    # Get real outcomes for ALL weeks (train + test)
    real_best = real_df.groupby(
        ["student_id", "question_unittest_id"]
    )["score"].max().reset_index()
    real_best["y_true"] = (real_best["score"] >= 1.0).astype(float)

    merged = feat_df.merge(
        real_best[["student_id", "question_unittest_id", "y_true"]],
        on=["student_id", "question_unittest_id"],
    )

    feature_cols = [
        "first_score", "best_score", "final_score", "n_attempts",
        "attempts_to_max", "improvement_rate", "stall_count",
        "improvement_count", "score_variance", "any_full_pass",
        "n_prechecks", "precheck_ratio",
    ]

    train = merged[merged["week"] <= cutoff_week]
    test = merged[merged["week"] > cutoff_week]

    if len(train) < 20 or len(test) < 10:
        return {"auc": np.nan, "accuracy": np.nan, "n_train": len(train),
                "n_test": len(test), "coefficients": {}}

    X_train = train[feature_cols].fillna(0).values
    y_train = train["y_true"].values
    X_test = test[feature_cols].fillna(0).values
    y_test = test["y_true"].values

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auc": np.nan, "accuracy": np.nan, "n_train": len(train),
                "n_test": len(test), "coefficients": {}}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred_prob = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, (y_pred_prob >= 0.5).astype(int))

    coefs = dict(zip(feature_cols, model.coef_[0]))

    return {"auc": auc, "accuracy": acc, "n_train": len(train),
            "n_test": len(test), "coefficients": coefs}


def strategy_calibrated_relative(
    v4_df: pd.DataFrame, real_df: pd.DataFrame,
    real_outcomes: pd.DataFrame, qid_to_week: Dict[int, int],
    cutoff_week: int,
) -> Dict[str, float]:
    # Per-question LLM difficulty: average LLM score per question
    submits = v4_df[v4_df["response_type"] == "Submit"].copy()
    submits = submits.dropna(subset=["score"])
    best_per_pair = submits.groupby(
        ["student_id", "question_unittest_id"]
    )["score"].max().reset_index()

    q_difficulty = best_per_pair.groupby("question_unittest_id")["score"].mean()
    q_difficulty_std = best_per_pair.groupby("question_unittest_id")["score"].std().fillna(0.01)

    # Per-student LLM ability: average LLM score per student
    s_ability = best_per_pair.groupby("student_id")["score"].mean()

    # Z-score: how much better/worse this pair is vs expected
    best_per_pair = best_per_pair.copy()
    best_per_pair["q_mean"] = best_per_pair["question_unittest_id"].map(q_difficulty)
    best_per_pair["q_std"] = best_per_pair["question_unittest_id"].map(q_difficulty_std)
    best_per_pair["s_mean"] = best_per_pair["student_id"].map(s_ability)
    best_per_pair["z_score"] = (best_per_pair["score"] - best_per_pair["q_mean"]) / best_per_pair["q_std"].clip(lower=0.01)

    # Student base rate from real training data
    student_base_rate = get_student_base_rates(real_df, qid_to_week, cutoff_week)

    best_per_pair["student_base_rate"] = best_per_pair["student_id"].map(student_base_rate)
    best_per_pair["calibrated_pred"] = (
        best_per_pair["student_base_rate"].fillna(0.3) +
        0.1 * best_per_pair["z_score"]
    ).clip(0, 1)

    # Filter to test weeks
    best_per_pair["week"] = best_per_pair["question_unittest_id"].map(qid_to_week)
    test_preds = best_per_pair[best_per_pair["week"] > cutoff_week]

    merged = test_preds.merge(
        real_outcomes, on=["student_id", "question_unittest_id"]
    )

    if len(merged) < 10 or len(np.unique(merged["y_true"])) < 2:
        return {"auc": np.nan, "accuracy": np.nan, "n": len(merged)}

    auc = roc_auc_score(merged["y_true"], merged["calibrated_pred"])
    acc = accuracy_score(merged["y_true"], (merged["calibrated_pred"] >= 0.5).astype(int))

    # Also compute AUC for just the z-score (no student base rate)
    auc_zscore = roc_auc_score(merged["y_true"], merged["z_score"])

    return {"auc": auc, "accuracy": acc, "auc_zscore_only": auc_zscore, "n": len(merged)}


def strategy_question_difficulty(
    v4_df: pd.DataFrame, real_df: pd.DataFrame,
    real_outcomes: pd.DataFrame, qid_to_week: Dict[int, int],
    cutoff_week: int,
) -> Dict[str, float]:
    # LLM-derived question difficulty: fraction of simulated students who fully pass
    submits = v4_df[v4_df["response_type"] == "Submit"].copy()
    submits = submits.dropna(subset=["score"])
    best_per_pair = submits.groupby(
        ["student_id", "question_unittest_id"]
    )["score"].max().reset_index()
    best_per_pair["full_pass"] = (best_per_pair["score"] >= 1.0).astype(float)

    q_pass_rate = best_per_pair.groupby("question_unittest_id")["full_pass"].mean()

    # Student historical pass rate from training weeks
    student_base_rate = get_student_base_rates(real_df, qid_to_week, cutoff_week)

    # Predict: student_base_rate * question_pass_rate (simple product)
    test_outcomes = real_outcomes.copy()
    test_outcomes["q_pass_rate"] = test_outcomes["question_unittest_id"].map(q_pass_rate)
    test_outcomes["s_base_rate"] = test_outcomes["student_id"].map(student_base_rate)
    test_outcomes = test_outcomes.dropna(subset=["q_pass_rate"])

    # Strategy A: question difficulty only
    merged_q = test_outcomes.dropna(subset=["q_pass_rate"])
    # Strategy B: student * question
    merged_sq = test_outcomes.dropna(subset=["q_pass_rate", "s_base_rate"])
    merged_sq["pred_sq"] = merged_sq["s_base_rate"] * merged_sq["q_pass_rate"]

    results = {}
    if len(merged_q) >= 10 and len(np.unique(merged_q["y_true"])) >= 2:
        results["auc_q_only"] = roc_auc_score(merged_q["y_true"], merged_q["q_pass_rate"])
        results["acc_q_only"] = accuracy_score(
            merged_q["y_true"], (merged_q["q_pass_rate"] >= 0.5).astype(int)
        )
        results["n_q"] = len(merged_q)

        # Also: real question difficulty for comparison
        real_q = real_df.copy()
        real_q["week"] = real_q["question_unittest_id"].map(qid_to_week)
        # Use training-week questions to compute real difficulty
        train_real = real_q[real_q["week"] <= cutoff_week]
        train_best = train_real.groupby(
            ["student_id", "question_unittest_id"]
        )["score"].max().reset_index()
        train_best["full_pass"] = (train_best["score"] >= 1.0).astype(float)
        real_q_pass = train_best.groupby("question_unittest_id")["full_pass"].mean()

        # For test questions: use LLM difficulty vs real difficulty correlation
        test_qids = merged_q["question_unittest_id"].unique()
        both = pd.DataFrame({
            "qid": test_qids,
            "llm_diff": [q_pass_rate.get(q, np.nan) for q in test_qids],
            "real_diff": [real_q_pass.get(q, np.nan) for q in test_qids],
        }).dropna()
        if len(both) > 2:
            from scipy.stats import spearmanr
            corr, pval = spearmanr(both["llm_diff"], both["real_diff"])
            results["difficulty_spearman"] = corr
            results["difficulty_pval"] = pval
            results["n_questions_compared"] = len(both)

    if len(merged_sq) >= 10 and len(np.unique(merged_sq["y_true"])) >= 2:
        results["auc_sq"] = roc_auc_score(merged_sq["y_true"], merged_sq["pred_sq"])
        results["acc_sq"] = accuracy_score(
            merged_sq["y_true"], (merged_sq["pred_sq"] >= 0.5).astype(int)
        )
        results["n_sq"] = len(merged_sq)

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(all_results: Dict[int, Dict], output_dir: str) -> str:
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25})

    cutoffs = sorted(all_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Panel 1: Variant comparison AUCs
    ax = axes[0, 0]
    for variant_name in ["v4_first", "v4_best", "v1_first", "v1_best"]:
        aucs = [all_results[c]["strategy1"].get(variant_name, {}).get("auc", np.nan) for c in cutoffs]
        ax.plot(cutoffs, aucs, marker="o", label=variant_name)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Cutoff Week")
    ax.set_ylabel("AUC")
    ax.set_title("Strategy 1: v1 vs v4 Variant Comparison")
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 0.8)

    # Panel 2: Trajectory LogReg AUC
    ax = axes[0, 1]
    traj_aucs = [all_results[c]["strategy2"].get("auc", np.nan) for c in cutoffs]
    first_aucs = [all_results[c]["strategy1"].get("v4_first", {}).get("auc", np.nan) for c in cutoffs]
    ax.plot(cutoffs, traj_aucs, marker="s", color="tab:red", label="Trajectory LogReg")
    ax.plot(cutoffs, first_aucs, marker="o", color="tab:blue", alpha=0.5, label="v4 first submit (baseline)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cutoff Week")
    ax.set_ylabel("AUC")
    ax.set_title("Strategy 2: Trajectory Features + LogReg")
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 0.8)

    # Panel 3: Calibrated predictions
    ax = axes[1, 0]
    cal_aucs = [all_results[c]["strategy3"].get("auc", np.nan) for c in cutoffs]
    zscore_aucs = [all_results[c]["strategy3"].get("auc_zscore_only", np.nan) for c in cutoffs]
    ax.plot(cutoffs, cal_aucs, marker="D", color="tab:green", label="Calibrated (base rate + z-score)")
    ax.plot(cutoffs, zscore_aucs, marker="x", color="tab:olive", label="Z-score only")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cutoff Week")
    ax.set_ylabel("AUC")
    ax.set_title("Strategy 3: Calibrated Relative Predictions")
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 0.8)

    # Panel 4: Question difficulty
    ax = axes[1, 1]
    q_aucs = [all_results[c]["strategy4"].get("auc_q_only", np.nan) for c in cutoffs]
    sq_aucs = [all_results[c]["strategy4"].get("auc_sq", np.nan) for c in cutoffs]
    ax.plot(cutoffs, q_aucs, marker="^", color="tab:purple", label="Question difficulty only")
    ax.plot(cutoffs, sq_aucs, marker="v", color="tab:brown", label="Student x Question")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cutoff Week")
    ax.set_ylabel("AUC")
    ax.set_title("Strategy 4: Question-Difficulty Baseline")
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 0.8)

    fig.suptitle("LLM Signal Extraction: 4 Strategies Across Temporal Horizons", fontsize=14)

    save_path = os.path.join(output_dir, "llm_signal_strategies.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_logreg_coefficients(all_results: Dict[int, Dict], output_dir: str) -> str:
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    cutoffs = sorted(all_results.keys())
    # Collect all coefficient dicts
    all_coefs = {}
    for c in cutoffs:
        coefs = all_results[c]["strategy2"].get("coefficients", {})
        if coefs:
            all_coefs[c] = coefs

    if not all_coefs:
        return ""

    features = list(next(iter(all_coefs.values())).keys())
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    x = np.arange(len(features))
    width = 0.8 / len(all_coefs)
    for i, (cutoff, coefs) in enumerate(all_coefs.items()):
        vals = [coefs.get(f, 0) for f in features]
        ax.bar(x + i * width, vals, width, label=f"W={cutoff}", alpha=0.8)

    ax.set_xticks(x + width * len(all_coefs) / 2)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient")
    ax.set_title("Logistic Regression Feature Coefficients by Cutoff Week")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

    save_path = os.path.join(output_dir, "llm_logreg_coefficients.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_results: Dict[int, Dict], output_dir: str) -> str:
    cutoffs = sorted(all_results.keys())
    lines = [
        "# LLM Signal Extraction Analysis",
        "",
        "## Overview",
        "",
        "We tested 4 strategies to extract predictive signal from LLM simulation data.",
        "The baseline (v4 first-submit pass fraction) gives AUC ~0.49 (random).",
        "",
        "---",
        "",
        "## Strategy 1: v1 vs v4 Variant Comparison",
        "",
        "Does the prompting strategy matter? v1 uses rewritten prompts without student profiles;",
        "v4 adds student profiles + 10 few-shot examples.",
        "",
        "| Cutoff | v4 First | v4 Best | v1 First | v1 Best |",
        "|--------|----------|---------|----------|---------|",
    ]
    for c in cutoffs:
        s1 = all_results[c]["strategy1"]
        row = f"| W={c} |"
        for name in ["v4_first", "v4_best", "v1_first", "v1_best"]:
            d = s1.get(name, {})
            auc = d.get("auc", np.nan)
            n = d.get("n", 0)
            row += f" {auc:.3f} (n={n}) |" if not np.isnan(auc) else f" N/A (n={n}) |"
        lines.append(row)

    lines += [
        "",
        "---",
        "",
        "## Strategy 2: Iteration Trajectory Features + Logistic Regression",
        "",
        "Extract features from the full 50-attempt iteration sequence and train a logistic",
        "regression on training-week pairs to predict test-week outcomes.",
        "",
        "Features: first_score, best_score, final_score, n_attempts, attempts_to_max,",
        "improvement_rate, stall_count, improvement_count, score_variance, any_full_pass,",
        "n_prechecks, precheck_ratio.",
        "",
        "| Cutoff | AUC | Accuracy | N Train | N Test |",
        "|--------|-----|----------|---------|--------|",
    ]
    for c in cutoffs:
        s2 = all_results[c]["strategy2"]
        auc = s2.get("auc", np.nan)
        acc = s2.get("accuracy", np.nan)
        nt = s2.get("n_train", 0)
        ne = s2.get("n_test", 0)
        auc_s = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        acc_s = f"{acc:.3f}" if not np.isnan(acc) else "N/A"
        lines.append(f"| W={c} | {auc_s} | {acc_s} | {nt} | {ne} |")

    # Coefficients
    for c in cutoffs:
        coefs = all_results[c]["strategy2"].get("coefficients", {})
        if coefs:
            lines += [
                f"",
                f"**Coefficients at W={c}:**",
                "",
            ]
            sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
            for feat, val in sorted_coefs:
                lines.append(f"- {feat}: {val:+.4f}")

    lines += [
        "",
        "---",
        "",
        "## Strategy 3: Calibrated Relative Predictions",
        "",
        "Instead of raw pass fraction, compute per-question z-scores (how much better/worse",
        "did the LLM do vs its average) and combine with the student's real base rate from",
        "training weeks.",
        "",
        "| Cutoff | AUC (calibrated) | AUC (z-score only) | N |",
        "|--------|------------------|--------------------|---|",
    ]
    for c in cutoffs:
        s3 = all_results[c]["strategy3"]
        auc = s3.get("auc", np.nan)
        auc_z = s3.get("auc_zscore_only", np.nan)
        n = s3.get("n", 0)
        auc_s = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        auc_z_s = f"{auc_z:.3f}" if not np.isnan(auc_z) else "N/A"
        lines.append(f"| W={c} | {auc_s} | {auc_z_s} | {n} |")

    lines += [
        "",
        "---",
        "",
        "## Strategy 4: Question-Difficulty-Only Baseline",
        "",
        "Use LLM-derived question pass rates as difficulty estimates. Optionally combine with",
        "student historical pass rate.",
        "",
        "| Cutoff | AUC (Q only) | AUC (S x Q) | Difficulty Spearman | N Questions |",
        "|--------|--------------|-------------|--------------------:|------------|",
    ]
    for c in cutoffs:
        s4 = all_results[c]["strategy4"]
        auc_q = s4.get("auc_q_only", np.nan)
        auc_sq = s4.get("auc_sq", np.nan)
        spear = s4.get("difficulty_spearman", np.nan)
        nq = s4.get("n_questions_compared", 0)
        lines.append(
            f"| W={c} | {auc_q:.3f} | {auc_sq:.3f} | {spear:.3f} | {nq} |"
            if not np.isnan(auc_q) else f"| W={c} | N/A | N/A | N/A | {nq} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Summary Comparison",
        "",
        "Best AUC per strategy across all horizons:",
        "",
        "| Strategy | Best AUC | At Cutoff | Description |",
        "|----------|----------|-----------|-------------|",
    ]

    strategies = [
        ("v4 First Submit", lambda c: all_results[c]["strategy1"].get("v4_first", {}).get("auc", np.nan)),
        ("v4 Best Submit", lambda c: all_results[c]["strategy1"].get("v4_best", {}).get("auc", np.nan)),
        ("v1 First Submit", lambda c: all_results[c]["strategy1"].get("v1_first", {}).get("auc", np.nan)),
        ("v1 Best Submit", lambda c: all_results[c]["strategy1"].get("v1_best", {}).get("auc", np.nan)),
        ("Trajectory LogReg", lambda c: all_results[c]["strategy2"].get("auc", np.nan)),
        ("Calibrated (base+z)", lambda c: all_results[c]["strategy3"].get("auc", np.nan)),
        ("Z-score Only", lambda c: all_results[c]["strategy3"].get("auc_zscore_only", np.nan)),
        ("Q Difficulty Only", lambda c: all_results[c]["strategy4"].get("auc_q_only", np.nan)),
        ("Student x Question", lambda c: all_results[c]["strategy4"].get("auc_sq", np.nan)),
    ]

    for name, fn in strategies:
        best_auc = -1
        best_c = None
        for c in cutoffs:
            v = fn(c)
            if not np.isnan(v) and v > best_auc:
                best_auc = v
                best_c = c
        if best_c is not None:
            lines.append(f"| {name} | {best_auc:.3f} | W={best_c} | |")
        else:
            lines.append(f"| {name} | N/A | - | |")

    lines += [
        "",
        "---",
        "",
        "## Visualizations",
        "",
        "![Strategy Comparison](llm_signal_strategies.png)",
        "",
        "![LogReg Coefficients](llm_logreg_coefficients.png)",
    ]

    report = "\n".join(lines)
    report_path = os.path.join(output_dir, "LLM_SIGNAL_ANALYSIS.md")
    with open(report_path, "w") as f:
        f.write(report)
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    v4_df = load_v4_data()
    v1_df = load_v1_data()
    real_df, qid_to_week = load_real_data()

    # Determine cutoff weeks from the data
    all_weeks = sorted(set(qid_to_week.values()))
    # Use cutoffs that leave at least 1 week for train and test
    cutoff_weeks = all_weeks[1:-1]  # skip first and last
    if len(cutoff_weeks) > 8:
        # Sample evenly
        indices = np.linspace(0, len(cutoff_weeks) - 1, 8, dtype=int)
        cutoff_weeks = [cutoff_weeks[i] for i in indices]

    print(f"\nWeeks: {all_weeks}")
    print(f"Cutoff weeks to evaluate: {cutoff_weeks}")

    all_results = {}

    for cutoff in cutoff_weeks:
        print(f"\n{'=' * 60}")
        print(f"CUTOFF WEEK = {cutoff}")
        print("=" * 60)

        real_outcomes = get_real_outcomes(real_df, qid_to_week, cutoff)
        print(f"  Real test outcomes: {len(real_outcomes)} pairs, "
              f"pass rate: {real_outcomes['y_true'].mean():.3f}")

        print("\n  Strategy 1: Variant Comparison...")
        s1 = strategy_variant_comparison(v4_df, v1_df, real_outcomes, qid_to_week, cutoff)
        for name, metrics in s1.items():
            print(f"    {name}: AUC={metrics.get('auc', 'N/A'):.3f}, n={metrics.get('n', 0)}"
                  if not np.isnan(metrics.get('auc', np.nan))
                  else f"    {name}: N/A, n={metrics.get('n', 0)}")

        print("\n  Strategy 2: Trajectory LogReg...")
        s2 = strategy_trajectory_logreg(v4_df, real_df, qid_to_week, cutoff)
        print(f"    AUC={s2.get('auc', 'N/A')}, "
              f"train={s2.get('n_train', 0)}, test={s2.get('n_test', 0)}")

        print("\n  Strategy 3: Calibrated Relative...")
        s3 = strategy_calibrated_relative(v4_df, real_df, real_outcomes, qid_to_week, cutoff)
        print(f"    AUC={s3.get('auc', 'N/A')}, z-score AUC={s3.get('auc_zscore_only', 'N/A')}")

        print("\n  Strategy 4: Question Difficulty...")
        s4 = strategy_question_difficulty(v4_df, real_df, real_outcomes, qid_to_week, cutoff)
        print(f"    Q-only AUC={s4.get('auc_q_only', 'N/A')}, "
              f"SxQ AUC={s4.get('auc_sq', 'N/A')}, "
              f"Difficulty r={s4.get('difficulty_spearman', 'N/A')}")

        all_results[cutoff] = {
            "strategy1": s1,
            "strategy2": s2,
            "strategy3": s3,
            "strategy4": s4,
        }

    print(f"\n{'=' * 60}")
    print("GENERATING OUTPUTS")
    print("=" * 60)

    plot_path = plot_results(all_results, OUTPUT_DIR)
    print(f"Strategy comparison plot: {plot_path}")

    coef_path = plot_logreg_coefficients(all_results, OUTPUT_DIR)
    if coef_path:
        print(f"Coefficient plot: {coef_path}")

    report_path = generate_report(all_results, OUTPUT_DIR)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
