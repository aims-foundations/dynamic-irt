import os
import json
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

# Try to import tueplots, but don't fail if not available
try:
    from tueplots import bundles
except ImportError:
    pass

def get_rating(store, key, initial_rating=0.0):
    return store.setdefault(key, initial_rating)

# Elo update with clipping
def rasch_update(theta, z, resp, K=0.4):
    """
    One-step update using a Rasch-style logistic for p_{ij}:
      p = 1 / (1 + exp[-(theta - z)])
    then delta = K*(resp - p), and clip back into [min_r, max_r].

    Args:
      theta   float: current ability (θ_i)
      z       float: current difficulty (z_j)
      resp    {0,1}: observed response R_{ij}
      K       float: learning rate
    Returns:
      (theta_new, z_new)
    """
    # Rasch-style probability
    p = 1.0 / (1.0 + np.exp(-(theta - z)))

    # compute update magnitude
    delta = K * (resp - p)

    # apply updates
    theta_new = theta + delta
    z_new = z - delta

    return theta_new, z_new

def load_and_prefix(path, model_name, param):
    """Load CSV and rename parameter column with model prefix."""
    try:
        df = pd.read_csv(path, usecols=["student_id", param])
        df = df.rename(columns={param: f"{param}_{model_name}"})
        return df
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None

def load_and_merge(prefix_map: dict, subfolder: str, id_col: str, metric: str, data_folder: str):
    """
    Load and merge ability/difficulty CSVs from multiple models.

    prefix_map: { df_prefix -> filename_prefix }
    subfolder: "ability" or "difficulty"
    id_col: "student_id" or "item_id"
    metric: "ability" or "difficulty"
    """
    dfs = []
    for prefix, fname in prefix_map.items():
        path = os.path.join(data_folder, subfolder, f"{fname}_{metric}.csv")
        df = load_and_prefix(path, prefix, metric)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # outer-merge all on id_col
    return reduce(lambda L, R: pd.merge(L, R, on=id_col, how="outer"), dfs)

def compute_corrs(merged_df: pd.DataFrame, real_col: str, metric: str):
    """
    Compute correlations between real (ground truth) and model columns.

    real_col: name of the "ground truth" column (e.g. "ability_student" or "difficulty_item")
    metric: "ability" or "difficulty"
    """
    if merged_df.empty or real_col not in merged_df.columns:
        return pd.DataFrame()

    cols = [c for c in merged_df.columns if c.startswith(f"{metric}_") and c != real_col]
    records = []

    for c in cols:
        sub = merged_df[[real_col, c]].dropna()
        if len(sub) < 3:
            continue

        r, _ = pearsonr(sub[real_col], sub[c])
        record = {
            "model": c.replace(f"{metric}_", ""),
            f"{metric}_pearson": r
        }

        if metric == "difficulty":
            s, _ = spearmanr(sub[real_col], sub[c])
            record[f"{metric}_spearman"] = s

        records.append(record)

    return pd.DataFrame(records)


# Model configurations
LIST_LLMS = [
    "gemma-3-27b-it",
    "llama-3.1-8b",
    "qwen2.5-14b",
    "gpt-4o",
    "claude-3-5",
    "gemini-2.5-pro",
    "mistral"
]

MODEL_MAP = {
    "claude-3-5": "claude-3-5",
    "mistral": "mistral",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gpt-4o": "gpt-4o",
    "llama-3.1-8b": "llama-3.1-8b",
    "gemma-3-27b-it": "gemma-3-27b-it",
    "student": "student",
    "item": "item"
}


def run_psychometric_analysis(data_folder: str, output_folder: str = None):
    """
    Run IRT-based psychometric analysis on LLM results.

    Args:
        data_folder: Path to folder containing scenario_results/
        output_folder: Path to save output files (defaults to data_folder)
    """
    if output_folder is None:
        output_folder = data_folder

    # Create output directories
    os.makedirs(os.path.join(output_folder, "ability"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "difficulty"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "correlations"), exist_ok=True)

    all_correlations = {}

    for llm in LIST_LLMS:
        print(f"Processing {llm}...")

        # Try to load scenario data
        dfs = []
        for scenario in [2, 3, 4]:
            scenario_path = os.path.join(
                data_folder,
                f"scenario_results/{llm}/{llm}_scenario{scenario}.csv"
            )
            if os.path.exists(scenario_path):
                try:
                    df = pd.read_csv(scenario_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"  Warning: Could not load {scenario_path}: {e}")

        if not dfs:
            print(f"  No data found for {llm}, skipping...")
            continue

        df = pd.concat(dfs, axis=0, ignore_index=True)

        # Create item_id if needed columns exist
        if "question_id" in df.columns and "test_case_id" in df.columns:
            df["item_id"] = df["question_id"].astype(str) + "_" + df["test_case_id"].astype(str)
        elif "question_id" in df.columns:
            df["item_id"] = df["question_id"].astype(str)
        else:
            print(f"  Warning: Missing question_id column for {llm}")
            continue

        # Check for correctness column
        correctness_col = None
        for col in ["LLM_correctness", "correctness", "correct", "pass"]:
            if col in df.columns:
                correctness_col = col
                break

        if correctness_col is None:
            print(f"  Warning: No correctness column found for {llm}")
            continue

        # Elo Rating Parameters
        K = 0.4
        initial_rating = 0.0
        student_ratings = {}
        item_ratings = {}

        for _, row in df.iterrows():
            sid = row.get("student_id", "unknown")
            iid = row["item_id"]
            resp = row[correctness_col]

            # Convert to 0/1 if needed
            if isinstance(resp, str):
                resp = 1 if resp.lower() in ["true", "1", "pass", "correct"] else 0
            else:
                resp = int(resp) if pd.notna(resp) else 0

            R_s = get_rating(student_ratings, sid, initial_rating)
            R_i = get_rating(item_ratings, iid, initial_rating)

            R_s_new, R_i_new = rasch_update(R_s, R_i, resp, K=K)
            student_ratings[sid] = R_s_new
            item_ratings[iid] = R_i_new

        # Create DataFrames
        students_df = (
            pd.DataFrame.from_dict(student_ratings, orient="index", columns=["ability"])
            .reset_index().rename(columns={"index": "student_id"})
        )
        items_df = (
            pd.DataFrame.from_dict(item_ratings, orient="index", columns=["difficulty"])
            .reset_index().rename(columns={"index": "item_id"})
        )

        # Save ability & difficulty data
        students_df.to_csv(
            os.path.join(output_folder, f"ability/{llm}_student_ability.csv"),
            index=False
        )
        items_df.to_csv(
            os.path.join(output_folder, f"difficulty/{llm}_difficulty.csv"),
            index=False
        )

        print(f"  Saved {len(students_df)} student abilities, {len(items_df)} item difficulties")

    # Compute correlations if we have real student data
    print("\nComputing correlations...")

    # Load and merge ability data
    merged_ability = load_and_merge(MODEL_MAP, "ability", "student_id", "ability", output_folder)
    if not merged_ability.empty and "ability_student" in merged_ability.columns:
        ability_corr_df = compute_corrs(merged_ability, "ability_student", "ability")
        all_correlations["ability"] = ability_corr_df.to_dict(orient="records")

    # Load and merge difficulty data
    merged_difficulty = load_and_merge(MODEL_MAP, "difficulty", "item_id", "difficulty", output_folder)
    if not merged_difficulty.empty and "difficulty_item" in merged_difficulty.columns:
        difficulty_corr_df = compute_corrs(merged_difficulty, "difficulty_item", "difficulty")
        all_correlations["difficulty"] = difficulty_corr_df.to_dict(orient="records")

    # Save correlations
    cor_out_path = os.path.join(output_folder, "correlations", "all_correlations.json")
    with open(cor_out_path, "w") as f:
        json.dump(all_correlations, f, indent=2, ensure_ascii=False)

    print(f"Saved correlations to {cor_out_path}")

    return all_correlations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run psychometric analysis on LLM results")
    parser.add_argument("--data", type=str, default="./",
                        help="Data folder containing scenario_results/")
    parser.add_argument("--output", type=str, default=None,
                        help="Output folder (defaults to data folder)")

    args = parser.parse_args()

    run_psychometric_analysis(args.data, args.output)
