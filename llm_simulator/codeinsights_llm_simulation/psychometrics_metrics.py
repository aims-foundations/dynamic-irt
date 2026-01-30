import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from tueplots import bundles

def get_rating(store, key):
    return store.setdefault(key, initial_rating)

# 3) Elo update with clipping
def rasch_update(theta, z, resp, K=0.4):
    """
    One‐step update using a Rasch‐style logistic for p_{ij}:
      p = 1 / (1 + exp[-(theta - z)])
    then delta = K*(resp - p), and clip back into [min_r, max_r].
    
    Args:
      theta   float: current ability (θ_i)
      z       float: current difficulty (z_j)
      resp    {0,1}: observed response R_{ij}
      K       float: learning rate
      min_r   float: lower bound for both θ_i and z_j
      max_r   float: upper bound for both θ_i and z_j
    Returns:
      (theta_new, z_new)
    """
    # 1) Rasch‐style probability
    p = 1.0 / (1.0 + np.exp(-(theta - z)))
    
    # 2) compute update magnitude
    delta = K * (resp - p)
    
    # 3) apply updates
    theta_new = theta + delta
    z_new     = z     - delta
    
    return theta_new, z_new

def load_and_prefix(path, model_name, param):
    # only read the columns we need
    df = pd.read_csv(path, usecols=["student_id", param])
    # rename ability → ability_<model_name>
    df = df.rename(columns={param: f"{param}_{model_name}"})
    return df

def load_and_merge(prefix_map: dict, subfolder: str, id_col: str, metric: str):
    """
    prefix_map: { df_prefix -> filename_prefix }  
    subfolder: "ability" or "difficulty"  
    id_col:   "student_id" or "item_id"  
    metric:   "ability" or "difficulty"  
    """
    dfs = []
    for prefix, fname in prefix_map.items():
        path = os.path.join(DATA_FOLDER,
                            subfolder,
                            f"{fname}_{metric}.csv")
        dfs.append(load_and_prefix(path, prefix, metric))
    # outer-merge all on id_col
    return reduce(lambda L, R: pd.merge(L, R, on=id_col, how="outer"), dfs)

def compute_corrs(merged_df: pd.DataFrame, real_col: str, metric: str):
    """
    real_col: name of the “ground truth” column (e.g. "ability_student" or "difficulty_item")
    metric:   same as above
    """
    cols = [c for c in merged_df if c.startswith(f"{metric}_") and c != real_col]
    records = []
    for c in cols:
        sub = merged_df[[real_col, c]].dropna()
        r, _ = pearsonr(sub[real_col], sub[c])
        # for difficulty you might also want spearmanr:
        record = {
            "model":          c.replace(f"{metric}_", ""),
            f"{metric}_pearson": r
        }
        if metric == "difficulty":
            s, _ = spearmanr(sub[real_col], sub[c])
            record[f"{metric}_spearman"] = s
        records.append(record)
    return pd.DataFrame(records)

LIST_LLMS = [
    "gemma-3-27b-it",
    "llama-8b",
    "qwen",
    "gpt-4o",
    "claude-3-5",
    "gemini-2.5-pro",
    "mistral"
]

DATA_FOLDER = "https://huggingface.co/datasets/Kazchoko/codeinsights_llm_simulation/resolve/main/"

model_map = {
    "claude3-5":   "claude-3-5",
    "mistral":     "mistral",
    "gemini":      "gemini-2.5-pro",
    "gpt4o":       "gpt-4o",
    "llama8b":     "llama-8b",
    "gemma2.5":    "gemma-3-27b-it",
    "student":     "student",
    "item":        "item"
}

cor_out_path = os.path.join(DATA_FOLDER, "correlations", "all_correlations.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

if __name__ == "__main__":
    for llm in LIST_LLMS:
        df1 = pd.read_csv(
            os.path.join(
                    DATA_FOLDER,
                    f"scenario_results/{llm}/{llm}_scenario2.csv",
                )
        )
            ('scenario_results/{llm}/{llm}_scenario2.csv')
        df2 = pd.read_csv(
            os.path.join(
                    DATA_FOLDER,
                    f"scenario_results/{llm}/{llm}_scenario3.csv",
                )
        )
        df3 = pd.read_csv(
            os.path.join(
                    DATA_FOLDER,
                    f"scenario_results/{llm}/{llm}_scenario4.csv",
                )
        )
        df = pd.concat(
            [df1, df2, df3],
            axis=0,
            ignore_index
        df["item_id"] = (df["question_id"].astype(str) + "_" + df["test_case_id"].astype(str))

        # Elo Rating Parameters
        K = 0.4
        initial_rating = 0.0
        student_ratings = {}
        item_ratings    = {}

        for _, row in df.iterrows():
            sid, iid, resp = (
                row["student_id"],
                row["item_id"],
                row["LLM_correctness"],  # assumed 0 or 1
            )
            R_s = get_rating(student_ratings, sid)
            R_i = get_rating(item_ratings,    iid)

            R_s_new, R_i_new = rasch_update(
                R_s, R_i, resp,
                K=K
            )
            student_ratings[sid] = R_s_new
            item_ratings[iid]    = R_i_new

        students_df = (
            pd.DataFrame.from_dict(student_ratings, orient="index", columns=["ability"])
              .reset_index().rename(columns={"index": "student_id"})
        )
        items_df = (
            pd.DataFrame.from_dict(item_ratings, orient="index", columns=["difficulty"])
              .reset_index().rename(columns={"index": "item_id"})
        )
        #Write ability & difficulty data
        students_df.to_csv(
                os.path.join(
                    DATA_FOLDER,
                    f"ability/{llm}_student_ability.csv"
                )
            )
        items_df.to_csv(
                os.path.join(
                    DATA_FOLDER,
                    f"difficulty/{llm}_difficulty.csv"
                )
            )
    #Load all question
    question_data = pd.read_csv(
        os.path.join(_FOLDER, "data/codeinsights_question.csv")
    )
    #Ability Correlation
    merged_ability    = load_and_merge(model_map, "ability", "student_id", "ability")
    ability_corr_df   = compute_corrs(merged_ability, "ability_student", "ability")
    #Difficulty Correlation
    merged_difficulty = load_and_merge(model_map, "difficulty", "item_id", "difficulty")
    difficulty_corr_df = compute_corrs(merged_difficulty, "difficulty_item", "difficulty")
    with open(out_path, "w") as f:
    json.dump({
        "ability":    ability_corr_df.to_dict(orient="records"),
        "difficulty": difficulty_corr_df.to_dict(orient="records")
    }, f, indent=2, ensure_ascii=False)



