import os
from typing import Dict

import pandas as pd
from huggingface_hub import snapshot_download


# ─── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_URL = "https://huggingface.co/datasets/Kazchoko/codeinsights_llm_simulation/resolve/main/"
REPO_ID    = "stair-lab/code_insights_csv"
DATA_DIR   = "data"

# ─── Helpers ────────────────────────────────────────────────────────────────────
def is_perfect_score(score: float) -> bool:
    """Returns True if the integer part of `score` is made up entirely of '1's."""
    return all(ch == "1" for ch in str(int(score)))

def load_data() -> pd.DataFrame:
    path = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    main = pd.read_csv(os.path.join(path, "main_data.csv"))
    questions = pd.read_csv(os.path.join(path, "question_infos.csv"))
    return main, questions

def preprocess_submissions(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df
        .dropna()
        .query("pass != 0.0 and response_type == 'Submit'")
        .sort_values("timestamp")
    )
    df["is_perfect"] = df["pass"].apply(is_perfect_score)
    return df.drop_duplicates(
        subset=["student_id", "question_unittest_id"],
        keep="last"
    ).reset_index(drop=True)

def merge_with_questions(df: pd.DataFrame, q_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "question_id", "week", "topic", "question_name",
        "question_text", "question_template", "question_unittests"
    ]
    return (
        df
        .merge(q_df[cols], 
               left_on="question_unittest_id",
               right_on="question_id",
               how="inner")
        .dropna()
    )

def extract_mistake_fix(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only student–question groups with both a mistake and a perfect attempt,
       then pulls their first-occurence rows."""
    grouped = df.groupby(["student_id", "question_unittest_id"])
    mixed = grouped.filter(lambda g: g["is_perfect"].any() and (~g["is_perfect"]).any())
    mixed = mixed.sort_values(["student_id", "question_unittest_id", "timestamp"])
    
    def first_by(cond):
        return (
            mixed[cond]
            .groupby(["student_id", "question_unittest_id"], as_index=False)
            .first()
        )
    
    imperfect = first_by(~mixed["is_perfect"])
    perfect   = first_by( mixed["is_perfect"])
    return pd.concat([imperfect, perfect], ignore_index=True)

def build_response_alignment(mf: pd.DataFrame) -> pd.DataFrame:
    """For each student–question, record their first mistake & first correct responses."""
    def pair_responses(group):
        return pd.Series({
            "response_mistake": group.loc[~group["is_perfect"], "response"].iloc[0],
            "response_correct": group.loc[group["is_perfect"],  "response"].iloc[0],
        })
    return (
        mf
        .groupby(["student_id", "question_unittest_id"], as_index=False)
        .apply(pair_responses)
        .reset_index(drop=True)
    )

def save_scenarios(dfs: Dict[str, pd.DataFrame], base_url: str):
    out_dir = os.path.join(base_url, DATA_DIR)
    os.makedirs(out_dir, exist_ok=True)
    for name, df in dfs.items():
        path = os.path.join(out_dir, f"{name}_full_data.csv")
        df.to_csv(path, index=False)

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    submissions, questions = load_data()
    clean_subs = preprocess_submissions(submissions)
    merged    = merge_with_questions(clean_subs, questions)

    mf       = extract_mistake_fix(clean_subs)
    mf_merged = merge_with_questions(mf, questions)
    alignment = build_response_alignment(mf_merged)

    final = merged.merge(alignment, on=["student_id", "question_unittest_id"])

    scenarios = {
        "Scenario1": merged.groupby("question_unittest_id", group_keys=False).head(1),
        "Scenario2": merged,
        "Scenario3": final[final["response"] == final["response_mistake"]],
        "Scenario4": final[final["response"] == final["response_correct"]],
        "Questions": questions,
    }
    save_scenarios(scenarios, OUTPUT_URL)

if __name__ == "__main__":
    main()