"""
Data preprocessing for CodeInsights dataset.

This script handles:
- Loading data from HuggingFace
- Cleaning and transforming student response data
- Creating temporally-ordered student response sequences

Note: This script has already been executed to generate the required datasets.
Running it again is not recommended unless regenerating data is needed.
"""

import os
import re
import pandas as pd
import numpy as np
from huggingface_hub import login, snapshot_download


def preprocess_codeinsights_data():
    """
    Preprocess CodeInsights data from HuggingFace.

    Returns:
        pd.DataFrame: Cleaned and processed data
    """
    # Login using environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not set. Using cached credentials if available.")

    path = snapshot_download(
        repo_id="stair-lab/code_insights_csv", repo_type="dataset"
    )
    code_insights = pd.read_csv(f"{path}/main_data.csv")
    questions = pd.read_csv(f"{path}/question_infos.csv")

    # Make Question Dataframe with Question Text and Student Solution
    def only_ones(value):
        """Check if the value consists only of '1's."""
        if pd.isna(value):
            return False
        s = str(value).strip()
        if s == "":
            return False
        try:
            s = str(int(float(s)))
        except ValueError:
            return False
        return bool(re.fullmatch(r"1+", s))

    # Create a boolean mask by applying the function to the "pass" column
    mask = code_insights["pass"].apply(only_ones)
    filtered_df = code_insights[mask]

    filtered_df = filtered_df.rename(
        columns={"question_unittest_id": "ItemID_SF"}
    )
    new_df = filtered_df.groupby("ItemID_SF", as_index=False).last()
    final_question = questions.merge(
        new_df[["ItemID_SF", "response"]],
        left_on="question_id", right_on="ItemID_SF"
    )

    # Clean Data
    def remove_decimal_if_whole(val):
        """Remove decimal point from whole numbers."""
        try:
            val_str = str(val)
            if "." in val_str:
                num = float(val_str)
                if num.is_integer():
                    return str(int(num))
                return val_str
            else:
                return val_str
        except ValueError:
            return str(val)

    filtered_code = code_insights[
        (code_insights["response_type"] == "Submit")
        | (code_insights["response_type"] == "Prechecked")
    ]
    filtered_code["pass"] = filtered_code["pass"].apply(remove_decimal_if_whole)
    filtered_code["pass"] = filtered_code["pass"].replace("nan", np.nan)
    filtered_code = filtered_code.dropna(subset=["pass"])
    filtered_code["timestamp"] = pd.to_datetime(
        filtered_code["timestamp"], format="%d/%m/%y, %H:%M:%S"
    )
    filtered_code["T"] = filtered_code.groupby("student_id")["timestamp"].transform(
        lambda x: (x - x.min()).dt.total_seconds()
    )
    # Make sure the DataFrame has a unique index and create an attempt_id.
    filtered_code = filtered_code.reset_index(drop=True)
    filtered_code["attempt_id"] = (
        filtered_code["student_id"].astype(str)
        + "_"
        + filtered_code["question_unittest_id"].astype(str)
        + "_"
        + filtered_code["T"].astype(str)
    )

    # Split the "pass" string into a list of characters.
    filtered_code["response_list"] = filtered_code["pass"].apply(list)

    # Explode the list so that each character becomes a separate row.
    code_exploded = filtered_code.explode("response_list")

    # Group by student_id, question_unittest_id,
    # and the unique attempt_id so numbering resets for each attempt.
    code_exploded["item_index"] = (
        code_exploded.groupby(
            ["student_id", "question_unittest_id", "attempt_id"]
        ).cumcount()
        + 1
    )

    # Form "item_id" by concatenating "question_unittest_id" with index.
    code_exploded["item_id"] = (
        code_exploded["question_unittest_id"].astype(str)
        + "_"
        + code_exploded["item_index"].astype(str)
    )

    # Rename the exploded column to "response"
    code_exploded = code_exploded.rename(
        columns={"response_list": "item_response"}
    )
    code_exploded = code_exploded.sort_values(["student_id", "T"])
    code_exploded["time_since_last_attempt"] = (
        code_exploded.groupby("student_id")["T"].diff().fillna(0)
    )
    columns = ["student_id", "item_id", "T",
               "item_response", "time_since_last_attempt"]
    code_clean_data = code_exploded[columns]
    code_clean_data = code_clean_data.rename(
        columns={
            "student_id": "StudentID_SF",
            "item_id": "ItemID_SF",
            "item_response": "ItemScore",
        }
    )
    code_clean_data["ItemScore"] = code_clean_data["ItemScore"].astype("Int64")

    return code_clean_data


if __name__ == "__main__":
    data = preprocess_codeinsights_data()
    print(f"Processed {len(data)} records")
    print(data.head())
