"""
Descriptive statistics and visualizations for Edmentum and CodeInsights datasets.

This script generates:
- Student attempt counts
- Score distributions
- Item difficulty distributions
"""

import random
import pandas as pd
import numpy as np

from dynamic_irt.elo import utils


# Constants
SPLIT_NUM = 5


if __name__ == "__main__":
    # Edmentum Data

    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    questions = utils.load_csv_from_gdrive_by_id(
        "19FlQXMJtp9uiqs6u90ajmV9J0JiWy4iq"
    )
    data = utils.load_csv_from_gdrive_by_id(
        "1M6tur2SD-yJi2q9FS9DwW7A8a0yioeAT"
    )
    data["day"] = data["time_since_last_attempt"] // 86400
    # Attempts per Item & Average Response
    attempts = (
        data.groupby(["StudentID_SF", "ItemID_SF"])
        .size()
        .reset_index(name="attempt_count")
    )
    avg_attempts = (
        attempts.groupby("StudentID_SF")["attempt_count"]
        .mean()
        .reset_index(name="avg_attempts_per_item")
    )
    overall_avg_attempts = avg_attempts["avg_attempts_per_item"].mean()
    print("Overall average attempts per item:", overall_avg_attempts)
    utils.plot_distribution(
        avg_attempts,
        "avg_attempts_per_item",
        "Edmentum Average Item Attempt per Student",
    )
    avg_item_score = (
        data.groupby(["StudentID_SF", "ItemID_SF"])["ItemScore"]
        .mean()
        .reset_index(name="avg_item_score")
    )
    utils.plot_distribution(
        avg_item_score,
        "avg_item_score",
        "Edmentum Average Response per Item per Student",
    )

    # CodeInsights
    code_clean_data = utils.load_csv_from_gdrive_by_id(
        "1-SfplBQW4Pi0-t2BvjQpqyIlwLjSjZgc"
    )
    code_clean_data["attempt_count"] = code_clean_data.groupby(
        ["StudentID_SF", "ItemID_SF"]
    )["T"].transform("count")
    item_dataset = utils.load_csv_from_gdrive_by_id(
        "1i_9rsIc0TM6_hib7PkBX58oQ2AxvdT1C"
    )
    # Filter item_dataset to get valid ItemID_SF values (student_count >= 10)
    valid_items = item_dataset[item_dataset["student_count"] >= 10]["ItemID_SF"]
    filtered_code_clean_data = code_clean_data[
        code_clean_data["ItemID_SF"].isin(valid_items)
    ]
    filtered_code_clean_data
    print(f"Number of data: {len(code_clean_data)}")
    print(f"Number of Students: {len(code_clean_data['StudentID_SF'].unique())}")
    print(f"Number of items: {len(code_clean_data['ItemID_SF'].unique())}")
    # Item-level
    # Item Dataframe to Judge Difficulty from Dataset
    last_attempts = code_clean_data.sort_values("T").drop_duplicates(
        subset=["StudentID_SF", "ItemID_SF"], keep="last"
    )
    last_attempts["student_count"] = last_attempts.groupby(["ItemID_SF"])[
        "T"
    ].transform("count")
    item_average_score = pd.DataFrame(
        last_attempts.groupby("ItemID_SF")["ItemScore"].mean()
    )
    item_average_attempt = pd.DataFrame(
        last_attempts.groupby("ItemID_SF")["attempt_count"].mean()
    )
    item_average_student = pd.DataFrame(
        last_attempts.groupby("ItemID_SF")["student_count"].mean()
    )
    item_data = item_average_score.merge(item_average_attempt, on="ItemID_SF")
    item_data = item_data.merge(item_average_student, on="ItemID_SF")
    item_data = item_data.sort_values(
        ["ItemScore", "attempt_count", "student_count"], ascending=False
    )
    utils.plot_distribution(
        item_data, "ItemScore", "CodeInsights Average Response per Item"
    )

    # Average Attempts Stats with Code Clean Data
    attempts = (
        code_clean_data.groupby(["StudentID_SF", "ItemID_SF"])
        .size()
        .reset_index(name="attempt_count")
    )
    avg_attempts = (
        attempts.groupby("StudentID_SF")["attempt_count"]
        .mean()
        .reset_index(name="avg_attempts_per_item")
    )
    overall_avg_attempts = avg_attempts["avg_attempts_per_item"].mean()
    print("Overall average attempts per item:", overall_avg_attempts)
    # Plot
    utils.plot_distribution(
        avg_attempts,
        "avg_attempts_per_item",
        "CodeInsights Average Item Attempt per Student",
    )
    avg_item_score = (
        code_clean_data.groupby(["StudentID_SF", "ItemID_SF"])["ItemScore"]
        .mean()
        .reset_index(name="avg_item_score")
    )
    utils.plot_distribution(
        avg_item_score,
        "avg_item_score",
        "CodeInsights Average Response per Item per Student",
    )
