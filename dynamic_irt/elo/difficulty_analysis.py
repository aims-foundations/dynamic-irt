"""
Difficulty analysis for Edmentum and CodeInsights datasets.

This script analyzes relationships between:
- Estimated difficulty and problem features
- Test-level difficulty distributions
- Correlations between difficulty, code length, and required steps
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from dynamic_irt.elo import utils


if __name__ == "__main__":
    # Constants
    SPLIT_NUM = 5

    # =============== Edmentum Rasch Logit and Estimated Difficulty Correlation ===============

    # Load required datasets
    ed_data = utils.load_csv_from_gdrive_by_id(
        "1yLQedZJioQdOjbNf15l9G4fkvZVF7U6P"
    )
    ed_both_update_forget = utils.load_csv_from_gdrive_by_id(
        "1diAFZsdY9IpA4I-E2L0vtp_4kNfEXsp9"
    )

    # Compute average difficulty and map it back
    # Fixed: Include include_groups=False to address the deprecation warning
    ed_av_difficulty_map = ed_both_update_forget.groupby("ItemID_SF").apply(
        utils.compute_difficulty, split_num=SPLIT_NUM, include_groups=False
    )
    ed_both_update_forget["AverageDifficulty"] = ed_both_update_forget["ItemID_SF"].map(
        ed_av_difficulty_map
    )

    # Get last entries for each student
    ed_last_entries_both = ed_both_update_forget.loc[
        ed_both_update_forget.groupby("StudentID_SF")["T"].idxmax()
    ]
    ed_last_entries = ed_data.loc[ed_data.groupby("StudentID_SF")["T"].idxmax()]

    # Create a filtered dataset with only the relevant columns
    ed_filtered = pd.DataFrame(
        {"x": ed_last_entries["RaschLogit"], "y": ed_last_entries_both["AverageDifficulty"]}
    ).dropna()

    # Calculate and print correlation
    corr_sp, p_value = spearmanr(ed_filtered["x"], ed_filtered["y"])
    print(f"Edmentum Spearman correlation: {corr_sp:.4f} (p-value: {p_value:.4f})")

    # =============== CodeInsights Difficulty Analysis ===============

    # Load datasets
    item_data = utils.load_csv_from_gdrive_by_id(
        "1i_9rsIc0TM6_hib7PkBX58oQ2AxvdT1C"
    )
    ci_code_forgetting = utils.load_csv_from_gdrive_by_id(
        "1_zXUjoBchwuFfJNLKkq92RQv53Dz3Pzg"
    )

    # First, check if 'AverageDifficulty' column exists
    ci_av_diff = ci_code_forgetting.groupby("ItemID_SF").apply(
        utils.compute_difficulty, split_num=SPLIT_NUM, include_groups=False
    )
    ci_code_forgetting["AverageDifficulty"] = ci_code_forgetting["ItemID_SF"].map(
        ci_av_diff
    )

    # Process test IDs and calculate averages
    ci_code_forgetting["TestID"] = (
        ci_code_forgetting["ItemID_SF"].str.split("_").str[0].astype(int)
    )
    ci_code_forgetting["Test_Average_Difficulty"] = ci_code_forgetting.groupby("TestID")[
        "AverageDifficulty"
    ].transform("mean")

    # Create a summary dataframe with test difficulties
    test_summary = pd.DataFrame(
        {
            "AverageDifficulty": ci_code_forgetting.groupby("ItemID_SF")[
                "AverageDifficulty"
            ].mean(),
            "TestID": ci_code_forgetting.groupby("ItemID_SF")["TestID"].mean(),
            "Test_Average_Difficulty": ci_code_forgetting.groupby("ItemID_SF")[
                "Test_Average_Difficulty"
            ].mean(),
        }
    )

    # Sort tests by difficulty for visualization
    order = test_summary.groupby("TestID")["Test_Average_Difficulty"].first().sort_values()
    mapping = {test_id: i for i, test_id in enumerate(order.index)}
    test_summary["TestID_order"] = test_summary["TestID"].map(mapping)

    # Plot test difficulties
    plt.figure(figsize=(10, 6))
    plt.scatter(test_summary["TestID_order"], test_summary["AverageDifficulty"])
    plt.xticks(ticks=list(mapping.values()), labels=list(mapping.keys()))
    plt.xlabel("Question (Ordered by Question Difficulty)")
    plt.ylabel("Test Difficulty")
    plt.title("Test Difficulty Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # =============== Item-Level Analysis ===============

    # Create item-level dataset
    estimated_difficulty = pd.DataFrame(
        ci_code_forgetting.groupby("ItemID_SF")["AverageDifficulty"].mean()
    )
    difficulty_full_dataframe = estimated_difficulty.merge(
        item_data, left_index=True, right_on="ItemID_SF"
    )

    # Calculate correlation between unittest difficulty and averaged item difficulty
    corr_unittest, p_unittest = spearmanr(
        difficulty_full_dataframe["AverageDifficulty"],
        difficulty_full_dataframe["ItemScore"],
    )
    print(
        f"Unittest difficulty correlation: {corr_unittest:.4f} (p-value: {p_unittest:.4f})"
    )

    # =============== Correlation with Steps and Lines ===============

    # Load question data
    question_df = utils.load_csv_from_gdrive_by_id(
        "1_GMlE3yyz1gq2OM5IpdoVyrQDAVyPx8G"
    )

    # Get test difficulties
    test_difficulties = ci_code_forgetting[
        ["TestID", "Test_Average_Difficulty"]
    ].drop_duplicates()

    # Merge data for correlation analysis
    test_items = question_df.merge(
        test_difficulties, left_on="question_id", right_on="TestID"
    ).drop_duplicates()

    # Calculate correlation between lines and difficulty
    corr_lines, p_lines = spearmanr(
        test_items["lines"], test_items["Test_Average_Difficulty"]
    )
    print(f"Lines-difficulty correlation: {corr_lines:.4f} (p-value: {p_lines:.4f})")
    corr_lines, p_lines = spearmanr(
        test_items["required_steps"], test_items["Test_Average_Difficulty"]
    )
    print(f"Steps-difficulty correlation: {corr_lines:.4f} (p-value: {p_lines:.4f})")
