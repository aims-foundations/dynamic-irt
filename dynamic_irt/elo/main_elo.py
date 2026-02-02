"""
Main Elo-based experiments for Edmentum and CodeInsights datasets.

This script runs various Elo rating model experiments:
- Dynamic updates with forgetting mechanisms
- Student ability tracking based on performance
- Difficulty estimation for coding problems
- Performance evaluation using AUC metrics with plots
"""

import random
import pandas as pd
import numpy as np

from dynamic_irt.elo import utils


if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 123
    random.seed(seed)
    np.random.seed(seed)

    # Load and prepare Edmentum data
    ed_data = utils.load_csv_from_gdrive_by_id(
        "1M6tur2SD-yJi2q9FS9DwW7A8a0yioeAT"
    )
    ed_data["day"] = ed_data["time_since_last_attempt"] // 86400
    ed_cols1 = [
        "StudentID_SF",
        "ItemID_SF",
        "day",
        "Base_Theta",
        "RaschLogit",
        "ItemScore",
        "time_since_last_attempt",
        "T",
    ]
    ed_cols2 = [
        "StudentID_SF",
        "ItemID_SF",
        "day",
        "ItemScore",
        "time_since_last_attempt",
        "T",
    ]

    # Common parameters
    mask_fraction = 0.2
    K = 0.4
    ed_difficulty_range = (-6.4, 3.9)

    # Experiment 1: Dynamic Update Considering Forgetting
    ed_data["Base_Theta"] = -2.5176
    ed_fit_data = (
        ed_data[ed_cols1].sort_values(["StudentID_SF", "T"]).reset_index(drop=True)
    )
    ed_fit_data = utils.mask_responses(ed_fit_data, mask_fraction=mask_fraction)
    ed_fit_data["ThetaUpdated"] = np.nan
    ed_fit_data = utils.run_update(
        ed_fit_data, utils.update_with_time, K=K, update_difficulty=False
    )
    utils.evaluate_edmentum(
        ed_data, ed_fit_data, ed_cols1, "RaschLogit", ed_difficulty_range
    )

    # Experiment 2: Dynamic Update Considering Previous Attempt
    ed_data["Base_Theta"] = -2.5176
    ed_attempt_data = (
        ed_data[ed_cols1].sort_values(["StudentID_SF", "T"]).reset_index(drop=True)
    )
    ed_attempt_data = utils.mask_responses(
        ed_attempt_data, mask_fraction=mask_fraction
    )
    ed_attempt_data["ThetaUpdated"] = np.nan
    ed_attempt_data = utils.run_update_with_new_attempt(
        ed_attempt_data,
        "ItemID_SF",
        K=K,
        update_difficulty=False,
    )
    utils.evaluate_edmentum(
        ed_data, ed_attempt_data, ed_cols1, "RaschLogit", ed_difficulty_range
    )

    # Experiment 3: Dynamic Update (Theta and Difficulty) Considering Forgetting
    ed_both_update_forget = (
        ed_data[ed_cols2].sort_values(["StudentID_SF", "T"]).reset_index(drop=True)
    )
    ed_both_update_forget["Base_Theta"] = 0
    ed_both_update_forget["Base_Difficulty"] = 0
    ed_both_update_forget = utils.mask_responses(
        ed_both_update_forget, mask_fraction=mask_fraction
    )
    ed_both_update_forget["ThetaUpdated"] = np.nan
    ed_both_update_forget["DifficultyUpdated"] = np.nan
    ed_both_update_forget = utils.run_update(
        ed_both_update_forget,
        utils.update_with_time,
        K=K,
        update_difficulty=True,
    )
    utils.evaluate_edmentum(
        ed_data, ed_both_update_forget, ed_cols2, "DifficultyUpdated", ed_difficulty_range
    )

    # Experiment 4: Dynamic Update (Theta and Difficulty)
    ed_both_update_base = (
        ed_data[ed_cols2].sort_values(["StudentID_SF", "T"]).reset_index(drop=True)
    )
    ed_both_update_base["Base_Theta"] = 0
    ed_both_update_base["Base_Difficulty"] = 0
    ed_both_update_base = utils.mask_responses(
        ed_both_update_base, mask_fraction=mask_fraction
    )
    ed_both_update_base["ThetaUpdated"] = np.nan
    ed_both_update_base["DifficultyUpdated"] = np.nan
    ed_both_update_base = utils.run_update(
        ed_both_update_base, utils.basic_update, K=K, update_difficulty=True
    )
    utils.evaluate_edmentum(
        ed_data, ed_both_update_base, ed_cols2, "DifficultyUpdated", ed_difficulty_range
    )

    # Load and prepare CodeInsights data
    ci_data = utils.load_csv_from_gdrive_by_id("1gJ3p5t3LYPEUt3IYc0YYeWxwqTT6YZ_BT")
    ci_data["day"] = ci_data["time_since_last_attempt"] // 86400
    ci_cols = [
        "StudentID_SF",
        "ItemID_SF",
        "day",
        "ItemScore",
        "time_since_last_attempt",
        "T",
    ]

    # Experiment 1: Dynamic Update (Theta & Difficulty) with basic_update
    ci_fit_data = ci_data.copy()
    ci_fit_data["Base_Theta"] = 0
    ci_fit_data["Base_Difficulty"] = 0
    ci_fit_data = ci_fit_data.sort_values(["StudentID_SF", "T"]).reset_index(drop=True)
    ci_fit_data = utils.mask_responses(ci_fit_data, mask_fraction=mask_fraction)
    ci_fit_data["ThetaUpdated"] = np.nan
    ci_fit_data["DifficultyUpdated"] = np.nan
    ci_fit_data = utils.run_update(
        ci_fit_data, utils.basic_update, K=K, update_difficulty=True
    )
    ci_av_diff = ci_fit_data.groupby("ItemID_SF").apply(
        utils.compute_difficulty, split_num=5
    )
    ci_fit_data["AverageDifficulty"] = ci_fit_data["ItemID_SF"].map(ci_av_diff)
    utils.evaluate_codeinsights(
        ci_data, ci_fit_data, ci_cols, "DifficultyUpdated", (-3.6, 3.6)
    )

    # Experiment 2: Dynamic Update (Theta & Difficulty) Considering Forgetting with update_with_time
    ci_code_forgetting = ci_data[ci_cols].copy()
    ci_code_forgetting["Base_Theta"] = 0
    ci_code_forgetting["Base_Difficulty"] = 0
    ci_code_forgetting = ci_code_forgetting.sort_values(["StudentID_SF", "T"]).reset_index(
        drop=True
    )
    ci_code_forgetting = utils.mask_responses(
        ci_code_forgetting, mask_fraction=mask_fraction
    )
    ci_code_forgetting["ThetaUpdated"] = np.nan
    ci_code_forgetting["DifficultyUpdated"] = np.nan
    ci_code_forgetting = utils.run_update(
        ci_code_forgetting, utils.update_with_time, K=K, update_difficulty=True
    )
    ci_av_diff = ci_code_forgetting.groupby("ItemID_SF").apply(
        utils.compute_difficulty, split_num=5
    )
    ci_code_forgetting["AverageDifficulty"] = ci_code_forgetting["ItemID_SF"].map(
        ci_av_diff
    )
    utils.evaluate_codeinsights(
        ci_data, ci_code_forgetting, ci_cols, "DifficultyUpdated", (-3.8, 3.5)
    )

    # Experiment 3: Dynamic Update (Theta & Difficulty) with New Item Attempts
    ci_attempt_data = ci_data[ci_cols].copy()
    ci_attempt_data["Base_Theta"] = 0
    ci_attempt_data["Base_Difficulty"] = 0
    ci_attempt_data = ci_attempt_data.sort_values(["StudentID_SF", "T"]).reset_index(
        drop=True
    )
    ci_attempt_data = utils.mask_responses(
        ci_attempt_data, mask_fraction=mask_fraction
    )
    ci_attempt_data["ThetaUpdated"] = np.nan
    ci_attempt_data["DifficultyUpdated"] = np.nan
    ci_attempt_data["ItemID"] = ci_attempt_data["ItemID_SF"].str.split("_").str[0]
    ci_attempt_data = utils.run_update_with_new_attempt(
        ci_attempt_data, "ItemID", K=K, update_difficulty=True
    )
    ci_av_diff = ci_attempt_data.groupby("ItemID_SF").apply(
        utils.compute_difficulty, split_num=5
    )
    ci_attempt_data["AverageDifficulty"] = ci_attempt_data["ItemID_SF"].map(ci_av_diff)
    utils.evaluate_codeinsights(
        ci_data, ci_attempt_data, ci_cols, "DifficultyUpdated", (-12.3, 12.9)
    )
