"""
Elo-based IRT utility functions for learning dynamics analysis.

This module provides Elo rating system implementations for tracking
student ability and problem difficulty over time, with support for:
- Time-based forgetting
- Multiple attempt tracking
- Edmentum and CodeInsights datasets
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import gdown


def load_csv_from_gdrive_by_id(file_id):
    """
    Load a CSV file from Google Drive using its file ID.

    Parameters:
        file_id (str): The Google Drive file ID

    Returns:
        pandas.DataFrame: DataFrame containing the CSV data
    """
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        output = 'temp_csv_file.csv'
        gdown.download(url, output, quiet=False)
        df = pd.read_csv(output)
        return df
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def mask_responses(df, mask_fraction=0.2):
    """
    Randomly mask a fraction of the responses in the DataFrame to separate train and test set.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        mask_fraction (float): Fraction of responses to mask (default is 0.2).

    Returns:
        pd.DataFrame: DataFrame with masked responses.
    """
    mask = np.random.rand(len(df)) < mask_fraction
    df.loc[mask, "ItemScore"] = np.nan
    return df


def basic_update(th, b, day, resp, K=0.4):
    """
    Update the ability and difficulty parameters using the standard Elo rating system.

    Args:
        th (float): Ability parameter.
        b (float): Difficulty parameter.
        day (int): Placeholder for consistency in update function formats for run_update
        resp (int): Response (1 for correct, 0 for incorrect).
        K (float): Learning rate (default is 0.4).

    Returns:
        tuple: Updated ability and difficulty parameters.
    """
    p = 1 / (1 + np.exp(-(th - b)))
    return th + K * (resp - p), b - K * (resp - p)


def update_with_time(th, b, day, resp, K=0.4):
    """
    Update the ability and difficulty parameters considering time intervals in Elo.

    Args:
        th (float): Ability parameter.
        b (float): Difficulty parameter.
        day (int): Time interval (in day) since last attempt.
        resp (int): Response (1 for correct, 0 for incorrect).
        K (float): Learning rate (default is 0.4).

    Returns:
        tuple: Updated ability and difficulty parameters.
    """
    p = 1 / (1 + np.exp(-(th - b)))
    # More time interval since last attempt makes the response correctness unreliable
    prob_coefficient = 1 / (day + 1)
    return th + K * (
        resp - (prob_coefficient * p + (1 - prob_coefficient) * 0.5)
    ), b - K * (resp - p)


def run_update(data, update_func, K=0.4, update_difficulty=False):
    """
    Enact the updates and add updated theta and difficulty to the dataframe.
    (for standard elo and time-interval model)

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        update_func (function): Function to update the parameters.
        K (float): Learning rate (default is 0.4).
        update_difficulty (bool): Whether to update difficulty (default is False).

    Returns:
        pd.DataFrame: DataFrame with updated theta and difficulty.
    """
    last_student, last_theta, last_difficulty = None, None, None
    theta_updated, difficulty_updated = [], []
    for idx, row in data.iterrows():
        if row["StudentID_SF"] != last_student:
            last_theta = row["Base_Theta"]
            last_difficulty = row["Base_Difficulty"] if update_difficulty else None
            last_student = row["StudentID_SF"]
        else:
            if not pd.isna(row["ItemScore"]):
                if update_difficulty:
                    last_theta, last_difficulty = update_func(
                        last_theta, last_difficulty, row["day"], row["ItemScore"], K=K
                    )
                else:
                    last_theta, _empty = update_func(
                        last_theta, row["RaschLogit"], row["day"], row["ItemScore"], K=K
                    )
        theta_updated.append(last_theta)
        if update_difficulty:
            difficulty_updated.append(last_difficulty)
    data["ThetaUpdated"] = theta_updated
    if update_difficulty:
        data["DifficultyUpdated"] = difficulty_updated
    return data


def run_update_with_new_attempt(data, item_column, K=0.4, update_difficulty=False):
    """
    Enact the updates and add updated theta and difficulty to the dataframe.
    (for update considering students' previous attempt)

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        item_column (str): Column name for item ID.
        K (float): Learning rate (default is 0.4).
        update_difficulty (bool): Whether to update difficulty (default is False).

    Returns:
        pd.DataFrame: DataFrame with updated theta and difficulty.
    """
    last_student, last_theta, last_difficulty = None, None, None
    theta_updated, difficulty_updated, attempted_items = [], [], []
    for idx, row in data.iterrows():
        if row["StudentID_SF"] != last_student:
            last_theta = row["Base_Theta"]
            last_difficulty = row["Base_Difficulty"] if update_difficulty else None
            last_student = row["StudentID_SF"]
            attempted_items = []
        else:
            if not pd.isna(row["ItemScore"]):
                if row[item_column] not in attempted_items:
                    w = 0
                    attempted_items.append(row[item_column])
                else:
                    w = 1
                if update_difficulty:
                    p = 1 / (1 + np.exp(-(last_theta - last_difficulty)))
                    last_theta = last_theta + K * (
                        row["ItemScore"] - (w * p + (1 - w) * 0.5)
                    )
                    last_difficulty = last_difficulty - K * (row["ItemScore"] - p)
                else:
                    p = 1 / (1 + np.exp(-(last_theta - row["RaschLogit"])))
                    last_theta = last_theta + K * (
                        row["ItemScore"] - (w * p + (1 - w) * 0.5)
                    )
                    last_difficulty = row["RaschLogit"] - K * (row["ItemScore"] - p)
        theta_updated.append(last_theta)
        if update_difficulty:
            difficulty_updated.append(last_difficulty)
    data["ThetaUpdated"] = theta_updated
    if update_difficulty:
        data["DifficultyUpdated"] = difficulty_updated
    return data


def compute_difficulty(group, split_num):
    """
    Compute the average Elo-estimated difficulty for each item in the group for CodeInsights dataset.

    Args:
        group (pd.DataFrame): DataFrame grouped by Item
        split_num (int): How we split the data

    Returns:
        float: Average difficulty for the item.
    """
    group = group.sort_values("T")
    difficulties = group["DifficultyUpdated"].values
    if len(difficulties) < split_num:
        return difficulties[-1]
    # Split into split_num parts and drop the first part (i.e., first 1/split_num)
    parts = np.array_split(difficulties, split_num)
    remaining = np.concatenate(parts[split_num - 1:])
    return np.mean(remaining)


def calculate_auc(raw_data, data, columns, difficulty_column):
    """
    Calculate AUC on the masked data.

    Args:
        raw_data (pd.DataFrame): Original data.
        data (pd.DataFrame): Data with updated theta and difficulty.
        columns (list): Selected columns for analysis
        difficulty_column (str): Column name for difficulty.

    Returns:
        None
    """
    # Copy and sort original data
    original_data = raw_data[columns].copy()
    original_data = original_data.sort_values(["StudentID_SF", "T"]).reset_index(
        drop=True
    )
    # Compute predicted probabilities
    data["PredictedProb"] = 1 / (
        1 + np.exp(-(data["ThetaUpdated"] - data[difficulty_column]))
    )
    # Identify mask indices
    valid_mask = (
        pd.isna(data["ItemScore"])
        & ~pd.isna(original_data["ItemScore"])
        & ~pd.isna(data["PredictedProb"])
    )
    masked_indices = data.index[valid_mask]

    # Build DataFrame with masked data
    masked_data = pd.DataFrame(
        {
            "OriginalScore": original_data.loc[masked_indices, "ItemScore"],
            "PredictedProb": data.loc[masked_indices, "PredictedProb"],
            "StudentID": data.loc[masked_indices, "StudentID_SF"],
            "ItemID": data.loc[masked_indices, "ItemID_SF"],
            "Theta": data.loc[masked_indices, "ThetaUpdated"],
            "Difficulty": data.loc[masked_indices, difficulty_column],
        }
    ).dropna()

    # Convert predicted probabilities to binary scores
    masked_data["PredictedScore"] = (masked_data["PredictedProb"] >= 0.5).astype(int)

    print("Out-of-Sample Evaluation Metrics (on masked data):")
    auc = roc_auc_score(masked_data["OriginalScore"], masked_data["PredictedProb"])
    print(f"AUC: {auc:.3f}")


def plot_trajectory(data, difficulty_column, low_bound, high_bound):
    """
    Plot the trajectory of ability and difficulty updates for students.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        difficulty_column (str): Column name for difficulty.
        low_bound (float): Lower bound for y-axis - lowest difficulty/ability
        high_bound (float): Upper bound for y-axis - highest difficulty/ability

    Returns:
        None
    """
    student_ids = data["StudentID_SF"].unique()[:9]

    # Create 3x3 subplots for Ability and Difficulty
    fig_ability, axes_ability = plt.subplots(3, 3, figsize=(20, 20))
    axes_ability = axes_ability.flatten()
    fig_difficulty, axes_difficulty = plt.subplots(3, 3, figsize=(20, 20))
    axes_difficulty = axes_difficulty.flatten()

    for ax_a, ax_d, student_id in zip(axes_ability, axes_difficulty, student_ids):
        df = data[data["StudentID_SF"] == student_id].sort_values("T")
        x = np.arange(len(df))
        ax_a.set_xlim(x[0], x[-1])
        ax_d.set_xlim(x[0], x[-1])

        mask0 = df["ItemScore"] == 0
        mask1 = df["ItemScore"] == 1

        # Plot Ability (ThetaUpdated)
        ax_a.plot(x, df["ThetaUpdated"], color="blue", alpha=0.6)
        ax_a.scatter(
            x[mask0],
            df.loc[mask0, "ThetaUpdated"],
            color="blue",
            edgecolor="black",
            marker="o",
            s=50,
            zorder=3,
        )
        ax_a.scatter(
            x[mask1],
            df.loc[mask1, "ThetaUpdated"],
            color="blue",
            edgecolor="black",
            marker="^",
            s=50,
            zorder=3,
        )
        first_point, last_point = df.iloc[0], df.iloc[-1]
        ax_a.scatter(
            x[0],
            first_point["ThetaUpdated"],
            s=100,
            color="green",
            marker="D",
            edgecolor="black",
            zorder=5,
            label="Start Ability",
        )
        ax_a.scatter(
            x[-1],
            last_point["ThetaUpdated"],
            s=100,
            color="purple",
            marker="D",
            edgecolor="black",
            zorder=5,
            label="End Ability",
        )
        ax_a.set_title(f"Student {student_id} Ability", fontsize=10)
        ax_a.set_xlabel("Data Point")
        ax_a.set_ylabel("ThetaUpdated", color="blue")
        ax_a.set_ylim(low_bound, high_bound)
        ax_a.tick_params(axis="both", labelsize=8)

        # Plot Difficulty
        ax_d.plot(x, df[difficulty_column], color="red", alpha=0.6)
        ax_d.scatter(
            x[mask0],
            df.loc[mask0, difficulty_column],
            color="red",
            edgecolor="black",
            marker="o",
            s=50,
            zorder=3,
        )
        ax_d.scatter(
            x[mask1],
            df.loc[mask1, difficulty_column],
            color="red",
            edgecolor="black",
            marker="^",
            s=50,
            zorder=3,
        )
        ax_d.scatter(
            x[0],
            first_point[difficulty_column],
            s=100,
            color="green",
            marker="*",
            edgecolor="black",
            zorder=5,
            label="Start Difficulty",
        )
        ax_d.scatter(
            x[-1],
            last_point[difficulty_column],
            s=100,
            color="purple",
            marker="*",
            edgecolor="black",
            zorder=5,
            label="End Difficulty",
        )
        ax_d.set_title(f"Student {student_id} Difficulty", fontsize=10)
        ax_d.set_xlabel("Data Point")
        ax_d.set_ylabel(difficulty_column, color="red")
        ax_d.set_ylim(low_bound, high_bound)
        ax_d.tick_params(axis="both", labelsize=8)

    # Global legend for Ability
    marker0 = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=8,
        label="ItemScore = 0",
    )
    marker1 = mlines.Line2D(
        [],
        [],
        color="black",
        marker="^",
        linestyle="None",
        markersize=8,
        label="ItemScore = 1",
    )
    line_theta = mlines.Line2D(
        [], [], color="blue", linestyle="-", label="ThetaUpdated"
    )
    star_start = mlines.Line2D(
        [],
        [],
        color="green",
        marker="D",
        linestyle="None",
        markersize=10,
        label="Start Ability",
    )
    star_end = mlines.Line2D(
        [],
        [],
        color="purple",
        marker="D",
        linestyle="None",
        markersize=10,
        label="End Ability",
    )
    fig_ability.legend(
        handles=[line_theta, marker0, marker1, star_start, star_end],
        loc="upper center",
        ncol=3,
        fontsize=10,
    )

    # Global legend for Difficulty
    line_diff = mlines.Line2D(
        [], [], color="red", linestyle="-", label=f"{difficulty_column} (Difficulty)"
    )
    star_diff_start = mlines.Line2D(
        [],
        [],
        color="green",
        marker="*",
        linestyle="None",
        markersize=10,
        label="Start Difficulty",
    )
    star_diff_end = mlines.Line2D(
        [],
        [],
        color="purple",
        marker="*",
        linestyle="None",
        markersize=10,
        label="End Difficulty",
    )
    fig_difficulty.legend(
        handles=[line_diff, marker0, marker1, star_diff_start, star_diff_end],
        loc="upper center",
        ncol=3,
        fontsize=10,
    )

    fig_ability.tight_layout(rect=[0, 0, 1, 0.95])
    fig_difficulty.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_distribution(data, x, title):
    """
    Plot the distribution of a specified column in the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        x (str): Column name to plot.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[x], bins=30, color="skyblue", edgecolor="black")
    plt.xlabel(x)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.show()


def evaluate_edmentum(orig_data, fit_data, columns, diff_col, traj_params):
    """
    Compute AUC, plot trajectory and distribution.

    Args:
        orig_data (pd.DataFrame): Original data.
        fit_data (pd.DataFrame): Data with updated theta and difficulty.
        columns (list): Selected columns for analysis
        diff_col (str): Column name for difficulty.
        traj_params (tuple): Parameters for trajectory plot.

    Returns:
        None
    """
    calculate_auc(orig_data, fit_data, columns, diff_col)
    plot_trajectory(fit_data, diff_col, *traj_params)
    plot_distribution(fit_data, diff_col, f"Distribution of {diff_col}")


def evaluate_codeinsights(orig_data, exp_data, columns, diff_col, traj_params):
    """
    Compute AUC, plot trajectory and distribution.

    Args:
        orig_data (pd.DataFrame): Original data.
        exp_data (pd.DataFrame): Data with updated theta and difficulty.
        columns (list): Selected columns for analysis
        diff_col (str): Column name for difficulty.
        traj_params (tuple): Parameters for trajectory plot.

    Returns:
        None
    """
    calculate_auc(orig_data, exp_data, columns, diff_col)
    calculate_auc(orig_data, exp_data, columns, "AverageDifficulty")
    plot_trajectory(exp_data, diff_col, *traj_params)
