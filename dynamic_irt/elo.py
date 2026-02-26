"""
Elo-based IRT for CodeInsights dataset (D1 human data).

Runs a standard Elo rating model that jointly updates student ability (theta)
and item difficulty (b) after each interaction.

Data is loaded from HuggingFace (stair-lab/code_insights_csv).
Results are saved to CodeInsights/results/elo/.

Usage:
    cd CodeInsights && python -m dynamic_irt.elo
"""

import json
import os
import random
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from huggingface_hub import login, snapshot_download
from sklearn.metrics import roc_auc_score
from tueplots import bundles, figsizes

matplotlib.use("Agg")
plt.rcParams.update(bundles.aaai2024())

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, REPO_ROOT)
from dynamic_irt.gpirt.utils import ensure_dir

# Standardized color palette (Paul Tol qualitative) — matches CIRT / Dynamic IRT
COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44"]


# ---------------------------------------------------------------------------
# Elo update functions
# ---------------------------------------------------------------------------

def mask_responses(df, mask_fraction=0.2):
    """Randomly mask a fraction of responses to create a held-out test set."""
    mask = np.random.rand(len(df)) < mask_fraction
    df.loc[mask, "ItemScore"] = np.nan
    return df


def basic_update(th, b, day, resp, K=0.4):
    """Standard Elo update for ability and difficulty."""
    p = 1 / (1 + np.exp(-(th - b)))
    return th + K * (resp - p), b - K * (resp - p)


def run_update(data, update_func, K=0.4, update_difficulty=False):
    """Run sequential Elo updates across all student interactions."""
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


def compute_difficulty(group, split_num):
    """Compute average Elo-estimated difficulty from the last 1/split_num of updates."""
    group = group.sort_values("T")
    difficulties = group["DifficultyUpdated"].values
    if len(difficulties) < split_num:
        return difficulties[-1]
    parts = np.array_split(difficulties, split_num)
    remaining = np.concatenate(parts[split_num - 1:])
    return np.mean(remaining)


def evaluate(raw_data, data, columns, difficulty_column):
    """Calculate AUC and RMSE on the masked held-out data.

    AUC is computed on binarized scores (all tests pass = success).
    RMSE is computed on the raw fractional scores.
    """
    original_data = raw_data[columns].copy()
    original_data = original_data.sort_values(["StudentID_SF", "T"]).reset_index(
        drop=True
    )
    data["PredictedProb"] = 1 / (
        1 + np.exp(-(data["ThetaUpdated"] - data[difficulty_column]))
    )
    valid_mask = (
        pd.isna(data["ItemScore"])
        & ~pd.isna(original_data["ItemScore"])
        & ~pd.isna(data["PredictedProb"])
    )
    masked_indices = data.index[valid_mask]

    masked_data = pd.DataFrame(
        {
            "OriginalScore": original_data.loc[masked_indices, "ItemScore"],
            "PredictedProb": data.loc[masked_indices, "PredictedProb"],
        }
    ).dropna()

    rmse = np.sqrt(np.mean((masked_data["OriginalScore"] - masked_data["PredictedProb"]) ** 2))

    binary_true = (masked_data["OriginalScore"] >= 1.0).astype(int)
    auc = roc_auc_score(binary_true, masked_data["PredictedProb"])

    print(f"Out-of-Sample:  AUC: {auc:.4f}  RMSE: {rmse:.4f}  N: {len(masked_data)}")

    return {"auc": auc, "rmse": rmse, "n_masked": len(masked_data)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load and preprocess CodeInsights D1 data from HuggingFace."""
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not set. Using cached credentials if available.")

    path = snapshot_download(
        repo_id="stair-lab/code_insights_csv", repo_type="dataset"
    )
    code_insights = pd.read_csv(f"{path}/main_data.csv")

    # Filter to Submit/Prechecked responses
    filtered_code = code_insights[
        code_insights["response_type"].isin(["Submit", "Prechecked"])
    ].copy()

    # Clean pass column
    def remove_decimal_if_whole(val):
        try:
            val_str = str(val)
            if "." in val_str:
                num = float(val_str)
                if num.is_integer():
                    return str(int(num))
                return val_str
            return val_str
        except ValueError:
            return str(val)

    filtered_code["pass"] = filtered_code["pass"].apply(remove_decimal_if_whole)
    filtered_code["pass"] = filtered_code["pass"].replace("nan", np.nan)
    filtered_code = filtered_code.dropna(subset=["pass"])

    # Compute time
    filtered_code["timestamp"] = pd.to_datetime(
        filtered_code["timestamp"], format="%d/%m/%y, %H:%M:%S"
    )
    filtered_code["T"] = filtered_code.groupby("student_id")["timestamp"].transform(
        lambda x: (x - x.min()).dt.total_seconds()
    )
    filtered_code = filtered_code.reset_index(drop=True)

    # Submission-level aggregation: one row per submission, fraction of tests passed.
    # Avoids exploding correlated test cases into independent Elo observations.
    def pass_fraction(s):
        s = str(s).strip()
        return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan

    filtered_code["ItemScore"] = filtered_code["pass"].apply(pass_fraction)
    filtered_code = filtered_code.dropna(subset=["ItemScore"])
    filtered_code = filtered_code.sort_values(["student_id", "T"])
    filtered_code["time_since_last_attempt"] = (
        filtered_code.groupby("student_id")["T"].diff().fillna(0)
    )

    data = filtered_code[
        ["student_id", "question_unittest_id", "T", "ItemScore",
         "time_since_last_attempt"]
    ].rename(columns={
        "student_id": "StudentID_SF",
        "question_unittest_id": "ItemID_SF",
    })

    data["day"] = data["time_since_last_attempt"] // 86400

    return data


# ---------------------------------------------------------------------------
# Visualization (matches CIRT / Dynamic IRT style)
# ---------------------------------------------------------------------------

def plot_param_hist(values, param_name, filename, result_dir, xlabel,
                    bins=30, pct_clip=(1, 99)):
    """Plot histogram + KDE, clipping x-axis to the given percentile range."""
    lo, hi = np.percentile(values, pct_clip)
    clipped = values[(values >= lo) & (values <= hi)]

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_half()["figure.figsize"])
    ax.hist(clipped, bins=bins, density=True, alpha=0.3, color=COLORS[0])
    sns.kdeplot(clipped, color=COLORS[0], linewidth=1.5, bw_adjust=0.5, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{param_name} Distribution")
    save_path = os.path.join(result_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_trajectories(fit_data, result_dir, prefix, n_students=5, window=30):
    """Plot ability trajectories as small multiples with rolling-mean smoothing.

    Each student gets its own subplot: raw theta in faint color, rolling mean
    in bold.  A shared y-axis makes cross-student comparison easy.
    """
    student_stats = fit_data.groupby("StudentID_SF").agg(
        first_theta=("ThetaUpdated", "first"),
        last_theta=("ThetaUpdated", "last"),
        n_points=("ThetaUpdated", "count"),
    )
    student_stats["gain"] = student_stats["last_theta"] - student_stats["first_theta"]
    positive = student_stats[
        (student_stats["gain"] > 0) & (student_stats["n_points"] >= 10)
    ]
    if positive.empty:
        print("  Warning: No students with positive ability gain — skipping trajectory.")
        return
    top_ids = positive.nlargest(n_students, "gain").index

    full_width = figsizes.aaai2024_full()["figure.figsize"][0]
    fig, axes = plt.subplots(
        n_students, 1, figsize=(full_width, 1.3 * n_students),
        sharex=True, sharey=True,
    )
    if n_students == 1:
        axes = [axes]

    cmap = plt.cm.tab20
    # Global question-to-color mapping so the same question has the same color
    # across all panels. Use hash for stable assignment.
    all_items = fit_data["ItemID_SF"].unique()
    item_color_idx = {q: hash(str(q)) % 20 for q in all_items}

    for ax, sid, color in zip(axes, top_ids, COLORS):
        df = fit_data[fit_data["StudentID_SF"] == sid].sort_values("T")
        x = np.arange(len(df))
        theta = df["ThetaUpdated"].values
        items = df["ItemID_SF"].values
        gain = student_stats.loc[sid, "gain"]

        # Shade consecutive runs of the same question
        n_unique = len(set(items))
        run_start = 0
        for i in range(1, len(items) + 1):
            if i == len(items) or items[i] != items[run_start]:
                q_color = cmap(item_color_idx[items[run_start]] / 20)
                ax.axvspan(run_start - 0.5, i - 0.5, alpha=0.10, color=q_color,
                           linewidth=0)
                run_start = i

        ax.plot(x, theta, color="0.4", alpha=0.2, linewidth=0.5)
        smoothed = pd.Series(theta).rolling(window, min_periods=1, center=True).mean()
        ax.plot(x, smoothed, color=color, linewidth=1.5)
        ax.set_ylabel(r"$\theta$", fontsize=7)
        ax.set_title(
            f"Student {sid}  ($\\Delta\\theta={gain:.2f}$,  n={len(df)},  "
            f"{n_unique} questions)",
            fontsize=7, pad=2,
        )
        ax.tick_params(labelsize=6)

    axes[-1].set_xlabel("Interaction index", fontsize=7)
    fig.tight_layout(h_pad=0.6)
    save_path = os.path.join(result_dir, f"{prefix}ability_trajectories.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def visualize(fit_data, result_dir, prefix):
    """Generate all result visualizations for one Elo experiment."""
    final_theta = fit_data.groupby("StudentID_SF")["ThetaUpdated"].last().values
    plot_param_hist(final_theta, r"$\theta$", f"{prefix}theta", result_dir,
                    r"$\theta$ (Final Ability)")

    avg_diff = fit_data.groupby("ItemID_SF")["AverageDifficulty"].first().dropna().values
    plot_param_hist(avg_diff, r"$b$", f"{prefix}difficulty", result_dir,
                    r"$b$ (Item Difficulty)", bins=50, pct_clip=(2, 98))

    plot_trajectories(fit_data, result_dir, prefix)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_estimates(fit_data, output_dir, prefix):
    """Save ability and difficulty estimates to CSV."""
    ability = fit_data.groupby("StudentID_SF").agg(
        final_ability=("ThetaUpdated", "last"),
        n_interactions=("ThetaUpdated", "count"),
    ).reset_index()
    ability_path = os.path.join(output_dir, f"{prefix}_ability.csv")
    ability.to_csv(ability_path, index=False)

    difficulty = fit_data.groupby("ItemID_SF").agg(
        difficulty_updated=("DifficultyUpdated", "last"),
        average_difficulty=("AverageDifficulty", "first"),
        n_responses=("DifficultyUpdated", "count"),
    ).reset_index()
    difficulty_path = os.path.join(output_dir, f"{prefix}_difficulty.csv")
    difficulty.to_csv(difficulty_path, index=False)

    print(f"  Saved: {ability_path} ({len(ability)} students)")
    print(f"  Saved: {difficulty_path} ({len(difficulty)} items)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seed = 123
    random.seed(seed)
    np.random.seed(seed)

    mask_fraction = 0.2
    K = 0.4
    output_dir = os.path.join(REPO_ROOT, "results", "elo")
    ensure_dir(output_dir)

    # Clean old results
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    ci_cols = [
        "StudentID_SF", "ItemID_SF", "day",
        "ItemScore", "time_since_last_attempt", "T",
    ]

    # Load D1
    print("Loading D1 (human) data from HuggingFace...")
    ci_data = load_data()
    print(
        f"D1 loaded: {len(ci_data)} records, "
        f"{ci_data['StudentID_SF'].nunique()} students, "
        f"{ci_data['ItemID_SF'].nunique()} items\n"
    )

    all_metrics = {}

    # ------------------------------------------------------------------
    # Elo: joint theta & difficulty update
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Basic Elo (Theta & Difficulty)")
    print("=" * 60)
    t0 = time.time()

    ci_fit_data = ci_data.copy()
    ci_fit_data["Base_Theta"] = 0
    ci_fit_data["Base_Difficulty"] = 0
    ci_fit_data = ci_fit_data.sort_values(["StudentID_SF", "T"]).reset_index(drop=True)
    ci_fit_data = mask_responses(ci_fit_data, mask_fraction=mask_fraction)
    ci_fit_data["ThetaUpdated"] = np.nan
    ci_fit_data["DifficultyUpdated"] = np.nan
    ci_fit_data = run_update(ci_fit_data, basic_update, K=K, update_difficulty=True)
    ci_av_diff = ci_fit_data.groupby("ItemID_SF").apply(
        compute_difficulty, split_num=5, include_groups=False
    )
    ci_fit_data["AverageDifficulty"] = ci_fit_data["ItemID_SF"].map(ci_av_diff)

    metrics = evaluate(ci_data, ci_fit_data, ci_cols, "AverageDifficulty")
    metrics["time_seconds"] = round(time.time() - t0, 1)
    all_metrics["basic_elo"] = metrics
    save_estimates(ci_fit_data, output_dir, "elo")
    print()

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    all_metrics["data_summary"] = {
        "source": "stair-lab/code_insights_csv",
        "n_records": len(ci_data),
        "n_students": int(ci_data["StudentID_SF"].nunique()),
        "n_items": int(ci_data["ItemID_SF"].nunique()),
        "mask_fraction": mask_fraction,
        "K": K,
        "seed": seed,
    }
    metrics_path = os.path.join(output_dir, "fit_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved: {metrics_path}")

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    visualize(ci_fit_data, output_dir, "elo_")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    m = all_metrics["basic_elo"]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  AUC: {m['auc']:.4f}  RMSE: {m['rmse']:.4f}  Time: {m['time_seconds']:.1f}s")
    print(f"\nResults saved to: {output_dir}/")
    print("=" * 60)
