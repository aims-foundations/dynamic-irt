import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from amortized_irt import IRT
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes
from utils import ensure_dir, set_seed

plt.rcParams.update(bundles.iclr2024())


# Compute Accuracy, AUC
def compute_metrics(response_matrix, abilities, difficulties):
    metrics = {
        "likelihood": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "auc": [],
    }
    total_samples = difficulties.shape[0]

    for sample_idx in tqdm(range(total_samples), desc="Computing metrics"):
        for sidx in range(n_testtakers):
            ability = abilities[sidx][sample_idx]
            difficulty = difficulties[sample_idx, question_expanding_indexes[sidx]]
            y_probs = torch.sigmoid(ability + difficulty)

            y_preds = (y_probs > 0.5).float()
            y_true = response_matrix[sidx][observation_mask[sidx]]
            y_dist = torch.distributions.Bernoulli(probs=y_probs)

            metrics["likelihood"].append(y_dist.log_prob(y_true).mean().item())
            metrics["accuracy"].append(accuracy_score(y_true.cpu(), y_preds.cpu()))
            metrics["f1"].append(f1_score(y_true.cpu(), y_preds.cpu()))
            metrics["precision"].append(precision_score(y_true.cpu(), y_preds.cpu()))
            metrics["recall"].append(recall_score(y_true.cpu(), y_preds.cpu()))
            try:
                metrics["auc"].append(roc_auc_score(y_true.cpu(), y_probs.cpu()))
            except:
                metrics["auc"].append(0.5)

    return metrics


def plot_metrics(metrics, plot_folder):
    # Plot metrics
    for metric in ["Likelihood", "Accuracy", "F1", "Precision", "Recall", "AUC"]:
        plt.figure(figsize=figsizes.aaai2024_half()["figure.figsize"])
        plt.hist(metrics[metric.lower()], bins=40, density=True, alpha=0.4)
        plt.axvline(np.mean(metrics[metric.lower()]), linestyle="--")
        plt.text(
            np.mean(metrics[metric.lower()]),
            plt.gca().get_ylim()[1],
            f"{np.mean(metrics[metric.lower()]):.2f} $\\pm$ {np.std(metrics[metric.lower()]):.2f}",
            ha="center",
            va="bottom",
        )
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.savefig(f"{plot_folder}/{metric.lower()}.png", dpi=300)
        plt.close()


# Compute Goodness of Fit
def compute_gof(response_matrix, abilities, difficulties, bin_size=6):
    # Get the min and max of abilities
    for i in range(n_testtakers):
        if i == 0:
            min_ability = abilities[i].min()
            max_ability = abilities[i].max()
        else:
            min_ability = min(min_ability, abilities[i].min())
            max_ability = max(max_ability, abilities[i].max())

    ability_bins = torch.linspace(min_ability, max_ability, bin_size + 1)

    diff_list = []
    for bin_start, bin_end in zip(ability_bins[:-1], ability_bins[1:]):
        print(f"Bin: {bin_start.item()} - {bin_end.item()}")
        for sidx, s_ability in enumerate(tqdm(abilities)):
            ability_mask = (s_ability >= bin_start) & (s_ability < bin_end)
            if ability_mask.sum() == 0:
                continue

            filtered_ability = s_ability[ability_mask]
            filtered_difficulty = difficulties[:, question_expanding_indexes[sidx]][
                ability_mask
            ]

            y_theoretical = IRT.compute_prob(
                filtered_ability,
                filtered_difficulty,
                disciminatory=1,
                guessing=0,
                loading_factor=1,
            )
            y_empirical = response_matrix[sidx][observation_mask[sidx]]
            y_empirical = y_empirical[None].expand_as(s_ability)[ability_mask]

            diff = 1 - torch.abs(y_empirical - y_theoretical)
            diff_list.append(diff.mean().item())

    diff_array = np.array(diff_list)
    mean_diff = diff_array.mean()
    return diff_array, mean_diff


def plot_gof(
    diff_array: np.ndarray,
    mean_diff: float,
    plot_path: str,
):
    sample_means = []
    for _ in range(100):
        indices = np.random.choice(
            len(diff_array), int(0.8 * len(diff_array)), replace=False
        )
        sample_mean = np.mean(diff_array[indices])
        sample_means.append(sample_mean)
    std_diff = np.std(sample_means)

    plt.figure(figsize=figsizes.aaai2024_half()["figure.figsize"])
    plt.hist(diff_array, bins=40, density=True, alpha=0.4)
    plt.xlabel(r"Difference between empirical and theoretical $P(y=1)$")
    plt.ylabel(
        r"Goodness of fit",
    )
    plt.tick_params(axis="both")
    plt.xlim(0, 1)
    plt.axvline(mean_diff, linestyle="--")
    plt.text(
        mean_diff,
        plt.gca().get_ylim()[1],
        f"{mean_diff:.2f} $\\pm$ {3 * std_diff:.2f}",
        ha="center",
        va="bottom",
    )
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return mean_diff, std_diff


if __name__ == "__main__":
    # wandb.init(project="code_insights")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument(
        "--kernel",
        help="Prior Kernel",
        type=str,
        default="RBF",
        choices=["RBF", "Matern"],
    )
    parser.add_argument("--length_scale", help="Length scale", type=float, default=1.0)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--fitting_method", type=str, default="hmc")
    parser.add_argument("--thinning", type=int, default=1)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_folder = f"results/{args.course_name}_s{args.seed}_D{args.D}_PL{args.PL}_{args.fitting_method}_kernel{args.kernel}_ls{args.length_scale}"
    plot_folder = f"plots/{args.course_name}_s{args.seed}_D{args.D}_PL{args.PL}_{args.fitting_method}_kernel{args.kernel}_ls{args.length_scale}"
    ensure_dir(plot_folder)

    #################################################################################

    # Download and load data
    data_folder = snapshot_download(
        repo_id=f"stair-lab/code_insights_matrices", repo_type="dataset"
    )
    data_folder = os.path.join(data_folder, args.course_name)

    # Load matrices
    response_matrix = torch.load(f"{data_folder}/correctness_matrix.pt").to(
        device, dtype=torch.float32
    )
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts

    response_time_matrix = torch.load(f"{data_folder}/time_matrix.pt").to(
        device, dtype=torch.float32
    )
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts

    # Load the ability and difficulty
    abilities = torch.load(f"{result_folder}/ability.pt")
    difficulties = torch.load(f"{result_folder}/difficulty.pt").to(device)

    if args.thinning > 1:
        abilities = [ab[:: args.thinning] for ab in abilities]
        difficulties = difficulties[:: args.thinning]

    # Get some necessary indexes
    n_testtakers, n_questions, n_max_attempts = response_matrix.shape
    observation_mask = response_matrix != -1
    response_time_indexes = []
    question_expanding_indexes = []
    for sidx in tqdm(range(n_testtakers), desc="Constructing indexes"):
        uni_time = response_time_matrix[sidx].unique()
        has_missing = 1 if -1 in uni_time else 0

        # Get the time index for each attempt
        time_index = (
            torch.searchsorted(uni_time, response_time_matrix[sidx]) - has_missing
        )
        ### REMEMBER: time_index is 0-indexed. Element 0 is -1
        ### We need to subtract 1 to get the correct index
        response_time_indexes.append(time_index[time_index != -1])
        question_expanding_indexes.append(
            torch.arange(n_questions, device=device)[:, None].expand(
                -1, n_max_attempts
            )[observation_mask[sidx]]
        )

    # Incase of hmc, index the abilities
    if args.fitting_method == "hmc":
        abilities = [
            abilities[sidx].to(device)[:, response_time_indexes[sidx]]
            for sidx in range(n_testtakers)
        ]

    # Incase of ess, remove warmup samples
    if args.fitting_method == "ess":
        num_wamrup = int(difficulties.shape[0] / 8)
        abilities = [
            abilities[sidx][num_wamrup:].to(device) for sidx in range(n_testtakers)
        ]
        difficulties = difficulties[num_wamrup:]

    # Compute Goodness of Fit
    diff_array, mean_diff = compute_gof(response_matrix, abilities, difficulties)
    plot_gof(diff_array, mean_diff, f"{plot_folder}/gof.png")

    # Compute metrics
    metrics = compute_metrics(response_matrix, abilities, difficulties)

    # Save metrics as json
    with open(f"{result_folder}/metrics.json", "w") as f:
        json.dump(metrics, f)

    # Plot metrics
    plot_metrics(metrics, plot_folder)
