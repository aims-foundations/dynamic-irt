import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes
from utils import ensure_dir, set_seed

plt.rcParams.update(bundles.iclr2024())


def item_response_fn_1PL(theta, z):
    return torch.sigmoid(theta + z)


# Compute Accuracy, AUC
def compute_metrics(response_matrix, abilities, difficulties):
    accuracy = []
    auc = []
    total_samples = difficulties.shape[0]

    for sample_idx in tqdm(range(total_samples), desc="Computing metrics"):
        for sidx in range(n_testtakers):
            ability = abilities[sidx][sample_idx]
            difficulty = difficulties[sample_idx, question_expanding_indexes[sidx]]
            y_probs = item_response_fn_1PL(ability, difficulty)

            y_preds = (y_probs > 0.5).float()
            y_true = response_matrix[sidx][observation_mask[sidx]]

            accuracy.append(accuracy_score(y_true.cpu(), y_preds.cpu()))
            try:
                auc.append(roc_auc_score(y_true.cpu(), y_probs.cpu()))
            except:
                auc.append(0.5)

    print(f"Accuracy: {np.mean(accuracy):.4f} +/- {np.std(accuracy):.4f}")
    print(f"AUC: {np.mean(auc):.4f} +/- {np.std(auc):.4f}")

    return accuracy, auc


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

            y_theoretical = item_response_fn_1PL(filtered_ability, filtered_difficulty)
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
    parser.add_argument("--iteration", help="# Iteration", type=int, default=100000)
    parser.add_argument("--sidx", help="Student index", type=int, default=0)
    parser.add_argument(
        "--kernel",
        help="Prior Kernel",
        type=str,
        default="RBF",
        choices=["RBF", "Matern"],
    )
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--fitting_method", type=str, default="hmc")
    parser.add_argument("--npoints", type=int, default=500)
    parser.add_argument("--length_scale", help="Length scale", type=float, default=1.0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_folder = f"results/{args.course_name}_s{args.seed}_D{args.D}_PL{args.PL}_{args.fitting_method}_kernel{args.kernel}_ls{args.length_scale}"
    ensure_dir("plots/")

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

    abilities = torch.load(f"{result_folder}/ability.pt")
    difficulties = torch.load(f"{result_folder}/difficulty.pt").to(device)

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
    plot_gof(diff_array, mean_diff, f"{result_folder}/gof.png")

    # Compute metrics
    compute_metrics(response_matrix, abilities, difficulties)
