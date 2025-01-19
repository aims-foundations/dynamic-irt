import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from tueplots import bundles, figsizes
from utils import ensure_dir, set_seed

plt.rcParams.update(bundles.icml2024())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--sidx", help="Student index", type=int, default=0)
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
    sampling_diagnostic = pickle.load(
        open(f"{result_folder}/sampling_diagnostic.pkl", "rb")
    )

    ability_keys = list(sampling_diagnostic.keys())
    ability_keys = list(filter(lambda x: x.startswith("ability"), ability_keys))

    total_students = len(ability_keys)
    print(f"Total students: {total_students}")

    metrics = {
        "ability_n_eff": [],
        "ability_r_hat": [],
        "difficulty_n_eff": [],
        "difficulty_r_hat": [],
    }
    for sidx in range(total_students):
        metrics["ability_n_eff"].extend(
            sampling_diagnostic[f"ability_{sidx}"]["n_eff"].tolist()
        )
        metrics["ability_r_hat"].extend(
            sampling_diagnostic[f"ability_{sidx}"]["r_hat"].tolist()
        )

    metrics["difficulty_n_eff"].extend(
        sampling_diagnostic["difficulty"]["n_eff"].tolist()
    )
    metrics["difficulty_r_hat"].extend(
        sampling_diagnostic["difficulty"]["r_hat"].tolist()
    )

    # Draw plots for abilities
    figsize = figsizes.icml2024_full(nrows=1, ncols=2)["figure.figsize"]
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs = axs.flatten()

    axs[0].hist(
        metrics["ability_n_eff"],
        bins=50,
        color="blue",
        alpha=0.7,
        label="Ability",
        density=True,
    )
    axs[0].set_title("Effective Sample Size")
    axs[0].set_xlabel("Number of effective samples")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(
        metrics["ability_r_hat"],
        bins=50,
        color="blue",
        alpha=0.7,
        label="Difficulty",
        density=True,
    )
    axs[1].set_title("Gelman-Rubin Diagnostic")
    axs[1].set_xlabel(r"$\hat{R}$")
    axs[1].set_ylabel("Frequency")
    axs[1].axvline(x=1.1, color="red", linestyle="--", label="Convergence threshold")

    plt.tight_layout()
    plt.savefig(f"{plot_folder}/sampling_diagnostic_ability.png", dpi=300)
    plt.close()

    # Draw plots for difficulties
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs = axs.flatten()

    axs[0].hist(
        metrics["difficulty_n_eff"],
        bins=50,
        color="blue",
        alpha=0.7,
        label="Difficulty",
        density=True,
    )
    axs[0].set_title("Effective Sample Size")
    axs[0].set_xlabel("Number of effective samples")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(
        metrics["difficulty_r_hat"],
        bins=50,
        color="blue",
        alpha=0.7,
        label="Difficulty",
        density=True,
    )
    axs[1].set_title("Gelman-Rubin Diagnostic")
    axs[1].set_xlabel(r"$\hat{R}$")
    axs[1].set_ylabel("Frequency")
    axs[1].axvline(x=1.1, color="red", linestyle="--", label="Convergence threshold")

    plt.tight_layout()
    plt.savefig(f"{plot_folder}/sampling_diagnostic_difficulty.png", dpi=300)
    plt.close()
