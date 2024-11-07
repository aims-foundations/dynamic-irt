import argparse
import json
import os
import pickle
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dynamic_irt.gpirt.utils import ensure_dir, set_seed
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from huggingface_hub import snapshot_download
from torch.distributions import Normal
from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm
from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes

plt.rcParams.update(bundles.aaai2024())


def item_response_fn_1PL(theta, z):
    return torch.sigmoid(theta - z)


def goodness_of_fit_1PL(
    z: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    bin_size: int = 6,
):
    # assert y.shape[1] == z.shape[0], f'{y.shape[1]} != {z.shape[0]}'
    assert y.shape[0] == theta.shape[0], f"{y.shape[0]} != {theta.shape[0]}"

    bin_start, bin_end = torch.min(theta), torch.max(theta)
    bins = torch.linspace(bin_start, bin_end, bin_size + 1)
    # print(bins) # [-3. -2. -1.  0.  1.  2.  3.]

    diff_list = []
    for i in tqdm(list(range(z.shape[0])), desc="Computing bins"):
        # for i in tqdm(list(range(1000)), desc="Computing bins"):
        single_z = z[i]

        for j in range(bins.shape[0] - 1):
            y_empirical = []
            y_theoretical = []
            for sidx, s_theta in enumerate(theta):
                s_bin_mask = (s_theta >= bins[j]) & (s_theta < bins[j + 1])
                s_bin_mask = s_bin_mask & (y[sidx] != -1)

                if s_bin_mask.sum() == 0:  # bin empty
                    continue

                y_empirical.append(y[sidx][s_bin_mask])
                y_theoretical.append(
                    item_response_fn_1PL(s_theta[s_bin_mask], single_z)
                )

            y_empirical = torch.concatenate(y_empirical).mean()
            y_theoretical = torch.concatenate(y_theoretical).mean()

            # theta_mid = (bins[j] + bins[j + 1]) / 2
            # y_theoretical = item_response_fn_1PL(
            #     theta_mid, single_z).item()

            diff = 1 - abs(y_empirical - y_theoretical)
            diff_list.append(diff.cpu())

    diff_array = np.array(diff_list)
    mean_diff = np.mean(diff_array)
    return mean_diff, diff_array


def goodness_of_fit_1PL_plot(
    z: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    plot_path: str,
    bin_size: int = 6,
):
    mean_diff, diff_array = goodness_of_fit_1PL(z, theta, y, bin_size)

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
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=1000)
    parser.add_argument(
        "--esp", help="Epsilon for avoiding zero score", type=float, default=1e-3
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)
    ensure_dir(f"results/{args.course_name}")
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}_wtc", repo_type="dataset"
    )

    y_obs = pickle.load(open(f"{data_folder}/correctness_bytc_matrix.pkl", "rb"))
    time_matrix = pickle.load(open(f"{data_folder}/time_matrix.pkl", "rb"))
    # u_obs has shape of n_student x n_question x n_attempt x n_testcase
    # The number of testcases can be different for each question

    # Number of students
    n_student = len(y_obs)

    # Number of questions
    n_question = len(y_obs[0])

    y_obs = torch.load("y_obs.pt")
    qidx_obs = torch.load("qidx_obs.pt")
    time_obs = torch.load("time_obs.pt")

    y_obs = torch.flatten(y_obs, start_dim=1)
    qidx_obs = torch.flatten(qidx_obs, start_dim=1)
    time_obs = torch.flatten(time_obs, start_dim=1)

    first_idx = torch.arange(start=0, end=n_student).reshape(-1, 1)
    sorted_idx = torch.argsort(time_obs, dim=1)
    y_obs = y_obs[first_idx, sorted_idx]
    qidx_obs = qidx_obs[first_idx, sorted_idx]
    time_obs = time_obs[first_idx, sorted_idx]

    masked_idx = y_obs != -1

    unique_time_obs = []
    aidx_obs = []  # student attempt index
    for tidx, time_ob in enumerate(time_obs):
        unique_time_obs.append(time_ob.unique())
        aidx_ob = torch.searchsorted(unique_time_obs[-1], time_ob)
        aidx_ob[aidx_ob == len(unique_time_obs[-1]) - 1] = (
            -1
        )  # Replace the last element with -1
        aidx_obs.append(aidx_ob)

    thetas = pickle.load(open(f"results/thetas_by_iter_{args.epochs}.pkl", "rb"))
    zs = pickle.load(open(f"results/zs_by_iter_{args.epochs}.pkl", "rb"))

    num_train = int(0.8 * n_student)
    num_test = n_student - num_train

    last_thetas = thetas[-1].copy()
    last_zs = zs[-1].clone()
    del thetas
    del zs

    list_saidx = []
    list_sqidx = []
    for sidx in range(n_student):
        if sidx < num_train:
            continue

        if masked_idx[sidx].sum() == 0:
            list_saidx.append(None)
            list_sqidx.append(None)
            continue

        saidx = aidx_obs[sidx][masked_idx[sidx]]  # attemp index for student
        list_saidx.append(saidx)

        sqidx = qidx_obs[sidx][masked_idx[sidx]]  # global testcase index
        list_sqidx.append(sqidx)

    visualized_theta = (torch.ones(num_test, y_obs.shape[1]) * -1.0).float().to(device)
    visualized_y = (torch.ones(num_test, y_obs.shape[1]) * -1.0).float().to(device)
    # visualized_z = (torch.ones((y_obs.shape[1],)) * -1.0).float().to(device)

    print("NUM THETA:", len(last_thetas))
    print("MASKED SHAPE:", masked_idx.shape)

    for sidx, st_theta in enumerate(tqdm(last_thetas)):
        if masked_idx[sidx].sum() == 0:
            continue
        if sidx < num_train:
            # Using test data
            continue

        first_idx = sidx - num_train
        visualized_theta[first_idx, masked_idx[sidx]] = st_theta[list_saidx[first_idx]]
        visualized_y[first_idx, masked_idx[sidx]] = y_obs[sidx][
            masked_idx[sidx]
        ].float()
        # visualized_z[masked_idx[sidx]] = last_zs[list_sqidx[first_idx]] # BUG???

    goodness_of_fit_1PL_plot(
        last_zs,
        visualized_theta,
        visualized_y,
        plot_path=f"results/goodness_of_fit_iter{args.epochs}.png",
        bin_size=6,
    )
