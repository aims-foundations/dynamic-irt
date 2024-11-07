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
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from huggingface_hub import snapshot_download
from torch.distributions import Normal
from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm
from tueplots import bundles
from utils import ensure_dir, parse_time, set_seed

plt.rcParams.update(bundles.aaai2024())


def plot_correlation(x, y, x_label, y_label, fig_title, save_file):
    plt.figure(figsize=(5, 5))
    axis_max = max(x.max(), y.max())
    axis_min = min(x.min(), y.min())
    plt.scatter(x, y)
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    plt.savefig(f"plots/{save_file}", dpi=300)


# Eliptical Slice Sampling
def ess_z(
    current_f, nu, previous_thetas, ll_dist, y_train, num_train, list_saidx, list_sqidx
):
    ll_current = ll_dist.log_prob(y_train).sum()
    # n_student = len(previous_thetas)
    # ll_thres = ll_current + (n_student * torch.log(torch.rand(1)).to(ll_current))
    ll_thres = ll_current + torch.log(torch.rand(1)).to(ll_current)
    print("Z log-likelihood", ll_current.item())

    angle = (torch.rand(1) * 2 * np.pi).to(ll_current)
    angle_min, angle_max = angle - 2 * np.pi, angle

    while True:
        next_f = torch.cos(angle) * current_f + torch.sin(angle) * nu
        tmz = []
        for sidx, st_theta in enumerate(previous_thetas):
            if sidx >= num_train:
                break
            if masked_idx[sidx].sum() == 0:
                continue
            st = st_theta[list_saidx[sidx]]  # get the theta value for the student
            sz = next_f[list_sqidx[sidx]]
            tmz.append(st - sz)
        tmz = torch.concatenate(tmz)

        likelihood_dist = Bernoulli(logits=tmz)

        log_likelihood = likelihood_dist.log_prob(y_train).sum()

        if log_likelihood > ll_thres:
            break
        else:
            if angle == 0:
                return current_f

            if angle < 0:
                angle_min = angle
            else:
                angle_max = angle
            angle = torch.rand(1).to(angle) * (angle_max - angle_min) + angle_min

    return next_f


# Eliptical Slice Sampling
def ess_theta(
    previous_thetas, nu, next_z, ll_dist, y_train, num_train, list_saidx, list_sqidx
):
    ll_current = ll_dist.log_prob(y_train).sum()
    # n_student = len(previous_thetas)
    # ll_thres = ll_current + (n_student * torch.log(torch.rand(1)).to(ll_current))
    ll_thres = ll_current + torch.log(torch.rand(1)).to(ll_current)
    print("Theta log-likelihood", ll_current.item())

    angle = (torch.rand(1) * 2 * np.pi).to(ll_current)
    angle_min, angle_max = angle - 2 * np.pi, angle

    while True:
        tmz = []
        next_thetas = []
        for sidx, st_theta in enumerate(previous_thetas):
            if masked_idx[sidx].sum() == 0:
                next_thetas.append([])
                continue

            next_theta = torch.cos(angle) * st_theta + torch.sin(angle) * nu[sidx]
            next_thetas.append(next_theta)

            if sidx >= num_train:
                continue

            next_st = next_theta[
                list_saidx[sidx]
            ]  # get the theta value for the student
            sz = next_z[list_sqidx[sidx]]
            tmz.append(next_st - sz)
        tmz = torch.concatenate(tmz)

        likelihood_dist = Bernoulli(logits=tmz)

        log_likelihood = likelihood_dist.log_prob(y_train).sum()

        if log_likelihood > ll_thres:
            break
        else:
            if angle == 0:
                return previous_thetas

            if angle < 0:
                angle_min = angle
            else:
                angle_max = angle
            angle = torch.rand(1).to(angle) * (angle_max - angle_min) + angle_min

    return next_thetas


if __name__ == "__main__":
    # wandb.init(project="code_insights")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10000)
    parser.add_argument("--is_continue", help="Continue sampling", type=int, default=0)
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

    # Number of questions
    n_question = len(y_obs[0])

    # Compute maximum number of testcases for each question
    list_max_testcases = {}
    for qidx in range(n_question):
        list_max_testcases[qidx] = 0
        for student in y_obs:
            list_n_testcases = []
            for x in student[qidx]:
                if not isinstance(x, int):
                    list_n_testcases.append(len(x))
                else:
                    list_n_testcases.append(0)
            list_max_testcases[qidx] = max(
                list_max_testcases[qidx], max(list_n_testcases)
            )

    total_testcases = sum(list_max_testcases.values())

    y_obs = torch.load("data/y_obs.pt")
    qidx_obs = torch.load("data/tidx_obs.pt")
    time_obs = torch.load("data/time_obs.pt")

    y_obs = torch.flatten(y_obs, start_dim=1)
    qidx_obs = torch.flatten(qidx_obs, start_dim=1)
    time_obs = torch.flatten(time_obs, start_dim=1)

    n_student = len(y_obs)
    first_idx = torch.arange(start=0, end=n_student).reshape(-1, 1)
    sorted_idx = torch.argsort(time_obs, dim=1)
    y_obs = y_obs[first_idx, sorted_idx]
    qidx_obs = qidx_obs[first_idx, sorted_idx]
    time_obs = time_obs[first_idx, sorted_idx]

    masked_idx = y_obs != -1

    # Create normal prior distribution for theta, where each theta_i is corresponding to a student at a specific time
    unique_time_obs = []
    aidx_obs = []  # student attempt index
    for tidx, time_ob in enumerate(time_obs):
        unique_time_obs.append(time_ob.unique())
        aidx_ob = torch.searchsorted(unique_time_obs[-1], time_ob)
        aidx_ob[aidx_ob == len(unique_time_obs[-1]) - 1] = (
            -1
        )  # Replace the last element with -1
        aidx_obs.append(aidx_ob)

    def get_theta_priors(sidx):
        time_obs_s = unique_time_obs[sidx][
            :-1
        ]  # Remove last element -- it's the empty one
        time_obs_s = time_obs_s.reshape(-1, 1)
        kernel = ScaleKernel(RBFKernel(length_scale=1.0)).to(device)
        covar = kernel(time_obs_s)

        return MultivariateNormal(
            torch.zeros(covar.shape[0], device=device), covariance_matrix=covar
        )

    def sample_theta_prior(sidx):
        if masked_idx[sidx].sum() == 0:
            return []
        else:
            return get_theta_priors(sidx).sample()

    # Create normal prior distribution for z, where each z_i is corresponding to a testcase in a question
    z_priors = []
    for qidx in range(total_testcases):
        z_priors.append(
            Normal(
                loc=torch.tensor(0.0, device=device),
                scale=torch.tensor(1.0, device=device),
            )
        )
    # covar = torch.ones((total_testcases,total_testcases), device=device)*1e-10
    # covar = covar.fill_diagonal_(1)
    # z_priors = MultivariateNormal(
    #     torch.zeros(total_testcases, device=device),
    #     covariance_matrix=covar
    # )

    # Create y_train
    num_train = int(0.8 * n_student)

    y_train = []
    y_test = []
    for sidx in range(n_student):
        if masked_idx[sidx].sum() == 0:
            continue

        if sidx < num_train:
            y_train.append(y_obs[sidx][masked_idx[sidx]])
        else:
            y_test.append(y_obs[sidx][masked_idx[sidx]])
    y_train = torch.concatenate(y_train).float()
    y_test = torch.concatenate(y_test).float()

    if args.is_continue:
        list_thetas = pickle.load(
            open(f"results/thetas_by_iter_{args.is_continue}.pkl", "rb")
        )
        list_zs = pickle.load(open(f"results/zs_by_iter_{args.is_continue}.pkl", "rb"))

        continue_iter = len(list_zs)
        list_thetas = list_thetas[-1:]
        list_zs = list_zs[-1:]
    else:
        list_thetas = [[sample_theta_prior(tp) for tp in range(n_student)]]
        list_zs = [
            torch.stack([z.sample() for z in z_priors])
            # z_priors.sample()
        ]
        continue_iter = 0

    list_saidx = []
    list_sqidx = []
    for sidx in range(n_student):
        if sidx >= num_train:
            break

        if masked_idx[sidx].sum() == 0:
            list_saidx.append(None)
            list_sqidx.append(None)
            continue

        saidx = aidx_obs[sidx][masked_idx[sidx]]  # attemp index for student
        list_saidx.append(saidx)

        sqidx = qidx_obs[sidx][masked_idx[sidx]]  # global testcase index
        list_sqidx.append(sqidx)

    for epoch in tqdm(range(continue_iter, args.epochs), desc="Sampling"):
        previous_thetas = list_thetas[-1]
        previous_zs = list_zs[-1]

        # Sampling for z
        tmz = []
        for sidx, st_theta in enumerate(previous_thetas):
            if sidx >= num_train:
                break

            if masked_idx[sidx].sum() == 0:
                continue

            st = st_theta[list_saidx[sidx]]  # get the theta value for the student
            sz = previous_zs[list_sqidx[sidx]]  # get the z value for the testcase
            tmz.append(st - sz)
        tmz = torch.concatenate(tmz)
        ll_dist = Bernoulli(logits=tmz)

        zs_noise = torch.stack([z.sample() for z in z_priors])
        # zs_noise = z_priors.sample()
        next_z = ess_z(
            previous_zs,
            zs_noise,
            previous_thetas,
            ll_dist,
            y_train,
            num_train,
            list_saidx,
            list_sqidx,
        )
        list_zs.append(next_z)

        # Sampling for theta
        tmz = []
        for sidx, st_theta in enumerate(previous_thetas):
            if sidx >= num_train:
                break
            if masked_idx[sidx].sum() == 0:
                continue
            st = st_theta[list_saidx[sidx]]  # get the theta value for the student
            # ==================
            sz = next_z[list_sqidx[sidx]]
            # ==================
            tmz.append(st - sz)
        tmz = torch.concatenate(tmz)

        ll_dist = Bernoulli(logits=tmz)
        thetas_noise = [sample_theta_prior(tp) for tp in range(n_student)]
        next_theta = ess_theta(
            previous_thetas,
            thetas_noise,
            next_z,
            ll_dist,
            y_train,
            num_train,
            list_saidx,
            list_sqidx,
        )
        list_thetas.append(next_theta)

        if epoch > 195 and (epoch + 1) % 200 == 0:
            # Extract optimized parameters
            with open(f"results/thetas_by_iter_{epoch + 1}.pkl", "wb") as f:
                pickle.dump(list_thetas, f)

            with open(f"results/zs_by_iter_{epoch + 1}.pkl", "wb") as f:
                pickle.dump(list_zs, f)

    # # wandb.finish()
