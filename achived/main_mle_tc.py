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
from utils import ensure_dir, parse_time, set_seed

matplotlib.rcParams["text.usetex"] = True


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
def ess_z(current_f, nu, previous_thetas, ll_dist, y_train, num_train):
    ll_current = ll_dist.log_prob(y_train).sum()
    ll_thres = ll_current + torch.log(torch.rand(1)).to(ll_current)

    angle = (torch.rand(1) * 2 * np.pi).to(ll_current)
    angle_min, angle_max = angle - 2 * np.pi, angle

    while True:
        next_f = torch.cos(angle) * current_f + torch.sin(angle) * nu

        tmz = []
        for sidx, st_theta in enumerate(previous_thetas):
            if masked_idx[sidx].sum() == 0:
                continue
            if sidx >= num_train:
                break
            sqidx = qidx_obs[sidx][masked_idx[sidx]]
            sz = next_f[sqidx]
            tmz.append(st_theta - sz)
        tmz = torch.concatenate(tmz)

        likelihood_dist = Bernoulli(logits=tmz)

        log_likelihood = likelihood_dist.log_prob(y_train).sum()

        if log_likelihood > ll_thres:
            break
        else:
            if angle < 0:
                angle_min = angle
            else:
                angle_max = angle
            angle = torch.rand(1).to(angle) * (angle_max - angle_min) + angle_min

    return next_f


# Eliptical Slice Sampling
def ess_theta(previous_theta, nu, next_z, ll_dist, y_train, num_train):
    ll_current = ll_dist.log_prob(y_train).sum()
    ll_thres = ll_current + torch.log(torch.rand(1)).to(ll_current)

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

            sqidx = qidx_obs[sidx][masked_idx[sidx]]
            sz = next_z[sqidx]
            tmz.append(next_theta - sz)
        tmz = torch.concatenate(tmz)

        likelihood_dist = Bernoulli(logits=tmz)

        log_likelihood = likelihood_dist.log_prob(y_train).sum()

        if log_likelihood > ll_thres:
            break
        else:
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

    # # Compute maximum number of testcases for each question
    # list_max_testcases = {}
    # for qidx in range(n_question):
    #     list_max_testcases[qidx] = 0
    #     for student in y_obs:
    #         list_n_testcases = []
    #         for x in student[qidx]:
    #             if not isinstance(x, int):
    #                 list_n_testcases.append(len(x))
    #             else:
    #                 list_n_testcases.append(0)
    #         list_max_testcases[qidx] = max(list_max_testcases[qidx], max(list_n_testcases))

    # # Flatten the testcases into questions
    # # The new shape of y_obs is n_student x (n_question*n_testcase) x n_attempt
    # y_tc_obs = []
    # time_tc_obs = []
    # qidx_tc_obs = []
    # for sidx, student in enumerate(tqdm(y_obs, desc="Preprocessing")):
    #     student_tc = []
    #     student_time = []
    #     student_qidx = []
    #     for qidx in range(n_question):
    #         for tidx in range(list_max_testcases[qidx]):
    #             student_tc.append([])
    #             student_time.append([])
    #             student_qidx.append([])

    #             for aidx, attempt in enumerate(student[qidx]):

    #                 if attempt == -1:
    #                     student_tc[-1].append(-1)
    #                     student_time[-1].append(
    #                         parse_time("01/01/30, 00:00:00")
    #                     )
    #                     student_qidx[-1].append(-1)
    #                 else:
    #                     if len(attempt) == 0:
    #                         student_tc[-1].append(-1)
    #                         student_time[-1].append(
    #                             parse_time("01/01/30, 00:00:00")
    #                         )
    #                         student_qidx[-1].append(-1)
    #                     elif tidx < len(attempt) and time_matrix[sidx][qidx][aidx] != "":
    #                         student_tc[-1].append(attempt[tidx])
    #                         student_time[-1].append(
    #                             parse_time(time_matrix[sidx][qidx][aidx])
    #                         )
    #                         student_qidx[-1].append(qidx)
    #                     else:
    #                         student_tc[-1].append(-1)
    #                         student_time[-1].append(
    #                             parse_time("01/01/30, 00:00:00")
    #                         )
    #                         student_qidx[-1].append(-1)
    #                         # raise ValueError("Testcase index out of bound")

    #     y_tc_obs.append(student_tc)
    #     time_tc_obs.append(student_time)
    #     qidx_tc_obs.append(student_qidx)

    # y_obs = torch.tensor(y_tc_obs, device=device)
    # qidx_obs = torch.tensor(qidx_tc_obs, device=device)
    # time_obs = torch.tensor(time_tc_obs, device=device)

    # torch.save(y_obs, "y_obs.pt")
    # torch.save(qidx_obs, "qidx_obs.pt")
    # torch.save(time_obs, "time_obs.pt")

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
    # Create multivariate normal prior distribution for each student
    # theta_priors = []
    # for sidx in range(n_student):
    #     # Compute covar matrix by using time_obs
    #     time_obs_s = time_obs[sidx][masked_idx[sidx]]
    #     time_obs_s = time_obs_s.reshape(-1, 1)
    #     kernel = ScaleKernel(RBFKernel(length_scale=1.0))
    #     covar = kernel(time_obs_s.cpu())#.to_dense()
    #     # covar += 1e-1 * torch.eye(covar.shape[0], device=device)

    #     theta_priors.append(MultivariateNormal(
    #         torch.zeros(covar.shape[0]),
    #         covariance_matrix=covar
    #     ))

    def get_theta_priors(sidx):
        time_obs_s = time_obs[sidx][masked_idx[sidx]]
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

    # Create normal prior distribution for z, where each z_i is corresponding to a question
    z_priors = []
    for qidx in range(n_question):
        z_priors.append(
            Normal(
                loc=torch.tensor(0.0, device=device),
                scale=torch.tensor(1.0, device=device),
            )
        )

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

    list_thetas = [[sample_theta_prior(tp) for tp in range(n_student)]]
    list_zs = [torch.stack([zp.sample() for zp in z_priors])]

    for epoch in tqdm(range(args.epochs), desc="Sampling"):
        previous_thetas = list_thetas[-1]
        previous_zs = list_zs[-1]

        # Sampling for z
        tmz = []
        for sidx, st_theta in enumerate(previous_thetas):
            if masked_idx[sidx].sum() == 0:
                continue
            if sidx >= num_train:
                break
            sqidx = qidx_obs[sidx][masked_idx[sidx]]
            sz = previous_zs[sqidx]
            tmz.append(st_theta - sz)
        tmz = torch.concatenate(tmz)

        ll_dist = Bernoulli(logits=tmz)
        zs_noise = torch.stack([zp.sample() for zp in z_priors])
        next_z = ess_z(
            previous_zs, zs_noise, previous_thetas, ll_dist, y_train, num_train
        )
        list_zs.append(next_z)

        # Sampling for theta
        tmz = []
        for sidx, st_theta in enumerate(previous_thetas):
            if masked_idx[sidx].sum() == 0:
                continue
            if sidx >= num_train:
                break
            sqidx = qidx_obs[sidx][masked_idx[sidx]]
            # ==================
            sz = next_z[sqidx]
            # ==================
            tmz.append(st_theta - sz)
        tmz = torch.concatenate(tmz)

        ll_dist = Bernoulli(logits=tmz)
        thetas_noise = [sample_theta_prior(tp) for tp in range(n_student)]
        next_theta = ess_theta(
            previous_thetas, thetas_noise, next_z, ll_dist, y_train, num_train
        )
        list_thetas.append(next_theta)

        if epoch > 195 and (epoch + 1) % 10 == 0:
            # Extract optimized parameters
            with open(f"results/thetas_by_iter.pkl", "wb") as f:
                pickle.dump(list_thetas, f)

            with open(f"results/zs_by_iter.pkl", "wb") as f:
                pickle.dump(list_zs, f)

    # # wandb.finish()
