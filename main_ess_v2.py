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

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    current_f, nu, previous_thetas, ll_dist, y_train, num_train, train_test_split_idx
):
    ll_current = ll_dist.log_prob(y_train).sum()
    ll_thres = ll_current + torch.log(torch.rand(1, device=device))
    print("Z log-likelihood", ll_current.item())

    angle = torch.rand(1, device=device) * 2 * np.pi
    angle_min, angle_max = angle - 2 * np.pi, angle

    while True:
        next_f = torch.cos(angle) * current_f + torch.sin(angle) * nu
        tmz = previous_thetas[:train_test_split_idx] - next_f[:train_test_split_idx]
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
            angle = torch.rand(1, device=device) * (angle_max - angle_min) + angle_min

    return next_f


# Eliptical Slice Sampling
def ess_theta(
    previous_thetas, nu, next_z, ll_dist, y_train, num_train, train_test_split_idx
):
    ll_current = ll_dist.log_prob(y_train).sum()
    ll_thres = ll_current + torch.log(torch.rand(1, device=device))
    print("Theta log-likelihood", ll_current.item())

    angle = torch.rand(1, device=device) * 2 * np.pi
    angle_min, angle_max = angle - 2 * np.pi, angle

    while True:
        next_thetas = torch.cos(angle) * previous_thetas + torch.sin(angle) * nu
        tmz = next_thetas[:train_test_split_idx] - next_z[:train_test_split_idx]

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
            angle = torch.rand(1, device=device) * (angle_max - angle_min) + angle_min

    return next_thetas


if __name__ == "__main__":
    # wandb.init(project="code_insights")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=100000)
    parser.add_argument("--is_continue", help="Continue sampling", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(f"results/{args.course_name}_seed{args.seed}")
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
    print("Computing maximum number of testcases for each question")
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

    print("Loading data")
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
    print("Creating theta priors")
    unique_time_obs = []
    aidx_obs = []  # student attempt index
    theta_priors = []
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

    for sidx in range(n_student):
        if masked_idx[sidx].sum() == 0:
            theta_priors.append(None)
            continue

        theta_priors.append(get_theta_priors(sidx))

    def sample_theta_prior(sidx):
        if theta_priors[sidx] is None:
            return []
        else:
            return theta_priors[sidx].sample()

    # Create normal prior distribution for z, where each z_i is corresponding to a testcase in a question
    print("Creating z priors")
    z_priors = []
    for qidx in range(total_testcases):
        z_priors.append(
            Normal(
                loc=torch.tensor(0.0, device=device),
                scale=torch.tensor(1.0, device=device),
            )
        )
    total_z = len(z_priors)

    # covar = torch.ones((total_testcases,total_testcases), device=device)*1e-10
    # covar = covar.fill_diagonal_(1)
    # z_priors = MultivariateNormal(
    #     torch.zeros(total_testcases, device=device),
    #     covariance_matrix=covar
    # )

    # Create y_train
    print("Splitting train and test data")
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

    list_saidx = []
    list_sqidx = []
    train_test_split_idx = 0
    student_idxs = []
    for sidx in range(n_student):
        if masked_idx[sidx].sum() == 0:
            list_saidx.append(None)
            # list_sqidx.append(None)
            continue

        saidx = aidx_obs[sidx][masked_idx[sidx]]  # attemp index for student
        list_saidx.append(saidx)

        sqidx = qidx_obs[sidx][masked_idx[sidx]]  # global testcase index
        list_sqidx.append(sqidx)

        student_idxs.extend([sidx] * saidx.shape[0])

        if sidx < num_train:
            train_test_split_idx += saidx.shape[0]

    student_idxs = torch.tensor(student_idxs)
    all_squidx = torch.cat(list_sqidx)
    # Save student indexes
    with open(
        f"results/{args.course_name}_seed{args.seed}/student_idxs.pkl", "wb"
    ) as f:
        pickle.dump(student_idxs, f)

    print("Sampling priors")
    if args.is_continue:
        list_thetas = pickle.load(
            open(
                f"results/{args.course_name}_seed{args.seed}/thetas_by_iter_{args.is_continue}.pkl",
                "rb",
            )
        )
        list_zs = pickle.load(
            open(
                f"results/{args.course_name}_seed{args.seed}/zs_by_iter_{args.is_continue}.pkl",
                "rb",
            )
        )

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

    num_theta_per_student = []
    previous_thetas = []
    for sidx, st_theta in enumerate(list_thetas[-1]):
        if list_saidx[sidx] is None:
            num_theta_per_student.append(0)
            continue

        num_theta_per_student.append(st_theta.shape[0])
        previous_thetas.append(st_theta[list_saidx[sidx]])
    previous_thetas = torch.cat(previous_thetas)
    previous_zs = list_zs[-1][all_squidx]

    for epoch in tqdm(range(continue_iter, args.epochs), desc="Sampling"):
        # Sampling for z
        tmz = (
            previous_thetas[:train_test_split_idx] - previous_zs[:train_test_split_idx]
        )
        ll_dist = Bernoulli(logits=tmz)

        zs_noise = torch.stack([z.sample() for z in z_priors])[all_squidx]
        # zs_noise = z_priors.sample()

        next_z = ess_z(
            previous_zs,
            zs_noise,
            previous_thetas,
            ll_dist,
            y_train,
            num_train,
            train_test_split_idx,
        )

        # Construct zs
        constructed_z = torch.zeros((total_z,), device=device)
        constructed_z[all_squidx] = next_z
        list_zs.append(constructed_z.to(torch.bfloat16).cpu())
        # list_zs.append(next_z.cpu())

        # Sampling for theta
        tmz = previous_thetas[:train_test_split_idx] - next_z[:train_test_split_idx]
        ll_dist = Bernoulli(logits=tmz)

        thetas_noise = []
        for sidx, st_theta in enumerate(
            [sample_theta_prior(tp) for tp in range(n_student)]
        ):
            if list_saidx[sidx] is None:
                continue
            thetas_noise.append(st_theta[list_saidx[sidx]])
        thetas_noise = torch.cat(thetas_noise)

        next_theta = ess_theta(
            previous_thetas,
            thetas_noise,
            next_z,
            ll_dist,
            y_train,
            num_train,
            train_test_split_idx,
        )

        # Construct thetas
        # constructed_theta = []
        # for sidx in range(n_student):
        #     if num_theta_per_student[sidx] == 0:
        #         constructed_theta.append([])
        #         continue
        #     st_theta = torch.zeros((num_theta_per_student[sidx],), device=device)
        #     st_theta[list_saidx[sidx]] = next_theta[student_idxs == sidx]
        #     constructed_theta.append(st_theta.cpu())
        # list_thetas.append(constructed_theta)
        list_thetas.append(next_theta.to(torch.bfloat16).cpu())

        # Update thetas and zs
        previous_thetas = next_theta
        previous_zs = next_z

        if (epoch + 1) % 5000 == 0:
            # Save thetas and zs with torch.save
            torch.save(
                list_thetas,
                f"results/{args.course_name}_seed{args.seed}/ess_thetas_by_iter_{epoch + 1}.pt",
            )
            torch.save(
                list_zs,
                f"results/{args.course_name}_seed{args.seed}/ess_zs_by_iter_{epoch + 1}.pt",
            )
            
            # Clear thetas and zs
            list_thetas = []
            list_zs = []
            
            # Delete old saved files
            # if epoch + 1 > 5000:
            #     os.remove(f"results/{args.course_name}_seed{args.seed}/ess_thetas_by_iter_{epoch + 1 - 5000}.pt")
            #     os.remove(f"results/{args.course_name}_seed{args.seed}/ess_zs_by_iter_{epoch + 1 - 5000}.pt")

    # wandb.finish()
