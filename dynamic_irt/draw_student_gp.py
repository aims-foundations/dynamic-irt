import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from huggingface_hub import snapshot_download
from torchmetrics.regression import SpearmanCorrCoef
from tqdm import tqdm
from tueplots import bundles, constants
from dynamic_irt.gpirt.utils import ensure_dir, set_seed

plt.rcParams.update(bundles.iclr2024())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_theta_priors(time_obs_s):
    # Create a GP model
    gp = SingleTaskGP(
        time_obs_s,
        torch.zeros_like(time_obs_s),
        covar_module=ScaleKernel(RBFKernel()),
        mean_module=ConstantMean(),
    )

    # Fit the model (even though we are not using any data, this is required to initialize the model properly)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    return gp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--iteration", help="# Iteration", type=int, default=100000)
    parser.add_argument("--sidx", help="Student index", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    #################################################################################

    time_obs = torch.load("data/time_obs.pt")
    # qidx_obs = torch.load("data/qidx_obs.pt")
    tidx_obs = torch.load("data/tidx_obs.pt")
    unique_time_obs = []
    for tidx, time_ob in enumerate(time_obs):
        unique_time_obs.append(time_ob.unique()[:-1].cpu().numpy())
    picked_sidx = args.sidx

    list_saidx = pickle.load(
        open(f"results/{args.course_name}_seed{args.seed}/list_saidx.pkl", "rb")
    )
    student_thetas = pickle.load(
        open(
            f"results/{args.course_name}_seed{args.seed}/student_{picked_sidx}_thetas.pkl",
            "rb",
        )
    )
    # >>> n_sample x n_time

    student_times = torch.tensor(unique_time_obs[picked_sidx]).to(device)
    student_times = student_times[list_saidx[picked_sidx]]
    student_tidx = tidx_obs[picked_sidx][tidx_obs[picked_sidx] != -1]
    student_thetas = torch.tensor(student_thetas).to(device)
    # >>> n_time

    sort_idx = student_tidx.argsort()
    sorted_student_times = student_times[sort_idx]
    sorted_student_thetas = [th[sort_idx] for th in student_thetas]
    student_tidx = student_tidx[sort_idx].cpu()

    # for sample in sorted_student_thetas:
    #     plt.plot(torch.arange(len(sample)),
    #              sample.cpu().numpy())

    # for tidx in student_tidx.unique():
    #     tmask = student_tidx == tidx
    #     st_theta = [th[tmask].tolist() for th in sorted_student_thetas]
    #     listX = torch.arange(len(st_theta[0]))
    #     for st in st_theta:
    #         plt.plot(listX, st)

    # plt.xlabel("Testcase unique index")
    # plt.ylabel(r"$\theta$")
    # plt.title(f"Student {picked_sidx}'s $\\theta$")
    # plt.savefig("test.png", dpi=300)

    X = torch.tensor(student_times)
    X = X[None].expand(len(student_thetas), -1).flatten()
    # >>> n_sample x n_time

    Y = torch.tensor(student_thetas).flatten()
    # >>> n_sample x n_time

    X = X[-5000:]
    Y = Y[-5000:]

    # breakpoint()
    # Create a GP model
    gp = SingleTaskGP(
        X.to(device).reshape(-1, 1),
        Y.to(device).reshape(-1, 1),
        covar_module=ScaleKernel(RBFKernel()),
        mean_module=ConstantMean(),
    )

    # Fit the model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    min_time = X.min().item()
    max_time = X.max().item()

    time_range = torch.linspace(min_time, max_time, 1000).to(device)
    # >>> n_time

    week_idx = [min(min_time + 14 * i, max_time) for i in range(7)]

    with torch.no_grad():
        gp.eval()
        pred = gp(time_range[:, None])
        # >>> n_time x 1

    mean_pred = pred.mean.cpu().numpy()
    std_pred = pred.stddev.cpu().numpy()

    # Draw the student's theta plot
    plt.figure()
    plt.plot(time_range.cpu().numpy(), mean_pred)
    plt.fill_between(
        time_range.cpu().numpy(),
        mean_pred - std_pred,
        mean_pred + std_pred,
        alpha=0.3,
    )

    # Draw horizontal lines
    for week in week_idx:
        plt.axvline(week, color="black", linestyle="--", alpha=0.5)
    plt.ylabel(r"$\theta$")
    plt.title(f"Student {picked_sidx}'s $\\theta$")
    plt.savefig(f"plots/student_{picked_sidx}_theta_plot_seed{args.seed}.png", dpi=300)
    plt.close()

    theta_prior = get_theta_priors(torch.tensor(student_times).reshape(-1, 1)).to(
        device
    )
    with torch.no_grad():
        theta_prior.eval()
        pred = theta_prior(time_range[:, None])
        # >>> n_time x 1

    mean_pred = pred.mean.cpu().numpy()
    std_pred = pred.stddev.cpu().numpy()

    # Draw the student's theta plot
    plt.figure()
    plt.plot(time_range.cpu().numpy(), mean_pred)
    plt.fill_between(
        time_range.cpu().numpy(),
        mean_pred - std_pred,
        mean_pred + std_pred,
        alpha=0.3,
    )

    # Draw horizontal lines
    for week in week_idx:
        plt.axvline(week, color="black", linestyle="--", alpha=0.5)

    plt.ylabel(r"$\theta$")
    plt.title(f"Student {picked_sidx}'s $\\theta$ prior")
    plt.savefig(
        f"plots/student_{picked_sidx}_theta_prior_plot_seed{args.seed}.png", dpi=300
    )
    plt.close()
