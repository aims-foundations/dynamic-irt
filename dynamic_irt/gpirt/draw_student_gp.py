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
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
from tueplots import bundles, constants
from utils import ensure_dir, set_seed

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
    student_points = pickle.load(
        open(
            f"results/{args.course_name}_seed{args.seed}/student_{picked_sidx}_points.pkl",
            "rb",
        )
    )
    student_xs, student_ys = pickle.load(
        open(
            f"results/{args.course_name}_seed{args.seed}/student_{picked_sidx}_xy.pkl",
            "rb",
        )
    )
    # >>> n_sample x n_time

    student_times = torch.tensor(unique_time_obs[picked_sidx]).to(device)
    student_times = student_times[list_saidx[picked_sidx]]
    student_tidx = tidx_obs[picked_sidx][tidx_obs[picked_sidx] != -1]
    student_thetas = torch.tensor(student_thetas).to(device)
    student_points = torch.stack(student_points).float()
    # >>> n_time

    sort_idx = student_tidx.argsort()
    sorted_student_times = student_times[sort_idx]
    sorted_student_thetas = [th[sort_idx] for th in student_thetas]
    student_tidx = student_tidx[sort_idx].cpu()

    min_time = student_times.min().item()
    max_time = student_times.max().item()
    time_range = torch.linspace(min_time, max_time, student_points.shape[-1])
    for sample_points in student_points:
        plt.plot(
            time_range.cpu().numpy(),
            sample_points.cpu().numpy(),
            color="gray",
            alpha=0.01,
        )

    mean_points = student_points.mean(dim=0)
    std_points = student_points.std(dim=0)
    plt.plot(time_range.cpu().numpy(), mean_points.cpu().numpy(), color="red")
    # plt.fill_between(
    #     time_range.cpu().numpy(),
    #     mean_points.cpu().numpy() - std_points.cpu().numpy(),
    #     mean_points.cpu().numpy() + std_points.cpu().numpy(),
    # )

    # Scatter student x, y
    plt.scatter(student_xs, student_ys, color="blue")
    uni_xs = np.unique(student_xs)
    uni_ys = []
    for x in uni_xs:
        uni_ys.append(student_ys[student_xs == x].mean().item())
    plt.plot(uni_xs, uni_ys, color="blue")

    # Plot week lines
    week_idx = [min(min_time + 14 * i, max_time) for i in range(7)]
    for week in week_idx:
        plt.axvline(week, color="black", linestyle="--", alpha=0.5)

    plt.xlabel("Days")
    plt.ylabel(r"$\theta$")
    plt.title(f"Student {picked_sidx}'s $\\theta$")
    plt.savefig(f"test_s{picked_sidx}_seed{args.seed}.png", dpi=300)

    # Load zs
    print("Loading zs")
    zs = torch.load(
        os.path.join(
            f"results/{args.course_name}_seed{args.seed}",
            f"ess_zs_by_iter_{args.iteration}.pt",
        )
    )

    student_zs = [z.to(device)[list_sqidx[picked_sidx]].cpu().tolist() for z in zs]
    student_zs = np.array(student_zs)

    student_zs_mean = student_zs.mean(axis=0)
    student_zs_std = student_zs.std(axis=0)

    # Scatter plot of student's z with error bars
    plt.figure()
    plt.errorbar(
        range(len(student_zs_mean)),
        student_zs_mean,
        yerr=student_zs_std,
        fmt="o",
        capsize=5,
    )
    # plt.xlabel("Testcase index")
    plt.ylabel(r"$z$")
    plt.title(f"Student {picked_sidx} z plot")
    plt.savefig(f"plots/student_{picked_sidx}_z_plot_seed{args.seed}.png", dpi=300)
    plt.close()
