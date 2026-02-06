import argparse
import os

import pickle
import random

import matplotlib.pyplot as plt
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from tqdm import tqdm
from tueplots import bundles, constants

plt.rcParams.update(bundles.iclr2024())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_dist(samples):
    # Sample having shape (N, D)
    # N: Number of samples
    # D: Dimension of the samples

    # Calculate the mean and covariance matrix of the samples
    mean = torch.mean(samples, dim=0)
    kernel = ScaleKernel(RBFKernel(length_scale=1.0)).to(device)
    covariance = kernel(samples.T)

    # Reconstruct the multivariate normal distribution
    return MultivariateNormal(mean, covariance_matrix=covariance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 45])
    parser.add_argument("--n_theta", type=int, default=10)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--start_iter", type=int, default=10000)
    parser.add_argument("--end_iter", type=int, default=100000)
    parser.add_argument("--step", type=int, default=5000)
    args = parser.parse_args()

    if len(args.seeds) != 2:
        raise ValueError("Please provide two seeds")

    first_folder = f"results/{args.course_name}_seed{args.seeds[0]}"
    second_folder = f"results/{args.course_name}_seed{args.seeds[1]}"

    theta_1 = []
    theta_2 = []
    for iter in tqdm(range(args.start_iter, args.end_iter + 1, args.step)):
        theta_1.extend(
            torch.load(os.path.join(first_folder, f"ess_thetas_by_iter_{iter}.pt"))
        )
        theta_2.extend(
            torch.load(os.path.join(second_folder, f"ess_thetas_by_iter_{iter}.pt"))
        )

    if args.draw:
        # Randomly select n_theta thetas from total_thetas without replacement
        total_thetas = len(theta_1[0])
        # selected_thetas = random.sample(range(total_thetas), args.n_theta)
        selected_thetas = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

        for thidx in tqdm(selected_thetas, desc="Drawing theta distributions"):
            list_theta_1 = [theta_1[i][thidx].float() for i in range(0, len(theta_1))]
            list_theta_2 = [theta_2[i][thidx].float() for i in range(0, len(theta_2))]

            # Draw theta_1 and theta_2 distributions on the same plot with different colors
            plt.hist(list_theta_1, bins=50, alpha=0.5, color="blue")
            plt.hist(list_theta_2, bins=50, alpha=0.5, color="red")
            plt.xlabel(r"$\theta$")
            plt.ylabel("Frequency")
            plt.title(r"$\theta$" + f" {thidx} Distribution")
            plt.savefig(f"plots/theta{thidx}.png", dpi=300)
            plt.close()

    student_idxs = pickle.load(
        open(os.path.join(first_folder, "student_idxs.pkl"), "rb")
    )
    list_saidx2idx = pickle.load(
        open(os.path.join(first_folder, "list_saidx2aidx.pkl"), "rb")
    )
    n_students = len(student_idxs.unique())

    list_kl_div = []
    for sidx in tqdm(student_idxs.unique(), desc="Computing KL Divergence"):
        student_idx = student_idxs == sidx
        if student_idx.sum() == 0:
            continue

        list_theta_1 = [th[student_idx][list_saidx2idx[sidx]] for th in theta_1]
        list_theta_2 = [th[student_idx][list_saidx2idx[sidx]] for th in theta_2]

        dist_theta_1 = construct_dist(torch.stack(list_theta_1).to(device).float())
        dist_theta_2 = construct_dist(torch.stack(list_theta_2).to(device).float())

        try:
            kl_div = torch.distributions.kl.kl_divergence(dist_theta_1, dist_theta_2)
            list_kl_div.append(kl_div.item())

            print(f"Student {sidx} KL Divergence: {kl_div.item()}")
        except:
            print(f"Student {sidx} KL Divergence: NaN")

        # Print mean KL Divergence
        print(f"Mean KL Divergence: {torch.mean(torch.tensor(list_kl_div)).item()}")

    # Save the list of KL Divergence
    pickle.dump(list_kl_div, open("results/kl_div.pkl", "wb"))

    plt.hist(list_kl_div, bins=50, alpha=0.5, color="blue")
    plt.axvline(x=torch.mean(torch.tensor(list_kl_div)), color="red", linestyle="--")
    plt.xlabel("KL Divergence")
    plt.ylabel("Frequency")
    plt.title("KL Divergence Distribution")
    plt.savefig(f"plots/kl_div.png", dpi=300)
