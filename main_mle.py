import argparse
import json
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from huggingface_hub import snapshot_download
from tqdm import tqdm
from utils import set_seed

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


# Negative log-likelihood function
def negative_log_likelihood(concentration, y_obs, student_idx, question_idx, t_flat):
    mean_correct = theta1[student_idx] * torch.sigmoid(
        theta0[student_idx] * t_flat - z[question_idx]
    )

    alpha = mean_correct * concentration
    beta = (1 - mean_correct) * concentration
    term1 = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    term2 = (alpha - 1) * torch.log(y_obs) + (beta - 1) * torch.log(1 - y_obs)
    nll = -(term1 + term2).mean()

    return nll


if __name__ == "__main__":
    wandb.init(project="code_insights")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument(
        "--concentration",
        help="Concentration hyperparameter",
        type=float,
        default=200.0,
    )
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10000)
    parser.add_argument(
        "--esp", help="Epsilon for avoiding zero score", type=float, default=1e-2
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}", repo_type="dataset"
    )

    y_obs = pickle.load(open(f"{data_folder}/correctness_matrix.pkl", "rb"))
    y_obs = torch.tensor(y_obs, device=device)

    student_info = pickle.load(open(f"{data_folder}/student_ids.pkl", "rb"))

    N, Q, T = y_obs.shape
    y_obs = y_obs.reshape(-1)
    t = np.linspace(1, T, T)
    t_flat = torch.from_numpy(np.tile(t, len(y_obs) // T)).to(device=device)
    student_idx = torch.from_numpy(np.repeat(np.arange(N), Q * T)).to(device=device)
    question_idx = torch.from_numpy(np.tile(np.repeat(np.arange(Q), T), N)).to(
        device=device
    )

    # Remove samples that have negative scores
    valid_idxs = y_obs != -1
    y_obs = y_obs[valid_idxs]

    # Scale scores
    y_obs = y_obs * (1 - 2 * args.esp) + args.esp

    t_flat = t_flat[valid_idxs]
    student_idx = student_idx[valid_idxs]
    question_idx = question_idx[valid_idxs]

    total_sample = y_obs.shape[0]
    randomized_idxs = torch.randperm(total_sample)
    training_idxs = randomized_idxs[: int(total_sample * 0.8)]
    testing_idxs = randomized_idxs[int(total_sample * 0.8) :]
    print("Number of training samples:", int(total_sample * 0.8))
    print("Number of testing samples:", int(total_sample * 0.2))

    y_obs_train = y_obs[training_idxs]
    t_flat_train = t_flat[training_idxs]
    student_idx_train = student_idx[training_idxs]
    question_idx_train = question_idx[training_idxs]

    y_obs_test = y_obs[testing_idxs]
    t_flat_test = t_flat[testing_idxs]
    student_idx_test = student_idx[testing_idxs]
    question_idx_test = question_idx[testing_idxs]

    # Define model parameters to optimize
    theta0 = nn.Parameter(torch.abs(torch.randn(N, requires_grad=True, device=device)))
    theta1 = nn.Parameter(
        torch.sigmoid(torch.randn(N, requires_grad=True, device=device))
    )
    z = nn.Parameter(torch.abs(torch.randn(Q, requires_grad=True, device=device)))

    # Set up optimizer
    optimizer = optim.Adam([theta0, theta1, z], lr=0.001)

    # Training loop
    for epoch in tqdm(range(args.epochs), desc="Fitting model"):
        optimizer.zero_grad()
        loss = negative_log_likelihood(
            args.concentration,
            y_obs_train,
            student_idx_train,
            question_idx_train,
            t_flat_train,
        )
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item():.4f}")

    # Extract optimized parameters
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.course_name}_{args.concentration}.pkl", "wb") as f:
        pickle.dump(
            {
                "theta0": theta0,
                "theta1": theta1,
                "z": z,
            },
            f,
        )

    wandb.finish()
