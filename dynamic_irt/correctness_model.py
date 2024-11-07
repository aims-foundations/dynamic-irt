import json

import arviz as az
import jax.random as random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import torch
import torch.nn as nn
import torch.optim as optim
from jax.scipy.special import expit as jexpit
from numpyro.infer import MCMC, NUTS
from scipy.special import expit, gammaln

matplotlib.rcParams["text.usetex"] = True


def generate_parameters(N, Q):
    theta0 = np.abs(np.random.normal(loc=0, scale=1, size=N))
    theta1 = np.random.beta(a=1, b=1, size=N)
    z = np.abs(np.random.normal(loc=0, scale=1, size=Q))
    return theta0, theta1, z


def oracle(theta0, theta1, z, N, Q, T):
    t = np.linspace(1, T, T)
    mean_correctness = np.zeros((N, Q, T))
    for i in range(N):
        for j in range(Q):
            for k in range(T):
                mean_correctness[i, j, k] = theta1[i] * expit(theta0[i] * t[k] - z[j])

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    return numpyro.sample(
        "obs",
        dist.BetaProportion(mean_correctness, 200),
        rng_key=rng_key,
        sample_shape=(1,),
    ).squeeze()


def plot_correctness(correctness, N):
    T = correctness.shape[2]
    t = np.linspace(1, T, T)

    plt.figure(figsize=(5, 5))
    for i in range(N):
        student_avg_correctness = np.mean(correctness[i], axis=0)
        plt.plot(t, student_avg_correctness, alpha=0.75, linewidth=0.5)

    plt.xlabel("Attempts")
    plt.ylabel("Average Correctness")
    plt.savefig("plots/avg_correctness.png", dpi=300)
    plt.close()


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


if __name__ == "__main__":
    N = 200  # 781
    Q = 100  # 179
    T = 50
    seed = 42
    np.random.seed(seed)
    inference = "mle"
    device = "cuda"

    theta0_gt, theta1_gt, z_gt = generate_parameters(N, Q)
    correctness = oracle(theta0_gt, theta1_gt, z_gt, N, Q, T)
    plot_correctness(correctness, N)

    y_obs = correctness.reshape(-1)
    t = np.linspace(1, T, T)
    t_flat = np.tile(t, len(y_obs) // T)
    student_idx = np.repeat(np.arange(N), Q * T)
    question_idx = np.tile(np.repeat(np.arange(Q), T), N)

    if inference == "mle":
        # Convert data to PyTorch tensors
        y_obs_torch = torch.from_numpy(np.array(y_obs)).to(device=device)
        student_idx_torch = (
            torch.from_numpy(np.array(student_idx)).to(device=device).long()
        )
        question_idx_torch = torch.from_numpy(np.array(question_idx)).to(device=device)
        t_flat_torch = torch.from_numpy(np.array(t_flat)).to(device=device)

        # Define model parameters to optimize
        theta0 = nn.Parameter(
            torch.abs(torch.randn(N, requires_grad=True, device=device))
        )
        theta1 = nn.Parameter(
            torch.sigmoid(torch.randn(N, requires_grad=True, device=device))
        )
        z = nn.Parameter(torch.abs(torch.randn(Q, requires_grad=True, device=device)))

        # Set up optimizer
        optimizer = optim.Adam([theta0, theta1, z], lr=0.0001)

        # Negative log-likelihood function
        def negative_log_likelihood(y_obs, student_idx, question_idx, t_flat):
            mean_correct = theta1[student_idx] * torch.sigmoid(
                theta0[student_idx] * t_flat - z[question_idx]
            )
            alpha = mean_correct * 200
            beta = (1 - mean_correct) * 200
            # nll = - torch.distributions.beta.Beta(alpha, beta).log_prob(y_obs).mean()
            term1 = (
                torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
            )
            term2 = (alpha - 1) * torch.log(y_obs) + (beta - 1) * torch.log(1 - y_obs)
            nll = -(term1 + term2).mean()

            return nll

        # Training loop
        num_epochs = 10000
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = negative_log_likelihood(
                y_obs_torch, student_idx_torch, question_idx_torch, t_flat_torch
            )
            cost = (theta1**2 * ((theta1 < 0).float() + (theta1 > 1).float())).mean()
            loss = loss + cost
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Extract optimized parameters
        theta0_optimized = theta0.cpu().detach().numpy()
        theta1_optimized = theta1.cpu().detach().numpy()
        z_optimized = z.cpu().detach().numpy()

        plot_correlation(
            theta0_gt,
            theta0_optimized,
            r"$\theta_0$",
            r"Predicted $\theta_0$",
            r"Correlation between $\theta_0$ and predicted $\theta_0$",
            "theta0_syn.png",
        )
        plot_correlation(
            theta1_gt,
            theta1_optimized,
            r"$\theta_1$",
            r"Predicted $\theta_1$",
            r"Correlation between $\theta_1$ and predicted $\theta_1$",
            "theta1_syn.png",
        )
        plot_correlation(
            z_gt,
            z_optimized,
            "$z$",
            "Predicted $z$",
            "Correlation between $z$ and predicted $z$",
            "z_syn.png",
        )
