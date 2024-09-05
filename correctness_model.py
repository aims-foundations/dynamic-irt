import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.random as random
from jax.scipy.special import expit as jexpit
import json
import arviz as az
from scipy.special import expit

def generate_parameters(N, Q):
    theta0 = np.random.normal(0, 1, N)
    theta1 = np.abs(np.random.normal(loc=0, scale=1, size=N))
    theta2 = np.random.beta(a=1, b=1, size=N)
    z = np.abs(np.random.normal(loc=0, scale=1, size=Q))
    return theta0, theta1, theta2, z

def oracle(theta0, theta1, theta2, z, N, Q, T):
    t = np.linspace(1, T, T)
    mean_correctness = np.zeros((N, Q, T))
    for i in range(N):
        for j in range(Q):
            for k in range(T):
                mean_correctness[i, j, k] = theta2[i]*expit(
                    theta0[i] + theta1[i] * t[k] - z[j]
                )

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    return numpyro.sample(
        "obs", dist.BetaProportion(mean_correctness, 200),
        rng_key=rng_key, sample_shape=(1,)
    ).squeeze()

def model(student_idx, question_idx, t, y_obs):
    theta0 = numpyro.sample("theta0", dist.Normal(0, 1).expand([N]))
    theta1 = numpyro.sample(
        "theta1",
        dist.TruncatedNormal(low=0.0, loc=0.0, scale=1.0).expand([N])
    )
    theta2 = numpyro.sample("theta2", dist.Beta(1, 1).expand([N]))
    z = numpyro.sample("z", dist.TruncatedNormal(low=-3, high=3, loc=0, scale=1).expand([Q]))

    mean_correct = theta2[student_idx]*jexpit(
        theta0[student_idx] + theta1[student_idx]*t - z[question_idx]
    )
    with numpyro.plate("data", len(y_obs)):
        numpyro.sample(
            "obs", dist.BetaProportion(mean_correct, 200), obs=y_obs
        )

def plot(correctness, N):
    T = correctness.shape[2]
    t = np.linspace(1, T, T)

    plt.figure(figsize=(5, 5))
    for i in range(N):
        student_avg_correctness = np.mean(correctness[i], axis=0)
        plt.plot(t, student_avg_correctness)

    plt.xlabel('Attempts')
    plt.ylabel('Average Correctness')
    plt.show()

if __name__ == "__main__":
    N = 70 # 781
    Q = 60 # 2647
    T = 60
    num_warmup = 200
    num_samples = 200
    seed = 42
    np.random.seed(42)
    inference = "mle"
    device = "cuda"

    theta0, theta1, theta2, z = generate_parameters(N, Q)
    print(f"Grountruth theta0: {theta0}")
    print(f"Grountruth theta1: {theta1}")
    print(f"Grountruth theta2: {theta2}")
    print(f"Grountruth z: {z}")
    correctness = oracle(theta0, theta1, theta2, z, N, Q, T)
    plot(correctness, N)

    y_obs = correctness.reshape(-1)
    t = np.linspace(1, T, T)
    t_flat = np.tile(t, len(y_obs) // T)
    student_idx = np.repeat(np.arange(N), Q * T)
    repeats_per_question = len(y_obs) // Q
    question_idx = np.repeat(np.arange(Q), repeats_per_question)

    if inference == "mle":
        # Convert data to PyTorch tensors
        y_obs_torch = torch.from_numpy(np.array(y_obs)).to(device=device)
        student_idx_torch = torch.from_numpy(np.array(y_obs)).to(device=device).long()
        question_idx_torch = torch.from_numpy(np.array(question_idx)).to(device=device)
        t_flat_torch = torch.from_numpy(np.array(t_flat)).to(device=device)

        # Define model parameters to optimize
        theta0 = nn.Parameter(torch.randn(N, requires_grad=True, device=device))
        theta1 = nn.Parameter(torch.abs(torch.randn(N, requires_grad=True, device=device)))
        theta2 = nn.Parameter(torch.sigmoid(torch.randn(N, requires_grad=True, device=device)))
        z = nn.Parameter(torch.abs(torch.randn(Q, requires_grad=True, device=device)))

        # Set up optimizer
        optimizer = optim.Adam([theta0, theta1, theta2, z], lr=0.001)

        # Negative log-likelihood function
        def negative_log_likelihood(y_obs, student_idx, question_idx, t_flat):
            mean_correct = theta2[student_idx] * torch.sigmoid(
                theta0[student_idx] + theta1[student_idx] * t_flat - z[question_idx]
            )
            alpha = mean_correct * 200
            beta = (1 - mean_correct) * 200
            nll = -torch.mean((alpha - 1) * torch.log(y_obs) + (beta - 1) * torch.log(1 - y_obs))
            return nll

        # Training loop
        num_epochs = 1000
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = negative_log_likelihood(y_obs_torch, student_idx_torch, question_idx_torch, t_flat_torch)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Extract optimized parameters
        theta0_optimized = theta0.cpu().detach().numpy()
        theta1_optimized = theta1.cpu().detach().numpy()
        theta2_optimized = theta2.cpu().detach().numpy()
        z_optimized = z.cpu().detach().numpy()

        print(f"Optimized theta0: {theta0_optimized}")
        print(f"Optimized theta1: {theta1_optimized}")
        print(f"Optimized theta2: {theta2_optimized}")
        print(f"Optimized z: {z_optimized}")

    elif inference == "mcmc":
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(random.PRNGKey(seed), student_idx, question_idx, t_flat, y_obs)

        posterior_samples = mcmc.get_samples()
        posterior_samples = {
            k: v.mean().item() for k, v in posterior_samples.items()
        }
        data = {
            "posterior_samples": posterior_samples,
            "theta0": theta0.tolist(),
            "theta1": theta1.tolist(),
            "z": z.tolist()
        }
        with open('synthetic_data.json', 'w') as f:
            json.dump(data, f, indent=4)

        # MCMC diagnostic
        data_pred = az.from_numpyro(mcmc)
        az.plot_trace(data_pred, compact=True, figsize=(15, 25))

        # Performance evaluation
        for key, value in {"theta0": theta0, "theta1": theta1, "z": z}.items():
            mse = np.mean((posterior_samples[key].mean(axis=0) - value) ** 2)
            print(f"MSE for {key}: {mse}")

        theta0_samples = posterior_samples['theta0']
        theta1_samples = posterior_samples['theta1']
        z_samples = posterior_samples['z']

        theta0_mean = np.mean(theta0_samples, axis=0)
        theta1_mean = np.mean(theta1_samples, axis=0)
        z_mean = np.mean(z_samples, axis=0)

        theta0_mse = np.mean((theta0_mean - theta0) ** 2)
        theta1_mse = np.mean((theta1_mean - theta1) ** 2)
        z_mse = np.mean((z_mean - z) ** 2)

        print(f"MSE for theta0: {theta0_mse}")
        print(f"MSE for theta1: {theta1_mse}")
        print(f"MSE for z: {z_mse}")