"""
Continuous IRT (CIRT) model: training and visualization.

Trains a parametric IRT model with sigmoid learning curves on CodeInsights data.
Each student has a learning rate (theta0) and asymptotic ability (theta1),
each item has a difficulty (z). Correctness is modeled as Beta-distributed.

Trains on all courses jointly — shared students and items are naturally deduplicated.

Usage:
    python -m dynamic_irt.cirt
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tueplots import bundles, figsizes

# Repo root (CodeInsights/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from dynamic_irt.gpirt.utils import ensure_dir, set_seed

plt.rcParams.update(bundles.aaai2024())

# Standardized color palette (Paul Tol qualitative)
COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(device='cpu'):
    """Load all courses from code_insights_csv and build combined matrices."""
    sys.path.insert(0, os.path.join(REPO_ROOT, 'data_collection'))
    from csv2matrices import build_matrices

    from huggingface_hub import snapshot_download

    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--stair-lab--code_insights_csv/"
        "snapshots/99d53fe7c11f6302fb28b82fab5ebd77c00e5d12"
    )
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        path = cache_path
    else:
        path = snapshot_download(
            repo_id="stair-lab/code_insights_csv", repo_type="dataset"
        )

    main_data = pd.read_csv(f"{path}/main_data.csv", low_memory=False)
    question_infos = pd.read_csv(f"{path}/question_infos.csv")

    # Filter to actual submissions
    main_data = main_data[main_data["response_type"].isin(["Submit", "Prechecked"])].copy()
    main_data = main_data.dropna(subset=["pass"])
    print(f"Loaded {len(main_data)} submissions across all courses")

    # Build combined matrices (shared students/items naturally deduplicated)
    result = build_matrices(main_data, question_infos, "all", device)
    student_info, question_info_list, correctness_matrix, time_matrix, is_exam_matrix = result

    N, Q, T = correctness_matrix.shape
    print(f"\nFinal matrices:")
    print(f"  Students: {N}")
    print(f"  Items (questions x testcases): {Q}")
    print(f"  Max attempts: {T}\n")

    return correctness_matrix.float(), N, Q, T


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def negative_log_likelihood(concentration, y_obs, student_idx, question_idx, t_flat, theta0, theta1, z):
    """Compute negative log-likelihood for CIRT model.

    mean_correct = theta1[s] * sigmoid(theta0[s] * t - z[q])
    y ~ BetaProportion(mean_correct, concentration)
    """
    eps = 1e-6
    mean_correct = theta1[student_idx] * torch.sigmoid(
        theta0[student_idx] * t_flat - z[question_idx]
    )
    mean_correct = mean_correct.clamp(eps, 1 - eps)

    alpha = mean_correct * concentration
    beta = (1 - mean_correct) * concentration
    term1 = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    term2 = (alpha - 1) * torch.log(y_obs) + (beta - 1) * torch.log(1 - y_obs)
    nll = -(term1 + term2).mean()

    return nll


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_losses(saving_losses, result_dir):
    """Plot training and test loss curves."""
    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    ax.plot(saving_losses["loss"], label="Train", alpha=0.7, color=COLORS[0])
    ax.plot(saving_losses["test_loss"], label="Test", alpha=0.7, color=COLORS[1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CIRT Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(result_dir, "losses.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_param_hist(values, param_name, filename, result_dir, xlabel,
                    bins=30, pct_clip=(1, 99)):
    """Plot histogram + KDE, clipping x-axis to the given percentile range."""
    lo, hi = np.percentile(values, pct_clip)
    clipped = values[(values >= lo) & (values <= hi)]

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_half()["figure.figsize"])
    ax.hist(clipped, bins=bins, density=True, alpha=0.3, color=COLORS[0])
    sns.kdeplot(clipped, color=COLORS[0], linewidth=1.5, bw_adjust=0.5, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{param_name} Distribution")
    save_path = os.path.join(result_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def visualize(parms_dict, result_dir):
    """Generate all result visualizations."""
    theta0 = parms_dict["theta0"].cpu().detach().numpy()
    theta1 = torch.sigmoid(parms_dict["theta1"]).cpu().detach().numpy()
    z = parms_dict["z"].cpu().detach().numpy()

    plot_param_hist(theta0, r"$\theta_0$", "theta0", result_dir, r"$\theta_0$ (Learning Rate)")
    plot_param_hist(theta1, r"$\theta_1$", "theta1", result_dir, r"$\theta_1$ (Asymptotic Ability)")
    plot_param_hist(z, r"$z$", "z", result_dir, r"$z$ (Difficulty)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(args.seed)
    result_dir = os.path.join(REPO_ROOT, "results", "cirt")
    ensure_dir(result_dir)

    # Load all courses jointly
    y_obs, N, Q, T = load_all_data(device='cpu')

    # Flatten and identify valid indices
    y_obs = y_obs.reshape(-1)
    valid_mask = (y_obs != -1).numpy()
    valid_count = valid_mask.sum()
    print(f"Valid observations: {valid_count:,} / {len(y_obs):,} ({100*valid_count/len(y_obs):.1f}%)")

    t = np.linspace(1, T, T)
    valid_indices = np.where(valid_mask)[0]

    student_idx_np = valid_indices // (Q * T)
    question_idx_np = (valid_indices // T) % Q
    t_idx_np = valid_indices % T
    t_flat_np = t[t_idx_np]

    y_obs = y_obs[valid_mask]
    y_obs = y_obs * (1 - 2 * args.esp) + args.esp

    # Move to device
    y_obs = y_obs.to(device)
    t_flat = torch.from_numpy(t_flat_np).to(device).float()
    student_idx = torch.from_numpy(student_idx_np).to(device).long()
    question_idx = torch.from_numpy(question_idx_np).to(device).long()

    # Train/test split
    total_samples = y_obs.shape[0]
    randomized_idxs = torch.randperm(total_samples)
    train_size = int(total_samples * 0.8)
    training_idxs = randomized_idxs[:train_size]
    testing_idxs = randomized_idxs[train_size:]

    print(f"\nDataset statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Training samples: {train_size}")
    print(f"  Testing samples: {total_samples - train_size}")
    print(f"  Students: {N}")
    print(f"  Questions: {Q}")
    print(f"  Max attempts: {T}\n")

    y_obs_train = y_obs[training_idxs]
    t_flat_train = t_flat[training_idxs]
    student_idx_train = student_idx[training_idxs]
    question_idx_train = question_idx[training_idxs]

    y_obs_test = y_obs[testing_idxs]
    t_flat_test = t_flat[testing_idxs]
    student_idx_test = student_idx[testing_idxs]
    question_idx_test = question_idx[testing_idxs]

    # Initialize parameters
    theta0 = nn.Parameter(torch.abs(torch.randn(N, requires_grad=True, device=device)))
    theta1 = nn.Parameter(torch.sigmoid(torch.randn(N, requires_grad=True, device=device)))
    z = nn.Parameter(torch.abs(torch.randn(Q, requires_grad=True, device=device)))

    optimizer = optim.Adam([theta0, theta1, z], lr=args.lr)

    # Training loop
    saving_losses = {"loss": [], "test_loss": []}
    best_test_loss = float('inf')

    print("Starting training...\n")
    for epoch in tqdm(range(args.epochs), desc="Training CIRT"):
        optimizer.zero_grad()

        loss = negative_log_likelihood(
            args.concentration, y_obs_train, student_idx_train,
            question_idx_train, t_flat_train, theta0, theta1, z
        )
        cost = (theta1**2 * ((theta1 < 0).float() + (theta1 > 1).float())).mean()
        loss = loss + cost
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_loss = negative_log_likelihood(
                args.concentration, y_obs_test, student_idx_test,
                question_idx_test, t_flat_test, theta0, theta1, z
            )
            test_loss = test_loss + cost

        saving_losses["loss"].append(loss.item())
        saving_losses["test_loss"].append(test_loss.item())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            with open(os.path.join(result_dir, "model_best.pkl"), "wb") as f:
                pickle.dump({
                    "theta0": theta0.detach().cpu(),
                    "theta1": theta1.detach().cpu(),
                    "z": z.detach().cpu(),
                    "epoch": epoch + 1,
                    "test_loss": test_loss.item()
                }, f)

    # Save final model
    parms_dict = {
        "theta0": theta0.detach().cpu(),
        "theta1": theta1.detach().cpu(),
        "z": z.detach().cpu(),
        "N": N, "Q": Q, "T": T
    }
    with open(os.path.join(result_dir, "model.pkl"), "wb") as f:
        pickle.dump(parms_dict, f)

    # Save loss history
    pd.DataFrame(saving_losses).to_json(os.path.join(result_dir, "losses.json"), indent=4)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final train loss: {saving_losses['loss'][-1]:.4f}")
    print(f"  Final test loss: {saving_losses['test_loss'][-1]:.4f}")
    print(f"  Best test loss: {best_test_loss:.4f}")
    print(f"  Results saved to: {result_dir}/")

    # Visualize
    print(f"\nGenerating plots...")
    plot_losses(saving_losses, result_dir)
    visualize(parms_dict, result_dir)

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CIRT model on CodeInsights data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--concentration", type=float, default=10.0,
                        help="Beta distribution concentration parameter")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--esp", type=float, default=1e-2,
                        help="Epsilon for avoiding zero/one scores")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
