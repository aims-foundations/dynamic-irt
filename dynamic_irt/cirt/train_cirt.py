"""
Modern CIRT training script that uses code_insights_csv dataset directly.

This script:
1. Loads data from stair-lab/code_insights_csv (cached)
2. Converts CSV to matrices on-the-fly
3. Trains CIRT model
4. Saves results
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from huggingface_hub import snapshot_download
from tqdm import tqdm
from tueplots import bundles

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from dynamic_irt.gpirt.utils import ensure_dir, set_seed

plt.rcParams.update(bundles.aaai2024())


def load_data_from_csv(course_name, device='cpu'):
    """Load and convert data from code_insights_csv to matrices using csv2matrices logic."""
    print(f"Loading data for {course_name}...")

    # Import csv2matrices functions
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../data_collection'))
    from csv2matrices import load_csv_data, build_matrices

    # Load CSV data
    main_data, question_infos, course_name = load_csv_data(course_name)
    print(f"Loaded {len(main_data)} submissions")

    # Build matrices
    result = build_matrices(main_data, question_infos, course_name, device)
    # Unpack result: (student_info, question_info_list, correctness_matrix, time_matrix, is_exam_matrix)
    student_info, question_info_list, correctness_matrix, time_matrix, is_exam_matrix = result

    N, Q, T = correctness_matrix.shape

    print(f"\nFinal matrices:")
    print(f"  Students: {N}")
    print(f"  Items (questions×testcases): {Q}")
    print(f"  Max attempts: {T}\n")

    return correctness_matrix.float(), N, Q, T


def negative_log_likelihood(concentration, y_obs, student_idx, question_idx, t_flat, theta0, theta1, z):
    """Compute negative log-likelihood for CIRT model."""
    mean_correct = theta1[student_idx] * torch.sigmoid(
        theta0[student_idx] * t_flat - z[question_idx]
    )

    alpha = mean_correct * concentration
    beta = (1 - mean_correct) * concentration
    term1 = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    term2 = (alpha - 1) * torch.log(y_obs) + (beta - 1) * torch.log(1 - y_obs)
    nll = -(term1 + term2).mean()

    return nll


def main():
    parser = argparse.ArgumentParser(description="Train CIRT model on CodeInsights data")
    parser.add_argument(
        "--course_name", type=str, default="dsa_hk231",
        help="Course name (e.g., dsa_hk231, dsa_hk222)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--concentration", type=float, default=10.0,
        help="Beta distribution concentration parameter"
    )
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument(
        "--esp", type=float, default=1e-2,
        help="Epsilon for avoiding zero/one scores"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="code_insights",
            config=vars(args),
            name=f"cirt_{args.course_name}_{args.concentration}"
        )

    set_seed(args.seed)
    result_dir = f"results/{args.course_name}_{args.concentration}"
    ensure_dir(result_dir)
    ensure_dir("plots")

    # Load data (load to CPU first to save memory)
    y_obs, N, Q, T = load_data_from_csv(args.course_name, device='cpu')

    # Flatten and identify valid indices on CPU
    y_obs = y_obs.reshape(-1)
    valid_mask = (y_obs != -1).numpy()
    valid_count = valid_mask.sum()

    print(f"Valid observations: {valid_count:,} / {len(y_obs):,} ({100*valid_count/len(y_obs):.1f}%)")

    # Create indices only for valid entries (much more memory efficient)
    t = np.linspace(1, T, T)
    valid_indices = np.where(valid_mask)[0]

    # Compute student, question, and time indices for valid entries
    student_idx_np = valid_indices // (Q * T)
    question_idx_np = (valid_indices // T) % Q
    t_idx_np = valid_indices % T
    t_flat_np = t[t_idx_np]

    # Filter y_obs and scale
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

    # Initialize model parameters
    theta0 = nn.Parameter(torch.abs(torch.randn(N, requires_grad=True, device=device)))
    theta1 = nn.Parameter(torch.sigmoid(torch.randn(N, requires_grad=True, device=device)))
    z = nn.Parameter(torch.abs(torch.randn(Q, requires_grad=True, device=device)))

    # Optimizer
    optimizer = optim.Adam([theta0, theta1, z], lr=0.001)

    # Training loop
    saving_losses = {"loss": [], "test_loss": []}
    best_test_loss = float('inf')

    print("Starting training...\n")
    for epoch in tqdm(range(args.epochs), desc="Training CIRT"):
        optimizer.zero_grad()

        loss = negative_log_likelihood(
            args.concentration,
            y_obs_train,
            student_idx_train,
            question_idx_train,
            t_flat_train,
            theta0, theta1, z
        )

        # Regularization: penalize theta1 outside [0,1]
        cost = (theta1**2 * ((theta1 < 0).float() + (theta1 > 1).float())).mean()
        loss = loss + cost

        loss.backward()
        optimizer.step()

        # Evaluate on test set
        with torch.no_grad():
            test_loss = negative_log_likelihood(
                args.concentration,
                y_obs_test,
                student_idx_test,
                question_idx_test,
                t_flat_test,
                theta0, theta1, z
            )
            test_loss = test_loss + cost

        saving_losses["loss"].append(loss.item())
        saving_losses["test_loss"].append(test_loss.item())

        if not args.no_wandb:
            wandb.log({"loss": loss.item(), "test_loss": test_loss.item()})

        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

        # Save best model
        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            with open(f"{result_dir}/model_best.pkl", "wb") as f:
                pickle.dump(
                    {
                        "theta0": theta0.detach().cpu(),
                        "theta1": theta1.detach().cpu(),
                        "z": z.detach().cpu(),
                        "epoch": epoch + 1,
                        "test_loss": test_loss.item()
                    },
                    f
                )

    # Save final model
    with open(f"{result_dir}/model.pkl", "wb") as f:
        pickle.dump(
            {
                "theta0": theta0.detach().cpu(),
                "theta1": theta1.detach().cpu(),
                "z": z.detach().cpu(),
                "N": N,
                "Q": Q,
                "T": T
            },
            f
        )

    # Save training history
    df = pd.DataFrame(saving_losses)
    df.to_json(f"{result_dir}/losses.json", indent=4)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(saving_losses["loss"], label="Train Loss", alpha=0.7)
    plt.plot(saving_losses["test_loss"], label="Test Loss", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"CIRT Training: {args.course_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/cirt_{args.course_name}_losses.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final train loss: {saving_losses['loss'][-1]:.4f}")
    print(f"  Final test loss: {saving_losses['test_loss'][-1]:.4f}")
    print(f"  Best test loss: {best_test_loss:.4f}")
    print(f"  Results saved to: {result_dir}/")
    print(f"{'='*60}\n")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
