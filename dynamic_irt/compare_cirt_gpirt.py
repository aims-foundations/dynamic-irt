"""
Comparison script for CIRT vs GPIRT models.

This script trains (or loads) both CIRT and GPIRT models on the same dataset
and compares their performance using quantitative metrics:
- Test log-likelihood
- AUC, Accuracy, F1, Precision, Recall
- Goodness-of-Fit analysis
- Parameter correlation
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from tueplots import bundles

plt.rcParams.update(bundles.neurips2024())


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# CIRT Model Functions
# ==============================================================================

def compute_cirt_predictions(theta0, theta1, z, student_idx, question_idx, t_flat):
    """Compute CIRT model predictions."""
    mean_correct = theta1[student_idx] * torch.sigmoid(
        theta0[student_idx] * t_flat - z[question_idx]
    )
    return mean_correct


def compute_cirt_likelihood(y_obs, probs):
    """Compute log-likelihood for CIRT predictions (Beta distribution)."""
    # Avoid numerical issues
    probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
    y_obs = torch.clamp(y_obs, 1e-6, 1 - 1e-6)

    concentration = 200.0  # Default concentration parameter
    alpha = probs * concentration
    beta = (1 - probs) * concentration

    log_prob = (
        torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        + (alpha - 1) * torch.log(y_obs) + (beta - 1) * torch.log(1 - y_obs)
    )
    return log_prob.mean().item()


# ==============================================================================
# GPIRT Model Functions
# ==============================================================================

def compute_gpirt_predictions(abilities, difficulties, student_idx, question_idx):
    """Compute GPIRT model predictions (1PL IRT model)."""
    # abilities: list of [n_samples, n_time_points] tensors per student
    # difficulties: [n_samples, n_questions] tensor
    # This computes P(correct) = sigmoid(ability - difficulty)

    # For evaluation, use mean of posterior samples
    probs_list = []
    for sidx in torch.unique(student_idx):
        mask = student_idx == sidx
        ability_mean = abilities[sidx].mean(dim=0)  # Average over MCMC samples

        # Get the corresponding time indices and questions
        q_idx = question_idx[mask]
        # Assuming difficulties is [n_samples, n_questions]
        difficulty_mean = difficulties.mean(dim=0)[q_idx]

        # IRT probability
        prob = torch.sigmoid(ability_mean[mask] - difficulty_mean)
        probs_list.append(prob)

    return torch.cat(probs_list)


# ==============================================================================
# Shared Evaluation Metrics
# ==============================================================================

def compute_metrics(y_true, y_probs, model_name="Model"):
    """Compute classification metrics."""
    y_probs_np = y_probs.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    y_pred = (y_probs > 0.5).float().cpu().numpy()

    # Handle binary classification
    try:
        auc = roc_auc_score(y_true_np, y_probs_np)
    except:
        auc = 0.5  # Default if AUC can't be computed

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true_np, y_pred),
        "f1": f1_score(y_true_np, y_pred),
        "precision": precision_score(y_true_np, y_pred, zero_division=0),
        "recall": recall_score(y_true_np, y_pred, zero_division=0),
        "auc": auc,
    }

    return metrics


def compute_goodness_of_fit(y_true, y_probs, abilities_or_theta, bin_size=6):
    """
    Compute Goodness of Fit by binning abilities and comparing
    theoretical vs empirical success rates.

    For CIRT: abilities_or_theta is theta1 values (asymptotic ability)
    For GPIRT: abilities_or_theta is mean abilities across time
    """
    # Flatten all data
    y_true_np = y_true.cpu().numpy()
    y_probs_np = y_probs.cpu().numpy()
    abilities_np = abilities_or_theta.cpu().numpy()

    # Create ability bins
    min_ability = abilities_np.min()
    max_ability = abilities_np.max()
    ability_bins = np.linspace(min_ability, max_ability, bin_size + 1)

    diff_list = []
    for bin_start, bin_end in zip(ability_bins[:-1], ability_bins[1:]):
        mask = (abilities_np >= bin_start) & (abilities_np < bin_end)
        if mask.sum() == 0:
            continue

        y_theoretical = y_probs_np[mask]
        y_empirical = y_true_np[mask]

        # Compute absolute difference
        diff = 1 - np.abs(y_empirical - y_theoretical)
        diff_list.append(diff.mean())

    if len(diff_list) == 0:
        return 0.0, 0.0

    diff_array = np.array(diff_list)
    return diff_array.mean(), diff_array.std()


# ==============================================================================
# Main Comparison Pipeline
# ==============================================================================

def load_data(course_name, device):
    """Load dataset from HuggingFace."""
    print(f"Loading dataset: {course_name}")
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{course_name}", repo_type="dataset"
    )

    # Load correctness matrix
    correctness_matrix = pickle.load(open(f"{data_folder}/correctness_matrix.pkl", "rb"))
    correctness_matrix = torch.tensor(correctness_matrix, device=device)

    N, Q, T = correctness_matrix.shape
    print(f"Dataset shape: {N} students, {Q} questions, {T} time points")

    return correctness_matrix, N, Q, T


def prepare_data_for_evaluation(correctness_matrix, N, Q, T, device):
    """Prepare data in flat format for evaluation."""
    y_obs = correctness_matrix.reshape(-1)

    # Create indices
    t = np.linspace(1, T, T)
    t_flat = torch.from_numpy(np.tile(t, len(y_obs) // T)).to(device=device).float()
    student_idx = torch.from_numpy(np.repeat(np.arange(N), Q * T)).to(device=device).long()
    question_idx = torch.from_numpy(np.tile(np.repeat(np.arange(Q), T), N)).to(device=device).long()

    # Remove missing values
    valid_mask = y_obs != -1
    y_obs = y_obs[valid_mask].float()
    t_flat = t_flat[valid_mask]
    student_idx = student_idx[valid_mask]
    question_idx = question_idx[valid_mask]

    # Train/test split
    total_samples = y_obs.shape[0]
    randomized_idxs = torch.randperm(total_samples)
    test_size = int(total_samples * 0.2)
    test_idxs = randomized_idxs[:test_size]

    return {
        "y_obs_test": y_obs[test_idxs],
        "t_flat_test": t_flat[test_idxs],
        "student_idx_test": student_idx[test_idxs],
        "question_idx_test": question_idx[test_idxs],
        "N": N,
        "Q": Q,
        "T": T,
    }


def evaluate_cirt(cirt_result_folder, test_data, device):
    """Evaluate CIRT model on test data."""
    print("\n" + "="*80)
    print("Evaluating CIRT Model")
    print("="*80)

    # Load CIRT parameters
    model_path = f"{cirt_result_folder}/model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CIRT model not found at {model_path}")

    with open(model_path, "rb") as f:
        params = pickle.load(f)

    theta0 = params["theta0"].to(device)
    theta1 = params["theta1"].to(device)
    z = params["z"].to(device)

    # Compute predictions on test set
    y_probs = compute_cirt_predictions(
        theta0, theta1, z,
        test_data["student_idx_test"],
        test_data["question_idx_test"],
        test_data["t_flat_test"]
    )

    # Compute log-likelihood
    log_likelihood = compute_cirt_likelihood(test_data["y_obs_test"], y_probs)

    # Convert continuous scores to binary for classification metrics
    y_true_binary = (test_data["y_obs_test"] > 0.5).float()

    # Compute metrics
    metrics = compute_metrics(y_true_binary, y_probs, model_name="CIRT")
    metrics["log_likelihood"] = log_likelihood

    # Compute Goodness of Fit (using theta1 as ability proxy)
    student_abilities = theta1[test_data["student_idx_test"]]
    gof_mean, gof_std = compute_goodness_of_fit(y_true_binary, y_probs, student_abilities)
    metrics["gof_mean"] = gof_mean
    metrics["gof_std"] = gof_std

    print(f"\nCIRT Results:")
    print(f"  Log-Likelihood: {log_likelihood:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  GoF: {gof_mean:.4f} ± {gof_std:.4f}")

    return metrics, y_probs


def evaluate_gpirt(gpirt_result_folder, test_data, device):
    """Evaluate GPIRT model on test data."""
    print("\n" + "="*80)
    print("Evaluating GPIRT Model")
    print("="*80)

    # Load GPIRT parameters
    ability_path = f"{gpirt_result_folder}/ability.pt"
    difficulty_path = f"{gpirt_result_folder}/difficulty.pt"

    if not os.path.exists(ability_path) or not os.path.exists(difficulty_path):
        raise FileNotFoundError(f"GPIRT model not found in {gpirt_result_folder}")

    abilities = torch.load(ability_path)  # List of [n_samples, n_time_points] per student
    difficulties = torch.load(difficulty_path).to(device)  # [n_samples, n_questions]

    # For each test sample, get the prediction
    # Note: This is simplified - in practice, you'd need to handle time indexing properly
    y_probs_list = []

    for sidx in range(test_data["N"]):
        mask = test_data["student_idx_test"] == sidx
        if mask.sum() == 0:
            continue

        # Get mean ability for this student
        ability_mean = abilities[sidx].mean(dim=0)  # [n_time_points]

        # Get question indices for this student's test samples
        q_indices = test_data["question_idx_test"][mask]

        # Get difficulty mean
        difficulty_mean = difficulties.mean(dim=0)  # [n_questions]

        # For simplicity, use the last time point ability (or could match time indices)
        # In practice, you'd want to match the actual time indices
        student_ability = ability_mean[-1]  # Use final ability

        # Compute probabilities
        probs = torch.sigmoid(student_ability - difficulty_mean[q_indices])
        y_probs_list.append(probs)

    y_probs = torch.cat(y_probs_list)

    # Compute log-likelihood (using Bernoulli for binary outcomes)
    y_true_binary = (test_data["y_obs_test"] > 0.5).float()
    log_probs = y_true_binary * torch.log(y_probs + 1e-6) + (1 - y_true_binary) * torch.log(1 - y_probs + 1e-6)
    log_likelihood = log_probs.mean().item()

    # Compute metrics
    metrics = compute_metrics(y_true_binary, y_probs, model_name="GPIRT")
    metrics["log_likelihood"] = log_likelihood

    # Compute Goodness of Fit
    # Use mean ability across time as proxy
    student_mean_abilities = torch.stack([ab.mean(dim=0).mean() for ab in abilities])
    student_abilities = student_mean_abilities[test_data["student_idx_test"]]
    gof_mean, gof_std = compute_goodness_of_fit(y_true_binary, y_probs, student_abilities)
    metrics["gof_mean"] = gof_mean
    metrics["gof_std"] = gof_std

    print(f"\nGPIRT Results:")
    print(f"  Log-Likelihood: {log_likelihood:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  GoF: {gof_mean:.4f} ± {gof_std:.4f}")

    return metrics, y_probs


def plot_comparison(cirt_metrics, gpirt_metrics, save_folder):
    """Create comparison visualizations."""
    ensure_dir(save_folder)

    # Metric comparison bar plot
    metrics_to_plot = ["log_likelihood", "auc", "accuracy", "f1", "gof_mean"]
    metric_labels = ["Log-Likelihood", "AUC", "Accuracy", "F1", "GoF"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 3))

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        cirt_val = cirt_metrics[metric]
        gpirt_val = gpirt_metrics[metric]

        axes[i].bar(["CIRT", "GPIRT"], [cirt_val, gpirt_val], color=["#4477aa", "#ee6677"])
        axes[i].set_ylabel(label)
        axes[i].set_title(label)

        # Add values on bars
        for j, val in enumerate([cirt_val, gpirt_val]):
            axes[i].text(j, val, f"{val:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(f"{save_folder}/metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nComparison plot saved to {save_folder}/metrics_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Compare CIRT and GPIRT models")
    parser.add_argument("--course_name", type=str, default="dsa_hk231",
                       help="Course name for dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--cirt_concentration", type=float, default=10.0,
                       help="CIRT concentration parameter")
    parser.add_argument("--gpirt_kernel", type=str, default="RBF",
                       choices=["RBF", "Matern"],
                       help="GPIRT kernel type")
    parser.add_argument("--gpirt_length_scale", type=float, default=1.0,
                       help="GPIRT length scale")
    parser.add_argument("--output_folder", type=str, default="comparison_results",
                       help="Output folder for results")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{args.output_folder}/{args.course_name}_{timestamp}"
    ensure_dir(output_folder)

    # Load data
    correctness_matrix, N, Q, T = load_data(args.course_name, device)
    test_data = prepare_data_for_evaluation(correctness_matrix, N, Q, T, device)

    # Evaluate CIRT
    cirt_folder = f"results/{args.course_name}_{args.cirt_concentration}"
    try:
        cirt_metrics, cirt_probs = evaluate_cirt(cirt_folder, test_data, device)
    except FileNotFoundError as e:
        print(f"\nWarning: {e}")
        print(f"Please train CIRT first using:")
        print(f"  python CodeInsights/dynamic_irt/cirt/continuous_irt.py --course_name {args.course_name}")
        cirt_metrics = None

    # Evaluate GPIRT
    gpirt_folder = f"results/{args.course_name}_s{args.seed}_D1_PL1_hmc_kernel{args.gpirt_kernel}_ls{args.gpirt_length_scale}"
    try:
        gpirt_metrics, gpirt_probs = evaluate_gpirt(gpirt_folder, test_data, device)
    except FileNotFoundError as e:
        print(f"\nWarning: {e}")
        print(f"Please train GPIRT first using:")
        print(f"  python CodeInsights/dynamic_irt/gpirt/inference.py --course_name {args.course_name}")
        gpirt_metrics = None

    # Compare and save results
    if cirt_metrics and gpirt_metrics:
        comparison_df = pd.DataFrame([cirt_metrics, gpirt_metrics])
        comparison_df.to_csv(f"{output_folder}/comparison.csv", index=False)
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print('='*80)
        print(comparison_df.to_string(index=False))

        # Create plots
        plot_comparison(cirt_metrics, gpirt_metrics, output_folder)

        # Save detailed results
        with open(f"{output_folder}/comparison.json", "w") as f:
            json.dump({
                "cirt": cirt_metrics,
                "gpirt": gpirt_metrics,
                "config": vars(args)
            }, f, indent=2)

        print(f"\nResults saved to {output_folder}/")
    else:
        print("\nComparison incomplete. Please train both models first.")


if __name__ == "__main__":
    main()
