"""
Dynamic IRT model with linear student growth (MLE via PyTorch).

Each student has an initial ability (theta0) and a growth rate (theta_growth).
Each item has a difficulty (beta). Correctness is modeled as:

    P(correct) = sigmoid(theta0[s] + theta_growth[s] * t - beta[q])

Usage:
    python -m dynamic_irt.dynamic_irt
    python -m dynamic_irt.dynamic_irt --subsample 100
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from dynamic_irt.gpirt import ensure_dir, set_seed

plt.rcParams.update(bundles.aaai2024())

# Standardized color palette (Paul Tol qualitative)
COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44"]


# ── Configuration ────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_data_path() -> str:
    return hf_hub_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        filename="codeinsights_student_response.csv",
        repo_type="dataset",
    )


def prepare_data(csv_path: str, subsample: int | None = None,
                 max_time: int | None = None, seed: int = 42, device: str = 'cpu'):
    """Load CSV, optionally subsample, and return tensors ready for training.

    Args:
        csv_path: Path to the student response CSV.
        subsample: If set, randomly sample this many students. Students are
            filtered to those with T_max < ``max_time`` first.
        max_time: Only used when ``subsample`` is set. Keep students whose
            maximum time_index is below this value (default 100).
        seed: RNG seed for reproducible subsampling.
        device: Torch device.

    Returns:
        (response, student_idx, item_idx, time_vals, N_persons, N_items,
         person_map, item_map)
    """
    df = pd.read_csv(csv_path)

    # Composite item key
    df["item_key"] = (
        df["question_unittest_id"].astype(str) + "_" + df["unit_test_id"].astype(str)
    )

    # Optional subsampling
    if subsample is not None:
        max_t = max_time if max_time is not None else 100
        student_max_time = df.groupby("student_id")["time_index"].max()
        eligible_students = student_max_time[student_max_time < max_t].index.tolist()
        print(f"Students with T_max < {max_t}: {len(eligible_students)} / {df['student_id'].nunique()}")

        if len(eligible_students) <= subsample:
            print(f"  Warning: only {len(eligible_students)} eligible students "
                  f"(requested {subsample}). Using all eligible.")
            sampled_students = eligible_students
        else:
            rng = np.random.default_rng(seed=seed)
            sampled_students = rng.choice(eligible_students, size=subsample, replace=False).tolist()
        print(f"  Sampled {len(sampled_students)} students.")
        df = df[df["student_id"].isin(sampled_students)].copy()

    # 0-based indices for PyTorch embedding-style lookup
    student_ids = sorted(df["student_id"].unique())
    item_keys = sorted(df["item_key"].unique())
    student_to_idx = {sid: i for i, sid in enumerate(student_ids)}
    item_to_idx = {ik: i for i, ik in enumerate(item_keys)}

    df["person_idx"] = df["student_id"].map(student_to_idx)
    df["item_idx"] = df["item_key"].map(item_to_idx)

    N_persons = len(student_ids)
    N_items = len(item_keys)

    # Time values (keep original scale)
    time_vals = df["time_index"].values.astype(np.float32)

    print(f"Persons: {N_persons}")
    print(f"Items:   {N_items}")
    print(f"Obs:     {len(df)}")
    print(f"T range: [{time_vals.min():.0f}, {time_vals.max():.0f}]")

    # Tensors
    response = torch.tensor(df["response"].values, dtype=torch.float32, device=device)
    student_idx = torch.tensor(df["person_idx"].values, dtype=torch.long, device=device)
    item_idx = torch.tensor(df["item_idx"].values, dtype=torch.long, device=device)
    time_t = torch.tensor(time_vals, dtype=torch.float32, device=device)

    # Lookup tables
    person_map = (
        df[["student_id", "person_idx"]]
        .drop_duplicates()
        .sort_values("person_idx")
        .reset_index(drop=True)
    )
    item_map = (
        df[["item_key", "item_idx"]]
        .drop_duplicates()
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    return response, student_idx, item_idx, time_t, N_persons, N_items, person_map, item_map


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def negative_log_likelihood(response, student_idx, item_idx, time_t,
                            theta0, theta_growth, beta):
    """Compute mean binary cross-entropy for the linear-growth Rasch model.

    logit = theta0[s] + theta_growth[s] * t - beta[q]
    """
    logit = theta0[student_idx] + theta_growth[student_idx] * time_t - beta[item_idx]
    return nn.functional.binary_cross_entropy_with_logits(logit, response)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_losses(losses, result_dir):
    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    ax.plot(losses["train"], label="Train", alpha=0.7, color=COLORS[0])
    ax.plot(losses["test"], label="Test", alpha=0.7, color=COLORS[1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Dynamic IRT Training")
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


def plot_trajectories(theta0_np, growth_np, person_map, t_max, result_dir):
    """Plot ability trajectories for top 5 students by positive growth."""
    positive_mask = growth_np > 0
    if not positive_mask.any():
        print("  Warning: No students with positive growth — skipping trajectory plot.")
        return

    top5_idx = np.argsort(growth_np)[::-1][:5]
    top5_idx = top5_idx[growth_np[top5_idx] > 0]

    ts = np.arange(0, t_max + 1)

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    for idx, color in zip(top5_idx, COLORS):
        trajectory = theta0_np[idx] + growth_np[idx] * ts
        student_id = person_map.loc[person_map["person_idx"] == idx, "student_id"].values[0]
        ax.plot(ts, trajectory, color=color,
                label=f"Student {student_id} ($g={growth_np[idx]:.4f}$)")

    ax.set_xlabel("Time index")
    ax.set_ylabel(r"$\theta(t)$")
    ax.set_title(r"Ability Trajectories: Top 5 Students by $\theta_{\mathrm{growth}}$")
    ax.legend()
    save_path = os.path.join(result_dir, "positive_growth_trajectories.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def visualize(parms_dict, person_map, t_max, result_dir):
    theta0 = parms_dict["theta0"].numpy()
    growth = parms_dict["theta_growth"].numpy()
    beta = parms_dict["beta"].numpy()

    plot_param_hist(theta0, r"$\theta_0$", "theta0", result_dir, r"$\theta_0$ (Initial Ability)")
    plot_param_hist(growth, r"$\theta_{\mathrm{growth}}$", "theta_growth", result_dir, r"$\theta_{\mathrm{growth}}$ (Growth Rate)", bins=80, pct_clip=(5, 95))
    plot_param_hist(beta, r"$\beta$", "beta", result_dir, r"$\beta$ (Item Difficulty)", bins=80)
    plot_trajectories(theta0, growth, person_map, t_max, result_dir)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(args.seed)
    result_dir = os.path.join(REPO_ROOT, "results", "dynamic_irt")
    ensure_dir(result_dir)

    # Load data
    data_path = get_data_path()
    response, student_idx, item_idx, time_t, N, Q, person_map, item_map = prepare_data(
        data_path,
        subsample=args.subsample,
        max_time=args.max_time,
        seed=args.seed,
        device=device,
    )

    # Train/test split
    total = response.shape[0]
    perm = torch.randperm(total)
    train_size = int(total * 0.8)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    print(f"\nDataset statistics:")
    print(f"  Total samples: {total}")
    print(f"  Training: {train_size}")
    print(f"  Testing: {total - train_size}")
    print(f"  Students: {N}")
    print(f"  Items: {Q}\n")

    # Initialize parameters
    theta0 = nn.Parameter(torch.randn(N, device=device) * 0.1)
    theta_growth = nn.Parameter(torch.randn(N, device=device) * 0.01)
    beta = nn.Parameter(torch.randn(Q, device=device) * 0.1)

    optimizer = optim.Adam([theta0, theta_growth, beta], lr=args.lr)

    # Training loop
    losses = {"train": [], "test": []}
    best_test_loss = float('inf')

    print("Starting training...\n")
    for epoch in tqdm(range(args.epochs), desc="Training Dynamic IRT"):
        optimizer.zero_grad()

        loss = negative_log_likelihood(
            response[train_idx], student_idx[train_idx], item_idx[train_idx],
            time_t[train_idx], theta0, theta_growth, beta
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_loss = negative_log_likelihood(
                response[test_idx], student_idx[test_idx], item_idx[test_idx],
                time_t[test_idx], theta0, theta_growth, beta
            )

        losses["train"].append(loss.item())
        losses["test"].append(test_loss.item())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}] "
                  f"Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            with open(os.path.join(result_dir, "model_best.pkl"), "wb") as f:
                pickle.dump({
                    "theta0": theta0.detach().cpu(),
                    "theta_growth": theta_growth.detach().cpu(),
                    "beta": beta.detach().cpu(),
                    "epoch": epoch + 1,
                    "test_loss": test_loss.item(),
                }, f)

    # Save final model
    parms_dict = {
        "theta0": theta0.detach().cpu(),
        "theta_growth": theta_growth.detach().cpu(),
        "beta": beta.detach().cpu(),
        "N": N, "Q": Q,
    }
    with open(os.path.join(result_dir, "model.pkl"), "wb") as f:
        pickle.dump(parms_dict, f)

    # Save loss history
    pd.DataFrame(losses).to_json(os.path.join(result_dir, "losses.json"), indent=4)

    # Save parameter CSVs
    person_map.to_csv(os.path.join(result_dir, "person_map.csv"), index=False)
    item_map.to_csv(os.path.join(result_dir, "item_map.csv"), index=False)

    student_params = person_map.copy()
    student_params["theta0"] = theta0.detach().cpu().numpy()
    student_params["theta_growth"] = theta_growth.detach().cpu().numpy()
    student_params.to_csv(os.path.join(result_dir, "student_params.csv"), index=False)

    item_params = item_map.copy()
    item_params["beta"] = beta.detach().cpu().numpy()
    item_params.to_csv(os.path.join(result_dir, "item_params.csv"), index=False)

    t_max = int(time_t.max().item())

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final train loss: {losses['train'][-1]:.4f}")
    print(f"  Final test loss:  {losses['test'][-1]:.4f}")
    print(f"  Best test loss:   {best_test_loss:.4f}")
    print(f"  Results saved to: {result_dir}/")

    # Visualize
    print(f"\nGenerating plots...")
    plot_losses(losses, result_dir)
    visualize(parms_dict, person_map, t_max, result_dir)

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit Dynamic IRT (linear growth) model via MLE.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Randomly sample this many students (for faster iteration).")
    parser.add_argument("--max-time", type=int, default=100,
                        help="When --subsample is set, only keep students with "
                             "T_max below this value (default: 100).")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
