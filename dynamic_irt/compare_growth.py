"""
Model comparison: growth curves, AUC, and rollout trajectories.

Compares CIRT, Dynamic IRT, and Elo (optionally GPIRT) on the same dataset.
Generates three figures:
  1. Growth curves — each model's estimated θ(t) for top-growth students
  2. AUC comparison — bar chart of predictive AUC across models
  3. Rollout trajectories — predicted P(correct) vs actual outcomes

Usage:
    cd CodeInsights && python -m dynamic_irt.compare_growth
    cd CodeInsights && python -m dynamic_irt.compare_growth --n_students 8
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tueplots import bundles, figsizes

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from dynamic_irt.gpirt.utils import ensure_dir

plt.rcParams.update(bundles.neurips2024())

# Paul Tol qualitative palette
MODEL_COLORS = {
    'CIRT': '#4477aa',
    'Dynamic IRT': '#ee6677',
    'Elo': '#228833',
    'GPIRT': '#aa3377',
}
MODEL_MARKERS = {
    'CIRT': 'o',
    'Dynamic IRT': 's',
    'Elo': '^',
    'GPIRT': 'D',
}


# ==============================================================================
# Data loading
# ==============================================================================

def load_observation_data(device='cpu'):
    """Load raw observation data for evaluation and trajectory plotting."""
    csv_path = hf_hub_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        filename="codeinsights_student_response.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(csv_path)
    df["item_key"] = (
        df["question_unittest_id"].astype(str) + "_" + df["unit_test_id"].astype(str)
    )
    return df


# ==============================================================================
# Model loaders
# ==============================================================================

def load_cirt(result_dir):
    """Load CIRT model parameters."""
    path = os.path.join(result_dir, "cirt", "model.pkl")
    if not os.path.exists(path):
        print(f"  CIRT: model not found at {path}")
        return None
    with open(path, "rb") as f:
        params = pickle.load(f)
    print(f"  CIRT: loaded (N={params['N']}, Q={params['Q']}, T={params['T']})")
    return params


def load_dynamic_irt(result_dir):
    """Load Dynamic IRT model parameters."""
    path = os.path.join(result_dir, "dynamic_irt", "model.pkl")
    if not os.path.exists(path):
        print(f"  Dynamic IRT: model not found at {path}")
        return None
    with open(path, "rb") as f:
        params = pickle.load(f)
    # Also load student_params for student_id mapping
    params_csv = os.path.join(result_dir, "dynamic_irt", "student_params.csv")
    if os.path.exists(params_csv):
        params["student_params"] = pd.read_csv(params_csv)
    print(f"  Dynamic IRT: loaded (N={params['N']}, Q={params['Q']})")
    return params


def load_elo(result_dir):
    """Load Elo model parameters."""
    ability_path = os.path.join(result_dir, "elo", "elo_ability.csv")
    difficulty_path = os.path.join(result_dir, "elo", "elo_difficulty.csv")
    metrics_path = os.path.join(result_dir, "elo", "fit_metrics.json")

    if not os.path.exists(ability_path):
        print(f"  Elo: model not found at {ability_path}")
        return None

    ability = pd.read_csv(ability_path)
    difficulty = pd.read_csv(difficulty_path)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    result = {
        "ability": ability,
        "difficulty": difficulty,
        "metrics": metrics,
    }
    print(f"  Elo: loaded ({len(ability)} students, {len(difficulty)} items)")
    return result


# ==============================================================================
# Growth curve computation
# ==============================================================================

def cirt_growth_curve(params, student_idx, t_range):
    """Compute CIRT ability curve: θ₁ · σ(θ₀ · t) for a student over time.

    Note: theta1 is stored raw (not logit-transformed) — the training code
    penalizes values outside [0, 1] but does not apply sigmoid.
    """
    theta0 = params["theta0"][student_idx].item()
    theta1 = params["theta1"][student_idx].item()
    theta1 = np.clip(theta1, 0, 1)
    curve = theta1 * 1.0 / (1.0 + np.exp(-theta0 * t_range))
    return curve


def dynamic_irt_growth_curve(params, student_idx, t_range):
    """Compute Dynamic IRT ability: θ₀ + g · t for a student over time."""
    theta0 = params["theta0"][student_idx].item()
    growth = params["theta_growth"][student_idx].item()
    curve = theta0 + growth * t_range
    return curve


def get_top_growth_students(dyn_irt_params, n=5):
    """Get indices of students with highest positive growth rate."""
    growth = dyn_irt_params["theta_growth"].numpy()
    top_indices = np.argsort(growth)[::-1][:n]
    # Filter to positive growth only
    top_indices = top_indices[growth[top_indices] > 0]
    return top_indices


# ==============================================================================
# Prediction functions
# ==============================================================================

def cirt_predict(params, student_idx, question_idx, t_flat):
    """CIRT prediction: P(correct) = θ₁[s] · σ(θ₀[s] · t - z[q]).

    Note: theta1 is stored raw (not logit-transformed).
    """
    theta0 = params["theta0"][student_idx]
    theta1 = params["theta1"][student_idx].clamp(0, 1)
    z = params["z"][question_idx]
    return theta1 * torch.sigmoid(theta0 * t_flat - z)


def dynamic_irt_predict(params, student_idx, question_idx, t_flat):
    """Dynamic IRT prediction: P(correct) = σ(θ₀[s] + g[s]·t - β[q])."""
    theta0 = params["theta0"][student_idx]
    growth = params["theta_growth"][student_idx]
    beta = params["beta"][question_idx]
    logit = theta0 + growth * t_flat - beta
    return torch.sigmoid(logit)


def elo_predict(elo_data, student_ids, item_ids):
    """Elo prediction: P(correct) = σ(θ[s] - b[q])."""
    ability_map = dict(zip(
        elo_data["ability"]["StudentID_SF"],
        elo_data["ability"]["final_ability"]
    ))
    difficulty_map = dict(zip(
        elo_data["difficulty"]["ItemID_SF"],
        elo_data["difficulty"]["average_difficulty"]
    ))

    probs = []
    valid = []
    for sid, iid in zip(student_ids, item_ids):
        if sid in ability_map and iid in difficulty_map:
            theta = ability_map[sid]
            b = difficulty_map[iid]
            probs.append(1.0 / (1.0 + np.exp(-(theta - b))))
            valid.append(True)
        else:
            probs.append(0.5)
            valid.append(False)
    return np.array(probs), np.array(valid)


# ==============================================================================
# Figure 1: Growth Curves
# ==============================================================================

def plot_growth_curves(cirt_params, dyn_irt_params, df, output_dir, n_students=5):
    """Plot ability trajectories from each model for top-growth students,
    overlaid with the actual (ground truth) rolling pass rate."""
    top_students = get_top_growth_students(dyn_irt_params, n=n_students)
    if len(top_students) == 0:
        print("  No students with positive growth — skipping growth curves.")
        return

    student_params = dyn_irt_params.get("student_params")

    t_max = min(cirt_params["T"] if cirt_params else 100, 100)
    t_range = np.arange(1, t_max + 1, dtype=np.float32)

    n_rows = len(top_students)
    fig_width = figsizes.neurips2024()["figure.figsize"][0]
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, 2.0 * n_rows),
                             sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, sidx in zip(axes, top_students):
        # Resolve student_id for data lookup
        growth_val = dyn_irt_params["theta_growth"][sidx].item()
        student_id = sidx
        if student_params is not None:
            match = student_params[student_params["person_idx"] == sidx]
            if len(match) > 0:
                student_id = match.iloc[0]["student_id"]

        # --- Ground truth: actual rolling pass rate ---
        student_df = df[df["student_id"] == student_id].sort_values("time_index")
        if len(student_df) >= 3:
            t_actual = student_df["time_index"].values.astype(np.float32)
            y_actual = student_df["response"].values.astype(np.float32)

            # Raw observations as faint dots
            ax.scatter(t_actual, y_actual, color='gray', alpha=0.12, s=4,
                       zorder=1, label=None)

            # Smoothed ground truth (rolling average)
            window = max(10, len(y_actual) // 8)
            y_smoothed = (pd.Series(y_actual)
                          .rolling(window, min_periods=1, center=True)
                          .mean().values)
            ax.plot(t_actual, y_smoothed, color='black', linewidth=1.5,
                    alpha=0.5, label='Actual (smoothed)', zorder=2)

        # --- Model curves ---
        # CIRT growth curve
        if cirt_params is not None:
            curve_cirt = cirt_growth_curve(cirt_params, sidx, t_range)
            ax.plot(t_range, curve_cirt, color=MODEL_COLORS['CIRT'],
                    linewidth=1.5, label='CIRT', alpha=0.9, zorder=3)

        # Dynamic IRT growth curve
        curve_dyn = dynamic_irt_growth_curve(dyn_irt_params, sidx, t_range)
        # Convert to probability scale for comparability
        curve_dyn_prob = 1.0 / (1.0 + np.exp(-curve_dyn))
        ax.plot(t_range, curve_dyn_prob, color=MODEL_COLORS['Dynamic IRT'],
                linewidth=1.5, label='Dynamic IRT', alpha=0.9, zorder=3)

        ax.set_ylabel(r"$P(\mathrm{correct})$", fontsize=7)
        ax.set_title(
            f"Student {student_id} ($g = {growth_val:.4f}$, "
            f"n = {len(student_df)})",
            fontsize=7, pad=2
        )
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=6, loc='lower right', ncol=3)
    axes[-1].set_xlabel("Attempt index $t$")
    fig.tight_layout(h_pad=0.8)

    save_path = os.path.join(output_dir, "growth_curves_comparison.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ==============================================================================
# Figure 2: AUC Comparison
# ==============================================================================

def compute_auc_comparison(cirt_params, dyn_irt_params, elo_data, df, output_dir):
    """Compute and plot AUC comparison across models."""
    # Build index mappings consistent with Dynamic IRT
    student_ids_sorted = sorted(df["student_id"].unique())
    item_keys_sorted = sorted(df["item_key"].unique())
    student_to_idx = {sid: i for i, sid in enumerate(student_ids_sorted)}
    item_to_idx = {ik: i for i, ik in enumerate(item_keys_sorted)}

    df_eval = df.copy()
    df_eval["person_idx"] = df_eval["student_id"].map(student_to_idx)
    df_eval["item_idx"] = df_eval["item_key"].map(item_to_idx)

    # Filter to valid mappings
    df_eval = df_eval.dropna(subset=["person_idx", "item_idx"])
    df_eval["person_idx"] = df_eval["person_idx"].astype(int)
    df_eval["item_idx"] = df_eval["item_idx"].astype(int)

    # Held-out test set (20%)
    np.random.seed(42)
    test_mask = np.random.rand(len(df_eval)) < 0.2
    df_test = df_eval[test_mask].copy()

    y_true = df_test["response"].values

    results = {}

    # Dynamic IRT
    if dyn_irt_params is not None:
        valid = (df_test["person_idx"] < dyn_irt_params["N"]) & \
                (df_test["item_idx"] < dyn_irt_params["Q"])
        df_valid = df_test[valid]
        if len(df_valid) > 0:
            s_idx = torch.tensor(df_valid["person_idx"].values, dtype=torch.long)
            q_idx = torch.tensor(df_valid["item_idx"].values, dtype=torch.long)
            t_vals = torch.tensor(df_valid["time_index"].values, dtype=torch.float32)
            with torch.no_grad():
                preds = dynamic_irt_predict(dyn_irt_params, s_idx, q_idx, t_vals).numpy()
            y_v = df_valid["response"].values
            results["Dynamic IRT"] = {
                "auc": roc_auc_score(y_v, preds),
                "accuracy": accuracy_score(y_v, (preds > 0.5).astype(int)),
                "f1": f1_score(y_v, (preds > 0.5).astype(int)),
            }

    # CIRT — skipped from AUC comparison because it uses a different item
    # indexing (Q=3888 via csv2matrices) than Dynamic IRT (Q=3312 via item_key).
    # Item indices don't map 1:1, so predictions with wrong indices are invalid.
    # CIRT is still included in growth curves and rollout plots (per-student params).
    if cirt_params is not None:
        print("  CIRT: skipped from AUC (different item indexing — "
              f"Q={cirt_params['Q']} vs Dynamic IRT Q={dyn_irt_params['Q']})")

    # Elo
    if elo_data is not None:
        preds_elo, valid_elo = elo_predict(
            elo_data,
            df_test["student_id"].values,
            df_test["question_unittest_id"].values,
        )
        if valid_elo.sum() > 100:
            y_v = y_true[valid_elo]
            p_v = preds_elo[valid_elo]
            results["Elo"] = {
                "auc": roc_auc_score(y_v, p_v),
                "accuracy": accuracy_score(y_v, (p_v > 0.5).astype(int)),
                "f1": f1_score(y_v, (p_v > 0.5).astype(int)),
            }

    if not results:
        print("  No models produced valid predictions — skipping AUC plot.")
        return results

    # Plot
    models = list(results.keys())
    metrics = ["auc", "accuracy", "f1"]
    metric_labels = ["AUC", "Accuracy", "F1"]

    fig_width = figsizes.neurips2024()["figure.figsize"][0]
    fig, axes = plt.subplots(1, len(metrics), figsize=(fig_width, 2.2))

    x = np.arange(len(models))
    bar_width = 0.6

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [results[m][metric] for m in models]
        colors = [MODEL_COLORS.get(m, '#999999') for m in models]
        bars = ax.bar(x, values, bar_width, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=6, rotation=30, ha='right')
        ax.set_ylabel(label, fontsize=7)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)
        ax.grid(True, axis='y', alpha=0.2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=5)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "auc_comparison.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return results


# ==============================================================================
# Figure 3: Rollout Trajectories
# ==============================================================================

def plot_rollout_trajectories(cirt_params, dyn_irt_params, elo_data,
                              df, output_dir, n_students=5):
    """Plot predicted vs actual per-student trajectories."""
    # Get top-growth students from Dynamic IRT
    top_students = get_top_growth_students(dyn_irt_params, n=n_students)
    if len(top_students) == 0:
        print("  No students with positive growth — skipping rollout trajectories.")
        return

    # Map person_idx back to student_id
    student_params = dyn_irt_params.get("student_params")
    if student_params is None:
        print("  No student_params CSV — skipping rollout trajectories.")
        return

    n_rows = len(top_students)
    fig_width = figsizes.neurips2024()["figure.figsize"][0]
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, 2.0 * n_rows),
                             sharex=False)
    if n_rows == 1:
        axes = [axes]

    for ax, sidx in zip(axes, top_students):
        match = student_params[student_params["person_idx"] == sidx]
        if len(match) == 0:
            continue
        student_id = match.iloc[0]["student_id"]

        # Get this student's actual submissions
        student_df = df[df["student_id"] == student_id].sort_values("time_index")
        if len(student_df) < 5:
            continue

        t_vals = student_df["time_index"].values.astype(np.float32)
        y_actual = student_df["response"].values.astype(np.float32)

        # Smoothed actual (rolling mean)
        window = max(5, len(y_actual) // 10)
        y_smoothed = pd.Series(y_actual).rolling(window, min_periods=1, center=True).mean().values
        x_axis = np.arange(len(y_actual))

        # Plot actual
        ax.scatter(x_axis, y_actual, color='gray', alpha=0.15, s=5, zorder=1)
        ax.plot(x_axis, y_smoothed, color='black', linewidth=1.0,
                label='Actual (smoothed)', alpha=0.6, zorder=2)

        # Item indices for predictions
        item_keys = student_df["item_key"].values if "item_key" in student_df.columns else None

        # Dynamic IRT predictions
        t_tensor = torch.tensor(t_vals, dtype=torch.float32)
        s_tensor = torch.full((len(t_vals),), sidx, dtype=torch.long)

        # Get item indices
        item_keys_sorted = sorted(df["item_key"].unique())
        item_to_idx = {ik: i for i, ik in enumerate(item_keys_sorted)}
        q_indices = student_df["item_key"].map(item_to_idx)
        valid_items = ~q_indices.isna()
        q_tensor = torch.tensor(q_indices[valid_items].values.astype(int), dtype=torch.long)

        if valid_items.sum() > 0:
            s_v = s_tensor[valid_items.values]
            t_v = t_tensor[valid_items.values]
            x_v = x_axis[valid_items.values]

            # Dynamic IRT
            if q_tensor.max() < dyn_irt_params["Q"]:
                with torch.no_grad():
                    dyn_preds = dynamic_irt_predict(dyn_irt_params, s_v, q_tensor, t_v).numpy()
                ax.plot(x_v, dyn_preds, color=MODEL_COLORS['Dynamic IRT'],
                        linewidth=1.0, alpha=0.7, label='Dynamic IRT', zorder=3)

            # CIRT
            if cirt_params is not None and q_tensor.max() < cirt_params["Q"]:
                with torch.no_grad():
                    cirt_preds = cirt_predict(cirt_params, s_v, q_tensor, t_v).numpy()
                cirt_preds = np.clip(cirt_preds, 0, 1)
                ax.plot(x_v, cirt_preds, color=MODEL_COLORS['CIRT'],
                        linewidth=1.0, alpha=0.7, label='CIRT', zorder=3)

        growth_val = dyn_irt_params["theta_growth"][sidx].item()
        ax.set_ylabel(r"$P(\mathrm{correct})$", fontsize=7)
        ax.set_title(f"Student {student_id} ($g = {growth_val:.4f}$, "
                     f"n = {len(student_df)})", fontsize=7, pad=2)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=5, loc='lower right', ncol=2)
    axes[-1].set_xlabel("Submission index")
    fig.tight_layout(h_pad=0.8)

    save_path = os.path.join(output_dir, "rollout_trajectory_comparison.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare growth models")
    parser.add_argument("--n_students", type=int, default=5,
                        help="Number of top-growth students to plot")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/comparison)")
    args = parser.parse_args()

    result_dir = os.path.join(REPO_ROOT, "results")
    output_dir = args.output_dir or os.path.join(result_dir, "comparison")
    ensure_dir(output_dir)

    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    cirt_params = load_cirt(result_dir)
    dyn_irt_params = load_dynamic_irt(result_dir)
    elo_data = load_elo(result_dir)

    if dyn_irt_params is None:
        print("\nDynamic IRT is required for student selection. "
              "Run: python -m dynamic_irt.dynamic_irt")
        return

    print("\n" + "=" * 60)
    print("Loading observation data...")
    print("=" * 60)
    df = load_observation_data()
    df["item_key"] = (
        df["question_unittest_id"].astype(str) + "_" + df["unit_test_id"].astype(str)
    )
    print(f"  Loaded {len(df)} observations")

    print("\n" + "=" * 60)
    print("[1/3] Growth Curves")
    print("=" * 60)
    plot_growth_curves(cirt_params, dyn_irt_params, df, output_dir,
                       n_students=args.n_students)

    print("\n" + "=" * 60)
    print("[2/3] AUC Comparison")
    print("=" * 60)
    metrics = compute_auc_comparison(cirt_params, dyn_irt_params, elo_data,
                                      df, output_dir)
    if metrics:
        print("\n  Model comparison:")
        for model, m in metrics.items():
            print(f"    {model:15s}  AUC={m['auc']:.4f}  "
                  f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")

    print("\n" + "=" * 60)
    print("[3/3] Rollout Trajectories")
    print("=" * 60)
    plot_rollout_trajectories(cirt_params, dyn_irt_params, elo_data,
                              df, output_dir, n_students=args.n_students)

    # Save metrics JSON
    if metrics:
        metrics_path = os.path.join(output_dir, "comparison_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics saved: {metrics_path}")

    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
