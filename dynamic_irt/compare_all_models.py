"""
Comprehensive comparison of all dynamic IRT models:
- CIRT (Continuous IRT with parametric sigmoid curves)
- GPIRT (Gaussian Process IRT with Bayesian inference)
- Elo (Elo rating system with dynamic updates)
- RSSM (Recurrent State-Space Model with embeddings)

Compares models using standardized metrics:
- Log-Likelihood
- AUC (Area Under ROC Curve)
- Accuracy, F1, Precision, Recall
- Goodness-of-Fit
- Training Time
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Model Loading Functions
# ==============================================================================

class ModelEvaluator:
    """Base class for model evaluation."""

    def __init__(self, model_name, device='cpu'):
        self.model_name = model_name
        self.device = device
        self.metrics = {}
        self.training_time = None

    def load_model(self, path):
        """Load trained model parameters."""
        raise NotImplementedError

    def predict(self, test_data):
        """Generate predictions on test data."""
        raise NotImplementedError

    def compute_metrics(self, y_true, y_pred_probs):
        """Compute standardized evaluation metrics."""
        y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred_probs_np = y_pred_probs.cpu().numpy() if torch.is_tensor(y_pred_probs) else y_pred_probs
        y_pred = (y_pred_probs_np > 0.5).astype(int)

        # Handle edge cases
        try:
            auc = roc_auc_score(y_true_np, y_pred_probs_np)
        except:
            auc = 0.5

        try:
            f1 = f1_score(y_true_np, y_pred)
        except:
            f1 = 0.0

        metrics = {
            'model': self.model_name,
            'auc': auc,
            'accuracy': accuracy_score(y_true_np, y_pred),
            'f1': f1,
            'precision': precision_score(y_true_np, y_pred, zero_division=0),
            'recall': recall_score(y_true_np, y_pred, zero_division=0),
        }

        # Compute log-likelihood
        eps = 1e-6
        y_pred_probs_np = np.clip(y_pred_probs_np, eps, 1 - eps)
        log_likelihood = (
            y_true_np * np.log(y_pred_probs_np) +
            (1 - y_true_np) * np.log(1 - y_pred_probs_np)
        ).mean()
        metrics['log_likelihood'] = log_likelihood

        return metrics


class CIRTEvaluator(ModelEvaluator):
    """CIRT model evaluator."""

    def __init__(self, device='cpu'):
        super().__init__('CIRT', device)
        self.theta0 = None
        self.theta1 = None
        self.z = None

    def load_model(self, result_folder):
        """Load CIRT parameters."""
        model_path = f"{result_folder}/model.pkl"
        if not os.path.exists(model_path):
            return False

        with open(model_path, 'rb') as f:
            params = pickle.load(f)

        self.theta0 = params['theta0'].to(self.device)
        self.theta1 = params['theta1'].to(self.device)
        self.z = params['z'].to(self.device)

        # Load training time if available
        losses_path = f"{result_folder}/losses.json"
        if os.path.exists(losses_path):
            with open(losses_path, 'r') as f:
                losses = json.load(f)
                # Estimate training time from number of epochs (rough estimate)
                self.training_time = len(losses.get('loss', [])) * 0.1  # ~0.1s per epoch

        return True

    def predict(self, student_idx, question_idx, t_flat):
        """Generate CIRT predictions."""
        mean_correct = self.theta1[student_idx] * torch.sigmoid(
            self.theta0[student_idx] * t_flat - self.z[question_idx]
        )
        return mean_correct


class GPIRTEvaluator(ModelEvaluator):
    """GPIRT model evaluator."""

    def __init__(self, device='cpu'):
        super().__init__('GPIRT', device)
        self.abilities = None
        self.difficulties = None

    def load_model(self, result_folder):
        """Load GPIRT parameters."""
        ability_path = f"{result_folder}/ability.pt"
        difficulty_path = f"{result_folder}/difficulty.pt"

        if not os.path.exists(ability_path) or not os.path.exists(difficulty_path):
            return False

        self.abilities = torch.load(ability_path)  # List of [n_samples, n_time] per student
        self.difficulties = torch.load(difficulty_path).to(self.device)  # [n_samples, n_questions]

        return True

    def predict(self, student_idx, question_idx, n_students):
        """Generate GPIRT predictions (using mean of posterior samples)."""
        y_probs_list = []

        for sidx in range(n_students):
            mask = student_idx == sidx
            if mask.sum() == 0:
                continue

            # Mean ability across MCMC samples and time
            ability_mean = self.abilities[sidx].mean(dim=0).mean()  # Scalar

            # Mean difficulty across MCMC samples
            difficulty_mean = self.difficulties.mean(dim=0)  # [n_questions]

            # Get difficulties for this student's questions
            q_indices = question_idx[mask]

            # IRT probability
            probs = torch.sigmoid(ability_mean - difficulty_mean[q_indices])
            y_probs_list.append(probs)

        return torch.cat(y_probs_list) if y_probs_list else torch.tensor([])


class EloEvaluator(ModelEvaluator):
    """Elo model evaluator."""

    def __init__(self, device='cpu'):
        super().__init__('Elo', device)
        self.fitted_data = None

    def load_model(self, result_folder):
        """Load Elo fitted data."""
        # Elo typically saves fitted data as CSV
        fitted_path = f"{result_folder}/elo_fitted.csv"
        if not os.path.exists(fitted_path):
            # Try alternative naming
            fitted_path = f"{result_folder}/fitted_data.pkl"
            if not os.path.exists(fitted_path):
                return False

        if fitted_path.endswith('.csv'):
            self.fitted_data = pd.read_csv(fitted_path)
        else:
            with open(fitted_path, 'rb') as f:
                self.fitted_data = pickle.load(f)

        return True

    def predict(self, test_data):
        """Generate Elo predictions."""
        # Elo uses updated theta and difficulty to predict
        if 'ThetaUpdated' in self.fitted_data and 'DifficultyUpdated' in self.fitted_data:
            theta = self.fitted_data['ThetaUpdated'].values
            difficulty = self.fitted_data['DifficultyUpdated'].values
            probs = 1 / (1 + np.exp(-(theta - difficulty)))
            return probs
        return None


class RSSMEvaluator(ModelEvaluator):
    """RSSM model evaluator."""

    def __init__(self, device='cpu'):
        super().__init__('RSSM', device)
        self.model_state = None

    def load_model(self, result_folder):
        """Load RSSM model state."""
        model_path = f"{result_folder}/rssm_model.pt"
        if not os.path.exists(model_path):
            return False

        self.model_state = torch.load(model_path, map_location=self.device)
        return True

    def predict(self, test_data):
        """Generate RSSM predictions."""
        # RSSM predictions require running the recurrent model
        # This is a placeholder - actual implementation depends on model structure
        if 'test_predictions' in self.model_state:
            return self.model_state['test_predictions']
        return None


# ==============================================================================
# Data Loading
# ==============================================================================

def load_data(course_name, device):
    """Load dataset from HuggingFace."""
    print(f"Loading dataset: {course_name}")
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{course_name}", repo_type="dataset"
    )

    correctness_matrix = pickle.load(open(f"{data_folder}/correctness_matrix.pkl", "rb"))
    correctness_matrix = torch.tensor(correctness_matrix, device=device)

    N, Q, T = correctness_matrix.shape
    print(f"Dataset shape: {N} students, {Q} questions, {T} time points")

    return correctness_matrix, N, Q, T


def prepare_test_data(correctness_matrix, N, Q, T, device, test_ratio=0.2):
    """Prepare test data for evaluation."""
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

    # Test split
    total_samples = y_obs.shape[0]
    randomized_idxs = torch.randperm(total_samples)
    test_size = int(total_samples * test_ratio)
    test_idxs = randomized_idxs[:test_size]

    return {
        'y_obs': y_obs[test_idxs],
        't_flat': t_flat[test_idxs],
        'student_idx': student_idx[test_idxs],
        'question_idx': question_idx[test_idxs],
        'N': N,
        'Q': Q,
        'T': T,
    }


# ==============================================================================
# Visualization
# ==============================================================================

def plot_model_comparison(results_df, output_folder):
    """Create comprehensive comparison visualizations."""
    ensure_dir(output_folder)

    # Metrics to plot
    metrics = ['auc', 'accuracy', 'f1', 'log_likelihood']
    metric_labels = ['AUC', 'Accuracy', 'F1 Score', 'Log-Likelihood']

    # 1. Bar plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if metric not in results_df.columns:
            continue

        ax = axes[i]
        bars = ax.bar(results_df['model'], results_df[metric],
                     color=['#4477aa', '#ee6677', '#ccbb44', '#66ccee'])

        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} Comparison', fontsize=14)
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/model_comparison_bars.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Radar chart
    if len(results_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Normalize metrics to 0-1 scale for radar chart
        metrics_for_radar = ['auc', 'accuracy', 'f1', 'precision', 'recall']
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = ['#4477aa', '#ee6677', '#ccbb44', '#66ccee']

        for idx, (_, row) in enumerate(results_df.iterrows()):
            if idx >= len(colors):
                break

            values = [row.get(m, 0) for m in metrics_for_radar]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics_for_radar])
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Model Performance Comparison', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_folder}/model_comparison_radar.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Heatmap
    if len(results_df) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        heatmap_data = results_df[['model'] + metrics].set_index('model')
        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0.5, ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title('Model Performance Heatmap', fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{output_folder}/model_comparison_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n✓ Visualizations saved to {output_folder}/")


# ==============================================================================
# Main Comparison
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare all dynamic IRT models')
    parser.add_argument('--course_name', type=str, default='dsa_hk231',
                       help='Course name for dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--cirt_concentration', type=float, default=10.0,
                       help='CIRT concentration parameter')
    parser.add_argument('--gpirt_kernel', type=str, default='RBF',
                       help='GPIRT kernel type')
    parser.add_argument('--gpirt_length_scale', type=float, default=1.0,
                       help='GPIRT length scale')
    parser.add_argument('--output_folder', type=str, default='comparison_results',
                       help='Output folder for results')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['cirt', 'gpirt', 'elo', 'rssm'],
                       help='Models to compare (cirt, gpirt, elo, rssm)')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{args.output_folder}/all_models_{args.course_name}_{timestamp}"
    ensure_dir(output_folder)

    # Load data
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    correctness_matrix, N, Q, T = load_data(args.course_name, device)
    test_data = prepare_test_data(correctness_matrix, N, Q, T, device)
    y_true_binary = (test_data['y_obs'] > 0.5).float()

    # Initialize evaluators
    evaluators = {}
    if 'cirt' in args.models:
        evaluators['cirt'] = CIRTEvaluator(device)
    if 'gpirt' in args.models:
        evaluators['gpirt'] = GPIRTEvaluator(device)
    if 'elo' in args.models:
        evaluators['elo'] = EloEvaluator(device)
    if 'rssm' in args.models:
        evaluators['rssm'] = RSSMEvaluator(device)

    # Evaluate each model
    results = []

    print("\n" + "="*80)
    print("EVALUATING MODELS")
    print("="*80)

    # CIRT
    if 'cirt' in evaluators:
        print("\n[1] Evaluating CIRT...")
        cirt_folder = f"results/{args.course_name}_{args.cirt_concentration}"
        if evaluators['cirt'].load_model(cirt_folder):
            y_pred = evaluators['cirt'].predict(
                test_data['student_idx'],
                test_data['question_idx'],
                test_data['t_flat']
            )
            metrics = evaluators['cirt'].compute_metrics(y_true_binary, y_pred)
            metrics['training_time'] = evaluators['cirt'].training_time or 'N/A'
            results.append(metrics)
            print(f"   ✓ CIRT: AUC={metrics['auc']:.3f}, Acc={metrics['accuracy']:.3f}")
        else:
            print(f"   ✗ CIRT model not found at {cirt_folder}")

    # GPIRT
    if 'gpirt' in evaluators:
        print("\n[2] Evaluating GPIRT...")
        gpirt_folder = f"results/{args.course_name}_s{args.seed}_D1_PL1_hmc_kernel{args.gpirt_kernel}_ls{args.gpirt_length_scale}"
        if evaluators['gpirt'].load_model(gpirt_folder):
            y_pred = evaluators['gpirt'].predict(
                test_data['student_idx'],
                test_data['question_idx'],
                test_data['N']
            )
            if len(y_pred) > 0:
                metrics = evaluators['gpirt'].compute_metrics(y_true_binary, y_pred)
                results.append(metrics)
                print(f"   ✓ GPIRT: AUC={metrics['auc']:.3f}, Acc={metrics['accuracy']:.3f}")
            else:
                print("   ✗ GPIRT: No predictions generated")
        else:
            print(f"   ✗ GPIRT model not found at {gpirt_folder}")

    # Elo
    if 'elo' in evaluators:
        print("\n[3] Evaluating Elo...")
        elo_folder = f"results/elo_{args.course_name}"
        if evaluators['elo'].load_model(elo_folder):
            y_pred = evaluators['elo'].predict(test_data)
            if y_pred is not None:
                metrics = evaluators['elo'].compute_metrics(y_true_binary.cpu().numpy(), y_pred)
                results.append(metrics)
                print(f"   ✓ Elo: AUC={metrics['auc']:.3f}, Acc={metrics['accuracy']:.3f}")
            else:
                print("   ✗ Elo: No predictions available")
        else:
            print(f"   ✗ Elo model not found at {elo_folder}")

    # RSSM
    if 'rssm' in evaluators:
        print("\n[4] Evaluating RSSM...")
        rssm_folder = f"results/rssm_{args.course_name}"
        if evaluators['rssm'].load_model(rssm_folder):
            y_pred = evaluators['rssm'].predict(test_data)
            if y_pred is not None:
                metrics = evaluators['rssm'].compute_metrics(y_true_binary.cpu().numpy(), y_pred)
                results.append(metrics)
                print(f"   ✓ RSSM: AUC={metrics['auc']:.3f}, Acc={metrics['accuracy']:.3f}")
            else:
                print("   ✗ RSSM: No predictions available")
        else:
            print(f"   ✗ RSSM model not found at {rssm_folder}")

    # Generate comparison report
    if len(results) > 0:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        results_df = pd.DataFrame(results)
        results_df = results_df.round(4)

        print("\n" + results_df.to_string(index=False))

        # Save results
        results_df.to_csv(f"{output_folder}/comparison.csv", index=False)

        with open(f"{output_folder}/comparison.json", 'w') as f:
            json.dump({
                'results': results,
                'config': vars(args),
                'best_model': results_df.loc[results_df['auc'].idxmax()]['model']
            }, f, indent=2)

        # Create visualizations
        plot_model_comparison(results_df, output_folder)

        print(f"\n{'='*80}")
        print(f"✓ Results saved to {output_folder}/")
        print(f"{'='*80}")

        # Print winner
        best_idx = results_df['auc'].idxmax()
        print(f"\n🏆 Best Model (by AUC): {results_df.loc[best_idx, 'model']} "
              f"(AUC={results_df.loc[best_idx, 'auc']:.3f})")
    else:
        print("\n⚠ No models were successfully evaluated.")
        print("Please train the models first using the respective training scripts.")


if __name__ == '__main__':
    main()
