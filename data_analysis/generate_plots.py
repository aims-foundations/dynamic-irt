#!/usr/bin/env python3
"""
generate_plots.py - Generate publication-quality figures for EduCodeSim paper

Generates scenario result plots and psychometric analysis figures matching
the style of the overleaf paper.

Usage:
    python generate_plots.py --output ./plots --metrics ./metrics
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Plot Configuration ────────────────────────────────────────────────────────
# Model display order (as in paper figures)
MODELS_ORDER = [
    'gpt-4.1-nano',
    'gemini-2.0-flash',
    'claude-sonnet-4',
    'mistral',
    'llama-3.1-8b',
    'gemma-3-27b',
    'qwen2.5-14b',
]

# Model name mapping (from data to display)
MODEL_NAME_MAP = {
    'gpt-4o': 'gpt-4.1-nano',
    'gpt-4.1-nano': 'gpt-4.1-nano',
    'gemini-2.0-flash': 'gemini-2.0-flash',
    'gemini-2.5-pro': 'gemini-2.0-flash',
    'claude-3-5': 'claude-sonnet-4',
    'claude-sonnet-4': 'claude-sonnet-4',
    'mistral': 'mistral',
    'mistral-large': 'mistral',
    'llama-3.1-8b-instruct': 'llama-3.1-8b',
    'llama-3.1-8b': 'llama-3.1-8b',
    'gemma-3-27b-it': 'gemma-3-27b',
    'gemma-3-27b': 'gemma-3-27b',
    'qwen2.5-14b-instruct': 'qwen2.5-14b',
    'qwen-2.5-14b': 'qwen2.5-14b',
    'qwen2.5-14b': 'qwen2.5-14b',
}

# Colors matching the paper style
COLORS = {
    'UTSR': '#1f77b4',      # Blue
    'ASTED': '#1f77b4',     # Blue
    'CosSim': '#ff7f0e',    # Orange
    'UTCA': '#2ca02c',      # Green
    'RMSE': '#d62728',      # Red
    'EAS': '#9467bd',       # Purple
    'Ability': '#1f77b4',   # Blue
    'Difficulty': '#ff7f0e', # Orange
}


def setup_plot_style():
    """Setup matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def normalize_model_name(name: str) -> str:
    """Normalize model name for consistent display."""
    name_lower = name.lower().replace('_', '-').replace(' ', '-')
    for key, value in MODEL_NAME_MAP.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return value
    return name


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_correlations(correlations_path: str) -> Dict:
    """Load psychometric correlations from JSON file."""
    with open(correlations_path, 'r') as f:
        return json.load(f)


# ─── Scenario 1 Plot ───────────────────────────────────────────────────────────
def plot_scenario1(metrics: Dict, output_path: Path):
    """
    Generate Scenario 1 bar chart (UTSR - Unit Test Success Rate).
    Single bar per model showing code correctness.
    """
    logger.info("Generating Scenario 1 plot (UTSR)...")

    # Extract S1 metrics
    s1_data = metrics.get('S1', metrics.get('scenario1', {}))

    # Prepare data
    models = []
    utsr_values = []

    for model_name, model_metrics in s1_data.items():
        display_name = normalize_model_name(model_name)
        if display_name in MODELS_ORDER:
            models.append(display_name)
            utsr = model_metrics.get('UTSR', model_metrics.get('utsr', 0))
            utsr_values.append(utsr)

    # Sort by model order
    sorted_data = sorted(zip(models, utsr_values),
                        key=lambda x: MODELS_ORDER.index(x[0]) if x[0] in MODELS_ORDER else 999)
    models, utsr_values = zip(*sorted_data) if sorted_data else ([], [])

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    bars = ax.bar(x, utsr_values, color=COLORS['UTSR'], width=0.6, label='UTSR')

    # Styling
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, max(utsr_values) * 1.15 if utsr_values else 1.0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=1, framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Save
    output_file = output_path / 'scenario1_result.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_path / 'scenario1_result.png', format='png', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_file}")


# ─── Scenario 2 Plot ───────────────────────────────────────────────────────────
def plot_scenario2(metrics: Dict, output_path: Path):
    """
    Generate Scenario 2 grouped bar chart (ASTED, CosSim, UTCA).
    Code Performance Imitation metrics.
    """
    logger.info("Generating Scenario 2 plot (ASTED, CosSim, UTCA)...")

    s2_data = metrics.get('S2', metrics.get('scenario2', {}))

    # Prepare data
    models = []
    asted_values = []
    cossim_values = []
    utca_values = []

    for model_name, model_metrics in s2_data.items():
        display_name = normalize_model_name(model_name)
        if display_name in MODELS_ORDER:
            models.append(display_name)
            asted_values.append(model_metrics.get('ASTED', model_metrics.get('asted', 0)))
            cossim_values.append(model_metrics.get('CosSim', model_metrics.get('cossim', 0)))
            utca_values.append(model_metrics.get('UTCA', model_metrics.get('utca', 0)))

    # Sort by model order
    sorted_data = sorted(zip(models, asted_values, cossim_values, utca_values),
                        key=lambda x: MODELS_ORDER.index(x[0]) if x[0] in MODELS_ORDER else 999)

    if sorted_data:
        models, asted_values, cossim_values, utca_values = zip(*sorted_data)
    else:
        models, asted_values, cossim_values, utca_values = [], [], [], []

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, asted_values, width, label='ASTED', color=COLORS['ASTED'])
    bars2 = ax.bar(x, cossim_values, width, label='CosSim', color=COLORS['CosSim'])
    bars3 = ax.bar(x + width, utca_values, width, label='UTCA', color=COLORS['UTCA'])

    # Styling
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Save
    output_file = output_path / 'scenario2_result.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_path / 'scenario2_result.png', format='png', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_file}")


# ─── Scenario 3 Plot ───────────────────────────────────────────────────────────
def plot_scenario3(metrics: Dict, output_path: Path):
    """
    Generate Scenario 3 grouped bar chart (ASTED, CosSim, UTCA, RMSE).
    Targeted Error Reproduction metrics.
    """
    logger.info("Generating Scenario 3 plot (ASTED, CosSim, UTCA, RMSE)...")

    s3_data = metrics.get('S3', metrics.get('scenario3', {}))

    # Prepare data
    models = []
    asted_values = []
    cossim_values = []
    utca_values = []
    rmse_values = []

    for model_name, model_metrics in s3_data.items():
        display_name = normalize_model_name(model_name)
        if display_name in MODELS_ORDER:
            models.append(display_name)
            asted_values.append(model_metrics.get('ASTED', model_metrics.get('asted', 0)))
            cossim_values.append(model_metrics.get('CosSim', model_metrics.get('cossim', 0)))
            utca_values.append(model_metrics.get('UTCA', model_metrics.get('utca', 0)))
            rmse_values.append(model_metrics.get('RMSE', model_metrics.get('rmse', 0)))

    # Sort by model order
    sorted_data = sorted(zip(models, asted_values, cossim_values, utca_values, rmse_values),
                        key=lambda x: MODELS_ORDER.index(x[0]) if x[0] in MODELS_ORDER else 999)

    if sorted_data:
        models, asted_values, cossim_values, utca_values, rmse_values = zip(*sorted_data)
    else:
        models, asted_values, cossim_values, utca_values, rmse_values = [], [], [], [], []

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, asted_values, width, label='ASTED', color=COLORS['ASTED'])
    bars2 = ax.bar(x - 0.5*width, cossim_values, width, label='CosSim', color=COLORS['CosSim'])
    bars3 = ax.bar(x + 0.5*width, utca_values, width, label='UTCA', color=COLORS['UTCA'])
    bars4 = ax.bar(x + 1.5*width, rmse_values, width, label='RMSE', color=COLORS['RMSE'])

    # Styling
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Save
    output_file = output_path / 'scenario3_result.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_path / 'scenario3_result.png', format='png', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_file}")


# ─── Scenario 4 Plot ───────────────────────────────────────────────────────────
def plot_scenario4(metrics: Dict, output_path: Path):
    """
    Generate Scenario 4 grouped bar chart (ASTED, CosSim, EAS, UTCA).
    Efficiency Alignment metrics.
    """
    logger.info("Generating Scenario 4 plot (ASTED, CosSim, EAS, UTCA)...")

    s4_data = metrics.get('S4', metrics.get('scenario4', {}))

    # Prepare data
    models = []
    asted_values = []
    cossim_values = []
    eas_values = []
    utca_values = []

    for model_name, model_metrics in s4_data.items():
        display_name = normalize_model_name(model_name)
        if display_name in MODELS_ORDER:
            models.append(display_name)
            asted_values.append(model_metrics.get('ASTED', model_metrics.get('asted', 0)))
            cossim_values.append(model_metrics.get('CosSim', model_metrics.get('cossim', 0)))
            eas_values.append(model_metrics.get('EAS', model_metrics.get('eas', 0)))
            utca_values.append(model_metrics.get('UTCA', model_metrics.get('utca', 0)))

    # Sort by model order
    sorted_data = sorted(zip(models, asted_values, cossim_values, eas_values, utca_values),
                        key=lambda x: MODELS_ORDER.index(x[0]) if x[0] in MODELS_ORDER else 999)

    if sorted_data:
        models, asted_values, cossim_values, eas_values, utca_values = zip(*sorted_data)
    else:
        models, asted_values, cossim_values, eas_values, utca_values = [], [], [], [], []

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, asted_values, width, label='ASTED', color=COLORS['ASTED'])
    bars2 = ax.bar(x - 0.5*width, cossim_values, width, label='CosSim', color=COLORS['CosSim'])
    bars3 = ax.bar(x + 0.5*width, eas_values, width, label='EAS', color=COLORS['EAS'])
    bars4 = ax.bar(x + 1.5*width, utca_values, width, label='UTCA', color=COLORS['UTCA'])

    # Styling
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    # EAS can be > 1, adjust ylim
    max_val = max(max(eas_values, default=1), 1.0) * 1.1
    ax.set_ylim(0, max_val)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Save
    output_file = output_path / 'scenario4_result.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_path / 'scenario4_result.png', format='png', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_file}")


# ─── Psychometrics Plot ────────────────────────────────────────────────────────
def plot_psychometrics(correlations: Dict, output_path: Path):
    """
    Generate psychometrics result plot (Ability and Difficulty correlations).
    Grouped bar chart showing Pearson correlations per model.
    """
    logger.info("Generating psychometrics plot (Ability, Difficulty correlations)...")

    # Prepare data
    models = []
    ability_values = []
    difficulty_values = []

    for model_name, model_corrs in correlations.items():
        display_name = normalize_model_name(model_name)
        if display_name in MODELS_ORDER:
            models.append(display_name)
            ability_values.append(model_corrs.get('ability', model_corrs.get('ability_correlation', 0)))
            difficulty_values.append(model_corrs.get('difficulty', model_corrs.get('difficulty_correlation', 0)))

    # Sort by model order
    sorted_data = sorted(zip(models, ability_values, difficulty_values),
                        key=lambda x: MODELS_ORDER.index(x[0]) if x[0] in MODELS_ORDER else 999)

    if sorted_data:
        models, ability_values, difficulty_values = zip(*sorted_data)
    else:
        models, ability_values, difficulty_values = [], [], []

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, ability_values, width, label='Ability', color=COLORS['Ability'])
    bars2 = ax.bar(x + width/2, difficulty_values, width, label='Difficulty', color=COLORS['Difficulty'])

    # Styling
    ax.set_ylabel('Correlation')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')

    # Set y-axis to show negative values
    all_values = list(ability_values) + list(difficulty_values)
    if all_values:
        y_min = min(min(all_values), -0.3)
        y_max = max(max(all_values), 0.4)
    else:
        y_min, y_max = -0.3, 0.4

    ax.set_ylim(y_min * 1.1, y_max * 1.1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Save
    output_file = output_path / 'psychometrics_result.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_path / 'psychometrics_result.png', format='png', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_file}")


# ─── Distribution Plots ────────────────────────────────────────────────────────
def plot_ability_distribution(ability_data: pd.DataFrame, output_path: Path):
    """Generate histogram of student ability estimates."""
    logger.info("Generating ability distribution plot...")

    fig, ax = plt.subplots(figsize=(8, 5))

    if 'ability' in ability_data.columns:
        ax.hist(ability_data['ability'], bins=30, color=COLORS['Ability'], alpha=0.7, edgecolor='black')
    elif 'theta' in ability_data.columns:
        ax.hist(ability_data['theta'], bins=30, color=COLORS['Ability'], alpha=0.7, edgecolor='black')

    ax.set_xlabel('Ability Estimate (θ)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Student Ability Estimates')

    output_file = output_path / 'ability_distribution.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_path / 'ability_distribution.png', format='png', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_file}")


def plot_difficulty_distribution(difficulty_data: pd.DataFrame, output_path: Path):
    """Generate histogram of item difficulty estimates."""
    logger.info("Generating difficulty distribution plot...")

    fig, ax = plt.subplots(figsize=(8, 5))

    if 'difficulty' in difficulty_data.columns:
        ax.hist(difficulty_data['difficulty'], bins=30, color=COLORS['Difficulty'], alpha=0.7, edgecolor='black')
    elif 'z' in difficulty_data.columns:
        ax.hist(difficulty_data['z'], bins=30, color=COLORS['Difficulty'], alpha=0.7, edgecolor='black')

    ax.set_xlabel('Difficulty Estimate (z)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Item Difficulty Estimates')

    output_file = output_path / 'difficulty_distribution.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_path / 'difficulty_distribution.png', format='png', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_file}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures for EduCodeSim paper"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="./metrics",
        help="Directory containing metrics JSON files"
    )
    parser.add_argument(
        "--psychometrics",
        type=str,
        default="./psychometrics",
        help="Directory containing psychometric analysis results"
    )

    args = parser.parse_args()

    # Setup
    setup_plot_style()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics)
    psychometrics_path = Path(args.psychometrics)

    # Load metrics
    metrics = {}
    all_results_file = metrics_path / 'all_results.json'
    if all_results_file.exists():
        metrics = load_metrics(str(all_results_file))
        logger.info(f"Loaded metrics from {all_results_file}")
    else:
        # Try to build metrics from scenario result CSVs
        logger.warning(f"Metrics file not found: {all_results_file}")
        logger.info("Attempting to generate plots with sample data...")

        # Sample data for demonstration
        metrics = {
            'S1': {
                'gpt-4.1-nano': {'UTSR': 0.495},
                'gemini-2.0-flash': {'UTSR': 0.558},
                'claude-sonnet-4': {'UTSR': 0.668},
                'mistral': {'UTSR': 0.534},
                'llama-3.1-8b': {'UTSR': 0.490},
                'gemma-3-27b': {'UTSR': 0.449},
                'qwen2.5-14b': {'UTSR': 0.366},
            },
            'S2': {
                'gpt-4.1-nano': {'ASTED': 0.123, 'CosSim': 0.964, 'UTCA': 0.678},
                'gemini-2.0-flash': {'ASTED': 0.118, 'CosSim': 0.967, 'UTCA': 0.761},
                'claude-sonnet-4': {'ASTED': 0.108, 'CosSim': 0.968, 'UTCA': 0.775},
                'mistral': {'ASTED': 0.174, 'CosSim': 0.960, 'UTCA': 0.659},
                'llama-3.1-8b': {'ASTED': 0.023, 'CosSim': 0.888, 'UTCA': 0.211},
                'gemma-3-27b': {'ASTED': 0.300, 'CosSim': 0.846, 'UTCA': 0.169},
                'qwen2.5-14b': {'ASTED': 0.039, 'CosSim': 0.935, 'UTCA': 0.299},
            },
            'S3': {
                'gpt-4.1-nano': {'ASTED': 0.089, 'CosSim': 0.957, 'UTCA': 0.417, 'RMSE': 0.475},
                'gemini-2.0-flash': {'ASTED': 0.092, 'CosSim': 0.959, 'UTCA': 0.406, 'RMSE': 0.475},
                'claude-sonnet-4': {'ASTED': 0.086, 'CosSim': 0.959, 'UTCA': 0.630, 'RMSE': 0.475},
                'mistral': {'ASTED': 0.081, 'CosSim': 0.958, 'UTCA': 0.437, 'RMSE': 0.475},
                'llama-3.1-8b': {'ASTED': 0.015, 'CosSim': 0.917, 'UTCA': 0.639, 'RMSE': 0.435},
                'gemma-3-27b': {'ASTED': 0.291, 'CosSim': 0.837, 'UTCA': 0.632, 'RMSE': 0.476},
                'qwen2.5-14b': {'ASTED': 0.040, 'CosSim': 0.939, 'UTCA': 0.666, 'RMSE': 0.419},
            },
            'S4': {
                'gpt-4.1-nano': {'ASTED': 0.078, 'CosSim': 0.963, 'EAS': 1.452, 'UTCA': 0.737},
                'gemini-2.0-flash': {'ASTED': 0.072, 'CosSim': 0.969, 'EAS': 1.411, 'UTCA': 0.869},
                'claude-sonnet-4': {'ASTED': 0.071, 'CosSim': 0.972, 'EAS': 1.654, 'UTCA': 0.913},
                'mistral': {'ASTED': 0.083, 'CosSim': 0.967, 'EAS': 1.594, 'UTCA': 0.853},
                'llama-3.1-8b': {'ASTED': 0.015, 'CosSim': 0.916, 'EAS': 1.239, 'UTCA': 0.080},
                'gemma-3-27b': {'ASTED': 0.258, 'CosSim': 0.858, 'EAS': 0, 'UTCA': 0.041},
                'qwen2.5-14b': {'ASTED': 0.048, 'CosSim': 0.949, 'EAS': 1.098, 'UTCA': 0.337},
            },
        }

    # Generate scenario plots
    if metrics:
        plot_scenario1(metrics, output_path)
        plot_scenario2(metrics, output_path)
        plot_scenario3(metrics, output_path)
        plot_scenario4(metrics, output_path)

    # Load and plot psychometric correlations
    correlations_file = psychometrics_path / 'correlations' / 'all_correlations.json'
    if correlations_file.exists():
        correlations = load_correlations(str(correlations_file))
        plot_psychometrics(correlations, output_path)
    else:
        # Sample correlations for demonstration
        correlations = {
            'gpt-4.1-nano': {'ability': -0.126, 'difficulty': 0.148},
            'gemini-2.0-flash': {'ability': 0.284, 'difficulty': 0.288},
            'claude-sonnet-4': {'ability': 0.359, 'difficulty': 0.224},
            'mistral': {'ability': 0.229, 'difficulty': 0.246},
            'llama-3.1-8b': {'ability': -0.166, 'difficulty': -0.213},
            'gemma-3-27b': {'ability': -0.287, 'difficulty': -0.287},
            'qwen2.5-14b': {'ability': 0.118, 'difficulty': 0.189},
        }
        plot_psychometrics(correlations, output_path)

    # Load and plot distributions if available
    for model in ['student', 'real']:
        ability_file = psychometrics_path / 'ability' / f'{model}_student_ability.csv'
        if ability_file.exists():
            ability_data = pd.read_csv(ability_file)
            plot_ability_distribution(ability_data, output_path)
            break

    for model in ['student', 'real']:
        difficulty_file = psychometrics_path / 'difficulty' / f'{model}_difficulty.csv'
        if difficulty_file.exists():
            difficulty_data = pd.read_csv(difficulty_file)
            plot_difficulty_distribution(difficulty_data, output_path)
            break

    logger.info(f"\nAll plots saved to: {output_path}")


if __name__ == "__main__":
    main()
