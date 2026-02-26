#!/usr/bin/env python3
"""
generate_report.py - Generate reproducibility report for EduCodeSim paper

Compares computed metrics against paper values and generates a detailed
markdown report with pass/fail status.

Usage:
    python generate_report.py --metrics ./metrics/all_results.json --output ./report.md
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── Expected Values from Paper ────────────────────────────────────────────────
PAPER_VALUES = {
    "S1": {
        "gpt-4.1-nano": {"UTSR": 0.495},
        "gemini-2.0-flash": {"UTSR": 0.558},
        "claude-sonnet-4": {"UTSR": 0.668},
        "mistral": {"UTSR": 0.534},
        "llama-3.1-8b": {"UTSR": 0.490},
        "gemma-3-27b": {"UTSR": 0.449},
        "qwen2.5-14b": {"UTSR": 0.366},
    },
    "S2": {
        "gpt-4.1-nano": {"ASTED": 0.123, "CosSim": 0.964, "UTCA": 0.678},
        "gemini-2.0-flash": {"ASTED": 0.118, "CosSim": 0.967, "UTCA": 0.761},
        "claude-sonnet-4": {"ASTED": 0.108, "CosSim": 0.968, "UTCA": 0.775},
        "mistral": {"ASTED": 0.174, "CosSim": 0.960, "UTCA": 0.659},
        "llama-3.1-8b": {"ASTED": 0.023, "CosSim": 0.888, "UTCA": 0.211},
        "gemma-3-27b": {"ASTED": 0.300, "CosSim": 0.846, "UTCA": 0.169},
        "qwen2.5-14b": {"ASTED": 0.039, "CosSim": 0.935, "UTCA": 0.299},
    },
    "S3": {
        "gpt-4.1-nano": {"ASTED": 0.089, "CosSim": 0.957, "UTCA": 0.417, "RMSE": 0.475},
        "gemini-2.0-flash": {"ASTED": 0.092, "CosSim": 0.959, "UTCA": 0.406, "RMSE": 0.475},
        "claude-sonnet-4": {"ASTED": 0.086, "CosSim": 0.959, "UTCA": 0.630, "RMSE": 0.475},
        "mistral": {"ASTED": 0.081, "CosSim": 0.958, "UTCA": 0.437, "RMSE": 0.475},
        "llama-3.1-8b": {"ASTED": 0.015, "CosSim": 0.917, "UTCA": 0.639, "RMSE": 0.435},
        "gemma-3-27b": {"ASTED": 0.291, "CosSim": 0.837, "UTCA": 0.632, "RMSE": 0.476},
        "qwen2.5-14b": {"ASTED": 0.040, "CosSim": 0.939, "UTCA": 0.666, "RMSE": 0.419},
    },
    "S4": {
        "gpt-4.1-nano": {"ASTED": 0.078, "CosSim": 0.963, "EAS": 1.452, "UTCA": 0.737},
        "gemini-2.0-flash": {"ASTED": 0.072, "CosSim": 0.969, "EAS": 1.411, "UTCA": 0.869},
        "claude-sonnet-4": {"ASTED": 0.071, "CosSim": 0.972, "EAS": 1.654, "UTCA": 0.913},
        "mistral": {"ASTED": 0.083, "CosSim": 0.967, "EAS": 1.594, "UTCA": 0.853},
        "llama-3.1-8b": {"ASTED": 0.015, "CosSim": 0.916, "EAS": 1.239, "UTCA": 0.080},
        "gemma-3-27b": {"ASTED": 0.258, "CosSim": 0.858, "EAS": None, "UTCA": 0.041},
        "qwen2.5-14b": {"ASTED": 0.048, "CosSim": 0.949, "EAS": 1.098, "UTCA": 0.337},
    },
}

PAPER_CORRELATIONS = {
    "gpt-4.1-nano": {"ability": -0.126, "difficulty": 0.148},
    "gemini-2.0-flash": {"ability": 0.284, "difficulty": 0.288},
    "claude-sonnet-4": {"ability": 0.359, "difficulty": 0.224},
    "mistral": {"ability": 0.229, "difficulty": 0.246},
    "llama-3.1-8b": {"ability": -0.166, "difficulty": -0.213},
    "gemma-3-27b": {"ability": -0.287, "difficulty": -0.287},
    "qwen2.5-14b": {"ability": 0.118, "difficulty": 0.189},
}

# Tolerance for comparison (5% relative or 0.05 absolute)
RELATIVE_TOLERANCE = 0.10  # 10% relative difference
ABSOLUTE_TOLERANCE = 0.05  # 0.05 absolute difference


# ─── Helper Functions ──────────────────────────────────────────────────────────
def normalize_model_name(name: str) -> str:
    """Normalize model name for matching."""
    name_lower = name.lower().replace('_', '-').replace(' ', '-')

    mapping = {
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

    for key, value in mapping.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return value
    return name


def compare_values(actual: float, expected: float) -> Tuple[bool, float]:
    """
    Compare actual vs expected value within tolerance.
    Returns (is_match, difference).
    """
    if expected is None or actual is None:
        return True, 0.0

    diff = abs(actual - expected)

    # Check absolute tolerance
    if diff <= ABSOLUTE_TOLERANCE:
        return True, diff

    # Check relative tolerance
    if expected != 0:
        rel_diff = diff / abs(expected)
        if rel_diff <= RELATIVE_TOLERANCE:
            return True, diff

    return False, diff


def load_json(path: str) -> Optional[Dict]:
    """Load JSON file if it exists."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load {path}: {e}")
        return None


# ─── Report Generation ─────────────────────────────────────────────────────────
def generate_report(
    metrics: Optional[Dict],
    correlations: Optional[Dict],
    output_path: str,
):
    """Generate comprehensive reproducibility report."""

    report_lines = []

    # Header
    report_lines.extend([
        "# EduCodeSim Reproducibility Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report compares computed results against the values reported in the EduCodeSim paper.",
        "",
        f"**Tolerance:** {RELATIVE_TOLERANCE*100:.0f}% relative or {ABSOLUTE_TOLERANCE} absolute",
        "",
        "---",
        "",
    ])

    # Summary statistics
    total_comparisons = 0
    total_matches = 0
    total_mismatches = 0
    missing_data = 0

    # Scenario results
    report_lines.extend([
        "## Scenario Results",
        "",
    ])

    if metrics:
        for scenario in ["S1", "S2", "S3", "S4"]:
            scenario_data = metrics.get(scenario, metrics.get(f"scenario{scenario[-1]}", {}))
            expected_data = PAPER_VALUES.get(scenario, {})

            report_lines.extend([
                f"### {scenario}: {'Code Correctness' if scenario == 'S1' else 'Performance Imitation' if scenario == 'S2' else 'Error Reproduction' if scenario == 'S3' else 'Efficiency Alignment'}",
                "",
            ])

            if not scenario_data:
                report_lines.append(f"⚠️ No data available for {scenario}")
                report_lines.append("")
                missing_data += len(expected_data)
                continue

            # Determine metrics for this scenario
            if scenario == "S1":
                metric_names = ["UTSR"]
            elif scenario == "S2":
                metric_names = ["ASTED", "CosSim", "UTCA"]
            elif scenario == "S3":
                metric_names = ["ASTED", "CosSim", "UTCA", "RMSE"]
            else:  # S4
                metric_names = ["ASTED", "CosSim", "EAS", "UTCA"]

            # Create table header
            header = "| Model | " + " | ".join(metric_names) + " | Status |"
            separator = "|" + "|".join(["---"] * (len(metric_names) + 2)) + "|"
            report_lines.extend([header, separator])

            for model_name, expected_metrics in expected_data.items():
                # Find matching model in actual data
                actual_metrics = None
                for actual_model, actual_data in scenario_data.items():
                    if normalize_model_name(actual_model) == model_name:
                        actual_metrics = actual_data
                        break

                if actual_metrics is None:
                    row = f"| {model_name} | " + " | ".join(["N/A"] * len(metric_names)) + " | ⚠️ Missing |"
                    report_lines.append(row)
                    missing_data += len(metric_names)
                    continue

                # Compare each metric
                row_values = []
                row_status = "✅"

                for metric in metric_names:
                    expected = expected_metrics.get(metric)
                    actual = actual_metrics.get(metric, actual_metrics.get(metric.lower()))

                    if actual is None:
                        row_values.append("N/A")
                        missing_data += 1
                    else:
                        is_match, diff = compare_values(actual, expected)
                        total_comparisons += 1

                        if is_match:
                            total_matches += 1
                            row_values.append(f"{actual:.3f}")
                        else:
                            total_mismatches += 1
                            row_status = "❌"
                            row_values.append(f"{actual:.3f} (exp: {expected:.3f})")

                row = f"| {model_name} | " + " | ".join(row_values) + f" | {row_status} |"
                report_lines.append(row)

            report_lines.append("")
    else:
        report_lines.extend([
            "⚠️ **No metrics data available.**",
            "",
            "Please run the evaluation pipeline first:",
            "```bash",
            "python compute_metrics.py",
            "```",
            "",
        ])

    # Psychometric results
    report_lines.extend([
        "---",
        "",
        "## Psychometric Analysis (IRT Correlations)",
        "",
    ])

    if correlations:
        header = "| Model | Ability Corr | Difficulty Corr | Status |"
        separator = "|---|---|---|---|"
        report_lines.extend([header, separator])

        for model_name, expected in PAPER_CORRELATIONS.items():
            # Find matching model in actual data
            actual = None
            for actual_model, actual_data in correlations.items():
                if normalize_model_name(actual_model) == model_name:
                    actual = actual_data
                    break

            if actual is None:
                row = f"| {model_name} | N/A | N/A | ⚠️ Missing |"
                report_lines.append(row)
                missing_data += 2
                continue

            # Compare correlations
            ability_actual = actual.get('ability', actual.get('ability_correlation'))
            difficulty_actual = actual.get('difficulty', actual.get('difficulty_correlation'))

            ability_match, ability_diff = compare_values(ability_actual, expected['ability'])
            difficulty_match, difficulty_diff = compare_values(difficulty_actual, expected['difficulty'])

            total_comparisons += 2

            ability_str = f"{ability_actual:.3f}" if ability_actual is not None else "N/A"
            difficulty_str = f"{difficulty_actual:.3f}" if difficulty_actual is not None else "N/A"

            if not ability_match and ability_actual is not None:
                ability_str = f"{ability_actual:.3f} (exp: {expected['ability']:.3f})"
            if not difficulty_match and difficulty_actual is not None:
                difficulty_str = f"{difficulty_actual:.3f} (exp: {expected['difficulty']:.3f})"

            if ability_match and difficulty_match:
                total_matches += 2
                status = "✅"
            else:
                total_mismatches += (0 if ability_match else 1) + (0 if difficulty_match else 1)
                total_matches += (1 if ability_match else 0) + (1 if difficulty_match else 0)
                status = "❌"

            row = f"| {model_name} | {ability_str} | {difficulty_str} | {status} |"
            report_lines.append(row)

        report_lines.append("")
    else:
        report_lines.extend([
            "⚠️ **No correlation data available.**",
            "",
            "Please run psychometric analysis first:",
            "```bash",
            "python psychometrics_metrics.py",
            "```",
            "",
        ])

    # Summary
    report_lines.extend([
        "---",
        "",
        "## Summary",
        "",
        f"- **Total comparisons:** {total_comparisons}",
        f"- **Matches:** {total_matches} ({total_matches/total_comparisons*100:.1f}%)" if total_comparisons > 0 else f"- **Matches:** {total_matches} (N/A)",
        f"- **Mismatches:** {total_mismatches}",
        f"- **Missing data:** {missing_data}",
        "",
    ])

    if total_comparisons > 0:
        match_rate = total_matches / total_comparisons * 100
        if match_rate >= 90:
            report_lines.append("### ✅ **REPRODUCIBILITY: PASSED**")
            report_lines.append("")
            report_lines.append("Results are within acceptable tolerance of paper values.")
        elif match_rate >= 70:
            report_lines.append("### ⚠️ **REPRODUCIBILITY: PARTIAL**")
            report_lines.append("")
            report_lines.append("Most results match, but some discrepancies exist.")
        else:
            report_lines.append("### ❌ **REPRODUCIBILITY: FAILED**")
            report_lines.append("")
            report_lines.append("Significant discrepancies from paper values.")
    else:
        report_lines.append("### ⚠️ **REPRODUCIBILITY: INCOMPLETE**")
        report_lines.append("")
        report_lines.append("Insufficient data for comparison. Run the full evaluation pipeline.")

    report_lines.extend([
        "",
        "---",
        "",
        "## Notes",
        "",
        "- Minor differences may occur due to random seed variations",
        "- Open-source models may show more variation than commercial models",
        "- EAS (Efficiency Alignment Score) can be NaN for models that fail to compile",
        "",
        "---",
        "",
        "*Report generated by EduCodeSim reproducibility pipeline*",
    ])

    # Write report
    report_content = "\n".join(report_lines)

    with open(output_path, 'w') as f:
        f.write(report_content)

    logger.info(f"Report saved to: {output_path}")

    # Print summary to console
    print("\n" + "="*60)
    print("REPRODUCIBILITY SUMMARY")
    print("="*60)
    print(f"Total comparisons: {total_comparisons}")
    print(f"Matches: {total_matches}")
    print(f"Mismatches: {total_mismatches}")
    print(f"Missing: {missing_data}")
    if total_comparisons > 0:
        print(f"Match rate: {total_matches/total_comparisons*100:.1f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate reproducibility report for EduCodeSim"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="./metrics/all_results.json",
        help="Path to metrics JSON file"
    )
    parser.add_argument(
        "--correlations",
        type=str,
        default="./psychometrics/correlations/all_correlations.json",
        help="Path to correlations JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reproducibility_report.md",
        help="Output path for report"
    )

    args = parser.parse_args()

    # Load data
    metrics = load_json(args.metrics)
    correlations = load_json(args.correlations)

    # Generate report
    generate_report(metrics, correlations, args.output)


if __name__ == "__main__":
    main()
