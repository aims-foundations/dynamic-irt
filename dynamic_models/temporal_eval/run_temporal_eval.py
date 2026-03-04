"""CLI entry point for temporal evaluation.

Usage:
    cd CodeInsights
    python -m dynamic_irt.temporal_eval.run_temporal_eval \
        --course_name dsa_hk231 \
        --models Elo CIRT DynamicIRT \
        --seed 42

    # Skip slow models (GPIRT):
    python -m dynamic_irt.temporal_eval.run_temporal_eval \
        --course_name dsa_hk231 --skip_slow

    # Specific horizons:
    python -m dynamic_irt.temporal_eval.run_temporal_eval \
        --course_name dsa_hk231 --cutoff_weeks 2 3 4
"""

import argparse
import os
import sys

# Ensure CodeInsights is on the path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from dynamic_irt.temporal_eval.harness import run_temporal_evaluation
from dynamic_irt.temporal_eval.plot_results import (
    plot_concept_pair_scatter,
    plot_metrics_vs_horizon,
    plot_student_trajectories,
    plot_summary_table,
)


def main():
    parser = argparse.ArgumentParser(
        description="Temporal evaluation of learning dynamics models"
    )
    parser.add_argument(
        "--course_name",
        type=str,
        default="dsa_hk231",
        help="Course name (dsa_hk231, dsa_hk221, pf_hk232, pf_hk222, all)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to evaluate (Elo, CIRT, DynamicIRT, GPIRT, RSSM). "
        "Default: all available.",
    )
    parser.add_argument(
        "--cutoff_weeks",
        type=int,
        nargs="+",
        default=None,
        help="Cutoff weeks for temporal splits. Default: auto-derive.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results/temporal_eval/)",
    )
    parser.add_argument(
        "--skip_slow",
        action="store_true",
        help="Skip models estimated to take >60 min per horizon (e.g., GPIRT)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        REPO_ROOT, "results", "temporal_eval"
    )

    results_df, predictions, data = run_temporal_evaluation(
        course_name=args.course_name,
        cutoff_weeks=args.cutoff_weeks,
        models=args.models,
        seed=args.seed,
        output_dir=output_dir,
        skip_slow=args.skip_slow,
    )

    if len(results_df) > 0 and not args.no_plot:
        print("\nGenerating plots...")
        plot_metrics_vs_horizon(results_df, output_dir)
        plot_summary_table(results_df, output_dir)
        plot_student_trajectories(predictions, data, output_dir)
        plot_concept_pair_scatter(predictions, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
