"""CLI entry point for temporal evaluation.

Usage:
    cd CodeInsights
    python -m dynamic_irt.temporal_eval.run_temporal_eval
    python -m dynamic_irt.temporal_eval.run_temporal_eval --course_name dsa_hk231
    python -m dynamic_irt.temporal_eval.run_temporal_eval --models Elo CIRT DynamicIRT RSSM
    python -m dynamic_irt.temporal_eval.run_temporal_eval --load_only --models Elo
    python -m dynamic_irt.temporal_eval.run_temporal_eval --load_only --course_name dsa_hk231
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from dynamic_models.temporal_eval.harness import load_saved_results, run_temporal_evaluation
from dynamic_models.temporal_eval.plot_results import (
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
        default="all",
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
        "--load_only",
        action="store_true",
        help="Skip training; load saved results and run analysis.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        REPO_ROOT, "results", "temporal_eval"
    )

    if args.load_only:
        print("Loading saved results...")
        results_df, predictions = load_saved_results(
            course_name=args.course_name,
            output_dir=output_dir,
            models=args.models,
        )
        if len(results_df) == 0 and not predictions:
            print("No saved results found.")
            return
        data = None
    else:
        results_df, predictions, data = run_temporal_evaluation(
            course_name=args.course_name,
            cutoff_weeks=args.cutoff_weeks,
            models=args.models,
            seed=args.seed,
            output_dir=output_dir,
            skip_slow=args.skip_slow,
        )

    if len(results_df) > 0:
        print("\nGenerating analysis...")
        plot_metrics_vs_horizon(results_df, output_dir)
        plot_summary_table(results_df, output_dir)

    if predictions:
        # When course_name="all", keys are (course, model, horizon) 3-tuples.
        # Plot functions expect (model, horizon) 2-tuples, so plot per course.
        sample_key = next(iter(predictions))
        if len(sample_key) == 3:
            courses = sorted({k[0] for k in predictions})
            for course in courses:
                course_preds = {
                    (k[1], k[2]): v for k, v in predictions.items() if k[0] == course
                }
                course_dir = os.path.join(output_dir, course)
                plot_concept_pair_scatter(course_preds, course_dir)
        else:
            if data is not None:
                plot_student_trajectories(predictions, data, output_dir)
            plot_concept_pair_scatter(predictions, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
