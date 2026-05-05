"""Train and evaluate the full RSSM model.

Usage:
    python scripts/train_rssm_full.py                    # single course, W=1
    python scripts/train_rssm_full.py --all               # all courses, all weeks
    python scripts/train_rssm_full.py --beta 0.01 --lr 5e-4
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynamic_models.temporal_eval.harness import run_temporal_evaluation


def main():
    parser = argparse.ArgumentParser(description="Train full RSSM")
    parser.add_argument("--course", default="dsa_hk231")
    parser.add_argument("--weeks", type=int, nargs="+", default=[1])
    parser.add_argument("--all", action="store_true", help="Run all courses x all weeks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/temporal_eval")
    args = parser.parse_args()

    if args.all:
        course = "all"
        weeks = None
    else:
        course = args.course
        weeks = args.weeks

    df, preds, data = run_temporal_evaluation(
        course_name=course,
        cutoff_weeks=weeks,
        models=["RSSMFull"],
        seed=args.seed,
        output_dir=args.output_dir,
    )

    if len(df) > 0:
        auc_rows = df[df["metric"] == "auc"]
        for _, row in auc_rows.iterrows():
            print(f"  {row.get('course', course)} W={row['horizon']}: "
                  f"AUC={row['value']:.4f}")


if __name__ == "__main__":
    main()
