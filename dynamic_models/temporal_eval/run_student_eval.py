"""Student-based evaluation: calibrate on train students, score test students.

Quality-filters students/questions across all weeks, then splits students
70/30. Train students calibrate item parameters; test students' weeks 1-3
data estimates ability; predictions are on test students' weeks 4-6.

Usage:
    python -m dynamic_models.temporal_eval.run_student_eval
    python -m dynamic_models.temporal_eval.run_student_eval --models IRT BKT DKT
"""

import argparse
import os
import pickle
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

from dynamic_models.temporal_eval.data_loader import load_student_split_data
from dynamic_models.temporal_eval.harness import get_adapter_registry
from dynamic_models.temporal_eval.metrics import compute_metrics


def run_student_evaluation(
    course_name: str = "dsa_hk231",
    models=None,
    seed: int = 42,
    output_dir: str = "results/student_eval",
    max_attempts: int = 10,
    test_frac: float = 0.3,
    train_week_cutoff: int = 3,
):
    data, split = load_student_split_data(
        course_name=course_name,
        max_attempts=max_attempts,
        test_frac=test_frac,
        train_week_cutoff=train_week_cutoff,
        seed=seed,
    )

    # Build adapters
    all_adapters = get_adapter_registry()
    if models is not None:
        adapters = {}
        for m in models:
            if m in all_adapters:
                adapters[m] = all_adapters[m]
            else:
                print(f"  WARNING: Unknown model '{m}'. "
                      f"Available: {list(all_adapters.keys())}")
    else:
        adapters = all_adapters

    # Train and evaluate
    os.makedirs(output_dir, exist_ok=True)
    results_rows = []
    predictions = {}

    for model_name, adapter in adapters.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print("=" * 60)

        t0 = time.time()
        try:
            prediction = adapter.fit_and_predict_student_split(
                data, split, seed=seed
            )
            metrics = compute_metrics(prediction.y_true, prediction.y_pred_prob)
            runtime = time.time() - t0

            predictions[model_name] = prediction

            print(f"  AUC={metrics.auc:.4f}  "
                  f"Acc={metrics.accuracy:.4f}  "
                  f"F1={metrics.f1:.4f}  "
                  f"LL={metrics.log_likelihood:.4f}  "
                  f"RMSE={metrics.rmse:.4f}  "
                  f"N={metrics.n_test_obs}  "
                  f"({runtime:.1f}s)")

            for metric_name, value in metrics.to_dict().items():
                if metric_name == "n_test_obs":
                    continue
                results_rows.append({
                    "course": course_name,
                    "model": model_name,
                    "metric": metric_name,
                    "value": float(value),
                    "n_test_obs": int(metrics.n_test_obs),
                    "runtime_seconds": float(runtime),
                })
        except NotImplementedError as e:
            print(f"  SKIPPED: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        # Save predictions
        if model_name in predictions:
            pred = predictions[model_name]
            pred_path = os.path.join(output_dir, f"{model_name}_student_pred.pkl")
            with open(pred_path, "wb") as f:
                pickle.dump(pred, f)

    results_df = pd.DataFrame(results_rows)
    if len(results_df) > 0:
        csv_path = os.path.join(output_dir, "student_eval.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved: {csv_path}")

    return results_df, predictions, data


def main():
    parser = argparse.ArgumentParser(description="Student-based evaluation")
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        REPO_ROOT, "results", "student_eval", args.course_name
    )

    run_student_evaluation(
        course_name=args.course_name,
        models=args.models,
        seed=args.seed,
        output_dir=output_dir,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
