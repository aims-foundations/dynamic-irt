"""Filtered temporal evaluation: single horizon determined by DataFilterConfig.

Applies student/question filters, then trains and evaluates models on the
filtered data with train=weeks 1..max_week, test=weeks max_week+1..end.

Usage:
    python -m dynamic_models.temporal_eval.run_filtered_eval
    python -m dynamic_models.temporal_eval.run_filtered_eval --course_name dsa_hk231
    python -m dynamic_models.temporal_eval.run_filtered_eval --models Elo IRT CIRT-Decay RSSM
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

from dynamic_models.temporal_eval.base_adapter import PredictionResult
from dynamic_models.temporal_eval.data_filter import DEFAULT_FILTER, DataFilterConfig, apply_filter
from dynamic_models.temporal_eval.data_loader import load_unified_data
from dynamic_models.temporal_eval.harness import _apply_index_filter, get_adapter_registry
from dynamic_models.temporal_eval.metrics import compute_metrics
from dynamic_models.temporal_eval.temporal_split import generate_temporal_splits


def run_filtered_evaluation(
    course_name: str = "dsa_hk231",
    models=None,
    seed: int = 42,
    output_dir: str = "results/filtered_eval",
    config: DataFilterConfig = None,
):
    if config is None:
        config = DEFAULT_FILTER

    print("=" * 60)
    print(f"LOADING DATA: {course_name}")
    print("=" * 60)
    data = load_unified_data(course_name)

    print("\nApplying data filter...")
    student_idx, train_item_idx, selected_qs = apply_filter(
        data.correctness_matrix, data.question_infos, config
    )

    # Test items: all items from weeks after max_week
    qi = data.question_infos
    test_item_idx = np.where(qi["week"].values > config.max_week)[0]

    # Combine filtered train items + all test items
    all_item_idx = np.unique(np.concatenate([train_item_idx, test_item_idx]))
    all_item_idx.sort()

    print(f"  {len(student_idx)} students (from {data.n_students}), "
          f"{len(train_item_idx)} train items ({len(selected_qs)} questions), "
          f"{len(test_item_idx)} test items")

    data = _apply_index_filter(data, student_idx, all_item_idx)

    # Single split at max_week
    splits = generate_temporal_splits(data.item_week, [config.max_week])
    if not splits:
        print("ERROR: No valid split produced. Check max_week vs available weeks.")
        return pd.DataFrame(), {}, data
    split = splits[0]
    print(f"\nSplit: {split}")

    # Build adapters
    all_adapters = get_adapter_registry()
    if models is not None:
        adapters = {}
        for m in models:
            if m in all_adapters:
                adapters[m] = all_adapters[m]
            else:
                print(f"  WARNING: Unknown model '{m}'. Available: {list(all_adapters.keys())}")
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
            prediction = adapter.fit_and_predict(data, split, seed=seed)
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

        # Save per-model
        if model_name in predictions:
            pred = predictions[model_name]
            if pred.model_state is not None:
                model_path = os.path.join(output_dir, f"{model_name}_W{config.max_week}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(pred.model_state, f)

    results_df = pd.DataFrame(results_rows)
    if len(results_df) > 0:
        csv_path = os.path.join(output_dir, "filtered_eval.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved: {csv_path}")

    return results_df, predictions, data


def main():
    parser = argparse.ArgumentParser(description="Filtered temporal evaluation")
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_slow", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        REPO_ROOT, "results", "filtered_eval", args.course_name
    )

    config = DEFAULT_FILTER

    results_df, predictions, data = run_filtered_evaluation(
        course_name=args.course_name,
        models=args.models,
        seed=args.seed,
        output_dir=output_dir,
        config=config,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
