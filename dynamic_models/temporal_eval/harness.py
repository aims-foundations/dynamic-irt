"""Evaluation harness: run all models across all temporal horizons."""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .base_adapter import ModelAdapter, PredictionResult
from .data_loader import UnifiedData, load_unified_data
from .metrics import compute_metrics
from .temporal_split import generate_temporal_splits


def get_adapter_registry() -> Dict[str, ModelAdapter]:
    """Build the full registry of model adapters."""
    from .adapters import (
        CIRTAdapter,
        DynamicIRTAdapter,
        EloAdapter,
        GPIRTAdapter,
        RSSMAdapter,
    )

    return {
        "Elo": EloAdapter(),
        "CIRT": CIRTAdapter(),
        "DynamicIRT": DynamicIRTAdapter(),
        "GPIRT": GPIRTAdapter(),
        "RSSM": RSSMAdapter(),
    }


def run_temporal_evaluation(
    course_name: str = "dsa_hk231",
    cutoff_weeks: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    seed: int = 42,
    output_dir: str = "results/temporal_eval",
    skip_slow: bool = False,
    slow_threshold_minutes: float = 60.0,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], PredictionResult], UnifiedData]:
    """Run all models across all temporal horizons.

    Args:
        course_name: Course to evaluate on.
        cutoff_weeks: List of cutoff week values. None = auto-derive.
        models: List of model names to include. None = all.
        seed: Random seed.
        output_dir: Where to save results.
        skip_slow: If True, skip models > slow_threshold_minutes per horizon.
        slow_threshold_minutes: Threshold for --skip_slow.

    Returns:
        Tuple of:
        - DataFrame with columns: model, horizon, metric, value,
          n_test_obs, runtime_seconds.
        - Dict mapping (model_name, horizon) to PredictionResult.
        - UnifiedData used for evaluation.
    """
    # 1. Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    data = load_unified_data(course_name)

    # 2. Generate splits
    splits = generate_temporal_splits(data.item_week, cutoff_weeks)
    print(f"\n{len(splits)} temporal splits:")
    for s in splits:
        print(f"  {s}")

    # 3. Build adapter registry
    all_adapters = get_adapter_registry()

    if models is not None:
        adapters = {}
        for m in models:
            if m in all_adapters:
                adapters[m] = all_adapters[m]
            else:
                print(f"  WARNING: Unknown model '{m}', skipping. "
                      f"Available: {list(all_adapters.keys())}")
    else:
        adapters = all_adapters

    if skip_slow:
        adapters = {
            k: v
            for k, v in adapters.items()
            if v.estimated_runtime_minutes(data) < slow_threshold_minutes
        }
        print(f"\nAfter filtering slow models: {list(adapters.keys())}")

    # 4. Main loop
    results_rows = []
    predictions = {}  # (model_name, horizon) -> PredictionResult

    for model_name, adapter in adapters.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name} "
              f"(~{adapter.estimated_runtime_minutes(data):.0f} min/horizon)")
        print("=" * 60)

        for split in splits:
            print(f"\n  W={split.cutoff_week}: "
                  f"{split.n_train_items} train / {split.n_test_items} test items")

            t0 = time.time()
            try:
                prediction = adapter.fit_and_predict(
                    data, split, seed=seed
                )
                metrics = compute_metrics(
                    prediction.y_true, prediction.y_pred_prob
                )
                runtime = time.time() - t0

                predictions[(model_name, int(split.cutoff_week))] = prediction

                print(
                    f"    AUC={metrics.auc:.4f}  "
                    f"Acc={metrics.accuracy:.4f}  "
                    f"F1={metrics.f1:.4f}  "
                    f"LL={metrics.log_likelihood:.4f}  "
                    f"RMSE={metrics.rmse:.4f}  "
                    f"N={metrics.n_test_obs}  "
                    f"({runtime:.1f}s)"
                )

                for metric_name, value in metrics.to_dict().items():
                    if metric_name == "n_test_obs":
                        continue
                    results_rows.append({
                        "model": model_name,
                        "horizon": int(split.cutoff_week),
                        "metric": metric_name,
                        "value": float(value),
                        "n_test_obs": int(metrics.n_test_obs),
                        "runtime_seconds": float(runtime),
                    })
            except NotImplementedError as e:
                print(f"    SKIPPED: {e}")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # 5. Save results
    results_df = pd.DataFrame(results_rows)

    if len(results_df) > 0:
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(
            output_dir, f"temporal_eval_{course_name}.csv"
        )
        results_df.to_csv(csv_path, index=False)

        json_path = os.path.join(
            output_dir, f"temporal_eval_{course_name}.json"
        )
        with open(json_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "course_name": course_name,
                        "cutoff_weeks": [s.cutoff_week for s in splits],
                        "seed": seed,
                        "models": list(adapters.keys()),
                    },
                    "results": results_rows,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to {csv_path}")
    else:
        print("\nNo results to save.")

    return results_df, predictions, data
