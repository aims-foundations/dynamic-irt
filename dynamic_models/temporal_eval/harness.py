"""Evaluation harness: run all models across all temporal horizons."""

import json
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .base_adapter import ModelAdapter, PredictionResult
from .data_filter import DataFilterConfig, apply_filter
from .data_loader import UnifiedData, load_unified_data
from .metrics import compute_metrics
from .temporal_split import generate_temporal_splits


def get_adapter_registry() -> Dict[str, ModelAdapter]:
    """Build the full registry of model adapters."""
    from .adapters import (
        BKTAdapter,
        CIRTAdapter,
        CIRTDecayAdapter,
        DKTAdapter,
        DynamicIRTAdapter,
        EloAdapter,
        GPIRTAdapter,
        IRTAdapter,
        LLMAdapter,
        RSSMAdapter,
        RSSMFullAdapter,
    )
    return {
        "BKT": BKTAdapter(),
        "DKT": DKTAdapter(),
        "Elo": EloAdapter(),
        "IRT": IRTAdapter(),
        "CIRT": CIRTAdapter(),
        "CIRT-Decay": CIRTDecayAdapter(),
        "DynamicIRT": DynamicIRTAdapter(),
        "GPIRT": GPIRTAdapter(),
        "LLM": LLMAdapter(),
        "RSSM": RSSMAdapter(),
        "RSSMFull": RSSMFullAdapter(),
    }


ALL_COURSES = ["dsa_hk231", "dsa_hk221", "pf_hk232", "pf_hk222"]


def _apply_index_filter(data: UnifiedData, student_idx, item_idx) -> UnifiedData:
    """Return a new UnifiedData containing only the selected students and items."""
    import numpy as np

    corr = data.correctness_matrix[student_idx][:, item_idx]
    time = data.time_matrix[student_idx][:, item_idx]
    qi = data.question_infos.iloc[item_idx].reset_index(drop=True)
    item_week = data.item_week[item_idx]
    student_ids = [data.student_ids[i] for i in student_idx]

    # Filter main_data to keep only selected students and items
    sid_set = set(student_ids)
    # Map item indices to question_unittest_ids for CSV filtering
    if "question_unittest_id" in data.main_data.columns:
        kept_qids = set()
        for idx in item_idx:
            row = data.question_infos.iloc[idx]
            if "question_unittest_id" in row.index:
                kept_qids.add(int(row["question_unittest_id"]))
        main_data = data.main_data[
            data.main_data["student_id"].isin(sid_set)
        ].copy()
        if kept_qids:
            main_data = main_data[
                main_data["question_unittest_id"].isin(kept_qids)
            ].copy()
    else:
        main_data = data.main_data[
            data.main_data["student_id"].isin(sid_set)
        ].copy()

    n_s, n_i, n_a = corr.shape
    return UnifiedData(
        main_data=main_data,
        correctness_matrix=corr,
        time_matrix=time,
        question_infos=qi,
        item_week=item_week,
        qid_to_week=data.qid_to_week,
        student_ids=student_ids,
        n_students=n_s,
        n_items=n_i,
        n_max_attempts=n_a,
        course_name=data.course_name,
    )


def run_temporal_evaluation(
    course_name: str = "all",
    cutoff_weeks: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    seed: int = 42,
    output_dir: str = "results/temporal_eval",
    skip_slow: bool = False,
    slow_threshold_minutes: float = 60.0,
    data_filter: Optional[DataFilterConfig] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], PredictionResult], UnifiedData]:
    """Run all models across all temporal horizons.

    Args:
        course_name: Course to evaluate on, or "all" for every course.
        cutoff_weeks: List of cutoff week values. None = auto-derive.
        models: List of model names to include. None = all.
        seed: Random seed.
        output_dir: Where to save results.
        skip_slow: If True, skip models > slow_threshold_minutes per horizon.
        slow_threshold_minutes: Threshold for --skip_slow.
        data_filter: Optional DataFilterConfig to filter students/questions.

    Returns:
        Tuple of:
        - DataFrame with columns: model, horizon, metric, value,
          n_test_obs, runtime_seconds, course.
        - Dict mapping (course, model_name, horizon) to PredictionResult.
        - UnifiedData from the last course evaluated.
    """
    if course_name == "all":
        all_results = []
        all_predictions = {}
        last_data = None
        for course in ALL_COURSES:
            print(f"\n{'#' * 60}")
            print(f"# COURSE: {course}")
            print(f"{'#' * 60}")
            course_dir = os.path.join(output_dir, course)
            df, preds, data = run_temporal_evaluation(
                course_name=course,
                cutoff_weeks=cutoff_weeks,
                models=models,
                seed=seed,
                output_dir=course_dir,
                skip_slow=skip_slow,
                slow_threshold_minutes=slow_threshold_minutes,
                data_filter=data_filter,
            )
            df["course"] = course
            all_results.append(df)
            for k, v in preds.items():
                all_predictions[(course, *k)] = v
            last_data = data
        combined_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        if len(combined_df) > 0:
            combined_csv = os.path.join(output_dir, "temporal_eval_all.csv")
            os.makedirs(output_dir, exist_ok=True)
            combined_df.to_csv(combined_csv, index=False)
            print(f"\nCombined results saved to {combined_csv}")
        return combined_df, all_predictions, last_data

    # 1. Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    data = load_unified_data(course_name)

    if data_filter is not None:
        student_idx, item_idx, selected_qs = apply_filter(
            data.correctness_matrix, data.question_infos, data_filter
        )
        print(f"\nData filter applied: {len(student_idx)} students, "
              f"{len(item_idx)} items, {len(selected_qs)} questions")
        data = _apply_index_filter(data, student_idx, item_idx)

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
    os.makedirs(output_dir, exist_ok=True)
    results_rows = []
    predictions = {}

    for model_name, adapter in adapters.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name} "
              f"(~{adapter.estimated_runtime_minutes(data):.0f} min/horizon)")
        print("=" * 60)

        model_rows = []
        model_preds = {}

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

                model_preds[int(split.cutoff_week)] = prediction

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
                    model_rows.append({
                        "course": course_name,
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

        # Save per-model immediately
        if model_rows:
            model_csv = os.path.join(output_dir, f"{model_name}.csv")
            pd.DataFrame(model_rows).to_csv(model_csv, index=False)
            print(f"  Metrics saved: {model_csv}")

        if model_preds:
            pred_path = os.path.join(output_dir, f"{model_name}_predictions.pkl")
            with open(pred_path, "wb") as f:
                pickle.dump(model_preds, f)

            for horizon, pred in model_preds.items():
                if pred.model_state is not None:
                    model_path = os.path.join(output_dir, f"{model_name}_W{horizon}.pkl")
                    with open(model_path, "wb") as f:
                        pickle.dump(pred.model_state, f)
            print(f"  Model states saved: {output_dir}/{model_name}_W*.pkl")

        results_rows.extend(model_rows)
        for h, p in model_preds.items():
            predictions[(model_name, h)] = p

    # 5. Combined results
    results_df = pd.DataFrame(results_rows)

    return results_df, predictions, data


def load_saved_results(
    course_name: str = "all",
    output_dir: str = "results/temporal_eval",
    models: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], PredictionResult]]:
    """Load saved predictions and metrics from a previous run.

    Args:
        course_name: Course to load, or "all" for every course subdirectory.
        output_dir: Root results directory.
        models: Model names to load. None = all available.

    Returns:
        Tuple of (results_df, predictions_dict).
    """
    if course_name == "all":
        all_dfs = []
        all_preds = {}
        for course in ALL_COURSES:
            course_dir = os.path.join(output_dir, course)
            if not os.path.isdir(course_dir):
                continue
            df, preds = load_saved_results(course, output_dir, models)
            if len(df) > 0:
                all_dfs.append(df)
            for k, v in preds.items():
                all_preds[k] = v
        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        return combined, all_preds

    course_dir = os.path.join(output_dir, course_name) if course_name != "all" else output_dir
    results_dfs = []
    predictions = {}

    if not os.path.isdir(course_dir):
        print(f"No saved results at {course_dir}")
        return pd.DataFrame(), {}

    # Discover available models from prediction pickle files
    for fname in sorted(os.listdir(course_dir)):
        if not fname.endswith("_predictions.pkl"):
            continue
        model_name = fname.replace("_predictions.pkl", "")
        if models is not None and model_name not in models:
            continue

        pred_path = os.path.join(course_dir, fname)
        with open(pred_path, "rb") as f:
            model_preds = pickle.load(f)  # dict[horizon -> PredictionResult]
        for horizon, pred in model_preds.items():
            predictions[(model_name, int(horizon))] = pred
        print(f"  Loaded {model_name}: {len(model_preds)} horizons")

        csv_path = os.path.join(course_dir, f"{model_name}.csv")
        if os.path.exists(csv_path):
            results_dfs.append(pd.read_csv(csv_path))

    results_df = pd.concat(results_dfs, ignore_index=True) if results_dfs else pd.DataFrame()
    return results_df, predictions
