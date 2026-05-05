"""Analyze per-student prediction std across temporal horizons."""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamic_models.temporal_eval.harness import load_saved_results

results_df, predictions = load_saved_results("all", "results/temporal_eval")

TARGET_MODELS = ["Elo", "CIRT", "DynamicIRT", "RSSM"]

rows = []
for (course, model, horizon), pred in sorted(predictions.items()):
    if model not in TARGET_MODELS:
        continue
    if pred.student_indices is None or pred.y_pred_prob is None:
        continue

    unique_students = np.unique(pred.student_indices)
    student_stds = []
    for s in unique_students:
        mask = pred.student_indices == s
        preds_s = pred.y_pred_prob[mask]
        if len(preds_s) >= 2:
            student_stds.append(np.std(preds_s))

    if student_stds:
        rows.append({
            "course": course,
            "model": model,
            "horizon": horizon,
            "mean_std": np.mean(student_stds),
            "median_std": np.median(student_stds),
            "n_students": len(student_stds),
        })

df = pd.DataFrame(rows)

# Aggregate across courses
agg = df.groupby(["model", "horizon"]).agg(
    mean_std=("mean_std", "mean"),
    median_std=("median_std", "mean"),
    n_students=("n_students", "sum"),
).reset_index()

agg = agg.sort_values(["model", "horizon"])

print("\n" + "=" * 65)
print("Per-Student Prediction Std by Model and Horizon (across courses)")
print("=" * 65)
print(agg.to_string(index=False, float_format="%.6f"))

out_path = "results/temporal_eval/psychometric/prediction_std_by_horizon.csv"
agg.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# Also print per-course detail
print("\n" + "=" * 65)
print("Per-Course Detail")
print("=" * 65)
print(df.to_string(index=False, float_format="%.6f"))
