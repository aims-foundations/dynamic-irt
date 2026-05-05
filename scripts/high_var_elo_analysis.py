"""Analyze high-variance Elo students across models and horizons."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from dynamic_models.temporal_eval.harness import load_saved_results
from dynamic_models.temporal_eval.data_loader import load_unified_data

results_df, predictions = load_saved_results("all", "results/temporal_eval")

courses = sorted(set(k[0] for k in predictions))
models_avail = sorted(set(k[1] for k in predictions))
horizons = sorted(set(k[2] for k in predictions))
print(f"Courses: {courses}")
print(f"Models: {models_avail}")
print(f"Horizons: {horizons}")

COURSE = "dsa_hk231"
HORIZON_FIRST = min(horizons)
MODELS = ["Elo", "CIRT", "DynamicIRT", "RSSM"]

# Load data for student ID mapping
data = load_unified_data(COURSE)
# student_ids[matrix_idx] = raw_student_id
raw_to_matrix = {sid: idx for idx, sid in enumerate(data.student_ids)}

# Step 1: Find high-variance Elo students at earliest horizon
key = (COURSE, "Elo", HORIZON_FIRST)
pred = predictions[key]
y_true = pred.y_true
y_prob = pred.y_pred_prob
students = pred.student_indices  # raw student IDs for Elo

unique_students = np.unique(students)
print(f"\n{COURSE}, Elo, horizon={HORIZON_FIRST}: {len(y_true)} obs, {len(unique_students)} students")

student_stds = {}
for s in unique_students:
    mask = students == s
    student_stds[s] = np.std(y_prob[mask])

std_series = pd.Series(student_stds).sort_values(ascending=False)
print("\nPrediction std distribution under Elo:")
print(std_series.describe())
print(f"\nTop 10 students by Elo prediction std:")
print(std_series.head(10))

top5_raw = std_series.head(5).index.tolist()
print(f"\nSelected top 5 high-variance students (raw IDs): {top5_raw}")

# Map to matrix indices for CIRT/DynamicIRT/RSSM
top5_matrix = {raw: raw_to_matrix.get(raw) for raw in top5_raw}
print(f"Mapping to matrix indices: {top5_matrix}")


def get_student_id(model, raw_sid):
    """Return the student index used by a given model."""
    if model == "Elo":
        return raw_sid
    return top5_matrix[raw_sid]


# Step 2: Compute per-student LL and AUC across all models/horizons
rows = []
for raw_sid in top5_raw:
    for model in MODELS:
        for h in horizons:
            key = (COURSE, model, h)
            if key not in predictions:
                continue
            p = predictions[key]
            sid = get_student_id(model, raw_sid)
            if sid is None:
                continue
            mask = p.student_indices == sid
            n = mask.sum()
            if n == 0:
                continue
            yt = p.y_true[mask]
            yp = np.clip(p.y_pred_prob[mask], 1e-15, 1 - 1e-15)

            ll = np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))

            if len(np.unique(yt)) >= 2:
                auc = roc_auc_score(yt, yp)
            else:
                auc = np.nan

            rows.append({
                "student_id": raw_sid,
                "model": model,
                "horizon": h,
                "student_LL": round(ll, 4),
                "student_AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
                "n_items": n,
            })

df = pd.DataFrame(rows)
print("\n" + "=" * 80)
print("Per-student results for top-5 high-variance Elo students")
print("=" * 80)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 140)
print(df.to_string(index=False))

# Step 3: Compare models for these students
print("\n" + "=" * 80)
print("Model comparison for high-variance Elo students (mean across horizons)")
print("=" * 80)

winner_counts = {m: 0 for m in MODELS}
for s in top5_raw:
    sdf = df[df.student_id == s]
    print(f"\nStudent {s} (Elo pred std = {std_series[s]:.4f}):")

    model_lls = {}
    model_aucs = {}
    for model in MODELS:
        mdf = sdf[sdf.model == model]
        if len(mdf) == 0:
            continue
        model_lls[model] = mdf["student_LL"].mean()
        valid_auc = mdf[mdf.student_AUC.notna()]["student_AUC"]
        model_aucs[model] = valid_auc.mean() if len(valid_auc) > 0 else np.nan

    ll_str = ", ".join(f"{m}: {v:.4f}" for m, v in model_lls.items())
    auc_str = ", ".join(f"{m}: {v:.4f}" for m, v in model_aucs.items() if not np.isnan(v))
    best_ll = max(model_lls, key=model_lls.get) if model_lls else "N/A"
    best_auc = max(
        ((m, v) for m, v in model_aucs.items() if not np.isnan(v)),
        key=lambda x: x[1], default=(None, None)
    )
    winner_counts[best_ll] = winner_counts.get(best_ll, 0) + 1

    print(f"  Mean LL  -> {ll_str}")
    print(f"  Mean AUC -> {auc_str}")
    print(f"  Best model by LL: {best_ll}, by AUC: {best_auc[0] if best_auc[0] else 'N/A'}")

print("\n" + "-" * 40)
print("Summary: Best model (by LL) counts across 5 students:")
for m, c in sorted(winner_counts.items(), key=lambda x: -x[1]):
    if c > 0:
        print(f"  {m}: {c} students")

# Step 4: Save
out_dir = "results/temporal_eval/psychometric"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "high_var_elo_students.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
