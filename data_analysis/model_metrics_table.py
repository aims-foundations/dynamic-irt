"""Compute threshold-free and thresholded metrics from saved student-split predictions.

Loads results/student_eval/{course}/{model}_student_pred.pkl (no re-training)
and computes balanced accuracy, accuracy, AUC, log loss, and Brier score per
course/model, plus per-attempt accuracy, balanced accuracy, and AUC.

Models within a course must be evaluated on identical (student, item, attempt)
observation sets; the script errors otherwise. Pass --intersection to instead
subset every model to the shared keys.

Usage:
    python data_analysis/model_metrics_table.py [--intersection] [--write-tex]
        [--courses COURSE ...] [--models MODEL ...]

Outputs:
    results/student_eval/metrics_summary.csv
    results/student_eval/metrics_per_attempt.csv
    overleaf/tables/threshold_free_metrics.tex (only with --write-tex)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from data_analysis.llm_eval_common import (
    COURSE_TITLES,
    MODEL_DISPLAY_NAMES as MODEL_DISPLAY,
)
from data_analysis.plot_filtered_accuracy import load_prediction_result

COURSES = ["dsa_hk231", "dsa_hk221", "pf_hk232", "pf_hk222"]
MODELS = ["IRT", "CIRT", "BKT", "DKT", "CodeDKT", "RSSM"]

EPS = 1e-7
THRESHOLD = 0.5
MIN_ATTEMPT_OBS = 10


def compute_summary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)
    y_pred = (y_prob >= THRESHOLD).astype(int)
    log_loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    both = len(np.unique(y_true)) == 2
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if both else np.nan,
        "accuracy": (y_pred == y_true).mean(),
        "auc": roc_auc_score(y_true, y_prob) if both else np.nan,
        "log_loss": log_loss,
        "brier": brier_score_loss(y_true, y_prob),
        "n": len(y_true),
        "base_rate": y_true.mean(),
    }


def prediction_keys(pred):
    return set(zip(
        np.asarray(pred.student_indices).astype(int).tolist(),
        np.asarray(pred.item_indices).astype(int).tolist(),
        np.asarray(pred.attempt_indices).astype(int).tolist(),
    ))


def keys_mask(pred, keys):
    return np.array([
        k in keys
        for k in zip(
            np.asarray(pred.student_indices).astype(int).tolist(),
            np.asarray(pred.item_indices).astype(int).tolist(),
            np.asarray(pred.attempt_indices).astype(int).tolist(),
        )
    ])


def check_key_alignment(course, preds, intersection):
    """Verify all models share one observation set, or subset to shared keys."""
    key_sets = {model: prediction_keys(pred) for model, pred in preds.items()}
    shared = set.intersection(*key_sets.values())
    mismatched = {m: len(ks) for m, ks in key_sets.items() if ks != shared}
    if not mismatched:
        return None
    detail = ", ".join(
        f"{m}: n={n} (shared {len(shared)}, base rate "
        f"{np.asarray(preds[m].y_true).mean():.3f})"
        for m, n in sorted(mismatched.items())
    )
    if not intersection:
        raise ValueError(
            f"{course}: models are evaluated on different observation sets "
            f"({detail}). Rerun the offending models on the canonical data, "
            f"or pass --intersection to compare on shared keys."
        )
    print(f"  {course}: subsetting to {len(shared)} shared keys ({detail})")
    return shared


def compute_per_attempt_metrics(prediction, course, model, max_attempts=10, mask=None):
    y_true = np.asarray(prediction.y_true).astype(int)
    y_prob = np.clip(np.asarray(prediction.y_pred_prob, dtype=float), EPS, 1 - EPS)
    attempts = np.asarray(prediction.attempt_indices)
    if mask is not None:
        y_true, y_prob, attempts = y_true[mask], y_prob[mask], attempts[mask]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    rows = []
    for a in range(max_attempts):
        mask = attempts == a
        n = int(mask.sum())
        if n < MIN_ATTEMPT_OBS:
            continue
        yt, yp, pr = y_true[mask], y_pred[mask], y_prob[mask]
        both_classes = len(np.unique(yt)) == 2
        rows.append({
            "course": course,
            "model": model,
            "attempt": a + 1,
            "accuracy": (yp == yt).mean(),
            "balanced_accuracy": balanced_accuracy_score(yt, yp) if both_classes else np.nan,
            "auc": roc_auc_score(yt, pr) if both_classes else np.nan,
            "n": n,
        })
    return rows


def fmt_cell(value):
    return "--" if np.isnan(value) else f"{value:.3f}"


def to_latex_table(summary_df):
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\caption{Balanced accuracy and threshold-free metrics (AUC, log loss,"
        " Brier score) for the trained models on all four courses. Predictions are"
        " binarized at 0.5 for balanced accuracy. Higher is better for balanced"
        " accuracy and AUC; lower is better for log loss and Brier score.}",
        "  \\label{tab:threshold_free_metrics}",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\resizebox{\\columnwidth}{!}{%",
        "  \\begin{tabular}{llcccc}",
        "    \\toprule",
        "    Course & Model & Bal. Acc & AUC & Log loss & Brier \\\\",
    ]
    for course in COURSES:
        block = summary_df[summary_df["course"] == course]
        if block.empty:
            continue
        lines.append("    \\midrule")
        for i, (_, row) in enumerate(block.iterrows()):
            course_cell = COURSE_TITLES[course] if i == 0 else ""
            model_cell = MODEL_DISPLAY.get(row["model"], row["model"])
            lines.append(
                f"    {course_cell} & {model_cell} & {fmt_cell(row['balanced_accuracy'])}"
                f" & {fmt_cell(row['auc'])} & {fmt_cell(row['log_loss'])}"
                f" & {fmt_cell(row['brier'])} \\\\"
            )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", action="store_true",
                        help="Subset all models to shared observation keys "
                             "instead of erroring on mismatch")
    parser.add_argument("--write-tex", action="store_true",
                        help="Also overwrite overleaf/tables/threshold_free_metrics.tex")
    parser.add_argument("--courses", nargs="+", default=COURSES)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    summary_rows = []
    attempt_rows = []
    for course in args.courses:
        output_dir = f"results/student_eval/{course}"
        preds = {}
        for model in args.models:
            pred = load_prediction_result(model, output_dir)
            if pred is None:
                print(f"  Missing predictions for {model} on {course}, skipping")
                continue
            if getattr(pred, "synthetic_indices", False):
                raise SystemExit(
                    f"{course}/{model}: prediction pickle is missing student/item "
                    "indices (zeros were substituted on load), so key alignment "
                    "would be meaningless. Regenerate the predictions with "
                    "indices saved.")
            preds[model] = pred
        if not preds:
            continue
        shared = check_key_alignment(course, preds, args.intersection)
        for model, pred in preds.items():
            mask = keys_mask(pred, shared) if shared is not None else None
            if mask is not None:
                y_true = np.asarray(pred.y_true)[mask]
                y_prob = np.asarray(pred.y_pred_prob)[mask]
            else:
                y_true, y_prob = pred.y_true, pred.y_pred_prob
            metrics = compute_summary_metrics(y_true, y_prob)
            summary_rows.append({"course": course, "model": model, **metrics})
            attempt_rows.extend(
                compute_per_attempt_metrics(pred, course, model, mask=mask))
            print(f"  {course}/{model}: " + ", ".join(
                f"{k}={v:.4f}" for k, v in metrics.items() if k != "n"))

    if not summary_rows:
        raise SystemExit("no predictions loaded; refusing to write outputs")

    if args.write_tex:
        loaded = {(r["course"], r["model"]) for r in summary_rows}
        missing_cells = [
            f"{course}/{model}"
            for course in args.courses for model in args.models
            if (course, model) not in loaded
        ]
        if missing_cells:
            raise SystemExit(
                "refusing to write a partial paper table; missing predictions "
                "for: " + ", ".join(missing_cells))

    summary_df = pd.DataFrame(summary_rows)
    attempt_df = pd.DataFrame(attempt_rows)

    summary_path = "results/student_eval/metrics_summary.csv"
    attempt_path = "results/student_eval/metrics_per_attempt.csv"
    summary_df.to_csv(summary_path, index=False)
    attempt_df.to_csv(attempt_path, index=False)
    print(f"Saved {summary_path}")
    print(f"Saved {attempt_path}")

    if args.write_tex:
        os.makedirs("overleaf/tables", exist_ok=True)
        table_path = "overleaf/tables/threshold_free_metrics.tex"
        with open(table_path, "w") as f:
            f.write(to_latex_table(summary_df))
        print(f"Saved {table_path}")
    else:
        print("Skipped overleaf table (pass --write-tex to overwrite)")


if __name__ == "__main__":
    main()
