"""Student-based evaluation: calibrate on train students, score test students.

Quality-filters students/questions across all weeks, then splits students
55/15/30 into train/val/test, with the validation students driving
checkpoint selection. Train students calibrate item parameters; test
students' weeks 1-3 data estimates ability; predictions are on test
students' weeks 4-6.

Usage:
    python -m dynamic_models.temporal_eval.run_student_eval
    python -m dynamic_models.temporal_eval.run_student_eval --models IRT BKT DKT CodeDKT RSSM
    python -m dynamic_models.temporal_eval.run_student_eval --courses dsa_hk231 dsa_hk221 pf_hk232 pf_hk222 --plot_losses
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

from dynamic_models.temporal_eval.data_filter import DEFAULT_FILTER
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
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
    min_question_coverage: float = 0.25,
    adapter_kwargs: dict = None,
):
    data, split = load_student_split_data(
        course_name=course_name,
        max_attempts=max_attempts,
        test_frac=test_frac,
        train_week_cutoff=train_week_cutoff,
        seed=seed,
        min_pass_rate=min_pass_rate,
        max_pass_rate=max_pass_rate,
        min_question_coverage=min_question_coverage,
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
    failed = []

    for model_name, adapter in adapters.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print("=" * 60)

        t0 = time.time()
        try:
            prediction = adapter.fit_and_predict_student_split(
                data, split, seed=seed,
                **((adapter_kwargs or {}).get(model_name, {}))
            )
            metrics = compute_metrics(prediction.y_true, prediction.y_pred_prob)
            runtime = time.time() - t0

            predictions[model_name] = prediction

            print(f"  AUC={metrics.auc:.4f}  "
                  f"Acc={metrics.accuracy:.4f}  "
                  f"BalAcc={metrics.balanced_accuracy:.4f}  "
                  f"F1={metrics.f1:.4f}  "
                  f"Brier={metrics.brier:.4f}  "
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
            failed.append(model_name)

        # Save predictions
        if model_name in predictions:
            pred = predictions[model_name]
            pred_path = os.path.join(output_dir, f"{model_name}_student_pred.pkl")
            with open(pred_path, "wb") as f:
                pickle.dump(pred, f)

    if failed:
        print(f"\n{'!' * 60}\nMODELS FAILED: {', '.join(failed)}\n{'!' * 60}",
              file=sys.stderr)

    results_df = pd.DataFrame(results_rows)
    if len(results_df) > 0:
        csv_path = os.path.join(output_dir, "student_eval.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved: {csv_path}")

    return results_df, predictions, data


COURSE_LABELS = {
    "dsa_hk231": "DSA 231",
    "dsa_hk221": "DSA 221",
    "pf_hk232": "PF 232",
    "pf_hk222": "PF 222",
}


def plot_loss_curves(all_losses, output_dir):
    """Plot training loss curves. One figure per model, one subplot per course."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tueplots import bundles

    plt.rcParams.update(bundles.icml2022())
    plt.rcParams.update({"text.usetex": True})

    models = sorted({m for cl in all_losses.values() for m in cl})
    courses = list(all_losses.keys())
    os.makedirs(output_dir, exist_ok=True)

    for model_name in models:
        model_data = {c: all_losses[c][model_name]
                      for c in courses if model_name in all_losses[c]}
        if not model_data:
            continue

        n = len(model_data)
        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.25 * ncols, 2.2 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for i, (course, losses) in enumerate(model_data.items()):
            ax = axes[i]
            losses = np.asarray(losses, dtype=float)
            epochs = np.arange(1, len(losses) + 1)
            if len(losses) >= 15:
                # 5-epoch moving average
                losses = np.convolve(losses, np.ones(5) / 5, mode="valid")
                epochs = epochs[2:2 + len(losses)]
            ax.plot(epochs, losses, linewidth=0.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training Loss")
            ax.set_title(COURSE_LABELS.get(course, course))

        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(model_name, y=1.02)
        fig.tight_layout()

        path = os.path.join(output_dir, f"{model_name.lower()}_loss.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Loss curve saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Student-based evaluation")
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    parser.add_argument("--courses", type=str, nargs="+", default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--plot_losses", action="store_true")
    parser.add_argument("--min_pass_rate", type=float,
                        default=DEFAULT_FILTER.min_pass_rate)
    parser.add_argument("--max_pass_rate", type=float,
                        default=DEFAULT_FILTER.max_pass_rate)
    parser.add_argument("--min_question_coverage", type=float,
                        default=DEFAULT_FILTER.min_question_coverage)
    parser.add_argument("--max_attempts", type=int,
                        default=DEFAULT_FILTER.max_attempts)
    args = parser.parse_args()

    courses = args.courses or [args.course_name]

    all_losses = {}
    missing_models = []
    for course in courses:
        output_dir = args.output_dir or os.path.join(
            REPO_ROOT, "results", "student_eval", course
        )

        _, predictions, _ = run_student_evaluation(
            course_name=course,
            models=args.models,
            seed=args.seed,
            output_dir=output_dir,
            max_attempts=args.max_attempts,
            min_pass_rate=args.min_pass_rate,
            max_pass_rate=args.max_pass_rate,
            min_question_coverage=args.min_question_coverage,
        )

        requested = args.models or list(get_adapter_registry().keys())
        missing_models.extend(
            f"{course}/{m}" for m in requested if m not in predictions
        )

        course_losses = {}
        for model_name, pred in predictions.items():
            if pred.losses and "train" in pred.losses:
                course_losses[model_name] = pred.losses["train"]
        all_losses[course] = course_losses

    if args.plot_losses and all_losses:
        loss_dir = os.path.join(REPO_ROOT, "results", "loss_curves")
        plot_loss_curves(all_losses, loss_dir)

    if missing_models:
        print(f"MODELS MISSING FROM RESULTS: {', '.join(missing_models)}",
              file=sys.stderr)
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
