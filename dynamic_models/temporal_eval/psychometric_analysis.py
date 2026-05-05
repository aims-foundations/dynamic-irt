"""Psychometric analysis of temporal evaluation models.

Usage:
    python -m dynamic_models.temporal_eval.psychometric_analysis
    python -m dynamic_models.temporal_eval.psychometric_analysis --course_name dsa_hk231
    python -m dynamic_models.temporal_eval.psychometric_analysis --models Elo CIRT
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from dynamic_models.temporal_eval.harness import load_saved_results

COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44", "#66ccee"]


def per_course_ll_table(df, output_dir):
    """Log-likelihood table: course x model, mean across horizons."""
    ll = df[df["metric"] == "log_likelihood"]
    pivot = ll.pivot_table(index="course", columns="model", values="value", aggfunc="mean")
    print("\n=== Mean Log-Likelihood (course x model) ===")
    print(pivot.round(4).to_string())

    csv_path = os.path.join(output_dir, "ll_course_model.csv")
    pivot.round(4).to_csv(csv_path)
    print(f"Saved: {csv_path}")
    return pivot


def per_course_metrics_plot(df, output_dir):
    """1x2 figure: LL and AUC vs train cutoff week, averaged across courses."""
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    metrics = [("log_likelihood", "Log-Likelihood ↑"), ("auc", "AUC ↑")]
    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    lines = []
    labels = []
    for ax, (metric, ylabel) in zip(axes, metrics):
        mdf = df[df["metric"] == metric]
        for i, model in enumerate(models):
            sub = mdf[mdf["model"] == model]
            if len(sub) == 0:
                continue
            grouped = sub.groupby("horizon")["value"]
            means = grouped.mean().sort_index()
            color = COLORS[i % len(COLORS)]
            line, = ax.plot(means.index, means.values, "o-", color=color,
                            linewidth=1.5, markersize=4)
            if metric == metrics[0][0]:
                lines.append(line)
                labels.append(model)
        ax.set_xlabel("Train Cutoff Week")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.legend(lines, labels, loc="upper center", ncol=len(models), fontsize=7,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(output_dir, "metrics_vs_horizon.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


def item_difficulty_analysis(predictions, output_dir):
    """Compare item difficulty estimates across models."""
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    max_horizon = max(h for _, h in predictions.keys())
    model_preds = {m: p for (m, h), p in predictions.items()
                   if h == max_horizon and p.item_indices is not None}

    if len(model_preds) < 2:
        print("Need at least 2 models with item indices for difficulty comparison.")
        return

    # For each model, compute actual vs predicted difficulty using its own predictions
    models = sorted(model_preds.keys())
    plotted = []
    for model in models:
        pred = model_preds[model]
        unique_items = np.unique(pred.item_indices)
        if len(unique_items) == 0:
            continue
        actual = np.array([1.0 - np.mean(pred.y_true[pred.item_indices == i]) for i in unique_items])
        predicted = np.array([1.0 - np.mean(pred.y_pred_prob[pred.item_indices == i]) for i in unique_items])
        plotted.append((model, actual, predicted))

    if not plotted:
        print("No item-level data available.")
        return

    fig, axes = plt.subplots(1, len(plotted), figsize=(3.5 * len(plotted), 3.5))
    if len(plotted) == 1:
        axes = [axes]

    for ax, (i, (model, x, y)) in zip(axes, enumerate(plotted)):
        r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else 0
        rmse = np.sqrt(np.mean((x - y) ** 2))

        ax.scatter(x, y, s=8, alpha=0.4, color=COLORS[i % len(COLORS)])
        lo, hi = min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Actual Difficulty")
        ax.set_ylabel("Predicted Difficulty")
        ax.set_title(f"{model} (r={r:.2f}, RMSE={rmse:.2f})", fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "item_difficulty_scatter.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


def model_agreement_analysis(predictions, output_dir):
    """How often do models agree on their predictions?"""
    max_horizon = max(h for _, h in predictions.keys())
    model_preds = {m: p for (m, h), p in predictions.items() if h == max_horizon}

    if len(model_preds) < 2:
        return

    models = sorted(model_preds.keys())
    n = len(models)

    # Pairwise correlation of predicted probabilities
    # Need to align predictions by (student, item) pairs
    print(f"\n=== Model Agreement (W={max_horizon}) ===")

    # Use binary predictions for agreement
    binary_preds = {}
    for m in models:
        binary_preds[m] = (model_preds[m].y_pred_prob >= 0.5).astype(int)

    corr_matrix = np.zeros((n, n))
    agree_matrix = np.zeros((n, n))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if len(binary_preds[m1]) == len(binary_preds[m2]):
                agree_matrix[i, j] = np.mean(binary_preds[m1] == binary_preds[m2])
                corr_matrix[i, j] = np.corrcoef(
                    model_preds[m1].y_pred_prob, model_preds[m2].y_pred_prob
                )[0, 1]
            else:
                agree_matrix[i, j] = np.nan
                corr_matrix[i, j] = np.nan

    print("\nProbability Correlation:")
    corr_df = pd.DataFrame(corr_matrix, index=models, columns=models)
    print(corr_df.round(3).to_string())

    print("\nBinary Agreement Rate:")
    agree_df = pd.DataFrame(agree_matrix, index=models, columns=models)
    print(agree_df.round(3).to_string())

    csv_path = os.path.join(output_dir, "model_agreement.csv")
    corr_df.round(4).to_csv(csv_path)
    print(f"Saved: {csv_path}")



def student_trajectory_analysis(predictions, output_dir, item_week=None,
                                question_infos=None, n_students=6, seed=42):
    """Plot predicted vs actual learning trajectories per model for selected students.

    For each model, picks students with the most test observations and plots
    question-level actual outcomes vs predicted P(correct), ordered by week.
    """
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    # Build test-case → question mapping from question_infos
    tc_to_qidx = None
    n_matrix_items = 0
    if question_infos is not None:
        tc_to_qidx = question_infos['qidx'].values
        n_matrix_items = len(tc_to_qidx)
        n_questions = question_infos['qidx'].nunique()
        # Build question-level week mapping (one week per question)
        q_week = question_infos.groupby('qidx')['week'].first()

    max_horizon = max(h for _, h in predictions.keys())
    model_preds = {m: p for (m, h), p in predictions.items()
                   if h == max_horizon and p.student_indices is not None and p.item_indices is not None}

    if not model_preds:
        print("No models with student/item indices for trajectory plots.")
        return

    models = sorted(model_preds.keys())

    for model in models:
        pred = model_preds[model]

        # Map test-case-level indices to question-level if needed
        raw_items = pred.item_indices
        if tc_to_qidx is not None and raw_items.max() < n_matrix_items and len(np.unique(raw_items)) > n_questions:
            item_indices = tc_to_qidx[raw_items]
            use_q_week = True
        else:
            item_indices = raw_items
            use_q_week = False

        unique_students = np.unique(pred.student_indices)

        student_n_questions = {}
        for s in unique_students:
            mask = pred.student_indices == s
            student_n_questions[s] = len(np.unique(item_indices[mask]))

        sorted_students = sorted(student_n_questions, key=student_n_questions.get, reverse=True)
        top_pool = sorted_students[:max(n_students * 3, int(len(sorted_students) * 0.7))]
        rng = np.random.RandomState(seed)
        if len(top_pool) <= n_students:
            selected = top_pool
        else:
            selected = list(rng.choice(top_pool, size=n_students, replace=False))

        if not selected:
            continue

        n_rows = len(selected)
        fig, axes = plt.subplots(n_rows, 1, figsize=(6, 2.2 * n_rows), sharex=False)
        if n_rows == 1:
            axes = [axes]

        for ax, sidx in zip(axes, selected):
            mask = pred.student_indices == sidx
            items = item_indices[mask]
            y_true = pred.y_true[mask].astype(np.float32)
            y_pred = pred.y_pred_prob[mask]

            # Aggregate to question level
            unique_items = np.unique(items)
            q_actual = np.array([y_true[items == q].mean() for q in unique_items])
            q_pred = np.array([y_pred[items == q].mean() for q in unique_items])

            # Sort by week
            if use_q_week and question_infos is not None:
                item_weeks = np.array([q_week.get(q, 0) for q in unique_items])
                sort_order = np.argsort(item_weeks, kind='stable')
                unique_items = unique_items[sort_order]
                q_actual = q_actual[sort_order]
                q_pred = q_pred[sort_order]
                item_weeks = item_weeks[sort_order]
                week_changes = np.where(np.diff(item_weeks))[0] + 0.5
                for wc in week_changes:
                    ax.axvline(wc, color="0.85", linewidth=0.5, zorder=0)
            elif item_week is not None:
                iw = item_week.numpy() if hasattr(item_week, 'numpy') else np.array(item_week)
                item_weeks = np.array([iw[i] if i < len(iw) else 0 for i in unique_items])
                sort_order = np.argsort(item_weeks, kind='stable')
                unique_items = unique_items[sort_order]
                q_actual = q_actual[sort_order]
                q_pred = q_pred[sort_order]
                item_weeks = item_weeks[sort_order]
                week_changes = np.where(np.diff(item_weeks))[0] + 0.5
                for wc in week_changes:
                    ax.axvline(wc, color="0.85", linewidth=0.5, zorder=0)

            x_axis = np.arange(len(unique_items))

            ax.scatter(x_axis, q_actual, color="black", alpha=0.3, s=12,
                       zorder=1, label="Actual")
            ax.plot(x_axis, q_pred, color=COLORS[0], linewidth=1.2, alpha=0.8,
                    zorder=3, label="Predicted")

            n_q = len(unique_items)
            ax.set_ylabel(r"$P(\mathrm{correct})$", fontsize=7)
            ax.set_title(f"Student {sidx} ({n_q} questions)", fontsize=7, pad=2)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

        axes[0].legend(fontsize=6, loc="lower right")
        axes[-1].set_xlabel("Question (ordered by week)")
        fig.suptitle(f"{model} — Student Trajectories (W={max_horizon})", fontsize=9)
        fig.tight_layout()

        save_path = os.path.join(output_dir, f"{model.lower()}_trajectories.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {save_path}")


def _map_to_question_level(pred, question_infos):
    """Map test-case-level item_indices to question-level qidx if needed."""
    raw = pred.item_indices
    tc_to_qidx = question_infos['qidx'].values
    n_matrix_items = len(tc_to_qidx)
    n_questions = question_infos['qidx'].nunique()
    if raw.max() < n_matrix_items and len(np.unique(raw)) > n_questions:
        return tc_to_qidx[raw]
    return raw


def student_all_models_plot(predictions, output_dir, question_infos, student_ids,
                            student_id=None, horizon=None, main_data=None):
    """Plot all models' predictions vs actual for one student across all questions.

    Uses the earliest horizon (W=1) for maximum course coverage.
    Automatically selects a student with high outcome variance if none specified.
    """
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    MODEL_COLORS = {"CIRT": "#4477aa", "DynamicIRT": "#ee6677",
                    "Elo": "#228833", "RSSM": "#aa3377"}
    q_week = question_infos.groupby('qidx')['week'].first()

    H = horizon or min(h for _, h in predictions.keys())
    models = sorted(m for m, h in predictions.keys() if h == H)

    # Build per-model question-level data
    def _extract(model, main_data=None):
        pred = predictions.get((model, H))
        if pred is None or pred.student_indices is None:
            return None, None, None, None
        items = _map_to_question_level(pred, question_infos)
        if model == 'Elo':
            sids = pred.student_indices
            # Elo uses question_unittest_id; map to qidx so all models share the same ID space.
            # csv2matrices enumerates question_unittest_ids in order of unique appearance in main_data.
            if main_data is not None:
                qids_ordered = main_data['question_unittest_id'].unique()
                quid_to_qidx = {int(qid): idx for idx, qid in enumerate(qids_ordered)}
                items = np.array([quid_to_qidx.get(int(i), i) for i in items])
        else:
            sids = np.array([student_ids[s] for s in pred.student_indices])
        return sids, items, pred.y_true.astype(np.float32), pred.y_pred_prob

    model_data = {}
    for m in models:
        sids, items, yt, yp = _extract(m, main_data=main_data)
        if sids is not None:
            model_data[m] = (sids, items, yt, yp)

    if not model_data:
        print("No model data available for student plot.")
        return

    # Auto-select student if not specified
    if student_id is None:
        # Find students present in all models with large question intersection and high outcome variance
        model_student_questions = {}
        for m, (sids, items, yt, yp) in model_data.items():
            sq = {}
            for s in np.unique(sids):
                mask = sids == s
                sq[s] = set(np.unique(items[mask]))
            model_student_questions[m] = sq

        common_students = set.intersection(*[set(d.keys()) for d in model_student_questions.values()])
        best_sid, best_score = None, -1
        ref_model = next(iter(model_data))
        ref_sids, ref_items, ref_yt, _ = model_data[ref_model]
        for s in common_students:
            q_intersection = set.intersection(*[model_student_questions[m][s] for m in model_data])
            if len(q_intersection) < 20:
                continue
            mask = ref_sids == s
            si = ref_items[mask]
            sy = ref_yt[mask]
            # Only count variance on intersection questions
            q_act = np.array([sy[si == q].mean() for q in q_intersection if q in set(si)])
            n_frac = ((q_act > 0.01) & (q_act < 0.99)).sum()
            if n_frac > best_score:
                best_score = n_frac
                best_sid = s
        student_id = best_sid

    if student_id is None:
        print("Could not find a suitable student.")
        return

    # Extract question-level results per model (unordered)
    model_question_data = {}
    for model, (sids, items, yt, yp) in model_data.items():
        mask = sids == student_id
        if not mask.any():
            continue
        s_items = items[mask]
        s_yt, s_yp = yt[mask], yp[mask]
        uq = np.unique(s_items)
        q_act = {q: s_yt[s_items == q].mean() for q in uq}
        q_pred = {q: s_yp[s_items == q].mean() for q in uq}
        model_question_data[model] = (q_act, q_pred)

    if not model_question_data:
        print(f"Student {student_id} not found in predictions.")
        return

    # Restrict to questions present in ALL models
    common_questions = set.intersection(*[set(qa.keys()) for qa, _ in model_question_data.values()])
    if len(common_questions) < 3:
        print(f"Student {student_id}: only {len(common_questions)} common questions, skipping.")
        return

    # Sort common questions by week
    common_sorted = sorted(common_questions, key=lambda q: q_week.get(q, 0))
    common_weeks = np.array([q_week.get(q, 0) for q in common_sorted])

    # Build aligned arrays
    ref_model = next(iter(model_question_data))
    ref_actual = np.array([model_question_data[ref_model][0][q] for q in common_sorted])
    model_preds = {}
    for model, (q_act, q_pred) in model_question_data.items():
        model_preds[model] = np.array([q_pred[q] for q in common_sorted])

    x = np.arange(len(common_sorted))
    n_qs = len(x)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Week boundaries and labels
    wc = np.where(np.diff(common_weeks))[0] + 0.5
    for w in wc:
        ax.axvline(w, color="0.85", linewidth=0.5, zorder=0)
    for w in sorted(np.unique(common_weeks)):
        w_indices = np.where(common_weeks == w)[0]
        ax.text(w_indices[len(w_indices)//2], 1.07, f"W{w}",
                ha='center', fontsize=6, color='0.5')

    ax.plot(x, ref_actual, color="black", linewidth=1.2, linestyle=':',
            alpha=0.6, zorder=2, label="Actual")

    for model in sorted(model_preds.keys()):
        ax.plot(x, model_preds[model],
                color=MODEL_COLORS.get(model, '0.5'), linewidth=1.0, alpha=0.8,
                zorder=3, label=model)

    ax.set_xlabel("Question (ordered by week)", fontsize=8)
    ax.set_ylabel(r"$P(\mathrm{correct})$", fontsize=8)
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=7, loc='lower right', ncol=len(model_preds) + 1)
    ax.set_title(f"Student {student_id} — Predicted vs Actual Across All Questions "
                 f"({n_qs} questions)", fontsize=9)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "student_all_models_all_questions.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Psychometric analysis of temporal eval models")
    parser.add_argument("--course_name", type=str, default="all")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(REPO_ROOT, "results", "temporal_eval")
    results_df, predictions = load_saved_results(args.course_name, output_dir, args.models)

    if len(results_df) == 0 and not predictions:
        print("No saved results found.")
        return

    analysis_dir = os.path.join(output_dir, "psychometric")
    os.makedirs(analysis_dir, exist_ok=True)

    # Per-course tables and plots
    if len(results_df) > 0:
        per_course_ll_table(results_df, analysis_dir)
        per_course_metrics_plot(results_df, analysis_dir)

    if predictions:
        from dynamic_models.temporal_eval.data_loader import load_unified_data

        # When keys are 3-tuples (course, model, horizon), run per course
        sample_key = next(iter(predictions))
        if len(sample_key) == 3:
            courses = sorted({k[0] for k in predictions})
            for course in courses:
                print(f"\n{'=' * 40}")
                print(f"COURSE: {course}")
                print(f"{'=' * 40}")
                course_preds = {(k[1], k[2]): v for k, v in predictions.items() if k[0] == course}
                course_dir = os.path.join(analysis_dir, course)
                os.makedirs(course_dir, exist_ok=True)
                data = load_unified_data(course)

                item_difficulty_analysis(course_preds, course_dir)
                model_agreement_analysis(course_preds, course_dir)
                student_trajectory_analysis(course_preds, course_dir, item_week=data.item_week,
                                            question_infos=data.question_infos)
                student_all_models_plot(course_preds, course_dir,
                                        question_infos=data.question_infos,
                                        student_ids=data.student_ids,
                                        main_data=data.main_data)
        else:
            data = load_unified_data(args.course_name)

            item_difficulty_analysis(predictions, analysis_dir)
            model_agreement_analysis(predictions, analysis_dir)
            student_trajectory_analysis(predictions, analysis_dir, item_week=data.item_week,
                                        question_infos=data.question_infos)
            student_all_models_plot(predictions, analysis_dir,
                                    question_infos=data.question_infos,
                                    student_ids=data.student_ids,
                                    main_data=data.main_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
