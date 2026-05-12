"""RSSM and CIRT prediction evolution across training horizons.

Shows how RSSM and CIRT predictions for a single student change as more
training data becomes available.

Output:
    overleaf/figures/rssm_horizon_evolution.png
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import bundles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamic_models.temporal_eval.data_loader import load_unified_data

plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

COURSE = "dsa_hk231"
COURSE_ID = 1
RESULTS_DIR = f"results/temporal_eval_full/{COURSE}"
STUDENT_ID = 1952

RSSM_COLORS = {1: "#d62728", 5: "#9467bd"}
CIRT_COLORS = {1: "#ff7f0e", 5: "#2ca02c"}


def aggregate_to_question(y_values, item_indices, student_mask):
    items = item_indices[student_mask]
    vals = y_values[student_mask]
    unique_items = np.unique(items)
    means = np.array([vals[items == q].mean() for q in unique_items])
    return unique_items, means


def main():
    data = load_unified_data(COURSE)

    # Question metadata
    q_infos = pd.read_csv(f".cache/matrices/{COURSE}/question_infos.csv")
    qidx_to_week = dict(q_infos.drop_duplicates("qidx")[["qidx", "week"]].values)

    # RSSM index mapping
    with open(f"data/multimodal/{COURSE}/metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    qi_complete = pd.read_csv("data_analysis/question_infos_complete.csv")
    qi_course = qi_complete[qi_complete["course_id"] == COURSE_ID]
    qid_to_name = dict(zip(qi_course["question_id"], qi_course["question_name"]))
    qname_to_qidx = dict(q_infos.drop_duplicates("qidx")[["qname", "qidx"]].values)
    rssm_to_qidx = {}
    for qid, rssm_idx in meta["question_to_idx"].items():
        name = qid_to_name.get(qid)
        if name and name in qname_to_qidx:
            rssm_to_qidx[rssm_idx] = qname_to_qidx[name]

    # Load RSSM and CIRT horizons
    with open(f"{RESULTS_DIR}/RSSM_predictions.pkl", "rb") as f:
        rssm_horizons = pickle.load(f)
    with open(f"{RESULTS_DIR}/CIRT_predictions.pkl", "rb") as f:
        cirt_horizons = pickle.load(f)

    sid_list = data.student_ids
    target_tidx = sid_list.index(STUDENT_ID)

    # Extract per-question RSSM predictions
    rssm_preds = {}
    actual_by_q = {}

    for horizon in sorted(rssm_horizons.keys()):
        pred = rssm_horizons[horizon]
        mask = pred.student_indices == target_tidx
        items_translated = np.array([rssm_to_qidx.get(i, -1) for i in pred.item_indices])
        valid = mask & (items_translated >= 0)
        if valid.sum() == 0:
            continue
        qidxs, pred_means = aggregate_to_question(pred.y_pred_prob, items_translated, valid)
        _, actual_means = aggregate_to_question(pred.y_true, items_translated, valid)
        rssm_preds[horizon] = dict(zip(qidxs, pred_means))
        actual_by_q.update(dict(zip(qidxs, actual_means)))

    # Extract per-question CIRT predictions (item_indices are into the correctness
    # matrix which has one entry per test case; aggregate to question level via qidx)
    valid_qidxs = set(qidx_to_week.keys())
    cirt_preds = {}
    for horizon in sorted(cirt_horizons.keys()):
        pred = cirt_horizons[horizon]
        mask = pred.student_indices == target_tidx
        items = pred.item_indices
        valid = mask & np.isin(items, list(valid_qidxs))
        if valid.sum() == 0:
            continue
        qidxs, pred_means = aggregate_to_question(pred.y_pred_prob, items, valid)
        _, actual_means = aggregate_to_question(pred.y_true, items, valid)
        cirt_preds[horizon] = dict(zip(qidxs, pred_means))
        actual_by_q.update(dict(zip(qidxs, actual_means)))

    # Get all questions that appear in any horizon of either model
    all_qs = set()
    for qm in list(rssm_preds.values()) + list(cirt_preds.values()):
        all_qs |= set(qm.keys())
    all_qs = np.array(sorted(all_qs))

    # Sort by week then question index
    weeks = np.array([qidx_to_week.get(q, 0) for q in all_qs])
    order = np.argsort(weeks * 10000 + all_qs)
    all_qs = all_qs[order]
    weeks = weeks[order]

    n_questions = len(all_qs)
    x = np.arange(n_questions)
    actual_vals = np.array([actual_by_q.get(q, np.nan) for q in all_qs])

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))

    # Smoothing helper
    def smooth(vals, window=5):
        return pd.Series(vals).rolling(window, min_periods=1, center=True).mean().values

    # Actual scores
    ax.plot(x, smooth(actual_vals), "k:", linewidth=1.0, label="Actual", alpha=0.6)

    # Plot RSSM and CIRT predictions for horizons 1 and 5
    for horizon in [1, 5]:
        for model_name, model_preds, colors in [
            ("RSSM", rssm_preds, RSSM_COLORS),
            ("CIRT", cirt_preds, CIRT_COLORS),
        ]:
            if horizon not in model_preds:
                continue
            qm = model_preds[horizon]
            pred_vals = np.full(n_questions, np.nan)
            for i, q in enumerate(all_qs):
                if q in qm:
                    pred_vals[i] = qm[q]
            valid_mask = ~np.isnan(pred_vals)
            if valid_mask.sum() == 0:
                continue
            smoothed = smooth(pred_vals[valid_mask])
            ax.plot(x[valid_mask], smoothed,
                    color=colors[horizon],
                    linewidth=1.3, alpha=0.85,
                    label=f"{model_name} W1--{horizon}")

    # Week boundary markers
    prev_week = None
    for i, w in enumerate(weeks):
        if w != prev_week and w > 0:
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
            ax.text(i + 0.5, 1.02, f"W{int(w)}", transform=ax.get_xaxis_transform(),
                    fontsize=7, ha="left", color="gray")
            prev_week = w

    ax.set_xlabel("Question (ordered by week)")
    ax.set_ylabel("$P$(correct)")
    ax.set_title(f"Student {STUDENT_ID} — RSSM vs CIRT across training horizons")
    ax.legend(loc="lower center", ncol=6, fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, n_questions - 0.5)

    plt.tight_layout()
    out_path = "overleaf/figures/rssm_horizon_evolution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
