"""Regenerate student_model_comparison.png from temporal eval predictions.

Handles different item indexing across models:
- Elo: item_indices = question_unittest_id, student_indices = student_id
- CIRT/DynamicIRT: item_indices = tensor (testcase) indices, student_indices = tensor student indices
- RSSM: item_indices = featurized question indices, student_indices = tensor student indices
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
MODELS = ["CIRT", "RSSM"]
MODEL_COLORS = {"CIRT": "#4477aa", "RSSM": "#aa3377"}

MAIN_STUDENT = 1952
APPENDIX_STUDENTS = [1901, 1640, 1639]


def aggregate_to_question(y_values, item_indices, student_mask):
    """Aggregate observations to unique item means for one student."""
    items = item_indices[student_mask]
    vals = y_values[student_mask]
    unique_items = np.unique(items)
    means = np.array([vals[items == q].mean() for q in unique_items])
    return unique_items, means


def plot_student(sid, data, preds, tidx_to_qidx, rssm_to_qidx, qidx_to_week,
                 out_path):
    sid_list = data.student_ids
    target_tidx = sid_list.index(sid)

    model_qmeans = {}

    # RSSM
    rssm = preds["RSSM"]
    rssm_mask = rssm.student_indices == target_tidx
    rssm_items_translated = np.array([rssm_to_qidx.get(i, -1) for i in rssm.item_indices])
    valid_rssm = rssm_mask & (rssm_items_translated >= 0)
    rssm_qidxs, rssm_actual = aggregate_to_question(rssm.y_true, rssm_items_translated, valid_rssm)
    _, rssm_pred = aggregate_to_question(rssm.y_pred_prob, rssm_items_translated, valid_rssm)
    model_qmeans["RSSM"] = dict(zip(rssm_qidxs, rssm_pred))
    actual_by_q = dict(zip(rssm_qidxs, rssm_actual))

    # CIRT
    for model in ["CIRT"]:
        pred = preds[model]
        mask = pred.student_indices == target_tidx
        if mask.sum() == 0:
            continue
        items = pred.item_indices[mask]
        vals = pred.y_pred_prob[mask]
        qidxs = tidx_to_qidx[items]
        unique_q = np.unique(qidxs)
        model_qmeans[model] = {q: vals[qidxs == q].mean() for q in unique_q}

    # Intersection of all models' question sets
    common_qs = set(actual_by_q.keys())
    for qset in model_qmeans.values():
        common_qs &= set(qset.keys())
    common_qs = np.array(sorted(common_qs))

    weeks = np.array([qidx_to_week.get(q, 0) for q in common_qs])
    order = np.argsort(weeks * 10000 + common_qs)
    common_qs = common_qs[order]
    weeks = weeks[order]

    n_questions = len(common_qs)
    x = np.arange(n_questions)
    actual_vals = np.array([actual_by_q[q] for q in common_qs])
    model_predictions = {m: np.array([qm[q] for q in common_qs]) for m, qm in model_qmeans.items()}

    def smooth(vals, window=2):
        return pd.Series(vals).rolling(window, min_periods=1, center=True).mean().values

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
    ax.plot(x, smooth(actual_vals), "k:", linewidth=1.2, label="Actual", alpha=0.7)
    for model in MODELS:
        if model not in model_predictions:
            continue
        ax.plot(x, smooth(model_predictions[model]), color=MODEL_COLORS[model], linewidth=1.2, label=model)

    prev_week = None
    for i, w in enumerate(weeks):
        if w != prev_week and w > 0:
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
            ax.text(i + 0.5, 1.02, f"W{int(w)}", transform=ax.get_xaxis_transform(),
                    fontsize=10, ha="left", color="gray")
            prev_week = w

    ax.set_xlabel("Question (ordered by week)", fontsize=12)
    ax.set_ylabel("$P$(correct)", fontsize=12)
    ax.set_title(f"Student {sid} — Predicted vs Actual Scores", fontsize=13)
    ax.legend(loc="lower center", ncol=3, fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, n_questions - 0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    data = load_unified_data(COURSE)

    q_infos = pd.read_csv(f".cache/matrices/{COURSE}/question_infos.csv")
    tidx_to_qidx = q_infos["qidx"].values
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

    # Load predictions
    preds = {}
    for model in MODELS:
        with open(f"{RESULTS_DIR}/{model}_predictions.pkl", "rb") as f:
            mp = pickle.load(f)
        preds[model] = mp[1]

    # Main figure
    plot_student(MAIN_STUDENT, data, preds, tidx_to_qidx, rssm_to_qidx,
                 qidx_to_week, "overleaf/figures/student_model_comparison.png")

    # Appendix figures
    for sid in APPENDIX_STUDENTS:
        plot_student(sid, data, preds, tidx_to_qidx, rssm_to_qidx,
                     qidx_to_week,
                     f"overleaf/figures/student_{sid}_model_comparison.png")


if __name__ == "__main__":
    main()
