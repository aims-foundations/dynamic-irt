"""Shared helpers for LLM-prediction analysis scripts.

Holds the JSONL-vs-ground-truth alignment and bootstrap logic used by
plot_filtered_accuracy.py and plot_llm_ablations.py, the _pass_fraction
helper used by kendall_tau_decomposition.py and llm_bug_comparison.py,
and the course/model display tables shared across figure and table scripts.
"""

import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

MODEL_STYLES = {
    "IRT": {"color": "#2080cc", "marker": "o"},
    "CIRT": {"color": "#d63030", "marker": "s"},
    "BKT": {"color": "#2e8b57", "marker": "D"},
    "DKT": {"color": "#7b2d8e", "marker": "^"},
    "CodeDKT": {"color": "#cc3399", "marker": "<"},
    "RSSM": {"color": "#66ccee", "marker": "v"},
}

MODEL_DISPLAY_NAMES = {
    "CodeDKT": "Code-DKT",
}

COURSE_TITLES = {
    "dsa_hk231": "DSA HK231",
    "dsa_hk221": "DSA HK221",
    "pf_hk232": "PF HK232",
    "pf_hk222": "PF HK222",
}


def _pass_fraction(s):
    s = str(s).strip()
    if not s or s == "nan":
        return np.nan
    return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan


def load_question_id_to_name():
    """Map HF question_id (as str) to question_name from the dataset snapshot."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    hf_dir = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        repo_type="dataset", local_files_only=True,
    )
    hf_qi = pd.read_csv(f"{hf_dir}/question_infos.csv")
    return {
        str(int(row["question_id"])): row["question_name"]
        for _, row in hf_qi.iterrows()
    }


def compute_llm_balanced_accuracy(jsonl_path, course, max_attempts=10):
    """Per-attempt balanced accuracy of LLM JSONL predictions vs ground truth.

    Aligns (student, question) pairs from the JSONL against
    load_student_split_data(course) and bootstraps a 95% CI (500 reps,
    seed 42) at each attempt. Returns (accs, ci_los, ci_his, counts, boots)
    where boots is the list of per-attempt bootstrap value arrays.
    """
    from dynamic_models.temporal_eval.data_loader import load_student_split_data

    data, split = load_student_split_data(course)
    qi = data.question_infos
    test_item_set = set(split.test_item_indices.tolist())
    qid_to_name = load_question_id_to_name()

    with open(jsonl_path) as f:
        rows = [json.loads(l) for l in f]

    llm_by_pair = defaultdict(list)
    for r in rows:
        llm_by_pair[(str(r["student_id"]), str(r["question_unittest_id"]))].append(r)

    accs = []
    ci_los = []
    ci_his = []
    counts = []
    boots = []

    for a in range(max_attempts):
        actuals_list = []
        preds_list = []

        for (sid, qid), attempts in llm_by_pair.items():
            s_indices = [i for i in split.test_student_indices
                         if str(data.student_ids[i]) == sid]
            if not s_indices:
                continue
            s_idx = s_indices[0]

            qname = qid_to_name.get(qid, "")
            q_items = [i for i in qi[qi["qname"] == qname].index
                       if i in test_item_set]
            if not q_items:
                continue

            real_results = []
            for qidx in q_items:
                obs = data.correctness_matrix[s_idx, qidx, :].numpy()
                valid = obs[obs != -1]
                if a < len(valid):
                    real_results.append(int(valid[a]))
                elif len(valid) > 0:
                    real_results.append(int(valid[-1]))

            if a < len(attempts):
                llm_pp = attempts[a]["pass"]
            else:
                llm_pp = attempts[-1]["pass"]

            if not real_results:
                continue

            llm_binary = [int(c) for c in str(llm_pp) if c in "01"]
            if not llm_binary:
                continue

            n = min(len(real_results), len(llm_binary))
            for ti in range(n):
                actuals_list.append(real_results[ti])
                preds_list.append(llm_binary[ti])

        if len(actuals_list) >= 10:
            actuals_arr = np.array(actuals_list)
            preds_arr = np.array(preds_list)
            pos = actuals_arr == 1
            neg = actuals_arr == 0
            tpr = (preds_arr[pos] == 1).mean() if pos.sum() > 0 else np.nan
            tnr = (preds_arr[neg] == 0).mean() if neg.sum() > 0 else np.nan
            bal_acc = (tpr + tnr) / 2 if not (np.isnan(tpr) or np.isnan(tnr)) else np.nan
            accs.append(bal_acc)

            rng = np.random.default_rng(42)
            boot_vals = []
            for _ in range(500):
                idx = rng.integers(0, len(actuals_arr), size=len(actuals_arr))
                ba, bp = actuals_arr[idx], preds_arr[idx]
                p = ba == 1; n = ba == 0
                t = (bp[p] == 1).mean() if p.sum() > 0 else np.nan
                tn = (bp[n] == 0).mean() if n.sum() > 0 else np.nan
                if not (np.isnan(t) or np.isnan(tn)):
                    boot_vals.append((t + tn) / 2)
            boot_vals = np.array(boot_vals)
            ci_los.append(np.quantile(boot_vals, 0.025) if len(boot_vals) > 10 else np.nan)
            ci_his.append(np.quantile(boot_vals, 0.975) if len(boot_vals) > 10 else np.nan)
            boots.append(boot_vals)
        else:
            accs.append(np.nan)
            ci_los.append(np.nan)
            ci_his.append(np.nan)
            boots.append(np.array([]))
        counts.append(len(actuals_list))

    return np.array(accs), np.array(ci_los), np.array(ci_his), np.array(counts), boots
