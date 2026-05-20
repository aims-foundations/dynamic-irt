"""Prototype variants of student_model_comparison figure."""

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
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

COURSE = "dsa_hk231"
COURSE_ID = 1
RESULTS_DIR = f"results/temporal_eval_full/{COURSE}"
MODELS = ["Elo", "CIRT", "DynamicIRT", "RSSM"]
MODEL_COLORS = {"Elo": "#228833", "CIRT": "#4477aa", "DynamicIRT": "#ee6677", "RSSM": "#aa3377"}

MAIN_STUDENT = 1952


def aggregate_to_question(y_values, item_indices, student_mask):
    items = item_indices[student_mask]
    vals = y_values[student_mask]
    unique_items = np.unique(items)
    means = np.array([vals[items == q].mean() for q in unique_items])
    return unique_items, means


def load_data():
    data = load_unified_data(COURSE)
    q_infos = pd.read_csv(f".cache/matrices/{COURSE}/question_infos.csv")
    tidx_to_qidx = q_infos["qidx"].values
    qidx_to_week = dict(q_infos.drop_duplicates("qidx")[["qidx", "week"]].values)

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

    quid_to_qidx = {}
    for qid, qname in qid_to_name.items():
        if qname in qname_to_qidx:
            quid_to_qidx[int(qid)] = int(qname_to_qidx[qname])

    HORIZON = 3
    preds = {}
    for model in MODELS:
        with open(f"{RESULTS_DIR}/{model}_predictions.pkl", "rb") as f:
            mp = pickle.load(f)
        preds[model] = mp[HORIZON]

    return data, preds, tidx_to_qidx, rssm_to_qidx, quid_to_qidx, qidx_to_week


def get_student_data(sid, data, preds, tidx_to_qidx, rssm_to_qidx, quid_to_qidx, qidx_to_week):
    sid_list = data.student_ids
    target_tidx = sid_list.index(sid)

    model_qmeans = {}

    rssm = preds["RSSM"]
    rssm_mask = rssm.student_indices == target_tidx
    rssm_items_translated = np.array([rssm_to_qidx.get(i, -1) for i in rssm.item_indices])
    valid_rssm = rssm_mask & (rssm_items_translated >= 0)
    rssm_qidxs, rssm_actual = aggregate_to_question(rssm.y_true, rssm_items_translated, valid_rssm)
    _, rssm_pred = aggregate_to_question(rssm.y_pred_prob, rssm_items_translated, valid_rssm)
    model_qmeans["RSSM"] = dict(zip(rssm_qidxs, rssm_pred))
    actual_by_q = dict(zip(rssm_qidxs, rssm_actual))

    for model in ["CIRT", "DynamicIRT"]:
        pred = preds[model]
        mask = pred.student_indices == target_tidx
        if mask.sum() == 0:
            continue
        items = pred.item_indices[mask]
        vals = pred.y_pred_prob[mask]
        qidxs = tidx_to_qidx[items]
        unique_q = np.unique(qidxs)
        model_qmeans[model] = {q: vals[qidxs == q].mean() for q in unique_q}

    elo = preds["Elo"]
    elo_mask = elo.student_indices == sid
    if elo_mask.sum() > 0:
        elo_items = elo.item_indices[elo_mask]
        elo_vals = elo.y_pred_prob[elo_mask]
        elo_qidxs = np.array([quid_to_qidx.get(int(q), -1) for q in elo_items])
        valid = elo_qidxs >= 0
        elo_qidxs, elo_vals = elo_qidxs[valid], elo_vals[valid]
        unique_q = np.unique(elo_qidxs)
        model_qmeans["Elo"] = {q: elo_vals[elo_qidxs == q].mean() for q in unique_q}

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

    return x, actual_vals, model_predictions, weeks, n_questions


def add_week_markers(ax, weeks):
    prev_week = None
    for i, w in enumerate(weeks):
        if w != prev_week and w > 0:
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
            ax.text(i + 0.5, 1.02, f"W{int(w)}", transform=ax.get_xaxis_transform(),
                    fontsize=7, ha="left", color="gray")
            prev_week = w


def prototype_v1(x, actual_vals, model_predictions, weeks, n_questions):
    """Smoothed, CIRT + RSSM only."""
    smooth_w = 5
    smooth = lambda v: pd.Series(v).rolling(smooth_w, center=True, min_periods=1).mean().values

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
    ax.plot(x, smooth(actual_vals), "k-", linewidth=1.2, label="Actual", alpha=0.7)
    for model in ["CIRT", "RSSM"]:
        if model not in model_predictions:
            continue
        ax.plot(x, smooth(model_predictions[model]), color=MODEL_COLORS[model],
                linewidth=1.5, label=model)

    add_week_markers(ax, weeks)
    ax.set_xlabel("Question (ordered by week)")
    ax.set_ylabel("$P$(correct)")
    ax.legend(loc="lower center", ncol=3, fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, n_questions - 0.5)

    plt.tight_layout()
    out = "overleaf/figures/proto_student_v1.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


CACHE_PATH = ".cache/proto_student_all.pkl"


def collect_all_students(data, preds, tidx_to_qidx, rssm_to_qidx, quid_to_qidx, qidx_to_week):
    """Collect per-student data, with disk caching."""
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached student data from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    all_student_data = []
    sid_list = data.student_ids
    for sid in sid_list:
        try:
            x, actual_vals, model_predictions, weeks, n_questions = get_student_data(
                sid, data, preds, tidx_to_qidx, rssm_to_qidx, quid_to_qidx, qidx_to_week)
        except (ValueError, KeyError):
            continue
        if "CIRT" not in model_predictions or "RSSM" not in model_predictions:
            continue
        all_student_data.append((actual_vals, model_predictions["CIRT"],
                                 model_predictions["RSSM"], weeks, n_questions))

    print(f"Collected data for {len(all_student_data)} students, caching...")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(all_student_data, f)
    return all_student_data


def prototype_v2(data, preds, tidx_to_qidx, rssm_to_qidx, quid_to_qidx, qidx_to_week):
    """All students as background curves, bold averages for Actual/CIRT/RSSM."""
    smooth_w = 5
    smooth = lambda v: pd.Series(v).rolling(smooth_w, center=True, min_periods=1).mean().values

    all_student_data = collect_all_students(
        data, preds, tidx_to_qidx, rssm_to_qidx, quid_to_qidx, qidx_to_week)

    # Use the max question count as reference; pad shorter students with NaN
    max_n = max(d[4] for d in all_student_data)
    actual_matrix = np.full((len(all_student_data), max_n), np.nan)
    cirt_matrix = np.full((len(all_student_data), max_n), np.nan)
    rssm_matrix = np.full((len(all_student_data), max_n), np.nan)

    # Find the most common week mapping (from the student with the most questions)
    ref_idx = np.argmax([d[4] for d in all_student_data])
    ref_weeks = all_student_data[ref_idx][3]

    for i, (actual, cirt, rssm, weeks, nq) in enumerate(all_student_data):
        actual_matrix[i, :nq] = actual
        cirt_matrix[i, :nq] = cirt
        rssm_matrix[i, :nq] = rssm

    ref_x = np.arange(max_n)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))

    # Background: individual student actual curves (smoothed)
    rng = np.random.default_rng(42)
    bg_indices = rng.choice(len(all_student_data), size=min(200, len(all_student_data)), replace=False)
    for i in bg_indices:
        nq = all_student_data[i][4]
        sx = np.arange(nq)
        ax.plot(sx, smooth(actual_matrix[i, :nq]), color="0.75", linewidth=0.3, alpha=0.25)

    # Bold averages
    avg_actual = smooth(np.nanmean(actual_matrix, axis=0))
    avg_cirt = smooth(np.nanmean(cirt_matrix, axis=0))
    avg_rssm = smooth(np.nanmean(rssm_matrix, axis=0))

    ax.plot(ref_x, avg_actual, "k-", linewidth=2.0, label="Actual (avg)")
    ax.plot(ref_x, avg_cirt, color=MODEL_COLORS["CIRT"], linewidth=2.0, label="CIRT (avg)")
    ax.plot(ref_x, avg_rssm, color=MODEL_COLORS["RSSM"], linewidth=2.0, label="RSSM (avg)")

    add_week_markers(ax, ref_weeks)
    ax.set_xlabel("Question (ordered by week)")
    ax.set_ylabel("$P$(correct)")
    ax.legend(loc="lower center", ncol=3, fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, max_n - 0.5)

    plt.tight_layout()
    out = "overleaf/figures/proto_student_v2.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def prototype_v3(x, actual_vals, model_predictions, weeks, n_questions):
    """Original style (all 4 models), but only test questions (week > 3)."""
    # Filter to weeks > 3 (test set after horizon 3 training)
    test_mask = weeks > 3
    tx = np.arange(test_mask.sum())
    t_actual = actual_vals[test_mask]
    t_weeks = weeks[test_mask]
    t_preds = {m: v[test_mask] for m, v in model_predictions.items()}

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
    ax.plot(tx, t_actual, "k:", linewidth=1.2, label="Actual", alpha=0.7)
    for model in MODELS:
        if model not in t_preds:
            continue
        ax.plot(tx, t_preds[model], color=MODEL_COLORS[model], linewidth=1.2, label=model)

    add_week_markers(ax, t_weeks)
    ax.set_xlabel("Question (ordered by week)")
    ax.set_ylabel("$P$(correct)")
    ax.set_title(f"Student {MAIN_STUDENT} — Test predictions after 3 weeks of training ({len(tx)} questions)")
    ax.legend(loc="lower center", ncol=5, fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, len(tx) - 0.5)

    plt.tight_layout()
    out = "overleaf/figures/proto_student_v3.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def prototype_metrics_distance():
    """AUC vs prediction distance (test_week - horizon), one line per model."""
    from dynamic_models.temporal_eval.metrics import compute_metrics

    COURSES = {
        "dsa_hk231": 1,
        "dsa_hk221": 2,
        "pf_hk232": 3,
        "pf_hk222": 4,
    }

    rows = []
    for course, course_id in COURSES.items():
        results_dir = f"results/temporal_eval_full/{course}"
        data = load_unified_data(course)
        item_week = data.item_week.numpy()

        # Build RSSM index -> week mapping
        with open(f"data/multimodal/{course}/metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        qi_complete = pd.read_csv("data_analysis/question_infos_complete.csv")
        qi_course = qi_complete[qi_complete["course_id"] == course_id]
        qid_to_name = dict(zip(qi_course["question_id"], qi_course["question_name"]))
        q_infos = pd.read_csv(f".cache/matrices/{course}/question_infos.csv")
        qname_to_week = dict(q_infos.drop_duplicates("qname")[["qname", "week"]].values)
        rssm_to_week = {}
        for qid, rssm_idx in meta["question_to_idx"].items():
            name = qid_to_name.get(qid)
            if name and name in qname_to_week:
                rssm_to_week[rssm_idx] = int(qname_to_week[name])

        # Elo: question_unittest_id -> week
        quid_to_week = {}
        for qid, qname in qid_to_name.items():
            if qname in qname_to_week:
                quid_to_week[int(qid)] = int(qname_to_week[qname])

        for model in MODELS:
            pred_path = f"{results_dir}/{model}_predictions.pkl"
            if not os.path.exists(pred_path):
                continue
            with open(pred_path, "rb") as f:
                all_preds = pickle.load(f)

            for horizon, pred in all_preds.items():
                # Map item_indices to weeks
                if model == "RSSM":
                    obs_weeks = np.array([rssm_to_week.get(int(i), -1) for i in pred.item_indices])
                elif model == "Elo":
                    obs_weeks = np.array([quid_to_week.get(int(i), -1) for i in pred.item_indices])
                else:
                    obs_weeks = item_week[pred.item_indices]

                for tw in sorted(set(obs_weeks)):
                    if tw <= horizon or tw < 0 or tw > 6:
                        continue
                    mask = obs_weeks == tw
                    if mask.sum() < 50:
                        continue
                    metrics = compute_metrics(pred.y_true[mask], pred.y_pred_prob[mask])
                    distance = tw - horizon
                    rows.append({
                        "course": course, "model": model, "horizon": horizon,
                        "test_week": tw, "distance": distance,
                        "auc": metrics.auc, "log_likelihood": metrics.log_likelihood,
                        "n": int(mask.sum()),
                    })

        print(f"  {course}: done")

    df = pd.DataFrame(rows)
    df.to_csv(".cache/per_week_metrics.csv", index=False)
    print(f"Saved metrics: {len(df)} rows")

    # Horizon 1 only: all 4 models, LL and AUC vs prediction distance (test week)
    h1 = df[df["horizon"] == 1]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))

    for ax, metric in zip(axes, ["log_likelihood", "auc"]):
        for model in MODELS:
            sub = h1[h1["model"] == model]
            grouped = sub.groupby("distance")[metric]
            means = grouped.mean().sort_index()
            sems = grouped.sem().sort_index()
            color = MODEL_COLORS[model]
            lw = 2.0 if model == "RSSM" else 1.3
            ms = 5 if model == "RSSM" else 3.5
            ax.fill_between(means.index, means - sems, means + sems, color=color, alpha=0.15)
            ax.plot(means.index, means.values, "o-", color=color,
                    label=model, linewidth=lw, markersize=ms)

        if metric == "auc":
            ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

        metric_label = r"AUC $\uparrow$" if metric == "auc" else r"Log-Likelihood $\uparrow$"
        ax.set_xlabel("Prediction Distance (weeks ahead)")
        ax.set_ylabel(metric_label)
        ax.set_xticks(sorted(h1["distance"].unique()))
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODELS),
               fontsize=7, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    out = "overleaf/figures/proto_metrics_v2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def prototype_kendall_tau():
    """Per-student Kendall Tau vs horizon and vs prediction distance."""
    from scipy.stats import kendalltau

    COURSES = {
        "dsa_hk231": 1,
        "dsa_hk221": 2,
        "pf_hk232": 3,
        "pf_hk222": 4,
    }

    CACHE = ".cache/kendall_tau_metrics.csv"
    if os.path.exists(CACHE):
        print(f"Loading cached Kendall Tau from {CACHE}")
        df = pd.read_csv(CACHE)
    else:
        rows = []
        for course, course_id in COURSES.items():
            results_dir = f"results/temporal_eval_full/{course}"
            data = load_unified_data(course)
            item_week = data.item_week.numpy()

            # Build week mappings (same as prototype_metrics_distance)
            with open(f"data/multimodal/{course}/metadata.pkl", "rb") as f:
                meta = pickle.load(f)
            qi_complete = pd.read_csv("data_analysis/question_infos_complete.csv")
            qi_course = qi_complete[qi_complete["course_id"] == course_id]
            qid_to_name = dict(zip(qi_course["question_id"], qi_course["question_name"]))
            q_infos = pd.read_csv(f".cache/matrices/{course}/question_infos.csv")
            qname_to_week = dict(q_infos.drop_duplicates("qname")[["qname", "week"]].values)
            rssm_to_week = {}
            for qid, rssm_idx in meta["question_to_idx"].items():
                name = qid_to_name.get(qid)
                if name and name in qname_to_week:
                    rssm_to_week[rssm_idx] = int(qname_to_week[name])
            quid_to_week = {}
            for qid, qname in qid_to_name.items():
                if qname in qname_to_week:
                    quid_to_week[int(qid)] = int(qname_to_week[qname])

            for model in MODELS:
                pred_path = f"{results_dir}/{model}_predictions.pkl"
                if not os.path.exists(pred_path):
                    continue
                with open(pred_path, "rb") as f:
                    all_preds = pickle.load(f)

                for horizon, pred in all_preds.items():
                    if horizon > 5:
                        continue
                    # Map observations to weeks
                    if model == "RSSM":
                        obs_weeks = np.array([rssm_to_week.get(int(i), -1) for i in pred.item_indices])
                    elif model == "Elo":
                        obs_weeks = np.array([quid_to_week.get(int(i), -1) for i in pred.item_indices])
                    else:
                        obs_weeks = item_week[pred.item_indices]

                    # Overall per-student tau (across all test weeks)
                    students = np.unique(pred.student_indices)
                    taus = []
                    for sid in students:
                        mask = pred.student_indices == sid
                        if mask.sum() < 10:
                            continue
                        yt, yp = pred.y_true[mask], pred.y_pred_prob[mask]
                        if yt.std() == 0 or yp.std() == 0:
                            continue
                        tau, _ = kendalltau(yp, yt)
                        taus.append(tau)
                    if taus:
                        rows.append({
                            "course": course, "model": model, "horizon": horizon,
                            "test_week": "all", "distance": "all",
                            "mean_tau": np.mean(taus), "n_students": len(taus),
                        })

                    # Per test-week per-student tau
                    for tw in sorted(set(obs_weeks)):
                        if tw <= horizon or tw < 0 or tw > 6:
                            continue
                        tw_taus = []
                        for sid in students:
                            mask = (pred.student_indices == sid) & (obs_weeks == tw)
                            if mask.sum() < 5:
                                continue
                            yt, yp = pred.y_true[mask], pred.y_pred_prob[mask]
                            if yt.std() == 0 or yp.std() == 0:
                                continue
                            tau, _ = kendalltau(yp, yt)
                            tw_taus.append(tau)
                        if tw_taus:
                            rows.append({
                                "course": course, "model": model, "horizon": horizon,
                                "test_week": int(tw), "distance": int(tw - horizon),
                                "mean_tau": np.mean(tw_taus), "n_students": len(tw_taus),
                            })

            print(f"  {course}: done")

        df = pd.DataFrame(rows)
        df.to_csv(CACHE, index=False)
        print(f"Saved: {len(df)} rows to {CACHE}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.8))

    # (a) Kendall Tau vs horizon (all test weeks aggregated)
    overall = df[df["test_week"] == "all"]
    for model in MODELS:
        sub = overall[overall["model"] == model]
        grouped = sub.groupby("horizon")["mean_tau"]
        means = grouped.mean().sort_index()
        color = MODEL_COLORS[model]
        lw = 2.0 if model == "RSSM" else 1.3
        ms = 5 if model == "RSSM" else 3.5
        ax1.plot(means.index, means.values, "o-", color=color,
                 label=model, linewidth=lw, markersize=ms)

    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Train Cutoff Week")
    ax1.set_ylabel(r"Kendall $\tau$ (per-student avg)")
    ax1.set_xticks(sorted(overall["horizon"].unique()))
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", va="top")

    # (b) Kendall Tau vs prediction distance (horizon 1 only)
    h1 = df[(df["horizon"] == 1) & (df["test_week"] != "all")]
    h1 = h1.copy()
    h1["distance"] = h1["distance"].astype(int)
    for model in MODELS:
        sub = h1[h1["model"] == model]
        grouped = sub.groupby("distance")["mean_tau"]
        means = grouped.mean().sort_index()
        color = MODEL_COLORS[model]
        lw = 2.0 if model == "RSSM" else 1.3
        ms = 5 if model == "RSSM" else 3.5
        ax2.plot(means.index, means.values, "o-", color=color,
                 label=model, linewidth=lw, markersize=ms)

    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Prediction Distance (weeks ahead)")
    ax2.set_ylabel(r"Kendall $\tau$ (per-student avg)")
    ax2.set_xticks(sorted(h1["distance"].unique()))
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes,
             fontsize=9, fontweight="bold", va="top")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODELS),
               fontsize=7, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    out = "overleaf/figures/proto_kendall_v1.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def load_horizon_predictions(horizon=3):
    """Load predictions for all models at a given horizon for dsa_hk231."""
    preds = {}
    for model in MODELS:
        with open(f"{RESULTS_DIR}/{model}_predictions.pkl", "rb") as f:
            mp = pickle.load(f)
        preds[model] = mp[horizon]
    return preds


def compute_per_student(pred, model):
    """Return arrays of (actual_mean, pred_mean, pred_std) per student."""
    students = np.unique(pred.student_indices)
    actual_means, pred_means, pred_stds = [], [], []
    for sid in students:
        mask = pred.student_indices == sid
        if mask.sum() < 10:
            continue
        yt = pred.y_true[mask]
        yp = pred.y_pred_prob[mask]
        actual_means.append(yt.mean())
        pred_means.append(yp.mean())
        pred_stds.append(yp.std())
    return np.array(actual_means), np.array(pred_means), np.array(pred_stds)


def load_all_courses_predictions(horizon=3):
    """Load predictions for all 4 courses at a given horizon."""
    COURSES = ["dsa_hk231", "dsa_hk221", "pf_hk232", "pf_hk222"]
    all_preds = {}
    for model in MODELS:
        yp_list, yt_list = [], []
        for course in COURSES:
            path = f"results/temporal_eval_full/{course}/{model}_predictions.pkl"
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                mp = pickle.load(f)
            if horizon not in mp:
                continue
            pred = mp[horizon]
            yp_list.append(pred.y_pred_prob)
            yt_list.append(pred.y_true)
        all_preds[model] = (np.concatenate(yp_list), np.concatenate(yt_list))
    return all_preds


def proto_fig_a_calibration(preds):
    """Calibration curve: binned predicted P(correct) vs actual pass rate, all courses."""
    all_course_preds = load_all_courses_predictions(horizon=3)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.2))
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    for model in MODELS:
        yp, yt = all_course_preds[model]
        bin_centers, bin_actuals = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (yp >= lo) & (yp < hi)
            if mask.sum() < 20:
                continue
            bin_centers.append((lo + hi) / 2)
            bin_actuals.append(yt[mask].mean())
        ax.plot(bin_centers, bin_actuals, "o-", color=MODEL_COLORS[model],
                linewidth=1.3, markersize=3.5, label=model)

    ax.set_xlabel("Predicted $P$(correct)")
    ax.set_ylabel("Actual pass rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = "overleaf/figures/proto_calibration.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


HORIZON_COLORS = {1: "#4477aa", 2: "#66ccee", 3: "#228833", 4: "#ccbb44", 5: "#ee6677"}


def calibration_bins(yp, yt, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, actuals = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (yp >= lo) & (yp < hi)
        if mask.sum() < 20:
            continue
        centers.append((lo + hi) / 2)
        actuals.append(yt[mask].mean())
    return centers, actuals


def proto_fig_a2_calibration_dsa(preds):
    """Two-panel calibration for dsa_hk231: (a) all models at H3, (b) RSSM across horizons."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 6))

    # (a) All 4 models at horizon 3
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    for model in MODELS:
        pred = preds[model]
        centers, actuals = calibration_bins(pred.y_pred_prob, pred.y_true)
        ax1.plot(centers, actuals, "o-", color=MODEL_COLORS[model],
                 linewidth=1.3, markersize=3.5, label=model)
    ax1.set_xlabel("Predicted $P$(correct)")
    ax1.set_ylabel("Actual pass rate")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title("(a) Model comparison")

    # (b) RSSM across horizons 1-5
    ax2.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    with open(f"{RESULTS_DIR}/RSSM_predictions.pkl", "rb") as f:
        all_rssm = pickle.load(f)
    for h in sorted(all_rssm.keys()):
        if h > 5:
            continue
        pred = all_rssm[h]
        centers, actuals = calibration_bins(pred.y_pred_prob, pred.y_true)
        ax2.plot(centers, actuals, "o-", color=HORIZON_COLORS[h],
                 linewidth=1.3, markersize=3.5, label=f"Cutoff W{h}")
    ax2.set_xlabel("Predicted $P$(correct)")
    ax2.set_ylabel("Actual pass rate")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title("(b) RSSM across training horizons")

    fig.tight_layout()
    out = "overleaf/figures/calibration_dsa.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


COURSES_ALL = {
    "dsa_hk231": "DSA 2023",
    "dsa_hk221": "DSA 2022",
    "pf_hk232": "PF 2023",
    "pf_hk222": "PF 2022",
}


def proto_fig_calibration_grid():
    """Calibration curves: 4 models x (4 courses x 5 horizons) grid."""
    courses = list(COURSES_ALL.keys())
    n_courses = len(courses)
    horizons = [1, 2, 3, 4, 5]
    n_horizons = len(horizons)

    fig, axes = plt.subplots(n_courses, n_horizons, figsize=(12, 9),
                             sharex=True, sharey=True)

    for row, course in enumerate(courses):
        for col, h in enumerate(horizons):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)

            for model in MODELS:
                path = f"results/temporal_eval_full/{course}/{model}_predictions.pkl"
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as f:
                    mp = pickle.load(f)
                if h not in mp:
                    continue
                pred = mp[h]
                centers, actuals = calibration_bins(pred.y_pred_prob, pred.y_true)
                ax.plot(centers, actuals, "o-", color=MODEL_COLORS[model],
                        linewidth=1.3, markersize=2.5, label=model if row == 0 and col == 0 else None)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(f"Cutoff W{h}")
            if col == 0:
                ax.set_ylabel(COURSES_ALL[course])
            if row == n_courses - 1:
                ax.set_xlabel("Predicted $P$(correct)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODELS),
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout()
    out = "overleaf/figures/calibration_grid.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def proto_fig_b_discrimination(preds):
    """Discrimination: distribution of P(correct) for correct vs incorrect outcomes."""
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.2), sharey=True)

    for ax, model in zip(axes, MODELS):
        pred = preds[model]
        yp = pred.y_pred_prob
        yt = pred.y_true

        bins = np.linspace(0, 1, 30)
        ax.hist(yp[yt == 0], bins=bins, alpha=0.5, color="#d62728", density=True, label="Incorrect")
        ax.hist(yp[yt == 1], bins=bins, alpha=0.5, color="#2ca02c", density=True, label="Correct")
        ax.set_title(model, fontsize=8)
        ax.set_xlabel("Predicted $P$(correct)", fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Density", fontsize=7)
    axes[-1].legend(fontsize=6, loc="upper center")

    fig.tight_layout()
    out = "overleaf/figures/proto_discrimination.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def proto_fig_c_ability_scatter(preds):
    """Predicted vs actual student ability (mean P(correct) per student)."""
    from scipy.stats import kendalltau

    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.2), sharey=True)

    for ax, model in zip(axes, MODELS):
        actual_means, pred_means, _ = compute_per_student(preds[model], model)
        tau, _ = kendalltau(pred_means, actual_means)

        ax.scatter(actual_means, pred_means, s=3, alpha=0.3, color=MODEL_COLORS[model])
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{model} ($\\tau$={tau:.3f})", fontsize=8)
        ax.set_xlabel("Actual pass rate", fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean predicted $P$(correct)", fontsize=7)

    fig.tight_layout()
    out = "overleaf/figures/proto_ability_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def proto_fig_d_prediction_variance(preds):
    """Per-student prediction variance across questions (boxplot)."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.8))

    all_stds = []
    labels = []
    colors = []
    for model in MODELS:
        _, _, pred_stds = compute_per_student(preds[model], model)
        all_stds.append(pred_stds)
        labels.append(model)
        colors.append(MODEL_COLORS[model])

    bp = ax.boxplot(all_stds, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Std of predicted $P$(correct)\nacross questions per student", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = "overleaf/figures/proto_pred_variance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def proto_fig_e_ranking_bar(preds):
    """Bar chart: Kendall Tau between model's student ranking and actual ranking."""
    from scipy.stats import kendalltau

    taus = []
    for model in MODELS:
        actual_means, pred_means, _ = compute_per_student(preds[model], model)
        tau, _ = kendalltau(pred_means, actual_means)
        taus.append(tau)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))
    bars = ax.bar(MODELS, taus, color=[MODEL_COLORS[m] for m in MODELS], width=0.5, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel(r"Kendall $\tau$ (student ranking vs actual)", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, tau in zip(bars, taus):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{tau:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out = "overleaf/figures/proto_ranking_bar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    preds = load_horizon_predictions(horizon=3)

    proto_fig_a_calibration(preds)
    proto_fig_a2_calibration_dsa(preds)
    proto_fig_calibration_grid()
    proto_fig_b_discrimination(preds)
    proto_fig_c_ability_scatter(preds)
    proto_fig_d_prediction_variance(preds)
    proto_fig_e_ranking_bar(preds)


if __name__ == "__main__":
    main()
