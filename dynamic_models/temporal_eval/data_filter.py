"""Shared data filtering thresholds for temporal evaluation.

These filters select a meaningful subset of questions and students
from the raw correctness matrix, used consistently across visualization
and model training.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DataFilterConfig:
    max_week: int = 3
    min_pass_rate: float = 0.10
    max_pass_rate: float = 0.90
    min_question_coverage: float = 0.25
    min_student_item_coverage: float = 0.50
    min_student_progress: float = 0.10
    max_attempts: int = 10


DEFAULT_FILTER = DataFilterConfig()


def apply_filter(correctness_matrix, question_infos, config=None):
    """Apply filtering thresholds to identify valid questions and students.

    Args:
        correctness_matrix: [n_students, n_items, n_max_attempts] tensor, -1 = missing.
        question_infos: DataFrame with columns qidx, week.
        config: DataFilterConfig. Uses DEFAULT_FILTER if None.

    Returns:
        Tuple of (student_indices, item_indices, selected_qidxs) as numpy arrays.
    """
    if config is None:
        config = DEFAULT_FILTER

    corr = correctness_matrix
    qi = question_infos
    n_students = corr.shape[0]

    # Collapse to last attempt (capped)
    matrix = _collapse_last_attempt(corr, max_attempt=config.max_attempts)

    # Filter questions
    qidx_to_week = dict(zip(qi["qidx"], qi["week"]))
    q_stats = []
    for qidx in qi["qidx"].unique():
        if qidx_to_week.get(qidx, 99) > config.max_week:
            continue
        item_mask = qi["qidx"] == qidx
        item_indices = np.where(item_mask)[0]
        sub = matrix[:, item_indices]
        students_attempted = np.any(sub != -1, axis=1).sum()
        coverage = students_attempted / n_students
        valid = sub[sub != -1]
        if len(valid) == 0:
            continue
        pass_rate = valid.mean()
        if config.min_pass_rate <= pass_rate <= config.max_pass_rate and coverage > config.min_question_coverage:
            q_stats.append((qidx, pass_rate, coverage))

    q_stats.sort(key=lambda x: -x[1])
    selected_qs = [q[0] for q in q_stats]

    # Collect item columns
    col_indices = []
    for qidx in selected_qs:
        item_mask = qi["qidx"] == qidx
        col_indices.extend(np.where(item_mask)[0])
    col_indices = np.array(col_indices)

    if len(col_indices) == 0:
        return np.array([], dtype=int), col_indices, selected_qs

    # Filter students: item coverage
    sub_matrix = matrix[:, col_indices]
    valid_counts = np.sum(sub_matrix != -1, axis=1)
    active_students = np.where(valid_counts > len(col_indices) * config.min_student_item_coverage)[0]

    # Filter students: progress between attempt 1 and attempt N
    if config.min_student_progress > 0:
        snapshots = _get_attempt_snapshots(corr, max_attempt=config.max_attempts)
        snap_first = snapshots[0][:, col_indices]
        snap_last = snapshots[-1][:, col_indices]
        progressing = []
        for s in active_students:
            v1 = snap_first[s][snap_first[s] != -1]
            vn = snap_last[s][snap_last[s] != -1]
            s1 = v1.mean() if len(v1) > 0 else 0.0
            sn = vn.mean() if len(vn) > 0 else 0.0
            if abs(sn - s1) >= config.min_student_progress:
                progressing.append(s)
        active_students = np.array(progressing) if progressing else np.array([], dtype=int)

    return active_students, col_indices, selected_qs


def _collapse_last_attempt(corr, max_attempt=None):
    n_s, n_i, n_a = corr.shape
    result = torch.full((n_s, n_i), -1, dtype=corr.dtype)
    if max_attempt is not None:
        count = torch.zeros((n_s, n_i), dtype=torch.long)
        for a in range(n_a):
            valid = corr[:, :, a] != -1
            under_cap = count < max_attempt
            use = valid & under_cap
            result[use] = corr[:, :, a][use]
            count[valid] += 1
    else:
        for a in range(n_a):
            valid = corr[:, :, a] != -1
            result[valid] = corr[:, :, a][valid]
    return result.numpy().astype(float)


def _get_attempt_snapshots(corr, max_attempt=10):
    n_s, n_i, n_a = corr.shape
    arr = corr.numpy()
    valid_mask = arr != -1
    cum_valid = np.cumsum(valid_mask, axis=2)

    prev = np.full((n_s, n_i), -1.0)
    snapshots = []
    for a in range(max_attempt):
        result = prev.copy()
        target = a + 1
        hits = cum_valid == target
        has_hit = hits.any(axis=2)
        first_idx = np.argmax(hits, axis=2)
        s_idx, i_idx = np.where(has_hit)
        result[s_idx, i_idx] = arr[s_idx, i_idx, first_idx[s_idx, i_idx]]
        snapshots.append(result)
        prev = result
    return snapshots
