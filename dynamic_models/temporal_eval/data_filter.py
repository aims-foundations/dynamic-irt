"""Shared data filtering for temporal evaluation.

Single entry point: filter_data(data, config) applies quality thresholds,
subsets the data to passing students/items, and caps the attempt dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .data_loader import UnifiedData


@dataclass
class DataFilterConfig:
    max_week: int = 3
    min_pass_rate: float = 0.10
    max_pass_rate: float = 0.90
    min_question_coverage: float = 0.25
    max_attempts: int = 10


DEFAULT_FILTER = DataFilterConfig()


def _compute_filter_indices(correctness_matrix, question_infos, config):
    """Compute which students/items pass quality thresholds.

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

    # Keep students with at least one observation on the selected questions;
    # empty rows carry no signal and break the adapters.
    sub_matrix = matrix[:, col_indices]
    valid_counts = np.sum(sub_matrix != -1, axis=1)
    active_students = np.where(valid_counts > 0)[0]

    return active_students, col_indices, selected_qs


def filter_data(data: "UnifiedData", config: DataFilterConfig = None) -> "UnifiedData":
    """Apply quality filter, subset data, and cap attempts. Single entry point.

    Steps:
      1. Select questions/students passing quality thresholds
      2. Subset matrices and metadata to those indices
      3. Truncate attempt dimension to config.max_attempts
    """
    from .data_loader import UnifiedData

    if config is None:
        config = DEFAULT_FILTER

    student_idx, item_idx, _ = _compute_filter_indices(
        data.correctness_matrix, data.question_infos, config
    )

    if len(item_idx) == 0 or len(student_idx) == 0:
        raise ValueError("No students or items passed quality filter")

    corr = data.correctness_matrix[student_idx][:, item_idx]
    time = data.time_matrix[student_idx][:, item_idx]
    qi = data.question_infos.iloc[item_idx].reset_index(drop=True)
    item_week = data.item_week[item_idx]
    student_ids = [data.student_ids[i] for i in student_idx]

    # Cap attempts
    max_a = config.max_attempts
    if corr.shape[2] > max_a:
        corr = corr[:, :, :max_a]
        time = time[:, :, :max_a]

    # Filter main_data CSV
    sid_set = set(student_ids)
    main_data = data.main_data[data.main_data["student_id"].isin(sid_set)].copy()
    if "question_unittest_id" in data.main_data.columns:
        kept_qids = set()
        for idx in item_idx:
            row = data.question_infos.iloc[idx]
            if "question_unittest_id" in row.index:
                kept_qids.add(int(row["question_unittest_id"]))
        if kept_qids:
            main_data = main_data[
                main_data["question_unittest_id"].isin(kept_qids)
            ].copy()

    n_s, n_i, n_a = corr.shape
    return UnifiedData(
        main_data=main_data,
        correctness_matrix=corr,
        time_matrix=time,
        question_infos=qi,
        item_week=item_week,
        qid_to_week=data.qid_to_week,
        student_ids=student_ids,
        n_students=n_s,
        n_items=n_i,
        n_max_attempts=n_a,
        course_name=data.course_name,
    )


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
