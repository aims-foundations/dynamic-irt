"""Student-based train/test split for evaluation.

Splits students 80/20 for calibration/scoring. All items remain
visible to both sets — the split is purely on the student dimension.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class StudentSplit:
    train_student_indices: np.ndarray
    test_student_indices: np.ndarray
    train_item_indices: np.ndarray  # items from weeks 1..W (scoring phase input)
    test_item_indices: np.ndarray   # items from weeks W+1..end (prediction target)
    train_week_cutoff: int
    n_train_students: int
    n_test_students: int

    def __repr__(self):
        return (
            f"StudentSplit(train={self.n_train_students} students, "
            f"test={self.n_test_students} students, "
            f"scoring_items={len(self.train_item_indices)} [weeks 1-{self.train_week_cutoff}], "
            f"prediction_items={len(self.test_item_indices)} [weeks {self.train_week_cutoff+1}+])"
        )


def generate_student_split(
    n_students: int,
    item_week: torch.Tensor,
    train_week_cutoff: int = 3,
    test_frac: float = 0.3,
    seed: int = 42,
) -> StudentSplit:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_students)
    n_test = int(n_students * test_frac)
    test_idx = np.sort(indices[:n_test])
    train_idx = np.sort(indices[n_test:])

    week_np = item_week.numpy()
    train_item_idx = np.where((week_np > 0) & (week_np <= train_week_cutoff))[0]
    test_item_idx = np.where(week_np > train_week_cutoff)[0]

    return StudentSplit(
        train_student_indices=train_idx,
        test_student_indices=test_idx,
        train_item_indices=train_item_idx,
        test_item_indices=test_item_idx,
        train_week_cutoff=train_week_cutoff,
        n_train_students=len(train_idx),
        n_test_students=len(test_idx),
    )
