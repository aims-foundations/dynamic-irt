"""Student-based train/val/test split for evaluation.

Students are shuffled once (seeded); the first `test_frac` become test
students and validation students are carved from the remainder, so the
test set is unchanged by the addition of a validation set. All items
remain visible to every set — the split is purely on the student
dimension. Validation students are excluded from training and exist so
model selection (checkpointing, early stopping) never touches the test
set.
"""

from dataclasses import dataclass, field

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
    val_student_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    n_val_students: int = 0

    def __repr__(self):
        return (
            f"StudentSplit(train={self.n_train_students} students, "
            f"val={self.n_val_students} students, "
            f"test={self.n_test_students} students, "
            f"scoring_items={len(self.train_item_indices)} [weeks 1-{self.train_week_cutoff}], "
            f"prediction_items={len(self.test_item_indices)} [weeks {self.train_week_cutoff+1}+])"
        )


def generate_student_split(
    n_students: int,
    item_week: torch.Tensor,
    train_week_cutoff: int = 3,
    test_frac: float = 0.3,
    val_frac: float = 0.15,
    seed: int = 42,
) -> StudentSplit:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_students)
    n_test = int(n_students * test_frac)
    n_val = int(n_students * val_frac)
    test_idx = np.sort(indices[:n_test])
    val_idx = np.sort(indices[n_test:n_test + n_val])
    train_idx = np.sort(indices[n_test + n_val:])

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
        val_student_indices=val_idx,
        n_val_students=len(val_idx),
    )
