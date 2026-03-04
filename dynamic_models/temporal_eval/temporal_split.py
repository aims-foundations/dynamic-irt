"""Week-based temporal split generation.

Creates train/test splits where train items come from weeks 1..W
and test items come from weeks W+1..max_week.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class TemporalSplit:
    """A single week-based train/test split."""

    cutoff_week: int

    # Item-level masks (length n_items)
    train_item_mask: torch.BoolTensor
    test_item_mask: torch.BoolTensor

    # Convenience indices
    train_item_indices: torch.LongTensor
    test_item_indices: torch.LongTensor

    # Stats
    n_train_items: int
    n_test_items: int
    train_weeks: List[int] = field(default_factory=list)
    test_weeks: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"TemporalSplit(W={self.cutoff_week}, "
            f"train={self.n_train_items} items weeks {self.train_weeks}, "
            f"test={self.n_test_items} items weeks {self.test_weeks})"
        )


def generate_temporal_splits(
    item_week: torch.Tensor,
    cutoff_weeks: Optional[List[int]] = None,
) -> List[TemporalSplit]:
    """Generate multiple temporal splits from week assignments.

    Args:
        item_week: [n_items] tensor of week numbers (integers).
        cutoff_weeks: List of cutoff week values W. Train on weeks 1..W,
            test on weeks W+1..max_week. If None, auto-generates
            all valid cutoffs (1 through max_week-1).

    Returns:
        List of TemporalSplit objects, one per valid cutoff week.
    """
    unique_weeks = sorted(w for w in item_week.unique().tolist() if w > 0)

    if not unique_weeks:
        raise ValueError("No valid weeks found in item_week (all zeros)")

    max_week = max(unique_weeks)

    if cutoff_weeks is None:
        cutoff_weeks = list(range(1, max_week))

    splits = []
    for W in cutoff_weeks:
        train_mask = (item_week > 0) & (item_week <= W)
        test_mask = item_week > W

        if train_mask.sum() == 0:
            print(f"  WARNING: cutoff_week={W} produces empty train set, skipping")
            continue
        if test_mask.sum() == 0:
            print(f"  WARNING: cutoff_week={W} produces empty test set, skipping")
            continue

        splits.append(TemporalSplit(
            cutoff_week=W,
            train_item_mask=train_mask,
            test_item_mask=test_mask,
            train_item_indices=torch.where(train_mask)[0],
            test_item_indices=torch.where(test_mask)[0],
            n_train_items=int(train_mask.sum()),
            n_test_items=int(test_mask.sum()),
            train_weeks=[w for w in unique_weeks if w <= W],
            test_weeks=[w for w in unique_weeks if w > W],
        ))

    return splits
