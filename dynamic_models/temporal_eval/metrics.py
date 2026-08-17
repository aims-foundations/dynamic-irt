"""Unified metric computation for temporal evaluation."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


@dataclass
class EvalMetrics:
    """Standardized metrics computed on every model x horizon combination."""

    auc: float
    accuracy: float
    balanced_accuracy: float
    f1: float
    log_likelihood: float
    brier: float
    rmse: float
    n_test_obs: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "auc": self.auc,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1": self.f1,
            "log_likelihood": self.log_likelihood,
            "brier": self.brier,
            "rmse": self.rmse,
            "n_test_obs": self.n_test_obs,
        }


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> EvalMetrics:
    """Compute all standardized metrics.

    Args:
        y_true: Binary ground truth (0 or 1), shape [N].
        y_pred_prob: Predicted P(correct), shape [N], values in [0, 1].
    """
    eps = 1e-7
    y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # AUC, balanced accuracy, and F1 are undefined when the test slice
    # contains a single class; report nan so degenerate slices are visible
    # instead of masquerading as chance-level scores.
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        f1 = float("nan")
        balanced_accuracy = float("nan")
    else:
        auc = roc_auc_score(y_true, y_pred_prob)
        f1 = f1_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    ll = np.mean(
        y_true * np.log(y_pred_prob)
        + (1 - y_true) * np.log(1 - y_pred_prob)
    )

    brier = brier_score_loss(y_true, y_pred_prob)

    rmse = np.sqrt(np.mean((y_true - y_pred_prob) ** 2))

    return EvalMetrics(
        auc=auc,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        f1=f1,
        log_likelihood=ll,
        brier=brier,
        rmse=rmse,
        n_test_obs=len(y_true),
    )
