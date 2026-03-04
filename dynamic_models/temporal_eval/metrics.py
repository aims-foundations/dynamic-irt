"""Unified metric computation for temporal evaluation."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class EvalMetrics:
    """Standardized metrics computed on every model x horizon combination."""

    auc: float
    accuracy: float
    f1: float
    log_likelihood: float
    rmse: float
    n_test_obs: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "auc": self.auc,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "log_likelihood": self.log_likelihood,
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

    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.5

    try:
        f1 = f1_score(y_true, y_pred)
    except ValueError:
        f1 = 0.0

    accuracy = accuracy_score(y_true, y_pred)

    ll = np.mean(
        y_true * np.log(y_pred_prob)
        + (1 - y_true) * np.log(1 - y_pred_prob)
    )

    rmse = np.sqrt(np.mean((y_true - y_pred_prob) ** 2))

    return EvalMetrics(
        auc=auc,
        accuracy=accuracy,
        f1=f1,
        log_likelihood=ll,
        rmse=rmse,
        n_test_obs=len(y_true),
    )
