"""Abstract base class for model adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .data_loader import UnifiedData
from .temporal_split import TemporalSplit


@dataclass
class PredictionResult:
    """Standardized prediction output from any model."""

    y_true: np.ndarray  # Binary ground truth (0 or 1), shape [N_test_obs]
    y_pred_prob: np.ndarray  # Predicted P(correct), shape [N_test_obs]

    # Optional metadata for diagnostic breakdown
    student_indices: Optional[np.ndarray] = None
    item_indices: Optional[np.ndarray] = None
    testcase_indices: Optional[np.ndarray] = None
    attempt_indices: Optional[np.ndarray] = None

    # Training diagnostics
    losses: Optional[Dict[str, List[float]]] = None  # e.g. {"train": [...], "test": [...]}

    # Learned parameter distributions (param_name -> 1D array of values)
    student_params: Optional[Dict[str, np.ndarray]] = None
    item_params: Optional[Dict[str, np.ndarray]] = None

    # Serializable model state for saving/loading trained models
    model_state: Optional[Dict] = None

    @property
    def n_observations(self) -> int:
        return len(self.y_true)


class ModelAdapter(ABC):
    """Abstract interface that each model must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @abstractmethod
    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        **kwargs,
    ) -> PredictionResult:
        """Train on train items, predict on test items.

        The model must ONLY use observations from train items for fitting.
        It must produce P(correct) predictions for ALL valid observations
        in the test items.
        """
        ...

    def fit_and_predict_student_split(
        self,
        data: "UnifiedData",
        split: "StudentSplit",
        seed: int = 42,
        **kwargs,
    ) -> PredictionResult:
        """Train on train students, calibrate test students, predict.

        Training: learn item params from train students (all weeks).
        Calibration: estimate test student ability from weeks 1..W.
        Prediction: predict test students' weeks W+1..end.
        """
        raise NotImplementedError(f"{self.name} does not support student splits")

    @abstractmethod
    def estimated_runtime_minutes(self, data: "UnifiedData") -> float:
        """Rough runtime estimate so the harness can plan."""
        ...
