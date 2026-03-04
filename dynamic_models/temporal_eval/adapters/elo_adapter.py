"""Elo model adapter for temporal evaluation.

Processes interactions chronologically. Only updates ability/difficulty
on train-week items; predicts on test-week items without updating.
"""

import numpy as np
import pandas as pd

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..temporal_split import TemporalSplit


class EloAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "Elo"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        K: float = 0.4,
        **kwargs,
    ) -> PredictionResult:
        np.random.seed(seed)

        # Use pre-built question_unittest_id -> week mapping
        qid_to_week = data.qid_to_week

        # Prepare Elo-format data from main_data
        df = data.main_data.copy()

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
        )
        df = df.dropna(subset=["timestamp"])

        # Compute time since first submission per student
        df["T"] = df.groupby("student_id")["timestamp"].transform(
            lambda x: (x - x.min()).dt.total_seconds()
        )

        # Compute pass fraction per submission
        def pass_fraction(s):
            s = str(s).strip()
            if "." in s:
                try:
                    s = str(int(float(s)))
                except ValueError:
                    return np.nan
            return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan

        df["ItemScore"] = df["pass"].apply(pass_fraction)
        df = df.dropna(subset=["ItemScore"])

        # Map each question_unittest_id to its week
        df["week"] = df["question_unittest_id"].map(qid_to_week)
        df = df.dropna(subset=["week"])
        df["week"] = df["week"].astype(int)

        # Determine train/test based on week
        df["is_train"] = df["week"] <= split.cutoff_week

        # Sort chronologically per student
        df = df.sort_values(["student_id", "T"]).reset_index(drop=True)

        # Run Elo with temporal split:
        # - Process ALL interactions chronologically
        # - Only update theta/difficulty on train-week items
        # - Record predictions on test-week items
        theta = {}  # student_id -> current ability
        difficulty = {}  # question_unittest_id -> current difficulty

        test_y_true = []
        test_y_pred = []
        test_student_ids = []
        test_item_ids = []

        for _, row in df.iterrows():
            sid = row["student_id"]
            qid = row["question_unittest_id"]

            # Initialize if new
            if sid not in theta:
                theta[sid] = 0.0
            if qid not in difficulty:
                difficulty[qid] = 0.0

            # Compute prediction
            p = 1.0 / (1.0 + np.exp(-(theta[sid] - difficulty[qid])))

            if row["is_train"]:
                # Update theta and difficulty
                resp = row["ItemScore"]
                theta[sid] += K * (resp - p)
                difficulty[qid] -= K * (resp - p)
            else:
                # Record prediction for test item
                # Binarize: all tests pass = 1
                binary_true = 1.0 if row["ItemScore"] >= 1.0 else 0.0
                test_y_true.append(binary_true)
                test_y_pred.append(p)
                test_student_ids.append(sid)
                test_item_ids.append(qid)

        if len(test_y_true) == 0:
            raise ValueError(
                f"No test observations for cutoff_week={split.cutoff_week}"
            )

        return PredictionResult(
            y_true=np.array(test_y_true),
            y_pred_prob=np.array(test_y_pred),
            student_indices=np.array(test_student_ids),
            item_indices=np.array(test_item_ids),
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 1.0
