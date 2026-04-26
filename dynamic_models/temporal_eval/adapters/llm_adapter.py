"""LLM Predictive Model (LLM-P2) adapter for temporal evaluation.

Loads pre-generated LLM simulation data from HuggingFace
(CodeInsightTeam/simulation_output) and uses graded code outcomes
as predictions for student performance.

The LLM generates C++ code conditioned on student history;
the code is compiled and graded against test cases. The test-case
pass fraction serves as the predicted P(correct).
"""

import json
import logging
import os
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..temporal_split import TemporalSplit

logger = logging.getLogger(__name__)

HF_SIM_REPO = "CodeInsightTeam/simulation_output"
V4_SUBDIR = "v4_profile_mindiff"

_sim_cache: Optional[pd.DataFrame] = None


def _pass_fraction(s):
    s = str(s).strip()
    if not s or s == "nan":
        return np.nan
    return sum(c == "1" for c in s) / len(s) if len(s) > 0 else np.nan


def _load_simulation_data(sim_data_path: Optional[str] = None) -> pd.DataFrame:
    global _sim_cache
    if _sim_cache is not None:
        logger.info("Using cached simulation data (%d rows)", len(_sim_cache))
        return _sim_cache

    if sim_data_path and os.path.exists(sim_data_path):
        jsonl_files = [sim_data_path]
    else:
        repo_dir = snapshot_download(
            repo_id=HF_SIM_REPO, repo_type="dataset",
            local_files_only=True,
        )
        shard_pattern = os.path.join(
            repo_dir, V4_SUBDIR, "glm_server_n10_attempts50_shard*of*.jsonl"
        )
        jsonl_files = sorted(glob(shard_pattern))
        if not jsonl_files:
            merged = os.path.join(repo_dir, V4_SUBDIR, "glm_v4_merged.jsonl")
            if os.path.exists(merged):
                jsonl_files = [merged]
            else:
                raise FileNotFoundError(
                    f"No simulation JSONL found in {repo_dir}/{V4_SUBDIR}/"
                )

    logger.info("Loading simulation data from %d file(s)…", len(jsonl_files))

    keep_cols = [
        "student_id", "question_unittest_id", "attempt_id",
        "response_type", "pass",
    ]
    rows = []
    for fpath in jsonl_files:
        logger.info("  Reading %s", os.path.basename(fpath))
        with open(fpath) as f:
            for line in f:
                rec = json.loads(line)
                rows.append({k: rec.get(k) for k in keep_cols})

    df = pd.DataFrame(rows)
    df["student_id"] = df["student_id"].astype(str)
    df["question_unittest_id"] = pd.to_numeric(
        df["question_unittest_id"], errors="coerce"
    )
    df["attempt_id"] = pd.to_numeric(df["attempt_id"], errors="coerce")
    df = df.dropna(subset=["question_unittest_id"])
    df["question_unittest_id"] = df["question_unittest_id"].astype(int)

    logger.info("Loaded %d simulation rows", len(df))
    logger.info(
        "  Unique students: %d, unique questions: %d",
        df["student_id"].nunique(),
        df["question_unittest_id"].nunique(),
    )

    _sim_cache = df
    return df


class LLMAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "LLM"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        sim_data_path: Optional[str] = None,
        attempt_mode: str = "first_submit",
        **kwargs,
    ) -> PredictionResult:
        sim_df = _load_simulation_data(sim_data_path)

        qid_to_week = data.qid_to_week

        # Map simulation questions to weeks, keep only test-week items
        sim_df = sim_df.copy()
        sim_df["week"] = sim_df["question_unittest_id"].map(qid_to_week)
        test_sim = sim_df[sim_df["week"] > split.cutoff_week].copy()

        if len(test_sim) == 0:
            raise ValueError(
                f"No simulation data for test weeks > {split.cutoff_week}"
            )

        # Select which attempt(s) to use as prediction
        if attempt_mode == "first_submit":
            submits = test_sim[test_sim["response_type"] == "Submit"]
            submits = submits.sort_values("attempt_id")
            pred_df = submits.groupby(
                ["student_id", "question_unittest_id"]
            ).first().reset_index()
        elif attempt_mode == "best_submit":
            submits = test_sim[test_sim["response_type"] == "Submit"].copy()
            submits["score"] = submits["pass"].apply(_pass_fraction)
            idx = submits.groupby(
                ["student_id", "question_unittest_id"]
            )["score"].idxmax()
            pred_df = submits.loc[idx].reset_index(drop=True)
        else:
            # Use first attempt regardless of type
            test_sim = test_sim.sort_values("attempt_id")
            pred_df = test_sim.groupby(
                ["student_id", "question_unittest_id"]
            ).first().reset_index()

        pred_df["y_pred"] = pred_df["pass"].apply(_pass_fraction)
        pred_df = pred_df.dropna(subset=["y_pred"])
        print(pred_df)

        # Get real student outcomes for the same test-week items
        real_df = data.main_data.copy()
        real_df["student_id"] = real_df["student_id"].astype(str)
        real_df["question_unittest_id"] = pd.to_numeric(
            real_df["question_unittest_id"], errors="coerce"
        )
        real_df = real_df.dropna(subset=["question_unittest_id", "pass"])
        real_df["question_unittest_id"] = real_df["question_unittest_id"].astype(int)
        real_df["week"] = real_df["question_unittest_id"].map(qid_to_week)
        real_test = real_df[real_df["week"] > split.cutoff_week].copy()

        real_test["score"] = real_test["pass"].apply(_pass_fraction)
        real_test = real_test.dropna(subset=["score"])

        # Aggregate real outcomes: best score per (student, question)
        real_best = real_test.groupby(
            ["student_id", "question_unittest_id"]
        )["score"].max().reset_index()
        real_best["y_true"] = (real_best["score"] >= 1.0).astype(float)

        # Inner join: only pairs in both simulation and real data
        merged = pred_df[["student_id", "question_unittest_id", "y_pred"]].merge(
            real_best[["student_id", "question_unittest_id", "y_true"]],
            on=["student_id", "question_unittest_id"],
            how="inner",
        )

        if len(merged) == 0:
            raise ValueError(
                f"No matching (student, question) pairs between simulation "
                f"and real data for cutoff_week={split.cutoff_week}"
            )

        logger.info(
            "  LLM predictions: %d matched pairs (sim=%d, real=%d, overlap=%d)",
            len(merged), len(pred_df),
            len(real_best), len(merged),
        )

        return PredictionResult(
            y_true=merged["y_true"].values,
            y_pred_prob=merged["y_pred"].values,
            student_indices=merged["student_id"].values,
            item_indices=merged["question_unittest_id"].values,
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 0.5
