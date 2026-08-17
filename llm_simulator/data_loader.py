"""Data structures, loading, and utilities for LLM student simulation.

Defines the EvalItem dataclass, test-case parsing, and the
student-split data loader that bridges the psychometric evaluation
framework to LLM eval items.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from dynamic_models.temporal_eval.data_loader import UnifiedData
from dynamic_models.temporal_eval.student_split import StudentSplit

logger = logging.getLogger(__name__)

HF_REPO_ID = "CodeInsightTeam/code_insights_csv"


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class EvalItem:
    """One evaluation item: a (student, question) pair ready for prompting."""
    question_id: str
    student_id: Optional[str]
    question_name: str
    question_text: str
    question_template: str
    question_unittests: str


# ── Test-case parsing (for iterative feedback) ──────────────────────────────


def parse_test_cases(unittests_str: str) -> List[dict]:
    """Parse the ``question_unittests`` field into a list of test case dicts.

    Each dict has keys: ``input``, ``std_in``, ``output``.
    Returns an empty list if the format is malformed.
    """
    test_cases = []
    for block in unittests_str.split("Unittest")[1:]:
        body = block[block.find(":") + 1:]
        i_input = body.find("Input:")
        i_stdin = body.find("STD input:")
        i_output = body.find("Output:")
        if -1 in (i_input, i_stdin, i_output):
            return []
        test_cases.append({
            "input": body[i_input + 6: i_stdin].strip(),
            "std_in": body[i_stdin + 10: i_output].strip(),
            "output": body[i_output + 7:].strip(),
        })
    return test_cases


# ── Item difficulty ──────────────────────────────────────────────────────────


@dataclass
class ItemDifficulty:
    question_id: int
    question_name: str
    topic: str
    train_pass_rate: float
    avg_attempts_to_pass: float
    fraction_eventually_pass: float
    n_train_students_attempted: int


def _load_hf_question_details() -> pd.DataFrame:
    """Load full question metadata (text, template, unittests) from HuggingFace."""
    hf_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset", local_files_only=True,
    )
    return pd.read_csv(f"{hf_dir}/question_infos.csv")


def _compute_item_difficulty(
    data: UnifiedData, split: StudentSplit,
) -> Dict[str, ItemDifficulty]:
    """Compute per-question difficulty stats from train students only."""
    corr = data.correctness_matrix
    qi = data.question_infos

    unique_qnames = qi.drop_duplicates(subset=["qname"])
    difficulties = {}

    for _, row in unique_qnames.iterrows():
        qname = row["qname"]
        item_mask = qi["qname"] == qname
        item_indices = np.where(item_mask)[0]

        # Only use test items (weeks 4+) — these are the prediction targets
        test_item_set = set(split.test_item_indices.tolist())
        target_indices = [i for i in item_indices if i in test_item_set]
        if not target_indices:
            continue

        # Train students' observations on these items
        train_corr = corr[split.train_student_indices][:, target_indices, :]
        n_train = len(split.train_student_indices)

        # Per train student: did they eventually pass all testcases?
        students_attempted = 0
        students_passed = 0
        total_attempts_to_pass = []

        for s in range(n_train):
            student_obs = train_corr[s]  # [n_target_items, n_attempts]
            has_obs = (student_obs != -1).any()
            if not has_obs:
                continue
            students_attempted += 1

            # Check if student eventually passed all testcases
            all_passed = True
            for tc in range(len(target_indices)):
                tc_obs = student_obs[tc]
                valid = tc_obs[tc_obs != -1]
                if len(valid) == 0 or valid[-1] != 1:
                    all_passed = False
                    break

            if all_passed:
                students_passed += 1
                # Count attempts to first all-pass
                for a in range(student_obs.shape[1]):
                    attempt_vals = student_obs[:, a]
                    if (attempt_vals == 1).all():
                        total_attempts_to_pass.append(a + 1)
                        break

        if students_attempted == 0:
            continue

        qid = int(row.get("question_unittest_id", 0)) if "question_unittest_id" in row.index else 0

        difficulties[qname] = ItemDifficulty(
            question_id=qid,
            question_name=qname,
            topic=str(row.get("topic", "")),
            train_pass_rate=students_passed / students_attempted,
            avg_attempts_to_pass=(
                np.mean(total_attempts_to_pass) if total_attempts_to_pass else float("nan")
            ),
            fraction_eventually_pass=students_passed / students_attempted,
            n_train_students_attempted=students_attempted,
        )

    return difficulties


# ── Student-split data loading ───────────────────────────────────────────────


def load_student_split_eval_items(
    data: UnifiedData,
    split: StudentSplit,
    seed: int = 42,
) -> Tuple[List[EvalItem], Dict[str, ItemDifficulty]]:
    """Convert StudentSplit data into EvalItems for the LLM predictor.

    Uses the same (data, split) that psychometric models use.

    Returns:
        Tuple of (eval_items, item_difficulties).
    """
    hf_qi = _load_hf_question_details()
    qi = data.question_infos

    # Join HF questions with course-id filtering as primary; question_name
    # matching is only a fallback for names absent from the course's own rows
    # (courses may share question banks across years, and a same-named question
    # from another course carries the wrong question_id).
    test_item_set_pre = set(split.test_item_indices.tolist())
    needed_qnames = set(qi[qi.index.isin(test_item_set_pre)]["qname"].unique())

    course_id = None
    if "course_id" in data.main_data.columns and len(data.main_data):
        course_id = int(data.main_data["course_id"].iloc[0])

    hf_qi = hf_qi[hf_qi["question_name"].isin(needed_qnames)]
    if course_id is not None and "course_id" in hf_qi.columns:
        in_course = hf_qi["course_id"].astype(int) == course_id
        course_qnames = set(hf_qi.loc[in_course, "question_name"])
        fallback = hf_qi[~in_course & ~hf_qi["question_name"].isin(course_qnames)]
        if len(fallback):
            logger.warning(
                "%d question names matched only by text from a different course; "
                "their question_ids will not match this course's submissions",
                fallback["question_name"].nunique(),
            )
        hf_qi = pd.concat([hf_qi[in_course], fallback])
    # The in-course and fallback blocks are disjoint by question_name; within
    # the fallback block keep="first" is arbitrary but deterministic.
    hf_qi = hf_qi.drop_duplicates(subset=["question_name"], keep="first")

    matched = needed_qnames & set(hf_qi["question_name"])
    if matched:
        logger.info("Matched %d/%d target questions in HF data", len(matched), len(needed_qnames))
    else:
        logger.warning("No question name overlap found in HF data for %s", data.course_name)

    # Map qname -> full question details from HuggingFace
    hf_lookup = {}
    for _, row in hf_qi.iterrows():
        hf_lookup[row["question_name"]] = row

    # Identify unique target questions (weeks 4+) from test items
    test_item_set = set(split.test_item_indices.tolist())
    target_qnames = qi[qi.index.isin(test_item_set)]["qname"].unique()

    logger.info("Target questions: %d", len(target_qnames))

    # Map test student indices to student_ids
    test_student_ids = [data.student_ids[i] for i in split.test_student_indices]

    main_df = data.main_data.copy()
    main_df["student_id"] = main_df["student_id"].astype(str)

    # Build EvalItems
    items = []
    for test_s_idx, sid in zip(split.test_student_indices, test_student_ids):
        sid_str = str(sid)

        for qname in target_qnames:
            # Check student has real observations on this question's items
            q_item_indices = qi[qi["qname"] == qname].index.tolist()
            q_test_items = [i for i in q_item_indices if i in test_item_set]
            if not q_test_items:
                continue

            has_obs = False
            for item_idx in q_test_items:
                obs = data.correctness_matrix[test_s_idx, item_idx, :]
                if (obs != -1).any():
                    has_obs = True
                    break
            if not has_obs:
                continue

            # Look up full question details
            hf_row = hf_lookup.get(qname)
            if hf_row is None:
                continue

            qid = str(int(hf_row["question_id"]))
            q_text = str(hf_row.get("question_text", ""))
            q_template = str(hf_row.get("question_template", ""))
            q_unittests = str(hf_row.get("question_unittests", ""))

            if not q_unittests or q_unittests == "nan":
                continue

            # Collect this student's real submissions on the target question
            target_rows = main_df[
                (main_df["student_id"] == sid_str)
                & (main_df["question_unittest_id"] == int(qid))
            ].sort_values("timestamp")

            real_attempts = []
            for _, r in target_rows.iterrows():
                if not r.get("response"):
                    continue
                real_attempts.append({
                    "response": str(r["response"]),
                    "pass": str(r.get("pass", "")),
                    "response_type": str(r.get("response_type", "Submit")),
                })

            # Cross-course fallback rows carry a foreign question_id that
            # matches no submissions here; such items can never emit output.
            if not real_attempts:
                continue

            item = EvalItem(
                question_id=qid,
                student_id=sid_str,
                question_name=qname,
                question_text=q_text,
                question_template=q_template,
                question_unittests=q_unittests,
            )
            item._real_attempts = real_attempts
            items.append(item)

    # Shuffle for representative coverage in partial runs
    rng = np.random.RandomState(seed)
    rng.shuffle(items)

    logger.info("Built %d eval items for %d test students", len(items), len(test_student_ids))

    # Compute item difficulty from train students
    difficulties = _compute_item_difficulty(data, split)

    return items, difficulties
