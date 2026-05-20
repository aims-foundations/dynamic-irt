"""Data loader bridging the student split framework to LLM eval items.

Uses the same (data, split) from load_student_split_data() that
psychometric models use, ensuring identical filtered students, items,
and split indices.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from dynamic_models.temporal_eval.data_loader import UnifiedData, load_student_split_data
from dynamic_models.temporal_eval.student_split import StudentSplit

from .data_loader import EvalItem, Example

logger = logging.getLogger(__name__)


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
        repo_id="CodeInsightTeam/code_insights_csv",
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


def load_student_split_eval_items(
    data: UnifiedData,
    split: StudentSplit,
    n_examples: int = 5,
    seed: int = 42,
) -> Tuple[List[EvalItem], Dict[str, ItemDifficulty]]:
    """Convert StudentSplit data into EvalItems for the LLM predictor.

    Uses the same (data, split) that psychometric models use.

    Returns:
        Tuple of (eval_items, item_difficulties).
    """
    hf_qi = _load_hf_question_details()
    qi = data.question_infos

    # Filter HF questions to this course
    course_infos = pd.read_csv(
        f"{snapshot_download(repo_id='CodeInsightTeam/code_insights_csv', repo_type='dataset', local_files_only=True)}/course_infos.csv"
    )
    course_row = course_infos[course_infos["course_name"] == data.course_name]
    if len(course_row) > 0:
        course_id = course_row["course_id"].values[0]
        hf_qi = hf_qi[hf_qi["course_id"] == course_id]

    # Map qname -> full question details from HuggingFace
    hf_lookup = {}
    qname_to_qid = {}
    for _, row in hf_qi.iterrows():
        hf_lookup[row["question_name"]] = row
        qname_to_qid[row["question_name"]] = int(row["question_id"])

    # Identify unique target questions (weeks 4+) from test items
    test_item_set = set(split.test_item_indices.tolist())
    target_qnames = qi[qi.index.isin(test_item_set)]["qname"].unique()

    logger.info("Target questions: %d", len(target_qnames))

    # Map test student indices to student_ids
    test_student_ids = [data.student_ids[i] for i in split.test_student_indices]

    # Build train question IDs (weeks 1-3) for context filtering
    train_qnames = qi.iloc[split.train_item_indices]["qname"].unique()
    train_qids = set()
    for qn in train_qnames:
        qid = qname_to_qid.get(qn)
        if qid is not None:
            train_qids.add(qid)

    # Build qid -> qname lookup from HF data
    qid_to_qname = {int(row["question_id"]): row["question_name"] for _, row in hf_qi.iterrows()}

    test_sid_set = set(str(sid) for sid in test_student_ids)
    main_df = data.main_data.copy()
    main_df["student_id"] = main_df["student_id"].astype(str)

    # Context submissions: test students' weeks 1-3 data
    context_df = main_df[
        main_df["student_id"].isin(test_sid_set)
        & main_df["question_unittest_id"].isin(train_qids)
    ].sort_values(["student_id", "timestamp"])

    # Pre-group context by student
    student_context = {}
    for sid, group in context_df.groupby("student_id"):
        student_context[str(sid)] = group.to_dict("records")

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

            # Build examples from this student's weeks 1-3 history
            examples = []
            rows = student_context.get(sid_str, [])

            # Get unique prior questions, sorted by timestamp (most recent last)
            prior_qs = {}
            for r in rows:
                pqid = r["question_unittest_id"]
                if str(pqid) != qid:
                    prior_qs.setdefault(pqid, []).append(r)

            # Take n_examples most recent prior questions
            sorted_pqs = sorted(
                prior_qs.items(),
                key=lambda x: x[1][-1]["timestamp"],
            )[-n_examples:]

            for pqid, pq_rows in sorted_pqs:
                pq_name = qid_to_qname.get(int(pqid), "")
                pq_hf = hf_lookup.get(pq_name)
                pq_text = str(pq_hf["question_text"]) if pq_hf is not None else ""
                pq_template = str(pq_hf["question_template"]) if pq_hf is not None else ""

                for r in pq_rows:
                    if not r.get("response"):
                        continue
                    examples.append(Example(
                        question_name=pq_name,
                        question_text=pq_text,
                        question_template=pq_template,
                        response=str(r["response"]),
                        response_type=str(r.get("response_type", "Submit")),
                        pass_pattern=str(r.get("pass", "")),
                    ))

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

            item = EvalItem(
                question_id=qid,
                student_id=sid_str,
                question_name=qname,
                question_text=q_text,
                question_template=q_template,
                question_unittests=q_unittests,
                examples=examples,
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
