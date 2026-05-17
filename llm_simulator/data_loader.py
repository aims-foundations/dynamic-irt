"""Unified data loading for LLM student simulation.

Loads scenario data from local CSVs (produced by data_preprocessing.py)
or directly from HuggingFace, and produces a list of EvalItem objects
ready for prompt building.

Key idea: one loader, parameterized by n_examples.
    - n_examples=0  → one item per question (zero-shot)
    - n_examples=N  → trajectory-based: for each (student, question),
      provide the student's N most recent prior questions (with all
      attempts on each) as context
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

HF_REPO_ID = "CodeInsightTeam/code_insights_csv"
DATA_URL = (
    "https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv/"
    "resolve/main/"
)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class Example:
    """An in-context example from a student's submission history."""
    question_name: str
    question_text: str
    question_template: str
    response: str
    response_type: str = "Submit"      # "Prechecked" or "Submit"
    pass_pattern: str = ""             # e.g. "1101" — test results after this attempt


@dataclass
class EvalItem:
    """One evaluation item: a (student, question) pair ready for prompting."""
    question_id: str
    student_id: Optional[str]
    question_name: str
    question_text: str
    question_template: str
    question_unittests: str
    examples: List[Example] = field(default_factory=list)


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


def infer_public_test_counts(main_df: pd.DataFrame) -> Dict[str, int]:
    """Infer N_public per question from "Prechecked" submission pass-string lengths.

    Students on the online judge run public tests via "Precheck", so
    len(pass) for those rows = N_public for that question.  We take the mode
    per question as the canonical N_public.

    Returns {question_unittest_id -> n_public}.  Questions with no
    Prechecked submissions are absent from the dict.
    """
    precheck = main_df[main_df["response_type"] == "Prechecked"].copy()
    precheck = precheck[precheck["pass"].notna()]
    precheck["pass_len"] = precheck["pass"].astype(str).str.len()
    n_public = {}
    for qid, group in precheck.groupby("question_unittest_id")["pass_len"]:
        modes = group.mode()
        if len(modes) > 0:
            n_public[str(qid)] = int(modes.iloc[0])
        elif len(group) > 0:
            n_public[str(qid)] = int(group.median())
    return n_public


# ── Internal helpers ─────────────────────────────────────────────────────────


def _load_csv(filename: str, data_dir: str = "data") -> pd.DataFrame:
    """Load a CSV from local directory or HuggingFace."""
    local = Path(data_dir) / filename
    if local.exists():
        logger.info("Loading local: %s", local)
        return pd.read_csv(local)
    url = f"{DATA_URL}data/{filename}"
    logger.info("Loading from HuggingFace: %s", url)
    return pd.read_csv(url)


def _load_hf_main_data() -> pd.DataFrame:
    """Load main_data.csv from HuggingFace (for public test counts + full data)."""
    hf_dir = snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset")
    return pd.read_csv(
        os.path.join(hf_dir, "main_data.csv"), dtype={"pass": str}
    )


def _load_hf_questions() -> pd.DataFrame:
    """Load question_infos.csv from HuggingFace."""
    hf_dir = snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset")
    return pd.read_csv(os.path.join(hf_dir, "question_infos.csv"))


# ── Main loader ──────────────────────────────────────────────────────────────


def load_eval_items(
    n_examples: int = 0,
    data_dir: str = "data",
    max_samples: Optional[int] = None,
    max_students: Optional[int] = None,
    max_questions: Optional[int] = None,
    seed: int = 42,
) -> List[EvalItem]:
    """Load evaluation items for the LLM simulator.

    Parameters
    ----------
    n_examples : int
        Number of previous questions to use as context.  0 = zero-shot (one
        item per question).  N > 0 = trajectory-based: for each (student,
        question), provide the student's N most recent prior questions (with
        ALL attempts on each) as in-context examples.
    data_dir : str
        Local directory containing scenario CSVs from data_preprocessing.py.
    max_samples : int, optional
        Limit the total number of items returned (for quick testing).
    max_students : int, optional
        Randomly sample this many students (for few-shot/trajectory mode).
    max_questions : int, optional
        Randomly sample this many questions (for few-shot/trajectory mode).
    seed : int
        Random seed for reproducible sampling (default: 42).

    Returns
    -------
    list of EvalItem
    """
    if n_examples == 0:
        items = _load_zero_shot(data_dir)
    else:
        items = _load_trajectory(
            n_examples, max_students=max_students,
            max_questions=max_questions, seed=seed,
        )

    if max_samples and len(items) > max_samples:
        items = items[:max_samples]
    return items


def _load_zero_shot(data_dir: str) -> List[EvalItem]:
    """Load one item per unique question (zero-shot, old S1)."""
    df = _load_csv("Scenario1_full_data.csv", data_dir)
    items = []
    for _, row in df.groupby("question_unittest_id").first().reset_index().iterrows():
        items.append(EvalItem(
            question_id=str(row["question_unittest_id"]),
            student_id=None,
            question_name=row["question_name"],
            question_text=row["question_text"],
            question_template=row["question_template"],
            question_unittests=row.get("question_unittests", ""),
        ))
    return items


def _load_trajectory(
    n_examples: int,
    max_students: Optional[int] = None,
    max_questions: Optional[int] = None,
    seed: int = 42,
) -> List[EvalItem]:
    """Load trajectory-based items from full submission history.

    For each student's last submission on each sampled question, provide the
    student's N most recent prior *questions* (with ALL attempts on each) as
    in-context examples.  This shows the LLM both the student's coding style
    and their debugging/iteration process.

    Parameters
    ----------
    n_examples : int
        Number of previous questions to include (each with all attempts).
    max_students, max_questions : int, optional
        Randomly sample this many students/questions for prediction targets.
    seed : int
        Random seed for reproducible sampling.
    """
    logger.info("Loading full submission data from HuggingFace…")
    main_df = _load_hf_main_data()
    q_df = _load_hf_questions()

    # Keep Prechecked + Submit responses, merge question info
    subs = main_df.dropna(subset=["response"]).copy()
    subs = subs[subs["response_type"].isin(["Prechecked", "Submit"])].sort_values("timestamp")

    q_cols = [
        "question_id", "question_name", "question_text",
        "question_template", "question_unittests",
    ]
    merged = subs.merge(
        q_df[q_cols],
        left_on="question_unittest_id",
        right_on="question_id",
        how="inner",
    )

    # Sample students and questions independently (unbiased)
    all_students = merged["student_id"].unique()
    all_questions = merged["question_unittest_id"].unique()

    if max_students and max_students < len(all_students):
        sampled_students = (
            pd.Series(all_students)
            .sample(n=max_students, random_state=seed)
            .values
        )
    else:
        sampled_students = all_students

    if max_questions and max_questions < len(all_questions):
        sampled_questions = (
            pd.Series(all_questions)
            .sample(n=max_questions, random_state=seed + 1)
            .values
        )
    else:
        sampled_questions = all_questions

    logger.info(
        "Sampled %d students, %d questions",
        len(sampled_students), len(sampled_questions),
    )

    # Keep full history for sampled students (needed for trajectory context)
    student_data = merged[merged["student_id"].isin(sampled_students)].copy()
    student_data = student_data.sort_values(["student_id", "timestamp"])

    sampled_q_set = set(sampled_questions)

    # Pre-group by student for fast lookup
    logger.info("Building trajectory items for %d students…", len(sampled_students))
    items = []
    n_processed = 0

    for student_id, student_df in student_data.groupby("student_id"):
        # Convert to list of dicts once (much faster than iterrows)
        rows = student_df.to_dict("records")
        if not rows:
            continue

        # Group rows by question_id for fast lookup
        q_rows: Dict[str, list] = {}
        for r in rows:
            qid = r["question_unittest_id"]
            q_rows.setdefault(qid, []).append(r)

        # Find target questions (last submission per sampled question)
        targets: Dict[str, dict] = {}
        for r in rows:
            qid = r["question_unittest_id"]
            if qid in sampled_q_set:
                targets[qid] = r  # last one wins (rows sorted by timestamp)

        for target_qid, target in targets.items():
            target_time = target["timestamp"]

            # Find unique OTHER questions attempted before target_time
            # Track the latest timestamp per prior question
            prior_latest: Dict[str, str] = {}
            for r in rows:
                if r["timestamp"] >= target_time:
                    break
                qid = r["question_unittest_id"]
                if qid != target_qid:
                    prior_latest[qid] = r["timestamp"]

            if not prior_latest:
                continue

            # Get N most recent prior questions by their latest timestamp
            sorted_priors = sorted(
                prior_latest.items(), key=lambda x: x[1]
            )[-n_examples:]
            prior_qid_set = {qid for qid, _ in sorted_priors}

            # Build examples: all attempts on each prior question before target
            examples = []
            for pq, _ in sorted_priors:
                for r in q_rows.get(pq, []):
                    if r["timestamp"] < target_time:
                        examples.append(Example(
                            question_name=r["question_name"],
                            question_text=r["question_text"],
                            question_template=r["question_template"],
                            response=r["response"],
                            response_type=r.get("response_type", "Submit"),
                            pass_pattern=str(r.get("pass", "")),
                        ))

            if not examples:
                continue

            items.append(EvalItem(
                question_id=str(target_qid),
                student_id=str(student_id),
                question_name=target["question_name"],
                question_text=target["question_text"],
                question_template=target["question_template"],
                question_unittests=str(
                    target.get("question_unittests", "")
                ),
                examples=examples,
            ))

        n_processed += 1
        if n_processed % 500 == 0:
            logger.info("  Processed %d/%d students (%d items so far)…",
                        n_processed, len(sampled_students), len(items))

    # Shuffle so partial runs give representative coverage across students
    import random
    rng = random.Random(seed)
    rng.shuffle(items)

    logger.info("Built %d trajectory items (shuffled)", len(items))
    return items
