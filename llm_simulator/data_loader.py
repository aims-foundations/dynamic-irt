"""Unified data loading for LLM student simulation.

Loads scenario data from local CSVs (produced by data_preprocessing.py)
or directly from HuggingFace, and produces a list of EvalItem objects
ready for prompt building.

Key idea: one loader, parameterized by n_examples.
    - n_examples=0  → one item per question (zero-shot, old S1)
    - n_examples=N  → one item per (student, question), with N examples
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

HF_REPO_ID = "stair-lab/code_insights_csv"
DATA_URL = (
    "https://huggingface.co/datasets/stair-lab/code_insights_csv/"
    "resolve/main/"
)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class Example:
    """An in-context example of a student's code on a different question."""
    question_name: str
    question_text: str
    question_template: str
    response: str


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
    precheck["pass_len"] = precheck["pass"].astype(str).str.len()
    n_public = (
        precheck.groupby("question_unittest_id")["pass_len"]
        .agg(lambda x: int(x.mode().iloc[0]))
        .to_dict()
    )
    return {str(k): v for k, v in n_public.items()}


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
) -> List[EvalItem]:
    """Load evaluation items for the LLM simulator.

    Parameters
    ----------
    n_examples : int
        Number of in-context examples per item.  0 = zero-shot (one item per
        question).  N > 0 = few-shot (one item per (student, question), with
        N examples from the student's other solved questions).
    data_dir : str
        Local directory containing scenario CSVs from data_preprocessing.py.
    max_samples : int, optional
        Limit the number of items returned (for quick testing).

    Returns
    -------
    list of EvalItem
    """
    if n_examples == 0:
        items = _load_zero_shot(data_dir)
    else:
        items = _load_few_shot(n_examples, data_dir)

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


def _load_few_shot(n_examples: int, data_dir: str) -> List[EvalItem]:
    """Load one item per (student, question) with N examples.

    Examples are drawn from the student's other solved questions (sorted by
    timestamp), giving the LLM in-context demonstrations of the student's
    coding style.
    """
    df = _load_csv("Scenario2_full_data.csv", data_dir)
    items = []
    for student_id, student_df in df.groupby("student_id"):
        student_df = student_df.sort_values("timestamp")
        if len(student_df) < n_examples + 1:
            continue

        for idx in range(len(student_df)):
            target = student_df.iloc[idx]

            # Pick examples from other questions by this student
            other = student_df.drop(student_df.index[idx])
            examples = [
                Example(
                    question_name=ex["question_name"],
                    question_text=ex["question_text"],
                    question_template=ex["question_template"],
                    response=ex["response"],
                )
                for _, ex in other.head(n_examples).iterrows()
            ]

            items.append(EvalItem(
                question_id=str(target.get(
                    "question_unittest_id", target.name
                )),
                student_id=str(student_id),
                question_name=target["question_name"],
                question_text=target["question_text"],
                question_template=target["question_template"],
                question_unittests=target.get("question_unittests", ""),
                examples=examples,
            ))
    return items
