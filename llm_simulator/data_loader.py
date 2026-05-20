"""Data structures and utilities for LLM student simulation.

Defines EvalItem, Example dataclasses and test-case parsing.
Data loading is handled by student_split_loader.py.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

HF_REPO_ID = "CodeInsightTeam/code_insights_csv"


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
    target_timestamp: Optional[str] = None  # timestamp of the target submission (for temporal filtering)


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


