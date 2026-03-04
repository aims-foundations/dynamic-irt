"""Unified prompt builder for LLM student simulation.

Replaces the old 4-scenario PromptGenerator + iterative prompt functions.
One scenario ("imitate this student"), parameterized by:
    - examples: None → zero-shot, List[dict] → few-shot
    - feedback: None → first attempt, dict → retry with test results
"""

import re
from typing import Dict, List, Optional

# ── Code extraction ──────────────────────────────────────────────────────────

_CPP_FENCE_RE = re.compile(r"```(?:cpp|c\+\+)\n(.*?)(?:\n```|$)", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```[\w+]*\n(.*?)(?:\n```|$)", re.DOTALL)


def extract_code(llm_output: str) -> Optional[str]:
    """Return the first C++ code block from markdown fences, or None."""
    # Prefer ```cpp or ```c++ fences
    m = _CPP_FENCE_RE.search(llm_output)
    if m:
        return m.group(1).strip()
    # Fall back to any fenced code block (```, ```c, etc.)
    m = _ANY_FENCE_RE.search(llm_output)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


# ── Shared instruction block ────────────────────────────────────────────────

_CODE_INSTRUCTIONS = (
    "Provide ONLY your C++ implementation that will replace the "
    "{{ STUDENT_ANSWER }} block in the template.\n"
    "- Do NOT reproduce any part of the template.\n"
    "- Do NOT emit `int main()` (it's already declared).\n"
    "- Include any needed class definitions.\n\n"
    "IMPORTANT: Your entire response must be exactly one Markdown C++ code-block.\n"
    "1. The first line of your output must be: ```cpp\n"
    "2. The last line of your output must be: ```\n"
    "3. No extra characters, whitespace, or text may appear before or after the block."
)


# ── Prompt builder ───────────────────────────────────────────────────────────


def build_prompt(
    question_name: str,
    question_text: str,
    question_template: str,
    examples: Optional[List[Dict[str, str]]] = None,
    feedback: Optional[Dict] = None,
) -> str:
    """Build a unified prompt for LLM student simulation.

    Parameters
    ----------
    question_name : str
        Name/title of the target question.
    question_text : str
        Full problem description.
    question_template : str
        C++ template with {{ STUDENT_ANSWER }} placeholder.
    examples : list of dict, optional
        In-context examples of the student's code. Each dict has keys:
        ``question_name``, ``question_text``, ``question_template``, ``response``.
        If None or empty, produces a zero-shot prompt.
    feedback : dict, optional
        Retry feedback from a previous attempt. Keys:
        ``previous_code`` (str), ``failed_tests`` (list of dict with keys
        ``input``, ``std_in``, ``expected``, ``actual``).

    Returns
    -------
    str
        The assembled prompt string.
    """
    parts: List[str] = []

    # ── In-context examples (few-shot / trajectory) ──
    if examples:
        parts.append("=== Student Submission History ===\n")
        parts.append(
            "Below are this student's previous submissions on other problems, "
            "showing how they code and iterate.\n"
        )

        # Group consecutive examples by question_name
        from itertools import groupby
        q_num = 0
        for qname, group in groupby(examples, key=lambda x: x["question_name"]):
            group = list(group)
            q_num += 1
            parts.append(
                f"--- Problem {q_num}: {qname} ---\n"
                f"{group[0]['question_text']}\n\n"
                f"Template:\n{group[0]['question_template']}\n"
            )
            if len(group) == 1:
                parts.append(f"Student's Code:\n{group[0]['response']}\n")
            else:
                for a, ex in enumerate(group, start=1):
                    parts.append(
                        f"Attempt {a}:\n{ex['response']}\n"
                    )

        parts.append(
            "Now, using the same student's coding style and approach, "
            "attempt this new problem:\n"
        )

    # ── Target question ──
    parts.append(
        f"Question: {question_name} — {question_text}\n\n"
        f"Template:\n{question_template}\n"
    )

    # ── Previous attempt feedback (iterative retry) ──
    if feedback:
        parts.append(
            f"\n=== Previous Attempt ===\n"
            f"```cpp\n{feedback['previous_code']}\n```\n\n"
            f"=== Feedback from Visible Tests ===\n"
            "Your previous submission failed the following visible test cases:\n"
        )
        for i, ft in enumerate(feedback["failed_tests"], start=1):
            actual = ft["actual"] or "(no output — likely a compile or runtime error)"
            parts.append(
                f"Test {i}:\n"
                f"  Test input:  {ft['input']}\n"
                f"  STD input:   {ft['std_in']}\n"
                f"  Expected:    {ft['expected']}\n"
                f"  Got:         {actual}\n"
            )
        parts.append(
            "\nPlease fix the issues and provide a corrected solution "
            "in the same format.\n"
        )

    # ── Instructions ──
    parts.append(f"\n{_CODE_INSTRUCTIONS}")

    return "\n".join(parts)
