"""Unified prompt builder for LLM student simulation.

Replaces the old 4-scenario PromptGenerator + iterative prompt functions.
One scenario ("imitate this student"), parameterized by:
    - examples: None → zero-shot, List[dict] → few-shot
    - feedback: None → first attempt, dict → retry with test results
"""

import re
from typing import Dict, List, Optional, Tuple, Union

# ── Code extraction ──────────────────────────────────────────────────────────

# Allow optional whitespace before closing ``` (GLM outputs indented closings).
# No end-of-string fallback — unclosed fences should not match (avoids capturing
# all reasoning text to end-of-string).
_CPP_FENCE_RE = re.compile(r"```(?:cpp|c\+\+)\n(.*?)\n\s*```", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```[\w+]*\n(.*?)\n\s*```", re.DOTALL)


def _find_code_in(text: str) -> Optional[str]:
    """Extract the first C++ code block from a text fragment."""
    m = _CPP_FENCE_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m = _ANY_FENCE_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


def extract_code(llm_output: str) -> Optional[str]:
    """Extract the model's actual C++ solution from its response.

    Many models (e.g. GLM-4) emit chain-of-thought reasoning that quotes
    student code in ```cpp fences before outputting the real answer.  To
    avoid grabbing reasoning-quoted code, we look for the *last*
    [Submit]/[Precheck] action tag and extract the first fence after it.
    Falls back to the last fence in the full response.
    """
    # Strategy 1: find last [Submit]/[Precheck], take first fence after it
    action_matches = list(_ACTION_RE.finditer(llm_output))
    if action_matches:
        last_action = action_matches[-1]
        code = _find_code_in(llm_output[last_action.end():])
        if code:
            return code

    # Strategy 2: take the last cpp/code fence in the entire response
    matches = _CPP_FENCE_RE.findall(llm_output)
    if matches:
        code = matches[-1].strip()
        if code:
            return code
    matches = _ANY_FENCE_RE.findall(llm_output)
    if matches:
        code = matches[-1].strip()
        if code:
            return code

    return None


_ACTION_RE = re.compile(r"\[(?:Precheck|Submit)\]", re.IGNORECASE)


def extract_action(llm_output: str) -> str:
    """Extract the student's chosen action from the LLM response.

    Uses the *last* action tag to skip any tags quoted in reasoning.
    Returns "Precheck" or "Submit".  Defaults to "Submit" if no action tag
    is found (conservative — treats ambiguous output as a final submission).
    """
    matches = list(_ACTION_RE.finditer(llm_output))
    if matches:
        tag = matches[-1].group(0).lower()
        if "precheck" in tag:
            return "Precheck"
        return "Submit"
    return "Submit"


# ── Shared instruction block ────────────────────────────────────────────────

_CODE_INSTRUCTIONS = (
    "You must first choose an action — either [Precheck] (run public tests only, "
    "no grade penalty) or [Submit] (final submission, graded against all tests).\n\n"
    "Then provide your C++ implementation that will replace the "
    "{{ STUDENT_ANSWER }} block in the template.\n"
    "- Do NOT reproduce any part of the template.\n"
    "- Do NOT emit `int main()` (it's already declared).\n"
    "- Include any needed class definitions.\n\n"
    "IMPORTANT: Your response format must be exactly:\n"
    "1. First line: either [Precheck] or [Submit]\n"
    "2. Second line: ```cpp\n"
    "3. Your code\n"
    "4. Last line: ```\n"
    "No extra characters, whitespace, or text may appear before or after."
)


# ── Prompt builder ───────────────────────────────────────────────────────────


def build_prompt(
    question_name: str,
    question_text: str,
    question_template: str,
    examples: Optional[List[Dict[str, str]]] = None,
    feedback: Optional[Dict] = None,
    persona_text: Optional[str] = None,
    rag_context: Optional[str] = None,
    summarized_history: Optional[str] = None,
) -> Union[str, Tuple[str, str]]:
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
    persona_text : str, optional
        Character Card-style persona for use as system message.
    rag_context : str, optional
        Retrieved reference submissions for additional context.

    Returns
    -------
    str or (str, str)
        If persona_text is provided, returns (system_message, user_message).
        Otherwise returns a single prompt string.
    """
    parts: List[str] = []

    # ── Summarized history (replaces raw examples when --summarize is used) ──
    if summarized_history:
        parts.append(summarized_history)

    # ── In-context examples (few-shot / trajectory) ──
    elif examples:
        parts.append("=== Student Submission History ===\n")
        parts.append(
            "Below are this student's previous submissions on other problems, "
            "showing how they code, iterate, and choose between Precheck and Submit.\n"
            "- [Precheck] runs public tests only (no grade penalty).\n"
            "- [Submit] is a final submission graded against all tests.\n"
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
            for a, ex in enumerate(group, start=1):
                rtype = ex.get("response_type", "Submit")
                action = "Precheck" if rtype == "Prechecked" else "Submit"
                result = ex.get("pass_pattern", "")
                result_str = f" → Result: {result}" if result else ""
                parts.append(
                    f"Attempt {a} [{action}]{result_str}:\n{ex['response']}\n"
                )

        parts.append(
            "Now, using the same student's coding style, approach, and "
            "Precheck/Submit strategy, attempt this new problem:\n"
        )

    # ── RAG context ──
    if rag_context:
        parts.append(f"{rag_context}\n")

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
            "\nEdit your previous code to fix the failing tests. "
            "Make the smallest change necessary — fix only the buggy "
            "lines and keep everything else identical. "
            "Do NOT rewrite the solution from scratch.\n"
        )

    # ── Instructions ──
    parts.append(f"\n{_CODE_INSTRUCTIONS}")

    user_message = "\n".join(parts)

    if persona_text:
        return (persona_text, user_message)
    return user_message
