"""Unified prompt builder for LLM student simulation.

One scenario ("imitate this student"): build_prompt assembles a structured
system + user message pair from the persona, self-trajectory summaries, and
question metadata.
Also provides extract_code / extract_action for parsing LLM responses.
"""

import re
from typing import List, Optional, Tuple

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
    "Choose an action based on how this student would behave:\n"
    "- [Submit] if the student would confidently submit directly.\n"
    "- [Precheck] if the student would test against public tests first.\n\n"
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
    persona_text: Optional[str] = None,
    self_summaries: Optional[List[str]] = None,
    question_metadata: Optional[str] = None,
    question_summary: Optional[str] = None,
) -> Tuple[str, str]:
    """Build a prompt for LLM student simulation.

    Returns a (system_message, user_message) pair assembled from the
    persona, self-trajectory summaries, and question metadata (each
    section is included only when provided).
    """
    # == SYSTEM MESSAGE ==
    sys_parts = [
        # Task framing
        "You are predicting what a specific university student would submit "
        "for a C++ programming assignment. Your goal is to mimic the full "
        "submission history of this student. If you believe the student "
        "would not pass on the first attempt, produce code with errors "
        "consistent with their coding abilities and profile. Continue "
        "producing realistic attempts until you believe the student would "
        "solve the question. Study the student profile and prior work "
        "to calibrate your response.",

        # Question description (short — full text + template are in user message)
        f"\n=== QUESTION: {question_name} ===\n",
    ]
    if question_summary:
        sys_parts.append(question_summary)
    if question_metadata:
        sys_parts.append(question_metadata)

    # Student identity (strip the old intro line from persona)
    if persona_text:
        persona_lines = persona_text.split("\n")
        identity_start = next(
            (i for i, l in enumerate(persona_lines) if "STUDENT IDENTITY" in l), None
        )
        if identity_start is not None:
            sys_parts.append("\n" + "\n".join(persona_lines[identity_start:]))
        else:
            sys_parts.append("\n" + persona_text)

    system_message = "\n".join(sys_parts)

    # == USER MESSAGE ==
    parts: List[str] = []

    # Self approaches (with question name + summary)
    if self_summaries:
        parts.append(
            "=== How This Student Approached Similar Problems ===\n"
        )
        for i, summary in enumerate(self_summaries, 1):
            parts.append(f"{i}. {summary}\n")

    # Format demonstration
    parts.append(
        "=== Submission Format ===\n"
        "You can either [Precheck] (run public tests only, no grade penalty) "
        "or [Submit] (final submission, graded against all tests).\n"
        "Example:\n"
        "[Precheck]\n"
        "```cpp\n"
        "// your implementation here\n"
        "```\n"
    )

    # Question text + template
    parts.append(f"=== Question ===\n{question_text}\n")
    parts.append(f"=== Code Template ===\n{question_template}\n")

    # Instructions at the very end
    parts.append(f"\n{_CODE_INSTRUCTIONS}")
    return (system_message, "\n".join(parts))
