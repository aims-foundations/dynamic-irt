"""
Iterative Code Runner for CodeInsights.

Implements the iteration loop: generate -> test -> feedback -> retry
Uses Claude API for code generation and existing CPPEvaluator for test execution.
"""

import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import anthropic
from huggingface_hub import hf_hub_download
import pandas as pd

from code_metrics import CPPEvaluator


# --- Data Structures ---

@dataclass
class TestCase:
    """Single test case with input, stdin, and expected output."""
    input: str      # Code to insert in template (test assertions)
    std_in: str     # Standard input for the program
    output: str     # Expected output


@dataclass
class Question:
    """A single question with its metadata and test cases."""
    question_id: str
    question_name: str
    question_text: str
    question_template: str
    test_cases: List[TestCase]


@dataclass
class TestResult:
    """Result of running test cases."""
    score: float                    # 0.0 to 1.0
    testcase_results: List[int]     # 0 or 1 per test case


@dataclass
class AttemptLog:
    """Log of a single attempt within an iteration session."""
    attempt_id: int
    timestamp: str
    response_type: str  # "precheck" (public tests) or "check" (all tests)
    code: str
    pass_pattern: str   # e.g., "11101" - matches main_data.csv format


@dataclass
class SessionLog:
    """Complete log of all attempts for one question."""
    question_id: str
    model_id: str
    attempts: List[AttemptLog] = field(default_factory=list)
    final_score: float = 0.0
    total_iterations: int = 0


# --- Main Runner Class ---

class IterativeCodeRunner:
    """
    Iterative code runner that uses Claude API for generation
    and CPPEvaluator for test execution.
    """

    # System prompt for Claude
    SYSTEM_PROMPT = """You are a skilled C++ programmer working on programming assignments.
Your task is to write correct, efficient C++ code that solves the given problem.
Write clean, well-structured code following good programming practices.

IMPORTANT RULES:
1. Provide ONLY your C++ implementation that replaces the {{ STUDENT_ANSWER }} block
2. Do NOT reproduce or include any template code
3. Do NOT write int main() - it's already in the template
4. Ensure your code handles all edge cases
5. Wrap your code in ```cpp and ``` markers"""

    FEEDBACK_TEMPLATE = """Your previous code attempt failed some test cases.

**Previous Code:**
```cpp
{code}
```

**Test Results:**
{test_results}

Please analyze the failures and provide a corrected implementation.
Remember: Only provide the code that replaces {{ STUDENT_ANSWER }} in the template.
Wrap your code in ```cpp and ``` markers."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 3,
        num_public_tests: int = 3,
        timeout_seconds: int = 10,
    ):
        """
        Initialize the iterative runner.

        Args:
            api_key: Anthropic API key
            model: Claude model ID
            max_iterations: Maximum iterations per question
            num_public_tests: Number of public tests (first N) shown during iteration
            timeout_seconds: Timeout for code execution
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.num_public_tests = num_public_tests
        self.timeout_seconds = timeout_seconds

    def run(self, questions: List[Question]) -> List[SessionLog]:
        """
        Run iterative code generation for all questions.

        Args:
            questions: List of questions to process

        Returns:
            List of session logs with all attempts
        """
        session_logs = []
        for i, question in enumerate(questions):
            print(f"\n{'='*60}")
            print(f"Processing question {i+1}/{len(questions)}: {question.question_id}")
            print(f"Question name: {question.question_name}")
            print(f"Test cases: {len(question.test_cases)} total, {min(self.num_public_tests, len(question.test_cases))} public")
            print('='*60)

            session_log = self._iterate(question)
            session_logs.append(session_log)

        return session_logs

    def _iterate(self, question: Question) -> SessionLog:
        """
        Main iteration loop for one question.

        Flow:
        1. Generate initial code
        2. Run public tests (first N)
        3. If score < 1.0 and iterations remain:
           - Build feedback prompt
           - Generate new code
           - Repeat from step 2
        4. Run ALL tests for final evaluation
        5. Return session log
        """
        session = SessionLog(
            question_id=question.question_id,
            model_id=self.model
        )

        # Split tests into public and private
        num_public = min(self.num_public_tests, len(question.test_cases))
        public_tests = question.test_cases[:num_public]
        all_tests = question.test_cases

        # Initial prompt
        prompt = self._build_initial_prompt(question)
        code = ""

        for attempt_num in range(self.max_iterations):
            print(f"\n--- Attempt {attempt_num + 1}/{self.max_iterations} ---")

            # Generate code
            response = self._make_request(prompt)
            code = self._extract_code(response)

            if not code:
                print("  WARNING: Failed to extract code from response")
                code = response  # Use raw response as fallback

            print(f"  Generated code length: {len(code)} chars")

            # Run public tests (precheck)
            public_result = self._run_tests(code, public_tests, question.question_template)

            # Log precheck attempt
            session.attempts.append(AttemptLog(
                attempt_id=attempt_num + 1,
                timestamp=datetime.now().isoformat(),
                response_type="precheck",
                code=code,
                pass_pattern="".join(map(str, public_result.testcase_results))
            ))

            print(f"  Public tests: {public_result.score:.0%} ({sum(public_result.testcase_results)}/{len(public_result.testcase_results)} passed)")

            # Check if all public tests pass
            if public_result.score == 1.0:
                print("  All public tests passed!")
                break

            # Build feedback prompt for next iteration
            if attempt_num < self.max_iterations - 1:
                prompt = self._build_feedback_prompt(question, code, public_result, public_tests)

        # Final evaluation on ALL tests
        print(f"\n--- Final Evaluation (all {len(all_tests)} tests) ---")
        final_result = self._run_tests(code, all_tests, question.question_template)

        # Log final check
        session.attempts.append(AttemptLog(
            attempt_id=len(session.attempts) + 1,
            timestamp=datetime.now().isoformat(),
            response_type="check",
            code=code,
            pass_pattern="".join(map(str, final_result.testcase_results))
        ))

        session.final_score = final_result.score
        session.total_iterations = len([a for a in session.attempts if a.response_type == "precheck"])

        print(f"  Final score: {session.final_score:.0%} ({sum(final_result.testcase_results)}/{len(final_result.testcase_results)} passed)")
        print(f"  Total iterations: {session.total_iterations}")

        return session

    def _make_request(self, prompt: str) -> str:
        """Make a request to Claude API."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            print(f"  API Error: {e}")
            return ""

    def _run_tests(
        self,
        code: str,
        test_cases: List[TestCase],
        template: str
    ) -> TestResult:
        """Run code against test cases using CPPEvaluator."""
        # Convert TestCase objects to dicts for CPPEvaluator
        tc_dicts = [
            {"input": tc.input, "std_in": tc.std_in, "output": tc.output}
            for tc in test_cases
        ]

        evaluator = CPPEvaluator(
            template=template,
            testcases=tc_dicts,
            timeout=self.timeout_seconds
        )

        result = evaluator.evaluate(code)

        return TestResult(
            score=result["score"],
            testcase_results=result["testcases"],
        )

    def _build_initial_prompt(self, question: Question) -> str:
        """Build the initial prompt for code generation."""
        return f"""**Question:** {question.question_name}

{question.question_text}

**Template:**
```cpp
{question.question_template}
```

Write the C++ code that replaces the `{{{{ STUDENT_ANSWER }}}}` placeholder.
Include any necessary class definitions but do NOT include main()."""

    def _build_feedback_prompt(
        self,
        question: Question,
        code: str,
        result: TestResult,
        test_cases: List[TestCase]
    ) -> str:
        """Build feedback prompt with test failure information."""
        # Format test results
        results_text = []
        for i, (tc, passed) in enumerate(zip(test_cases, result.testcase_results)):
            status = "PASSED" if passed else "FAILED"
            results_text.append(f"Test {i+1}: {status}")
            if not passed:
                # Show truncated test info for failed tests
                test_input_preview = tc.input[:200] + "..." if len(tc.input) > 200 else tc.input
                expected_preview = tc.output[:100] + "..." if len(tc.output) > 100 else tc.output
                results_text.append(f"  Test code: {test_input_preview}")
                results_text.append(f"  Expected output: {expected_preview}")

        test_results_str = "\n".join(results_text)

        # Combine with original question context
        return f"""**Question:** {question.question_name}

{question.question_text}

**Template:**
```cpp
{question.question_template}
```

{self.FEEDBACK_TEMPLATE.format(code=code, test_results=test_results_str)}"""

    def _extract_code(self, response: str) -> str:
        """Extract C++ code from model response."""
        # Look for code blocks with cpp language specifier
        code_blocks = re.findall(r"```cpp\n(.*?)\n```", response, flags=re.DOTALL)
        if code_blocks:
            code = code_blocks[0].strip()
        else:
            # Try without language specifier
            code_blocks = re.findall(r"```\n(.*?)\n```", response, flags=re.DOTALL)
            if code_blocks:
                code = code_blocks[0].strip()
            else:
                code = response.strip()

        # Remove main() if present (shouldn't be there but safety check)
        if "int main" in code:
            code = code.split("int main")[0].strip()

        return code


# --- Data Loading ---

def load_questions(num_questions: int = 5) -> List[Question]:
    """
    Load questions from HuggingFace dataset.
    Uses same data source as existing CodeInsightsCorrectCodeScenario.
    """
    print(f"Loading {num_questions} questions from HuggingFace...")

    # Download data file from HuggingFace
    data_file = hf_hub_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        repo_type="dataset",
        filename="codeinsights_llm_simulation/data/Scenario1_full_data.csv",
    )

    df = pd.read_csv(data_file, dtype={"pass": "str"})

    questions = []
    for question_id, question_df in df.groupby("question_unittest_id"):
        if len(questions) >= num_questions:
            break

        target = question_df.iloc[0]
        test_cases = []

        # Parse test cases (same logic as existing scenario)
        for testcase_str in target["question_unittests"].split("Unittest")[1:]:
            testcase_str = testcase_str[testcase_str.find(":") + 1:]
            input_idx = testcase_str.find("Input:")
            std_in_idx = testcase_str.find("STD input:")
            output_idx = testcase_str.find("Output:")

            if input_idx == -1 or std_in_idx == -1 or output_idx == -1:
                continue

            test_cases.append(TestCase(
                input=testcase_str[input_idx + 6: std_in_idx].strip(),
                std_in=testcase_str[std_in_idx + 10: output_idx].strip(),
                output=testcase_str[output_idx + 7:].strip(),
            ))

        # Need at least 3 tests for public/private split
        if len(test_cases) >= 3:
            questions.append(Question(
                question_id=str(question_id),
                question_name=target.get("question_name", ""),
                question_text=target.get("question_text", ""),
                question_template=target["question_template"],
                test_cases=test_cases
            ))

    print(f"Loaded {len(questions)} questions")
    return questions
