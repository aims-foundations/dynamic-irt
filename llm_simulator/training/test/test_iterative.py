"""Unit tests for run_iterative_model.py components.

Tests are grouped by function and cover:
  - parse_test_cases: string parsing of the question_unittests field
  - extract_code: regex extraction of C++ code from LLM markdown output
  - infer_public_test_counts: pandas-based N_public inference
  - build_initial_prompt / build_feedback_prompt: prompt builders
  - CPPEvaluator.evaluate_with_outputs: compile + run + output capture

Run with:
    python -m pytest training/test/test_iterative.py -v
or:
    python training/test/test_iterative.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None  # tests are runnable standalone without pytest

from run_iterative_model import (
    build_feedback_prompt,
    build_initial_prompt,
    extract_code,
    infer_public_test_counts,
    parse_test_cases,
)
from grading_engine import CPPEvaluator


# ── Fixtures / shared data ────────────────────────────────────────────────────

# Matches the format produced by data_preprocessing.py / the online judge.
# Format:  "Unittest N: Input: <code> STD input: <stdin> Output: <expected>"
SAMPLE_UNITTEST_STR = (
    "Unittest 1: "
    "Input: int x = 5; cout << x << endl; "
    "STD input:  "
    "Output: 5\n"
    "Unittest 2: "
    "Input: int x = 10; cout << x << endl; "
    "STD input: 42\n"
    "Output: 10"
)

SIMPLE_TEMPLATE = """
#include <iostream>
using namespace std;
{{ STUDENT_ANSWER }}
int main() {
    {% for TEST in TESTCASES %}
    {
        {{ TEST.extra }};
        {{ TEST.testcode }};
    }
    {% endfor %}
    return 0;
}
"""


# ── parse_test_cases ──────────────────────────────────────────────────────────


def test_parse_test_cases_returns_list():
    cases = parse_test_cases(SAMPLE_UNITTEST_STR)
    assert isinstance(cases, list)
    assert len(cases) == 2


def test_parse_test_cases_fields_present():
    cases = parse_test_cases(SAMPLE_UNITTEST_STR)
    for tc in cases:
        assert "input" in tc
        assert "std_in" in tc
        assert "output" in tc


def test_parse_test_cases_values():
    cases = parse_test_cases(SAMPLE_UNITTEST_STR)
    assert cases[0]["output"] == "5"
    assert cases[1]["output"] == "10"
    assert cases[1]["std_in"] == "42"


def test_parse_test_cases_empty_stdin():
    cases = parse_test_cases(SAMPLE_UNITTEST_STR)
    # First test case has empty STD input
    assert cases[0]["std_in"] == ""


def test_parse_test_cases_malformed_missing_output():
    # Missing "Output:" tag → should return []
    bad = "Unittest 1: Input: int x = 1; STD input: "
    assert parse_test_cases(bad) == []


def test_parse_test_cases_malformed_missing_stdin():
    bad = "Unittest 1: Input: int x = 1; Output: 1"
    assert parse_test_cases(bad) == []


def test_parse_test_cases_single_unittest():
    single = (
        "Unittest 1: "
        "Input: cout << 42; "
        "STD input:  "
        "Output: 42"
    )
    cases = parse_test_cases(single)
    assert len(cases) == 1
    assert cases[0]["output"] == "42"


# ── extract_code ──────────────────────────────────────────────────────────────


def test_extract_code_cpp_fence():
    output = "```cpp\nint x = 1;\n```"
    assert extract_code(output) == "int x = 1;"


def test_extract_code_plain_fence():
    output = "```\nint x = 2;\n```"
    assert extract_code(output) == "int x = 2;"


def test_extract_code_cppplus_fence():
    output = "```c++\nint y = 3;\n```"
    assert extract_code(output) == "int y = 3;"


def test_extract_code_multiline_body():
    output = "```cpp\nint add(int a, int b) {\n    return a + b;\n}\n```"
    code = extract_code(output)
    assert code is not None
    assert "int add" in code
    assert "return a + b" in code


def test_extract_code_no_fence_returns_none():
    assert extract_code("just some explanation text") is None


def test_extract_code_only_opening_fence():
    # No closing ``` — regex falls back to $ (end of string)
    output = "```cpp\nint x = 1;"
    code = extract_code(output)
    assert code is not None
    assert "int x = 1;" in code


def test_extract_code_with_surrounding_text():
    output = "Here is my solution:\n```cpp\nreturn 42;\n```\nHope it helps!"
    assert extract_code(output) == "return 42;"


# ── infer_public_test_counts ──────────────────────────────────────────────────


def test_infer_public_test_counts_basic():
    df = pd.DataFrame({
        "response_type": ["Prechecked", "Prechecked", "Submit", "Prechecked"],
        "question_unittest_id": [1, 1, 1, 2],
        "pass": ["101", "101", "10110", "11"],
    })
    result = infer_public_test_counts(df)
    assert result["1"] == 3   # len("101") = 3
    assert result["2"] == 2   # len("11") = 2


def test_infer_public_test_counts_ignores_submit():
    df = pd.DataFrame({
        "response_type": ["Submit", "Submit"],
        "question_unittest_id": [1, 2],
        "pass": ["111", "00000"],
    })
    result = infer_public_test_counts(df)
    # No Prechecked rows → empty dict
    assert result == {}


def test_infer_public_test_counts_mode_with_variation():
    # Three rows for Q1: lengths 3, 3, 4 → mode is 3
    df = pd.DataFrame({
        "response_type": ["Prechecked", "Prechecked", "Prechecked"],
        "question_unittest_id": [1, 1, 1],
        "pass": ["101", "011", "1011"],
    })
    result = infer_public_test_counts(df)
    assert result["1"] == 3


def test_infer_public_test_counts_keys_are_strings():
    df = pd.DataFrame({
        "response_type": ["Prechecked"],
        "question_unittest_id": [42],
        "pass": ["10"],
    })
    result = infer_public_test_counts(df)
    assert "42" in result


# ── build_initial_prompt ──────────────────────────────────────────────────────


def test_build_initial_prompt_contains_question_name():
    prompt = build_initial_prompt("Sort Array", "Sort the given array.", "{{ STUDENT_ANSWER }}")
    assert "Sort Array" in prompt


def test_build_initial_prompt_contains_question_text():
    prompt = build_initial_prompt("Sort Array", "Sort the given array.", "{{ STUDENT_ANSWER }}")
    assert "Sort the given array." in prompt


def test_build_initial_prompt_contains_template():
    prompt = build_initial_prompt("Q", "desc", "template_code_here")
    assert "template_code_here" in prompt


def test_build_initial_prompt_contains_cpp_instruction():
    prompt = build_initial_prompt("Q", "desc", "tmpl")
    assert "```cpp" in prompt


# ── build_feedback_prompt ─────────────────────────────────────────────────────


def test_build_feedback_prompt_includes_initial():
    initial = "Original question prompt"
    feedback = build_feedback_prompt(
        initial, "int x = 1;",
        [{"input": "1", "std_in": "", "output": "2"}],
        [0], ["1"],
    )
    assert initial in feedback


def test_build_feedback_prompt_shows_previous_code():
    feedback = build_feedback_prompt(
        "prompt", "int answer = 99;",
        [{"input": "x", "std_in": "", "output": "y"}],
        [0], ["z"],
    )
    assert "int answer = 99;" in feedback
    assert "Previous Attempt" in feedback


def test_build_feedback_prompt_shows_failed_test_info():
    feedback = build_feedback_prompt(
        "prompt", "int x = 1;",
        [{"input": "test_input_code", "std_in": "42", "output": "expected_val"}],
        [0], ["actual_val"],
    )
    assert "test_input_code" in feedback
    assert "42" in feedback
    assert "expected_val" in feedback
    assert "actual_val" in feedback


def test_build_feedback_prompt_no_failed_tests():
    # All tests pass — feedback still includes "Previous Attempt" structure
    feedback = build_feedback_prompt(
        "prompt", "int x = 2;",
        [{"input": "1", "std_in": "", "output": "2"}],
        [1], ["2"],
    )
    assert isinstance(feedback, str)
    assert "Previous Attempt" in feedback
    # No failure details (no failed tests)
    assert "Expected" not in feedback


def test_build_feedback_prompt_partial_failures():
    # 2 tests: first passes, second fails
    feedback = build_feedback_prompt(
        "prompt", "code",
        [
            {"input": "in1", "std_in": "", "output": "out1"},
            {"input": "in2", "std_in": "", "output": "out2"},
        ],
        [1, 0], ["out1", "wrong"],
    )
    # Only the failing test (index 1) should appear in feedback
    assert "in2" in feedback
    assert "out2" in feedback


# ── CPPEvaluator.evaluate_with_outputs ───────────────────────────────────────

_HELLO_TEMPLATE = """
#include <iostream>
using namespace std;
{{ STUDENT_ANSWER }}
int main() {
    {% for TEST in TESTCASES %}
    {
        {{ TEST.extra }};
        {{ TEST.testcode }};
    }
    {% endfor %}
    return 0;
}
"""

_HELLO_TESTCASES = [
    {"input": 'cout << greet() << endl;', "std_in": "", "output": "hello"},
    {"input": 'cout << greet() << endl;', "std_in": "", "output": "hello"},
]


def test_evaluate_with_outputs_correct_code():
    evaluator = CPPEvaluator(_HELLO_TEMPLATE, _HELLO_TESTCASES)
    code = 'string greet() { return "hello"; }'
    result = evaluator.evaluate_with_outputs(code)
    assert result["score"] == 1.0
    assert all(r == 1 for r in result["testcases"])
    assert all("hello" in o for o in result["outputs"])


def test_evaluate_with_outputs_wrong_code():
    evaluator = CPPEvaluator(_HELLO_TEMPLATE, _HELLO_TESTCASES)
    code = 'string greet() { return "world"; }'
    result = evaluator.evaluate_with_outputs(code)
    assert result["score"] == 0.0
    assert all(r == 0 for r in result["testcases"])


def test_evaluate_with_outputs_compile_error():
    evaluator = CPPEvaluator(_HELLO_TEMPLATE, _HELLO_TESTCASES)
    code = "this is not valid C++"
    result = evaluator.evaluate_with_outputs(code)
    assert result["score"] == 0.0
    assert all(r == 0 for r in result["testcases"])
    assert all(o == "" for o in result["outputs"])


def test_evaluate_with_outputs_returns_outputs_list():
    evaluator = CPPEvaluator(_HELLO_TEMPLATE, _HELLO_TESTCASES)
    code = 'string greet() { return "hello"; }'
    result = evaluator.evaluate_with_outputs(code)
    assert "outputs" in result
    assert len(result["outputs"]) == len(_HELLO_TESTCASES)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        # parse_test_cases
        test_parse_test_cases_returns_list,
        test_parse_test_cases_fields_present,
        test_parse_test_cases_values,
        test_parse_test_cases_empty_stdin,
        test_parse_test_cases_malformed_missing_output,
        test_parse_test_cases_malformed_missing_stdin,
        test_parse_test_cases_single_unittest,
        # extract_code
        test_extract_code_cpp_fence,
        test_extract_code_plain_fence,
        test_extract_code_cppplus_fence,
        test_extract_code_multiline_body,
        test_extract_code_no_fence_returns_none,
        test_extract_code_only_opening_fence,
        test_extract_code_with_surrounding_text,
        # infer_public_test_counts
        test_infer_public_test_counts_basic,
        test_infer_public_test_counts_ignores_submit,
        test_infer_public_test_counts_mode_with_variation,
        test_infer_public_test_counts_keys_are_strings,
        # build_initial_prompt
        test_build_initial_prompt_contains_question_name,
        test_build_initial_prompt_contains_question_text,
        test_build_initial_prompt_contains_template,
        test_build_initial_prompt_contains_cpp_instruction,
        # build_feedback_prompt
        test_build_feedback_prompt_includes_initial,
        test_build_feedback_prompt_shows_previous_code,
        test_build_feedback_prompt_shows_failed_test_info,
        test_build_feedback_prompt_no_failed_tests,
        test_build_feedback_prompt_partial_failures,
        # CPPEvaluator.evaluate_with_outputs
        test_evaluate_with_outputs_correct_code,
        test_evaluate_with_outputs_wrong_code,
        test_evaluate_with_outputs_compile_error,
        test_evaluate_with_outputs_returns_outputs_list,
    ]

    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests.")
    sys.exit(0 if failed == 0 else 1)
