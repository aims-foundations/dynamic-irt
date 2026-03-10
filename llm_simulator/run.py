"""Unified LLM student simulation entry point.

Replaces the old run_single_turn.py and run_iterative_model.py with one CLI
parameterized by n_examples (few-shot) and max_attempts (iterative feedback).

Usage:
    # Zero-shot (old S1)
    python -m llm_simulator.run --models claude gpt --n_examples 0

    # Few-shot student simulation (old S2/S3/S4)
    python -m llm_simulator.run --models claude gpt --n_examples 3

    # Iterative with feedback
    python -m llm_simulator.run --models claude --max_attempts 100

    # Few-shot + iterative
    python -m llm_simulator.run --models claude --n_examples 3 --max_attempts 5

    # Quick test
    python -m llm_simulator.run --models claude --n_examples 0 --max_samples 2 --dry_run
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .data_loader import (
    EvalItem,
    Example,
    infer_public_test_counts,
    load_eval_items,
    parse_test_cases,
)
from .prompts import build_prompt, extract_action, extract_code
from .runners import MODEL_CONFIGS, create_runner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ── Standalone grading (no nested executors — safe for ProcessPoolExecutor) ──


def _grade_single(template: str, testcases: list, code: str) -> dict:
    """Grade a single code submission against test cases.

    Runs sequentially (no internal process pool) so it's safe to call
    from ProcessPoolExecutor without deadlocking.

    Returns dict with 'score', 'testcases' (List[int]), 'outputs' (List[str]).
    """
    # Format test cases
    formatted = []
    std_inputs = []
    for tc in testcases:
        formatted.append({"extra": "", "testcode": tc["input"],
                          "expected_output": tc["output"]})
        std_inputs.append(tc.get("std_in", ""))

    # Generate code variants (one per test case)
    start_idx = template.find("{% for TEST in TESTCASES %}")
    end_idx = template.find("{% endfor %}") + len("{% endfor %}")
    code_with_answer = template.replace("{{ STUDENT_ANSWER }}", code)
    start_idx = code_with_answer.find("{% for TEST in TESTCASES %}")
    end_idx = code_with_answer.find("{% endfor %}") + len("{% endfor %}")

    codes = []
    for tc in formatted:
        codes.append(code_with_answer[:start_idx] + tc["testcode"]
                     + code_with_answer[end_idx:])

    # Compile and run in a temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        results_list = []
        outputs = []
        for i, src in enumerate(codes):
            cpp_file = os.path.join(temp_dir, f"tc_{i}.cpp")
            exe_file = os.path.join(temp_dir, f"tc_{i}.out")
            with open(cpp_file, "w") as f:
                f.write(src)

            # Compile
            comp = subprocess.run(
                ["g++", "-std=c++11", cpp_file, "-o", exe_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            if comp.returncode != 0:
                results_list.append(0)
                outputs.append("")
                continue

            # Run
            try:
                run = subprocess.run(
                    ["timeout", "10", exe_file],
                    input=std_inputs[i],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True,
                )
                if run.returncode != 0:
                    results_list.append(0)
                    outputs.append("")
                    continue
                outputs.append(run.stdout)
                expected = testcases[i]["output"]
                if expected.strip() == run.stdout.strip():
                    results_list.append(1)
                else:
                    results_list.append(0)
            except Exception:
                results_list.append(0)
                outputs.append("")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    score = sum(results_list) / len(results_list) if results_list else 0
    return {"score": score, "testcases": results_list, "outputs": outputs}


# ── Output records ───────────────────────────────────────────────────────────


@dataclass
class AttemptRecord:
    attempt_id: int
    timestamp: str
    response_type: str  # "Prechecked" or "Submit"
    prompt: str
    raw_response: str
    code: Optional[str]
    pass_pattern: str


@dataclass
class EvalResult:
    question_id: str
    student_id: Optional[str]
    model: str
    n_examples: int
    attempts: List[AttemptRecord] = field(default_factory=list)
    # Lookups from real data
    course_id: str = ""
    section_id: str = ""
    is_exam: str = ""

    def to_rows(self) -> List[dict]:
        """Convert to dicts matching human data schema exactly.

        Shared columns (same name/type as main_data.csv):
            student_id, course_id, section_id, question_unittest_id,
            attempt_id, timestamp, is_exam, response_type, response, pass
        LLM-specific columns:
            model, n_examples, prompt, raw_response
        """
        return [
            {
                # Schema-compatible fields (match main_data.csv exactly)
                "student_id": self.student_id or "",
                "course_id": self.course_id or "",
                "section_id": self.section_id or "",
                "question_unittest_id": self.question_id,
                "attempt_id": str(rec.attempt_id),  # String to match real data
                "timestamp": rec.timestamp,  # Already in DD/MM/YY format
                "is_exam": self.is_exam or "",
                "response_type": rec.response_type,
                "response": rec.code or "",
                "pass": rec.pass_pattern,
                # LLM-specific fields
                "model": self.model,
                "n_examples": self.n_examples,
                "prompt": rec.prompt,
                "raw_response": rec.raw_response,
            }
            for rec in self.attempts
        ]


# ── Prompt building ──────────────────────────────────────────────────────────


def _build_initial_prompt(item: EvalItem, max_prompt_chars: int = 300_000) -> str:
    """Build the initial prompt for an item, trimming examples if too long."""
    examples = [
        {
            "question_name": ex.question_name,
            "question_text": ex.question_text,
            "question_template": ex.question_template,
            "response": ex.response,
            "response_type": ex.response_type,
            "pass_pattern": ex.pass_pattern,
        }
        for ex in item.examples
    ] or None

    prompt = build_prompt(
        question_name=item.question_name,
        question_text=item.question_text,
        question_template=item.question_template,
        examples=examples,
    )

    # Trim oldest examples until prompt fits
    while len(prompt) > max_prompt_chars and examples:
        examples = examples[1:]
        prompt = build_prompt(
            question_name=item.question_name,
            question_text=item.question_text,
            question_template=item.question_template,
            examples=examples or None,
        )

    return prompt


def _build_feedback_prompt(
    item: EvalItem, code: Optional[str], pub: dict, public_tests: list,
    max_prompt_chars: int = 300_000,
) -> str:
    """Build a feedback prompt from failed test results."""
    examples = [
        {
            "question_name": ex.question_name,
            "question_text": ex.question_text,
            "question_template": ex.question_template,
            "response": ex.response,
            "response_type": ex.response_type,
            "pass_pattern": ex.pass_pattern,
        }
        for ex in item.examples
    ] or None

    if code is None:
        feedback = {
            "previous_code": "(could not parse code block)",
            "failed_tests": [{
                "input": "", "std_in": "", "expected": "",
                "actual": "Your response did not contain a valid "
                          "```cpp ... ``` code block.",
            }],
        }
    else:
        failed_tests = []
        for j, r in enumerate(pub["testcases"]):
            if r == 0:
                tc = public_tests[j]
                actual = (
                    pub["outputs"][j].strip()[:500] if pub["outputs"][j] else None
                )
                failed_tests.append({
                    "input": tc["input"][:500],
                    "std_in": tc["std_in"][:500],
                    "expected": tc["output"][:500],
                    "actual": actual,
                })
        # Limit to first 5 failed tests to avoid huge prompts
        feedback = {
            "previous_code": code,
            "failed_tests": failed_tests[:5],
        }

    prompt = build_prompt(
        question_name=item.question_name,
        question_text=item.question_text,
        question_template=item.question_template,
        examples=examples,
        feedback=feedback,
    )

    # Trim oldest examples until prompt fits
    while len(prompt) > max_prompt_chars and examples:
        examples = examples[1:]
        prompt = build_prompt(
            question_name=item.question_name,
            question_text=item.question_text,
            question_template=item.question_template,
            examples=examples or None,
            feedback=feedback,
        )

    # Safety net: if still too long (feedback/question text alone is huge),
    # rebuild without feedback to avoid context-length errors
    if len(prompt) > max_prompt_chars:
        prompt = build_prompt(
            question_name=item.question_name,
            question_text=item.question_text,
            question_template=item.question_template,
        )

    return prompt


# ── Batch-iterative evaluation ──────────────────────────────────────────────


def _run_chunk(
    chunk_items: List[EvalItem],
    runner,
    model_key: str,
    n_examples: int,
    max_attempts: int,
    n_public_map: Dict[str, int],
    chunk_label: str,
    batch_size: int = 50,
    max_submits: int = 50,
    early_stop_patience: int = 5,
    student_to_course: Optional[Dict[str, str]] = None,
    student_to_section: Optional[Dict[str, str]] = None,
    question_to_is_exam: Optional[Dict[str, str]] = None,
) -> List[EvalResult]:
    """Run all attempts (up to max_attempts) on a chunk of items.

    The LLM decides whether each attempt is a [Precheck] or [Submit],
    learning the pattern from the student's few-shot examples.
    - [Precheck]: graded against public tests only, free, feedback given.
    - [Submit]: graded against ALL tests, counts toward submit budget.
    - Items are done when: submit passes all tests, submit budget
      (max_submits) exhausted, max_attempts reached, or early-stop
      triggered (no improvement in pass pattern for early_stop_patience
      consecutive submits).
    """
    student_to_course = student_to_course or {}
    student_to_section = student_to_section or {}
    question_to_is_exam = question_to_is_exam or {}
    # Pre-parse test cases
    test_data = {}
    for idx, item in enumerate(chunk_items):
        test_cases = parse_test_cases(item.question_unittests)
        if not test_cases or not isinstance(item.question_template, str):
            test_data[idx] = None
            continue
        n_pub = n_public_map.get(item.question_id, len(test_cases))
        n_pub = max(1, min(n_pub, len(test_cases)))
        public_tests = test_cases[:n_pub]
        test_data[idx] = {
            "test_cases": test_cases,
            "public_tests": public_tests,
        }

    results = [
        EvalResult(
            question_id=item.question_id,
            student_id=item.student_id,
            model=model_key,
            n_examples=n_examples,
            course_id=student_to_course.get(str(item.student_id), ""),
            section_id=student_to_section.get(str(item.student_id), ""),
            is_exam=question_to_is_exam.get(str(item.question_id), ""),
        )
        for item in chunk_items
    ]
    last_code: List[Optional[str]] = [None] * len(chunk_items)
    last_eval: dict = {}   # idx -> last eval result (for feedback prompts)
    last_tests: dict = {}  # idx -> test cases used in last eval (pub or all)
    last_pass_pattern: dict = {}  # idx -> last submit pass_pattern (for stall detection)
    stall_count: List[int] = [0] * len(chunk_items)
    submits_used: List[int] = [0] * len(chunk_items)
    active = list(range(len(chunk_items)))

    for attempt_id in range(max_attempts):
        if not active:
            logger.info(
                "%s | All items done after %d attempts!",
                chunk_label, attempt_id,
            )
            break

        # Build prompts for active items
        prompts = []
        for idx in active:
            item = chunk_items[idx]
            if attempt_id == 0:
                prompts.append(_build_initial_prompt(item))
            else:
                td = test_data[idx]
                if td is None:
                    prompts.append(_build_initial_prompt(item))
                else:
                    tests_used = last_tests.get(idx, td["public_tests"])
                    ev = last_eval.get(idx, {
                        "testcases": [0] * len(tests_used),
                        "outputs": [""] * len(tests_used),
                    })
                    prompts.append(_build_feedback_prompt(
                        item, last_code[idx], ev, tests_used,
                    ))

        # Batch generate in sub-batches
        n_sub = (len(prompts) + batch_size - 1) // batch_size
        logger.info(
            "%s | Attempt %d/%d: %d active items (%d sub-batches of %d)",
            chunk_label, attempt_id + 1, max_attempts,
            len(active), n_sub, batch_size,
        )
        all_responses = []
        for sb, start in enumerate(range(0, len(prompts), batch_size), 1):
            sub = prompts[start:start + batch_size]
            logger.info(
                "%s |   sub-batch %d/%d (%d items)",
                chunk_label, sb, n_sub, len(sub),
            )
            all_responses.extend(runner.generate(sub))

        # Parse actions and codes
        # Use DD/MM/YY format to match real student data (required by IRT models)
        timestamp = time.strftime("%d/%m/%y, %H:%M:%S")
        actions = []
        codes_for_grading = []
        for i, idx in enumerate(active):
            action = extract_action(all_responses[i])
            code = extract_code(all_responses[i])
            # If model chose Submit but budget exhausted, downgrade to Precheck
            if action == "Submit" and submits_used[idx] >= max_submits:
                action = "Precheck"
            actions.append(action)
            last_code[idx] = code
            codes_for_grading.append(code)

        # Split into precheck vs submit groups for grading
        precheck_jobs = {}  # i -> idx (grade against public tests)
        submit_jobs = {}    # i -> idx (grade against ALL tests)
        for i, idx in enumerate(active):
            td = test_data[idx]
            if td is not None and codes_for_grading[i] is not None:
                if actions[i] == "Submit":
                    submit_jobs[i] = idx
                else:
                    precheck_jobs[i] = idx

        # Grade prechecks (public tests) in parallel
        precheck_results = {}
        if precheck_jobs:
            logger.info("%s | Grading %d prechecks (public tests)…",
                        chunk_label, len(precheck_jobs))
            n_workers = min(32, len(precheck_jobs))
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {}
                for i, idx in precheck_jobs.items():
                    td = test_data[idx]
                    futures[pool.submit(
                        _grade_single,
                        chunk_items[idx].question_template,
                        td["public_tests"],
                        codes_for_grading[i],
                    )] = i
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        precheck_results[i] = future.result()
                    except Exception as e:
                        logger.error("Precheck grading failed for item %d: %s", i, e)

        # Grade submits (ALL tests) in parallel
        submit_results = {}
        if submit_jobs:
            logger.info("%s | Grading %d submits (all tests)…",
                        chunk_label, len(submit_jobs))
            n_workers = min(32, len(submit_jobs))
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {}
                for i, idx in submit_jobs.items():
                    td = test_data[idx]
                    futures[pool.submit(
                        _grade_single,
                        chunk_items[idx].question_template,
                        td["test_cases"],
                        codes_for_grading[i],
                    )] = i
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        submit_results[i] = future.result()
                    except Exception as e:
                        logger.error("Submit grading failed for item %d: %s", i, e)

        # Process results and determine next active set
        next_active = []
        n_prechecks = 0
        n_submits = 0
        n_submit_passed = 0
        for i, idx in enumerate(active):
            raw = all_responses[i]
            prompt = prompts[i]
            code = codes_for_grading[i]
            action = actions[i]
            td = test_data[idx]

            if action == "Submit":
                n_submits += 1
                submits_used[idx] += 1
                if i in submit_results:
                    ev = submit_results[i]
                    pass_pattern = "".join(str(x) for x in ev["testcases"])
                    sub_passed = bool(pass_pattern) and all(c == "1" for c in pass_pattern)
                elif td is None or code is None:
                    n_tests = len(td["test_cases"]) if td else 0
                    pass_pattern = "0" * n_tests if n_tests else ""
                    ev = {"testcases": [0] * n_tests, "outputs": [""] * n_tests}
                    sub_passed = False
                else:
                    pass_pattern = ""
                    ev = {"testcases": [], "outputs": []}
                    sub_passed = False

                results[idx].attempts.append(AttemptRecord(
                    attempt_id=attempt_id,
                    timestamp=timestamp,
                    response_type="Submit",
                    prompt=prompt,
                    raw_response=raw,
                    code=code,
                    pass_pattern=pass_pattern,
                ))

                if sub_passed:
                    n_submit_passed += 1
                    # Done — passed all tests
                elif submits_used[idx] < max_submits:
                    # Check for stall: no improvement in pass pattern
                    prev_pp = last_pass_pattern.get(idx)
                    if prev_pp is not None and pass_pattern == prev_pp:
                        stall_count[idx] += 1
                    else:
                        stall_count[idx] = 0
                    last_pass_pattern[idx] = pass_pattern

                    if stall_count[idx] >= early_stop_patience:
                        # Stuck — stop retrying this item
                        pass
                    else:
                        # Failed but still have submits — continue
                        last_eval[idx] = ev
                        last_tests[idx] = td["test_cases"] if td else []
                        next_active.append(idx)
                # else: budget exhausted → done

            else:  # Precheck
                n_prechecks += 1
                if i in precheck_results:
                    ev = precheck_results[i]
                    pass_pattern = "".join(str(x) for x in ev["testcases"])
                elif td is None or code is None:
                    n_pub = len(td["public_tests"]) if td else 0
                    pass_pattern = "0" * n_pub if n_pub else ""
                    ev = {"testcases": [0] * n_pub, "outputs": [""] * n_pub}
                else:
                    pass_pattern = ""
                    ev = {"testcases": [], "outputs": []}

                results[idx].attempts.append(AttemptRecord(
                    attempt_id=attempt_id,
                    timestamp=timestamp,
                    response_type="Prechecked",
                    prompt=prompt,
                    raw_response=raw,
                    code=code,
                    pass_pattern=pass_pattern,
                ))

                last_eval[idx] = ev
                last_tests[idx] = td["public_tests"] if td else []
                next_active.append(idx)

        n_done = len(active) - len(next_active)
        n_early_stopped = sum(1 for idx in active
                              if stall_count[idx] >= early_stop_patience
                              and idx not in next_active)
        logger.info(
            "%s | Attempt %d summary: %d precheck, %d submit (%d passed), "
            "%d done (%d early-stopped), %d continue",
            chunk_label, attempt_id + 1,
            n_prechecks, n_submits, n_submit_passed,
            n_done, n_early_stopped, len(next_active),
        )
        active = next_active

    # Summary
    n_submitted = sum(1 for s in submits_used if s > 0)
    n_all_pass = sum(
        1 for r in results
        if any(a.response_type == "Submit"
               and a.pass_pattern and all(c == "1" for c in a.pass_pattern)
               for a in r.attempts)
    )
    logger.info(
        "%s | Done! %d/%d submitted, %d passed all tests (%.0f%%)",
        chunk_label, n_submitted, len(chunk_items),
        n_all_pass, 100.0 * n_all_pass / len(chunk_items),
    )
    return results


def run_batch_iterative(
    items: List[EvalItem],
    model_key: str,
    n_examples: int,
    max_attempts: int,
    n_public_map: Dict[str, int],
    student_to_course: Optional[Dict[str, str]] = None,
    student_to_section: Optional[Dict[str, str]] = None,
    question_to_is_exam: Optional[Dict[str, str]] = None,
    tensor_parallel_size: int = 1,
    output_dir: Optional[str] = None,
    batch_size: int = 50,
    chunk_size: int = 500,
    shard: Optional[str] = None,
    port: Optional[int] = None,
    max_submits: int = 50,
    early_stop_patience: int = 5,
) -> List[EvalResult]:
    """Chunked batch-iterative evaluation.

    Processes items in chunks of `chunk_size`. Each chunk runs all attempts
    (up to max_attempts) before moving to the next, so we get complete
    multi-attempt data for finished chunks immediately.

    Within each chunk, items are batched (batch_size) for efficient vLLM
    inference, and only failed items retry in subsequent attempts.
    """
    runner = create_runner(model_key, tensor_parallel_size=tensor_parallel_size, port=port)
    all_results: List[EvalResult] = []
    n_chunks = (len(items) + chunk_size - 1) // chunk_size

    # Resume: load existing results and skip completed chunks
    skip_chunks = 0
    existing_rows = []
    if output_dir:
        shard_suffix = f"_shard{shard.replace('/', 'of')}" if shard else ""
        filename = f"{model_key}_n{n_examples}_attempts{max_attempts}{shard_suffix}.jsonl"
        existing_path = os.path.join(output_dir, filename)
        if os.path.exists(existing_path):
            with open(existing_path) as f:
                existing_rows = [json.loads(line) for line in f]
            if existing_rows:
                existing_pairs = set(
                    (r["student_id"], r["question_unittest_id"])
                    for r in existing_rows
                )
                skip_chunks = len(existing_pairs) // chunk_size
                logger.info(
                    "Resume: found %d existing pairs (%d complete chunks) in %s",
                    len(existing_pairs), skip_chunks, existing_path,
                )

    for chunk_idx, chunk_start in enumerate(range(0, len(items), chunk_size)):
        if chunk_idx < skip_chunks:
            continue

        chunk_items = items[chunk_start:chunk_start + chunk_size]
        chunk_label = f"[{model_key}] Chunk {chunk_idx + 1}/{n_chunks}"
        logger.info(
            "========== %s: items %d–%d of %d ==========",
            chunk_label, chunk_start + 1,
            chunk_start + len(chunk_items), len(items),
        )

        chunk_results = _run_chunk(
            chunk_items, runner, model_key, n_examples,
            max_attempts, n_public_map, chunk_label, batch_size,
            max_submits=max_submits,
            early_stop_patience=early_stop_patience,
            student_to_course=student_to_course,
            student_to_section=student_to_section,
            question_to_is_exam=question_to_is_exam,
        )
        all_results.extend(chunk_results)

        # Save after each chunk — complete multi-attempt data
        if output_dir:
            save_results(
                all_results, output_dir, model_key,
                n_examples, max_attempts, shard=shard,
                prepend_rows=existing_rows if skip_chunks > 0 else None,
            )
            logger.info(
                "%s | Saved %d total results so far (%d new + %d resumed)",
                chunk_label,
                len(all_results) + (len(existing_rows) if skip_chunks > 0 else 0),
                len(all_results),
                len(existing_rows) if skip_chunks > 0 else 0,
            )

    if hasattr(runner, "cleanup"):
        runner.cleanup()

    return all_results


# ── Output ───────────────────────────────────────────────────────────────────


def save_results(
    results: List[EvalResult], output_dir: str, model_key: str,
    n_examples: int, max_attempts: int, shard: Optional[str] = None,
    prepend_rows: Optional[List[dict]] = None,
) -> str:
    """Save results to JSONL, return output path."""
    os.makedirs(output_dir, exist_ok=True)
    shard_suffix = f"_shard{shard.replace('/', 'of')}" if shard else ""
    filename = f"{model_key}_n{n_examples}_attempts{max_attempts}{shard_suffix}.jsonl"
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w") as f:
        # Write resumed rows first
        if prepend_rows:
            for row in prepend_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Write new results
        for r in results:
            for row in r.to_rows():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_total = (len(prepend_rows) if prepend_rows else 0) + sum(
        len(r.attempts) for r in results
    )
    logger.info("Saved %d rows → %s", n_total, out_path)
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Unified LLM student simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m llm_simulator.run --models claude gpt --n_examples 0\n"
            "  python -m llm_simulator.run --models claude --n_examples 3\n"
            "  python -m llm_simulator.run --models claude --max_attempts 100\n"
            "  python -m llm_simulator.run --models claude --n_examples 0 "
            "--max_samples 2 --dry_run\n"
        ),
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_CONFIGS.keys()),
        default=["claude"],
        help="Models to evaluate (default: claude)",
    )
    parser.add_argument(
        "--n_examples", type=int, default=0,
        help="Number of in-context examples (0=zero-shot, N=few-shot)",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=50,
        help="Max attempts per question (default: 50, matching p99 student behavior)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit total number of items (for testing)",
    )
    parser.add_argument(
        "--max_students", type=int, default=None,
        help="Randomly sample N students (for trajectory mode)",
    )
    parser.add_argument(
        "--max_questions", type=int, default=None,
        help="Randomly sample N questions (for trajectory mode)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for student/question sampling (default: 42)",
    )
    parser.add_argument(
        "--data_dir", default="data",
        help="Directory with scenario CSVs (default: data/)",
    )
    parser.add_argument(
        "--output_dir", default="results/llm_eval",
        help="Output directory for result CSVs (default: results/llm_eval/)",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor-parallel size for vLLM models (default: 1)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load data and build prompts, but don't call any LLM",
    )
    parser.add_argument(
        "--shard", type=str, default=None,
        help="Shard index/total, e.g. '0/2' or '1/2' for parallel runs",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override vLLM server port (for glm_server mode)",
    )
    parser.add_argument(
        "--max_submits", type=int, default=50,
        help="Max submit attempts per question (default: 50, safety cap covering 99.5%% of real student behavior)",
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=5,
        help="Stop retrying an item after N consecutive submits with no improvement (default: 5)",
    )
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    items = load_eval_items(
        n_examples=args.n_examples,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        max_students=args.max_students,
        max_questions=args.max_questions,
        seed=args.seed,
    )
    logger.info(
        "Loaded %d items (n_examples=%d)", len(items), args.n_examples,
    )

    # ── Shard items for parallel runs ─────────────────────────────────────
    if args.shard:
        shard_idx, n_shards = map(int, args.shard.split("/"))
        shard_size = len(items) // n_shards
        start = shard_idx * shard_size
        end = start + shard_size if shard_idx < n_shards - 1 else len(items)
        items = items[start:end]
        logger.info(
            "Shard %d/%d: items %d–%d (%d items)",
            shard_idx, n_shards, start, end - 1, len(items),
        )

    if args.dry_run:
        # Show sample prompts
        for item in items[:2]:
            examples = [
                {
                    "question_name": ex.question_name,
                    "question_text": ex.question_text,
                    "question_template": ex.question_template,
                    "response": ex.response,
                }
                for ex in item.examples
            ] or None
            prompt = build_prompt(
                question_name=item.question_name,
                question_text=item.question_text,
                question_template=item.question_template,
                examples=examples,
            )
            logger.info(
                "--- Item: Q=%s, S=%s ---\n%s\n",
                item.question_id, item.student_id, prompt[:500],
            )
        logger.info("Dry run complete. %d items loaded.", len(items))
        return

    # ── Load public test counts and student info ────────────────────────
    from .data_loader import _load_hf_main_data
    logger.info("Loading main_data.csv for public-test inference and student info…")
    main_df = _load_hf_main_data()
    n_public_map = infer_public_test_counts(main_df)
    logger.info("Inferred public-test counts for %d questions.", len(n_public_map))

    # Build student → course_id and section_id mappings from real data
    student_info = main_df[["student_id", "course_id", "section_id"]].drop_duplicates(
        subset=["student_id"]
    )
    student_to_course = dict(
        zip(student_info["student_id"].astype(str), student_info["course_id"].astype(str))
    )
    student_to_section = dict(
        zip(student_info["student_id"].astype(str), student_info["section_id"].astype(str))
    )
    logger.info("Built student info mappings for %d students.", len(student_to_course))

    # Build question → is_exam mapping from real data
    question_info = main_df[["question_unittest_id", "is_exam"]].drop_duplicates(
        subset=["question_unittest_id"]
    )
    question_to_is_exam = dict(
        zip(question_info["question_unittest_id"].astype(str), question_info["is_exam"].astype(str))
    )
    logger.info("Built question info mappings for %d questions.", len(question_to_is_exam))

    # ── Run each model ───────────────────────────────────────────────────
    for model_key in args.models:
        logger.info("=== Running %s ===", model_key)
        try:
            results = run_batch_iterative(
                items, model_key, args.n_examples,
                max_attempts=args.max_attempts,
                n_public_map=n_public_map,
                student_to_course=student_to_course,
                student_to_section=student_to_section,
                question_to_is_exam=question_to_is_exam,
                tensor_parallel_size=args.tp,
                output_dir=args.output_dir,
                shard=args.shard,
                port=args.port,
                max_submits=args.max_submits,
                early_stop_patience=args.early_stop_patience,
            )
        except Exception as e:
            logger.error("Model %s failed: %s", model_key, e, exc_info=True)


if __name__ == "__main__":
    main()
