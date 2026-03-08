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
import time
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
from .prompts import build_prompt, extract_code
from .runners import MODEL_CONFIGS, create_runner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


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
                "course_id": "0",  # Default; can be mapped if needed
                "section_id": "0",  # Default; not used by IRT models
                "question_unittest_id": self.question_id,
                "attempt_id": str(rec.attempt_id),  # String to match real data
                "timestamp": rec.timestamp,  # Already in DD/MM/YY format
                "is_exam": "0",  # Default; used by RSSM featurization
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


def _build_initial_prompt(item: EvalItem, max_prompt_chars: int = 600_000) -> str:
    """Build the initial prompt for an item, trimming examples if too long."""
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

    # Trim oldest examples until prompt fits
    while len(prompt) > max_prompt_chars and examples and len(examples) > 1:
        examples = examples[1:]
        prompt = build_prompt(
            question_name=item.question_name,
            question_text=item.question_text,
            question_template=item.question_template,
            examples=examples,
        )

    return prompt


def _build_feedback_prompt(
    item: EvalItem, code: Optional[str], pub: dict, public_tests: list,
) -> str:
    """Build a feedback prompt from failed public test results."""
    examples = [
        {
            "question_name": ex.question_name,
            "question_text": ex.question_text,
            "question_template": ex.question_template,
            "response": ex.response,
        }
        for ex in item.examples
    ] or None

    if code is None:
        return build_prompt(
            question_name=item.question_name,
            question_text=item.question_text,
            question_template=item.question_template,
            examples=examples,
            feedback={
                "previous_code": "(could not parse code block)",
                "failed_tests": [{
                    "input": "", "std_in": "", "expected": "",
                    "actual": "Your response did not contain a valid "
                              "```cpp ... ``` code block.",
                }],
            },
        )

    failed_tests = []
    for j, r in enumerate(pub["testcases"]):
        if r == 0:
            tc = public_tests[j]
            actual = (
                pub["outputs"][j].strip() if pub["outputs"][j] else None
            )
            failed_tests.append({
                "input": tc["input"],
                "std_in": tc["std_in"],
                "expected": tc["output"],
                "actual": actual,
            })

    return build_prompt(
        question_name=item.question_name,
        question_text=item.question_text,
        question_template=item.question_template,
        examples=examples,
        feedback={"previous_code": code, "failed_tests": failed_tests},
    )


# ── Batch-iterative evaluation ──────────────────────────────────────────────


def run_batch_iterative(
    items: List[EvalItem],
    model_key: str,
    n_examples: int,
    max_attempts: int,
    n_public_map: Dict[str, int],
    tensor_parallel_size: int = 1,
    output_dir: Optional[str] = None,
    batch_size: int = 50,
    shard: Optional[str] = None,
) -> List[EvalResult]:
    """Batch-iterative evaluation: batch generate, compile/test, retry failures.

    Round 0: batch-generate all items
    Round 1..N: collect failures, build feedback, batch-generate only failures
    Each round uses runner.generate() for efficient batched inference.

    Every item gets response_type="Prechecked" for each attempt, then a final
    "Submit" with evaluation against ALL test cases — matching the human data.
    """
    from .grading_engine import CPPEvaluator

    runner = create_runner(model_key, tensor_parallel_size=tensor_parallel_size)

    # Pre-parse test cases and build evaluators
    test_data = {}  # idx -> {test_cases, public_tests, eval_pub, eval_all}
    for idx, item in enumerate(items):
        test_cases = parse_test_cases(item.question_unittests)
        if not test_cases:
            logger.warning("Q%s: no test cases, will skip grading", item.question_id)
            test_data[idx] = None
            continue
        n_pub = n_public_map.get(item.question_id, len(test_cases))
        n_pub = max(1, min(n_pub, len(test_cases)))
        public_tests = test_cases[:n_pub]
        test_data[idx] = {
            "test_cases": test_cases,
            "public_tests": public_tests,
            "eval_pub": CPPEvaluator(item.question_template, public_tests),
            "eval_all": CPPEvaluator(item.question_template, test_cases),
        }

    # Initialize results and tracking
    results: List[EvalResult] = [
        EvalResult(
            question_id=item.question_id,
            student_id=item.student_id,
            model=model_key,
            n_examples=n_examples,
        )
        for item in items
    ]
    last_code: List[Optional[str]] = [None] * len(items)
    # Active set: indices of items that still need more attempts
    active = list(range(len(items)))

    for attempt_id in range(max_attempts):
        if not active:
            break

        # Build prompts for active items
        prompts = []
        for idx in active:
            item = items[idx]
            if attempt_id == 0:
                prompts.append(_build_initial_prompt(item))
            else:
                # Build feedback from last attempt
                td = test_data[idx]
                if td is None:
                    prompts.append(_build_initial_prompt(item))
                else:
                    pub = td["eval_pub"].evaluate_with_outputs(
                        last_code[idx]
                    ) if last_code[idx] else {
                        "testcases": [0] * len(td["public_tests"]),
                        "outputs": [""] * len(td["public_tests"]),
                    }
                    prompts.append(_build_feedback_prompt(
                        item, last_code[idx], pub, td["public_tests"],
                    ))

        # Batch generate in chunks
        all_responses = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            logger.info(
                "[%s] Attempt %d/%d — generating %d–%d of %d active items…",
                model_key, attempt_id + 1, max_attempts,
                start + 1, start + len(chunk), len(active),
            )
            all_responses.extend(runner.generate(chunk))

        # Process results and determine who needs retry
        # Use DD/MM/YY format to match real student data (required by IRT models)
        timestamp = time.strftime("%d/%m/%y, %H:%M:%S")
        next_active = []
        for i, idx in enumerate(active):
            raw = all_responses[i]
            prompt = prompts[i]
            code = extract_code(raw)
            last_code[idx] = code

            td = test_data[idx]
            if td is None or code is None:
                # No test cases or no code — record and retry
                pass_pattern = ""
                if td and code is None:
                    pass_pattern = "0" * len(td["public_tests"])
            else:
                pub = td["eval_pub"].evaluate_with_outputs(code)
                pass_pattern = "".join(str(x) for x in pub["testcases"])

            results[idx].attempts.append(AttemptRecord(
                attempt_id=attempt_id,
                timestamp=timestamp,
                response_type="Prechecked",
                prompt=prompt,
                raw_response=raw,
                code=code,
                pass_pattern=pass_pattern,
            ))

            # Check if passed all public tests
            passed = td is not None and code is not None and all(
                c == "1" for c in pass_pattern
            )
            if not passed:
                next_active.append(idx)

        n_passed = len(active) - len(next_active)
        logger.info(
            "[%s] Attempt %d: %d/%d passed, %d remaining",
            model_key, attempt_id + 1, n_passed, len(active), len(next_active),
        )
        active = next_active

        # Save incrementally after each round
        if output_dir:
            save_results(
                results, output_dir, model_key,
                n_examples, max_attempts, shard=shard,
            )

    # Final evaluation on ALL tests → "Submit" record
    logger.info("[%s] Running final evaluation on all test cases…", model_key)
    for idx, item in enumerate(items):
        td = test_data[idx]
        if td is not None and last_code[idx]:
            final = td["eval_all"].evaluate_with_outputs(last_code[idx])
        else:
            n_tests = len(td["test_cases"]) if td else 0
            final = {
                "score": 0.0,
                "testcases": [0] * n_tests,
                "outputs": [""] * n_tests,
            }
        final_pass = "".join(str(x) for x in final["testcases"])
        results[idx].attempts.append(AttemptRecord(
            attempt_id=len(results[idx].attempts),
            timestamp=time.strftime("%d/%m/%y, %H:%M:%S"),
            response_type="Submit",
            prompt="",
            raw_response=last_code[idx] or "",
            code=last_code[idx],
            pass_pattern=final_pass,
        ))

    if output_dir:
        save_results(
            results, output_dir, model_key,
            n_examples, max_attempts, shard=shard,
        )

    if hasattr(runner, "cleanup"):
        runner.cleanup()

    return results


# ── Output ───────────────────────────────────────────────────────────────────


def save_results(
    results: List[EvalResult], output_dir: str, model_key: str,
    n_examples: int, max_attempts: int, shard: Optional[str] = None,
) -> str:
    """Save results to JSONL, return output path."""
    rows = []
    for r in results:
        rows.extend(r.to_rows())

    os.makedirs(output_dir, exist_ok=True)
    shard_suffix = f"_shard{shard.replace('/', 'of')}" if shard else ""
    filename = f"{model_key}_n{n_examples}_attempts{max_attempts}{shard_suffix}.jsonl"
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Saved %d rows → %s", len(rows), out_path)
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

    # ── Load public test counts ─────────────────────────────────────────
    from .data_loader import _load_hf_main_data
    logger.info("Loading main_data.csv for public-test inference…")
    main_df = _load_hf_main_data()
    n_public_map = infer_public_test_counts(main_df)
    logger.info("Inferred public-test counts for %d questions.", len(n_public_map))

    # ── Run each model ───────────────────────────────────────────────────
    for model_key in args.models:
        logger.info("=== Running %s ===", model_key)
        try:
            results = run_batch_iterative(
                items, model_key, args.n_examples,
                max_attempts=args.max_attempts,
                n_public_map=n_public_map,
                tensor_parallel_size=args.tp,
                output_dir=args.output_dir,
                shard=args.shard,
            )
        except Exception as e:
            logger.error("Model %s failed: %s", model_key, e, exc_info=True)


if __name__ == "__main__":
    main()
