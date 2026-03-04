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
        return [
            {
                "question_id": self.question_id,
                "student_id": self.student_id or "",
                "model": self.model,
                "n_examples": self.n_examples,
                "attempt_id": rec.attempt_id,
                "timestamp": rec.timestamp,
                "response_type": rec.response_type,
                "raw_response": rec.raw_response,
                "code": rec.code or "",
                "pass_pattern": rec.pass_pattern,
            }
            for rec in self.attempts
        ]


# ── Single-shot batch evaluation ─────────────────────────────────────────────


def run_single_shot(
    items: List[EvalItem],
    model_key: str,
    n_examples: int,
    tensor_parallel_size: int = 1,
) -> List[EvalResult]:
    """Run all items in a single batch (no feedback loop)."""
    runner = create_runner(model_key, tensor_parallel_size=tensor_parallel_size)

    # Build prompts
    prompts = []
    for item in items:
        examples = [
            {
                "question_name": ex.question_name,
                "question_text": ex.question_text,
                "question_template": ex.question_template,
                "response": ex.response,
            }
            for ex in item.examples
        ] or None
        prompts.append(build_prompt(
            question_name=item.question_name,
            question_text=item.question_text,
            question_template=item.question_template,
            examples=examples,
        ))

    logger.info("[%s] Generating %d responses (single-shot)…", model_key, len(prompts))
    responses = runner.generate(prompts)

    # Wrap as EvalResults
    results = []
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for item, raw in zip(items, responses):
        code = extract_code(raw)
        result = EvalResult(
            question_id=item.question_id,
            student_id=item.student_id,
            model=model_key,
            n_examples=n_examples,
        )
        result.attempts.append(AttemptRecord(
            attempt_id=0,
            timestamp=timestamp,
            response_type="Submit",
            raw_response=raw,
            code=code,
            pass_pattern="",  # computed later if --compute_metrics
        ))
        results.append(result)

    # Clean up GPU if vLLM
    if hasattr(runner, "cleanup"):
        runner.cleanup()

    return results


# ── Iterative evaluation with feedback ───────────────────────────────────────


def run_iterative(
    items: List[EvalItem],
    model_key: str,
    n_examples: int,
    max_attempts: int,
    n_public_map: Dict[str, int],
    tensor_parallel_size: int = 1,
) -> List[EvalResult]:
    """Run each item iteratively with compile-and-test feedback."""
    from .grading_engine import CPPEvaluator

    runner = create_runner(model_key, tensor_parallel_size=tensor_parallel_size)
    results = []

    for i, item in enumerate(tqdm(items, desc=f"[{model_key}] iterative")):
        test_cases = parse_test_cases(item.question_unittests)
        if not test_cases:
            logger.warning("Q%s: failed to parse test cases, skipping", item.question_id)
            continue

        n_public = n_public_map.get(item.question_id, len(test_cases))
        n_public = max(1, min(n_public, len(test_cases)))

        public_tests = test_cases[:n_public]
        evaluator_public = CPPEvaluator(item.question_template, public_tests)
        evaluator_all = CPPEvaluator(item.question_template, test_cases)

        # Build initial prompt
        examples = [
            {
                "question_name": ex.question_name,
                "question_text": ex.question_text,
                "question_template": ex.question_template,
                "response": ex.response,
            }
            for ex in item.examples
        ] or None

        initial_prompt = build_prompt(
            question_name=item.question_name,
            question_text=item.question_text,
            question_template=item.question_template,
            examples=examples,
        )

        result = EvalResult(
            question_id=item.question_id,
            student_id=item.student_id,
            model=model_key,
            n_examples=n_examples,
        )

        current_prompt = initial_prompt
        last_code: Optional[str] = None

        for attempt_id in range(max_attempts):
            raw_output = runner.call(current_prompt)
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            code = extract_code(raw_output)

            if code is None:
                result.attempts.append(AttemptRecord(
                    attempt_id=attempt_id,
                    timestamp=timestamp,
                    response_type="Prechecked",
                    raw_response=raw_output,
                    code=None,
                    pass_pattern="0" * len(public_tests),
                ))
                current_prompt = build_prompt(
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
                continue

            last_code = code

            # Evaluate against public tests
            pub = evaluator_public.evaluate_with_outputs(code)
            pass_pattern = "".join(str(x) for x in pub["testcases"])

            result.attempts.append(AttemptRecord(
                attempt_id=attempt_id,
                timestamp=timestamp,
                response_type="Prechecked",
                raw_response=raw_output,
                code=code,
                pass_pattern=pass_pattern,
            ))

            logger.info(
                "  [%s] Q%s attempt %d/%d: public %.0f%% (%s)",
                model_key, item.question_id,
                attempt_id + 1, max_attempts,
                pub["score"] * 100, pass_pattern,
            )

            if pub["score"] == 1.0:
                break

            # Build feedback for next attempt
            failed_tests = []
            for j, r in enumerate(pub["testcases"]):
                if r == 0:
                    tc = public_tests[j]
                    actual = (
                        pub["outputs"][j].strip()
                        if pub["outputs"][j]
                        else None
                    )
                    failed_tests.append({
                        "input": tc["input"],
                        "std_in": tc["std_in"],
                        "expected": tc["output"],
                        "actual": actual,
                    })

            current_prompt = build_prompt(
                question_name=item.question_name,
                question_text=item.question_text,
                question_template=item.question_template,
                examples=examples,
                feedback={
                    "previous_code": code,
                    "failed_tests": failed_tests,
                },
            )

        # Final evaluation on ALL tests
        if last_code:
            final = evaluator_all.evaluate_with_outputs(last_code)
        else:
            final = {
                "score": 0.0,
                "testcases": [0] * len(test_cases),
                "outputs": [""] * len(test_cases),
            }

        final_pass = "".join(str(x) for x in final["testcases"])
        result.attempts.append(AttemptRecord(
            attempt_id=len(result.attempts),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            response_type="Submit",
            raw_response=last_code or "",
            code=last_code,
            pass_pattern=final_pass,
        ))

        logger.info(
            "  [%s] Q%s SUBMIT: %.0f%% (%s) after %d attempt(s)",
            model_key, item.question_id,
            final["score"] * 100, final_pass,
            len(result.attempts) - 1,
        )
        results.append(result)

    if hasattr(runner, "cleanup"):
        runner.cleanup()

    return results


# ── Output ───────────────────────────────────────────────────────────────────


def save_results(
    results: List[EvalResult], output_dir: str, model_key: str,
    n_examples: int, max_attempts: int,
) -> str:
    """Save results to CSV, return output path."""
    rows = []
    for r in results:
        rows.extend(r.to_rows())

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_key}_n{n_examples}_attempts{max_attempts}.csv"
    out_path = os.path.join(output_dir, filename)
    pd.DataFrame(rows).to_csv(out_path, index=False)
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
        "--max_attempts", type=int, default=1,
        help="Max attempts per question (1=single-shot, >1=iterative with feedback)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit number of items (for testing)",
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
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    items = load_eval_items(
        n_examples=args.n_examples,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
    )
    logger.info(
        "Loaded %d items (n_examples=%d)", len(items), args.n_examples,
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

    # ── Load public test counts (for iterative mode) ─────────────────────
    n_public_map: Dict[str, int] = {}
    if args.max_attempts > 1:
        from .data_loader import _load_hf_main_data
        logger.info("Loading main_data.csv for public-test inference…")
        main_df = _load_hf_main_data()
        n_public_map = infer_public_test_counts(main_df)
        logger.info("Inferred public-test counts for %d questions.", len(n_public_map))

    # ── Run each model ───────────────────────────────────────────────────
    for model_key in args.models:
        logger.info("=== Running %s ===", model_key)
        try:
            if args.max_attempts <= 1:
                results = run_single_shot(
                    items, model_key, args.n_examples,
                    tensor_parallel_size=args.tp,
                )
            else:
                results = run_iterative(
                    items, model_key, args.n_examples,
                    max_attempts=args.max_attempts,
                    n_public_map=n_public_map,
                    tensor_parallel_size=args.tp,
                )
            save_results(
                results, args.output_dir, model_key,
                args.n_examples, args.max_attempts,
            )
        except Exception as e:
            logger.error("Model %s failed: %s", model_key, e, exc_info=True)


if __name__ == "__main__":
    main()
