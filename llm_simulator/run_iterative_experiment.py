#!/usr/bin/env python3
"""
Run Iterative Code Generation Experiment with Claude.

Usage:
    python run_iterative_experiment.py --api-key YOUR_API_KEY

Or set ANTHROPIC_API_KEY environment variable:
    export ANTHROPIC_API_KEY=YOUR_API_KEY
    python run_iterative_experiment.py
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, login

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from codeinsights_iterative_runner import (
    IterativeCodeRunner,
    load_questions,
    SessionLog,
)


# --- Configuration ---

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_NUM_QUESTIONS = 5
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_NUM_PUBLIC_TESTS = 3

HF_ORGANIZATION = "CodeInsightTeam"
HF_REPO_NAME = "iterative_llm_results"


# --- Result Formatting ---

def session_logs_to_dataframe(session_logs: list) -> pd.DataFrame:
    """
    Convert session logs to DataFrame matching main_data.csv format.

    Columns:
    - model_id: LLM identifier (replaces student_id)
    - question_unittest_id: Question ID
    - attempt_id: Iteration number
    - timestamp: When attempt was made
    - response_type: "precheck" or "check"
    - response: Generated code
    - pass: Pass pattern (e.g., "11101")
    """
    rows = []
    for session in session_logs:
        for attempt in session.attempts:
            rows.append({
                "model_id": session.model_id,
                "question_unittest_id": session.question_id,
                "attempt_id": attempt.attempt_id,
                "timestamp": attempt.timestamp,
                "response_type": attempt.response_type,
                "response": attempt.code,
                "pass": attempt.pass_pattern,
            })

    return pd.DataFrame(rows)


def upload_to_huggingface(df: pd.DataFrame, model_name: str, hf_token: str = None):
    """Upload results to HuggingFace."""
    # Create dataset
    dataset = Dataset.from_pandas(df)

    # Generate unique config name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = model_name.replace("/", "_").replace("-", "_")
    config_name = f"{model_safe_name}_{timestamp}"

    # Push to hub
    repo_id = f"{HF_ORGANIZATION}/{HF_REPO_NAME}"
    print(f"\nUploading to HuggingFace: {repo_id} (config: {config_name})")

    try:
        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            token=hf_token
        )
        print(f"Successfully uploaded to {repo_id}")
        print(f"View at: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        # Save locally as backup
        backup_path = f"iterative_results_{config_name}.csv"
        df.to_csv(backup_path, index=False)
        print(f"Results saved locally to {backup_path}")
        return False


def print_summary(df: pd.DataFrame, session_logs: list):
    """Print experiment summary statistics."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nTotal attempts logged: {len(df)}")
    print(f"  - Prechecks: {len(df[df['response_type'] == 'precheck'])}")
    print(f"  - Final checks: {len(df[df['response_type'] == 'check'])}")

    # Calculate success metrics
    final_checks = df[df['response_type'] == 'check']

    def is_perfect(pass_pattern):
        return all(c == '1' for c in pass_pattern) if pass_pattern else False

    perfect_scores = final_checks[final_checks['pass'].apply(is_perfect)]
    print(f"\nPerfect scores: {len(perfect_scores)}/{len(final_checks)} ({100*len(perfect_scores)/len(final_checks):.1f}%)")

    # Per-question summary
    print("\nPer-question results:")
    for session in session_logs:
        status = "PASS" if is_perfect(session.attempts[-1].pass_pattern) else "FAIL"
        print(f"  {session.question_id}: {session.final_score:.0%} ({session.total_iterations} iterations) [{status}]")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Run iterative code generation experiment with Claude"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=DEFAULT_NUM_QUESTIONS,
        help=f"Number of questions to process (default: {DEFAULT_NUM_QUESTIONS})"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum iterations per question (default: {DEFAULT_MAX_ITERATIONS})"
    )
    parser.add_argument(
        "--num-public-tests",
        type=int,
        default=DEFAULT_NUM_PUBLIC_TESTS,
        help=f"Number of public tests for feedback (default: {DEFAULT_NUM_PUBLIC_TESTS})"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for upload (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local output file path (CSV)"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip HuggingFace upload"
    )

    args = parser.parse_args()

    # Validate API key
    if not args.api_key:
        print("Error: Anthropic API key required.")
        print("Set via --api-key or ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    # HuggingFace login if uploading
    if not args.skip_upload and args.hf_token:
        print("Logging in to HuggingFace...")
        login(token=args.hf_token)

    print("=" * 60)
    print("Iterative LLM Code Generation Experiment")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Questions: {args.num_questions}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Public tests: {args.num_public_tests}")
    print("=" * 60)

    # Load questions
    questions = load_questions(num_questions=args.num_questions)

    if not questions:
        print("Error: No questions loaded. Check HuggingFace dataset access.")
        sys.exit(1)

    # Initialize runner
    runner = IterativeCodeRunner(
        api_key=args.api_key,
        model=args.model,
        max_iterations=args.max_iterations,
        num_public_tests=args.num_public_tests,
    )

    # Run iterations
    print("\nStarting experiment...")
    session_logs = runner.run(questions)

    # Convert to DataFrame
    df = session_logs_to_dataframe(session_logs)

    # Print summary
    print_summary(df, session_logs)

    # Save locally if specified
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    # Upload to HuggingFace
    if not args.skip_upload:
        if not args.hf_token:
            print("\nWarning: No HF_TOKEN provided. Saving locally instead.")
            local_path = f"iterative_results_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(local_path, index=False)
            print(f"Results saved to {local_path}")
        else:
            upload_to_huggingface(df, args.model, args.hf_token)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
