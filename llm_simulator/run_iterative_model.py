"""run_iterative_model.py

Iterative LLM code generation for S1 (correct code) scenario.

The LLM receives feedback from failed **public** tests after each attempt,
mirroring how real students experience the online judge's "Precheck" feature.

Public-test discovery
---------------------
Rather than a hard-coded N, we infer N_public per question from the raw
main_data.csv: "Prechecked" submissions only run on public tests, so the
length of their `pass` string = N_public for that question.

Output format
-------------
The output CSV matches the main_data.csv schema:
    model_id, question_unittest_id, attempt_id, timestamp, response_type,
    response, pass

where:
    response_type = "Prechecked"  — intermediate iterations (public tests only)
    response_type = "Submit"      — final evaluation on all tests

Usage
-----
    python run_iterative_model.py \\
        --scenario_dir data/ \\
        --models claude-sonnet-4 gpt-4.1-nano \\
        --max_attempts 100 \\
        --output_dir data/iterative_results/
"""

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import anthropic
import google.generativeai as genai
import openai
import pandas as pd
from huggingface_hub import snapshot_download
from mistralai import Mistral

from grading_engine import CPPEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_REPO_ID = "stair-lab/code_insights_csv"

# ── LLM runners ──────────────────────────────────────────────────────────────
# Minimal self-contained runners (no module-level env-var side effects).


class _BaseLLMRunner:
    def call(self, prompt: str) -> str:
        raise NotImplementedError


class ClaudeRunner(_BaseLLMRunner):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def call(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            stop_sequences=["\n```"],
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class GeminiRunner(_BaseLLMRunner):
    def __init__(self, model: str = "gemini-2.0-flash"):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model)

    def call(self, prompt: str) -> str:
        resp = self.model.generate_content(
            contents=[prompt],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=4000,
                stop_sequences=["\n```"],
            ),
        )
        return resp.text


class OpenAIRunner(_BaseLLMRunner):
    def __init__(self, model: str = "gpt-4.1-nano"):
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def call(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            stop=["\n```"],
        )
        return resp.choices[0].message.content


class MistralRunner(_BaseLLMRunner):
    def __init__(self, model: str = "mistral-large-latest"):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.model = model

    def call(self, prompt: str) -> str:
        resp = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        text = resp.choices[0].message.content
        # Manual trim for stop sequence (Mistral doesn't always honour it)
        idx = text.find("\n```")
        if idx != -1:
            text = text[:idx]
        return text


class VLLMRunner(_BaseLLMRunner):
    """Open-source model runner using a local vLLM engine.

    The engine is initialised once at construction time (weights are loaded
    into GPU memory), so subsequent ``call()`` invocations are fast.

    Args:
        model_hf_id:          HuggingFace model ID or local path.
        max_tokens:           Maximum tokens to generate per call.
        temperature:          Sampling temperature (0.0 = greedy).
        tensor_parallel_size: Number of GPUs to shard across.
    """

    def __init__(
        self,
        model_hf_id: str,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        tensor_parallel_size: int = 4,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vllm is required for open-source model inference. "
                "Install with: pip install vllm"
            ) from exc

        self.llm = LLM(
            model=model_hf_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
            trust_remote_code=True,   # required for GLM and some other models
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n```"],
        )

    def call(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text


MODEL_REGISTRY: Dict[str, callable] = {
    "claude-sonnet-4":  lambda: ClaudeRunner("claude-sonnet-4-20250514"),
    "gpt-4.1-nano":     lambda: OpenAIRunner("gpt-4.1-nano"),
    "gemini-2.0-flash": lambda: GeminiRunner("gemini-2.0-flash"),
    "mistral-large":    lambda: MistralRunner("mistral-large-latest"),
    # Open-source models (served via vLLM on local GPUs)
    "glm":              lambda: VLLMRunner("QuantTrio/GLM-4.7-AWQ", tensor_parallel_size=4),
}

# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class AttemptRecord:
    attempt_id: int
    timestamp: str
    response_type: str   # "Prechecked" or "Submit"
    response: str        # raw LLM output
    pass_pattern: str    # binary string, e.g. "10110"


@dataclass
class IterativeSession:
    model_id: str
    question_unittest_id: str
    attempts: List[AttemptRecord] = field(default_factory=list)

    def to_rows(self) -> List[dict]:
        return [
            {
                "model_id": self.model_id,
                "question_unittest_id": self.question_unittest_id,
                "attempt_id": rec.attempt_id,
                "timestamp": rec.timestamp,
                "response_type": rec.response_type,
                "response": rec.response,
                "pass": rec.pass_pattern,
            }
            for rec in self.attempts
        ]


# ── Public test inference ─────────────────────────────────────────────────────


def infer_public_test_counts(main_df: pd.DataFrame) -> Dict[str, int]:
    """Infer N_public per question from "Prechecked" submission pass-string lengths.

    Students on the online judge only run public tests when they click "Precheck",
    so len(pass) for those rows = N_public for that question.  We take the mode
    (most common length) per question as the canonical N_public.

    Returns a dict {question_unittest_id (str) -> n_public (int)}.
    Questions with no Prechecked submissions are absent from the dict.
    """
    precheck = main_df[main_df["response_type"] == "Prechecked"].copy()
    precheck["pass_len"] = precheck["pass"].astype(str).str.len()
    n_public = (
        precheck.groupby("question_unittest_id")["pass_len"]
        .agg(lambda x: int(x.mode().iloc[0]))
        .to_dict()
    )
    return {str(k): v for k, v in n_public.items()}


# ── Test-case parsing ─────────────────────────────────────────────────────────


def parse_test_cases(unittests_str: str) -> List[dict]:
    """Parse the `question_unittests` field into a list of test case dicts."""
    test_cases = []
    for block in unittests_str.split("Unittest")[1:]:
        body = block[block.find(":") + 1:]
        i_input  = body.find("Input:")
        i_stdin  = body.find("STD input:")
        i_output = body.find("Output:")
        if -1 in (i_input, i_stdin, i_output):
            return []  # malformed — caller should skip this question
        test_cases.append({
            "input":  body[i_input + 6  : i_stdin].strip(),
            "std_in": body[i_stdin + 10 : i_output].strip(),
            "output": body[i_output + 7 :].strip(),
        })
    return test_cases


# ── Code extraction ───────────────────────────────────────────────────────────

_CPP_FENCE_RE = re.compile(r"```(?:cpp|c\+\+)?\n(.*?)(?:\n```|$)", re.DOTALL)


def extract_code(llm_output: str) -> Optional[str]:
    """Return the first C++ code block from markdown fences, or None."""
    m = _CPP_FENCE_RE.search(llm_output)
    return m.group(1).strip() if m else None


# ── Prompts ───────────────────────────────────────────────────────────────────

_BASE_INSTRUCTIONS = (
    "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} "
    "block in the template. "
    "– Do NOT reproduce any part of the template. "
    "– Do NOT emit `int main()` (it's already declared). "
    "– Ensure your code is correct, efficient, handles all edge cases, and includes "
    "any needed class definitions.\n\n"
    "IMPORTANT: Your entire response must be exactly one Markdown C++ code-block.\n"
    "1. The first line of your output must be: ```cpp\n"
    "2. The last line of your output must be: ```\n"
    "3. No extra characters, whitespace, or text may appear before or after the block."
)


def build_initial_prompt(question_name: str, question_text: str, template: str) -> str:
    return (
        f"Question: {question_name} — {question_text}\n\n"
        f"Template:\n{template}\n\n"
        f"{_BASE_INSTRUCTIONS}"
    )


def build_feedback_prompt(
    initial_prompt: str,
    previous_code: str,
    public_test_cases: List[dict],
    public_results: List[int],
    public_outputs: List[str],
) -> str:
    """Construct a retry prompt that includes which public tests failed and why."""
    failed = [i for i, r in enumerate(public_results) if r == 0]
    lines = ["Your previous submission failed the following visible test cases:\n"]
    for i in failed:
        tc = public_test_cases[i]
        actual = (
            public_outputs[i].strip()
            if public_outputs[i]
            else "(no output — likely a compile or runtime error)"
        )
        lines.append(
            f"Test {i + 1}:\n"
            f"  Test input:  {tc['input']}\n"
            f"  STD input:   {tc['std_in']}\n"
            f"  Expected:    {tc['output']}\n"
            f"  Got:         {actual}\n"
        )
    return (
        f"{initial_prompt}\n\n"
        f"=== Previous Attempt ===\n"
        f"```cpp\n{previous_code}\n```\n\n"
        f"=== Feedback from Visible Tests ===\n"
        + "\n".join(lines)
        + "\nPlease fix the issues and provide a corrected solution in the same format."
    )


# ── Core iterative runner ─────────────────────────────────────────────────────


class IterativeRunner:
    """Runs one LLM through multiple attempts on a single coding question,
    providing public-test feedback after each failed iteration."""

    def __init__(self, llm_runner: _BaseLLMRunner, model_id: str, max_attempts: int = 100):
        self.llm_runner = llm_runner
        self.model_id = model_id
        self.max_attempts = max_attempts

    def run_session(
        self,
        question_unittest_id: str,
        question_template: str,
        all_test_cases: List[dict],
        n_public: int,
        initial_prompt: str,
    ) -> IterativeSession:
        """Run iterative attempts for a single question.

        Args:
            question_unittest_id: Unique question ID (used in output rows).
            question_template:    Jinja2 C++ template with {{ STUDENT_ANSWER }}.
            all_test_cases:       Full list of test cases (public + private).
            n_public:             How many leading test cases are "public" (shown
                                  as feedback during iteration).
            initial_prompt:       The S1 prompt for this question.

        Returns:
            IterativeSession with one "Prechecked" AttemptRecord per iteration
            plus a final "Submit" record evaluated against all tests.
        """
        session = IterativeSession(
            model_id=self.model_id,
            question_unittest_id=str(question_unittest_id),
        )

        public_tests = all_test_cases[:n_public]
        evaluator_public = CPPEvaluator(question_template, public_tests)
        evaluator_all    = CPPEvaluator(question_template, all_test_cases)

        current_prompt = initial_prompt
        last_code: Optional[str] = None

        for attempt_id in range(self.max_attempts):
            # ── LLM call ────────────────────────────────────────────────────
            raw_output = self.llm_runner.call(current_prompt)
            timestamp  = time.strftime("%Y-%m-%dT%H:%M:%S")
            code       = extract_code(raw_output)

            if code is None:
                # Could not parse a code block — record as all-fail and retry
                session.attempts.append(AttemptRecord(
                    attempt_id=attempt_id,
                    timestamp=timestamp,
                    response_type="Prechecked",
                    response=raw_output,
                    pass_pattern="0" * len(public_tests),
                ))
                current_prompt = (
                    f"{initial_prompt}\n\n"
                    "Your response did not contain a valid ```cpp ... ``` code block. "
                    "Please try again."
                )
                continue

            last_code = code

            # ── Evaluate against public tests ────────────────────────────
            pub = evaluator_public.evaluate_with_outputs(code)
            pass_pattern = "".join(str(x) for x in pub["testcases"])

            session.attempts.append(AttemptRecord(
                attempt_id=attempt_id,
                timestamp=timestamp,
                response_type="Prechecked",
                response=raw_output,
                pass_pattern=pass_pattern,
            ))

            logger.info(
                "  [%s] Q%s attempt %d/%d: public %.0f%% (%s)",
                self.model_id, question_unittest_id,
                attempt_id + 1, self.max_attempts,
                pub["score"] * 100, pass_pattern,
            )

            # Stop early when all public tests pass
            if pub["score"] == 1.0:
                break

            # ── Build feedback prompt for next iteration ─────────────────
            current_prompt = build_feedback_prompt(
                initial_prompt,
                code,
                public_tests,
                pub["testcases"],
                pub["outputs"],
            )

        # ── Final evaluation on ALL tests ────────────────────────────────
        if last_code:
            final = evaluator_all.evaluate_with_outputs(last_code)
        else:
            final = {
                "score": 0.0,
                "testcases": [0] * len(all_test_cases),
                "outputs": [""] * len(all_test_cases),
            }

        final_pass = "".join(str(x) for x in final["testcases"])
        session.attempts.append(AttemptRecord(
            attempt_id=len(session.attempts),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            response_type="Submit",
            response=last_code or "",
            pass_pattern=final_pass,
        ))

        logger.info(
            "  [%s] Q%s SUBMIT: %.0f%% (%s) after %d attempt(s)",
            self.model_id, question_unittest_id,
            final["score"] * 100, final_pass,
            len(session.attempts) - 1,  # exclude the Submit record
        )
        return session


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Iterative S1 (correct code) LLM evaluation"
    )
    parser.add_argument(
        "--scenario_dir", default="data",
        help="Directory containing Scenario1_full_data.csv (default: data/)",
    )
    parser.add_argument(
        "--models", nargs="*", choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Models to run (default: all registered models)",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=100,
        help="Max iterations per question before giving up (default: 100)",
    )
    parser.add_argument(
        "--output_dir", default="data/iterative_results",
        help="Directory for output CSVs (default: data/iterative_results/)",
    )
    args = parser.parse_args()

    # ── Load scenario data ───────────────────────────────────────────────────
    scenario1_path = os.path.join(args.scenario_dir, "Scenario1_full_data.csv")
    logger.info("Loading Scenario 1 data from %s", scenario1_path)
    scenario1_df = pd.read_csv(scenario1_path, dtype={"pass": str})

    # ── Load raw data to infer public test counts ────────────────────────────
    logger.info("Downloading raw data from HuggingFace (%s) for public-test inference…", HF_REPO_ID)
    hf_dir = snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset")
    main_df = pd.read_csv(os.path.join(hf_dir, "main_data.csv"), dtype={"pass": str})

    n_public_map = infer_public_test_counts(main_df)
    logger.info("Inferred public-test counts for %d questions.", len(n_public_map))

    # ── Build question list ──────────────────────────────────────────────────
    questions = []
    for question_id, group in scenario1_df.groupby("question_unittest_id"):
        row = group.iloc[0]
        qid = str(question_id)

        test_cases = parse_test_cases(row["question_unittests"])
        if not test_cases:
            logger.warning("Q%s: failed to parse test cases, skipping", qid)
            continue

        # Fall back to all-public if the question has no Prechecked data
        n_public = n_public_map.get(qid, len(test_cases))
        n_public = max(1, min(n_public, len(test_cases)))  # clamp to valid range

        prompt = build_initial_prompt(
            row["question_name"],
            row["question_text"],
            row["question_template"],
        )
        questions.append((qid, row["question_template"], test_cases, n_public, prompt))

    logger.info("Running iterative S1 on %d questions.", len(questions))

    # ── Run each model ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    for model_id in args.models:
        try:
            runner = MODEL_REGISTRY[model_id]()
        except Exception as e:
            logger.error("Could not initialise runner for %s: %s", model_id, e)
            continue

        iter_runner = IterativeRunner(runner, model_id=model_id, max_attempts=args.max_attempts)
        all_rows: List[dict] = []

        for i, (qid, template, test_cases, n_public, prompt) in enumerate(questions):
            logger.info(
                "[%s] Q%s (%d/%d) — %d public / %d total tests",
                model_id, qid, i + 1, len(questions), n_public, len(test_cases),
            )
            try:
                session = iter_runner.run_session(
                    question_unittest_id=qid,
                    question_template=template,
                    all_test_cases=test_cases,
                    n_public=n_public,
                    initial_prompt=prompt,
                )
                all_rows.extend(session.to_rows())
            except Exception as e:
                logger.error("Q%s failed: %s", qid, e)

        out_path = os.path.join(args.output_dir, f"{model_id}_s1_iterative.csv")
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        logger.info("Saved %d rows → %s", len(all_rows), out_path)


if __name__ == "__main__":
    main()
