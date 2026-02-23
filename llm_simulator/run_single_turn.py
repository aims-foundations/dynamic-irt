#!/usr/bin/env python3
"""
run_single_turn.py - Single-turn LLM evaluation across all 4 scenarios.

The LLM sees the prompt once and produces one response — no feedback loop.
This is the correct mode for S2 (style imitation), S3 (error replication),
and S4 (efficiency alignment). For S1 with iterative public-test feedback,
use run_iterative_model.py instead.

Supports both commercial API models (Claude, GPT, Gemini, Mistral) and
open-source models served via vLLM (LLaMA, Gemma, Qwen).

Usage:
    python run_single_turn.py --models claude gpt --scenarios S2 S3 S4
    python run_single_turn.py --models llama qwen --output ./results
    python run_single_turn.py --models claude --max-samples 10    # quick test
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    # Commercial API models
    "claude": {
        "name": "claude_claude-sonnet-4",
        "api_model": "claude-sonnet-4-20250514",
        "backend": "anthropic",
        "max_workers": 2,
        "delay": 1.0,
    },
    "gpt": {
        "name": "openai_gpt-4.1-nano",
        "api_model": "gpt-4.1-nano",
        "backend": "openai",
        "max_workers": 2,
        "delay": 0.8,
    },
    "gemini": {
        "name": "google_gemini-2.0-flash",
        "api_model": "gemini-2.0-flash",
        "backend": "google",
        "max_workers": 3,
        "delay": 0.5,
    },
    "mistral": {
        "name": "mistral_mistral-large",
        "api_model": "mistral-large-latest",
        "backend": "mistral",
        "max_workers": 3,
        "delay": 0.5,
    },
    # Open-source models (vLLM)
    "llama": {
        "name": "meta_llama-3.1-8b-instruct",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "vllm",
        "max_tokens": 2048,
        "temperature": 0.0,
    },
    "gemma": {
        "name": "google_gemma-3-27b-it",
        "hf_id": "google/gemma-3-27b-it",
        "backend": "vllm",
        "max_tokens": 2048,
        "temperature": 0.0,
    },
    "qwen": {
        "name": "qwen_qwen2.5-14b-instruct",
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "backend": "vllm",
        "max_tokens": 2048,
        "temperature": 0.0,
    },
}

SCENARIOS = ["S1", "S2", "S3", "S4"]
DATA_URL = "https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv/codeinsights_llm_simulation/resolve/main/"


# ── Prompt generation ──────────────────────────────────────────────────────────

class PromptGenerator:
    """Generate prompts for each evaluation scenario."""

    @staticmethod
    def scenario1(row: pd.Series) -> str:
        return (
            f"Question: {row['question_name']} — {row['question_text']}\n\n"
            "Template:\n"
            f"{row['question_template']}\n\n"
            "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template. "
            "– Do NOT reproduce any part of the template "
            "– Do NOT emit `int main()` (it's already declared) "
            "– Ensure your code is correct, efficient, handles all edge cases, and includes any needed class definitions "
            "IMPORTANT: Your entire response must be exactly one Markdown C++ code-block. "
            "1. The first line of your output must be: ```cpp "
            "2. The last line of your output must be: ``` "
            "3. No extra characters, whitespace, or text may appear before the opening ```cpp or after the closing ```."
        )

    @staticmethod
    def scenario2(row: pd.Series, examples: List[pd.Series]) -> str:
        prompt = (
            "=== Student Profile ===\n"
            f"Week: {row['week']}\n"
            f"Topic: {row['topic']}\n\n"
        )
        for n, ex in enumerate(examples, start=1):
            prompt += (
                f"Example {n}:\n"
                f"Question: {ex['question_name']} — {ex['question_text']}\n"
                f"Template:\n{ex['question_template']}\n"
                f"Your Code:\n{ex['response']}\n\n"
            )
        prompt += (
            "Now, using that same student style, attempt this:\n"
            f"Question: {row['question_name']} — {row['question_text']}\n"
            f"Template:\n{row['question_template']}\n\n"
            "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template. "
            "– Do NOT reproduce any part of the template "
            "– Do NOT emit `int main()` (it's already declared) "
            "– Ensure your code mirrors the style of the previous examples and includes any necessary class definitions "
            "IMPORTANT: your entire response must be exactly one Markdown C++ code-block:\n"
            "1. First line: ```cpp\n"
            "2. Last line: ```\n"
        )
        return prompt

    @staticmethod
    def scenario3(row: pd.Series, examples: List[pd.Series]) -> str:
        prompt = (
            "=== Student Profile ===\n"
            "When students submit code to the platform, it will be tested by unit tests, where\n"
            "- Unit test pass rate = proportion of unit tests passed with the code\n"
            "- Full pass rate = proportion of code passing all unit tests\n\n"
            "=== Past Mistake Examples ===\n"
        )
        for n, ex in enumerate(examples, start=1):
            response_col = "response_mistake" if "response_mistake" in ex.index else "response"
            prompt += (
                f"Example {n} (Week {ex['week']}, Topic: {ex['topic']}):\n"
                f"Question: {ex['question_name']} — {ex['question_text']}\n"
                "Template:\n"
                f"{ex['question_template']}\n"
                "Student's Response Code with Error:\n"
                f"{ex[response_col]}\n\n"
            )
        prompt += (
            "=== New Target Problem ===\n"
            f"Week: {row['week']}, Topic: {row['topic']}\n"
            f"Question: {row['question_name']} — {row['question_text']}\n"
            "Template:\n"
            f"{row['question_template']}\n\n"
            "**Instructions:**\n"
            "1. Mimic your own coding style, naming conventions, indentation, and typical error patterns from the examples.\n"
            "2. Introduce a mistake you are likely to make (e.g., off-by-one index, wrong initialization, missing edge case).\n"
            "3. Do **not** produce a fully correct solution or add unfamiliar optimizations.\n\n"
            "4. Include any needed class definitions, and make sure the code is compatible with the Unit Test Input.\n"
            "5. Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template.\n"
            "6. Do NOT reproduce any part of the template.\n"
            "7. Do NOT emit `int main()` (it's already declared).\n\n"
            "IMPORTANT: your entire response must be exactly one Markdown C++ code-block:\n"
            "1. First line: ```cpp\n"
            "2. Last line: ```\n"
            "No extra characters, whitespace, or text before/after.\n"
        )
        return prompt

    @staticmethod
    def scenario4(row: pd.Series, examples: List[pd.Series]) -> str:
        prompt = (
            f"Week: {row['week']}\n"
            f"Topic: {row['topic']}\n\n"
        )
        for n, ex in enumerate(examples, start=1):
            response_col = "response_correct" if "response_correct" in ex.index else "response"
            prompt += (
                f"Example {n}:\n"
                f"Question: {ex['question_name']} — {ex['question_text']}\n"
                "Template:\n"
                f"{ex['question_template']}\n"
                "Your Code:\n"
                f"{ex[response_col]}\n\n"
            )
        prompt += (
            "Now, using that same student's coding style, attempt this:\n"
            f"Question: {row['question_name']} — {row['question_text']}\n\n"
            "Template:\n"
            f"{row['question_template']}\n\n"
            "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template. "
            "– Do NOT reproduce any part of the template "
            "– Do NOT emit `int main()` (it's already declared) "
            "– Ensure your code is correct, handles all edge cases, and includes any needed class definitions "
            "– Match the student's usual efficiency style.\n\n"
            "IMPORTANT: your entire response must be exactly one Markdown C++ code-block:\n"
            "1. First line: ```cpp\n"
            "2. Last line: ```\n"
            "No extra whitespace or text before/after.\n"
        )
        return prompt


# ── Data loading ───────────────────────────────────────────────────────────────

class DataLoader:
    """Load and prepare scenario data from local files or HuggingFace."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)

    def _load_csv(self, filename: str) -> pd.DataFrame:
        local = self.data_dir / filename
        if local.exists():
            logger.info("Loading local data: %s", local)
            return pd.read_csv(local)
        url = f"{DATA_URL}data/{filename}"
        logger.info("Loading from HuggingFace: %s", url)
        return pd.read_csv(url)

    def prepare(self, scenario: str) -> List[Dict]:
        num = scenario.replace("S", "")
        df = self._load_csv(f"Scenario{num}_full_data.csv")
        if scenario == "S1":
            return self._prepare_s1(df)
        return self._prepare_s234(df, scenario)

    def _prepare_s1(self, df: pd.DataFrame) -> List[Dict]:
        items = []
        for _, row in df.groupby("question_unittest_id").first().reset_index().iterrows():
            items.append({
                "question_id": str(row["question_unittest_id"]),
                "student_id": None,
                "prompt": PromptGenerator.scenario1(row),
            })
        return items

    def _prepare_s234(self, df: pd.DataFrame, scenario: str) -> List[Dict]:
        prompt_fn = {
            "S2": PromptGenerator.scenario2,
            "S3": PromptGenerator.scenario3,
            "S4": PromptGenerator.scenario4,
        }[scenario]
        items = []
        for student_id, student_df in df.groupby("student_id"):
            student_df = student_df.sort_values("timestamp")
            if len(student_df) < 4:
                continue
            attempts = student_df.iloc[:4]
            for idx in range(len(attempts)):
                target = attempts.iloc[idx]
                examples = [attempts.iloc[i] for i in range(len(attempts)) if i != idx][:3]
                items.append({
                    "question_id": str(target.get("question_unittest_id", target.name)),
                    "student_id": str(student_id),
                    "prompt": prompt_fn(target, examples),
                })
        return items


# ── Model runners ──────────────────────────────────────────────────────────────

class _APIRunner:
    """Base class for commercial API runners with parallel execution."""

    def __init__(self, max_workers: int = 2, delay: float = 1.0):
        self.max_workers = max_workers
        self.delay = delay

    def _call(self, prompt: str) -> str:
        raise NotImplementedError

    def generate(self, prompts: List[str]) -> List[str]:
        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._call, p): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    logger.info("✓ %d/%d", idx + 1, len(prompts))
                except Exception as e:
                    logger.error("✗ prompt %d: %s", idx + 1, e)
                    results[idx] = f"ERROR: {e}"
                time.sleep(self.delay / self.max_workers)
        return results


class ClaudeRunner(_APIRunner):
    def __init__(self, api_model: str, **kwargs):
        super().__init__(**kwargs)
        import anthropic as _anthropic
        self.client = _anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.api_model = api_model

    def _call(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.api_model, max_tokens=4000,
            stop_sequences=["\n```"],
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class GeminiRunner(_APIRunner):
    def __init__(self, api_model: str, **kwargs):
        super().__init__(**kwargs)
        import google.generativeai as _genai
        _genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = _genai.GenerativeModel(api_model)
        self._genai = _genai

    def _call(self, prompt: str) -> str:
        resp = self.model.generate_content(
            contents=[prompt],
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=4000,
                stop_sequences=["\n```"],
            ),
        )
        return resp.text


class OpenAIRunner(_APIRunner):
    def __init__(self, api_model: str, **kwargs):
        super().__init__(**kwargs)
        import openai as _openai
        self.client = _openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.api_model = api_model

    def _call(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000, stop=["\n```"],
        )
        return resp.choices[0].message.content


class MistralRunner(_APIRunner):
    def __init__(self, api_model: str, **kwargs):
        super().__init__(**kwargs)
        from mistralai import Mistral as _Mistral
        self.client = _Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.api_model = api_model

    def _call(self, prompt: str) -> str:
        resp = self.client.chat.complete(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        text = resp.choices[0].message.content
        idx = text.find("\n```")
        return text[:idx] if idx != -1 else text


class VLLMRunner:
    """Open-source model runner using vLLM for batch inference."""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.sampling_params = None

    def initialize(self):
        from vllm import LLM, SamplingParams
        logger.info("Initializing vLLM: %s", self.config["hf_id"])
        self.model = LLM(model=self.config["hf_id"], tensor_parallel_size=1, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            stop=["```\n", "\n```"],
        )

    def generate(self, prompts: List[str]) -> List[str]:
        if self.model is None:
            self.initialize()
        outputs = self.model.generate(prompts, self.sampling_params)
        return [o.outputs[0].text if o.outputs else "" for o in outputs]


_API_RUNNER_MAP = {
    "anthropic": ClaudeRunner,
    "openai":    OpenAIRunner,
    "google":    GeminiRunner,
    "mistral":   MistralRunner,
}


def create_runner(config: Dict):
    """Return the appropriate runner for a model config."""
    backend = config["backend"]
    if backend == "vllm":
        return VLLMRunner(config)
    cls = _API_RUNNER_MAP[backend]
    return cls(
        api_model=config["api_model"],
        max_workers=config.get("max_workers", 2),
        delay=config.get("delay", 1.0),
    )


# ── Evaluation pipeline ────────────────────────────────────────────────────────

def run_evaluation(
    models: List[str],
    scenarios: List[str],
    output_dir: str,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
):
    output_path = Path(output_dir)
    data_loader = DataLoader()

    for model_key in models:
        if model_key not in MODEL_CONFIGS:
            logger.warning("Unknown model '%s'. Skipping.", model_key)
            continue

        config = MODEL_CONFIGS[model_key]
        model_name = config["name"]
        logger.info("\n%s\nEvaluating: %s\n%s", "=" * 60, model_name, "=" * 60)

        runner = create_runner(config)
        # Eagerly initialize vLLM so weights load once for all scenarios
        if isinstance(runner, VLLMRunner):
            runner.initialize()

        for scenario in scenarios:
            logger.info("Scenario %s ...", scenario)
            items = data_loader.prepare(scenario)
            if max_samples:
                items = items[:max_samples]
            logger.info("%d samples", len(items))

            prompts = [item["prompt"] for item in items]

            # Generate in batches (vLLM benefits from batching;
            # API runners handle their own concurrency internally)
            all_responses: List[str] = []
            for i in tqdm(range(0, len(prompts), batch_size), desc=scenario):
                all_responses.extend(runner.generate(prompts[i: i + batch_size]))

            results = [
                {
                    "question_id": item["question_id"],
                    "student_id": item["student_id"],
                    "text": resp,
                    "model": model_name,
                }
                for item, resp in zip(items, all_responses)
            ]

            # Save CSV
            scenario_num = scenario.replace("S", "")
            model_short = "_".join(model_name.split("_")[1:])
            result_dir = output_path / "scenario_results" / model_short
            result_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(
                result_dir / f"{model_short}_scenario{scenario_num}.csv", index=False
            )

            # Save JSON (for compute_metrics.py compatibility)
            json_dir = output_path / "llm_output" / model_name / scenario
            json_dir.mkdir(parents=True, exist_ok=True)
            (json_dir / "scenario_state.json").write_text(json.dumps(results, indent=2))

            logger.info("Saved %d results for %s / %s", len(results), model_name, scenario)

        # Free GPU memory between models
        if isinstance(runner, VLLMRunner):
            del runner
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Single-shot LLM evaluation across all 4 scenarios"
    )
    parser.add_argument(
        "--models", type=str, default=",".join(MODEL_CONFIGS.keys()),
        help=f"Comma-separated models: {', '.join(MODEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--scenarios", type=str, default="S1,S2,S3,S4",
        help="Comma-separated scenarios (S1, S2, S3, S4)",
    )
    parser.add_argument(
        "--output", type=str, default="./",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for vLLM inference",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap on samples per scenario (for quick testing)",
    )
    args = parser.parse_args()

    models = [m.strip().lower() for m in args.models.split(",")]
    scenarios = [s.strip().upper() for s in args.scenarios.split(",")]

    logger.info("Models:    %s", models)
    logger.info("Scenarios: %s", scenarios)
    logger.info("Output:    %s", args.output)

    run_evaluation(
        models=models,
        scenarios=scenarios,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
