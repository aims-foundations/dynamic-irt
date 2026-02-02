#!/usr/bin/env python3
"""
run_opensource_models.py - Run open-source LLMs for EduCodeSim evaluation using vLLM

This script evaluates open-source models (LLaMA, Gemma, Qwen) on student simulation
scenarios without requiring API keys.

Usage:
    python run_opensource_models.py                          # Run all models
    python run_opensource_models.py --models llama           # Run specific model
    python run_opensource_models.py --scenario S1            # Run specific scenario
    python run_opensource_models.py --output ./results       # Custom output directory
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "llama": {
        "name": "meta_llama-3.1-8b-instruct",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "max_tokens": 2048,
        "temperature": 0.0,
    },
    "gemma": {
        "name": "google_gemma-3-27b-it",
        "hf_id": "google/gemma-3-27b-it",
        "max_tokens": 2048,
        "temperature": 0.0,
    },
    "qwen": {
        "name": "qwen_qwen2.5-14b-instruct",
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "max_tokens": 2048,
        "temperature": 0.0,
    },
}

SCENARIOS = ["S1", "S2", "S3", "S4"]
DATA_URL = "https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv/codeinsights_llm_simulation/resolve/main/"


# ─── Data Classes ──────────────────────────────────────────────────────────────
@dataclass
class ScenarioResult:
    """Result from a single scenario evaluation."""
    question_id: str
    student_id: Optional[str]
    prompt: str
    response: str
    model: str
    scenario: str


# ─── Prompt Generation ─────────────────────────────────────────────────────────
class PromptGenerator:
    """Generate prompts for each scenario."""

    @staticmethod
    def scenario1_prompt(row: pd.Series) -> str:
        """Generate Scenario 1 (Code Correctness) prompt."""
        return (
            f"Question: {row['question_name']} — {row['question_text']}\n\n"
            "Template:\n"
            f"{row['question_template']}\n\n"
            "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template. "
            "– Do NOT reproduce any part of the template "
            "– Do NOT emit `int main()` (it's already declared) "
            "– Ensure your code is correct, efficient, handles all edge cases, and includes any needed class definitions "
            "IMPORTANT: "
            "Your entire response must be exactly one Markdown C++ code-block. "
            "1. The first line of your output must be: ```cpp "
            "2. The last line of your output must be: ``` "
            "3. No extra characters, whitespace, or text may appear before the opening ```cpp or after the closing ```."
        )

    @staticmethod
    def scenario2_prompt(row: pd.Series, examples: List[pd.Series]) -> str:
        """Generate Scenario 2 (Code Performance Imitation) prompt."""
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
    def scenario3_prompt(row: pd.Series, examples: List[pd.Series]) -> str:
        """Generate Scenario 3 (Targeted Error Reproduction) prompt."""
        prompt = (
            "=== Student Profile ===\n"
            "When students submit code to the platform, it will be tested by unit tests, where\n"
            "- Unit test pass rate = proportion of unit tests passed with the code\n"
            "- Full pass rate = proportion of code passing all unit tests\n\n"
            "=== Past Mistake Examples ===\n"
        )

        for n, ex in enumerate(examples, start=1):
            response_col = 'response_mistake' if 'response_mistake' in ex.index else 'response'
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
    def scenario4_prompt(row: pd.Series, examples: List[pd.Series]) -> str:
        """Generate Scenario 4 (Efficiency Alignment) prompt."""
        prompt = (
            f"Week: {row['week']}\n"
            f"Topic: {row['topic']}\n\n"
        )

        for n, ex in enumerate(examples, start=1):
            response_col = 'response_correct' if 'response_correct' in ex.index else 'response'
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


# ─── Data Loading ──────────────────────────────────────────────────────────────
class DataLoader:
    """Load scenario data from local files or HuggingFace."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.cache = {}

    def load_scenario_data(self, scenario: str) -> pd.DataFrame:
        """Load data for a specific scenario."""
        scenario_num = scenario.replace("S", "")
        filename = f"Scenario{scenario_num}_full_data.csv"

        # Try local file first
        local_path = self.data_dir / filename
        if local_path.exists():
            logger.info(f"Loading local data: {local_path}")
            return pd.read_csv(local_path)

        # Fall back to HuggingFace
        url = f"{DATA_URL}data/{filename}"
        logger.info(f"Loading data from HuggingFace: {url}")
        return pd.read_csv(url)

    def prepare_scenario1_data(self) -> List[Dict]:
        """Prepare data for Scenario 1."""
        df = self.load_scenario_data("S1")
        items = []

        for _, row in df.groupby("question_unittest_id").first().iterrows():
            items.append({
                "question_id": str(row.name),
                "student_id": None,
                "prompt": PromptGenerator.scenario1_prompt(row),
                "row": row,
            })

        return items

    def prepare_scenario234_data(self, scenario: str) -> List[Dict]:
        """Prepare data for Scenarios 2, 3, or 4."""
        df = self.load_scenario_data(scenario)
        items = []

        for student_id, student_df in df.groupby("student_id"):
            student_df = student_df.sort_values("timestamp")
            if len(student_df) < 4:
                continue

            attempts = student_df.iloc[:4]

            # Use each attempt as target, others as examples
            for idx in range(min(4, len(attempts))):
                target = attempts.iloc[idx]
                examples = [attempts.iloc[i] for i in range(len(attempts)) if i != idx][:3]

                if scenario == "S2":
                    prompt = PromptGenerator.scenario2_prompt(target, examples)
                elif scenario == "S3":
                    prompt = PromptGenerator.scenario3_prompt(target, examples)
                else:  # S4
                    prompt = PromptGenerator.scenario4_prompt(target, examples)

                items.append({
                    "question_id": str(target.get("question_unittest_id", target.name)),
                    "student_id": str(student_id),
                    "prompt": prompt,
                    "row": target,
                })

        return items


# ─── Model Runner ──────────────────────────────────────────────────────────────
class VLLMRunner:
    """Run inference using vLLM."""

    def __init__(self, model_config: Dict):
        self.config = model_config
        self.model = None
        self.tokenizer = None

    def initialize(self):
        """Initialize vLLM model."""
        try:
            from vllm import LLM, SamplingParams
            logger.info(f"Initializing vLLM with {self.config['hf_id']}...")

            self.model = LLM(
                model=self.config["hf_id"],
                tensor_parallel_size=1,  # Adjust based on GPU count
                trust_remote_code=True,
            )

            self.sampling_params = SamplingParams(
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                stop=["```\n", "\n```"],
            )

            logger.info("vLLM model initialized successfully.")

        except ImportError:
            logger.error("vLLM not installed. Install with: pip install vllm")
            raise

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        if self.model is None:
            self.initialize()

        outputs = self.model.generate(prompts, self.sampling_params)

        responses = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            responses.append(text)

        return responses


class TransformersRunner:
    """Fallback runner using HuggingFace Transformers (slower but more compatible)."""

    def __init__(self, model_config: Dict):
        self.config = model_config
        self.model = None
        self.tokenizer = None

    def initialize(self):
        """Initialize Transformers model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"Initializing Transformers with {self.config['hf_id']}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["hf_id"],
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["hf_id"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Transformers model initialized successfully.")

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for prompts (one at a time for Transformers)."""
        import torch

        if self.model is None:
            self.initialize()

        responses = []
        for prompt in tqdm(prompts, desc="Generating"):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"] or 0.01,
                    do_sample=self.config["temperature"] > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            responses.append(response)

        return responses


# ─── Main Evaluation Pipeline ──────────────────────────────────────────────────
def run_evaluation(
    models: List[str],
    scenarios: List[str],
    output_dir: str,
    use_vllm: bool = True,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
):
    """Run evaluation pipeline."""

    output_path = Path(output_dir)
    data_loader = DataLoader()

    for model_key in models:
        if model_key not in MODEL_CONFIGS:
            logger.warning(f"Unknown model: {model_key}. Skipping.")
            continue

        config = MODEL_CONFIGS[model_key]
        model_name = config["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*60}")

        # Initialize runner
        try:
            if use_vllm:
                runner = VLLMRunner(config)
            else:
                runner = TransformersRunner(config)
            runner.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            continue

        for scenario in scenarios:
            logger.info(f"\nRunning {scenario}...")

            # Prepare data
            if scenario == "S1":
                items = data_loader.prepare_scenario1_data()
            else:
                items = data_loader.prepare_scenario234_data(scenario)

            if max_samples:
                items = items[:max_samples]

            logger.info(f"Processing {len(items)} samples...")

            # Extract prompts
            prompts = [item["prompt"] for item in items]

            # Generate responses in batches
            all_responses = []
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"{scenario}"):
                batch = prompts[i:i+batch_size]
                responses = runner.generate(batch)
                all_responses.extend(responses)

            # Build results
            results = []
            for item, response in zip(items, all_responses):
                results.append({
                    "question_id": item["question_id"],
                    "student_id": item["student_id"],
                    "text": response,
                    "model": model_name,
                })

            # Save results
            scenario_num = scenario.replace("S", "")
            model_short = "_".join(model_name.split("_")[1:])  # Remove prefix

            result_dir = output_path / "scenario_results" / model_short
            result_dir.mkdir(parents=True, exist_ok=True)

            result_file = result_dir / f"{model_short}_scenario{scenario_num}.csv"
            df = pd.DataFrame(results)
            df.to_csv(result_file, index=False)
            logger.info(f"Saved results to {result_file}")

            # Also save as JSON for compatibility
            json_dir = output_path / "opensource_llm_output" / model_name / scenario
            json_dir.mkdir(parents=True, exist_ok=True)

            json_file = json_dir / "scenario_state.json"
            with open(json_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved JSON to {json_file}")

        # Clean up model to free memory
        del runner
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Run open-source LLMs for EduCodeSim evaluation"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="llama,gemma,qwen",
        help="Comma-separated list of models to run (llama, gemma, qwen)"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="S1,S2,S3,S4",
        help="Comma-separated list of scenarios (S1, S2, S3, S4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./",
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Use Transformers instead of vLLM (slower but more compatible)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per scenario (for testing)"
    )

    args = parser.parse_args()

    models = [m.strip().lower() for m in args.models.split(",")]
    scenarios = [s.strip().upper() for s in args.scenarios.split(",")]

    logger.info("Starting EduCodeSim open-source model evaluation")
    logger.info(f"Models: {models}")
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Using vLLM: {not args.no_vllm}")

    run_evaluation(
        models=models,
        scenarios=scenarios,
        output_dir=args.output,
        use_vllm=not args.no_vllm,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
