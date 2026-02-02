#!/usr/bin/env python3
"""
Run CodeInsights evaluation using vLLM OpenAI-compatible API.
This script evaluates open-source models on the CodeInsights scenarios.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add HELM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "helm" / "src"))

import pandas as pd
from openai import OpenAI

# Import CodeInsights scenarios
from helm.benchmark.scenarios.codeinsights_correct_code_scenario import CodeInsightsCorrectCodeScenario
from helm.benchmark.scenarios.codeinsights_student_coding_scenario import CodeInsightsStudentCodingScenario
from helm.benchmark.scenarios.codeinsights_student_mistake_scenario import CodeInsightsStudentMistakeScenario
from helm.benchmark.scenarios.codeinsights_code_efficiency_scenario import CodeInsightsCodeEfficiencyScenario

# Model configurations
MODELS = {
    "llama": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "base_url": "http://localhost:8001/v1",
    },
    "gemma": {
        "name": "google/gemma-2-27b-it",
        "base_url": "http://localhost:8002/v1",
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "base_url": "http://localhost:8003/v1",
    },
}

SCENARIOS = {
    "S1": {
        "name": "Correct Code (S1)",
        "class": CodeInsightsCorrectCodeScenario,
        "instruction": (
            "You are a skilled C++ programmer working on a foundational programming course assignment. "
            "Your task is to write correct, efficient C++ code that solves the given problem. "
            "Write clean, well-structured code that follows good programming practices. "
            "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template. "
            "DO NOT reproduce the template part as the generated code would be inserted to the template, "
            "and make sure the code is compatible with the Unit Test Input. "
            "Ensure your code is correct, efficient, includes any class definition when needed, and handles all edge cases properly."
        ),
    },
    "S2": {
        "name": "Student Coding (S2)",
        "class": CodeInsightsStudentCodingScenario,
        "instruction": (
            "You are the same student who wrote the three examples below in your foundational C++ course. "
            "Mimic exactly your personal coding style, conventions, and level of proficiency—"
            "do not over-optimize or introduce unfamiliar patterns. "
            "Include the same sort of formatting, variable names, and minor imperfections you demonstrated. "
            "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template. "
            "DO NOT reproduce the template part as the generated code would be inserted to the template, "
            "and make sure the code is compatible with the Unit Test Input. "
            "Ensure your code includes any class definition when needed."
        ),
    },
    "S3": {
        "name": "Student Mistake (S3)",
        "class": CodeInsightsStudentMistakeScenario,
        "instruction": (
            "You are a C++ student with a consistent personal style, conventions, and proficiency level.\n"
            "Your task is to attempt the target problem **but introduce realistic mistake** you would typically make—"
            "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template. "
            "DO NOT reproduce the template part as the generated code would be inserted to the template, "
            "and make sure the code is compatible with the Unit Test Input. "
            "Ensure your code includes any class definition when needed."
        ),
    },
    "S4": {
        "name": "Code Efficiency (S4)",
        "class": CodeInsightsCodeEfficiencyScenario,
        "instruction": (
            "You are the same student who wrote the three examples below in your foundational C++ course. "
            "Mimic exactly your personal coding style, conventions, and make sure to generate a correct code. "
            "Do not over-optimize or introduce unfamiliar patterns. If the code is correct but inefficient, "
            "imitate the inefficiency. "
            "If the student writes efficiently, write efficiently too. "
            "Include the same sort of formatting, variable names, and minor imperfections you demonstrated. "
            "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template. "
            "DO NOT reproduce the template part as the generated code would be inserted to the template, "
            "and make sure the code is compatible with the Unit Test Input. "
            "Ensure your code is correct, includes any class definition when needed, and handles all edge cases properly."
        ),
    },
}


def create_client(model_config):
    """Create OpenAI client for vLLM server."""
    return OpenAI(
        api_key="EMPTY",
        base_url=model_config["base_url"],
    )


def generate_completion(client, model_name, prompt, instruction, max_tokens=4000, temperature=0.0):
    """Generate completion using vLLM server."""
    try:
        # Gemma doesn't support system role, so we prepend instruction to user message
        if "gemma" in model_name.lower():
            messages = [
                {"role": "user", "content": f"{instruction}\n\n{prompt}"},
            ]
        else:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating completion: {e}")
        return None


def run_scenario(model_key, scenario_key, output_dir, max_instances=None):
    """Run evaluation for a specific model and scenario."""
    model_config = MODELS[model_key]
    scenario_config = SCENARIOS[scenario_key]

    print(f"\n{'='*60}")
    print(f"Running {scenario_config['name']} with {model_config['name']}")
    print(f"{'='*60}")

    # Check if already completed
    output_path = Path(output_dir) / model_key / scenario_key
    results_file = output_path / "results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                existing = json.load(f)
            # Skip if we have at least 300 results (S1 has 301)
            if len(existing) >= 300:
                print(f"Already completed {len(existing)} instances, skipping...")
                return existing
        except:
            pass

    # Create client
    client = create_client(model_config)

    # Get scenario instances
    scenario = scenario_config["class"](num_testcases=-1)
    instances = scenario.get_instances("all")

    # Default limit for large scenarios
    if max_instances is None and len(instances) > 500:
        max_instances = 500
        print(f"Limiting to {max_instances} instances (original: {len(instances)})")

    if max_instances:
        instances = instances[:max_instances]

    print(f"Total instances: {len(instances)}")

    # Prepare output directory
    output_path = Path(output_dir) / model_key / scenario_key
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for i, instance in enumerate(instances):
        print(f"\rProcessing instance {i+1}/{len(instances)}...", end="", flush=True)

        prompt = instance.input.text

        # Generate completion
        start_time = time.time()
        completion = generate_completion(
            client,
            model_config["name"],
            prompt,
            scenario_config["instruction"],
        )
        elapsed_time = time.time() - start_time

        # Extract reference (if available)
        references = []
        if instance.references:
            references = [ref.output.text for ref in instance.references]

        result = {
            "instance_id": i,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "completion": completion,
            "references": references,
            "elapsed_time": elapsed_time,
        }

        # Add extra fields from instance if available
        if hasattr(instance, "extra_data") and instance.extra_data:
            result.update(instance.extra_data)

        results.append(result)

        # Save periodically
        if (i + 1) % 10 == 0:
            save_results(results, output_path)

    print(f"\nCompleted {len(results)} instances")

    # Final save
    save_results(results, output_path)

    return results


def save_results(results, output_path):
    """Save results to JSON and CSV."""
    # Save JSON
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save CSV (simplified)
    df = pd.DataFrame([{
        "instance_id": r["instance_id"],
        "completion": r["completion"][:1000] if r["completion"] else None,
        "elapsed_time": r["elapsed_time"],
    } for r in results])
    df.to_csv(output_path / "results.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Run CodeInsights evaluation")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Model to evaluate (default: all)")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default=None,
                        help="Scenario to run (default: all)")
    parser.add_argument("--output", type=str, default="./helm_results/evaluation",
                        help="Output directory")
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Maximum instances per scenario (default: all)")

    args = parser.parse_args()

    # Determine models and scenarios to run
    models = [args.model] if args.model else list(MODELS.keys())
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS.keys())

    print(f"Starting CodeInsights Evaluation")
    print(f"Models: {models}")
    print(f"Scenarios: {scenarios}")
    print(f"Output: {args.output}")
    print(f"Max instances: {args.max_instances or 'all'}")

    # Run evaluations
    all_results = {}

    for model_key in models:
        for scenario_key in scenarios:
            try:
                results = run_scenario(
                    model_key,
                    scenario_key,
                    args.output,
                    args.max_instances,
                )
                all_results[f"{model_key}_{scenario_key}"] = len(results)
            except Exception as e:
                print(f"\nError running {model_key}/{scenario_key}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    for key, count in all_results.items():
        print(f"  {key}: {count} instances")


if __name__ == "__main__":
    main()
