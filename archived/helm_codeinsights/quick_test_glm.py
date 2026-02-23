#!/usr/bin/env python3
"""Quick test of GLM-4.7-AWQ on CodeInsights S1 (correct_code) scenario."""

import time
import json
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from openai import OpenAI

# Config
VLLM_BASE_URL = "http://localhost:8240/v1"
MODEL_NAME = "glm-4.7-awq"
MAX_INSTANCES = 5

INSTRUCTION = (
    "You are a skilled C++ programmer working on a foundational programming course assignment. "
    "Your task is to write correct, efficient C++ code that solves the given problem. "
    "Write clean, well-structured code that follows good programming practices. "
    "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template. "
    "DO NOT reproduce the template part as the generated code would be inserted to the template, "
    "and make sure the code is compatible with the Unit Test Input. "
    "Ensure your code is correct, efficient, includes any class definition when needed, and handles all edge cases properly."
)


def load_scenario_data():
    """Load S1 correct_code data from HuggingFace."""
    # Use cached data (gated repo requires auth)
    data_file = "/lfs/skampere1/0/shared_hf_cache/datasets--CodeInsightTeam--code_insights_csv/snapshots/b2ed07387d109af257089734a14fd7beee273bd9/codeinsights_llm_simulation/data/Scenario1_full_data.csv"
    df = pd.read_csv(data_file, dtype={"pass": "str"})

    instances = []
    for question_id, question_df in df.groupby("question_unittest_id"):
        target = question_df.iloc[0]
        question_test_cases = []
        tc_parsing_success = True

        for testcase_str in target["question_unittests"].split("Unittest")[1:]:
            testcase_str = testcase_str[testcase_str.find(":") + 1:]
            input_idx = testcase_str.find("Input:")
            std_in_idx = testcase_str.find("STD input:")
            output_idx = testcase_str.find("Output:")
            if input_idx == -1 or std_in_idx == -1 or output_idx == -1:
                tc_parsing_success = False
                break
            testcase = {
                "input": testcase_str[input_idx + 6: std_in_idx].strip(),
                "std_in": testcase_str[std_in_idx + 10: output_idx].strip(),
                "output": testcase_str[output_idx + 7:].strip(),
            }
            question_test_cases.append(testcase)

        if not tc_parsing_success:
            continue

        prompt = (
            f"Question: {target['question_name']} — {target['question_text']}\n\n"
            "Template:\n"
            f"{target['question_template']}\n\n"
            "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template.\n"
            "– Do NOT reproduce any part of the template\n"
            "– Do NOT emit `int main()` (it's already declared)\n"
            "– Ensure your code is correct, efficient, handles all edge cases, and includes any needed class definitions\n"
            "IMPORTANT:\n"
            "Your entire response must be exactly one Markdown C++ code-block.\n"
            "1. The first line of your output must be: ```cpp\n"
            "2. The last line of your output must be: ```\n"
            "3. No extra characters, whitespace, or text may appear before the opening ```cpp or after the closing ```.\n"
        )
        instances.append({
            "question_id": str(question_id),
            "question_name": target.get("question_name", ""),
            "prompt": prompt,
            "template": target["question_template"],
            "test_cases": question_test_cases,
        })

    return instances


def main():
    print("Loading S1 (correct_code) scenario data from HuggingFace...")
    instances = load_scenario_data()
    print(f"Loaded {len(instances)} instances, testing with {MAX_INSTANCES}")

    client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)
    results = []

    for i, inst in enumerate(instances[:MAX_INSTANCES]):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{MAX_INSTANCES}] Question: {inst['question_name']} (ID: {inst['question_id']})")
        print(f"{'='*60}")

        start = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": inst["prompt"]},
                ],
                max_tokens=4000,
                temperature=0.0,
            )
            completion = response.choices[0].message.content
            elapsed = time.time() - start
            tokens = response.usage.completion_tokens

            print(f"Time: {elapsed:.1f}s | Tokens: {tokens}")
            print(f"Response (first 500 chars):\n{completion[:500]}")

            results.append({
                "question_id": inst["question_id"],
                "question_name": inst["question_name"],
                "completion": completion,
                "elapsed_time": elapsed,
                "completion_tokens": tokens,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "question_id": inst["question_id"],
                "question_name": inst["question_name"],
                "completion": None,
                "error": str(e),
            })

    # Save results
    output_dir = Path(__file__).parent / "glm_test_results"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "glm_s1_quick_test.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DONE — {len(results)} results saved to {output_dir / 'glm_s1_quick_test.json'}")
    success = sum(1 for r in results if r.get("completion"))
    print(f"Successful completions: {success}/{len(results)}")


if __name__ == "__main__":
    main()
