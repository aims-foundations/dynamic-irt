"""
Inspect a HELM run from HuggingFace CodeInsightTeam/evaluation_results.

Usage:
    python inspect_hf_run.py --scenario correct_code --model gemma-3-27b-it --temp 0.6
    python inspect_hf_run.py --scenario student_coding --model llama-3.1-8b-instruct --temp 0.3 --num 5
"""

import argparse
import json
import os
from huggingface_hub import hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "CodeInsightTeam/evaluation_results"
CACHE_DIR = "/lfs/skampere1/0/sttruong/.cache/huggingface"


def download_file(scenario, model, temp, filename):
    path = f"runs/{scenario}/{model}/temp_{temp}/{filename}"
    try:
        return hf_hub_download(
            REPO_ID, path,
            repo_type="dataset", token=HF_TOKEN, cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(f"Could not download {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Inspect a HELM run from HuggingFace")
    parser.add_argument("--scenario", required=True, help="e.g. correct_code, student_coding, student_mistake, code_efficiency")
    parser.add_argument("--model", required=True, help="e.g. gemma-3-27b-it, llama-3.1-8b-instruct, qwen2.5-14b-instruct")
    parser.add_argument("--temp", required=True, help="e.g. 0.0, 0.3, 0.6, 0.9")
    parser.add_argument("--num", type=int, default=3, help="Number of completions to print (default: 3)")
    args = parser.parse_args()

    print(f"=== {args.scenario} / {args.model} / temp_{args.temp} ===\n")

    # 1. Run spec
    run_spec_path = download_file(args.scenario, args.model, args.temp, "run_spec.json")
    if run_spec_path:
        with open(run_spec_path) as f:
            spec = json.load(f)
        print("--- Run Spec ---")
        print(f"  Name: {spec['name']}")
        print(f"  Scenario class: {spec['scenario_spec']['class_name'].split('.')[-1]}")
        print(f"  stop_sequences: {spec['adapter_spec']['stop_sequences']}")
        print(f"  temperature: {spec['adapter_spec']['temperature']}")
        print(f"  max_tokens: {spec['adapter_spec']['max_tokens']}")
        print(f"  Instructions: {spec['adapter_spec']['instructions'][:200]}...")
        print()

    # 2. Scenario state (completions)
    ss_path = download_file(args.scenario, args.model, args.temp, "scenario_state.json")
    if ss_path:
        with open(ss_path) as f:
            data = json.load(f)
        states = data["request_states"]
        print(f"--- Scenario State: {len(states)} request_states ---\n")

        for i in range(min(args.num, len(states))):
            state = states[i]
            instance = state["instance"]
            completion = state["result"]["completions"][0]["text"]

            print(f"--- Completion {i} ---")
            print(f"  Instance ID: {instance['id']}")

            # Show prompt (truncated)
            prompt = instance["input"]["text"]
            if len(prompt) > 500:
                print(f"  Prompt: {prompt[:500]}...[truncated, {len(prompt)} chars total]")
            else:
                print(f"  Prompt: {prompt}")

            # Show completion
            print(f"  Completion ({len(completion)} chars):")
            if len(completion) > 1000:
                print(f"    {repr(completion[:1000])}...[truncated]")
            else:
                print(f"    {repr(completion)}")
            print()

    # 3. Per-instance stats summary
    stats_path = download_file(args.scenario, args.model, args.temp, "per_instance_stats.json")
    if stats_path:
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"--- Per-Instance Stats: {len(stats)} entries ---")

        # Aggregate stats by name
        from collections import defaultdict
        agg = defaultdict(list)
        for entry in stats:
            for s in entry["stats"]:
                name = s["name"] if isinstance(s["name"], str) else s["name"].get("name", str(s["name"]))
                if "mean" in s:
                    agg[name].append(s["mean"])

        for name, values in sorted(agg.items()):
            avg = sum(values) / len(values)
            nonzero = sum(1 for v in values if v > 0)
            print(f"  {name}: mean={avg:.4f}, nonzero={nonzero}/{len(values)}")


if __name__ == "__main__":
    main()
