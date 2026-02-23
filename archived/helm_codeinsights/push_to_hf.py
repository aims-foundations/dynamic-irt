#!/usr/bin/env python3
"""Push GLM evaluation results to HuggingFace."""
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "CodeInsightTeam/evaluation_result_new"
RESULTS_DIR = Path(__file__).parent / "helm_results/evaluation/glm"

SCENARIO_MAP = {
    "S1": "correct_code",
    "S2": "student_coding",
    "S3": "student_mistake",
    "S4": "code_efficiency",
}

MODEL_NAME = "glm-4.7-awq"

# Load token from env or stored credential file
token = os.environ.get("HF_TOKEN") or (Path("~/.cache/huggingface/token").expanduser().read_text().strip()
    if Path("~/.cache/huggingface/token").expanduser().exists() else None)
print(f"Using token: {token[:10]}..." if token else "No token found!")

api = HfApi(token=token)

for scenario_key, scenario_name in SCENARIO_MAP.items():
    results_file = RESULTS_DIR / scenario_key / "results.json"
    if not results_file.exists():
        print(f"Skipping {scenario_key} — no results file")
        continue

    with open(results_file) as f:
        results = json.load(f)

    total = {"S1": 301, "S2": 1000, "S3": 1000, "S4": 1000}[scenario_key]
    status = "complete" if len(results) >= total else f"partial_{len(results)}of{total}"
    path_in_repo = f"{MODEL_NAME}/{scenario_name}/results_{status}.json"

    print(f"Uploading {scenario_key} ({len(results)}/{total}) → {path_in_repo} ...")
    api.upload_file(
        path_or_fileobj=str(results_file),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Add GLM-4.7-AWQ {scenario_name} results ({len(results)}/{total})",
    )
    print(f"  Done!")

print("\nAll uploads complete!")
print(f"View at: https://huggingface.co/datasets/{REPO_ID}")
