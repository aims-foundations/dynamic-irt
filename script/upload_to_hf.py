"""
Upload all D2 simulation results to CodeInsightTeam/evaluation_results on HuggingFace.

Consolidates three sources into a clean directory structure:
  runs/{scenario}/{model}/temp_{temperature}/
      per_instance_stats.json
      run_spec.json
      scenario.json
      scenario_state.json
      stats.json

Sources:
  A. HuggingFace CodeInsightTeam/evaluation_results (temp=0.0, 4 models, 3 scenarios)
  B. Local codeinsights_Dec8 (temp=0.3/0.6/0.9, 3 models, 4 scenarios)
  C. Local codeinsights_Oct3 (temp=0.0/0.5/1.0, 4 models, 5 scenarios, many incomplete)

Strategy: Build a staging directory with copies/symlinks, then upload_folder in one commit.
This script is READ-ONLY on source data.
"""

import os
import re
import shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, list_repo_tree

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
TARGET_REPO = "CodeInsightTeam/evaluation_results"
STAGING_DIR = Path("/lfs/skampere1/0/sttruong/support/codeinsight/_hf_staging")
HELM_FILES = ["per_instance_stats.json", "run_spec.json", "scenario.json", "scenario_state.json", "stats.json"]

# Model name normalization: strip vendor prefix for clean paths
MODEL_CLEAN = {
    "google_gemma-3-27b-it": "gemma-3-27b-it",
    "meta_llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "meta_llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "qwen_qwen2.5-14b-instruct": "qwen2.5-14b-instruct",
    "deepseek-ai_deepseek-llm-67b-chat": "deepseek-llm-67b-chat",
    "openai_gpt-4o-2024-08-06": "gpt-4o-2024-08-06",
}

# Scenario name normalization
SCENARIO_CLEAN = {
    "codeinsights_correct_code": "correct_code",
    "codeinsights_student_coding": "student_coding",
    "codeinsights_student_mistake": "student_mistake",
    "codeinsights_student_mistake_coding": "student_mistake",
    "codeinsights_code_efficiency": "code_efficiency",
    "codeinsights_edge_case": "edge_case",
    "correct_code": "correct_code",
    "student_coding": "student_coding",
    "student_mistake_coding": "student_mistake",
}


def parse_local_dirname(dirname):
    """Parse HELM directory name into (scenario, temperature, model)."""
    parts = dirname.split(":", 1)
    if len(parts) < 2:
        return None
    scenario_raw = parts[0]
    params = parts[1]
    temp_match = re.search(r"temperature=([\d.]+)", params)
    if not temp_match:
        return None
    temperature = temp_match.group(1)
    model_match = re.search(r"model=(.+)$", params)
    if not model_match:
        return None
    model_raw = model_match.group(1)
    scenario = SCENARIO_CLEAN.get(scenario_raw, scenario_raw)
    model = MODEL_CLEAN.get(model_raw, model_raw)
    return scenario, temperature, model


def parse_hf_dirname(dirname):
    """Parse HF CodeInsightTeam directory name into (scenario, model)."""
    parts = dirname.split(":model=")
    if len(parts) != 2:
        return None
    scenario = SCENARIO_CLEAN.get(parts[0], parts[0])
    model = MODEL_CLEAN.get(parts[1], parts[1])
    return scenario, model


def stage_local_source(base_path, batch_name, staged_paths):
    """Copy local HELM results into the staging directory."""
    base = Path(base_path)
    if not base.exists():
        print(f"  SKIP: {base} does not exist")
        return 0, 0

    staged = 0
    skipped = 0

    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "eval_cache":
            continue

        parsed = parse_local_dirname(run_dir.name)
        if parsed is None:
            print(f"  SKIP (unparseable): {run_dir.name}")
            skipped += 1
            continue

        scenario, temperature, model = parsed
        target_dir = STAGING_DIR / "runs" / scenario / model / f"temp_{temperature}"
        has_metrics = (run_dir / "per_instance_stats.json").exists()

        # Don't overwrite a complete run with an incomplete one
        if str(target_dir) in staged_paths and staged_paths[str(target_dir)] and not has_metrics:
            print(f"  SKIP (would overwrite complete): runs/{scenario}/{model}/temp_{temperature}")
            skipped += 1
            continue

        # Check which HELM files exist
        found_files = [f for f in HELM_FILES if (run_dir / f).exists()]
        if not found_files:
            print(f"  SKIP (no HELM files): {run_dir.name}")
            skipped += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in found_files:
            src = run_dir / fname
            dst = target_dir / fname
            shutil.copy2(str(src), str(dst))

        status = "complete" if has_metrics else "partial"
        print(f"  [{batch_name}] runs/{scenario}/{model}/temp_{temperature} ({len(found_files)} files, {status})")
        staged_paths[str(target_dir)] = has_metrics
        staged += 1

    return staged, skipped


def stage_hf_source(staged_paths):
    """Download from CodeInsightTeam/evaluation_results and stage locally."""
    source_repo = "CodeInsightTeam/evaluation_results"
    temperature = "0.0"

    all_items = list(list_repo_tree(
        source_repo, repo_type="dataset", recursive=True, token=HF_TOKEN
    ))

    # Group files by run directory
    run_files = {}
    for item in all_items:
        if not item.path.startswith("runs/codeinsights/"):
            continue
        parts = item.path.split("/")
        if len(parts) >= 4:
            run_key = parts[2]
            filename = parts[3]
            if filename in HELM_FILES:
                run_files.setdefault(run_key, []).append((item.path, filename))

    staged = 0
    for run_key, files in sorted(run_files.items()):
        parsed = parse_hf_dirname(run_key)
        if parsed is None:
            print(f"  SKIP (unparseable): {run_key}")
            continue

        scenario, model = parsed
        target_dir = STAGING_DIR / "runs" / scenario / model / f"temp_{temperature}"
        target_dir.mkdir(parents=True, exist_ok=True)

        has_metrics = any(f[1] == "per_instance_stats.json" for f in files)
        print(f"  [HF] runs/{scenario}/{model}/temp_{temperature} ({len(files)} files)")

        for source_path, filename in files:
            local_path = hf_hub_download(
                source_repo, source_path,
                repo_type="dataset", token=HF_TOKEN,
                cache_dir="/lfs/skampere1/0/sttruong/.cache/huggingface",
            )
            shutil.copy2(local_path, str(target_dir / filename))

        staged_paths[str(target_dir)] = has_metrics
        staged += 1

    return staged


def main():
    os.environ["HF_HOME"] = "/lfs/skampere1/0/sttruong/.cache/huggingface"
    api = HfApi(token=HF_TOKEN)

    # Clean staging directory
    if STAGING_DIR.exists():
        shutil.rmtree(str(STAGING_DIR))
    STAGING_DIR.mkdir(parents=True)

    print("=" * 60)
    print(f"Target: {TARGET_REPO}")
    print(f"Staging: {STAGING_DIR}")
    print("=" * 60)

    staged_paths = {}

    # Source A: HuggingFace CodeInsightTeam (temp=0.0) — stage first (these have metrics)
    print("\n--- Source A: CodeInsightTeam/evaluation_results (temp=0.0) ---")
    try:
        hf_count = stage_hf_source(staged_paths)
        print(f"  Staged: {hf_count} runs")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Source B: Local Dec8 (temp=0.3, 0.6, 0.9) — most complete
    print("\n--- Source B: codeinsights_Dec8 (temp=0.3/0.6/0.9) ---")
    dec8_path = "/dfs/scratch0/sttruong/helm_code_staging/benchmark_output/runs/codeinsights_Dec8"
    dec8_staged, dec8_skip = stage_local_source(dec8_path, "Dec8", staged_paths)
    print(f"  Staged: {dec8_staged}, Skipped: {dec8_skip}")

    # Source C: Local Oct3 (temp=0.0, 0.5, 1.0) — partial, won't overwrite complete
    print("\n--- Source C: codeinsights_Oct3 (temp=0.0/0.5/1.0) ---")
    oct3_path = "/dfs/scratch0/sttruong/helm_code_staging/benchmark_output/runs/codeinsights_Oct3"
    oct3_staged, oct3_skip = stage_local_source(oct3_path, "Oct3", staged_paths)
    print(f"  Staged: {oct3_staged}, Skipped: {oct3_skip}")

    # Count total staged
    total_files = sum(1 for _ in STAGING_DIR.rglob("*.json"))
    total_size_mb = sum(f.stat().st_size for f in STAGING_DIR.rglob("*.json")) / (1024 * 1024)
    print(f"\n--- Staging complete: {total_files} files, {total_size_mb:.0f} MB ---")

    # Upload using upload_folder with create_pr=True (single commit as a PR)
    print(f"\n--- Uploading to {TARGET_REPO} (as pull request) ---")
    api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=TARGET_REPO,
        repo_type="dataset",
        create_pr=True,
        commit_message="Add consolidated D2 simulation results with clean folder structure",
        commit_description=(
            "Consolidates HELM evaluation results from 3 sources:\n"
            "- CodeInsightTeam/evaluation_results (temp=0.0, 4 models, 3 scenarios)\n"
            "- Local Dec8 batch (temp=0.3/0.6/0.9, 3 models, 4 scenarios)\n"
            "- Local Oct3 batch (temp=0.0/0.5/1.0, partial runs)\n\n"
            "Clean structure: runs/{scenario}/{model}/temp_{temperature}/"
        ),
    )

    print("\n" + "=" * 60)
    print("Upload complete!")
    print("=" * 60)

    # Clean up staging
    shutil.rmtree(str(STAGING_DIR))
    print("Staging directory cleaned up.")


if __name__ == "__main__":
    main()
