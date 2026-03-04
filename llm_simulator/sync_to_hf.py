"""Sync simulation results from skampere2 to HuggingFace.

Usage:
    python llm_simulator/sync_to_hf.py

Copies latest shard files from skampere2, merges into a single JSONL, and
uploads to CodeInsightTeam/simulation_output on HuggingFace.

Handles both old CSV shards and new JSONL shards transparently.
"""

import json
import os
import subprocess
import sys

import pandas as pd


REMOTE = "sttruong@skampere2.stanford.edu"
REMOTE_DIR = "/lfs/skampere2/0/sttruong/support/codeinsight/llm_simulator/results/llm_eval"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "results", "llm_eval")
HF_REPO = "CodeInsightTeam/simulation_output"


def _load_shard(path: str) -> pd.DataFrame:
    """Load a shard file (CSV or JSONL)."""
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def sync():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # 1. Copy shard files from skampere2 (try JSONL first, fall back to CSV)
    print("Syncing from skampere2...")
    shard_files = []
    for shard in ["shard0of2", "shard1of2"]:
        local_path = None
        for ext in [".jsonl", ".csv"]:
            remote_path = f"{REMOTE}:{REMOTE_DIR}/glm_n10_attempts1_{shard}{ext}"
            local = os.path.join(LOCAL_DIR, f"glm_n10_attempts1_{shard}{ext}")
            result = subprocess.run(
                ["scp", remote_path, local],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                local_path = local
                break
        if local_path:
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            print(f"  {shard}: {size_mb:.1f} MB ({os.path.basename(local_path)})")
            shard_files.append(local_path)
        else:
            print(f"  {shard}: not found")

    if not shard_files:
        print("No shard files found, nothing to upload.")
        return

    # 2. Merge shards into a single JSONL
    dfs = [_load_shard(f) for f in shard_files]
    merged = pd.concat(dfs, ignore_index=True)
    if "prompt" not in merged.columns:
        merged["prompt"] = ""
    print(f"Merged: {len(merged)} rows, {merged['student_id'].nunique()} students, "
          f"{merged['question_id'].nunique()} questions")

    jsonl_path = os.path.join(LOCAL_DIR, "glm_n10_attempts1.jsonl")
    with open(jsonl_path, "w") as f:
        for _, row in merged.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + "\n")
    print(f"Saved JSONL: {os.path.getsize(jsonl_path) / 1024 / 1024:.1f} MB")

    # 3. Upload to HuggingFace (JSONL only)
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            with open(token_path) as fh:
                token = fh.read().strip()
    if not token:
        print("ERROR: No HF_TOKEN found. Set HF_TOKEN or run `huggingface-cli login`.")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"Uploading to {HF_REPO}...")

    api.upload_file(
        path_or_fileobj=jsonl_path,
        path_in_repo="glm/glm_n10_attempts1.jsonl",
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"Done! https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    sync()
