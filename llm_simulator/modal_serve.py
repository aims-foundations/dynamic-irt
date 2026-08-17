"""Serve OSS models on Modal with vLLM (OpenAI-compatible API).

Usage:
    modal deploy llm_simulator/modal_serve.py::qwen_app
    modal deploy llm_simulator/modal_serve.py::gemma_app

    # Run eval:
    python -m llm_simulator.eval_student_split \
        --models qwen_server \
        --base_url https://<your-app>.modal.run/v1
"""

import subprocess

import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "huggingface_hub")
)

cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
CACHE_DIR = "/root/.cache/vllm"

# Model weights land in the huggingface_hub cache, not the vLLM cache;
# without this volume every cold container re-downloads 28-62GB of weights.
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_CACHE_DIR = "/root/.cache/huggingface"

# ── Qwen3-14B ──

qwen_app = modal.App("codeinsight-qwen")

@qwen_app.function(
    image=vllm_image, gpu="A100-40GB", timeout=3600, scaledown_window=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_DIR: cache_vol, HF_CACHE_DIR: hf_vol},
    max_containers=6,
)
@modal.web_server(port=8000, startup_timeout=900)
def qwen_serve():
    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3-14B",
        "--dtype", "auto", "--trust-remote-code",
        "--max-model-len", "16384",
        "--async-scheduling",
        "--host", "0.0.0.0", "--port", "8000",
    ])

# ── Gemma 4 31B ──

gemma_app = modal.App("codeinsight-gemma")

@gemma_app.function(
    image=vllm_image, gpu="H100", timeout=3600, scaledown_window=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CACHE_DIR: cache_vol, HF_CACHE_DIR: hf_vol},
    max_containers=4,
)
@modal.web_server(port=8000, startup_timeout=900)
def gemma_serve():
    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "google/gemma-4-31B-it",
        "--dtype", "auto", "--trust-remote-code",
        "--max-model-len", "16384",
        "--max-num-batched-tokens", "16384",
        "--async-scheduling",
        "--host", "0.0.0.0", "--port", "8000",
    ])
