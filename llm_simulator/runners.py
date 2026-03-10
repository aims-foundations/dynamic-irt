"""Unified LLM model runners and registry.

Supports commercial API models (Claude, GPT, Gemini, Mistral) and
open-source models served via vLLM (LLaMA, Gemma, Qwen, GLM).

Each runner exposes two methods:
    call(prompt)     -> str          # single prompt (for iterative loop)
    generate(prompts) -> List[str]   # batch (for single-shot evaluation)
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

logger = logging.getLogger(__name__)


# ── Base class ────────────────────────────────────────────────────────────────


class LLMRunner:
    """Base class for all model runners."""

    def __init__(self, max_workers: int = 2, delay: float = 1.0):
        self.max_workers = max_workers
        self.delay = delay

    def call(self, prompt: str) -> str:
        """Generate a response for a single prompt."""
        raise NotImplementedError

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts.

        Default implementation uses ThreadPoolExecutor over call().
        VLLMRunner overrides this for true batch inference.
        """
        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.call, p): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("prompt %d failed: %s", idx + 1, e)
                    results[idx] = f"ERROR: {e}"
                time.sleep(self.delay / self.max_workers)
        return results


# ── Commercial API runners ────────────────────────────────────────────────────


class ClaudeRunner(LLMRunner):
    def __init__(self, api_model: str, **kwargs):
        super().__init__(**kwargs)
        import anthropic as _anthropic
        self.client = _anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.api_model = api_model

    def call(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.api_model, max_tokens=4000,
            stop_sequences=["\n```"],
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class GeminiRunner(LLMRunner):
    def __init__(self, api_model: str, **kwargs):
        super().__init__(**kwargs)
        import google.generativeai as _genai
        _genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = _genai.GenerativeModel(api_model)
        self._genai = _genai

    def call(self, prompt: str) -> str:
        resp = self.model.generate_content(
            contents=[prompt],
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=4000,
                stop_sequences=["\n```"],
            ),
        )
        return resp.text


class OpenAIRunner(LLMRunner):
    def __init__(self, api_model: str, base_url: str = None,
                 api_key: str = None, max_tokens: int = 4000,
                 stop: list = None, **kwargs):
        super().__init__(**kwargs)
        import openai as _openai
        self.client = _openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url,
            timeout=1800.0,  # 30 min — large prompts + busy servers need headroom
        )
        self.api_model = api_model
        self.max_tokens = max_tokens
        self.stop = stop

    def call(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            stop=self.stop,
        )
        return resp.choices[0].message.content


class MistralRunner(LLMRunner):
    def __init__(self, api_model: str, **kwargs):
        super().__init__(**kwargs)
        from mistralai import Mistral as _Mistral
        self.client = _Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.api_model = api_model

    def call(self, prompt: str) -> str:
        resp = self.client.chat.complete(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        text = resp.choices[0].message.content
        # Mistral doesn't always honour stop sequences
        idx = text.find("\n```")
        return text[:idx] if idx != -1 else text


# ── Open-source model runner (vLLM) ──────────────────────────────────────────


class VLLMRunner(LLMRunner):
    """Open-source model runner using vLLM for batch inference.

    Supports both call() (single prompt, for iterative) and generate()
    (true batch, for single-shot evaluation).
    """

    def __init__(self, hf_id: str, max_tokens: int = 4000,
                 temperature: float = 0.0, tensor_parallel_size: int = 1,
                 enforce_eager: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self._model = None
        self._sampling_params = None

    def _ensure_initialized(self):
        if self._model is not None:
            return
        from vllm import LLM, SamplingParams
        logger.info("Initializing vLLM: %s (enforce_eager=%s)", self.hf_id, self.enforce_eager)
        self._model = LLM(
            model=self.hf_id,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="auto",
            trust_remote_code=True,
            enforce_eager=self.enforce_eager,
        )
        self._sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _make_messages(self, prompt: str):
        """Wrap a text prompt in chat message format."""
        return [{"role": "user", "content": prompt}]

    def call(self, prompt: str) -> str:
        self._ensure_initialized()
        outputs = self._model.chat(
            [self._make_messages(prompt)], self._sampling_params,
        )
        return outputs[0].outputs[0].text

    def generate(self, prompts: List[str]) -> List[str]:
        self._ensure_initialized()
        conversations = [self._make_messages(p) for p in prompts]
        outputs = self._model.chat(conversations, self._sampling_params)
        return [o.outputs[0].text if o.outputs else "" for o in outputs]

    def cleanup(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


# ── Model configuration and factory ──────────────────────────────────────────


MODEL_CONFIGS: Dict[str, dict] = {
    # Commercial API models
    "claude": {
        "api_model": "claude-sonnet-4-20250514",
        "backend": "anthropic",
        "max_workers": 2,
        "delay": 1.0,
    },
    "gpt": {
        "api_model": "gpt-4.1-nano",
        "backend": "openai",
        "max_workers": 2,
        "delay": 0.8,
    },
    "gemini": {
        "api_model": "gemini-2.0-flash",
        "backend": "google",
        "max_workers": 3,
        "delay": 0.5,
    },
    "mistral": {
        "api_model": "mistral-large-latest",
        "backend": "mistral",
        "max_workers": 3,
        "delay": 0.5,
    },
    # Open-source models (vLLM)
    "llama": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "vllm",
        "max_tokens": 2048,
        "temperature": 0.0,
    },
    "gemma": {
        "hf_id": "google/gemma-3-27b-it",
        "backend": "vllm",
        "max_tokens": 2048,
        "temperature": 0.0,
    },
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "backend": "vllm",
        "max_tokens": 2048,
        "temperature": 0.0,
    },
    "glm": {
        "hf_id": "QuantTrio/GLM-4.7-AWQ",
        "backend": "vllm",
        "max_tokens": 8000,
        "temperature": 0.0,
    },
    # vLLM serve mode — connect to a running vLLM server via OpenAI API
    "glm_server": {
        "api_model": "QuantTrio/GLM-4.7-AWQ",
        "backend": "openai",
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "max_tokens": 2048,
        "max_workers": 32,
        "delay": 0.0,
    },
}

_API_RUNNER_MAP = {
    "anthropic": ClaudeRunner,
    "openai": OpenAIRunner,
    "google": GeminiRunner,
    "mistral": MistralRunner,
}


def create_runner(model_key: str, tensor_parallel_size: int = 1,
                  port: int = None) -> LLMRunner:
    """Create a runner for the given model key.

    Args:
        port: Override the server port for OpenAI-compatible backends
              (e.g., glm_server). Sets base_url to http://localhost:{port}/v1.
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {list(MODEL_CONFIGS.keys())}"
        )
    config = MODEL_CONFIGS[model_key]
    backend = config["backend"]

    if backend == "vllm":
        return VLLMRunner(
            hf_id=config["hf_id"],
            max_tokens=config.get("max_tokens", 4000),
            temperature=config.get("temperature", 0.0),
            tensor_parallel_size=tensor_parallel_size,
        )

    cls = _API_RUNNER_MAP[backend]
    kwargs = {
        "api_model": config["api_model"],
        "max_workers": config.get("max_workers", 2),
        "delay": config.get("delay", 1.0),
    }
    if port is not None:
        kwargs["base_url"] = f"http://localhost:{port}/v1"
    elif "base_url" in config:
        kwargs["base_url"] = config["base_url"]
    if "api_key" in config:
        kwargs["api_key"] = config["api_key"]
    if "max_tokens" in config:
        kwargs["max_tokens"] = config["max_tokens"]
    if "stop" in config:
        kwargs["stop"] = config["stop"]
    return cls(**kwargs)
