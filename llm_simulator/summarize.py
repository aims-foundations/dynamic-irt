"""Behavioral summarization of student submission history.

Compresses raw code examples into compact behavioral descriptions
using a cheap/fast LLM (Haiku). Summaries capture problem-solving
approach, mistake patterns, and iteration style without reproducing code.
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

RAG_SUMMARY_PROMPT = (
    "Here is a code submission by a student on a C++ programming problem. "
    "State factually what the student implemented in 1-2 sentences. "
    "Do not interpret, analyze, or speculate about why tests passed or failed. "
    "Do not reproduce any code."
)

SELF_SUMMARY_PROMPT = (
    "Here is a student's own submission history on a prior C++ problem. "
    "Summarize their coding style and debugging approach in 1-2 sentences. "
    "Focus on: their strategy, common mistake patterns, how they iterate. "
    "Do not reproduce any code."
)

QUESTION_SUMMARY_PROMPT = (
    "Summarize this C++ programming problem in one sentence. "
    "Focus on: what the student must implement and the key data structure or algorithm involved. "
    "Do not reproduce any code."
)


class HistorySummarizer:
    def __init__(
        self,
        summary_model: str = "claude-haiku-4-5-20251001",
        cache_dir: Optional[str] = None,
    ):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = summary_model
        self.cache_dir = cache_dir or os.path.join("results", "llm_eval", "summary_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        cache_file = os.path.join(self.cache_dir, "summaries.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                self._cache = json.load(f)
            logger.info("Loaded %d cached summaries.", len(self._cache))

    def _save_cache(self):
        cache_file = os.path.join(self.cache_dir, "summaries.json")
        with open(cache_file, "w") as f:
            json.dump(self._cache, f)

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _call_llm(self, prompt: str) -> str:
        key = self._cache_key(prompt)
        if key in self._cache:
            return self._cache[key]

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        result = msg.content[0].text.strip()
        lines = result.split("\n")
        lines = [l for l in lines if not l.startswith("#")]
        result = "\n".join(lines).strip()
        self._cache[key] = result
        self._save_cache()
        return result

    def batch_call_llm(self, prompts: List[str], max_workers: int = 20) -> List[str]:
        """Call Haiku in parallel for uncached prompts. Returns results in order."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(prompts)
        uncached = []

        for i, prompt in enumerate(prompts):
            key = self._cache_key(prompt)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached.append((i, prompt))

        if not uncached:
            return results

        logger.info("Haiku batch: %d cached, %d to call (%d workers)",
                     len(prompts) - len(uncached), len(uncached), min(max_workers, len(uncached)))

        def _single_call(prompt):
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            result = msg.content[0].text.strip()
            lines = result.split("\n")
            lines = [l for l in lines if not l.startswith("#")]
            return "\n".join(lines).strip()

        n_failed = 0
        with ThreadPoolExecutor(max_workers=min(max_workers, len(uncached))) as pool:
            futures = {pool.submit(_single_call, prompt): (i, prompt) for i, prompt in uncached}
            for future in as_completed(futures):
                i, prompt = futures[future]
                try:
                    result = future.result()
                    key = self._cache_key(prompt)
                    self._cache[key] = result
                    results[i] = result
                except Exception as e:
                    logger.error("Haiku call failed: %s", e)
                    results[i] = ""
                    n_failed += 1

        if n_failed:
            logger.warning("Haiku batch: %d/%d calls failed (empty summaries)",
                           n_failed, len(uncached))

        self._save_cache()
        return results

