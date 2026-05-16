"""Behavioral summarization of student submission history.

Compresses raw code examples into compact behavioral descriptions
using a cheap/fast LLM (Haiku). Summaries capture problem-solving
approach, mistake patterns, and iteration style without reproducing code.
"""

import hashlib
import json
import logging
import os
from itertools import groupby
from typing import Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = (
    "Here is a student's submission history on a C++ programming problem. "
    "Describe their problem-solving approach in 2-3 sentences. "
    "Focus on: what strategy they used, what mistakes they made, how they debugged. "
    "Do not reproduce any code."
)

RAG_SUMMARY_PROMPT = (
    "Here is a code submission by a student on a C++ programming problem. "
    "Describe their approach in 1-2 sentences. "
    "Focus on: what strategy they used and what the result was. "
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

    def summarize_history(self, examples: List[dict]) -> str:
        """Summarize a student's submission history into behavioral descriptions.

        Takes the same example dicts that build_prompt uses (with keys:
        question_name, question_text, question_template, response,
        response_type, pass_pattern).

        Returns a compact summary string replacing the raw code examples.
        """
        if not examples:
            return ""

        parts = ["=== Student Submission History (summarized) ===\n"]

        q_num = 0
        for qname, group in groupby(examples, key=lambda x: x["question_name"]):
            group = list(group)
            q_num += 1

            attempts_block = []
            for a, ex in enumerate(group, start=1):
                rtype = ex.get("response_type", "Submit")
                action = "Precheck" if rtype == "Prechecked" else "Submit"
                pp = ex.get("pass_pattern", "")
                code = ex.get("response", "")
                n_lines = code.count("\n") + 1 if code else 0
                attempts_block.append(
                    f"Attempt {a} [{action}] result: {pp}, {n_lines} lines\n{code}"
                )

            prompt = (
                f"{SUMMARY_PROMPT}\n\n"
                f"Problem: {qname}\n"
                f"{group[0].get('question_text', '')}\n\n"
                + "\n\n".join(attempts_block)
            )

            summary = self._call_llm(prompt)
            n_attempts = len(group)
            final_result = group[-1].get("pass_pattern", "?")
            q_desc = group[0].get("question_text", "").strip()
            parts.append(
                f"--- Problem {q_num}: {qname} ({n_attempts} attempts, final: {final_result}) ---\n"
                f"Description: {q_desc}\n\n"
                f"Student's Problem-Solving Approach:\n{summary}\n"
            )

        parts.append(
            "=== New Problem ===\n\n"
            "Now, using the same student's coding style, approach, and "
            "Precheck/Submit strategy, attempt this new problem:\n"
        )
        return "\n".join(parts)

    def summarize_rag_context(self, rag_text: str) -> str:
        """Summarize RAG-retrieved peer/self submissions into compact descriptions."""
        if not rag_text:
            return ""

        sections = rag_text.split("---")
        parts = ["=== Reference Submissions (summarized) ===\n"]

        for section in sections:
            section = section.strip()
            if not section or section.startswith("=== Reference"):
                continue

            header_end = section.find("---")
            if header_end == -1:
                header = ""
                body = section
            else:
                header = section[:header_end].strip()
                body = section[header_end + 3:].strip()

            if len(section) < 50:
                parts.append(section)
                continue

            prompt = f"{RAG_SUMMARY_PROMPT}\n\n{section}"
            summary = self._call_llm(prompt)
            parts.append(f"--- {header} ---\n{summary}\n" if header else f"{summary}\n")

        return "\n".join(parts)
