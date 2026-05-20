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

FEEDBACK_SUMMARY_PROMPT = (
    "Here are test case results from running a student's code submission. "
    "Condense the failing test cases into a compact log format. "
    "For each failed test, state: the test input, the expected output, "
    "and the actual output the code produced. "
    "Output only the factual test results as a log. "
    "Do not interpret, analyze, or explain the results."
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

        self._save_cache()
        return results

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

    def summarize_self_trajectory(self, traj, question_name: str = "") -> str:
        """Summarize the test student's own trajectory on a prior question.

        Args:
            traj: SelfTrajectory dataclass from rag.py
            question_name: Overrides traj.question_name if provided
        """
        name = question_name or traj.question_name
        attempts_block = []
        for i, a in enumerate(traj.attempts, 1):
            rtype = "Precheck" if a["response_type"] == "Prechecked" else "Submit"
            attempts_block.append(
                f"Attempt {i} [{rtype}] result: {a['pass_pattern']}\n{a['response']}"
            )

        prompt = (
            f"{SELF_SUMMARY_PROMPT}\n\n"
            f"Problem: {name}\n\n"
            + "\n\n".join(attempts_block)
        )
        return self._call_llm(prompt)

    def summarize_question(self, question_text: str, question_name: str = "") -> str:
        """Summarize a programming problem into one sentence."""
        prompt = f"{QUESTION_SUMMARY_PROMPT}\n\nProblem: {question_name}\n{question_text[:1000]}"
        return self._call_llm(prompt)

    def summarize_feedback(self, previous_code: str, failed_tests: list) -> str:
        """Condense failed test cases into a compact log. No code is sent to Haiku."""
        test_lines = []
        for i, t in enumerate(failed_tests[:5], 1):
            test_lines.append(
                f"Test {i}:\n"
                f"  Input: {(t.get('input') or '')[:300]}\n"
                f"  Expected: {(t.get('expected') or '')[:300]}\n"
                f"  Got: {(t.get('actual') or '(no output)')[:300]}"
            )

        prompt = f"{FEEDBACK_SUMMARY_PROMPT}\n\n" + "\n\n".join(test_lines)
        return self._call_llm(prompt)

    def summarize_rag_context(self, rag_text: str) -> str:
        """Summarize RAG-retrieved self submissions into compact descriptions."""
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
