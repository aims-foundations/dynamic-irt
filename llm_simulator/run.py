"""LLM student simulation: iterative code generation and grading.

Core module providing run_evaluation() which runs the trajectory-grounded
attempt loop: at each step the model sees the student's real prior attempts
and predicts the next submission, which is then compiled and graded.

Entry point is eval_student_split.py, not this file directly.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .data_loader import EvalItem, parse_test_cases
from .prompts import build_prompt, extract_action, extract_code
from .runners import MODEL_CONFIGS, create_runner

logger = logging.getLogger(__name__)


# ── Grading ─────────────────────────────────────────────────────────────────


def _grade_single(template: str, testcases: list, code: str) -> list:
    """Grade a code submission against test cases. Returns per-test 0/1 results."""
    formatted = []
    std_inputs = []
    for tc in testcases:
        formatted.append({"extra": "", "testcode": tc["input"],
                          "expected_output": tc["output"]})
        std_inputs.append(tc.get("std_in", ""))

    code_with_answer = template.replace("{{ STUDENT_ANSWER }}", code)
    start_idx = code_with_answer.find("{% for TEST in TESTCASES %}")
    end_idx = code_with_answer.find("{% endfor %}") + len("{% endfor %}")

    codes = [code_with_answer[:start_idx] + tc["testcode"] + code_with_answer[end_idx:]
             for tc in formatted]

    temp_dir = tempfile.mkdtemp()
    try:
        results_list = []
        for i, src in enumerate(codes):
            cpp_file = os.path.join(temp_dir, f"tc_{i}.cpp")
            exe_file = os.path.join(temp_dir, f"tc_{i}.out")
            with open(cpp_file, "w") as f:
                f.write(src)

            comp = subprocess.run(
                ["g++", "-std=c++11", cpp_file, "-o", exe_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            if comp.returncode != 0:
                results_list.append(0)
                continue

            try:
                run = subprocess.run(
                    ["timeout", "10", exe_file], input=std_inputs[i],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                if run.returncode != 0:
                    results_list.append(0)
                    continue
                results_list.append(1 if testcases[i]["output"].strip() == run.stdout.strip() else 0)
            except Exception:
                results_list.append(0)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results_list


def _grade_batch(jobs, chunk_items, test_data, codes):
    """Grade a batch of jobs in parallel. Returns {i: per-test 0/1 results}."""
    if not jobs:
        return {}
    results = {}
    n_workers = min(32, len(jobs))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for i, idx in jobs.items():
            td = test_data[idx]
            futures[pool.submit(
                _grade_single, chunk_items[idx].question_template,
                td["test_cases"], codes[i],
            )] = i
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                logger.error("Grading failed for item %d: %s", i, e)
    return results


# ── Output records ──────────────────────────────────────────────────────────


@dataclass
class AttemptRecord:
    attempt_id: int
    timestamp: str
    response_type: str
    prompt: str
    raw_response: str
    code: Optional[str]
    pass_pattern: str


@dataclass
class EvalResult:
    question_id: str
    student_id: Optional[str]
    model: str
    attempts: List[AttemptRecord] = field(default_factory=list)
    course_id: str = ""
    section_id: str = ""
    is_exam: str = ""

    def to_rows(self) -> List[dict]:
        return [
            {
                "student_id": self.student_id or "",
                "course_id": self.course_id or "",
                "section_id": self.section_id or "",
                "question_unittest_id": self.question_id,
                "attempt_id": str(rec.attempt_id),
                "timestamp": rec.timestamp,
                "is_exam": self.is_exam or "",
                "response_type": rec.response_type,
                "response": rec.code or "",
                "pass": rec.pass_pattern,
                "model": self.model,
                "prompt": rec.prompt,
                "raw_response": rec.raw_response,
            }
            for rec in self.attempts
        ]


# ── Initial prompt construction ─────────────────────────────────────────────


def _build_initial_prompt(item, persona_builder=None):
    """Build the first-attempt prompt from pre-computed item attributes."""
    persona_text = None
    if persona_builder and item.student_id:
        persona_text = persona_builder.build_persona_text(item.student_id)

    return build_prompt(
        question_name=item.question_name,
        question_text=item.question_text,
        question_template=item.question_template,
        persona_text=persona_text,
        self_summaries=getattr(item, "_self_summaries", None),
        question_metadata=getattr(item, "_question_metadata", None),
        question_summary=getattr(item, "_question_summary", None),
    )


# ── Main evaluation loop ───────────────────────────────────────────────────


def run_evaluation(
    items: List[EvalItem],
    model_key: str,
    max_attempts: int,
    student_to_course: Optional[Dict[str, str]] = None,
    student_to_section: Optional[Dict[str, str]] = None,
    question_to_is_exam: Optional[Dict[str, str]] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 20,
    chunk_size: int = 50,
    persona_builder=None,
    history_summarizer=None,
    prompt_log_file=None,
    base_url: Optional[str] = None,
    no_trajectory: bool = False,
) -> List[EvalResult]:
    """Run iterative LLM evaluation on a list of items.

    Each item follows the student's real attempt trajectory (grounded mode):
    the LLM sees the student's actual code submissions step-by-step and
    predicts the next attempt, matching their coding style and approach.
    Items are processed in chunks with results saved after each chunk.
    """
    student_to_course = student_to_course or {}
    student_to_section = student_to_section or {}
    question_to_is_exam = question_to_is_exam or {}

    runner = create_runner(model_key, base_url=base_url)
    all_results: List[EvalResult] = []

    # Resume: skip already-completed (student, question) pairs
    existing_rows = []
    if output_dir:
        filename = f"{model_key}_attempts{max_attempts}.jsonl"
        existing_path = os.path.join(output_dir, filename)
        if os.path.exists(existing_path):
            with open(existing_path) as f:
                existing_rows = [json.loads(line) for line in f]
            completed = {(str(r["student_id"]), str(r["question_unittest_id"])) for r in existing_rows}
            original_count = len(items)
            items = [it for it in items if (str(it.student_id), str(it.question_id)) not in completed]
            logger.info("Resume: %d completed, %d remaining (was %d).",
                        len(completed), len(items), original_count)

    n_chunks = max(1, (len(items) + chunk_size - 1) // chunk_size)

    for chunk_idx, chunk_start in enumerate(range(0, len(items), chunk_size)):
        chunk = items[chunk_start:chunk_start + chunk_size]
        label = f"[{model_key}] Chunk {chunk_idx + 1}/{n_chunks}"
        logger.info("========== %s: %d items ==========", label, len(chunk))

        chunk_results = _run_chunk(
            chunk, runner, model_key, label, max_attempts,
            batch_size,
            student_to_course, student_to_section, question_to_is_exam,
            persona_builder, history_summarizer, prompt_log_file,
            no_trajectory=no_trajectory,
        )
        all_results.extend(chunk_results)

        if output_dir:
            save_results(all_results, output_dir, model_key, max_attempts,
                         prepend_rows=existing_rows or None)
            logger.info("%s | Saved %d results", label,
                        len(all_results) + len(existing_rows))

    if hasattr(runner, "cleanup"):
        runner.cleanup()

    return all_results


def _run_chunk(
    chunk_items, runner, model_key, label, max_attempts,
    batch_size,
    student_to_course, student_to_section, question_to_is_exam,
    persona_builder, history_summarizer, prompt_log_file,
    no_trajectory=False,
):
    """Run the iterative attempt loop on a chunk of items."""
    # Parse test cases
    test_data = {}
    for idx, item in enumerate(chunk_items):
        tcs = parse_test_cases(item.question_unittests)
        if not tcs or not isinstance(item.question_template, str):
            test_data[idx] = None
            continue
        test_data[idx] = {"test_cases": tcs}

    # Init state
    results = [
        EvalResult(
            question_id=item.question_id, student_id=item.student_id,
            model=model_key,
            course_id=student_to_course.get(str(item.student_id), ""),
            section_id=student_to_section.get(str(item.student_id), ""),
            is_exam=question_to_is_exam.get(str(item.question_id), ""),
        )
        for item in chunk_items
    ]
    active = list(range(len(chunk_items)))
    conversations = {}  # idx -> {"system": str, "messages": [...]}
    error_indices = set()  # items dropped due to API errors (retried on resume)

    from .summarize import RAG_SUMMARY_PROMPT
    item_max_attempts = {}
    for idx, item in enumerate(chunk_items):
        real = getattr(item, "_real_attempts", None) or []
        item_max_attempts[idx] = min(len(real), max_attempts)

    for attempt in range(max_attempts):
        if not active:
            break

        active = [idx for idx in active if attempt < item_max_attempts.get(idx, 0)]
        if not active:
            break

        # ── Build prompts ──
        prompts_for_log = []
        user_prompts = []
        system_prompts = []
        conv_histories = []

        for idx in active:
            item = chunk_items[idx]

            if attempt == 0:
                rp = _build_initial_prompt(item, persona_builder)
                prompts_for_log.append(rp)
                if isinstance(rp, tuple):
                    sys_msg, user_msg = rp
                    conversations[idx] = {
                        "system": sys_msg,
                        "messages": [{"role": "user", "content": user_msg}],
                    }
                    system_prompts.append(sys_msg)
                    user_prompts.append(user_msg)
                    conv_histories.append(None)
                else:
                    system_prompts.append(None)
                    user_prompts.append(rp)
                    conv_histories.append(None)
            else:
                conv = conversations.get(idx)

                if no_trajectory:
                    # Only the real student's attempts are removed; the model
                    # still sees its own prior turns via the conversation.
                    feedback_msg = f"This is attempt {attempt + 1}. Predict what the student would submit."
                else:
                    # Show full student trajectory so far (attempts 0..t-1)
                    real = getattr(item, "_real_attempts", [])
                    parts = ["=== Student's Attempt Trajectory ===\n"]

                    for prev_t in range(attempt):
                        prev_real = real[prev_t] if prev_t < len(real) else None
                        if not prev_real:
                            continue
                        rtype = "Precheck" if prev_real["response_type"] == "Prechecked" else "Submit"
                        if history_summarizer:
                            summary = history_summarizer._call_llm(
                                f"{RAG_SUMMARY_PROMPT}\n\n"
                                f"Problem: {item.question_name}\n"
                                f"Test result: {prev_real['pass']}\n\n"
                                f"```cpp\n{prev_real['response'][:3000]}\n```"
                            )
                            parts.append(
                                f"Attempt {prev_t + 1} [{rtype}] -> {prev_real['pass']}\n"
                                f"Summary: {summary}\n"
                            )
                        else:
                            parts.append(
                                f"Attempt {prev_t + 1} [{rtype}] -> {prev_real['pass']}\n"
                            )

                    parts.append(
                        "Now predict what the student would submit next. "
                        "Match their coding style and debugging approach."
                    )
                    feedback_msg = "\n".join(parts)

                if conv:
                    conv["messages"].append({"role": "user", "content": feedback_msg})
                    prompts_for_log.append(("(multi-turn)", feedback_msg))
                    system_prompts.append(conv["system"])
                    user_prompts.append(feedback_msg)
                    conv_histories.append(conv["messages"])
                else:
                    prompts_for_log.append(feedback_msg)
                    system_prompts.append(None)
                    user_prompts.append(feedback_msg)
                    conv_histories.append(None)

        # ── Call LLM ──
        n_sub = max(1, (len(user_prompts) + batch_size - 1) // batch_size)
        logger.info("%s | Attempt %d/%d: %d active (%d sub-batches)",
                    label, attempt + 1, max_attempts, len(active), n_sub)

        all_responses = []
        for start in range(0, len(user_prompts), batch_size):
            all_responses.extend(runner.generate(
                user_prompts[start:start + batch_size],
                system_prompts=system_prompts[start:start + batch_size],
                conversations=conv_histories[start:start + batch_size],
            ))

        # Update conversation histories
        for i, resp in enumerate(all_responses):
            conv = conversations.get(active[i])
            if conv:
                conv["messages"].append({"role": "assistant", "content": resp})
                # Cap context growth: keep the first user message (full question
                # prompt) plus the last 8 messages. Trimming in whole user/
                # assistant pairs keeps the alternation valid.
                msgs = conv["messages"]
                if len(msgs) > 9:
                    conv["messages"] = [msgs[0]] + msgs[-8:]

        # ── Log ──
        if prompt_log_file:
            with open(prompt_log_file, "a") as logf:
                for i, (rp, resp) in enumerate(zip(prompts_for_log, all_responses)):
                    item = chunk_items[active[i]]
                    logf.write(f"\n{'='*70}\n")
                    logf.write(f"ATTEMPT {attempt+1} | Q={item.question_name} | S={item.student_id}\n")
                    logf.write(f"{'='*70}\n\n")
                    if isinstance(rp, tuple):
                        if rp[0] == "(multi-turn)":
                            logf.write(f"--- FEEDBACK ---\n{rp[1]}\n\n")
                        else:
                            logf.write(f"--- SYSTEM ---\n{rp[0]}\n\n--- USER ---\n{rp[1]}\n\n")
                    else:
                        logf.write(f"--- PROMPT ---\n{rp}\n\n")
                    logf.write(f"--- RESPONSE ---\n{resp}\n\n")

        # ── Parse actions + grade ──
        timestamp = time.strftime("%d/%m/%y, %H:%M:%S")
        actions = []
        codes = []
        for i, idx in enumerate(active):
            action = extract_action(all_responses[i])
            code = extract_code(all_responses[i])
            actions.append(action)
            codes.append(code)

        grade_jobs = {i: idx for i, idx in enumerate(active)
                      if test_data[idx] and codes[i]}

        if grade_jobs:
            logger.info("%s | Grading %d items", label, len(grade_jobs))

        grade_results = _grade_batch(grade_jobs, chunk_items, test_data, codes)

        # ── Process results ──
        next_active = []
        n_graded = n_passed = 0

        for i, idx in enumerate(active):
            raw = all_responses[i]
            td = test_data[idx]

            # Runner failure sentinel: don't record a fake all-tests-failed
            # attempt. Drop the item entirely so a resume retries it.
            if isinstance(raw, str) and raw.startswith("ERROR:"):
                error_indices.add(idx)
                continue

            if i in grade_results:
                n_graded += 1
                pp = "".join(str(x) for x in grade_results[i])
                passed = bool(pp) and all(c == "1" for c in pp)
            else:
                n_t = len(td["test_cases"]) if td else 0
                pp = "0" * n_t
                passed = False

            # "Prechecked" (not "Precheck") matches the response_type values in
            # the real submission data, keeping the output schema compatible.
            rtype = actions[i] if actions[i] == "Submit" else "Prechecked"
            results[idx].attempts.append(AttemptRecord(
                attempt, timestamp, rtype, user_prompts[i], raw, codes[i], pp))

            next_active.append(idx)
            if passed:
                n_passed += 1

        logger.info("%s | Attempt %d: %d graded, %d passed all, %d continue",
                    label, attempt + 1, n_graded, n_passed, len(next_active))
        active = next_active

    if error_indices:
        logger.warning("%s | %d items hit API errors; left unrecorded so a "
                       "resume retries them", label, len(error_indices))
        results = [r for idx, r in enumerate(results) if idx not in error_indices]

    n_all_pass = sum(1 for r in results
                     if any(a.pass_pattern and all(c == "1" for c in a.pass_pattern)
                            for a in r.attempts))
    logger.info("%s | Done! %d/%d passed all tests (%.0f%%)",
                label, n_all_pass, len(chunk_items),
                100.0 * n_all_pass / len(chunk_items) if chunk_items else 0)
    return results


# ── Output ──────────────────────────────────────────────────────────────────


def save_results(
    results: List[EvalResult], output_dir: str, model_key: str,
    max_attempts: int, prepend_rows: Optional[List[dict]] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_key}_attempts{max_attempts}.jsonl")
    with open(out_path, "w") as f:
        if prepend_rows:
            for row in prepend_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for r in results:
            for row in r.to_rows():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_total = (len(prepend_rows) if prepend_rows else 0) + sum(len(r.attempts) for r in results)
    logger.info("Saved %d rows → %s", n_total, out_path)
    return out_path
