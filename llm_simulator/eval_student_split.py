"""LLM student simulation using the student split evaluation framework.

Uses the same (data, split) loader as the psychometric models (IRT, CIRT,
BKT, DKT). With val_frac=0.0 the test split is identical to theirs, but the
train pool additionally includes their validation students.

Usage:
    python -m llm_simulator.eval_student_split --course dsa_hk231 --models haiku
    python -m llm_simulator.eval_student_split --course dsa_hk231 --models haiku --max_students 5 --max_questions 3 --dry_run
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynamic_models.temporal_eval.data_loader import load_student_split_data

from .prompts import build_prompt
from .rag import RAGRetriever
from .run import MODEL_CONFIGS, run_evaluation, save_results
from .data_loader import load_student_split_eval_items

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _format_question_metadata(difficulty) -> str:
    """Format item difficulty as natural language for the prompt."""
    parts = []
    if difficulty.topic:
        parts.append(f"Topic: {difficulty.topic}")
    parts.append(f"{difficulty.train_pass_rate:.0%} of students pass all tests on this problem.")
    if not np.isnan(difficulty.avg_attempts_to_pass):
        parts.append(f"Average {difficulty.avg_attempts_to_pass:.1f} attempts to solve.")
    parts.append(f"{difficulty.n_train_students_attempted} students attempted this question.")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="LLM simulation with student split")
    parser.add_argument("--course", type=str, default="dsa_hk231")
    parser.add_argument("--models", nargs="+", default=["haiku"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--max_attempts", type=int, default=10)
    parser.add_argument("--max_students", type=int, default=None)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--n_self", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/llm_student_eval")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_prompts", type=str, default=None,
                        help="Save all prompts and responses to this file")
    parser.add_argument("--no_persona", action="store_true")
    parser.add_argument("--no_summarize", action="store_true")
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--no_rag", action="store_true")
    parser.add_argument("--no_trajectory", action="store_true")
    parser.add_argument("--recent_questions", action="store_true",
                        help="Replace TF-IDF+recency RAG with most recent questions")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Override base_url for OpenAI-compatible server backends")
    parser.add_argument("--direct_solve", action="store_true",
                        help="Ablation: LLM solves each unique question directly, "
                             "with no student-specific context")
    args = parser.parse_args()

    if args.recent_questions and args.no_rag:
        parser.error("--recent_questions has no effect with --no_rag")
    if args.direct_solve and (args.no_persona or args.no_rag or args.no_metadata
                              or args.no_trajectory or args.recent_questions):
        parser.error("--direct_solve already removes all student context; "
                     "do not combine it with other ablation flags")
    if args.direct_solve:
        # Direct solve uses none of the student-context machinery; reuse the
        # existing skip paths.
        args.no_persona = args.no_rag = args.no_metadata = True
        args.no_summarize = True

    # 1. Load data (same as psychometric models)
    # val_frac=0: the simulator has no validation phase, and the default 0.15
    # would silently shrink the RAG/persona corpus and train_pass_rate stats.
    data, split = load_student_split_data(args.course, seed=args.seed, val_frac=0.0)

    # 2. Convert to EvalItems (all questions, all students)
    items, difficulties = load_student_split_eval_items(data, split, seed=args.seed)

    if args.direct_solve:
        # One item per unique question; the LLM plays itself, not a student.
        seen_questions = set()
        deduped = []
        for item in items:
            if item.question_id not in seen_questions:
                seen_questions.add(item.question_id)
                item.student_id = "direct_solve"
                item._real_attempts = []
                deduped.append(item)
        items = deduped
        logger.info("Direct solve: deduplicated to %d unique questions", len(items))

    # 3. Filter students first, then cap questions per student
    if args.max_students:
        unique_sids = list(set(it.student_id for it in items))
        if args.max_students < len(unique_sids):
            rng = np.random.RandomState(args.seed)
            keep = set(rng.choice(unique_sids, args.max_students, replace=False))
            items = [it for it in items if it.student_id in keep]

    if args.max_questions:
        rng = np.random.RandomState(args.seed + 1)
        filtered = []
        for sid in set(it.student_id for it in items):
            student_items = [it for it in items if it.student_id == sid]
            if len(student_items) > args.max_questions:
                indices = rng.choice(len(student_items), args.max_questions, replace=False)
                student_items = [student_items[i] for i in sorted(indices)]
            filtered.extend(student_items)
        items = filtered

    logger.info("Loaded %d eval items (%d students, %d questions)",
                len(items), len(set(it.student_id for it in items)),
                len(set(it.question_name for it in items)))

    # 3. Use main_data from the loaded split (already filtered to course)
    full_main_df = data.main_data
    hf_dir = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        repo_type="dataset", local_files_only=True,
    )
    question_infos = pd.read_csv(f"{hf_dir}/question_infos.csv")
    course_infos = pd.read_csv(f"{hf_dir}/course_infos.csv")

    # 4. Build RAG with student split boundary
    train_sids = set(str(data.student_ids[i]) for i in split.train_student_indices)
    test_sids = set(str(data.student_ids[i]) for i in split.test_student_indices)

    rag = None
    if not args.no_rag:
        rag = RAGRetriever.from_student_split(
            full_main_df, question_infos,
            train_student_ids=train_sids,
            test_student_ids=test_sids,
            train_week_cutoff=split.train_week_cutoff,
            qid_to_week=data.qid_to_week,
        )

    # 5. Build persona builder
    persona_builder = None
    if not args.no_persona:
        from .persona import PersonaBuilder
        section_infos = pd.read_csv(f"{hf_dir}/section_infos.csv")
        # Persona uses same filtered data as RAG
        train_mask = full_main_df["student_id"].astype(str).isin(train_sids)
        test_mask = full_main_df["student_id"].astype(str).isin(test_sids)
        test_week = full_main_df["question_unittest_id"].map(
            lambda q: data.qid_to_week.get(int(q), 99) if pd.notna(q) else 99
        )
        persona_df = full_main_df[train_mask | (test_mask & (test_week <= split.train_week_cutoff))]
        persona_builder = PersonaBuilder(
            persona_df, question_infos,
            course_infos=course_infos, section_infos=section_infos,
        )
        logger.info("PersonaBuilder: %d students", len(persona_builder))

    # 6. Init summarizer
    summarizer = None
    if not args.no_summarize:
        from .summarize import HistorySummarizer
        summarizer = HistorySummarizer()

    # Build question name -> HF row lookup for self-summary question descriptions
    hf_lookup = {}
    for _, row in question_infos.iterrows():
        hf_lookup[row["question_name"]] = row

    # 7. Pre-compute summaries for each item
    logger.info("Pre-computing self summaries...")
    item_self_summaries = {}
    item_metadata = {}

    # Collect RAG self-trajectories (fast, no LLM calls)
    item_selfs = {}
    if not args.no_rag:
        for i, item in enumerate(items):
            target_week = data.qid_to_week.get(int(item.question_id))
            if args.recent_questions:
                item_selfs[i] = rag.retrieve_recent_trajectories(
                    item.student_id, item.question_id,
                    max_self=args.n_self, target_week=target_week,
                )
            else:
                item_selfs[i] = rag.retrieve_self_trajectories(
                    item.student_id, item.question_id,
                    max_self=args.n_self, target_week=target_week,
                )

    if not args.no_metadata:
        for i, item in enumerate(items):
            diff = difficulties.get(item.question_name)
            if diff:
                item_metadata[i] = _format_question_metadata(diff)

    if summarizer:
        from .summarize import SELF_SUMMARY_PROMPT, QUESTION_SUMMARY_PROMPT

        all_prompts = []
        prompt_map = []  # (item_idx, type, sub_idx)

        for i, item in enumerate(items):
            # Self prompts (question summary + approach)
            for j, s in enumerate(item_selfs.get(i, [])):
                hf_row = hf_lookup.get(s.question_name)
                q_text_str = str(hf_row["question_text"]) if hf_row is not None else ""
                q_prompt = f"{QUESTION_SUMMARY_PROMPT}\n\nProblem: {s.question_name}\n{q_text_str[:1000]}"
                all_prompts.append(q_prompt)
                prompt_map.append((i, "self_q", j))
                attempts_block = []
                for k, a in enumerate(s.attempts, 1):
                    rtype = "Precheck" if a["response_type"] == "Prechecked" else "Submit"
                    attempts_block.append(f"Attempt {k} [{rtype}] result: {a['pass_pattern']}\n{a['response']}")
                a_prompt = f"{SELF_SUMMARY_PROMPT}\n\nProblem: {s.question_name}\n\n" + "\n\n".join(attempts_block)
                all_prompts.append(a_prompt)
                prompt_map.append((i, "self_a", j))

            # Target question summary
            q_prompt = f"{QUESTION_SUMMARY_PROMPT}\n\nProblem: {item.question_name}\n{item.question_text[:1000]}"
            all_prompts.append(q_prompt)
            prompt_map.append((i, "target_q", 0))

        logger.info("Batch summarizing %d prompts...", len(all_prompts))
        all_results = summarizer.batch_call_llm(all_prompts, max_workers=20)

        self_q_results = {}
        self_a_results = {}
        target_q_results = {}

        for idx, (i, typ, j) in enumerate(prompt_map):
            r = all_results[idx] or ""
            if typ == "self_q":
                self_q_results[(i, j)] = r
            elif typ == "self_a":
                self_a_results[(i, j)] = r
            elif typ == "target_q":
                target_q_results[i] = r

        for i, item in enumerate(items):
            selfs = item_selfs.get(i, [])
            if selfs:
                entries = []
                for j, s in enumerate(selfs):
                    q_sum = self_q_results.get((i, j), "")
                    approach = self_a_results.get((i, j), "")
                    entries.append(f"[{s.question_name}] {q_sum}\nStudent's approach: {approach}")
                item_self_summaries[i] = entries

            item._question_summary = target_q_results.get(i)
    else:
        for i, item in enumerate(items):
            selfs = item_selfs.get(i, [])
            if selfs:
                item_self_summaries[i] = [
                    f"[{s.question_name}] {len(s.attempts)} attempts" for s in selfs
                ]
            item._question_summary = None

    logger.info("Summaries complete: %d self, %d metadata",
                len(item_self_summaries), len(item_metadata))

    # 8. Inject summaries into items
    for i, item in enumerate(items):
        item._self_summaries = item_self_summaries.get(i, [])
        item._question_metadata = item_metadata.get(i)

    # 9. Dry run: show sample prompts
    if args.dry_run:
        for item in items[:2]:
            persona_text = None
            if persona_builder and item.student_id:
                persona_text = persona_builder.build_persona_text(item.student_id)

            prompt = build_prompt(
                question_name=item.question_name,
                question_text=item.question_text,
                question_template=item.question_template,
                persona_text=persona_text,
                self_summaries=getattr(item, "_self_summaries", None),
                question_metadata=getattr(item, "_question_metadata", None),
                question_summary=getattr(item, "_question_summary", None),
            )

            if isinstance(prompt, tuple):
                sys_msg, user_msg = prompt
                print(f"\n{'='*60}")
                print(f"Item: Q={item.question_name}, S={item.student_id}")
                print(f"{'='*60}")
                print(f"=== SYSTEM ({len(sys_msg)} chars) ===")
                print(sys_msg[:500])
                print(f"\n=== USER ({len(user_msg)} chars) ===")
                print(user_msg[:2000])
            else:
                print(f"\n{'='*60}")
                print(f"Item: Q={item.question_name}, S={item.student_id}")
                print(prompt[:2000])

        logger.info("Dry run complete. %d items loaded.", len(items))
        return

    # 10. Run models
    student_info = full_main_df[["student_id", "course_id", "section_id"]].drop_duplicates(subset=["student_id"])
    student_to_course = dict(zip(student_info["student_id"].astype(str), student_info["course_id"].astype(str)))
    student_to_section = dict(zip(student_info["student_id"].astype(str), student_info["section_id"].astype(str)))
    question_info = full_main_df[["question_unittest_id", "is_exam"]].drop_duplicates(subset=["question_unittest_id"])
    question_to_is_exam = dict(zip(question_info["question_unittest_id"].astype(str), question_info["is_exam"].astype(str)))

    output_dir = os.path.join(args.output_dir, args.course)

    # Route ablation runs to subdirectories (all active flags joined)
    flags = [name for name, on in [
        ("no_persona", args.no_persona),
        ("no_rag", args.no_rag),
        ("no_metadata", args.no_metadata),
        ("no_trajectory", args.no_trajectory),
        ("no_summarize", args.no_summarize),
        ("recent_questions", args.recent_questions),
    ] if on]
    if args.direct_solve:
        # Matches the existing results directory for this condition.
        output_dir = os.path.join(output_dir, "direct_solve")
    elif flags:
        output_dir = os.path.join(output_dir, "ablation_" + "_".join(flags))

    for model_key in args.models:
        logger.info("=== Running %s ===", model_key)
        try:
            results = run_evaluation(
                items, model_key,
                max_attempts=args.max_attempts,
                student_to_course=student_to_course,
                student_to_section=student_to_section,
                question_to_is_exam=question_to_is_exam,
                output_dir=output_dir,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                persona_builder=persona_builder,
                history_summarizer=summarizer,
                prompt_log_file=args.log_prompts,
                no_trajectory=args.no_trajectory,
                base_url=args.base_url,
                direct_solve=args.direct_solve,
            )
        except Exception as e:
            logger.error("Model %s failed: %s", model_key, e, exc_info=True)

    logger.info("Done!")


if __name__ == "__main__":
    main()
