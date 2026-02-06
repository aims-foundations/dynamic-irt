import copy
import json
import pickle
import random
import re
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import (
    EXAM_INST,
    FEEDBACK_INST,
    FIRST_ATTEMPT_PREFIX,
    MAX_QUES_PER_WEEK,
    MAX_TRIALS_PER_QUES,
    QUESTION_INST,
    RESPONSE_INST,
    RETRY_END_TAG,
    RETRY_PREFIX,
    RETRY_START_TAG,
    START_WEEK,
    STOP_ANSWER,
    SYSTEM_PROMPT,
    USER_TAG,
)

from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

from utils import (
    check_health,
    compute_ed,
    ensure_dir,
    format_prompt,
    get_cpp_evaluator,
    get_model_api,
    infer_completion_api,
    infer_score,
    parse_replace_string,
    parse_score_from_feedback,
    restore_code_answer,
    run_server,
    set_seed,
    shutdown_server,
)


def parse_question(instruction):
    questions = instruction.split("\n\nQuestion ")[1:]
    return_questions = {}
    for q in questions:
        idx = q[: q.find(":")]
        return_questions[idx] = q[q.find(":") + 2 :].strip().replace(RESPONSE_INST, "")

    return return_questions


def parse_question_name(question_names):
    return_names = {}
    for qidx, q in enumerate(question_names):
        if "|" in q:
            names = q.split("|")
            for sub_idx, name in enumerate(names):
                return_names[f"{qidx+1}.{sub_idx+1}"] = name
        else:
            return_names[str(qidx + 1)] = q

    return return_names


def choose_init_student(dataset):
    student_idxs = []
    for idx, hist in enumerate(ds["history"]):
        if len(hist) == 0:
            student_idxs.append(idx)

    student_idxs.append(len(dataset))
    list_stu_scores = []

    # Choose a student
    for start_student_idx, stop_student_idx in tqdm(
        zip(student_idxs[:-1], student_idxs[1:]), desc="Chosing student"
    ):
        actual_exam_scores = []
        exam_flag = False
        student_ds = dataset.select(list(range(start_student_idx, stop_student_idx)))

        for sample in student_ds:
            if sample["week"] < START_WEEK:
                continue

            if (
                sample["instruction"].startswith("Your score")
                and QUESTION_INST in sample["instruction"]
            ):
                if sample["week"] > START_WEEK:
                    exc = parse_score_from_feedback(sample["instruction"])

                    if ex_ques_idx in actual_exam_scores[-1]:
                        if exc > actual_exam_scores[-1][ex_ques_idx]:
                            actual_exam_scores[-1][ex_ques_idx] = exc
                    else:
                        actual_exam_scores[-1][ex_ques_idx] = exc
                    ex_ques_idx = sample["output"].split(":")[0].split(" ")[-1]
                    exam_flag = False

            elif sample["instruction"].startswith("Your score") and exam_flag:
                exc = parse_score_from_feedback(sample["instruction"])

                if ex_ques_idx in actual_exam_scores[-1]:
                    if exc > actual_exam_scores[-1][ex_ques_idx]:
                        actual_exam_scores[-1][ex_ques_idx] = exc
                else:
                    actual_exam_scores[-1][ex_ques_idx] = exc
                ex_ques_idx = sample["output"].split(":")[0].split(" ")[-1]

            if EXAM_INST in sample["instruction"] and sample["week"] > START_WEEK:
                ex_ques_idx = sample["output"].split(":")[0].split(" ")[-1]
                if len(ex_ques_idx):
                    exam_flag = True
                    actual_exam_scores.append({})

        # Compute score student with min exam score
        min_score = 1
        for x in actual_exam_scores[0].values():
            if x < min_score and x > 0.1:
                min_score = x
        list_stu_scores.append(min_score)

    # Choose the student with the lowest average score
    print("Student has min exam score:", min(list_stu_scores))
    stu_idx = np.argmin(list_stu_scores)
    return student_idxs[stu_idx], student_idxs[stu_idx + 1]


def get_answer_and_score(
    trial,
    qid,
    history,
    evaluator,
    client,
    model,
    generation_config,
    best_answer="",
    last_answer="",
):
    # Format prompt for getting answer
    prompt = format_prompt(SYSTEM_PROMPT, history, tokenizer, is_answer=True)

    # Add some hints for generation
    if trial == 0:
        postfix = FIRST_ATTEMPT_PREFIX + f"{qid}:\n" + best_answer[:50]
    else:
        postfix = RETRY_PREFIX + f"{qid}:\n" + RETRY_START_TAG
    prompt += postfix

    # Infer student's answer
    completions = infer_completion_api(prompt, client, model, generation_config, n=32)
    chosen_answer = None
    chosen_score = None
    chosen_code = None

    for answer in completions:
        # spliting_idx = answer.find(":")
        # if spliting_idx < 1:
        #     continue
        # ques_idx = int(answer[:spliting_idx].strip())
        # if ques_idx != round_idx+1:
        #     continue
        # clean_answer = answer[spliting_idx + 1 :].strip()
        if trial == 0:
            clean_answer = best_answer[:50] + answer
        else:
            clean_answer = restore_code_answer(last_answer, RETRY_START_TAG + answer)

        ans_score = infer_score(evaluator, clean_answer)
        if chosen_score is None or ans_score > chosen_score:
            chosen_answer = answer
            chosen_score = ans_score
            chosen_code = clean_answer

    return postfix + chosen_answer, chosen_code, chosen_score


if __name__ == "__main__":
    # wandb.init()
    parser = ArgumentParser()
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    parser.add_argument("--cls", type=str, default="CC01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=9820)
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(f"results/{args.cls}")

    server_process = run_server(
        f"vllm serve saves/All --tensor-parallel-size 4 --port {args.port}"
    )
    check_health(f"http://localhost:{args.port}/health")
    client, model, tokenizer, generation_config = get_model_api(
        f"saves/All", port=args.port
    )

    ds = load_dataset(
        f"stair-lab/{args.course_name}_v3_per_student_sft_lf_splited",
        split=args.cls + "_test",
    )

    # Load question infos to construct Autograding system
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}_wtc", repo_type="dataset"
    )
    question_infos = pickle.load(open(f"{data_folder}/unique_questions.pkl", "rb"))
    question_name2idx = pickle.load(open(f"{data_folder}/question_name2idx.pkl", "rb"))
    list_best_answers = pickle.load(open(f"data/{args.cls}/best_answers.pkl", "rb"))
    list_evaluators = {}

    start_student_idx, stop_student_idx = choose_init_student(ds)
    student_ds = ds.select(list(range(start_student_idx, stop_student_idx)))
    pickle.dump(student_ds, open(f"results/{args.cls}/student_ds.pkl", "wb"))

    student_history = None
    list_questions = []
    list_exams = []
    actual_exam_scores = []
    optimized_exam_scores = []

    # Pick history at START_WEEK
    exam_flag = False
    ex_ques_idx = None
    for sample in student_ds:
        if sample["week"] < START_WEEK:
            continue

        if (
            sample["instruction"].startswith("Your score")
            and QUESTION_INST in sample["instruction"]
        ):
            if sample["week"] == START_WEEK:
                # Here is the history
                student_history = sample["history"]

                # Add previous exam score
                student_history.append(
                    [
                        sample["instruction"].split(QUESTION_INST)[0],
                        None,
                    ]
                )
                list_questions.append(
                    [
                        parse_question(sample["instruction"]),
                        parse_question_name(sample["question_name"]),
                    ]
                )
                list_evaluators.update(
                    get_cpp_evaluator(
                        sample["question_name"], question_name2idx, question_infos
                    )
                )

            elif sample["week"] > START_WEEK:
                list_questions.append(
                    [
                        parse_question(sample["instruction"]),
                        parse_question_name(sample["question_name"]),
                    ]
                )
                list_evaluators.update(
                    get_cpp_evaluator(
                        sample["question_name"], question_name2idx, question_infos
                    )
                )

                exc = parse_score_from_feedback(sample["instruction"])
                if ex_ques_idx in actual_exam_scores[-1]:
                    if exc > actual_exam_scores[-1][ex_ques_idx]:
                        actual_exam_scores[-1][ex_ques_idx] = exc
                else:
                    actual_exam_scores[-1][ex_ques_idx] = exc
                ex_ques_idx = sample["output"].split(":")[0].split(" ")[-1]
                exam_flag = False

        elif sample["instruction"].startswith("Your score") and exam_flag:
            exc = parse_score_from_feedback(sample["instruction"])
            if ex_ques_idx in actual_exam_scores[-1]:
                if exc > actual_exam_scores[-1][ex_ques_idx]:
                    actual_exam_scores[-1][ex_ques_idx] = exc
            else:
                actual_exam_scores[-1][ex_ques_idx] = exc
            ex_ques_idx = sample["output"].split(":")[0].split(" ")[-1]

        if EXAM_INST in sample["instruction"] and sample["week"] > START_WEEK:
            list_exams.append(
                [
                    parse_question(sample["instruction"]),
                    parse_question_name(sample["question_name"]),
                ]
            )
            list_evaluators.update(
                get_cpp_evaluator(
                    sample["question_name"], question_name2idx, question_infos
                )
            )

            ex_ques_idx = sample["output"].split(":")[0].split(" ")[-1]
            exam_flag = True
            actual_exam_scores.append({})

    # Now we have the student history, questions, and exams
    # student_history: List of [user, assistant]
    # list_questions: List of [questions, question_names].
    # -- Each question is a dict of question index and question content.
    # -- Each question_name is a dict of question index and question name.
    # list_exams: List of [questions, question_names].
    # -- Each question is a dict of question index and question content.
    # -- Each question_name is a dict of question index and question name.

    # Perform search
    print("Total optimized week:", min(len(list_questions), len(list_exams)))
    for wi, (w_ques, ex_ques) in enumerate(zip(list_questions, list_exams)):  # Week
        # w_ques: [questions, question_names]
        # ex_ques: [questions, question_names]
        print(f"Searching on week {START_WEEK+wi}...")

        # Filter an exam question that student has low score
        min_ex_score = 1
        chosen_ex_ques = None
        for exid, ex_score in actual_exam_scores[wi].items():
            if ex_score < min_ex_score and ex_score > 0.1:
                min_ex_score = ex_score
                chosen_ex_ques = exid
        ex_ques_content = {k: v for k, v in ex_ques[0].items() if k == chosen_ex_ques}
        ex_names = {k: v for k, v in ex_ques[1].items() if k == chosen_ex_ques}
        ex_ques = [ex_ques_content, ex_names]

        num_exam_ques = len(set([x.split(".")[0] for x in ex_ques[1].keys()]))

        print("Week question:", w_ques[1].keys())
        print("Exam question:", ex_ques[1].keys())
        print("Num exam question:", num_exam_ques)
        print("Actual exam scores:", min_ex_score)

        # Multi practice rounds for each week
        for round_idx in range(MAX_QUES_PER_WEEK):
            list_trajs = {}
            list_scores = {}

            # ==========================================================================

            def process_question(qidx, question, question_name):
                hist = copy.deepcopy(student_history)
                hist[-1][0] = (
                    hist[-1][0]
                    + f"\n\n{QUESTION_INST}\n\n"
                    + f"Question {round_idx+1}: "
                    + question
                    + RESPONSE_INST
                )

                last_answer = ""
                for trial in range(MAX_TRIALS_PER_QUES):
                    print("Trial:", trial)

                    # Infer and append answer to history
                    chosen_answer, complete_code, chosen_score = get_answer_and_score(
                        trial=trial,
                        qid=round_idx + 1,
                        history=hist,
                        best_answer=list_best_answers[question_name],
                        last_answer=last_answer,
                        evaluator=list_evaluators[question_name],
                        client=client,
                        model=model,
                        generation_config=generation_config,
                    )
                    last_answer = complete_code
                    hist[-1][1] = chosen_answer

                    # Format prompt for scoring
                    hist.append(["", ""])

                    # Infer student's score
                    print(f"Trial Score ({question_name}):", chosen_score)
                    hist[-1][0] = FEEDBACK_INST.format(score=round(chosen_score, 2))

                if round_idx < MAX_QUES_PER_WEEK - 1:
                    list_trajs[qidx] = copy.deepcopy(hist)

                # Force student to take exam (imagination)
                hist[-1][0] = (
                    hist[-1][0]
                    + f"\n\n{EXAM_INST}\n\n"
                    + "\n\n".join(
                        [f"Question {eqi}: " + q for eqi, q in ex_ques[0].items()]
                    )
                    + RESPONSE_INST
                )
                exam_scores = {}

                last_ex_answer = {}
                for trial in range(MAX_TRIALS_PER_QUES * num_exam_ques):
                    print("Exam trial:", trial)
                    ex_ques_idx = random.choice(list(ex_ques[0].keys()))
                    exam_question_name = ex_ques[1][ex_ques_idx]

                    chosen_ex_answer, complete_code, chosen_ex_score = (
                        get_answer_and_score(
                            trial=trial,
                            qid=ex_ques_idx,
                            history=hist,
                            best_answer=list_best_answers[exam_question_name],
                            last_answer=(
                                last_ex_answer[ex_ques_idx]
                                if ex_ques_idx in last_ex_answer
                                else None
                            ),
                            evaluator=list_evaluators[exam_question_name],
                            client=client,
                            model=model,
                            generation_config=generation_config,
                        )
                    )
                    last_ex_answer[ex_ques_idx] = complete_code
                    chosen_ex_ques_idx = ex_ques_idx
                    chosen_ex_ques_name = exam_question_name

                    hist[-1][1] = chosen_ex_answer

                    # Format prompt for scoring
                    hist.append(["", ""])

                    # Infer student's score
                    print(
                        "Chosing exam question:",
                        chosen_ex_ques_idx,
                        chosen_ex_ques_name,
                    )
                    print(f"Exam Score ({chosen_ex_ques_name}):", chosen_ex_score)
                    hist[-1][0] = FEEDBACK_INST.format(score=round(chosen_ex_score, 2))

                    # Update exam scores
                    __chosen_ex_score = (
                        chosen_ex_score + chosen_score / MAX_QUES_PER_WEEK
                    )
                    if chosen_ex_ques_idx in exam_scores:
                        if chosen_ex_score > exam_scores[chosen_ex_ques_idx]:
                            exam_scores[chosen_ex_ques_idx] = chosen_ex_score
                            exam_scores[chosen_ex_ques_idx + "combined"] = (
                                __chosen_ex_score
                            )
                    else:
                        exam_scores[chosen_ex_ques_idx] = chosen_ex_score
                        exam_scores[chosen_ex_ques_idx + "combined"] = __chosen_ex_score

                # Update results of completed question
                list_scores[qidx] = exam_scores
                if round_idx == MAX_QUES_PER_WEEK - 1:
                    list_trajs[qidx] = copy.deepcopy(hist)

            # ==========================================================================
            # Run all questions parallelly
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [
                    executor.submit(
                        process_question,
                        qidx,
                        w_ques[0][qidx],  # question
                        w_ques[1][qidx],  # question_name
                    )
                    for qidx in list(w_ques[0].keys())
                ]
                for future in tqdm(futures, desc="Question"):
                    future.result()

            # Choose the best trajectory
            best_scores = None
            best_traj = None

            for qidx in w_ques[0].keys():
                traj = list_trajs[qidx]
                scores = list_scores[qidx]
                score = sum(
                    [v for k, v in scores.items() if not k.endswith("_combined")]
                )
                if best_scores is not None:
                    best_score = sum(
                        [
                            v
                            for k, v in best_scores.items()
                            if not k.endswith("_combined")
                        ]
                    )
                else:
                    best_score = None

                if best_score is None or score > best_score:
                    best_scores = scores
                    best_traj = traj

            # Continue with best trajectory and best scores
            student_history = best_traj
            optimized_exam_scores.append(best_scores)

            # Save student history
            json.dump(
                best_traj,
                open(
                    f"results/{args.cls}/week{START_WEEK+wi}_ques{round_idx+1}.json",
                    "w",
                ),
            )

            print(
                f"Best optimzied score for week {START_WEEK + wi}:",
                best_scores,
            )
            print(
                f"Actual score for week {START_WEEK + wi}:",
                actual_exam_scores[wi],
            )

            # Save exam scores
            json.dump(
                list_scores,
                open(
                    f"results/{args.cls}/week{START_WEEK+wi}_round{round_idx}.json", "w"
                ),
            )
            json.dump(
                {
                    "actual": actual_exam_scores[wi],
                    "optimized": best_scores,
                },
                open(
                    f"results/{args.cls}/week{START_WEEK+wi}__round{round_idx}_exam.json",
                    "w",
                ),
            )

    print("Actual exam scores:")
    print(actual_exam_scores)
    print("Optimized exam scores:")
    print(optimized_exam_scores)

    with open(f"results/{args.cls}/final_results.json", "w") as f:
        json.dump({"actual": actual_exam_scores, "optimized": optimized_exam_scores}, f)

    shutdown_server(server_process)
    # wandb.finish()
