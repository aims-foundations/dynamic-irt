import copy
import json
import os
import pickle
import random
import re
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import wandb
from datasets import load_dataset
from grading_engine.engine import CPPEvaluator
from huggingface_hub import snapshot_download
from Levenshtein import distance
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from utils import ensure_dir, set_seed

# Modify OpenAI's API key and API base to use vLLM's API server.
START_WEEK = 3
MAX_QUES_PER_WEEK = 3
MAX_TRIALS_PER_QUES = 10

openai_api_key = "EMPTY"
openai_api_base = f"http://localhost:{os.environ.get('PORT', 8000)}/v1"
pattern = r"\s*([0-9]*\.?[0-9]+)\/[0-9]+"
USER_TAG = "<|start_header_id|>user<|end_header_id|>\n\n"
STOP_ANSWER = ["<|eot_id|>", "Your score"]
STOP_EVAL = ["<|eot_id|>", "Answer for"]

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id
tokenizer = AutoTokenizer.from_pretrained(
    model,
    use_fast=True,
)
generation_config = GenerationConfig.from_pretrained(model)


def format_prompt(system, history, is_answer=True):
    conversation = [{"role": "system", "content": system}]
    for hist in history[:-1]:
        conversation.append({"role": "user", "content": hist[0]})
        conversation.append({"role": "assistant", "content": hist[1]})
    if is_answer:
        conversation.append({"role": "user", "content": history[-1][0]})
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=is_answer
    )
    if not is_answer:
        prompt = prompt + USER_TAG
    return prompt


def parse_question(instruction):
    questions = instruction.split("\n\nQuestion ")[1:]
    return_questions = {}
    for q in questions:
        idx = q[: q.find(":")]
        return_questions[idx] = (
            q[q.find(":") + 2 :]
            .strip()
            .replace("Please write your answer for the above question in C++.", "")
        )

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


def infer_completion(prompt, is_answer=True, n=1):
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        # temperature=generation_config.temperature,
        temperature=1,
        # top_p=generation_config.top_p,
        top_p=1,
        stop=STOP_ANSWER if is_answer else STOP_EVAL,
        echo=False,
        n=n,
        best_of=32,
        logprobs=1,
        max_tokens=1024 if is_answer else 20,
        stream=False,
        timeout=1800,
    )
    return completion


def get_question_info_by_name(question_name, question_name2idx, question_infos):
    return question_infos[question_name2idx[question_name]]


def get_cpp_evaluator(question_names, *args, **kwargs):
    list_evaluators = {}
    for combine_name in question_names:
        names = combine_name.split("|")
        for name in names:
            content, template, testcases = get_question_info_by_name(
                name, *args, **kwargs
            )
            evaluator = CPPEvaluator(template, testcases, max_workers=4)
            list_evaluators[name] = evaluator
    return list_evaluators


def infer_score(evaluator, answer):
    student_results = evaluator.evaluate(answer)
    return student_results["score"]


def choose_init_student(dataset):
    student_idxs = []
    for idx, hist in enumerate(ds["history"]):
        if len(hist) == 0:
            student_idxs.append(idx)

    student_idxs.append(len(dataset))

    list_avg_scores = []
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
                and "Here are the exercise questions" in sample["instruction"]
            ):
                if sample["week"] > START_WEEK:
                    exc = float(
                        re.search(
                            r"Your score:\s*([0-9]*\.?[0-9]+)\/[0-9]+",
                            sample["instruction"],
                        ).group(1)
                    )

                    if ex_ques_idx in actual_exam_scores[-1]:
                        if exc > actual_exam_scores[-1][ex_ques_idx]:
                            actual_exam_scores[-1][ex_ques_idx] = exc
                    else:
                        actual_exam_scores[-1][ex_ques_idx] = exc
                    ex_ques_idx = sample["output"][20 : sample["output"].find(":")]
                    exam_flag = False

            elif sample["instruction"].startswith("Your score") and exam_flag:
                exc = float(
                    re.search(
                        r"Your score:\s*([0-9]*\.?[0-9]+)\/[0-9]+",
                        sample["instruction"],
                    ).group(1)
                )
                if ex_ques_idx in actual_exam_scores[-1]:
                    if exc > actual_exam_scores[-1][ex_ques_idx]:
                        actual_exam_scores[-1][ex_ques_idx] = exc
                else:
                    actual_exam_scores[-1][ex_ques_idx] = exc
                ex_ques_idx = sample["output"][20 : sample["output"].find(":")]

            if (
                "Here are the exam questions." in sample["instruction"]
                and sample["week"] > START_WEEK
            ):
                ex_ques_idx = sample["output"][20 : sample["output"].find(":")]
                exam_flag = True
                actual_exam_scores.append({})

        # Compute score avg score
        avg_score = sum([sum(x.values()) / len(x) for x in actual_exam_scores]) / len(
            actual_exam_scores
        )
        list_avg_scores.append(avg_score)

    # Choose the student with the lowest average score
    print("Student has avg exam score:", min(list_avg_scores))
    stu_idx = list_avg_scores.index(min(list_avg_scores))
    return student_idxs[stu_idx], student_idxs[stu_idx + 1]


if __name__ == "__main__":
    # wandb.init()
    parser = ArgumentParser()
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    parser.add_argument("--cls", type=str, default="CC01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(f"results/{args.cls}")

    ds = load_dataset(
        f"stair-lab/{args.course_name}_wtc_per_student_sft_lf_splited",
        split=args.cls + "_test",
    )

    # Load question infos to construct Autograding system
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}_wtc", repo_type="dataset"
    )
    question_infos = pickle.load(open(f"{data_folder}/unique_questions.pkl", "rb"))
    question_name2idx = pickle.load(open(f"{data_folder}/question_name2idx.pkl", "rb"))
    list_evaluators = {}

    start_student_idx, stop_student_idx = choose_init_student(ds)
    student_ds = ds.select(list(range(start_student_idx, stop_student_idx)))
    pickle.dump(student_ds, open(f"results/{args.cls}/student_ds.pkl", "wb"))

    student_system = ""
    student_history = None
    list_questions = []
    list_exams = []
    actual_exam_scores = []
    optimized_exam_scores = []

    # Pick history at week 3
    exam_flag = False
    ex_ques_idx = None
    for sample in student_ds:
        if sample["week"] < START_WEEK:
            continue

        if (
            sample["instruction"].startswith("Your score")
            and "Here are the exercise questions" in sample["instruction"]
        ):
            if sample["week"] == START_WEEK:
                # Here is the history
                student_history = sample["history"]
                student_system = sample["system"]

                # Add previous exam score
                student_history.append(
                    [
                        sample["instruction"].split("Here are the exercise questions")[
                            0
                        ],
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

                exc = float(
                    re.search(
                        r"Your score:\s*([0-9]*\.?[0-9]+)\/[0-9]+",
                        sample["instruction"],
                    ).group(1)
                )

                if ex_ques_idx in actual_exam_scores[-1]:
                    if exc > actual_exam_scores[-1][ex_ques_idx]:
                        actual_exam_scores[-1][ex_ques_idx] = exc
                else:
                    actual_exam_scores[-1][ex_ques_idx] = exc
                ex_ques_idx = sample["output"][20 : sample["output"].find(":")]
                exam_flag = False

        elif sample["instruction"].startswith("Your score") and exam_flag:
            exc = float(
                re.search(
                    r"Your score:\s*([0-9]*\.?[0-9]+)\/[0-9]+", sample["instruction"]
                ).group(1)
            )
            if ex_ques_idx in actual_exam_scores[-1]:
                if exc > actual_exam_scores[-1][ex_ques_idx]:
                    actual_exam_scores[-1][ex_ques_idx] = exc
            else:
                actual_exam_scores[-1][ex_ques_idx] = exc
            ex_ques_idx = sample["output"][20 : sample["output"].find(":")]

        if (
            "Here are the exam questions." in sample["instruction"]
            and sample["week"] > START_WEEK
        ):
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

            ex_ques_idx = sample["output"][20 : sample["output"].find(":")]
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
    print("Total optimized week:", min(len(list_questions[:-1]), len(list_exams)))
    for w_ques, ex_ques in zip(list_questions[:-1], list_exams):  # Week
        # w_ques: [questions, question_names]
        # ex_ques: [questions, question_names]
        print(f"Searching on week {len(optimized_exam_scores)+1}...")
        num_exam_ques = len(set([x.split(".")[0] for x in ex_ques[1].keys()]))

        print("Week question:", w_ques[1].keys())
        print("Exam question:", ex_ques[1].keys())
        print("Num exam question:", num_exam_ques)

        # Multi practice rounds for each week
        for round_idx in range(MAX_QUES_PER_WEEK):
            list_trajs = {}
            list_scores = {}

            def process_question(qidx, question, question_name):
                hist = copy.deepcopy(student_history)
                hist[-1][0] = (
                    hist[-1][0]
                    + "\n\nHere are the exercise questions for practice.\n\n"
                    + f"Question {round_idx+1}: "
                    + question
                    + "Please write your answer for the above question in C++."
                )

                for trial in range(MAX_TRIALS_PER_QUES):
                    print("Trial:", trial)
                    # Format prompt for getting answer
                    prompt = format_prompt(student_system, hist, is_answer=True)
                    prompt += "Answer for question "

                    # Infer student's answer
                    completions = infer_completion(prompt, is_answer=True, n=32)
                    chosen_answer = None
                    chosen_score = None

                    # dist_mat = [distance(a.text,b.text) for a,b in zip(completions.choices[:-1], completions.choices[1:])]
                    # print("Answer distance:", dist_mat)
                    for completion in completions.choices:
                        answer = completion.text

                        spliting_idx = answer.find(":")
                        if spliting_idx < 1:
                            continue

                        clean_answer = answer[spliting_idx + 1 :].strip()
                        ans_score = infer_score(
                            list_evaluators[question_name], clean_answer
                        )

                        if chosen_score is None or ans_score > chosen_score:
                            chosen_answer = answer
                            chosen_score = ans_score

                    # answer = random.choice(completions.choices).text
                    hist[-1][1] = "Answer for question " + chosen_answer

                    # Format prompt for scoring
                    hist.append(["", ""])

                    # Infer student's score
                    print(f"Trial Score ({question_name}):", chosen_score)
                    hist[-1][0] = f"Your score: {chosen_score}/1."

                if round_idx < MAX_QUES_PER_WEEK - 1:
                    list_trajs[qidx] = copy.deepcopy(hist)

                # Force student to take exam (imagination)
                hist[-1][0] = (
                    "Here are the exam questions.\n\n"
                    + "\n\n".join(
                        [f"Question {eqi}: " + q for eqi, q in ex_ques[0].items()]
                    )
                    + "Please write your answer for the above question in C++."
                )
                exam_scores = {}

                for trial in range(MAX_TRIALS_PER_QUES * num_exam_ques):
                    print("Exam trial:", trial)

                    # Format prompt for getting answer
                    prompt = format_prompt(student_system, hist, is_answer=True)
                    prompt += "Answer for question "

                    # Infer student's answer
                    completions = infer_completion(prompt, is_answer=True, n=32)
                    chosen_ex_ques_idx = None
                    chosen_ex_ques_name = None
                    chosen_ex_answer = None
                    chosen_ex_score = None
                    for completion in completions.choices:
                        ex_answer = completion.text

                        spliting_idx = ex_answer.find(":")
                        ex_ques_idx = ex_answer[:spliting_idx].strip()
                        if ex_ques_idx not in ex_ques[1]:
                            continue

                        clean_ex_answer = ex_answer[spliting_idx + 1 :].strip()
                        exam_question_name = ex_ques[1][ex_ques_idx]
                        ex_score = infer_score(
                            list_evaluators[exam_question_name], clean_ex_answer
                        )

                        if chosen_ex_score is None or ex_score > chosen_ex_score:
                            chosen_ex_ques_idx = ex_ques_idx
                            chosen_ex_ques_name = exam_question_name
                            chosen_ex_answer = ex_answer
                            chosen_ex_score = ex_score

                    if chosen_ex_score is None:
                        # If no question found, assume the model answers first question
                        print("Could not find exam question idx")
                        chosen_ex_ques_idx = random.choice(list(ex_ques[1].values()))
                        chosen_ex_ques_name = ex_ques[1][chosen_ex_ques_idx]
                        chosen_ex_answer = random.choice(completions.choices).text
                        spliting_idx = chosen_ex_answer.find(":")
                        clean_ex_answer = chosen_ex_answer[spliting_idx + 1 :].strip()
                        chosen_ex_score = infer_score(
                            list_evaluators[chosen_ex_ques_name], clean_ex_answer
                        )

                    hist[-1][1] = "Answer for question " + chosen_ex_answer

                    # Format prompt for scoring
                    hist.append(["", ""])

                    # Infer student's score
                    print(
                        "Chosing exam question:",
                        chosen_ex_ques_idx,
                        chosen_ex_ques_name,
                    )
                    print(f"Exam Score ({chosen_ex_ques_name}):", chosen_ex_score)

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

                    hist[-1][0] = f"Your score: {chosen_ex_score}/1."

                list_scores[qidx] = exam_scores

                if round_idx == MAX_QUES_PER_WEEK - 1:
                    list_trajs[qidx] = copy.deepcopy(hist)

            with ThreadPoolExecutor(max_workers=32) as executor:
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
                score = sum([v for k, v in scores.items() if k.endswith("combined")])
                if best_scores is not None:
                    best_score = sum(
                        [v for k, v in best_scores.items() if k.endswith("combined")]
                    )
                else:
                    best_score = None

                if best_score is None or score > best_score:
                    best_scores = scores
                    best_traj = traj

            student_history = best_traj

            # Save student history
            with open(
                f"results/{args.cls}/week{START_WEEK+len(optimized_exam_scores)}_ques{round_idx+1}.json",
                "w",
            ) as f:
                json.dump(best_traj, f)

            if round_idx == MAX_QUES_PER_WEEK - 1:
                optimized_exam_scores.append(best_scores)
                wi = len(optimized_exam_scores)
                print(
                    "Best optimzied score for week",
                    START_WEEK + wi - 1,
                    ": ",
                    best_scores,
                )
                print(
                    "Actual score for week",
                    START_WEEK + wi - 1,
                    ": ",
                    actual_exam_scores[wi - 1],
                )
                # Save exam scores
                with open(
                    f"results/{args.cls}/week{START_WEEK+len(optimized_exam_scores)-1}_exam.json",
                    "w",
                ) as f:
                    json.dump(
                        {
                            "actual": actual_exam_scores[wi - 1],
                            "optimized": best_scores,
                        },
                        f,
                    )

    print("Actual exam scores:")
    print(actual_exam_scores)
    print("Optimized exam scores:")
    print(optimized_exam_scores)

    with open(f"results/{args.cls}/final_results.json", "w") as f:
        json.dump({"actual": actual_exam_scores, "optimized": optimized_exam_scores}, f)

    # wandb.finish()
