import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from config import CLASSES, FEEDBACK_INST, SYSTEM_PROMPT, WEEK_FILES
from datasets import concatenate_datasets, Dataset, DatasetDict
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_time(time_str):
    # Parsing the string into a datetime object
    parsed_datetime = datetime.strptime(time_str, "%d/%m/%y, %H:%M:%S")
    return parsed_datetime


def format_question(question):
    answer_phrase = "Answer:(penalty regime:"
    question_phrase = "Question text "
    if question_phrase in question:
        question = question.replace(question_phrase, "").strip()

    if answer_phrase in question:
        question = question.split(answer_phrase)[0].strip()
    return question


def format_questions(list_questions):
    formated_questions = []
    for qi, q in enumerate(list_questions):
        if isinstance(q, list):
            for sub_idx, sub_q in enumerate(q):
                formated_questions.append(
                    f"Question {qi+1}.{sub_idx+1}: "
                    + format_question(sub_q["question"])
                )
        else:
            formated_questions.append(
                f"Question {qi+1}: " + format_question(q["question"])
            )
    return "\n\n".join(formated_questions)


def get_question_names(list_questions):
    list_names = []
    for qi, q in enumerate(list_questions):
        if isinstance(q, list):
            list_names.append("|".join([sub_q["name"] for sub_q in q]))
        else:
            list_names.append(q["name"])
    return list_names


def format_response(response):
    for prefix in ["Saved: ", "Prechecked: ", "Submit: "]:
        response = response.replace(prefix, "")
    return response


def format_chat_template(tokenizer, exams, exercises, style="trl"):
    ex_questions, ex_responses = exams
    questions, responses = exercises
    # questions: List[str]
    # responses: List[List[List[str]]]. Shape: week x (qidx,....)

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    conv_weeks = [1]
    conv_question_names = [[]]
    testcase_scores = [[]]
    last_feedback = ""
    for wi, (w_ex_questions, w_ex_responses, w_questions, w_responses) in enumerate(
        zip(ex_questions, ex_responses, questions, responses)
    ):
        last_ex_feedback = ""

        if len(w_ex_questions) > 0:
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        last_feedback
                        + "Here are the exam questions.\n\n"
                        + format_questions(w_ex_questions)
                        + "Please write your answer for the above question in C++."
                    ),
                }
            )
            last_feedback = ""
            conv_weeks.append(wi + 1)
            conv_question_names.append(get_question_names(w_ex_questions))
            testcase_scores.append(res[1]["testcases"])

            if len(w_ex_responses) == 0:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": "<|no_answer|>",
                    }
                )
                conv_weeks.append(wi + 1)
                conv_question_names.append([])
                testcase_scores.append([])
            else:
                for ridx, res in enumerate(w_ex_responses):
                    # res: 0: qidx, 1: student_exm, 2: exam_scores, 3: exam_tc_scores
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": f"Answer for question {res[0]}:\n"
                            + format_response(res[1]["action"]),
                        }
                    )
                    conv_weeks.append(wi + 1)
                    conv_question_names.append([])
                    testcase_scores.append([])
                    if ridx < len(w_ex_responses) - 1:
                        conversation.append(
                            {
                                "role": "user",
                                "content": FEEDBACK_INST.format(score=round(res[2], 2)),
                            }
                        )
                        conv_weeks.append(wi + 1)
                        conv_question_names.append([])
                        testcase_scores.append(res[1]["testcases"])

                    else:
                        last_ex_feedback = (
                            FEEDBACK_INST.format(score=round(res[2], 2)) + "\n\n"
                        )

        ### Week exercises
        conversation.append(
            {
                "role": "user",
                "content": (
                    last_ex_feedback
                    + "Here are the exercise questions for practice.\n\n"
                    + format_questions(w_questions)
                    + "Please write your answer for the above question in C++."
                ),
            }
        )
        conv_weeks.append(wi + 1)
        conv_question_names.append(get_question_names(w_questions))
        if last_ex_feedback:
            testcase_scores.append(res[1]["testcases"])
        else:
            testcase_scores.append([])

        if len(w_responses) == 0:
            conversation.append(
                {
                    "role": "assistant",
                    "content": "<|no_answer|>",
                }
            )
            conv_weeks.append(wi + 1)
            conv_question_names.append([])
            testcase_scores.append([])
        else:
            for ridx, res in enumerate(w_responses):
                # res: 0: qidx, 1: student_exm, 2: exam_scores, 3: exam_tc_scores
                conversation.append(
                    {
                        "role": "assistant",
                        "content": f"Answer for question {res[0]}:\n"
                        + format_response(res[1]["action"]),
                    }
                )
                conv_weeks.append(wi + 1)
                conv_question_names.append([])
                testcase_scores.append([])
                if ridx < len(w_responses) - 1:
                    conversation.append(
                        {
                            "role": "user",
                            "content": FEEDBACK_INST.format(score=round(res[2], 2)),
                        }
                    )
                    conv_weeks.append(wi + 1)
                    conv_question_names.append([])
                    testcase_scores.append(res[1]["testcases"])

                else:
                    last_feedback = (
                        FEEDBACK_INST.format(score=round(res[2], 2)) + "\n\n"
                    )

    if style == "easycontext":
        prompts = tokenizer.apply_chat_template(
            conversation, tokenize=True, add_special_tokens=False
        )
    elif style == "trl":
        prompts = tokenizer.apply_chat_template(
            conversation, tokenize=True, add_special_tokens=False
        )
    elif style == "lf":
        prompts = {
            "system": [],
            "instruction": [],
            "output": [],
            "history": [],
            "week": [],
            "question_name": [],
            "testcase_scores": [],
        }
        history = []
        for idx in range(1, len(conversation), 2):
            prompts["system"].append(conversation[0]["content"])
            prompts["history"].append(history.copy())
            prompts["instruction"].append(conversation[idx]["content"])
            prompts["output"].append(conversation[idx + 1]["content"])
            prompts["week"].append(conv_weeks[idx])
            prompts["question_name"].append(conv_question_names[idx])
            prompts["testcase_scores"].append(testcase_scores[idx])
            history.append(
                [conversation[idx]["content"], conversation[idx + 1]["content"]]
            )

    return prompts

def split_ds_lf(ds, test_size=0.2):
    new_student_idxs = []
    for idx, hist in enumerate(ds["history"]):
        if len(hist) == 0:
            new_student_idxs.append(idx)

    total_students = len(new_student_idxs)
    start_test_idx = new_student_idxs[int(total_students * (1-test_size))]
    new_ds = ds.select(list(range(0, start_test_idx)))
    new_ds_test = ds.select(list(range(start_test_idx, len(ds))))

    data_dict = DatasetDict({
        "train": new_ds,
        "test": new_ds_test,
    })
    return data_dict

def split_ds_normal(ds, test_size=0.2):
    return ds.train_test_split(test_size=test_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument(
        "--model",
        help="Model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--style",
        help="style of dataset",
        type=str,
        choices=["trl", "lf", "easycontext"],
        default="lf",
    )
    args = parser.parse_args()

    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Download and load data
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}_records_wtc", repo_type="dataset"
    )

    final_ds = {}
    list_classes = CLASSES[args.course_name]
    for cls in tqdm(list_classes, desc="Processing"):
        print(f"*** CLASS {cls} ***")

        list_files = os.listdir(os.path.join(data_folder, cls))
        list_files = [f for f in list_files if f.endswith(".json")]
        list_files = sorted(list_files)

        # Check if list file has type WEEK_FILES[args.course_name][0] or WEEK_FILES[args.course_name][1]
        if list_files[0] in WEEK_FILES[args.course_name][1][0]:
            weeks = WEEK_FILES[args.course_name][1]
        elif "W6_All.json" in list_files:
            weeks = WEEK_FILES[args.course_name][2]
        else:
            weeks = WEEK_FILES[args.course_name][0]

        # Read all exam file and find students that have done all exams
        print("Finding students that do all exams")
        exam_files = WEEK_FILES[args.course_name]["exam"]
        student_done_exam = {}
        for exam_file in exam_files:
            with open(os.path.join(data_folder, cls, exam_file), "r") as f:
                data = json.load(f)

            for student_ans in data["student_answers"]:
                student_id = student_ans["id"]
                if student_id not in student_done_exam:
                    student_done_exam[student_id] = 0
                student_done_exam[student_id] += 1

        student_done_exam = [
            k for k, v in student_done_exam.items() if v == len(exam_files)
        ]

        ##########################################################################
        # Loading student exercises
        ##########################################################################
        print("Loading student exercises")

        question_by_week = []
        # >>> num_weeks x num_questions

        student_ans_by_week = {}
        # >>> student_id: num_weeks x num_questions x num_attempt

        # Read all exercise questions in each week
        for wi, week in enumerate(weeks):
            question_pw = []
            # >>> num_questions

            for json_file in week:
                if not os.path.exists(os.path.join(data_folder, cls, json_file)):
                    continue

                with open(os.path.join(data_folder, cls, json_file), "r") as f:
                    data = json.load(f)

                num_prev_questions = len(question_pw)
                question_pw.extend(data["list_questions"])
                student_res_by_topic = {}

                for student_ans in data["student_answers"]:
                    student_id = student_ans["id"]
                    if student_id not in student_done_exam:
                        continue

                    # student_res = [[] for _ in data["list_questions"]]
                    student_res = {}
                    # >>> num_question x num_attempt

                    for res in student_ans["response_history"]:
                        if "." in res["question"]:
                            dotidx = res["question"].rfind(".")
                            qidx = int(float(res["question"].split(" ")[-1]))
                            sub_qidx = int(res["question"][dotidx + 1 :])
                        else:
                            qidx = int(res["question"].split(" ")[-1])
                            sub_qidx = None

                        question_idx = str(num_prev_questions + qidx)
                        if sub_qidx is not None:
                            question_idx += f".{sub_qidx}"

                        if not question_idx in student_res:
                            student_res[question_idx] = res["results"]
                        else:
                            student_res[question_idx].extend(res["results"])

                    if student_id not in student_res_by_topic:
                        student_res_by_topic[student_id] = student_res
                    else:
                        for question_idx in student_res:
                            if question_idx in student_res_by_topic[student_id]:
                                student_res_by_topic[student_id][question_idx] = (
                                    student_res_by_topic[student_id][question_idx]
                                    + student_res[question_idx]
                                )
                            else:
                                student_res_by_topic[student_id][question_idx] = (
                                    student_res[question_idx]
                                )

                        # student_res_by_topic[student_id] = [
                        #     a + b
                        #     for a, b in zip(
                        #         student_res_by_topic[student_id], student_res
                        #     )
                        # ]

                for student_id in student_res_by_topic:
                    if student_id not in student_ans_by_week:
                        student_ans_by_week[student_id] = []

                    if len(student_ans_by_week[student_id]) == wi:
                        student_ans_by_week[student_id].append({})
                    elif len(student_ans_by_week[student_id]) < wi:
                        continue

                    student_ans_by_week[student_id][-1].update(
                        student_res_by_topic[student_id]
                    )

            question_by_week.append(question_pw)

        ##########################################################################
        # Loading student exams
        ##########################################################################
        print("Loading student exams")
        exam_by_week = [
            [],
        ]
        # >>> num_weeks x num_questions

        student_exm_by_week = {}
        # >>> student_id: num_weeks x num_questions x num_attempt

        # Read all exam questions in each week
        for wi, exam_file in enumerate(exam_files):
            with open(os.path.join(data_folder, cls, exam_file), "r") as f:
                data = json.load(f)

            exam_by_week.append(data["list_questions"])
            for student_ans in data["student_answers"]:
                student_id = student_ans["id"]
                if student_id not in student_done_exam:
                    continue

                if student_id not in student_exm_by_week:
                    student_exm_by_week[student_id] = [
                        {},
                    ]

                # student_res = [[] for _ in data["list_questions"]]
                student_res = {}
                # >>> num_question x num_attempt

                for res in student_ans["response_history"]:
                    # question_idx = int(res["question"].split(" ")[-1]) - 1
                    question_idx = res["question"].split(" ")[-1]
                    if not question_idx in student_res:
                        student_res[question_idx] = res["results"]
                    else:
                        student_res[question_idx].extend(res["results"])

                if len(student_exm_by_week[student_id]) == wi + 1:
                    student_exm_by_week[student_id].append(student_res)

        ##########################################################################
        # Gather and format data
        ##########################################################################
        print("Formating data")
        if args.style == "lf":
            list_prompts = {
                "system": [],
                "instruction": [],
                "output": [],
                "history": [],
                "week": [],
                "question_name": [],
                "testcase_scores": [],
            }
        else:
            list_prompts = []

        for stid in tqdm(student_exm_by_week, desc="Running students"):
            if len(student_ans_by_week) == 0:
                continue

            questions_all_weeks = []
            answer_all_weeks = []

            exams_all_weeks = []
            exam_answer_all_weeks = []

            for wi, (week_questions, week_exms, student_ans, student_exm) in enumerate(
                zip(
                    question_by_week,
                    exam_by_week,
                    student_ans_by_week[stid],
                    student_exm_by_week[stid],
                )
            ):
                # Prepare scores
                # Remove attempt with empty student_ans[question][attempt]["score"]
                for ridx, res in student_ans.items():
                    clean_res = []
                    for attempt in res:
                        if (
                            attempt["action"].startswith("Attempt finished")
                            and attempt["score"] == ""
                        ):
                            attempt["score"] = "0.0"
                            attempt["action"] = ""

                        if attempt["score"] != "" and not attempt["action"].startswith(
                            "Attempt finished"
                        ):
                            clean_res.append(attempt)

                    student_ans[ridx] = clean_res

                # Score can be accessd by student_ans[question][attempt]["score"]
                # Score must be normalized to [0, 1] by max_score, which can be accessed by week_questions[question]["max_score"]
                ques_scores = {}
                ques_tc_scores = {}
                for qidx, res in student_ans.items():
                    if isinstance(week_questions[int(float(qidx)) - 1], list):
                        # Set random question
                        max_score = week_questions[int(float(qidx)) - 1][0]["max_score"]
                    else:
                        max_score = week_questions[int(float(qidx)) - 1]["max_score"]
                    if max_score == 0:
                        # This question has bug
                        max_score = 1

                    score = []
                    tc_score = []
                    for attempt in student_ans[qidx]:
                        # score.append(float(attempt["score"]) / max_score)
                        if len(attempt["testcases"]) == 0:
                            tc_score.append([])
                            score.append(0.0)
                        else:
                            tc_score.append(attempt["testcases"])
                            score.append(
                                sum(attempt["testcases"]) / len(attempt["testcases"])
                            )

                    ques_tc_scores[qidx] = tc_score
                    ques_scores[qidx] = score

                # Prepare scores
                # Remove attempt with empty student_exm[question][attempt]["score"]
                for ridx, res in student_exm.items():
                    clean_res = []
                    for attempt in res:
                        if (
                            attempt["action"].startswith("Attempt finished")
                            and attempt["score"] == ""
                        ):
                            attempt["score"] = "0.0"
                            attempt["action"] = ""

                        if attempt["score"] != "" and not attempt["action"].startswith(
                            "Attempt finished"
                        ):
                            clean_res.append(attempt)
                    student_exm[ridx] = clean_res

                # Score can be accessed by student_exm[question][attempt]["score"]
                # Score must be normalized to [0, 1] by max_score, which can be accessed by week_exms[question]["max_score"]
                exam_scores = {}
                exam_tc_scores = {}
                for qidx, res in student_exm.items():
                    if isinstance(week_exms[int(float(qidx)) - 1], list):
                        # Set random question
                        max_score = week_exms[int(float(qidx)) - 1][0]["max_score"]
                    else:
                        max_score = week_exms[int(float(qidx)) - 1]["max_score"]

                    score = []
                    tc_score = []
                    for attempt in student_exm[qidx]:
                        # score.append(float(attempt["score"]) / max_score)
                        if len(attempt["testcases"]) == 0:
                            tc_score.append([])
                            score.append(0.0)
                        else:
                            tc_score.append(attempt["testcases"])
                            score.append(
                                sum(attempt["testcases"]) / len(attempt["testcases"])
                            )

                    exam_scores[qidx] = score
                    exam_tc_scores[qidx] = tc_score

                # Convert student_ans[question][attempt]["time"] from string to datetime
                for res in student_ans.values():
                    for attempt in res:
                        attempt["time"] = parse_time(attempt["time"])

                # Convert student_exm[question][attempt]["time"] from string to datetime
                for res in student_exm.values():
                    for attempt in res:
                        attempt["time"] = parse_time(attempt["time"])

                # Now we have:
                # student_ans: num_questions x num_attempt
                # student_exm: num_questions x num_attempt
                # ques_scores: num_questions x num_attempt
                # exam_scores: num_questions x num_attempt
                # Now we flatten the answers and scores. And we sort them by the student_ans[question][attempt]["time"]

                # Flatten the answers and scores
                student_ans_flat = []
                student_exm_flat = []
                for qidx in student_ans:
                    for aidx in range(len(student_ans[qidx])):
                        student_ans_flat.append(
                            (
                                qidx,
                                student_ans[qidx][aidx],
                                ques_scores[qidx][aidx],
                                ques_tc_scores[qidx][aidx],
                            )
                        )

                for qidx in student_exm:
                    for aidx in range(len(student_exm[qidx])):
                        student_exm_flat.append(
                            (
                                qidx,
                                student_exm[qidx][aidx],
                                exam_scores[qidx][aidx],
                                exam_tc_scores[qidx][aidx],
                            )
                        )

                # Sort the answers and scores by the student_ans[question][attempt]["time"]
                student_ans_flat = sorted(student_ans_flat, key=lambda x: x[1]["time"])
                student_exm_flat = sorted(student_exm_flat, key=lambda x: x[1]["time"])

                # Append to the list
                questions_all_weeks.append(week_questions)
                answer_all_weeks.append(student_ans_flat)

                exams_all_weeks.append(week_exms)
                exam_answer_all_weeks.append(student_exm_flat)

            # Prepare the data for training
            prompt = format_chat_template(
                tokenizer,
                exams=(exams_all_weeks, exam_answer_all_weeks),
                exercises=(questions_all_weeks, answer_all_weeks),
                style=args.style,
            )
            # pickle.dump(
            #     [(exams_all_weeks, exam_answer_all_weeks), (questions_all_weeks, answer_all_weeks)],
            #     open(f"student_data/{cls}/{stid}.pkl", "wb")
            # )
            if args.style == "lf":
                list_prompts["system"].extend(prompt["system"])
                list_prompts["instruction"].extend(prompt["instruction"])
                list_prompts["output"].extend(prompt["output"])
                list_prompts["history"].extend(prompt["history"])
                list_prompts["week"].extend(prompt["week"])
                list_prompts["question_name"].extend(prompt["question_name"])
                list_prompts["testcase_scores"].extend(prompt["testcase_scores"])
            else:
                list_prompts.append(prompt)

        if args.style == "easycontext":
            cls_ds = Dataset.from_dict({"input_ids": list_prompts})
            cls_ds = split_ds_normal(cls_ds)
        elif args.style == "trl":
            cls_ds = Dataset.from_dict({"text": list_prompts})
            cls_ds = split_ds_normal(cls_ds)
        elif args.style == "lf":
            cls_ds = Dataset.from_dict(list_prompts)
            cls_ds = split_ds_lf(cls_ds)
        final_ds[cls] = cls_ds

    # Create final dataset and push to hf hub
    all_ds_train = concatenate_datasets([ds["train"] for ds in final_ds.values()])
    all_ds_test = concatenate_datasets([ds["test"] for ds in final_ds.values()])
    final_ds["all_cls"] = DatasetDict(
        {
            "train": all_ds_train,
            "test": all_ds_test,
        }
    )

    if args.style == "lf":
        repo_name = f"stair-lab/{args.course_name}_sft"
    elif args.style == "trl":
        repo_name = f"stair-lab/{args.course_name}_sft_trl"
    elif args.style == "easycontext":
        repo_name = f"stair-lab/{args.course_name}_sft_easycontext"

    for cls, ds in final_ds.items():
        ds.push_to_hub(repo_name, config_name=cls)
