import json
import os
import pickle
import shutil
from argparse import ArgumentParser
from urllib.parse import unquote, urlsplit

import Levenshtein
import numpy as np
import pandas as pd
import requests
import warning
from bs4 import BeautifulSoup
from datasets import config, load_dataset
from huggingface_hub import HfApi, snapshot_download


def format_dataset(repo_id, directory_json_files, threshold=0.8):
    question_attempts = []
    student_id_to_index = {}
    correctness = []
    all_questions = {}

    for directory, files in directory_json_files.items():
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as json_file:
                        data = json.load(json_file)
                        for q in data["list_questions"]:
                            processed_question = preprocess_question(q["question"])
                            all_questions.append(processed_question)

            for q in data_q["train"]:
                if q["name"] not in all_questions:
                    all_questions[q["name"]] = q["question"]

                            for response_history in answer["response_history"]:
                                num_attempts = len(response_history["results"]) - 2
                                question_attempts.append(num_attempts)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file: {file_path}: {e}")

    student_ids = [{}] * len(student_id_to_index)
    for sid, idx in student_id_to_index.items():
        student_ids[idx] = {"student_id": sid}

    question_index = {qid: idx for idx, qid in enumerate(all_questions)}
    unique_questions = list(all_questions.values())

    N = len(student_id_to_index)
    Q = len(unique_questions)
    T = max(question_attempts, default=0)
    correctness_matrix = np.full((N, Q, T), -1).tolist()
    correctness_bytc_matrix = np.full((N, Q, T), -1).tolist()
    time_matrix = np.full((N, Q, T), "").tolist()
    response_matrix = np.full((N, Q, T), "").tolist()

    for root, dirs, files in os.walk(repo_id):
        for file in files:
            print(f"Processing {dirs}/{file}")
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as json_file:
                        data = json.load(json_file)
                        question_content_map = {
                            f"Question {idx + 1}": (q["question"], q["max_scores"])
                            for idx, q in enumerate(data["list_questions"])
                            if "max_scores" in q
                        }

            question_content_map = {
                f"Question {idx + 1}": (q["name"], q["max_score"])
                for idx, q in enumerate(data_q["train"])
                if "max_scores" in q
            }
            for answer in data_s["train"]:
                s_idx = student_id_to_index[answer["id"]]
                if "class" not in student_ids[s_idx]:
                    student_ids[s_idx]["class"] = directory

                for response_history in answer["response_history"]:
                    current_question_name, question_max_score = (
                        question_content_map.get(response_history["question"], ("", 0))
                    )
                    if not current_question_name or float(question_max_score) == 0:
                        continue

                    q_idx = question_index[current_question_name]

                    for t, result in enumerate(response_history["results"][1:-1]):
                        if result["score"] == "":
                            score = 0
                        elif float(result["score"]) > float(question_max_score):
                            warning.warn("Student score exceeds max score!")
                        else:
                            score = float(result["score"]) / float(question_max_score)

                        response = result["action"]
                        for prefix in ["Prechecked: ", "Saved: ", "Submit: "]:
                            if result["action"].startswith(prefix):
                                response = response.replace(prefix, "")

                        correctness_matrix[s_idx][q_idx][t] = score
                        correctness_bytc_matrix = result["testcase"]
                        time_matrix[s_idx][q_idx][t] = result["time"]
                        response_matrix[s_idx][q_idx][t] = response

    return (
        student_ids,
        unique_questions,
        correctness_matrix,
        correctness_bytc_matrix,
        time_matrix,
        response_matrix,
    )


def upload_files(repo_id, file_paths):
    api = HfApi()

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"Uploaded {file_name} successfully.")
        except Exception as e:
            print(f"Failed to upload {file_name}: {str(e)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--course_name", help="Class Name", type=str, default="dsa_hk231"
    )
    args = parser.parse_args()

    directory_json_files = find_json_files(f"stair-lab/{args.course_name}_records_wtc")
    (
        student_ids,
        unique_questions,
        correctness_matrix,
        correctness_bytc_matrix,
        time_matrix,
        response_matrix,
    ) = format_dataset(
        f"stair-lab/{args.course_name}_records_wtc", directory_json_files
    )

    matrices = [
        "data/student_ids.pkl",
        "data/unique_questions.pkl",
        "data/correctness_matrix.pkl",
        "data/correctness_bytc_matrix.pkl",
        "data/time_matrix.pkl",
        "data/response_matrix.pkl",
    ]

    with open("student_ids.pkl", "wb") as file:
        pickle.dump(student_ids, file)

    with open("unique_questions.pkl", "wb") as file:
        pickle.dump(unique_questions, file)

    with open("correctness_matrix.pkl", "wb") as file:
        pickle.dump(correctness_matrix, file)

    with open("data/correctness_bytc_matrix.pkl", "wb") as file:
        pickle.dump(correctness_matrix, file)

    with open("data/time_matrix.pkl", "wb") as file:
        pickle.dump(time_matrix, file)

    with open("response_matrix.pkl", "wb") as file:
        pickle.dump(response_matrix, file)

    upload_files(f"stair-lab/{args.course_name}_wtc", matrices)
