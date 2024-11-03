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
from huggingface_hub import HfApi


def find_json_files(repo_id):
    directories = []
    base_url = f"https://huggingface.co/datasets/{repo_id}/tree/main/"
    response = requests.get(base_url)

    directory_json_files = {}

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", href=True)
        for link in links:
            href = link["href"]
            if "/tree/main/" in href and href.count("/") == 6:
                directory_name = href.split("/")[-1]
                directories.append(directory_name)

    for directory in directories:
        dir_url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{directory}"
        response = requests.get(dir_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.find_all("a", href=True)
            json_files = []
            for link in links:
                href = link["href"]
                if href.endswith(".json"):
                    path = urlsplit(href).path
                    filename = path.split("/")[-1]
                    filename = unquote(filename)
                    json_files.append(filename)

            if json_files:
                directory_json_files[directory] = json_files

    return directory_json_files


def format_dataset(repo_id, directory_json_files, threshold=0.8):
    question_attempts = []
    student_id_to_index = {}
    correctness = []
    all_questions = {}

    for directory, files in directory_json_files.items():
        for file in files:
            data_q = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="list_questions"
            )

            for q in data_q["train"]:
                if q["name"] not in all_questions:
                    all_questions[q["name"]] = q["question"]

            data_s = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="student_answers"
            )
            base_name, _ = os.path.splitext(file)

            for answer in data_s["train"]:
                student_id = answer["id"]
                if student_id not in student_id_to_index:
                    student_id_to_index[student_id] = len(student_id_to_index)

                for response_history in answer["response_history"]:
                    num_attempts = len(response_history["results"]) - 2
                    question_attempts.append(num_attempts)

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

    for directory, files in directory_json_files.items():
        for file in files:
            print(f"Processing {directory}/{file}")
            data_q = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="list_questions"
            )
            data_s = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="student_answers"
            )

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

    with open("data/student_ids.pkl", "wb") as file:
        pickle.dump(student_ids, file)

    with open("data/unique_questions.pkl", "wb") as file:
        pickle.dump(unique_questions, file)

    with open("data/correctness_matrix.pkl", "wb") as file:
        pickle.dump(correctness_matrix, file)

    with open("data/correctness_bytc_matrix.pkl", "wb") as file:
        pickle.dump(correctness_matrix, file)

    with open("data/time_matrix.pkl", "wb") as file:
        pickle.dump(time_matrix, file)

    with open("data/response_matrix.pkl", "wb") as file:
        pickle.dump(response_matrix, file)

    upload_files(f"stair-lab/{args.course_name}_wtc", matrices)
