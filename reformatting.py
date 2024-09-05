import json
import os
import pickle
import shutil
from urllib.parse import unquote, urlsplit

import Levenshtein
import numpy as np
import pandas as pd
import requests
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


def preprocess_question(question):
    answer_phrase = "Answer:(penalty regime:"
    question_phrase = "Question text "
    if question_phrase in question:
        question = question.replace(question_phrase, "").strip()

    if answer_phrase in question:
        question = question.split(answer_phrase)[0].strip()

    return question


def tokenize(text):
    tokens = text.split()
    return " ".join(tokens[:150])


def edit_distance(s1, s2):
    if not s1 or not s2:
        return 0
    distance = Levenshtein.distance(s1, s2)

    max_len = max(len(s1), len(s2))
    similarity = 1 - (distance / max_len)
    return similarity


def filter_unique_questions(questions, threshold=0.8):
    if not questions:
        return []
    full_filtered_questions = [questions[0]]
    filtered_questions = [tokenize(questions.pop(0))]

    while questions:
        full_current_question = questions[0]
        current_question = tokenize(questions.pop(0))
        is_unique = True

        for accepted_question in filtered_questions:
            if edit_distance(current_question, accepted_question) > threshold:
                is_unique = False
                break

        if is_unique:
            filtered_questions.append(current_question)
            full_filtered_questions.append(full_current_question)

    return filtered_questions, full_filtered_questions


def format_dataset(repo_id, directory_json_files, threshold=0.8):
    question_attempts = []
    student_id_to_index = {}
    correctness = []
    all_questions = []

    for directory, files in directory_json_files.items():
        for file in files:
            data_q = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="list_questions",
                cache_dir="./cache"
            )

            for q in data_q["train"]:
                processed_question = preprocess_question(q["question"])
                all_questions.append(processed_question)

            data_s = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="student_answers",
                cache_dir="./cache"
            )
            base_name, _ = os.path.splitext(file)

            for answer in data_s["train"]:
                student_id = answer["id"]
                if student_id not in student_id_to_index:
                    student_id_to_index[student_id] = len(student_id_to_index)

                for response in answer["response_history"]:
                    num_attempts = len(response["results"])
                    question_attempts.append(num_attempts)

    student_ids = [{}] * len(student_id_to_index)
    for sid, idx in student_id_to_index.items():
        student_ids[idx] = {"student_id": sid}
    unique_questions = list(set(all_questions))
    unique_questions_by_distance, full_unique_questions_by_distance = (
        filter_unique_questions(unique_questions)
    )
    question_index = {qid: idx for idx, qid in enumerate(unique_questions_by_distance)}

    N = len(student_id_to_index)
    Q = len(unique_questions_by_distance)
    T = max(question_attempts, default=0)
    correctness_matrix = np.full((N, Q, T), -1).tolist()
    time_matrix = np.full((N, Q, T), "").tolist()
    response_matrix = np.full((N, Q, T), "").tolist()
    print(Q)

    for directory, files in directory_json_files.items():
        for file in files:
            data_q = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="list_questions"
            )
            data_s = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="student_answers"
            )

            question_content_map = {
                f"Question {idx + 1}": q["question"]
                for idx, q in enumerate(data_q["train"])
            }
            for answer in data_s["train"]:
                s_idx = student_id_to_index[answer["id"]]
                if "class" not in student_ids[s_idx]:
                    student_ids[s_idx]["class"] = directory
                for response in answer["response_history"]:
                    current_question = question_content_map.get(
                        response["question"], ""
                    )

                    q_idx = None
                    for idx, q_unique in enumerate(unique_questions_by_distance):
                        if (
                            edit_distance(tokenize(current_question), q_unique)
                            > threshold
                        ):
                            q_idx = idx
                            break
                    if q_idx is not None:
                        for t, result in enumerate(response["results"]):
                            score = (
                                float(result["marks"])
                                if result["marks"] != ""
                                else float(-1)
                            )
                            timestamp = result["time"]
                            response = result["action"]
                            correctness_matrix[s_idx][q_idx][t] = score
                            time_matrix[s_idx][q_idx][t] = timestamp
                            response_matrix[s_idx][q_idx][t] = response

    return (
        student_ids,
        full_unique_questions_by_distance,
        correctness_matrix,
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
    repo_id = "stair-lab/dsa_hk231"

    directory_json_files = find_json_files(repo_id)
    student_ids, unique_questions, correctness_matrix, time_matrix, response_matrix = (
        format_dataset(repo_id, directory_json_files)
    )

    matrices = [
        "student_ids.pkl",
        "unique_questions.pkl",
        "correctness_matrix.pkl",
        "time_matrix.pkl",
        "response_matrix.pkl",
    ]

    with open("student_ids.pkl", "wb") as file:
        pickle.dump(student_ids, file)

    with open("unique_questions.pkl", "wb") as file:
        pickle.dump(unique_questions, file)

    with open("correctness_matrix.pkl", "wb") as file:
        pickle.dump(correctness_matrix, file)

    with open("time_matrix.pkl", "wb") as file:
        pickle.dump(time_matrix, file)

    with open("response_matrix.pkl", "wb") as file:
        pickle.dump(response_matrix, file)

    upload_files(repo_id, matrices)

    cache_dir = config.HF_DATASETS_CACHE
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
