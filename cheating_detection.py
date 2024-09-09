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
import datetime


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


def fetch_unique_questions(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        try:
            # print(response.content)
            unique_questions = pickle.loads(response.content)
            return unique_questions
        except Exception as e:
            print(f"Failed to load and deserialize the pickle file: {e}")
            return []
    else:
        print(f"Failed to download data: HTTP {response.status_code}")
        return []

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

def datetime_converter(o):
    if isinstance(o, datetime.datetime):
        return o.isoformat()

def edit_distance(s1, s2):
    if not s1 or not s2:
        return 0
    distance = Levenshtein.distance(s1, s2)

    return distance

def similarity(s1, s2):
    if not s1 or not s2:
        return 0
    return Levenshtein.ratio(s1, s2)

def detect_cheating(records, change_threshold=0.9, time_threshold=datetime.timedelta(minutes=10)):
    if not records:
        return []

    details = []

    for i in range(1, len(records)):
        current_record = records[i]
        previous_record = records[i - 1]

        max_len = max(len(current_record[0]), len(previous_record[0]))
        edit_distance_fraction = edit_distance(current_record[0], previous_record[0]) / max_len

        time_difference = (current_record[1] - previous_record[1]).total_seconds()

        if edit_distance_fraction > change_threshold and time_difference < time_threshold.total_seconds():
            details.append({
                "edit_distance_fraction": edit_distance_fraction,
                "time_difference_seconds": time_difference,
                "action_pair": [previous_record[0], current_record[0]],
                "state_pair": [previous_record[2], current_record[2]]
            })

    return details

def find_matching_question(question_text, unique_questions, threshold=0.8):
    processed_text = tokenize(preprocess_question(question_text))
    
    for unique_question in unique_questions:
        processed_unique_question = tokenize(preprocess_question(unique_question))
        if similarity(processed_text, processed_unique_question) >= threshold:
            return unique_question
    
    return None

def main(repo_id, directory_json_files, threshold=0.9):
    unique_questions = fetch_unique_questions(f"https://huggingface.co/datasets/{repo_id}/resolve/main/unique_questions.pkl")
    all_records = {}
    cheaters = {}

    for directory, files in directory_json_files.items():
        for file in files:
            data_q = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="list_questions"
            )
            list_questions = [q["question"] for q in data_q["train"]]
                
            data_s = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="student_answers"
            )
            base_name, _ = os.path.splitext(file)

            for answer in data_s["train"]:
                student_id = answer["id"]
                
                if student_id in cheaters:
                    continue

                for q_idx, response in enumerate(answer["response_history"]):
                    matching_question = find_matching_question(list_questions[q_idx], unique_questions)

                    if not matching_question:
                        print(base_name, file_path, list_questions[q_idx])
                        matching_question = "Unmatched Questions"
                    
                    # question_id = response["question"]
                    student_responses = []
                    last_response = None
                    last_correct_response = None
                    
                    for r_text in response["results"]:
                        if r_text["state"] not in ["Not complete", "Precheck results", "Incomplete answer"] and "Attempt finished" not in r_text["action"]:
                            timestamp = datetime.datetime.strptime(r_text["time"], "%d/%m/%y, %H:%M:%S")
                            student_responses.append((r_text["action"], timestamp, r_text["state"]))

                            last_response = {
                                "response": r_text["action"],
                                "file_path": file
                            }
                            if r_text["state"] == "Correct":
                                last_correct_response = last_response

                    # print(student_responses)
                    cheating_details = detect_cheating(student_responses)
                    if cheating_details != []:
                        cheaters.setdefault(student_id, []).append({
                            "lab": base_name,
                            "file": file,
                            "question_id": response["question"],
                            "cheating_details": cheating_details
                        })
                        break

                    else:
                        final_response = last_correct_response if last_correct_response else last_response
                        if final_response:
                            all_records.setdefault(matching_question, {}).setdefault(student_id, []).append(final_response)

     with open('cheating_records.json', 'w') as f:
        json.dump(cheaters, f, indent=4, default=datetime_converter)

    with open('non_cheating_records.json', 'w') as f:
        json.dump(all_records, f, indent=4, default=datetime_converter)

    return

if __name__ == "__main__":
    repo_id = "stair-lab/dsa_hk231_records"

    directory_json_files = find_json_files(repo_id)
    main(
        repo_id, directory_json_files
    )

    cache_dir = config.HF_DATASETS_CACHE
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
