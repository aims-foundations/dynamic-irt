import datetime
import json
import os
import pickle
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlsplit

import Levenshtein
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm


def fetch_unique_questions(filepath):
    try:
        with open(filepath, "rb") as file:  # Open the file in binary read mode
            unique_questions = pickle.load(file)
        return unique_questions
    except Exception as e:
        print(f"Failed to load and deserialize the pickle file: {e}")
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


def filter_cheating_between_students(responses, question, threshold=0.1):
    if not responses:
        return [], {}

    filtered_records = [responses.pop(0)]
    potential_copies_by_question = []
    seen_pairs = set()

    for current_response in responses:
        is_unique = True

        for accepted_response in filtered_records:
            id_pair = tuple(sorted([current_response["id"], accepted_response["id"]]))

            dist = levenshtein_distance(
                current_response["response"], accepted_response["response"]
            )
            normalized_distance = dist / max(
                len(current_response["response"]), len(accepted_response["response"])
            )

            if (
                normalized_distance <= threshold
                and current_response["id"] != accepted_response["id"]
                and id_pair not in seen_pairs
            ):
                is_unique = False
                seen_pairs.add(id_pair)

                potential_copies_by_question.append(
                    {
                        current_response["id"]: current_response["response"],
                        accepted_response["id"]: accepted_response["response"],
                        "edit_distance": dist,
                        "normalized_distance": normalized_distance,
                    }
                )
                break

        if is_unique:
            filtered_records.append(current_response)

    return filtered_records, potential_copies_by_question


def detect_cheating(
    records, change_threshold=0.9, time_threshold=datetime.timedelta(minutes=10)
):
    if not records:
        return []

    details = []

    for i in range(1, len(records)):
        current_record = records[i]
        previous_record = records[i - 1]

        max_len = max(len(current_record[0]), len(previous_record[0]))
        edit_distance_fraction = (
            edit_distance(current_record[0], previous_record[0]) / max_len
        )

        time_difference = (current_record[1] - previous_record[1]).total_seconds()

        if (
            edit_distance_fraction > change_threshold
            and time_difference < time_threshold.total_seconds()
        ):
            details.append(
                {
                    "edit_distance_fraction": edit_distance_fraction,
                    "time_difference_seconds": time_difference,
                    "action_pair": [previous_record[0], current_record[0]],
                    "state_pair": [previous_record[2], current_record[2]],
                }
            )

    return details


def find_matching_question(question_text, unique_questions, threshold=0.8):
    processed_text = tokenize(preprocess_question(question_text))

    for unique_question in unique_questions:
        processed_unique_question = tokenize(preprocess_question(unique_question))
        if similarity(processed_text, processed_unique_question) >= threshold:
            return unique_question

    return None


def main(repo_id, threshold=0.9):
    unique_questions = fetch_unique_questions(f"{repo_id}/unique_questions.pkl")
    all_records = {}
    cheaters = {}
    copying_records = {}

    for root, dirs, files in os.walk(repo_id):
        for file in tqdm(files, desc="Processing files"):
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as json_file:
                        data = json.load(json_file)
                        lab_name = data["lab_name"]
                        list_questions = [q["question"] for q in data["list_questions"]]

                        for answer in tqdm(
                            data["student_answers"],
                            desc="Processing answers",
                            leave=False,
                        ):
                            student_id = answer["id"]
                            if student_id in cheaters:
                                continue

                            for q_idx, response in enumerate(
                                answer["response_history"]
                            ):
                                matching_question = find_matching_question(
                                    list_questions[q_idx], unique_questions
                                )

                                if not matching_question:
                                    print(lab_name, file_path, list_questions[q_idx])
                                    matching_question = "Unmatched Questions"

                                # question_id = response["question"]
                                student_responses = []
                                last_response = None
                                last_correct_response = None

                                for r_text in response["results"]:
                                    if (
                                        r_text["state"]
                                        not in [
                                            "Not complete",
                                            "Precheck results",
                                            "Incomplete answer",
                                        ]
                                        and "Attempt finished" not in r_text["action"]
                                    ):
                                        timestamp = datetime.datetime.strptime(
                                            r_text["time"], "%d/%m/%y, %H:%M:%S"
                                        )
                                        student_responses.append(
                                            (
                                                r_text["action"],
                                                timestamp,
                                                r_text["state"],
                                            )
                                        )

                                        last_response = {
                                            "response": r_text["action"],
                                            "file_path": file_path,
                                        }
                                        if r_text["state"] == "Correct":
                                            last_correct_response = last_response

                                # print(student_responses)
                                cheating_details = detect_cheating(student_responses)
                                if cheating_details != []:
                                    cheaters.setdefault(student_id, []).append(
                                        {
                                            "lab": lab_name,
                                            "file": file_path,
                                            "question_id": response["question"],
                                            "cheating_details": cheating_details,
                                        }
                                    )

                                else:
                                    final_response = (
                                        last_correct_response
                                        if last_correct_response
                                        else last_response
                                    )
                                    if final_response:
                                        all_records.setdefault(
                                            matching_question, {}
                                        ).setdefault(student_id, []).append(
                                            final_response
                                        )

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file: {file_path}: {e}")

    filtered_all_records = {}

    for question, responses_by_students in tqdm(
        all_records.items(), desc="Filtering records"
    ):
        filtered_all_records[question] = {}
        all_responses = []
        for student_id, responses in responses_by_students.items():
            for response in responses:
                all_responses.append(
                    {
                        "id": student_id,
                        "response": response["response"],
                        "question": question,
                    }
                )

        if len(all_responses) > 1:
            filtered_responses, copies = filter_cheating_between_students(
                all_responses, question
            )
            for response in filtered_responses:
                filtered_all_records[question].setdefault(response["id"], []).append(
                    {"response": response["response"]}
                )
            if copies:
                copying_records[question] = copies  # Store copies by question
        else:
            for response in all_responses:
                filtered_all_records[question].setdefault(response["id"], []).append(
                    {"response": response["response"]}
                )

    with open("cheating_records.json", "w") as f:
        json.dump(cheaters, f, indent=4, default=datetime_converter)

    with open("copying_records.json", "w") as f:
        json.dump(copying_records, f, indent=4)

    return


if __name__ == "__main__":
    repo_id = "stair-lab/dsa_hk231_records"
    data_folder = snapshot_download(repo_id=repo_id, repo_type="dataset")

    main(data_folder)
