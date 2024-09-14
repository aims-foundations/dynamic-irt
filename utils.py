import json
import os
import re
from urllib.parse import unquote, urlsplit

import requests
from bs4 import BeautifulSoup
from datasets import load_dataset


def parse_score(header_text):
    search_result = re.search(r"/(\d+\.\d+)", header_text)
    if search_result:
        return float(search_result.group(1))
    else:
        return None


def filter_class_group(details, sid, class_name):
    for detail in details:
        if detail["Class Group"] == "No groups":
            if detail["ID"] == sid:
                print(
                    f"Processing student ID: {detail['ID']} with only a class group."
                )
                return True
        
        if detail["ID"] == sid and detail["Class Group"] == class_name:
            print(
                f"Processing student ID: {detail['ID']} from Class Group: {detail['Class Group']}"
            )
            return True

    return False


def find_global_max(repo_id, course_name, class_name):
    global_max = 0

    url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{class_name}/"
    response = requests.get(url)
    json_files = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a")

        for link in links:
            href = link.get("href")
            if href.endswith(".json"):
                path = urlsplit(href).path
                filename = path.split("/")[-1]
                filename = unquote(filename)
                json_files.append(filename)
    else:
        print("Failed to retrieve data:", response.status_code)

    for json_file in json_files:
        data_q = load_dataset(
            repo_id, data_files=f"{class_name}/{json_file}", field="list_questions"
        )
        data_s = load_dataset(
            repo_id, data_files=f"{class_name}/{json_file}", field="student_answers"
        )

        ids = []
        for answers in data_s["train"]:
            ids.append(answers["id"])

        for idx in range(len(data_q["train"])):
            max_score = data_q["train"][idx]["max_scores"]
            q_index = idx + 1

            records = []
            for answers in data_s["train"]:
                for answer in answers["response_history"]:
                    marks = []
                    if answer["question"] == f"Question {q_index}":
                        mark_per_attempt = []
                        for score_idx in range(len(answer["results"])):
                            if answer["results"][score_idx]["marks"] != "":
                                mark_per_attempt.append(
                                    answer["results"][score_idx]["marks"]
                                )

                    marks.append(mark_per_attempt)
                records.extend(marks)

            try:
                records = [
                    [float(mark) * 10 / max_score for mark in student_marks]
                    for student_marks in records
                ]
            except:
                continue

            max_attempts = [len(student_marks) for student_marks in records]
            global_max = max(global_max, max(max_attempts))

    return global_max
