import argparse
import json
import os
from urllib.parse import unquote, urlsplit

import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from utils import find_global_max


def plot_per_question(repo_id, course_name, class_name):
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
            plt.clf()

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

            print("PLOT")

            max_attempts = max(len(student_marks) for student_marks in records)

            x = list(range(1, max_attempts + 1))

            # Find the average score for each number of attemps
            # Pad the marks of each student with highest marks
            padded_records = []
            for student_marks in records:
                if student_marks:
                    padded_records.append(
                        student_marks
                        + [max(student_marks)] * (max_attempts - len(student_marks))
                    )
                else:
                    padded_records.append([0] * max_attempts)

            padded_records = np.array(padded_records)
            average_marks = np.nanmean(padded_records, axis=0)

            for i, student_marks in enumerate(padded_records):
                plt.plot(x, student_marks, label=f"{ids[i]}", color="blue", alpha=0.3)

            plt.plot(x, average_marks, label="Average Marks", linewidth=3, color="red")

            lab_name = json_file.split(".")[0]
            # Add labels and legend
            plt.xlabel("Attempts")
            plt.ylabel("Marks")
            plt.title(f"{lab_name} - Q{q_index}")
            plt.savefig(f"plots/{course_name}/{class_name}/{lab_name}-Q{q_index}.png")


def plot_all_questions(repo_id, course_name, class_name):
    plt.clf()

    all_padded_records = []
    global_max = find_global_max(repo_id, course_name, class_name)

    x = list(range(1, global_max + 1))

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
            print(f"PLOT {idx}")

            padded_records = []
            for student_marks in records:
                if student_marks:
                    padded_records.append(
                        student_marks
                        + [max(student_marks)] * (global_max - len(student_marks))
                    )
                    all_padded_records.append(
                        student_marks
                        + [max(student_marks)] * (global_max - len(student_marks))
                    )
                else:
                    padded_records.append([0] * global_max)
                    all_padded_records.append([0] * global_max)

            padded_records = np.array(padded_records)

            # Plot each student's attempts
            for i, student_marks in enumerate(padded_records):
                plt.plot(x, student_marks, label=f"{ids[i]}", color="blue", alpha=0.2)

    all_padded_records = np.array(all_padded_records)
    all_average_marks = np.nanmean(all_padded_records, axis=0)

    plt.plot(x, all_average_marks, label="All Average Marks", linewidth=3, color="Red")
    plt.xlabel("Attempts")
    plt.ylabel("Marks")
    plt.title(f"All questions")
    plt.savefig(f"plots/{course_name}/{class_name}/all-questions.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="DSA-HK231"
    )
    parser.add_argument("--class_name", help="Class Name", type=str, default="L01")
    args = parser.parse_args()
    course_name = args.course_name
    class_name = args.class_name

    repo_id = "stair-lab/dsa_records"

    os.makedirs("plots", exist_ok=True)
    os.makedirs(f"plots/{course_name}", exist_ok=True)
    os.makedirs(f"plots/{course_name}/{class_name}", exist_ok=True)

    plot_per_question(repo_id, course_name, class_name)
    plot_all_questions(repo_id, course_name, class_name)


if __name__ == "__main__":
    main()
