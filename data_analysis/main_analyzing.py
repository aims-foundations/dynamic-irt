import argparse
import json
import os
import re
from urllib.parse import unquote, urlsplit

import matplotlib.pyplot as plt
import numpy as np
import wandb
from huggingface_hub import snapshot_download

try:
    from .utils import find_global_max  # When used as a module
except ImportError:
    from utils import find_global_max  # When run as a script


def normalize_name(name):
    # Replace all non-alphanumeric characters (excluding underscores) with '_'
    return re.sub(r"[^\w\s]", "_", name)


def plot_per_question(repo_id, course_name, class_name):
    for root, dirs, files in os.walk(repo_id):
        if class_name in dirs:
            class_dir_path = os.path.join(root, class_name)

            for subdir_root, subdir_dirs, subdir_files in os.walk(class_dir_path):
                for file in subdir_files:
                    if file.endswith(".json"):
                        file_path = os.path.join(subdir_root, file)
                        try:
                            with open(file_path, "r") as json_file:
                                data = json.load(json_file)
                                print(data["lab_name"])
                                ids = []
                                for answers in data["student_answers"]:
                                    ids.append(answers["id"])

                                for idx in range(len(data["list_questions"])):
                                    plt.clf()

                                    max_score = data["list_questions"][idx][
                                        "max_scores"
                                    ]
                                    q_index = idx + 1

                                    records = []
                                    for answers in data["student_answers"]:
                                        for answer in answers["response_history"]:
                                            marks = []
                                            if (
                                                answer["question"]
                                                == f"Question {q_index}"
                                            ):
                                                mark_per_attempt = []
                                                for score_idx in range(
                                                    len(answer["results"])
                                                ):
                                                    if (
                                                        answer["results"][score_idx][
                                                            "marks"
                                                        ]
                                                        != ""
                                                    ):
                                                        mark_per_attempt.append(
                                                            answer["results"][
                                                                score_idx
                                                            ]["marks"]
                                                        )

                                            marks.append(mark_per_attempt)
                                        records.extend(marks)

                                    try:
                                        records = [
                                            [
                                                float(mark) * 10 / max_score
                                                for mark in student_marks
                                            ]
                                            for student_marks in records
                                        ]
                                    except:
                                        continue

                                    print("PLOT")

                                    max_attempts = max(
                                        len(student_marks) for student_marks in records
                                    )

                                    x = list(range(1, max_attempts + 1))

                                    # Find the average score for each number of attemps
                                    # Pad the marks of each student with highest marks
                                    padded_records = []
                                    for student_marks in records:
                                        if student_marks:
                                            padded_records.append(
                                                student_marks
                                                + [max(student_marks)]
                                                * (max_attempts - len(student_marks))
                                            )
                                        else:
                                            padded_records.append([0] * max_attempts)

                                    padded_records = np.array(padded_records)
                                    average_marks = np.nanmean(padded_records, axis=0)

                                    for i, student_marks in enumerate(padded_records):
                                        plt.plot(
                                            x,
                                            student_marks,
                                            label=f"{ids[i]}",
                                            color="blue",
                                            alpha=0.3,
                                        )

                                    plt.plot(
                                        x,
                                        average_marks,
                                        label="Average Marks",
                                        linewidth=3,
                                        color="red",
                                    )

                                    # Add labels and legend
                                    plt.xlabel("Attempts")
                                    plt.ylabel("Marks")
                                    plt.title(
                                        f"{normalize_name(data['lab_name'])} - Q{q_index}"
                                    )
                                    plt.savefig(
                                        f"plots/{course_name}/{class_name}/{normalize_name(data['lab_name'])}-Q{q_index}.png"
                                    )
                                    plt.clf()

                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from file: {file_path}: {e}")


def plot_all_questions(repo_id, course_name, class_name):
    plt.clf()

    all_padded_records = []
    global_max = find_global_max(repo_id, course_name, class_name)

    x = list(range(1, global_max + 1))

    for root, dirs, files in os.walk(repo_id):
        if class_name in dirs:
            class_dir_path = os.path.join(root, class_name)

            for subdir_root, subdir_dirs, subdir_files in os.walk(class_dir_path):
                for file in subdir_files:
                    if file.endswith(".json"):
                        file_path = os.path.join(subdir_root, file)
                        try:
                            with open(file_path, "r") as json_file:
                                data = json.load(json_file)

                                ids = []
                                for answers in data["student_answers"]:
                                    ids.append(answers["id"])

                                for idx in range(len(data["list_questions"])):
                                    max_score = data["list_questions"][idx][
                                        "max_scores"
                                    ]
                                    q_index = idx + 1

                                    records = []
                                    for answers in data["student_answers"]:
                                        for answer in answers["response_history"]:
                                            marks = []
                                            if (
                                                answer["question"]
                                                == f"Question {q_index}"
                                            ):
                                                mark_per_attempt = []
                                                for score_idx in range(
                                                    len(answer["results"])
                                                ):
                                                    if (
                                                        answer["results"][score_idx][
                                                            "marks"
                                                        ]
                                                        != ""
                                                    ):
                                                        mark_per_attempt.append(
                                                            answer["results"][
                                                                score_idx
                                                            ]["marks"]
                                                        )

                                            marks.append(mark_per_attempt)
                                        records.extend(marks)

                                    try:
                                        records = [
                                            [
                                                float(mark) * 10 / max_score
                                                for mark in student_marks
                                            ]
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
                                                + [max(student_marks)]
                                                * (global_max - len(student_marks))
                                            )
                                            all_padded_records.append(
                                                student_marks
                                                + [max(student_marks)]
                                                * (global_max - len(student_marks))
                                            )
                                        else:
                                            padded_records.append([0] * global_max)
                                            all_padded_records.append([0] * global_max)

                                    padded_records = np.array(padded_records)

                                    # Plot each student's attempts
                                    for i, student_marks in enumerate(padded_records):
                                        plt.plot(
                                            x,
                                            student_marks,
                                            label=f"{ids[i]}",
                                            color="blue",
                                            alpha=0.2,
                                        )

                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from file: {file_path}: {e}")

        all_padded_records = np.array(all_padded_records)
        all_average_marks = np.nanmean(all_padded_records, axis=0)

        plt.plot(
            x, all_average_marks, label="All Average Marks", linewidth=3, color="Red"
        )
        plt.xlabel("Attempts")
        plt.ylabel("Marks")
        plt.title(f"All questions")
        plt.savefig(f"plots/{course_name}/{class_name}/all-questions.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk221"
    )
    parser.add_argument("--class_name", help="Class Name", type=str, default="L01")
    args = parser.parse_args()
    course_name = args.course_name
    class_name = args.class_name

    wandb.init(project="student-score-analysis")

    directory = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}_records", repo_type="dataset"
    )

    os.makedirs("plots", exist_ok=True)
    os.makedirs(f"plots/{course_name}", exist_ok=True)
    os.makedirs(f"plots/{course_name}/{class_name}", exist_ok=True)

    plot_per_question(directory, course_name, class_name)
    plot_all_questions(directory, course_name, class_name)

    wandb.finish()


if __name__ == "__main__":
    main()
