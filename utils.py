"""
This file provides support functions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt


def parse_score(name):
    """
    Parse score.
    """
    return float(name.split("\n")[1][1:])


def check(data):
    """
    Check.
    """
    data = data["attemps"]
    print(f"Checking: n={len(data)}, sample: {data[0]}")


def process_scores_per_question(data):
    """
    Process scores.
    """
    processed_data = []
    for dataset in data:
        for question in range(len(dataset["max_scores"])):
            max_score = dataset["max_scores"][question]
            attempts = dataset["attempts"]
            ids = [attempt["id"] for attempt in attempts]
            records = [attempt["records"][question] for attempt in attempts]
            scaled_records = [
                [float(mark) * 10 / max_score for mark in student_marks]
                for student_marks in records
            ]
            processed_data.append((ids, scaled_records))
    return processed_data


def plot_scores_per_question(namepath, processed_data):
    """
    Plot scores.
    """
    for ids, records in processed_data:
        max_attempts = max(len(student_marks) for student_marks in records)
        x_value = list(range(1, max_attempts + 1))

        padded_records = np.array(
            [
                (
                    marks + [max(marks)] * (max_attempts - len(marks))
                    if marks
                    else [0] * max_attempts
                )
                for marks in records
            ]
        )
        average_marks = np.nanmean(padded_records, axis=0)

        plt.figure()
        for i, student_marks in enumerate(padded_records):
            plt.plot(x_value, student_marks, label=f"{ids[i]}", color="blue", alpha=0.3)
        plt.plot(
            x_value, average_marks, label="Average Marks", linewidth=3, color="blue"
        )

        plt.xlabel("Attempts")
        plt.ylabel("Marks")
        plt.title(f"{namepath} - Analysis")
        plt.legend()
        plt.savefig(f"plots/{namepath}.png")
        plt.close()


def process_scores_all_questions(data, global_max):
    """
    Process records.
    """
    processed_records = []
    for dataset in data:
        for question in range(len(dataset["max_scores"])):
            # max_score = dataset["max_content"][question]
            attempts = dataset["attempts"]
            ids = [attempt["id"] for attempt in attempts]
            records = [attempt["records"][question] for attempt in attempts]
            # Adjusting scores
            # adjusted_records = [
            #     [float(mark) * 10 / max_score for mark in student_marks]
            #     for student_marks in records
            # ]
            # Padding records
            padded_records = []
            for student_marks in records:
                if student_marks:
                    padded_records.append(
                        student_marks
                        + [max(student_marks)] * (global_max - len(student_marks))
                    )
                else:
                    padded_records.append([0] * global_max)
            processed_records.append((ids, padded_records))
    return processed_records


def plot_scores_all_questions(namepath, processed_records, global_max):
    """
    Plot scores all questions.
    """
    plt.clf()
    all_padded_records = []
    x_value = list(range(1, global_max + 1))

    for ids, records in processed_records:
        for i, student_marks in enumerate(records):
            plt.plot(x_value, student_marks, label=f"{ids[i]}", color="blue", alpha=0.2)
            all_padded_records.append(student_marks)

    all_padded_records = np.array(all_padded_records)
    all_average_marks = np.mean(all_padded_records, axis=0)
    plt.plot(
        x_value, all_average_marks, label="All Average Marks", linewidth=3, color="Red"
    )

    plt.xlabel("Attempts")
    plt.ylabel("Marks")
    plt.title(f"All questions ({namepath})")
    plt.legend()
    plt.savefig(f"plots/{namepath}/all-questions.png")
    plt.close()


def process_scores_per_week(weeks, student_ids, namepath):
    """
    Process scores in weeks.
    """
    students = {student_id: [] for student_id in student_ids}
    for week in weeks:
        for student_id in student_ids:
            students[student_id].append([])
        for exercise in week:
            with open(
                os.path.join(f"data/{namepath}", exercise), "r", encoding="utf-8"
            ) as file:
                data = json.load(file)
            for attempt in data["attempts"]:
                student_id = attempt["id"]
                for record in attempt["records"]:
                    if data["exercise"] == "Graph.json":
                        max_scores = max(float(mark) * 5 / 7 for mark in record)
                    elif data["exercise"] == "Week_5_Exam.json":
                        max_scores = max(float(mark) * 10 / 11 for mark in record)
                    else:
                        max_scores = max(record, default=0, key=float)
                students[student_id][-1].append(max(max_scores, default=0))
    return students


def plot_scores_per_week(students, num_tests):
    """
    Plot results.
    """
    test_order = list(range(1, num_tests + 1))
    for student_id, scores in students.items():
        plt.plot(
            test_order, scores, label=f"Student {student_id}", color="blue", alpha=0.3
        )
    avg_results = [sum(scores) / len(scores) for scores in zip(*students.values())]
    plt.plot(test_order, avg_results, label="Average", linewidth=2, color="black")
    plt.xlabel("Week")
    plt.ylabel("Result")
    plt.title("Weekly Results of Students")
    plt.legend()
    plt.grid(True)
    plt.show()
