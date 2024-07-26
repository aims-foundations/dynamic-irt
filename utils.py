"""
This file provides support functions.
"""
import os
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

def load_data(namepath):
    """
        Load data.
    """
    data = []
    directory = f"data/{namepath}"
    for json_file in os.listdir(directory):
        if json_file.endswith(".json"):
            file_path = os.path.join(directory, json_file)
            with open(file_path, "r", encoding='utf-8') as file:
                data.append(json.load(file))
    return data

def process_scores(data):
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

def plot_scores(namepath, processed_data):
    """
        Plot scores.
    """
    for ids, records in processed_data:
        max_attempts = max(len(student_marks) for student_marks in records)
        x_value = list(range(1, max_attempts + 1))

        padded_records = np.array([
            marks + [max(marks)] * (max_attempts - len(marks)) if marks else [0] * max_attempts
            for marks in records
        ])
        average_marks = np.nanmean(padded_records, axis=0)

        plt.figure()
        for i, student_marks in enumerate(padded_records):
            plt.plot(x_value, student_marks, label=f"{ids[i]}", color="blue", alpha=0.3)
        plt.plot(x_value, average_marks, label="Average Marks", linewidth=3, color="blue")

        plt.xlabel("Attempts")
        plt.ylabel("Marks")
        plt.title(f"{namepath} - Analysis")
        plt.legend()
        plt.savefig(f"plots/{namepath}.png")
        plt.close()

def load_and_filter_data(namepath):
    """
        Load and filter data.
    """
    all_data = []
    for json_file in os.listdir(f"data/{namepath}"):
        if json_file.endswith(".json"):
            file_path = os.path.join(f"data/{namepath}", json_file)
            with open(file_path, "r", encoding='utf-8') as file:
                data = json.load(file)
                all_data.append((json_file, data))
    return all_data

def process_records(data, global_max):
    """
        Process records.
    """
    processed_records = []
    for dataset in data:
        for question in range(len(dataset["max_scores"])):
            max_score = dataset["max_content"][question]
            attempts = dataset["attempts"]
            ids = [attempt["id"] for attempt in attempts]
            records = [attempt["records"][question] for attempt in attempts]
            # Adjusting scores
            adjusted_records = [
                [float(mark) * 10 / max_score for mark in student_marks]
                for student_marks in records
            ]
            # Padding records
            padded_records = pad_records(adjusted_records, global_max)
            processed_records.append((ids, padded_records))
    return processed_records

def pad_records(records, max_attempts):
    """
        Pad records.
    """
    padded = []
    for student_marks in records:
        if student_marks:
            padded.append(student_marks +
                [max(student_marks)] * (max_attempts - len(student_marks)))
        else:
            padded.append([0] * max_attempts)
    return padded

def plot_data(namepath, processed_records, global_max):
    """
        Plot data.
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
    plt.plot(x_value, all_average_marks, label="All Average Marks", linewidth=3, color="Red")

    plt.xlabel("Attempts")
    plt.ylabel("Marks")
    plt.title(f"All questions ({namepath})")
    plt.legend()
    plt.savefig(f"plots/{namepath}/all-questions.png")
    plt.close()

def load_json_data(filepath):
    """
        Load json data.
    """
    with open(filepath, "r", encoding='utf-8') as file:
        return json.load(file)

def collect_student_ids(weeks, namepath):
    """
        Collect student ids.
    """
    ids_set = set()
    for week in weeks:
        for exercise in week:
            data = load_json_data(os.path.join(f"data/{namepath}", exercise))
            ids_set.update(attempt["id"] for attempt in data["attempts"])
    return ids_set

def process_scores_in_weeks(weeks, student_ids, namepath):
    """
        Process scores in weeks.
    """
    students = {student_id: [] for student_id in student_ids}
    for week in weeks:
        for student_id in student_ids:
            students[student_id].append([])
        for exercise in week:
            data = load_json_data(os.path.join(f"data/{namepath}", exercise))
            process_exercise(data, students)
    return students

def process_exercise(data, students):
    """
        Process exercise.
    """
    for attempt in data["attempts"]:
        student_id = attempt["id"]
        max_scores = calculate_max_scores(attempt["records"], data["exercise"])
        # Update the student's record for the current week
        students[student_id][-1].append(max(max_scores, default=0))

def calculate_max_scores(records, exercise_name):
    """
        Calculate max scores.
    """
    for record in records:
        if exercise_name == "Graph.json":
            yield max(float(mark) * 5 / 7 for mark in record)
        elif exercise_name == "Week_5_Exam.json":
            yield max(float(mark) * 10 / 11 for mark in record)
        else:
            yield max(record, default=0, key=float)

def calculate_weekly_averages(students):
    """
        Calculate weekly averages.
    """
    return {student_id: [sum(week) / len(week) for week in weeks_scores]
        for student_id, weeks_scores in students.items()}

def plot_results(students, num_tests):
    """
        Plot results.
    """
    test_order = list(range(1, num_tests + 1))
    for student_id, scores in students.items():
        plt.plot(test_order, scores, label=f"Student {student_id}", color="blue", alpha=0.3)
    avg_results = [sum(scores) / len(scores) for scores in zip(*students.values())]
    plt.plot(test_order, avg_results, label="Average", linewidth=2, color="black")
    plt.xlabel("Week")
    plt.ylabel("Result")
    plt.title("Weekly Results of Students")
    plt.legend()
    plt.grid(True)
    plt.show()
    