"""
This is the main process for analyzing crawled data.
"""

import os
import json
import argparse
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--course_name", help="Class Name", type=str, default="DSA-HK231")
parser.add_argument("--class_name", help="Class Name", type=str, default="L09")
args = parser.parse_args()


def plot_questions(course_name, class_name):
    """
    This function plots student grades per question.
    """
    for json_file in os.listdir(f"data/{course_name}/{class_name}"):
        if not json_file.endswith(".json"):
            continue

        name = json_file
        json_file = os.path.join(f"data/{course_name}/{class_name}", json_file)
        with open(json_file, "r", encoding="utf8") as f:
            data = json.load(f)

    # Question q
    for q in range(len(data["max_scores"])):
        plt.clf()

        max_score = data["max_scores"][q]
        attemps = data["attemps"]
        ids = [attemp["id"] for attemp in attemps]
        records = [attemp["records"][q] for attemp in attemps]
        try:
            records = [
                [float(mark) * 10 / max_score for mark in student_marks]
                for student_marks in records
            ]
        except RuntimeError:
            continue

        # Find the maximum number of attempts across all students
        max_attempts = max(len(student_marks) for student_marks in records)

        # Generate x values (attempts)
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
        # Convert the list of lists to a NumPy array
        padded_records = np.array(padded_records)
        # Calculate the average marks for each attempt number across all students
        average_marks = np.nanmean(padded_records, axis=0)

        # Plot each student's attempts
        for i, student_marks in enumerate(padded_records):
            plt.plot(x, student_marks, label=f"{ids[i]}", color="blue", alpha=0.3)

        # Plot the average marks as a bold line
        plt.plot(x, average_marks, label="Average Marks", linewidth=3, color="blue")

        # Add labels and legend
        plt.xlabel("Attempts")
        plt.ylabel("Marks")
        plt.title(f"{name} - Q{q+1}")

        plt.savefig(f"plots/{course_name}/{class_name}/{name}-Q{q+1}.png")
        plt.close()


def plot_all_questions(course_name, class_name):
    """
    This function plots all student grades of all questions in each week.
    """
    global_max = 0
    for json_file in os.listdir(f"data/{course_name}/{class_name}"):
        if not json_file.endswith(".json"):
            continue

        name = json_file
        print(name)
        json_file = os.path.join(f"data/{course_name}/{class_name}", json_file)
        with open(json_file, "r", encoding="utf8") as f:
            data = json.load(f)

        # Question q
        for q in range(len(data["max_scores"])):

            max_score = data["max_scores"][q]
            attemps = data["attemps"]
            ids = [attemp["id"] for attemp in attemps]
            records = [attemp["records"][q] for attemp in attemps]
            try:
                records = [
                    [float(mark) * 10 / max_score for mark in student_marks]
                    for student_marks in records
                ]
            except RuntimeError:
                continue

            # Find the maximum number of attempts across all students
            max_attempts = max(len(student_marks) for student_marks in records)
            global_max = max(global_max, max_attempts)

    plt.clf()

    all_padded_records = []

    for json_file in os.listdir(f"data/{course_name}/{class_name}"):
        if not json_file.endswith(".json"):
            continue

        name = json_file
        json_file = os.path.join(f"data/{course_name}/{class_name}", json_file)
        with open(json_file, "r", encoding="utf8") as f:
            data = json.load(f)

        # Question q
        for q in range(len(data["max_scores"])):

            max_score = data["max_scores"][q]
            attemps = data["attemps"]
            ids = [attemp["id"] for attemp in attemps]
            records = [attemp["records"][q] for attemp in attemps]
            try:
                if name == "Graph.json":
                    max_score = 2
                records = [
                    [float(mark) * 10 / max_score for mark in student_marks]
                    for student_marks in records
                ]
            except RuntimeError:
                continue

            # Generate x values (attempts)
            x = list(range(1, global_max + 1))

            # Find the average score for each number of attemps
            # Pad the marks of each student with highest marks
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

            # Convert the list of lists to a NumPy array
            padded_records = np.array(padded_records)

            # Plot each student's attempts
            for i, student_marks in enumerate(padded_records):
                plt.plot(x, student_marks, label=f"{ids[i]}", color="blue", alpha=0.2)

    all_padded_records = np.array(all_padded_records)
    all_average_marks = np.nanmean(all_padded_records, axis=0)
    plt.plot(x, all_average_marks, label="All Average Marks", linewidth=3, color="Red")

    # Add labels and legend
    plt.xlabel("Attempts")
    plt.ylabel("Marks")
    plt.title(f"All questions ({class_name})")

    plt.savefig(f"plots/{course_name}/{class_name}/all-questions.png")
    plt.close()


def plot_weeks(weeks, course_name, class_name):
    """
    This function plots all student grades of all questions across all weeks.
    """
    # Get all unique student IDs
    ids_set = set()
    for week in weeks:
        for path in week:
            with open(
                os.path.join(f"data/{course_name}/{class_name}", path),
                "r",
                encoding="utf8",
            ) as f:
                data = json.load(f)
                attemps = data["attemps"]
                ids = [attemp["id"] for attemp in attemps]
            ids_set.update(ids)
            print("Week:", week, "->", len(ids))
    print(f"Total -> {len(ids_set)}")
    student_ids = list(ids_set)

    students = {sid: [] for sid in student_ids}
    for week in weeks[:]:
        for sid in student_ids:
            students[sid].append([])
        for exercise in week[:]:
            with open(
                os.path.join(f"data/{course_name}/{class_name}", exercise),
                "r",
                encoding="utf8",
            ) as f:
                data = json.load(f)
            attemps = data["attemps"]

            this_exercise_result = {}
            ids_do_exercise = []
            for attemp in attemps:
                sid = attemp["id"]
                records = attemp["records"]
                records: list[list]

                maxes = []
                for question_marks in records:

                    # Special cases
                    if exercise == "Graph.json":
                        question_marks = [
                            float(mark) * 5 / 7 for mark in question_marks
                        ]
                    if exercise == "Week_5_Exam.json":
                        question_marks = [
                            float(mark) * 10 / 11 for mark in question_marks
                        ]

                    if question_marks:
                        try:
                            maxes.append(max(float(mark) for mark in question_marks))
                        except RuntimeError:
                            maxes.append(0)
                    else:
                        maxes.append(0)

                result = sum(maxes)

                student_last_result = this_exercise_result.get(sid, 0)
                if result >= student_last_result:
                    this_exercise_result[sid] = result

                ids_do_exercise.append(sid)

            for sid in student_ids:
                if sid in ids_do_exercise:
                    students[sid][-1].append(this_exercise_result[sid])
                else:
                    students[sid][-1].append(0)

    grouped_by_week = students.copy()
    for rid, records in grouped_by_week.items():
        grouped_by_week[rid] = [sum(record) / len(record) for record in records]

    # Calculate the average result for each test
    num_tests = 6
    avg_results = [
        sum(all_student_scores_in_one_week) / len(all_student_scores_in_one_week)
        for all_student_scores_in_one_week in zip(*grouped_by_week.values())
    ]

    # Generate x values (test order)
    test_order = list(range(1, num_tests + 1))

    # Plot each student's test results
    for wid, week_avg in grouped_by_week.items():
        plt.plot(test_order, week_avg, label=f"Student {wid}", color="blue", alpha=0.3)

    # Plot the average line
    plt.plot(test_order, avg_results, label="Average", linewidth=2, color="black")

    # Add labels and title
    plt.xlabel("Week")
    plt.ylabel("Result")
    plt.title("Weekly Results of Students")

    # Show the plot
    plt.grid(True)
    plt.savefig(f"plots/{course_name}/{class_name}/all-weeks.png")
    plt.close()


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    os.makedirs(f"plots/{args.course_name}", exist_ok=True)
    os.makedirs(f"plots/{args.course_name}/{args.class_name}", exist_ok=True)

    plot_questions(args.course_name, args.class_name)

    plot_all_questions(args.course_name, args.class_name)

    weeks = [
        [
            "OOP_Review.json",
            "Recursion.json",
            "Array_List.json",
            "Singly_Linked_List.json",
        ],
        [
            "Week_2_Exam.json",
            "Doubly_Linked_List.json",
            "Stack.json",
            "Queue.json",
            "Sorting_(Easy).json",
        ],
        [
            "Week_3_Exam.json",
            "Sorting_(Advance).json",
            "Binary_Tree.json",
            "Binary_Search_Tree.json",
        ],
        ["Week_4_Exam.json", "AVL_Tree.json", "B-Tree.json"],
        ["Week_5_Exam.json", "Heap.json", "Search.json"],
        ["Week_6_Exam.json", "Graph.json", "Hash.json"],
    ]

    plot_weeks(weeks, args.course_name, args.class_name)
