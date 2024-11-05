"""
Run this file to get students' score analysis.
"""

import argparse
import json
import os

from utils import (
    plot_scores_all_questions,
    plot_scores_per_question,
    plot_scores_per_week,
    process_scores_all_questions,
    process_scores_per_question,
    process_scores_per_week,
)


def plot_per_question(course_question, class_question):
    """
    This function plots student grades per question.
    """
    namepath = f"{course_question}/{class_question}"
    data = []
    directory = f"data/{namepath}"
    for json_file in os.listdir(directory):
        if json_file.endswith(".json"):
            file_path = os.path.join(directory, json_file)
            with open(file_path, "r", encoding="utf-8") as file:
                data.append(json.load(file))
    processed_data = process_scores_per_question(data)
    plot_scores_per_question(namepath, processed_data)


def plot_all_questions(course_all_questions, class_all_questions):
    """
    This function plots student grades for all questions.
    """
    namepath = f"{course_all_questions}/{class_all_questions}"
    all_data = []
    for json_file in os.listdir(f"data/{namepath}"):
        if json_file.endswith(".json"):
            file_path = os.path.join(f"data/{namepath}", json_file)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_data.append((json_file, data))
    global_max = max(
        len(attempt["records"])
        for _, dataset in all_data
        for attempt in dataset["attempts"]
    )
    processed_records = process_scores_all_questions(all_data, global_max)
    plot_scores_all_questions(namepath, processed_records, global_max)


def plot_weeks(weeks, course_weeks, class_weeks):
    """
    This function plots student grades in weeks.
    """
    namepath = f"{course_weeks}/{class_weeks}"
    student_ids = set()
    for week in weeks:
        for exercise in week:
            with open(
                os.path.join(f"data/{namepath}", exercise), "r", encoding="utf-8"
            ) as file:
                data = json.load(file)
            student_ids.update(attempt["id"] for attempt in data["attempts"])
    students = process_scores_per_week(weeks, student_ids, namepath)
    weekly_averages = {
        student_id: [sum(week) / len(week) for week in weeks_scores]
        for student_id, weeks_scores in students.items()
    }
    plot_scores_per_week(weekly_averages, len(weeks[0]))


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Class Name", type=str, default="DSA-HK231"
    )
    parser.add_argument("--class_name", help="Class Name", type=str, default="L09")
    args = parser.parse_args()
    course_name = args.course_name
    class_name = args.class_name
    os.makedirs("plots", exist_ok=True)
    os.makedirs(f"plots/{course_name}", exist_ok=True)
    os.makedirs(f"plots/{course_name}/{class_name}", exist_ok=True)

    plot_per_question(course_name, class_name)
    plot_all_questions(course_name, class_name)

    weeks_file_list = [
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

    plot_weeks(weeks_file_list, course_name, class_name)


if __name__ == "__main__":
    main()
