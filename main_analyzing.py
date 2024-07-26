"""
Run this file to get students' score analysis.
"""
import os
import argparse
from utils import (
    load_data, process_scores, plot_scores, load_and_filter_data,
    process_records, plot_data, collect_student_ids,
    calculate_weekly_averages, plot_results,
    process_scores_in_weeks
)

def plot_questions(course_question, class_question):
    """
    This function plots student grades per question.
    """
    namepath = f"{course_question}/{class_question}"
    data = load_data(namepath)
    processed_data = process_scores(data)
    plot_scores(namepath, processed_data)

def plot_all_questions(course_all_questions, class_all_questions):
    """
    This function plots student grades for all questions.
    """
    namepath = f"{course_all_questions}/{class_all_questions}"
    data = load_and_filter_data(namepath)
    global_max = max(
        len(attempt["records"])
        for _, dataset in data
        for attempt in dataset["attempts"]
    )
    processed_records = process_records(data, global_max)
    plot_data(namepath, processed_records, global_max)

def plot_weeks(weeks, course_weeks, class_weeks):
    """
    This function plots student grades in weeks.
    """
    namepath = f"{course_weeks}/{class_weeks}"
    student_ids = collect_student_ids(weeks, namepath)
    students = process_scores_in_weeks(weeks, student_ids, namepath)
    weekly_averages = calculate_weekly_averages(students)
    plot_results(weekly_averages, len(weeks[0]))

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--course_name", help="Class Name", type=str, default="DSA-HK231")
    parser.add_argument("--class_name", help="Class Name", type=str, default="L09")
    args = parser.parse_args()
    course_name = args.course_name
    class_name = args.class_name
    os.makedirs("plots", exist_ok=True)
    os.makedirs(f"plots/{course_name}", exist_ok=True)
    os.makedirs(f"plots/{course_name}/{class_name}", exist_ok=True)

    plot_questions(course_name, class_name)
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
    