import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from datasets import load_dataset

def load_data_from_huggingface(dataset_name, file_path):
    dataset_question = load_dataset(dataset_name, data_files=file_path, field='list_questions')
    dataset_student = load_dataset(dataset_name, data_files=file_path, field='student_answers')
    return dataset_question, dataset_student

def extract_data(dataset_questions, dataset_students):
    question_index = random.randint(0, len(dataset_questions['train']) - 1)

    student_data = {}
    for student_answers in dataset_students['train']:
        for response in student_answers['response_history']:
            if response['question'] == f'Question {question_index + 1}':
                attempts = [(i+1, result['state'] == 'Correct') for i, result in enumerate(response['results'])]
                student_data[student_answers['name']] = attempts
    return student_data

def plot(student_data):
    plt.figure(figsize=(12, 8))
    attempt_dict = {}

    for student, attempts in student_data.items():
        x, y = zip(*attempts) if attempts else ([], [])
        y_correct = [1 if correct else 0 for correct in y]
        x_floats = np.array(x) + np.random.normal(0, 0.1, size=len(x))

        for xi, yi in zip(x_floats, y_correct):
            plt.plot([xi, xi], [0, yi], color='blue', alpha=0.5)

        for i, correct in zip(x, y_correct):
            if i not in attempt_dict:
                attempt_dict[i] = []
            attempt_dict[i].append(correct)

    if attempt_dict:
        avg_x = sorted(attempt_dict.keys())
        avg_y = [np.mean(attempt_dict[att]) for att in avg_x]

        def sigmoid(x, a, b, c):
            return a / (1.0 + np.exp(-c * (x - b)))

        initial_guesses = [1, np.median(avg_x), 1]
        popt, _ = curve_fit(sigmoid, avg_x, avg_y, p0=initial_guesses, maxfev=5000)

        xs = np.linspace(min(avg_x), max(avg_x), 300)
        ys = sigmoid(xs, *popt)

        plt.plot(xs, ys, label='Average', color='red', linewidth=2)

    plt.title(f"Performance of {len(student_data)} students")
    plt.xlabel("Attempt Number")
    plt.ylabel("Correctness")
    plt.legend()
    plt.grid(True)
    plt.savefig("image.png")
    plt.close()
    print("Plot saved as 'image.png'")


dataset_name = "stair-lab/dsa-records"
file_path = {"train": "CC02/LAB4.json"}

dataset_questions, dataset_students = load_data_from_huggingface(dataset_name, file_path)
if dataset_questions and dataset_students:
    student_data = extract_data(dataset_questions, dataset_students)
    if student_data:
        plot(student_data)
    else:
        print("No student data found for the question.")
else:
    print("Failed to load dataset.")