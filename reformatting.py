import os
import json
import numpy as np
import pandas as pd

def find_total_student(directory):
    student_set = set()

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        if 'student_answers' in data:
                            for answer in data['student_answers']:
                                student_id = answer.get('id')
                                if student_id:
                                    student_set.add(student_id)
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")

    return len(student_set)

def format_dataset(directory):
    question_attempts = {}
    student_id_to_index = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        for student in data.get('student_answers', []):
                            student_id = student.get('id')
                            if student_id not in student_id_to_index:
                                student_id_to_index[student_id] = len(student_id_to_index)
                            for response in student.get('response_history', []):
                                question_id = f"{folder_name}_{data.get('lab_name', '')}_{response['question'].split()[-1]}"
                                num_attempts = len(response.get('results', []))
                                question_attempts[question_id] = max(question_attempts.get(question_id, 0), num_attempts)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file: {file_path}: {e}")

    question_index = {qid: idx for idx, qid in enumerate(sorted(question_attempts.keys()))}
    N = len(student_id_to_index)
    Q = len(question_index)
    T = max(question_attempts.values(), default=0)
    correctness = np.full((N, Q, T), np.nan)
    # print(N, Q, T)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(file_path))
                try:
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        for student in data.get('student_answers', []):
                            student_id = student.get('id')
                            if student_id and student_id in student_id_to_index:
                                s_idx = student_id_to_index[student_id]
                                for response in student.get('response_history', []):
                                    question_id = f"{folder_name}_{data.get('lab_name', '')}_{response['question'].split()[-1]}"
                                    if question_id in question_index:
                                        q_idx = question_index[question_id]
                                        for t, result in enumerate(response.get('results', [])):
                                            score = float(result.get('marks', 0)) if result.get('state') == 'Correct' else 0
                                            correctness[s_idx, q_idx, t] = score
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file: {file_path}: {e}")

    return correctness

if __name__ == "__main__":
    directory = 'dsa_records'
    data_array = format_dataset(directory)
    print(data_array)