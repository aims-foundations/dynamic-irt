import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, unquote
import shutil
from datasets import config
import pickle

def find_json_files(repo_id):
    directories = []
    base_url = f"https://huggingface.co/datasets/{repo_id}/tree/main/"
    response = requests.get(base_url)

    directory_json_files = {}

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            if "/tree/main/" in href and href.count('/') == 6:
                directory_name = href.split('/')[-1]
                directories.append(directory_name)

    for directory in directories:
        dir_url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{directory}"
        response = requests.get(dir_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            json_files = []
            for link in links:
                href = link['href']
                if href.endswith('.json'):
                    path = urlsplit(href).path
                    filename = path.split('/')[-1]
                    filename = unquote(filename)
                    json_files.append(filename)

            if json_files:
                directory_json_files[directory] = json_files

    return directory_json_files

# def find_total_student(repo_id, directory_json_files):
#     student_set = set()

#     for directory, files in directory_json_files.items():
#         for file in files:
#             data_s = load_dataset(repo_id, data_files=f"{directory}/{file}", field='student_answers')
            
#             student_set.update([answer['id'] for answer in data_s['train'] if 'id' in answer])

#     return len(student_set)

def format_dataset(repo_id, directory_json_files):
    question_attempts = {}
    student_id_to_index = {}
    correctness = []

    for directory, files in directory_json_files.items():
        for file in files:
            data_s = load_dataset(repo_id, data_files=f"{directory}/{file}", field='student_answers')
            base_name, _ = os.path.splitext(file)
            for answer in data_s['train']:
                student_id = answer['id']
                if student_id not in student_id_to_index:
                    student_id_to_index[student_id] = len(student_id_to_index)
                process_answer(answer, base_name, directory, question_attempts, student_id_to_index, correctness)

    question_index = {qid: idx for idx, qid in enumerate(sorted(question_attempts.keys()))}
    N = len(student_id_to_index)
    Q = len(question_index)
    T = max(question_attempts.values(), default=0)
    correctness_matrix = np.full((N, Q, T), np.nan)

    for entry in correctness:
        correctness_matrix[entry['s_idx'], entry['q_idx'], entry['t']] = entry['score']

    return correctness_matrix

def process_answer(answer, base_name, directory, question_attempts, student_id_to_index, correctness):
    for response in answer['response_history']:
        question_id = f"{directory}_{base_name}_{response['question'].split()[-1]}"
        num_attempts = len(response['results'])
        question_attempts[question_id] = max(question_attempts.get(question_id, 0), num_attempts)
        s_idx = student_id_to_index[answer['id']]
        q_idx = question_attempts[question_id]

        for t, result in enumerate(response['results']):
            score = float(result['marks']) if result['marks'] != "" else float(-1)
            correctness.append({'s_idx': s_idx, 'q_idx': q_idx, 't': t, 'score': score})

if __name__ == "__main__":
    repo_id = "stair-lab/dsa_records"

    directory_json_files = find_json_files(repo_id)
    correctness_matrix = format_dataset(repo_id, directory_json_files)

    with open('correctness_matrix.pkl', 'wb') as file:
        pickle.dump(correctness_matrix, file)

    nan_mask = np.isnan(correctness_matrix)
    nan_count = np.sum(nan_mask)
    print("Total number of NaN values:", nan_count)

    cache_dir = config.HF_DATASETS_CACHE    
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
