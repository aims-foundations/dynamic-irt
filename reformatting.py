import os
import json
import numpy as np

def load_and_process_data(directory):
    question_attempts = {}
    total_students = 0

    # Process each JSON file in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)

                        # Update total_students count
                        total_students += len(data['student_answers'])

                        # Update the question_attempts dictionary
                        for student in data['student_answers']:
                            for response in student['response_history']:
                                # Construct question_id with folder name, lab name, and question part
                                question_id = f"{folder_name}_{data['lab_name']}_{response['question'].split()[-1]}"
                                num_attempts = len(response['results'])
                                if question_id not in question_attempts:
                                    question_attempts[question_id] = num_attempts
                                else:
                                    question_attempts[question_id] = max(question_attempts[question_id], num_attempts)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {file_path}")

    # Now create the question_index mapping
    question_index = {qid: idx for idx, qid in enumerate(sorted(question_attempts.keys()))}

    # Dimensions for the 3D array
    N = total_students
    Q = len(question_index)  # Use the length of question_index
    T = max(question_attempts.values())
    correctness = np.full((N, Q, T), np.nan)

    # Reset student index for array filling
    current_student_index = 0

    # Reprocess files to fill the correctness array
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        for student in data['student_answers']:
                            for response in student['response_history']:
                                question_id = f"{folder_name}_{data['lab_name']}_{response['question'].split()[-1]}"
                                q_idx = question_index[question_id]  # Safely get the index for the question
                                for t, result in enumerate(response['results']):
                                    score = float(result['marks']) if result['state'] == 'Correct' else 0
                                    correctness[current_student_index, q_idx, t] = score
                            current_student_index += 1
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {file_path}")

    return correctness

# Usage
directory = 'dsa_records'
data_array = load_and_process_data(directory)
print(data_array)