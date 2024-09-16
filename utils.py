import json
import os
import re
from urllib.parse import unquote, urlsplit


def parse_score(header_text):
    search_result = re.search(r"/(\d+\.\d+)", header_text)
    if search_result:
        return float(search_result.group(1))
    else:
        return None


def filter_class_group(details, sid, class_name):
    for detail in details:
        if detail["Class Group"] == "No groups":
            if detail["ID"] == sid:
                print(
                    f"Processing student ID: {detail['ID']} with only a class group."
                )
                return True
        
        if detail["ID"] == sid and detail["Class Group"] == class_name:
            print(
                f"Processing student ID: {detail['ID']} from Class Group: {detail['Class Group']}"
            )
            return True

    return False


def find_global_max(repo_id, course_name, class_name):
    global_max = 0

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
                                    max_score = data["list_questions"][idx]["max_scores"]
                                    q_index = idx + 1
                        
                                    records = []
                                    for answers in data["student_answers"]:
                                        for answer in answers["response_history"]:
                                            marks = []
                                            if answer["question"] == f"Question {q_index}":
                                                mark_per_attempt = []
                                                for score_idx in range(len(answer["results"])):
                                                    if answer["results"][score_idx]["marks"] != "":
                                                        mark_per_attempt.append(
                                                            answer["results"][score_idx]["marks"]
                                                        )
                        
                                            marks.append(mark_per_attempt)
                                        records.extend(marks)
                        
                                    try:
                                        records = [
                                            [float(mark) * 10 / max_score for mark in student_marks]
                                            for student_marks in records
                                        ]
                                    except:
                                        continue
                        
                                    max_attempts = [len(student_marks) for student_marks in records]
                                    global_max = max(global_max, max(max_attempts))

                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from file: {file_path}: {e}")
    return global_max