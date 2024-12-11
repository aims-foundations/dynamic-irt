import io
import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm
from utils import parse_time


def format_dataset(repo_id, directory_json_files):
    question_attempts = []
    student_id_to_index = {}
    all_questions = {}
    all_question_info = {}

    for directory, files in directory_json_files.items():
        for file in files:
            try:
                week_idx = int(file.split("_")[0][1:])
                topic = file[file.find("_") + 1 : file.find(".json")]
            except:
                continue

            data_q = json.load(open(f"{repo_id}/{directory}/{file}", "r"))[
                "list_questions"
            ]

            for q in data_q:
                if isinstance(q, list):
                    for sq in q:
                        if sq["name"] not in all_questions:
                            all_questions[sq["name"]] = (
                                sq["question"],
                                sq["template"],
                                sq["testcases"],
                            )
                            all_question_info[sq["name"]] = {
                                "qname": sq["name"],
                                "week": week_idx,
                                "topic": topic,
                            }
                else:
                    if q["name"] not in all_questions:
                        all_questions[q["name"]] = (
                            q["question"],
                            q["template"],
                            q["testcases"],
                        )
                        all_question_info[q["name"]] = {
                            "qname": q["name"],
                            "week": week_idx,
                            "topic": topic,
                        }

            data_s = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="student_answers"
            )
            base_name, _ = os.path.splitext(file)

            for answer in data_s["train"]:
                student_id = answer["id"]
                if student_id not in student_id_to_index:
                    student_id_to_index[student_id] = len(student_id_to_index)

                for response_history in answer["response_history"]:
                    num_attempts = len(response_history["results"]) - 2
                    question_attempts.append(num_attempts)

    student_ids = [{}] * len(student_id_to_index)
    for sid, idx in student_id_to_index.items():
        student_ids[idx] = {"student_id": sid}

    question_name2idx = {qid: idx for idx, qid in enumerate(all_questions)}
    unique_questions = list(all_questions.values())
    # all_question_info = pd.DataFrame(all_question_info)
    all_question_info = list(all_question_info.values())

    N = len(student_id_to_index)
    Q = len(unique_questions)
    T = max(question_attempts, default=0)
    correctness_matrix = np.full((N, Q, T), -1).tolist()
    is_exam_matrix = np.full((N, Q, T), -1).tolist()
    correctness_bytc_matrix = np.full((N, Q, T), -1).tolist()
    time_matrix = np.full((N, Q, T), "").tolist()
    response_matrix = np.full((N, Q, T), "").tolist()

    for directory, files in directory_json_files.items():
        for file in files:
            is_exam = "exam" in file.lower()
            print(f"Processing {directory}/{file}")
            data_q = {
                "train": json.load(open(f"{repo_id}/{directory}/{file}", "r"))[
                    "list_questions"
                ]
            }
            data_s = load_dataset(
                repo_id, data_files=f"{directory}/{file}", field="student_answers"
            )

            question_content_map = {}
            for idx, q in enumerate(data_q["train"]):
                if isinstance(q, list):
                    for sub_idx, sq in enumerate(q):
                        question_content_map[f"Question {idx + 1}.{sub_idx + 1}"] = (
                            sq["name"],
                            sq["max_score"],
                        )
                else:
                    question_content_map[f"Question {idx + 1}"] = (
                        q["name"],
                        q["max_score"],
                    )

            for answer in data_s["train"]:
                s_idx = student_id_to_index[answer["id"]]
                if "class" not in student_ids[s_idx]:
                    student_ids[s_idx]["class"] = directory

                for response_history in answer["response_history"]:
                    current_question_name, question_max_score = (
                        question_content_map.get(response_history["question"], ("", 0))
                    )
                    if not current_question_name or float(question_max_score) == 0:
                        continue

                    q_idx = question_name2idx[current_question_name]

                    for t, result in enumerate(response_history["results"][1:-1]):
                        if result["score"] == "":
                            score = 0
                        elif float(result["score"]) > float(question_max_score):
                            print("Student score exceeds max score!")
                        else:
                            score = float(result["score"]) / float(question_max_score)

                        response = result["action"]
                        for prefix in ["Prechecked: ", "Saved: ", "Submit: "]:
                            if result["action"].startswith(prefix):
                                response = response.replace(prefix, "")

                        if "testcases" not in result:
                            print(
                                f"{directory}/{file} does not have 'testcase' results."
                            )
                            result["testcases"] = []

                        is_exam_matrix[s_idx][q_idx][t] = is_exam
                        correctness_matrix[s_idx][q_idx][t] = score
                        correctness_bytc_matrix[s_idx][q_idx][t] = result["testcases"]
                        time_matrix[s_idx][q_idx][t] = result["time"]
                        response_matrix[s_idx][q_idx][t] = response

    return (
        student_ids,
        unique_questions,
        all_question_info,
        question_name2idx,
        is_exam_matrix,
        correctness_matrix,
        correctness_bytc_matrix,
        time_matrix,
        response_matrix,
    )


def upload_files(repo_id, file_paths):
    api = HfApi()

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"Uploaded {file_name} successfully.")
        except Exception as e:
            print(f"Failed to upload {file_name}: {str(e)}")


def convert_to_tc_matrix(course_name, student_ids, y_obs, is_exam_matrix, time_matrix, device):
    # Number of questions
    n_question = len(y_obs[0])

    # Compute maximum number of testcases for each question
    list_max_testcases = {}
    for qidx in range(n_question):
        list_max_testcases[qidx] = 0
        for student in y_obs:
            list_n_testcases = []
            for x in student[qidx]:
                if not isinstance(x, int):
                    list_n_testcases.append(len(x))
                else:
                    list_n_testcases.append(0)
            list_max_testcases[qidx] = max(
                list_max_testcases[qidx], max(list_n_testcases)
            )

    # Flatten the testcases into questions
    # The new shape of y_obs is n_student x (n_question*n_testcase) x n_attempt
    y_tc_obs = []
    is_exam_obs = []
    time_tc_obs = []
    qidx_tc_obs = []
    tidx_tc_obs = []
    global_tidx = 0
    for qidx in range(n_question):
        for tidx in range(list_max_testcases[qidx]):
            qidx_tc_obs.append(qidx)
            tidx_tc_obs.append(global_tidx)
            global_tidx += 1
                
    for sidx, student in enumerate(tqdm(y_obs, desc="Preprocessing")):
        student_tc = []
        student_is_exam = []
        student_time = []
        for qidx in range(n_question):
            for tidx in range(list_max_testcases[qidx]):
                student_tc.append([])
                student_is_exam.append([])
                student_time.append([])

                for aidx, attempt in enumerate(student[qidx]):

                    if attempt == -1:
                        student_tc[-1].append(-1)
                        student_is_exam[-1].append(-1)
                        student_time[-1].append(-1)
                    else:
                        if len(attempt) == 0:
                            student_tc[-1].append(-1)
                            student_is_exam[-1].append(-1)
                            student_time[-1].append(-1)
                        elif (
                            tidx < len(attempt) and time_matrix[sidx][qidx][aidx] != ""
                        ):
                            student_tc[-1].append(attempt[tidx])
                            student_is_exam[-1].append(is_exam_matrix[sidx][qidx][aidx])
                            student_time[-1].append(
                                parse_time(time_matrix[sidx][qidx][aidx], course_name)
                            )
                        else:
                            student_tc[-1].append(-1)
                            student_is_exam[-1].append(-1)
                            student_time[-1].append(-1)
                            # raise ValueError("Testcase index out of bound")

        y_tc_obs.append(student_tc)
        is_exam_obs.append(student_is_exam)
        time_tc_obs.append(student_time)

    y_obs = torch.tensor(y_tc_obs, device=device, dtype=torch.int8)
    is_exam_obs = torch.tensor(is_exam_obs, device=device, dtype=torch.int8)
    time_obs = torch.tensor(time_tc_obs, device=device)
    qidx_obs = torch.tensor(qidx_tc_obs, device=device)
    
    # Remove students with no data
    accept_idxs = []
    for idx, row in enumerate(time_obs):
        if row.mean() > -1:
            accept_idxs.append(idx)
            
    y_obs = y_obs[accept_idxs]
    time_obs = time_obs[accept_idxs]
    is_exam_obs = is_exam_obs[accept_idxs]
    student_ids = [student_ids[idx] for idx in accept_idxs]
    
    # Remove questions with no data
    accept_idxs = []
    for tidx in range(y_obs.size(1)):
        all_submissions = y_obs[:, tidx, :]
        all_submissions = all_submissions[all_submissions == -1]
        if len(all_submissions) == 0:
            continue
        
        mean = all_submissions.float().mean()
        if mean != 0 and mean != 1:
            accept_idxs.append(tidx)
            
    y_obs = y_obs[:, accept_idxs, :]
    time_obs = time_obs[:, accept_idxs]
    is_exam_obs = is_exam_obs[:, accept_idxs]
    qidx_obs = qidx_obs[accept_idxs]

    return student_ids, y_obs, is_exam_obs, time_obs, qidx_obs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--course_name", help="Class Name", type=str, default="dsa_hk231"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download and load data
    data_folder = snapshot_download(
        repo_id=f"stair-lab/code_insights_jsons", repo_type="dataset"
    )
    data_folder = os.path.join(data_folder, args.course_name)
    directory_json_files = {}
    for folder in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, folder)):
            list_jsons = os.listdir(os.path.join(data_folder, folder))
            list_jsons = [x for x in list_jsons if x.endswith(".json")]
            directory_json_files[folder] = list_jsons

    (
        student_ids,
        unique_questions,
        question_infos,
        question_name2idx,
        is_exam_matrix,
        correctness_matrix,
        correctness_bytc_matrix,
        time_matrix,
        response_matrix,
    ) = format_dataset(data_folder, directory_json_files)

    student_ids, y_obs, is_exam_obs, time_obs, qidx_obs = convert_to_tc_matrix(
        args.course_name, student_ids, correctness_bytc_matrix, is_exam_matrix, time_matrix, device
    )

    assert len(student_ids) == y_obs.size(0)
    assert len(qidx_obs) == y_obs.size(1)
    
    upload_api = HfApi()
    print("Uploading files...")

    sorted_question_infos = []
    for qidx in qidx_obs:
        qinfo = question_infos[qidx]
        qinfo["qidx"] = qidx.item()
        sorted_question_infos.append(qinfo)

    question_info_file = io.BytesIO()
    sorted_question_infos = pd.DataFrame(sorted_question_infos)
    sorted_question_infos.to_csv(question_info_file, index=False)
    upload_api.upload_file(
        repo_id=f"stair-lab/code_insights_matrices",
        repo_type="dataset",
        path_in_repo=f"{args.course_name}/question_infos.csv",
        path_or_fileobj=question_info_file,
    )

    student_info_file = io.BytesIO()
    student_df = pd.DataFrame(student_ids)
    student_df.to_csv(student_info_file, index=False)
    upload_api.upload_file(
        repo_id=f"stair-lab/code_insights_matrices",
        repo_type="dataset",
        path_in_repo=f"{args.course_name}/student_info.csv",
        path_or_fileobj=student_info_file,
    )

    correctness_matrix_file = io.BytesIO()
    torch.save(y_obs, correctness_matrix_file)
    upload_api.upload_file(
        repo_id=f"stair-lab/code_insights_matrices",
        repo_type="dataset",
        path_in_repo=f"{args.course_name}/correctness_matrix.pt",
        path_or_fileobj=correctness_matrix_file,
    )

    is_exam_matrix_file = io.BytesIO()
    torch.save(is_exam_obs, is_exam_matrix_file)
    upload_api.upload_file(
        repo_id=f"stair-lab/code_insights_matrices",
        repo_type="dataset",
        path_in_repo=f"{args.course_name}/is_exam_matrix.pt",
        path_or_fileobj=is_exam_matrix_file,
    )

    time_obs_file = io.BytesIO()
    torch.save(time_obs, time_obs_file)
    upload_api.upload_file(
        repo_id=f"stair-lab/code_insights_matrices",
        repo_type="dataset",
        path_in_repo=f"{args.course_name}/time_matrix.pt",
        path_or_fileobj=time_obs_file,
    )

    #### ARCHIVED - REMOVE LATER ####
    # unique_questions_file = io.BytesIO()
    # pickle.dump(unique_questions, unique_questions_file)
    # upload_api.upload_file(
    #     repo_id=f"stair-lab/code_insights_matrices",
    #     repo_type="dataset",
    #     path_in_repo=f"{args.course_name}/unique_questions.pkl",
    #     path_or_fileobj=unique_questions_file,
    # )

    # question_name2idx_file = io.BytesIO()
    # pickle.dump(question_name2idx, question_name2idx_file)
    # upload_api.upload_file(
    #     repo_id=f"stair-lab/code_insights_matrices",
    #     repo_type="dataset",
    #     path_in_repo=f"{args.course_name}/question_name2idx.pkl",
    #     path_or_fileobj=question_name2idx_file,
    # )

    # response_matrix_file = io.BytesIO()
    # pickle.dump(response_matrix, response_matrix_file)
    # upload_api.upload_file(
    #     repo_id=f"stair-lab/code_insights_matrices",
    #     repo_type="dataset",
    #     path_in_repo=f"{args.course_name}/response_matrix.pkl",
    #     path_or_fileobj=response_matrix_file,
    # )
