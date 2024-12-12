import io
import json
import os

import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download and load data
    origin_data_folder = snapshot_download(
        repo_id=f"stair-lab/code_insights_jsons", repo_type="dataset"
    )

    # Load data
    data_dict = {
        "student_id": [],
        "course_id": [],
        "section_id": [],
        "question_unittest_id": [],
        "attempt_id": [],
        "response": [],
        "pass": [],
    }

    student_infos = {
        "student_id": [],
        "student_uid": [],
    }

    course_infos = {
        "course_id": [0, 1, 2, 3],
        "course_name": ["pf_hk232", "dsa_hk231", "pf_hk222", "dsa_hk221"],
    }

    section_infos = {
        "section_id": [],
        "course_id": [],
        "section_name": [],
    }

    question_infos = {
        "question_id": [],
        "question_name": [],
        "question_text": [],
        "testcase_input": [],
        "testcase_std_input": [],
        "testcase_output": [],
    }

    for course_id, course_name in enumerate(course_infos["course_name"]):
        data_folder = os.path.join(origin_data_folder, course_name)
        directory_json_files = {}

        for folder in os.listdir(data_folder):
            if os.path.isdir(os.path.join(data_folder, folder)):
                list_jsons = os.listdir(os.path.join(data_folder, folder))
                list_jsons = [x for x in list_jsons if x.endswith(".json")]
                directory_json_files[folder] = list_jsons

        directory_json_files = dict(sorted(directory_json_files.items()))

        # Iterate over sections
        for section_name, json_files in tqdm(directory_json_files.items()):
            current_section_id = len(section_infos["section_id"])

            # Process section
            section_infos["section_id"].append(current_section_id)
            section_infos["course_id"].append(course_id)
            section_infos["section_name"].append(section_name)

            for json_file in json_files:
                print(f"Processing {course_name}/{section_name}/{json_file}")
                with open(os.path.join(data_folder, section_name, json_file), "r") as f:
                    data = json.load(f)

                data_q = data["list_questions"]
                data_s = data["student_answers"]

                question_content_map = {}
                for idx, q in enumerate(data_q):
                    if isinstance(q, list):
                        for sub_idx, sq in enumerate(q):
                            count_testcases = len(sq["testcases"])

                            if (
                                sq["name"] in question_infos["question_name"]
                                and question_infos["question_name"].count(sq["name"])
                                == count_testcases
                            ):
                                question_global_idx = question_infos[
                                    "question_name"
                                ].index(sq["name"])
                                question_content_map[
                                    f"Question {idx + 1}.{sub_idx + 1}"
                                ] = (question_global_idx, count_testcases)
                            else:
                                question_global_idx = len(
                                    question_infos["question_name"]
                                )
                                question_content_map[
                                    f"Question {idx + 1}.{sub_idx + 1}"
                                ] = (question_global_idx, count_testcases)

                                for tcid, tc in enumerate(sq["testcases"]):
                                    question_infos["question_id"].append(
                                        question_global_idx + tcid
                                    )
                                    question_infos["question_name"].append(sq["name"])
                                    question_infos["question_text"].append(
                                        sq["question"]
                                    )
                                    question_infos["testcase_input"].append(tc["input"])
                                    question_infos["testcase_std_input"].append(
                                        tc["std_input"]
                                    )
                                    question_infos["testcase_output"].append(
                                        tc["output"]
                                    )

                    else:
                        count_testcases = len(q["testcases"])

                        if (
                            q["name"] in question_infos["question_name"]
                            and question_infos["question_name"].count(q["name"])
                            == count_testcases
                        ):
                            question_global_idx = question_infos["question_name"].index(
                                q["name"]
                            )
                            question_content_map[f"Question {idx + 1}"] = (
                                question_global_idx,
                                count_testcases,
                            )
                        else:
                            question_global_idx = len(question_infos["question_name"])
                            question_content_map[f"Question {idx + 1}"] = (
                                question_global_idx,
                                count_testcases,
                            )

                            for tcid, tc in enumerate(q["testcases"]):
                                question_infos["question_id"].append(
                                    question_global_idx + tcid
                                )
                                question_infos["question_name"].append(q["name"])
                                question_infos["question_text"].append(q["question"])
                                question_infos["testcase_input"].append(tc["input"])
                                question_infos["testcase_std_input"].append(
                                    tc["std_input"]
                                )
                                question_infos["testcase_output"].append(tc["output"])

                for answer in data_s:
                    student_uid = answer["id"]
                    if student_uid == "":
                        continue
                    if student_uid in student_infos["student_uid"]:
                        student_id = student_infos["student_uid"].index(student_uid)
                    else:
                        student_id = len(student_infos["student_uid"])
                        student_infos["student_id"].append(student_id)
                        student_infos["student_uid"].append(student_uid)

                    for response_history in answer["response_history"]:
                        question_global_idx, num_testcases = question_content_map.get(
                            response_history["question"], (-1, 0)
                        )
                        if question_global_idx == -1 or num_testcases == 0:
                            continue

                        for attempt_id, result in enumerate(
                            response_history["results"][1:-1]
                        ):
                            if (
                                "testcases" not in result
                                or len(result["testcases"]) != num_testcases
                            ):
                                continue

                            response = result["action"]
                            for prefix in ["Prechecked: ", "Saved: ", "Submit: "]:
                                if result["action"].startswith(prefix):
                                    response = response.replace(prefix, "")

                            for tcid, is_pass in enumerate(result["testcases"]):
                                data_dict["student_id"].append(student_id)
                                data_dict["course_id"].append(course_id)
                                data_dict["section_id"].append(current_section_id)
                                data_dict["question_unittest_id"].append(
                                    question_global_idx + tcid
                                )
                                data_dict["attempt_id"].append(attempt_id)
                                data_dict["response"].append(response)
                                data_dict["pass"].append(is_pass)

    upload_api = HfApi()
    os.makedirs("results", exist_ok=True)
    print("Uploading files...")

    df = pd.DataFrame(data_dict)
    data_dict_file = io.BytesIO()
    df.to_csv(data_dict_file, index=False)
    upload_api.upload_file(
        repo_id=f"stair-lab/code_insights_csv",
        repo_type="dataset",
        path_in_repo="main_data.csv",
        path_or_fileobj=data_dict_file,
    )

    df = pd.DataFrame(student_infos)
    student_infos_file = io.BytesIO()
    df.to_csv(student_infos_file, index=False)
    upload_api.upload_file(
        repo_id=f"stair-lab/code_insights_csv",
        repo_type="dataset",
        path_in_repo="student_infos.csv",
        path_or_fileobj=student_infos_file,
    )

    df = pd.DataFrame(course_infos)
    course_infos_file = io.BytesIO()
    df.to_csv(course_infos_file, index=False)
    upload_api.upload_file(
        repo_id=f"stair-lab/code_insights_csv",
        repo_type="dataset",
        path_in_repo="course_infos.csv",
        path_or_fileobj=course_infos_file,
    )

    df = pd.DataFrame(section_infos)
    section_infos_file = io.BytesIO()
    df.to_csv(section_infos_file, index=False)
    upload_api.upload_file(
        repo_id=f"stair-lab/code_insights_csv",
        repo_type="dataset",
        path_in_repo="section_infos.csv",
        path_or_fileobj=section_infos_file,
    )

    df = pd.DataFrame(question_infos)
    question_infos_file = io.BytesIO()
    df.to_csv(question_infos_file, index=False)
    upload_api.upload_file(
        repo_id=f"stair-lab/code_insights_csv",
        repo_type="dataset",
        path_in_repo="question_infos.csv",
        path_or_fileobj=question_infos_file,
    )
