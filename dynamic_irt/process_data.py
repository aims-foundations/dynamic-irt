import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from dynamic_irt.gpirt.utils import ensure_dir, parse_time, set_seed


if __name__ == "__main__":
    # wandb.init(project="code_insights")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}_wtc", repo_type="dataset"
    )

    y_obs = pickle.load(open(f"{data_folder}/correctness_bytc_matrix.pkl", "rb"))
    time_matrix = pickle.load(open(f"{data_folder}/time_matrix.pkl", "rb"))
    # u_obs has shape of n_student x n_question x n_attempt x n_testcase
    # The number of testcases can be different for each question

    # Number of students
    n_student = len(y_obs)

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
    time_tc_obs = []
    qidx_tc_obs = []
    tidx_tc_obs = []
    for sidx, student in enumerate(tqdm(y_obs, desc="Preprocessing")):
        student_tc = []
        student_time = []
        student_qidx = []
        student_tidx = []
        global_tidx = 0
        for qidx in range(n_question):
            for tidx in range(list_max_testcases[qidx]):
                student_tc.append([])
                student_time.append([])
                student_qidx.append([])
                student_tidx.append([])

                for aidx, attempt in enumerate(student[qidx]):

                    if attempt == -1:
                        student_tc[-1].append(-1)
                        student_time[-1].append(parse_time("01/01/30, 00:00:00"))
                        student_qidx[-1].append(-1)
                        student_tidx[-1].append(-1)
                    else:
                        if len(attempt) == 0:
                            student_tc[-1].append(-1)
                            student_time[-1].append(parse_time("01/01/30, 00:00:00"))
                            student_qidx[-1].append(-1)
                            student_tidx[-1].append(-1)
                        elif (
                            tidx < len(attempt) and time_matrix[sidx][qidx][aidx] != ""
                        ):
                            student_tc[-1].append(attempt[tidx])
                            student_time[-1].append(
                                parse_time(time_matrix[sidx][qidx][aidx])
                            )
                            student_qidx[-1].append(qidx)
                            student_tidx[-1].append(global_tidx)
                        else:
                            student_tc[-1].append(-1)
                            student_time[-1].append(parse_time("01/01/30, 00:00:00"))
                            student_qidx[-1].append(-1)
                            student_tidx[-1].append(-1)
                            # raise ValueError("Testcase index out of bound")

                global_tidx += 1

        y_tc_obs.append(student_tc)
        time_tc_obs.append(student_time)
        qidx_tc_obs.append(student_qidx)
        tidx_tc_obs.append(student_tidx)

    y_obs = torch.tensor(y_tc_obs, device=device)
    qidx_obs = torch.tensor(qidx_tc_obs, device=device)
    time_obs = torch.tensor(time_tc_obs, device=device)
    tidx_obs = torch.tensor(tidx_tc_obs, device=device)

    torch.save(y_obs, "data/y_obs.pt")
    torch.save(qidx_obs, "data/qidx_obs.pt")
    torch.save(time_obs, "data/time_obs.pt")
    torch.save(tidx_obs, "data/tidx_obs.pt")
