import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from dynamic_irt.gpirt.utils import ensure_dir, set_seed


def item_response_fn_1PL(theta, z):
    return torch.sigmoid(theta - z)


if __name__ == "__main__":
    # wandb.init(project="code_insights")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=1000)
    parser.add_argument(
        "--esp", help="Epsilon for avoiding zero score", type=float, default=1e-3
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)
    ensure_dir(f"results/{args.course_name}")
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

    y_obs = torch.load("y_obs.pt")
    qidx_obs = torch.load("qidx_obs.pt")
    time_obs = torch.load("time_obs.pt")

    y_obs = torch.flatten(y_obs, start_dim=1)
    qidx_obs = torch.flatten(qidx_obs, start_dim=1)
    time_obs = torch.flatten(time_obs, start_dim=1)

    first_idx = torch.arange(start=0, end=n_student).reshape(-1, 1)
    sorted_idx = torch.argsort(time_obs, dim=1)
    y_obs = y_obs[first_idx, sorted_idx]
    qidx_obs = qidx_obs[first_idx, sorted_idx]
    time_obs = time_obs[first_idx, sorted_idx]

    masked_idx = y_obs != -1

    unique_time_obs = []
    aidx_obs = []  # student attempt index
    for tidx, time_ob in enumerate(time_obs):
        unique_time_obs.append(time_ob.unique())
        aidx_ob = torch.searchsorted(unique_time_obs[-1], time_ob)
        aidx_ob[aidx_ob == len(unique_time_obs[-1]) - 1] = (
            -1
        )  # Replace the last element with -1
        aidx_obs.append(aidx_ob)

    thetas = pickle.load(open(f"results/thetas_by_iter_{args.epochs}.pkl", "rb"))
    zs = pickle.load(open(f"results/zs_by_iter_{args.epochs}.pkl", "rb"))

    num_train = int(0.8 * n_student)
    num_test = n_student - num_train

    last_thetas = thetas[-1].copy()
    last_zs = zs[-1].clone()
    del thetas
    del zs

    list_saidx = []
    list_sqidx = []
    for sidx in range(n_student):
        if masked_idx[sidx].sum() == 0:
            list_saidx.append(None)
            list_sqidx.append(None)
            continue

        saidx = aidx_obs[sidx][masked_idx[sidx]]  # attemp index for student
        list_saidx.append(saidx)

        sqidx = qidx_obs[sidx][masked_idx[sidx]]  # global testcase index
        list_sqidx.append(sqidx)

    list_y_true = []
    list_y_pred = []
    for sidx, st_theta in enumerate(tqdm(last_thetas)):
        if masked_idx[sidx].sum() == 0:
            continue

        y_pred = item_response_fn_1PL(
            st_theta[list_saidx[sidx]], last_zs[list_sqidx[sidx]]
        )
        y_pred = (y_pred >= 0.5).float()
        y_true = y_obs[sidx][masked_idx[sidx]].float()

        list_y_pred.append(y_pred)
        list_y_true.append(y_true)

    train_y_true = torch.cat(list_y_true[:num_train]).cpu().numpy()
    test_y_true = torch.cat(list_y_true[num_train:]).cpu().numpy()
    train_y_pred = torch.cat(list_y_pred[:num_train]).cpu().numpy()
    test_y_pred = torch.cat(list_y_pred[num_train:]).cpu().numpy()

    train_acc = accuracy_score(train_y_true, train_y_pred)
    train_f1 = f1_score(train_y_true, train_y_pred)
    train_precision = precision_score(train_y_true, train_y_pred)
    train_recall = recall_score(train_y_true, train_y_pred)
    train_roc_auc = roc_auc_score(train_y_true, train_y_pred)

    train_eval_res = {
        "accuracy": train_acc,
        "f1": train_f1,
        "precision": train_precision,
        "recall": train_recall,
        "roc_auc": train_roc_auc,
    }
    print(train_eval_res)

    test_acc = accuracy_score(test_y_true, test_y_pred)
    test_f1 = f1_score(test_y_true, test_y_pred)
    test_precision = precision_score(test_y_true, test_y_pred)
    test_recall = recall_score(test_y_true, test_y_pred)
    test_roc_auc = roc_auc_score(test_y_true, test_y_pred)

    test_eval_res = {
        "accuracy": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "roc_auc": test_roc_auc,
    }
    print(test_eval_res)

    # Save evaluation results
    ensure_dir(f"eval_results/all_cls")
    with open(f"eval_results/all_cls/dirt_train_res.json", "w") as f:
        json.dump(train_eval_res, f)
    with open(f"eval_results/all_cls/dirt_test_res.json", "w") as f:
        json.dump(test_eval_res, f)
