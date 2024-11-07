import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torchmetrics.regression import SpearmanCorrCoef
from tqdm import tqdm
from tueplots import bundles, constants
from utils import ensure_dir, set_seed

plt.rcParams.update(bundles.iclr2024())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # wandb.init(project="code_insights")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--iteration", help="# Iteration", type=int, default=100000)
    parser.add_argument("--sidx", help="Student index", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(f"results/{args.course_name}_seed{args.seed}")
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}_wtc", repo_type="dataset"
    )

    y_obs = pickle.load(open(f"{data_folder}/correctness_bytc_matrix.pkl", "rb"))
    time_matrix = pickle.load(open(f"{data_folder}/time_matrix.pkl", "rb"))
    # u_obs has shape of n_student x n_question x n_attempt x n_testcase
    # The number of testcases can be different for each question

    # Number of questions
    n_question = len(y_obs[0])

    # Compute maximum number of testcases for each question
    print("Computing maximum number of testcases for each question")
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

    total_testcases = sum(list_max_testcases.values())

    print("Loading data")
    y_obs = torch.load("data/y_obs.pt")
    qidx_obs = torch.load("data/tidx_obs.pt")
    time_obs = torch.load("data/time_obs.pt")

    y_obs = torch.flatten(y_obs, start_dim=1)
    qidx_obs = torch.flatten(qidx_obs, start_dim=1)
    time_obs = torch.flatten(time_obs, start_dim=1)

    n_student = len(y_obs)
    first_idx = torch.arange(start=0, end=n_student).reshape(-1, 1)
    sorted_idx = torch.argsort(time_obs, dim=1)
    y_obs = y_obs[first_idx, sorted_idx]
    qidx_obs = qidx_obs[first_idx, sorted_idx]
    time_obs = time_obs[first_idx, sorted_idx]

    masked_idx = y_obs != -1

    # Process the attempt index for each student
    unique_time_obs = []
    aidx_obs = []  # student attempt index
    for tidx, time_ob in enumerate(time_obs):
        unique_time_obs.append(time_ob.unique())
        aidx_ob = torch.searchsorted(unique_time_obs[-1], time_ob)
        aidx_ob[aidx_ob == len(unique_time_obs[-1]) - 1] = (
            -1
        )  # Replace the last element with -1
        aidx_obs.append(aidx_ob)

    # Create y_train
    print("Splitting train and test data")
    num_train = int(0.8 * n_student)
    y_train = []
    y_test = []
    for sidx in range(n_student):
        if masked_idx[sidx].sum() == 0:
            continue

        if sidx < num_train:
            y_train.append(y_obs[sidx][masked_idx[sidx]])
        else:
            y_test.append(y_obs[sidx][masked_idx[sidx]])
    # y_train = torch.concatenate(y_train).float()
    # y_test = torch.concatenate(y_test).float()

    list_saidx = []
    list_sqidx = []
    train_test_split_idx = 0
    student_idxs = []
    for sidx in range(n_student):
        if masked_idx[sidx].sum() == 0:
            list_saidx.append(None)
            # list_sqidx.append(None)
            continue

        saidx = aidx_obs[sidx][masked_idx[sidx]]  # attemp index for student
        list_saidx.append(saidx)

        sqidx = qidx_obs[sidx][masked_idx[sidx]]  # global testcase index
        list_sqidx.append(sqidx)

        student_idxs.extend([sidx] * saidx.shape[0])

        if sidx < num_train:
            train_test_split_idx += saidx.shape[0]

    # Pick a student
    print("Picking a student")
    picked_sidx = args.sidx + num_train
    student_y = y_test[args.sidx].float().cpu().numpy()

    # Draw the student's performance
    plt.figure()
    plt.plot(student_y)
    # plt.xlabel("Attempt index")
    plt.ylabel("Correctness")
    plt.title(f"Student {picked_sidx} performance")
    plt.savefig(f"plots/student_{picked_sidx}_performance_seed{args.seed}.png", dpi=300)

    # Load thetas
    print("Loading thetas")
    first_folder = f"results/{args.course_name}_seed{args.seed}"
    thetas = torch.load(
        os.path.join(first_folder, f"ess_thetas_by_iter_{args.iteration}.pt")
    )
    time_obs = torch.load("data/time_obs.pt")
    unique_time_obs = []
    for tidx, time_ob in enumerate(time_obs):
        unique_time_obs.append(time_ob.unique()[:-1].cpu().numpy())
    student_idxs = pickle.load(
        open(os.path.join(first_folder, "student_idxs.pkl"), "rb")
    )
    list_saidx2idx = pickle.load(
        open(os.path.join(first_folder, "list_saidx2aidx.pkl"), "rb")
    )

    # student_thetas = [th[student_idxs == picked_sidx][list_saidx2idx[picked_sidx]].float().tolist() for th in thetas]
    student_thetas = [th[student_idxs == picked_sidx].float().tolist() for th in thetas]

    # Save thetas
    with open(
        f"results/{args.course_name}_seed{args.seed}/student_{picked_sidx}_thetas.pkl",
        "wb",
    ) as f:
        pickle.dump(student_thetas, f)
    # exit(0)
    student_thetas = np.array(student_thetas)
    student_thetas_mean = student_thetas.mean(axis=0)
    # student_thetas_std = student_thetas.std(axis=0)
    # # Draw the student's theta plot
    # plt.figure()
    # plt.plot(unique_time_obs[picked_sidx], student_thetas_mean)
    # plt.fill_between(
    #     unique_time_obs[picked_sidx],
    #     student_thetas_mean - student_thetas_std,
    #     student_thetas_mean + student_thetas_std,
    #     alpha=0.3,
    # )
    # plt.ylabel(r"$\theta$")
    # plt.title(f"Student {picked_sidx} theta plot")
    # plt.savefig(f"plots/student_{picked_sidx}_theta_plot_seed{args.seed}.png", dpi=300)
    # plt.close()

    # Compute the Spearman correlation coefficient
    print("Computing Spearman correlation coefficient")
    spearman = SpearmanCorrCoef()
    spearman_value = spearman(
        torch.tensor(student_thetas_mean, device=device),
        torch.tensor(student_y, device=device)[list_saidx2idx[picked_sidx]],
    )
    print(
        f"Student {picked_sidx} Spearman correlation coefficient: {spearman_value:.2f}"
    )
    fig, ax = spearman.plot()
    plt.savefig(
        f"plots/student_{picked_sidx}_spearman_plot_seed{args.seed}.png", dpi=300
    )

    # Load zs
    print("Loading zs")
    zs = torch.load(os.path.join(first_folder, f"ess_zs_by_iter_{args.iteration}.pt"))

    student_zs = [z.to(device)[list_sqidx[picked_sidx]].cpu().tolist() for z in zs]
    student_zs = np.array(student_zs)

    student_zs_mean = student_zs.mean(axis=0)
    student_zs_std = student_zs.std(axis=0)

    # Scatter plot of student's z with error bars
    plt.figure()
    plt.errorbar(
        range(len(student_zs_mean)),
        student_zs_mean,
        yerr=student_zs_std,
        fmt="o",
        capsize=5,
    )
    # plt.xlabel("Testcase index")
    plt.ylabel(r"$z$")
    plt.title(f"Student {picked_sidx} z plot")
    plt.savefig(f"plots/student_{picked_sidx}_z_plot_seed{args.seed}.png", dpi=300)
    plt.close()
