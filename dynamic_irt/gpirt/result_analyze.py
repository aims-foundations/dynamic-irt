import argparse
import os
import pickle
import random

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
    parser.add_argument("--npoints", type=int, default=500)
    parser.add_argument(
        "--kernel",
        help="Prior Kernel",
        type=str,
        default="RBF",
        choices=["RBF", "Matern"],
    )
    parser.add_argument("--length_scale", help="Length scale", type=float, default=10.0)
    parser.add_argument("--start_iter", type=int, default=2000)
    parser.add_argument("--end_iter", type=int, default=10000)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--sidx", help="Student index", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    result_folder = f"results/{args.course_name}_seed{args.seed}_npoints{args.npoints}_kernel{args.kernel}_lengthscale{args.length_scale}"
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
        uni_time = time_ob.unique()
        aidx_ob = torch.searchsorted(uni_time, time_ob)
        # Replace the last element with -1
        aidx_ob[aidx_ob == len(uni_time) - 1] = -1
        aidx_obs.append(aidx_ob)
        unique_time_obs.append(uni_time[:-1])

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
    picked_sidx = args.sidx
    student_x = unique_time_obs[picked_sidx][list_saidx[picked_sidx]].cpu().numpy()
    if picked_sidx < num_train:
        student_y = y_train[picked_sidx].float().cpu().numpy()
    else:
        student_y = y_test[picked_sidx - num_train].float().cpu().numpy()

    # Save the student's x and y
    with open(
        f"{result_folder}/{args.course_name}_seed{args.seed}/student_{picked_sidx}_xy.pkl",
        "wb",
    ) as f:
        pickle.dump((student_x, student_y), f)

    # Draw the student's performance
    plt.figure()
    plt.scatter(student_x, student_y)
    plt.ylabel("Correctness")
    plt.title(f"Student {picked_sidx} performance")
    plt.savefig(f"plots/student_{picked_sidx}_performance_seed{args.seed}.png", dpi=300)
    plt.close()

    # Load thetas
    print("Loading thetas")
    thetas = []
    points = []

    for iter in tqdm(range(args.start_iter, args.end_iter + 1, args.step)):
        thetas.extend(
            torch.load(os.path.join(result_folder, f"ess_thetas_by_iter_{iter}.pt"))
        )
        points.extend(
            torch.load(os.path.join(result_folder, f"ess_points_by_iter_{iter}.pt"))
        )

    time_obs = torch.load("data/time_obs.pt")
    unique_time_obs = []
    for tidx, time_ob in enumerate(time_obs):
        unique_time_obs.append(time_ob.unique()[:-1].cpu().numpy())
    student_idxs = pickle.load(
        open(os.path.join(result_folder, "student_idxs.pkl"), "rb")
    )
    list_saidx2idx = pickle.load(
        open(os.path.join(result_folder, "list_saidx2aidx.pkl"), "rb")
    )
    list_available_sidx = pickle.load(
        open(os.path.join(result_folder, "list_available_sidx.pkl"), "rb")
    )

    student_thetas = [th[student_idxs == picked_sidx].float().tolist() for th in thetas]
    student_available_idx = list_available_sidx.index(picked_sidx)
    student_points = [
        pt[100 * student_available_idx : 100 * (student_available_idx + 1)]
        for pt in points
    ]

    # Save thetas
    with open(
        f"{result_folder}/{args.course_name}_seed{args.seed}/student_{picked_sidx}_thetas.pkl",
        "wb",
    ) as f:
        pickle.dump(student_thetas, f)

    # Save points
    with open(
        f"{result_folder}/{args.course_name}_seed{args.seed}/student_{picked_sidx}_points.pkl",
        "wb",
    ) as f:
        pickle.dump(student_points, f)

    # Compute the Spearman correlation coefficient
    print("Computing Spearman correlation coefficient")
    spearman = SpearmanCorrCoef()
    list_spearman = []
    for sampled_thetas in tqdm(
        random.choices(student_thetas, k=1000), desc="Computing SCC"
    ):
        list_spearman.append(
            spearman(
                torch.tensor(sampled_thetas, device=device)[
                    list_saidx2idx[picked_sidx]
                ],
                torch.tensor(student_y, device=device)[list_saidx2idx[picked_sidx]],
            )
            .cpu()
            .item()
        )

    # Plot the Spearman correlation coefficient histogram
    plt.figure()
    plt.hist(list_spearman, bins=20)

    # Draw mean axvline
    plt.axvline(np.mean(list_spearman), color="red", linestyle="dashed")
    plt.xlabel("Spearman correlation coefficient")
    plt.ylabel("Frequency")
    plt.title(f"Student {picked_sidx} SCC")
    plt.savefig(
        f"plots/student_{picked_sidx}_spearman_hist_seed{args.seed}_npoints{args.npoints}_kernel{args.kernel}_lengthscale{args.length_scale}.png",
        dpi=300,
    )

    # # Load zs
    # print("Loading zs")
    # zs = torch.load(os.path.join(first_folder, f"ess_zs_by_iter_{args.iteration}.pt"))

    # student_zs = [z.to(device)[list_sqidx[picked_sidx]].cpu().tolist() for z in zs]
    # student_zs = np.array(student_zs)

    # # Save zs
    # with open(
    #     f"{result_folder}/student_{picked_sidx}_zs.pkl",
    #     "wb",
    # ) as f:
    #     pickle.dump(student_zs, f)

    # student_zs_mean = student_zs.mean(axis=0)
    # student_zs_std = student_zs.std(axis=0)

    # # Scatter plot of student's z with error bars
    # plt.figure()
    # plt.errorbar(
    #     range(len(student_zs_mean)),
    #     student_zs_mean,
    #     yerr=student_zs_std,
    #     fmt="o",
    #     capsize=5,
    # )
    # # plt.xlabel("Testcase index")
    # plt.ylabel(r"$z$")
    # plt.title(f"Student {picked_sidx} z plot")
    # plt.savefig(f"plots/student_{picked_sidx}_z_plot_seed{args.seed}.png", dpi=300)
    # plt.close()
