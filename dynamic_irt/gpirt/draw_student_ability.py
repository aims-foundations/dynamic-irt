import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from tueplots import bundles
from utils import ensure_dir, moving_average, set_seed

plt.rcParams.update(bundles.iclr2024())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--sidx", help="Student index", type=int, default=0)
    parser.add_argument(
        "--kernel",
        help="Prior Kernel",
        type=str,
        default="RBF",
        choices=["RBF", "Matern"],
    )
    parser.add_argument("--length_scale", help="Length scale", type=float, default=1.0)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--fitting_method", type=str, default="hmc")
    parser.add_argument("--thinning", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_folder = f"results/{args.course_name}_s{args.seed}_D{args.D}_PL{args.PL}_{args.fitting_method}_kernel{args.kernel}_ls{args.length_scale}"
    plot_folder = f"plots/{args.course_name}_s{args.seed}_D{args.D}_PL{args.PL}_{args.fitting_method}_kernel{args.kernel}_ls{args.length_scale}"
    ensure_dir(plot_folder)

    #################################################################################

    # Download and load data
    data_folder = snapshot_download(
        repo_id=f"stair-lab/code_insights_matrices", repo_type="dataset"
    )
    data_folder = os.path.join(data_folder, args.course_name)

    # Load matrices
    response_matrix = torch.load(f"{data_folder}/correctness_matrix.pt").to(
        device, dtype=torch.float32
    )
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts

    response_time_matrix = torch.load(f"{data_folder}/time_matrix.pt").to(
        device, dtype=torch.float32
    )
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts

    abilities = torch.load(f"{result_folder}/ability.pt")
    difficulties = torch.load(f"{result_folder}/difficulty.pt").to(device)

    if args.thinning > 1:
        abilities = [ab[:: args.thinning] for ab in abilities]
        difficulties = difficulties[:: args.thinning]

    # Get some necessary indexes
    n_testtakers, n_questions, n_max_attempts = response_matrix.shape
    observation_mask = response_matrix != -1
    response_time_indexes = []
    question_expanding_indexes = []
    for sidx in tqdm(range(n_testtakers), desc="Constructing indexes"):
        uni_time = response_time_matrix[sidx].unique()
        has_missing = 1 if -1 in uni_time else 0

        # Get the time index for each attempt
        time_index = (
            torch.searchsorted(uni_time, response_time_matrix[sidx]) - has_missing
        )
        ### REMEMBER: time_index is 0-indexed. Element 0 is -1
        ### We need to subtract 1 to get the correct index
        response_time_indexes.append(time_index[time_index != -1])
        question_expanding_indexes.append(
            torch.arange(n_questions, device=device)[:, None].expand(
                -1, n_max_attempts
            )[observation_mask[sidx]]
        )

    student_times = (
        response_time_matrix[args.sidx, observation_mask[args.sidx]].cpu().numpy()
    )
    time_sort_idx = np.argsort(student_times)
    plt.figure()
    plt.scatter(
        student_times,
        response_matrix[args.sidx, observation_mask[args.sidx]].cpu().numpy() * 6 - 3,
        s=5,
    )

    ability = abilities[args.sidx]
    for saidx in range(ability.shape[0]):
        plt.scatter(
            student_times,
            ability[saidx, response_time_indexes[args.sidx]].cpu().numpy(),
            color="gray",
            s=5,
        )
    plt.plot(
        student_times[time_sort_idx],
        ability.mean(0)[response_time_indexes[args.sidx]][time_sort_idx].cpu().numpy(),
        color="green",
    )

    # Plot week lines
    min_time = student_times.min().item()
    max_time = student_times.max().item()
    week_idx = [
        min(min_time + 14 * i, max_time)
        for i in range(int((max_time - min_time) // 14 + 1))
    ]
    for week in week_idx:
        plt.axvline(week, color="black", linestyle="--", alpha=0.5)

    plt.xlabel("Time (Days)")
    plt.ylabel("Correctness/Ability")
    plt.title(f"Student {args.sidx}'s Correctness and Ability")
    # plt.xlim(min_time-0.01, min_time+0.1)
    plt.savefig(
        f"{plot_folder}/student{args.sidx}_correctness_ability.png",
        dpi=300,
    )
