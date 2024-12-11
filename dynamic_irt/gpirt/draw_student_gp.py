import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tueplots import bundles
from utils import ensure_dir, moving_average, set_seed

plt.rcParams.update(bundles.iclr2024())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--iteration", help="# Iteration", type=int, default=100000)
    parser.add_argument("--sidx", help="Student index", type=int, default=0)
    parser.add_argument(
        "--kernel",
        help="Prior Kernel",
        type=str,
        default="RBF",
        choices=["RBF", "Matern"],
    )
    parser.add_argument("--npoints", type=int, default=500)
    parser.add_argument("--length_scale", help="Length scale", type=float, default=50.0)

    args = parser.parse_args()

    set_seed(args.seed)
    result_folder = f"results/{args.course_name}_seed{args.seed}_npoints{args.npoints}_kernel{args.kernel}_lengthscale{args.length_scale}"
    ensure_dir("plots/")

    #################################################################################

    y_obs = torch.load("data/y_obs.pt")
    time_obs = torch.load("data/time_obs.pt")
    qidx_obs = torch.load("data/qidx_obs.pt")
    tidx_obs = torch.load("data/tidx_obs.pt")

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

    unique_time_obs = []
    for tidx, time_ob in enumerate(time_obs):
        unique_time_obs.append(time_ob.unique()[:-1].cpu().numpy())
    picked_sidx = args.sidx

    list_saidx = pickle.load(open(f"{result_folder}/{args.course_name}_seed{args.seed}/list_saidx.pkl", "rb"))
    student_thetas = pickle.load(
        open(
            f"{result_folder}/{args.course_name}_seed{args.seed}/student_{picked_sidx}_thetas.pkl",
            "rb",
        )
    )
    student_points = pickle.load(
        open(
            f"{result_folder}/{args.course_name}_seed{args.seed}/student_{picked_sidx}_points.pkl",
            "rb",
        )
    )
    student_xs, student_ys = pickle.load(
        open(
            f"{result_folder}/{args.course_name}_seed{args.seed}/student_{picked_sidx}_xy.pkl",
            "rb",
        )
    )
    # >>> n_sample x n_time

    student_times = torch.tensor(unique_time_obs[picked_sidx]).to(device)
    student_times = student_times[list_saidx[picked_sidx]]
    student_qidx = qidx_obs[picked_sidx][masked_idx[picked_sidx]]
    student_thetas = torch.tensor(student_thetas).to(device)
    student_points = torch.stack(student_points).float()
    # >>> n_time

    sort_idx = student_qidx.argsort()
    sorted_student_times = student_times[sort_idx]
    sorted_ys = torch.tensor(student_ys, device=device)[sort_idx]
    sorted_student_thetas = [th[sort_idx] for th in student_thetas]
    student_qidx = student_qidx[sort_idx].cpu()

    print("Done Loading!")

    sorted_ys_by_qidx = []
    for qidx in student_qidx.unique():
        qidx_mask = student_qidx == qidx
        q_time = sorted_student_times[qidx_mask]
        q_y = sorted_ys[qidx_mask]

        for quni_time in q_time.unique():
            q_time_mask = q_time == quni_time
            q_y_time = q_y[q_time_mask].mean().item()
            sorted_ys_by_qidx.append(q_y_time)

    # Plot student sorted y
    plt.scatter(range(len(sorted_ys_by_qidx)), sorted_ys_by_qidx)
    plt.xlabel("Questions")
    plt.ylabel("Student y")
    plt.title(f"Student {picked_sidx}'s y")
    plt.savefig(
        f"plots/test_s{picked_sidx}_y_seed{args.seed}_npoints{args.npoints}_kernel{args.kernel}_lengthscale{args.length_scale}.png",
        dpi=300,
    )
    plt.close()

    min_time = student_times.min().item()
    max_time = student_times.max().item()
    time_range = torch.linspace(min_time, max_time, student_points.shape[-1])
    for sample_points in student_points:
        plt.plot(
            time_range.cpu().numpy(),
            sample_points.cpu().numpy(),
            color="gray",
            alpha=0.01,
        )

    mean_points = student_points.mean(dim=0)
    std_points = student_points.std(dim=0)
    plt.plot(time_range.cpu().numpy(), mean_points.cpu().numpy(), color="red")

    # Scatter student x, y
    student_ys = student_ys * 6 - 3
    plt.scatter(student_xs, student_ys, color="blue")

    ma_ys = []
    for uni_x in np.unique(student_xs):
        end_x = uni_x + 30
        ma_y = student_ys[(student_xs >= uni_x) & (student_xs < end_x)]
        ma_ys.append(ma_y.mean())
    plt.plot(np.unique(student_xs), ma_ys, color="blue")

    # Plot week lines
    week_idx = [min(min_time + 14 * i, max_time) for i in range(7)]
    for week in week_idx:
        plt.axvline(week, color="black", linestyle="--", alpha=0.5)

    plt.xlabel("Days")
    plt.ylabel(r"$\theta$")
    plt.title(f"Student {picked_sidx}'s $\\theta$")
    plt.savefig(
        f"plots/test_s{picked_sidx}_seed{args.seed}_npoints{args.npoints}_kernel{args.kernel}_lengthscale{args.length_scale}.png",
        dpi=300,
    )
    plt.close()