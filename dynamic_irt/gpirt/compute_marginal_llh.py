import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from es_sampler import IRTLikelihood
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
    args = parser.parse_args()

    set_seed(args.seed)
    result_folder = f"results/{args.course_name}_seed{args.seed}_npoints{args.npoints}_kernel{args.kernel}_lengthscale{args.length_scale}"

    y_obs = torch.load("data/y_obs.pt")
    time_obs = torch.load("data/time_obs.pt")

    y_obs = torch.flatten(y_obs, start_dim=1)
    time_obs = torch.flatten(time_obs, start_dim=1)

    n_student = len(y_obs)
    first_idx = torch.arange(start=0, end=n_student).reshape(-1, 1)
    sorted_idx = torch.argsort(time_obs, dim=1)
    y_obs = y_obs[first_idx, sorted_idx]
    masked_idx = y_obs != -1

    print("Splitting train and test data")
    list_saidx = pickle.load(open(f"{result_folder}/list_saidx.pkl", "rb"))
    num_train = int(0.8 * n_student)
    y_test = []
    for sidx in range(num_train, n_student):
        if masked_idx[sidx].sum() == 0:
            continue

        y_test.append(y_obs[sidx][masked_idx[sidx]])
    y_test = torch.concatenate(y_test).float()
    total_test = y_test.shape[0]

    all_squidx = pickle.load(open(f"{result_folder}/all_squidx.pkl", "rb")).cpu()

    thetas = []
    zs = []
    for iter in tqdm(range(args.start_iter, args.end_iter + 1, args.step)):
        theta_iter = torch.load(
            os.path.join(result_folder, f"ess_thetas_by_iter_{iter}.pt")
        )
        thetas.extend([thetai[-total_test:].to(device) for thetai in theta_iter])

        z_iter = torch.load(os.path.join(result_folder, f"ess_zs_by_iter_{iter}.pt"))
        zs.extend([zi[all_squidx][-total_test:].to(device) for zi in z_iter])

    # Randomly choose 1000 sampled thetas and zs
    # chosen_idx = random.sample(range(len(thetas)), 1000)
    # thetas = [thetas[i] for i in chosen_idx]
    # zs = [zs[i] for i in chosen_idx]

    llh_fn = IRTLikelihood(device=device)
    list_llh = []
    for i, (theta, z) in tqdm(
        enumerate(zip(thetas, zs)), desc="Computing LLH", total=len(thetas)
    ):
        llh = llh_fn.log_likelihood(theta, z, y_test)
        list_llh.append(llh.item())

    # Draw LLH histogram
    plt.figure()
    plt.hist(list_llh, bins=20)

    # Mean vertical line
    mean_llh = np.mean(list_llh)
    plt.axvline(mean_llh, color="red", linestyle="--")

    plt.xlabel("Log Likelihood")
    plt.ylabel("Frequency")
    plt.title("LLH Histogram (Mean LLH: {:.2f})".format(mean_llh))
    plt.savefig(f"{result_folder}/llh_histogram.png", dpi=300)
    plt.close()
