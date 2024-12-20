import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal
from tqdm import tqdm
from tueplots import bundles
from utils import ensure_dir, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument(
        "--seeds", help="Random seed", type=int, nargs="+", default=[42, 45]
    )
    parser.add_argument("--npoints", type=int, default=500)
    parser.add_argument(
        "--kernel",
        help="Prior Kernel",
        type=str,
        default="RBF",
        choices=["RBF", "Matern"],
    )
    parser.add_argument("--length_scale", help="Length scale", type=float, default=10.0)
    parser.add_argument("--nu", help="Nu", type=float, default=2.5)
    parser.add_argument("--start_iter", type=int, default=2000)
    parser.add_argument("--end_iter", type=int, default=10000)
    parser.add_argument("--step", type=int, default=1000)
    args = parser.parse_args()

    seed1, seed2 = args.seeds

    if args.kernel == "Matern":
        result_folder_seed1 = f"results/{args.course_name}_seed{seed1}_npoints{args.npoints}_kernel{args.kernel}_nu{args.nu}"
        result_folder_seed2 = f"results/{args.course_name}_seed{seed2}_npoints{args.npoints}_kernel{args.kernel}_nu{args.nu}"

    elif args.kernel == "RBF":
        result_folder_seed1 = f"results/{args.course_name}_seed{seed1}_npoints{args.npoints}_kernel{args.kernel}_lengthscale{args.length_scale}"
        result_folder_seed2 = f"results/{args.course_name}_seed{seed2}_npoints{args.npoints}_kernel{args.kernel}_lengthscale{args.length_scale}"

    plot_folder = f"plots/{args.course_name}_seed{seed1}+{seed2}_npoints{args.npoints}_kernel{args.kernel}_lengthscale{args.length_scale}"

    ensure_dir(plot_folder)

    zs_seed1 = []
    zs_seed2 = []
    for iter in tqdm(range(args.start_iter, args.end_iter + 1, args.step)):
        zs_seed1.extend(
            torch.load(os.path.join(result_folder_seed1, f"ess_zs_by_iter_{iter}.pt"))
        )
        zs_seed2.extend(
            torch.load(os.path.join(result_folder_seed2, f"ess_zs_by_iter_{iter}.pt"))
        )

    zs_seed1 = torch.stack(zs_seed1).to(device=device).float()
    zs_seed2 = torch.stack(zs_seed2).to(device=device).float()

    mean_zs_seed1 = zs_seed1.mean(dim=0)
    mean_zs_seed2 = zs_seed2.mean(dim=0)

    std_zs_seed1 = zs_seed1.std(dim=0)
    std_zs_seed2 = zs_seed2.std(dim=0)

    zs_seed1_dist = Normal(mean_zs_seed1, std_zs_seed1)
    zs_seed2_dist = Normal(mean_zs_seed2, std_zs_seed2)

    # Compute KL Divergence
    kl_div = torch.distributions.kl.kl_divergence(zs_seed1_dist, zs_seed2_dist)

    # Plot KL Divergence histogram
    plt.figure()
    plt.hist(kl_div.cpu().numpy(), bins=100, color="#4477aa")
    # Draw vertical line at the mean
    plt.axvline(kl_div.mean().item(), color="#ee6677", linestyle="dashed")
    # Put the mean value on the plot
    plt.text(kl_div.mean().item(), 105, f"Mean: {kl_div.mean().item():.2f}")
    plt.xlim(0, 20)
    plt.xlabel("$D_{KL} [p_{42}(z) || p_{45}(z)]$")
    plt.ylabel("Count")
    plt.title("KL Divergence Histogram")
    plt.savefig(os.path.join(plot_folder, "kl_divergence_histogram.png"), dpi=300)
    plt.close()

    # Draw histrgram of zs_seed1 and zs_seed2 on the same plot
    # Pick 10 random indices
    random_indices = np.random.choice(range(zs_seed1.shape[1]), 10)

    for idx in random_indices:
        plt.figure()
        plt.hist(
            zs_seed1[:, idx].cpu().numpy(),
            bins=50,
            alpha=0.5,
            color="#4477aa",
            density=True,
        )
        plt.hist(
            zs_seed2[:, idx].cpu().numpy(),
            bins=50,
            alpha=0.5,
            color="#ee6677",
            density=True,
        )

        g1 = torch.distributions.Normal(mean_zs_seed1[idx], std_zs_seed1[idx])
        g2 = torch.distributions.Normal(mean_zs_seed2[idx], std_zs_seed2[idx])

        x = torch.linspace(-3, 3, 1000).to(device)
        y1 = g1.log_prob(x).exp().cpu().numpy()
        y2 = g2.log_prob(x).exp().cpu().numpy()

        plt.plot(x.cpu(), y1, label=f"Seed {args.seeds[0]}", color="#004488")
        plt.plot(x.cpu(), y2, label=f"Seed {args.seeds[1]}", color="#bb5566")

        plt.xlabel("$z$")
        plt.xlim(-3, 3)
        plt.ylabel("Frequency")
        plt.title("Histogram of $z_{" + str(idx) + "}$")
        plt.legend()
        plt.savefig(os.path.join(plot_folder, f"z_histogram_index_{idx}.png"), dpi=300)
        plt.close()
