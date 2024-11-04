import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from tueplots import bundles, constants

plt.rcParams.update(bundles.iclr2024())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 45])
    parser.add_argument("--n_theta", type=int, default=10)
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args()

    if len(args.seeds) != 2:
        raise ValueError("Please provide two seeds")

    first_folder = f"results/{args.course_name}_seed{args.seeds[0]}"
    second_folder = f"results/{args.course_name}_seed{args.seeds[1]}"

    theta_1 = torch.load(
        os.path.join(first_folder, f"ess_thetas_by_iter_{args.iteration}.pt")
    )
    theta_2 = torch.load(
        os.path.join(second_folder, f"ess_thetas_by_iter_{args.iteration}.pt")
    )

    total_students = len(theta_1[0])
    # Randomly select n_theta thetas from total_thetas without replacement
    selected_students = random.sample(range(total_students), args.n_theta)

    for sidx in tqdm(selected_students, desc="Drawing theta distributions"):
        thidx = random.randint(0, len(theta_1[0][0]))
        list_theta_1 = [theta_1[i][sidx][thidx] for i in range(2000, len(theta_1))]
        list_theta_2 = [theta_2[i][sidx][thidx] for i in range(2000, len(theta_2))]

        # Draw theta_1 and theta_2 distributions on the same plot with different colors
        plt.hist(list_theta_1, bins=50, alpha=0.5, color="blue")
        plt.hist(list_theta_2, bins=50, alpha=0.5, color="red")
        plt.xlabel(r"$\theta$")
        plt.ylabel("Frequency")
        plt.title(f"Student {sidx} - " + r"$\theta$" + f" {thidx} Distribution")
        plt.savefig(
            f"plots/student{sidx}_theta{thidx}_iter{args.iteration}.png", dpi=300
        )
        plt.close()
