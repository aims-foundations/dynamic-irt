import argparse
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import torch
from huggingface_hub import snapshot_download
from tueplots import bundles, constants

plt.rcParams.update(bundles.neurips2024())


def draw_hist(class_students, value, save_file, theta=0):
    plt.figure()
    for cid, (class_name, sidxs) in enumerate(class_students.items()):
        color = "#" + constants.color.palettes.paultol_bright[cid]
        line = plt.axvline(
            value[sidxs].mean(), linestyle="dashed", linewidth=1, color=color
        )
        hist = plt.hist(
            value[sidxs],
            bins=50,
            density=True,
            stacked=True,
            histtype="step",
            label=class_name,
            color=color,
        )

    plt.xlabel(r"$\theta_" + str(theta) + "$")
    plt.legend()
    plt.savefig(f"plots/{save_file}", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument(
        "--concentration",
        help="Concentration hyperparameter",
        type=float,
        default=10.0,
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}", repo_type="dataset"
    )

    student_info = pickle.load(open(f"{data_folder}/student_ids.pkl", "rb"))

    parms_file = f"results/{args.course_name}_{args.concentration}.pkl"
    if not os.path.exists(parms_file):
        raise RuntimeError(f"File {parms_file} does not exist.")

    parms_dict = pickle.load(open(parms_file, "rb"))

    # Group students by class
    class_students = {}
    for sidx, student in enumerate(student_info):
        class_id = student["class"][:-2]
        if class_id not in class_students:
            class_students[class_id] = []
        class_students[class_id].append(sidx)

    draw_hist(
        class_students,
        parms_dict["theta0"].cpu().detach().numpy(),
        "theta0_real.png",
        theta=0,
    )
    draw_hist(
        class_students,
        parms_dict["theta1"].cpu().detach().numpy(),
        "theta1_real.png",
        theta=1,
    )
