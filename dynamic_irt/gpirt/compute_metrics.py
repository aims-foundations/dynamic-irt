import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from gpirt.utils import ensure_dir, set_seed
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from tueplots import bundles

plt.rcParams.update(bundles.iclr2024())


def item_response_fn_1PL(theta, z):
    return torch.sigmoid(theta + z)


if __name__ == "__main__":
    # wandb.init(project="code_insights")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--step", help="Random seed", type=int, default=50)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)
    ensure_dir(f"results/{args.course_name}")
    data_folder = snapshot_download(
        repo_id="stair-lab/code_insights_matrices", repo_type="dataset"
    )

    response_matrix = torch.load("data/correctness_matrix.pt").to(device)
    response_time_matrix = torch.load("data/time_matrix.pt").to(device)
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts

    ##### TEMPORARY FIX -- REMOVE LATER #####
    accept_idxs = []
    for idx, row in enumerate(response_time_matrix):
        if row.unique().shape[0] > 1:
            accept_idxs.append(idx)

    response_matrix = response_matrix[accept_idxs]
    response_time_matrix = response_time_matrix[accept_idxs]
    response_time_matrix[response_time_matrix == 2314] = -1
    #########################################

    n_students, n_questions, n_max_attempts = response_matrix.shape

    # Compute the mask for the observed responses
    observation_mask = (response_matrix != -1).cpu()
    y_true = response_matrix[observation_mask].cpu().numpy()

    # Load the ability
    ability = torch.load("data/ability.pt")
    ability = torch.concatenate(ability, dim=-1)[2000 :: args.step]

    # Load the difficulty
    difficulty = torch.load("data/difficulty.pt")[2000 :: args.step]

    # Compute the predicted responses
    metrics = {
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "auc": [],
    }
    for ab, di in tqdm(
        zip(ability, difficulty), desc="Computing metrics", total=len(ability)
    ):
        ab = ab.to(device)
        di = di.to(device)

        di = di[None, :, None].expand(n_students, -1, n_max_attempts)
        di = di[observation_mask]

        prob_pred = item_response_fn_1PL(ab, di).cpu().numpy()
        y_pred = (prob_pred > 0.5).astype(int)

        metrics["accuracy"].append(accuracy_score(y_true, y_pred))
        metrics["f1"].append(f1_score(y_true, y_pred))
        metrics["precision"].append(precision_score(y_true, y_pred))
        metrics["recall"].append(recall_score(y_true, y_pred))
        metrics["auc"].append(roc_auc_score(y_true, prob_pred))

    # Save the metrics
    mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}

    with open(f"results/{args.course_name}/metrics.txt", "w") as f:
        for k, v in mean_metrics.items():
            f.write(f"{k}: {v} +/- {std_metrics[k]}\n")

    # Plot the metrics
    plt.figure()
    plt.hist(metrics["accuracy"], bins=20)
    plt.axvline(mean_metrics["accuracy"], color="red", linestyle="dashed")
    plt.xlim(0, 1)
    plt.title("Accuracy")
    plt.ylabel("Count")
    plt.title(f"Accuracy - Mean: {mean_metrics['accuracy']:.3f}")
    plt.savefig(f"results/{args.course_name}/accuracy.png", dpi=300)
    plt.close()

    plt.figure()
    plt.hist(metrics["f1"], bins=20)
    plt.axvline(mean_metrics["f1"], color="red", linestyle="dashed")
    plt.xlim(0, 1)
    plt.title("F1 Score")
    plt.ylabel("Count")
    plt.title(f"F1 Score - Mean: {mean_metrics['f1']:.3f}")
    plt.savefig(f"results/{args.course_name}/f1.png", dpi=300)
    plt.close()

    plt.figure()
    plt.hist(metrics["auc"], bins=20)
    plt.axvline(mean_metrics["auc"], color="red", linestyle="dashed")
    plt.xlim(0, 1)
    plt.title("AUC")
    plt.ylabel("Count")
    plt.title(f"AUC - Mean: {mean_metrics['auc']:.3f}")
    plt.savefig(f"results/{args.course_name}/auc.png", dpi=300)
    plt.close()
