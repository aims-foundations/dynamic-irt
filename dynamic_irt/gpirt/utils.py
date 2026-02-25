import json
import os
import random
import re
from datetime import datetime

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch
from Levenshtein import distance
from tqdm import tqdm
from tueplots import bundles

plt.rcParams.update(bundles.aaai2024())

STARTING_TIME = datetime.strptime("1/9/23, 00:00:00", "%d/%m/%y, %H:%M:%S")


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_prior_distribution(theta_priors, sidx, npoints, save_file):
    # Plot prior distribution for theta at student 415
    prior = theta_priors[sidx]
    samples = prior.sample(torch.Size([20]))[:, -npoints:]
    for sample in samples:
        plt.plot(sample.cpu().numpy(), color="black", alpha=0.1)
    plt.savefig(save_file, dpi=300)
    plt.close()


def plot_correlation(x, y, x_label, y_label, fig_title, save_file):
    plt.figure(figsize=(5, 5))
    axis_max = max(x.max(), y.max())
    axis_min = min(x.min(), y.min())
    plt.scatter(x, y)
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    plt.savefig(f"plots/{save_file}", dpi=300)


def compute_ed(original, list_str):
    return [distance(original, x) for x in list_str]


def parse_time(time_str):
    # Parsing the string into a datetime object
    parsed_datetime = datetime.strptime(time_str, "%d/%m/%y, %H:%M:%S") - STARTING_TIME
    # Convert to days
    return parsed_datetime.total_seconds() / 86400


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def parse_score(header_text):
    search_result = re.search(r"/(\d+\.\d+)", header_text)
    if search_result:
        return float(search_result.group(1))
    else:
        return None


def get_ability_priors_pyro(
    unique_time_vec, kernel, length_scale=1.0, npoints=0, device="cpu"
):
    uni_time = unique_time_vec.unsqueeze(-1).double()
    if kernel == "Matern":
        kernel = gp.kernels.Matern52(
            input_dim=1, lengthscale=torch.tensor(length_scale)
        ).double()
    elif kernel == "RBF":
        kernel = gp.kernels.RBF(
            input_dim=1,
            lengthscale=torch.tensor(length_scale),
        ).double()
    else:
        raise ValueError("Invalid kernel type")

    covar = kernel(uni_time) + 1e-4 * torch.eye(
        uni_time.shape[0],
        device=device,
        dtype=torch.double,
    )

    return dist.MultivariateNormal(
        torch.zeros(covar.shape[0], device=device, dtype=torch.double), covar
    )


def preprocess(
    response_matrix,
    response_time_matrix,
    low_rank_configs,
    device,
):
    n_students, n_questions = response_matrix.shape[:2]
    if response_matrix.ndim == 3:
        n_max_attempts = response_matrix.shape[2]

    # response_time_matrix: n_students x n_questions x n_attempts
    if low_rank_configs["type"] == "GP":
        observation_mask = (response_matrix != -1).cpu()
        response_time_indexes = []
        question_expanding_indexes = []
        ability_prior_dists = []

        for sidx in tqdm(range(n_students), desc="Constructing indexes"):
            uni_time = response_time_matrix[sidx].unique()
            has_missing = 1 if -1 in uni_time else 0

            # Get the time index for each attempt
            time_index = (
                torch.searchsorted(uni_time, response_time_matrix[sidx]) - has_missing
            )
            ### REMEMBER: time_index is 0-indexed. In case, element 0 is -1
            ### We need to subtract 1 to get the correct index
            response_time_indexes.append(time_index[time_index != -1])
            question_expanding_indexes.append(
                torch.arange(n_questions, device=device)[:, None].expand(
                    -1, n_max_attempts
                )[observation_mask[sidx]]
            )

            # Remove the first element since it is -1
            uni_time = uni_time[has_missing:]
            ability_prior_dists.append(
                get_ability_priors_pyro(
                    uni_time,
                    kernel=low_rank_configs["kernel"],
                    length_scale=low_rank_configs["length_scale"],
                    device=device,
                )
            )

        question_expanding_indexes = torch.concatenate(question_expanding_indexes).cpu()

        question_observation_indexes = []
        for qidx in tqdm(range(n_questions), desc="Constructing question indexes"):
            matches = torch.where(question_expanding_indexes == qidx)[0]
            if len(matches) > 0:
                question_observation_indexes.append(matches[0].item())
            else:
                question_observation_indexes.append(-1)
        question_observation_indexes = torch.tensor(
            question_observation_indexes, device="cpu"
        )

    else:
        raise NotImplementedError("Work in progress")

    if low_rank_configs["type"] == "GP":
        item_prior_dists = dist.Normal(
            torch.zeros(n_questions, device=device),
            torch.ones(n_questions, device=device),
        )
    else:
        raise NotImplementedError("Work in progress")

    return (
        observation_mask,
        response_time_indexes,
        question_expanding_indexes,
        ability_prior_dists,
        item_prior_dists,
    )


def find_global_max(repo_id, course_name, class_name):
    global_max = 0

    for root, dirs, files in os.walk(repo_id):
        if class_name in dirs:
            class_dir_path = os.path.join(root, class_name)

            for subdir_root, subdir_dirs, subdir_files in os.walk(class_dir_path):
                for file in subdir_files:
                    if file.endswith(".json"):
                        file_path = os.path.join(subdir_root, file)
                        try:
                            with open(file_path, "r") as json_file:
                                data = json.load(json_file)
                                ids = []
                                for answers in data["student_answers"]:
                                    ids.append(answers["id"])

                                for idx in range(len(data["list_questions"])):
                                    max_score = data["list_questions"][idx][
                                        "max_scores"
                                    ]
                                    q_index = idx + 1

                                    records = []
                                    for answers in data["student_answers"]:
                                        for answer in answers["response_history"]:
                                            marks = []
                                            if (
                                                answer["question"]
                                                == f"Question {q_index}"
                                            ):
                                                mark_per_attempt = []
                                                for score_idx in range(
                                                    len(answer["results"])
                                                ):
                                                    if (
                                                        answer["results"][score_idx][
                                                            "marks"
                                                        ]
                                                        != ""
                                                    ):
                                                        mark_per_attempt.append(
                                                            answer["results"][
                                                                score_idx
                                                            ]["marks"]
                                                        )

                                            marks.append(mark_per_attempt)
                                        records.extend(marks)

                                    try:
                                        records = [
                                            [
                                                float(mark) * 10 / max_score
                                                for mark in student_marks
                                            ]
                                            for student_marks in records
                                        ]
                                    except:
                                        continue

                                    max_attempts = [
                                        len(student_marks) for student_marks in records
                                    ]
                                    global_max = max(global_max, max(max_attempts))

                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from file: {file_path}: {e}")
    return global_max
