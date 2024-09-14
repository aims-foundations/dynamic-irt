import os
import pickle
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from huggingface_hub import snapshot_download


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def count_tokens(text):
    return len(text.split())


def count_per_student(directory):
    response_matrix = load_pickle(f"{directory}/response_matrix.pkl")
    unique_questions = load_pickle(f"{directory}/unique_questions.pkl")
    tokens_per_student = [0] * len(response_matrix)

    for s_idx, student_responses in enumerate(response_matrix):
        for q_idx, responses in enumerate(student_responses):
            question_text = unique_questions[q_idx]
            tokens_per_student[s_idx] += count_tokens(question_text)

            for response in responses:
                tokens_per_student[s_idx] += count_tokens(response)

    return tokens_per_student


def plot(repo_id, token_counts, num_bins=None):
    plt.figure(figsize=(10, 6))
    if num_bins is None:
        num_bins = "auto"
    counts, bins, patches = plt.hist(
        token_counts, bins=num_bins, color="blue", alpha=0.7
    )
    plt.title(f"Total tokens per student in class {repo_id}")
    plt.xlabel("Number of tokens")
    plt.ylabel("Number of students")
    plt.grid(True)
    plt.savefig(f"{repo_id}_total_tokens.png")
    plt.close()
    print("total students processed:", sum(counts))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--course_name", help="Class Name", type=str, default="dsa_hk231"
    )
    args = parser.parse_args()
    directory = snapshot_download(
        repo_id=f"stair-lab/{args.repo_id}", repo_type="dataset"
    )

    tokens_per_student = count_per_student(directory)
    plot(args.repo_id, tokens_per_student, 20)
