import os
import pickle

from argparse import ArgumentParser

from huggingface_hub import snapshot_download

from utils import parse_time

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    args = parser.parse_args()

    # Download and load data
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}_wtc", repo_type="dataset"
    )

    unique_questions = pickle.load(
        open(os.path.join(data_folder, "unique_questions.pkl"), "rb")
    )
    # >>> n_question

    correctness_matrix = pickle.load(
        open(os.path.join(data_folder, "correctness_matrix.pkl"), "rb")
    )
    time_matrix = pickle.load(open(os.path.join(data_folder, "time_matrix.pkl"), "rb"))
    # >>> n_student x n_question x n_attempt

    breakpoint()
    print("ABCD")
