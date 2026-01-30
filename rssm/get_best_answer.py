import argparse
import os
import pickle
import random

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from utils import ensure_dir
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        default="stair-lab/dsa_hk231_wtc",
    )
    args = parser.parse_args()

    data_folder = snapshot_download(repo_id=args.dataset, repo_type="dataset")
    response_matrix = pickle.load(open(f"{data_folder}/response_matrix.pkl", "rb"))
    correctness_matrix = pickle.load(
        open(f"{data_folder}/correctness_matrix.pkl", "rb")
    )
    student_ids = pickle.load(open(f"{data_folder}/student_ids.pkl", "rb"))
    unique_questions = pickle.load(open(f"{data_folder}/unique_questions.pkl", "rb"))
    question_name2idx = pickle.load(open(f"{data_folder}/question_name2idx.pkl", "rb"))

    # Filter students in target class
    list_class_student_idxs = []
    for sid, student in enumerate(student_ids):
        if student["class"] == args.cls or args.cls == "all_cls":
            list_class_student_idxs.append(sid)

    question_idx2name = {v: k for k, v in question_name2idx.items()}

    # Filter questions that these student have done
    list_best_answers = {}
    list_best_scores = {}
    for sid in tqdm(list_class_student_idxs):
        for qidx, question_attempts in enumerate(response_matrix[sid]):
            for aidx, attempt in enumerate(question_attempts):
                if attempt == "":
                    continue
                qname = question_idx2name[qidx]
                if qname in list_best_answers:
                    if correctness_matrix[sid][qidx][aidx] > list_best_scores[qname]:
                        list_best_answers[qname] = response_matrix[sid][qidx][aidx]
                        list_best_scores[qname] = correctness_matrix[sid][qidx][aidx]
                else:
                    list_best_answers[qname] = response_matrix[sid][qidx][aidx]
                    list_best_scores[qname] = correctness_matrix[sid][qidx][aidx]
    ensure_dir(f"data/{args.cls}")
    print("Total questions:", len(list_best_answers))
    pickle.dump(list_best_answers, open(f"data/{args.cls}/best_answers.pkl", "wb"))
    pickle.dump(list_best_answers, open(f"data/{args.cls}/best_scores.pkl", "wb"))
