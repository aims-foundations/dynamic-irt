import argparse
import os
import pickle
import re

from datasets import Dataset, load_dataset
from embed_text_package.embed_text_v2 import Embedder
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import (
    ensure_dir,
    format_template_testcase,
    get_question_info_by_name,
    parse_question_name,
)


def parse_score_from_feedback(sample):
    return float(re.search(r"Your score:\s*([0-9]*\.?[0-9]+)\/[0-9]+", sample).group(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="stair-lab/dsa_hk231_wtc_per_student_sft_lf"
    )
    parser.add_argument("--cls", type=str, default="all_cls")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--model",
        help="Model",
        type=str,
        # default="/lfs/local/0/nqduc/Llama-3.1-8B-embedding",
        default="deepseek-ai/DeepSeek-Coder-V2-Instruct",
    )
    parser.add_argument(
        "--num_gpu",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.cls)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load question infos ans best_answers
    data_folder = snapshot_download(
        repo_id=f"stair-lab/dsa_hk231_wtc", repo_type="dataset"
    )
    question_infos = pickle.load(open(f"{data_folder}/unique_questions.pkl", "rb"))
    question_name2idx = pickle.load(open(f"{data_folder}/question_name2idx.pkl", "rb"))
    best_answers = pickle.load(open(f"data/{args.cls}/best_answers.pkl", "rb"))

    list_questions_by_week = []
    list_tcs_by_week = []
    list_best_ans_by_week = []
    list_student_attempts = []
    list_student_scores = []
    list_student_tc_scores = []
    student_idxs = []
    week_idxs = []

    # Find maximum number of testcases per question
    max_num_tcs = 0
    for row in tqdm(ds):
        if len(row["testcase_scores"]) > max_num_tcs:
            max_num_tcs = len(row["testcase_scores"])

    print(f"Max number of testcases: {max_num_tcs}")

    student_idx = -1
    is_practice = False
    total_rows = len(ds)
    for ri, row in enumerate(tqdm(ds)):
        if len(row["history"]) == 0:
            # New student
            student_idx += 1

        if "Here are the exercise questions for practice." in row["instruction"]:
            ques = (
                row["instruction"]
                .replace("Here are the exercise questions for practice.", "")
                .strip()
            )
            list_questions_by_week.append(ques)
            ques_names = parse_question_name(row["question_name"])
            tcs_by_week = ""
            best_ans_by_week = ""

            for qid, q_name in ques_names.items():
                content, template, testcases = get_question_info_by_name(
                    q_name, question_name2idx, question_infos
                )
                tcs_by_week += f"Question {qid}:\n" + format_template_testcase(
                    tokenizer, template, testcases
                )

                best_ans_by_week += (
                    f"Solution for question {qid}:\n" f"{best_answers[q_name]}\n"
                )

            list_tcs_by_week.append(tcs_by_week)
            list_best_ans_by_week.append(best_ans_by_week)
            is_practice = True

        if "Here are the exam questions." in row["instruction"]:
            is_practice = False

        if ri == total_rows - 1:
            break

        if is_practice:
            if ds[ri + 1]["instruction"].startswith("Your score"):
                # week_idxs.append(row["week"])
                week_idxs.append(len(list_questions_by_week) - 1)
                student_idxs.append(student_idx)
                list_student_attempts.append(row["output"])

                score = parse_score_from_feedback(ds[ri + 1]["instruction"])
                list_student_scores.append(score)

                tc_score = ds[ri + 1]["testcase_scores"]
                if len(tc_score) < max_num_tcs:
                    tc_score += [-1] * (max_num_tcs - len(tc_score))
                list_student_tc_scores.append(tc_score)

    # Load model
    embedder = Embedder()
    embedder.load(
        args.model,
        tensor_parallel_size=args.num_gpu,
        enable_chunked_prefill=False,
        enforce_eager=True,
        trust_remote_code=True,
        # gpu_memory_utilization=0.9
    )

    # Embed questions
    print("Embedding questions")
    ds_ques = Dataset.from_dict({"text": list_questions_by_week})
    ques_emb = (
        embedder.get_embeddings(
            DataLoader(ds_ques, batch_size=args.batch_size),
            embedder.which_model,
            ["text"],
        )
        .data["text"]
        .to_pylist()
    )

    print("Embedding testcases")
    ds_tcs = Dataset.from_dict({"text": list_tcs_by_week})
    tcs_emb = (
        embedder.get_embeddings(
            DataLoader(ds_tcs, batch_size=args.batch_size),
            embedder.which_model,
            ["text"],
        )
        .data["text"]
        .to_pylist()
    )

    print("Embedding best answers")
    ds_bans = Dataset.from_dict({"text": list_best_ans_by_week})
    bans_emb = (
        embedder.get_embeddings(
            DataLoader(ds_bans, batch_size=args.batch_size),
            embedder.which_model,
            ["text"],
        )
        .data["text"]
        .to_pylist()
    )

    print("Embedding answers")
    ds_answer = Dataset.from_dict({"text": list_student_attempts})
    answer_emb = (
        embedder.get_embeddings(
            DataLoader(ds_answer, batch_size=args.batch_size),
            embedder.which_model,
            ["text"],
        )
        .data["text"]
        .to_pylist()
    )

    # Save
    ensure_dir(f"data/{args.cls}")

    with open(f"data/{args.cls}/questions.pkl", "wb") as f:
        pickle.dump(ques_emb, f)

    with open(f"data/{args.cls}/testcases.pkl", "wb") as f:
        pickle.dump(tcs_emb, f)

    with open(f"data/{args.cls}/best_answer_by_week.pkl", "wb") as f:
        pickle.dump(bans_emb, f)

    with open(f"data/{args.cls}/answers.pkl", "wb") as f:
        pickle.dump(answer_emb, f)

    with open(f"data/{args.cls}/scores.pkl", "wb") as f:
        pickle.dump(list_student_scores, f)

    with open(f"data/{args.cls}/testcase_scores.pkl", "wb") as f:
        pickle.dump(list_student_tc_scores, f)

    with open(f"data/{args.cls}/student_idxs.pkl", "wb") as f:
        pickle.dump(student_idxs, f)

    with open(f"data/{args.cls}/week_idxs.pkl", "wb") as f:
        pickle.dump(week_idxs, f)
