import pickle
import re
import sys
from argparse import ArgumentParser

import wandb
from config import FIRST_ATTEMPT_PREFIX, RETRY_PREFIX, START_WEEK
from datasets import load_dataset
from grading_engine.engine import CPPEvaluator
from huggingface_hub import snapshot_download
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from utils import (
    get_cpp_evaluator,
    get_question_info_by_name,
    infer_completion,
    infer_score,
    parse_question_name,
)

pattern = r"\s*([0-9]*\.?[0-9]+)\/[0-9]+"

tokenizer = AutoTokenizer.from_pretrained(
    model,
    use_fast=True,
)
generation_config = GenerationConfig.from_pretrained(model)


def format_sample(sample):
    conversation = [{"role": "system", "content": sample["system"]}]
    for hist in sample["history"]:
        conversation.append({"role": "user", "content": hist[0]})
        conversation.append({"role": "assistant", "content": hist[1]})

    conversation.append({"role": "user", "content": sample["instruction"]})
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    return prompt


if __name__ == "__main__":
    wandb.init()
    parser = ArgumentParser()
    parser.add_argument("--cls", type=str, default="CC01")
    args = parser.parse_args()

    ds = load_dataset(
        "stair-lab/dsa_hk231_v3_per_student_sft_lf_splited", split=args.cls + "_test"
    )
    model, tokenizer, generation_config = get_model(f"saves/All")

    data_folder = snapshot_download(
        repo_id=f"stair-lab/dsa_hk231_wtc", repo_type="dataset"
    )
    question_infos = pickle.load(open(f"{data_folder}/unique_questions.pkl", "rb"))
    question_name2idx = pickle.load(open(f"{data_folder}/question_name2idx.pkl", "rb"))

    acc = []
    mae = []
    list_evaluators = {}
    question_names = {}
    for sample in tqdm(ds):
        if len(sample["question_name"]) > 0:
            list_evaluators.update(
                get_cpp_evaluator(
                    sample["question_name"], question_name2idx, question_infos
                )
            )
            question_names = parse_question_name(sample["question_name"])

        if (
            not sample["instruction"].startswith("Your score:")
            or sample["week"] < START_WEEK
        ):
            continue

        prompt = format_sample(sample)
        prompt_length = len(tokenizer(prompt)["input_ids"])
        if prompt_length > 129024:
            continue

        prompt += sample["output"].split(":")[0]
        completions = infer_completion(prompt, n=32)
        chosen_ex_ques_idx = None
        chosen_ex_ques_name = None
        chosen_ex_answer = None
        chosen_ex_score = None

        for completion in completions.choices:
            clean_ex_answer = completion.text.strip()

            # spliting_idx = ex_answer.find(":")
            # ex_ques_idx = ex_answer[:spliting_idx].strip()
            # if ex_ques_idx not in question_names:
            #     continue
            # clean_ex_answer = ex_answer[spliting_idx + 1 :].strip()

            exam_question_name = question_names[ex_ques_idx]
            ex_score = infer_score(list_evaluators[exam_question_name], clean_ex_answer)

            if chosen_ex_score is None or ex_score > chosen_ex_score:
                chosen_ex_ques_idx = ex_ques_idx
                chosen_ex_ques_name = exam_question_name
                chosen_ex_answer = ex_answer
                chosen_ex_score = ex_score

        pred_score = chosen_ex_score
        match = re.search(pattern, sample["instruction"])
        gt_score = float(match.group(1))

        if pred_score == gt_score:
            acc.append(1)
        else:
            acc.append(0)

        mae.append(abs(gt_score - pred_score))
        res = {
            "gt": gt_score,
            "pred": pred_score,
            "acc": acc[-1],
            "mae": mae[-1],
            "mean_acc": sum(acc) / len(acc),
            "mean_mae": sum(mae) / len(mae),
        }
        print(res)
        wandb.log(res)

    mean_acc = sum(acc) / len(acc)
    mean_mae = sum(mae) / len(mae)
    with open(f"{args.cls}.csv", "w") as f:
        f.write("Mean Acc,Mean MAE,Number of trials\n")
        f.write(f"{mean_acc},{mean_mae},{len(mae)}")

    wandb.finish()
