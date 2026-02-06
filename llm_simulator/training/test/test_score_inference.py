import pickle
import re
import sys
from argparse import ArgumentParser

import wandb
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
pattern = r"\s*([0-9]*\.?[0-9]+)\/[0-9]+"
USER_TAG = "<|start_header_id|>user<|end_header_id|>\n\n"
STOP_TOKEN = "<|eot_id|>"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id
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
    # conversation.append(
    #      {"role": "user", "content": sample["instruction"]}
    # )
    # conversation.append(
    #      {"role": "assistant", "content": sample["output"]}
    # )
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False) + USER_TAG
    return prompt


def infer_completion(prompt):
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        stop=[
            STOP_TOKEN,
        ],
        echo=False,
        n=1,
        logprobs=1,
        max_tokens=20,
        stream=False,
    )
    return completion


if __name__ == "__main__":
    wandb.init()
    parser = ArgumentParser()
    parser.add_argument("--cls", type=str, default="CC01")
    args = parser.parse_args()

    ds = load_dataset(
        "stair-lab/dsa_hk231_per_student_sft_lf_splited", split=args.cls + "_test"
    )

    acc = []
    mae = []
    for sample in tqdm(ds):
        if not sample["instruction"].startswith("Your score:") or sample["week"] < 3:
            continue

        prompt = format_sample(sample)
        prompt_length = len(tokenizer(prompt)["input_ids"])

        pred_score = None
        trial = 0
        while trial < 10:
            try:
                completion = infer_completion(prompt + "Your score: ")
                match = re.search(pattern, completion.choices[0].text)
                pred_score = float(match.group(1))
                break
            except:
                trial += 1

        if pred_score is None:
            pred_score = 0.0

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
