import argparse
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams


def format_prompt(tokenizer, system, history, query, response):
    conversation = [{"role": "system", "content": system}]
    for hist in history:
        conversation.append({"role": "user", "content": hist[0]})
        conversation.append({"role": "assistant", "content": hist[1]})

    conversation.append({"role": "user", "content": query})
    conversation.append({"role": "assistant", "content": response})
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    return prompt


def compute_perplexity(model, tokenizer, sample, sampling_params={}):
    prompt = format_prompt(
        tokenizer,
        sample["system"],
        sample["history"],
        sample["instruction"],
        sample["output"],
    )
    if len(tokenizer.encode(prompt)) > 131000:
        return None

    completion = infer_completion(model, prompt, sampling_params)
    prompt_logprobs = completion[0].prompt_logprobs

    total_tokens = len(prompt_logprobs)

    # Count tokens of output
    output_tokens = tokenizer.encode(sample["output"])
    list_logprobs = []
    for i, _ in enumerate(output_tokens):
        token_prob = prompt_logprobs[total_tokens - i - 1]
        if (-list(token_prob.values())[0].logprob) == np.inf:
            continue
        list_logprobs.append(-list(token_prob.values())[0].logprob)

    list_logprobs = np.array(list_logprobs)

    # Compute perplexity
    perplexity = np.exp(np.mean(list_logprobs))
    return perplexity


def infer_completion(model, prompt, sampling_params={}):
    return model.generate(prompt, sampling_params=sampling_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, nargs="+")
    parser.add_argument(
        "--dataset",
        type=str,
        default="stair-lab/dsa_hk231_wtc_per_student_sft_lf_splited",
    )
    args = parser.parse_args()

    for cls in args.cls:
        model_id = f"saves/{cls}"
        # model_id = "meta-llama/Llama-3.2-3B-Instruct"

        model = LLM(model_id, dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        generation_config = GenerationConfig.from_pretrained(model_id)
        sampling_params = SamplingParams(
            temperature=1.0,
            top_k=1,
            prompt_logprobs=1,
            max_tokens=1,
            truncate_prompt_tokens=130000,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        dataset = load_dataset(args.dataset, split=cls + "_test")

        # Random sampling for choosing 100 samples to evaluate
        chosen_idxs = random.choices(list(range(len(dataset))), k=1000)
        eval_dataset = dataset.select(chosen_idxs)

        list_perplexity = []
        for i, sample in enumerate(tqdm(eval_dataset)):
            preplexity = compute_perplexity(
                model, tokenizer, sample, sampling_params=sampling_params
            )

            if preplexity is not None:
                list_perplexity.append(preplexity)

            if (i + 1) % 50 == 0:
                print(f"Mean perplexity: {sum(list_perplexity) / len(list_perplexity)}")

        print(f"Class: {cls}")
        print(f"Mean perplexity: {sum(list_perplexity) / len(list_perplexity)}")

        # Save to file
        with open(f"results/perplexity_{cls}.txt", "w") as f:
            f.write("\n".join([str(x) for x in list_perplexity]))
            f.write(
                "\n" + f"Mean perplexity: {sum(list_perplexity) / len(list_perplexity)}"
            )
