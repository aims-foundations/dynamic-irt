import copy
import difflib
import os
import random
import re
import subprocess
import time

import numpy as np
import requests
import torch

from config import (
    RETRY_END_TAG,
    RETRY_SEPARATOR,
    RETRY_START_TAG,
    STOP_ANSWER,
    USER_TAG,
)
from grading_engine import CPPEvaluator
from Levenshtein import distance
from openai import OpenAI
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams


def restore_code_answer(last_answer, edit_response):
    list_lines = last_answer.split("\n")
    list_replace = parse_replace_string(edit_response)

    for original_code, replace_code in list_replace:
        eds = compute_ed(original_code, list_lines)
        chosen_idx = np.argmin(eds)
        list_lines[chosen_idx] = replace_code
    return "\n".join(list_lines)


def parse_replace_string(s):
    result = []
    lines = s.strip().splitlines()
    current_pair = ["", ""]

    is_new_res = False
    for line in lines:
        line = line.strip()
        if line == RETRY_START_TAG:
            is_new_res = False
            current_pair = ["", ""]  # Start a new pair
            continue
        elif line == RETRY_SEPARATOR:
            is_new_res = True
            continue  # Skip the "WITH" line
        elif line == RETRY_END_TAG:
            result.append(current_pair.copy())
        elif line:  # Only add non-empty lines
            if is_new_res:
                idx = 1
            else:
                idx = 0
            if current_pair[idx] != "":
                line = "\n" + line
            current_pair[idx] += line

    return result


def compute_ed(original, list_str):
    return [distance(original, x) for x in list_str]


def parse_score_from_feedback(sample):
    return float(re.search(r"Your score:\s*([0-9]*\.?[0-9]+)\/[0-9]+", sample).group(1))


def parse_question_name(question_names):
    return_names = {}
    for qidx, q in enumerate(question_names):
        if "|" in q:
            names = q.split("|")
            for sub_idx, name in enumerate(names):
                return_names[f"{qidx+1}.{sub_idx+1}"] = name
        else:
            return_names[str(qidx + 1)] = q

    return return_names


def get_model(model_name_or_path):
    model = LLM(model_name_or_path, gpu_memory_utilization=0.9, tensor_parallel_size=4)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    return model, tokenizer, generation_config


def get_model_api(model_name_or_path, port=9820):
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )
    models = client.models.list()
    model = models.data[0].id
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    return client, model, tokenizer, generation_config


def format_prompt(system, history, tokenizer, is_answer=True):
    conversation = [{"role": "system", "content": system}]
    for hist in history[:-1]:
        conversation.append({"role": "user", "content": hist[0]})
        conversation.append({"role": "assistant", "content": hist[1]})
    if is_answer:
        conversation.append({"role": "user", "content": history[-1][0]})
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=is_answer
    )
    if not is_answer:
        prompt = prompt + USER_TAG
    return prompt


def infer_completion(prompt, model, generation_config, n=1):
    sampling_params = SamplingParams(
        n=n,
        best_of=32,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        max_tokens=2048,
        stop=STOP_ANSWER,
    )
    completion = model.generate(prompt, sampling_params)
    completion = [[go.text for go in g.outputs] for g in completion]
    return completion


def infer_completion_api(prompt, client, model, generation_config, n=1):
    completions = client.completions.create(
        model=model,
        prompt=prompt,
        echo=False,
        n=n,
        stream=False,
        best_of=32,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        max_tokens=2048,
        stop=STOP_ANSWER,
        timeout=1800,
    )
    completion = [c.text for c in completions.choices]
    return completion


def get_question_info_by_name(question_name, question_name2idx, question_infos):
    return question_infos[question_name2idx[question_name]]


def get_cpp_evaluator(question_names, *args, **kwargs):
    list_evaluators = {}
    for combine_name in question_names:
        names = combine_name.split("|")
        for name in names:
            content, template, testcases = get_question_info_by_name(
                name, *args, **kwargs
            )
            evaluator = CPPEvaluator(template, testcases, max_workers=4)
            list_evaluators[name] = evaluator
    return list_evaluators


def infer_score(evaluator, answer):
    student_results = evaluator.evaluate(answer)
    return student_results["score"]


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def string_diff(tokenizer, a, b):
    # Initialize the result dictionary
    diff_result = []

    tokenized_a = a.split("\n")
    tokenized_b = b.split("\n")

    # Get the differences between the two strings using SequenceMatcher
    seq_matcher = difflib.SequenceMatcher(
        None, tokenized_a, tokenized_b, autojunk=False
    )

    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        if tag == "replace":
            for ii, (tka, tkb) in enumerate(
                zip(tokenized_a[i1:i2], tokenized_b[j1:j2])
            ):
                diff_result.append((tka, tkb))
        elif tag == "delete":
            diff_result.append(("\n".join(tokenized_a[i1:i2]), ""))
        elif tag == "insert":
            diff_result.append(
                (
                    "\n".join(tokenized_a[i1 - 1 : i2]),
                    "\n".join(tokenized_b[j1 - 1 : j2]),
                )
            )

    return diff_result


def remove_spaces(s):
    ss = re.sub("[^\S\n]{2,}", " ", s)
    return re.sub("[\n]{2,}", "\n", ss)


def run_server(cmd_string):
    try:
        server_process = subprocess.Popen(
            cmd_string,
            shell=True,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE
        )
        return server_process
    except Exception as e:
        print(f"Error starting server: {e}")
        return None


def shutdown_server(process):
    try:
        kill(process.pid)
        # process.terminate()
        print("Server shutdown successfully.")
    except Exception as e:
        print(f"Error shutting down server: {e}")


def check_health(url):
    time.sleep(60)
    server_ok = False
    while server_ok is False:
        try:
            # Send a GET request to the health check endpoint
            response = requests.get(url)

            # Check if the server is healthy
            if response.status_code == 200:
                server_ok = True
            else:
                time.sleep(1)

        except requests.exceptions.RequestException as e:
            time.sleep(1)
    return server_ok
