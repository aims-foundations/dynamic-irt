"""
LLM-based feature extraction for code problems.

This script extracts code complexity metrics using vLLM models:
- Lines of code
- Required steps to solve

Note: Requires GPU configuration for vLLM. Not recommended to run
without proper setup.
"""

import re
import numpy as np
import torch
import vllm

from dynamic_irt.elo import utils


def create_vllm_model(model_name="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=4):
    """
    Initialize vLLM model for feature extraction.

    Parameters:
        model_name (str): HuggingFace model name
        tensor_parallel_size (int): Number of GPUs to use

    Returns:
        tuple: (vllm.LLM model, vllm.SamplingParams)
    """
    model = vllm.LLM(
        model_name,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=False,
        enforce_eager=True,
        dtype=torch.float16,
        swap_space=32,
        max_num_seqs=128,
        max_model_len=12800,
        tensor_parallel_size=tensor_parallel_size,
    )

    generation_config = vllm.SamplingParams(
        n=8,
        best_of=16,
    )
    generation_config.n = 16
    generation_config.best_of = 32

    return model, generation_config


def run_llama(model, generation_config, prompt):
    """
    Generate output from the vLLM model.

    Parameters:
        model: vLLM model instance
        generation_config: vLLM sampling parameters
        prompt (str): Input prompt

    Returns:
        str: Generated text
    """
    generation_results = model.generate([prompt], generation_config)
    generated_text = "".join(
        [output.text for output in generation_results[0].outputs]
    ).strip()
    return generated_text


def extract_lines_of_code(model, generation_config, code):
    """
    Extract number of lines of code using LLM.

    Parameters:
        model: vLLM model instance
        generation_config: vLLM sampling parameters
        code (str): Source code

    Returns:
        int or np.nan: Number of lines
    """
    prompt = (
        "Given the following code in text format, "
        "count the number of lines it would take if typed on a programming platform.\n\n"
        "Return ONLY a single integer on its own line. "
        "No words. No punctuation. No explanation. Just a number like:\n"
        "7\n\n"
        "Here is the code:\n{}\n\n"
        "Integer only:"
    ).format(code)

    llm_response = run_llama(model, generation_config, prompt)
    match = re.search(r"^\s*(\d+)\s*$", llm_response.strip(), re.MULTILINE)

    if match:
        return int(match.group(1))
    else:
        return np.nan


def extract_required_steps(model, generation_config, question_text, solution):
    """
    Extract number of required steps to solve using LLM.

    Parameters:
        model: vLLM model instance
        generation_config: vLLM sampling parameters
        question_text (str): Problem description
        solution (str): Solution code

    Returns:
        int or np.nan: Number of steps
    """
    prompt = (
        "You are an expert evaluator for first-year undergraduate "
        "computer science students in Vietnam solving C++ problems.\n"
        "Given the following question:"
        "{}\n\n".format(question_text)
        + "And the following correct solution:"
        "{}\n\n".format(solution)
        + "Determine how many steps a student must follow to arrive at the correct answer. "
        "Do not explain the process but ONLY Return a single integer "
        "representing the number of steps, like 5 or 10. "
        "DO NOT PRODUCE MORE THAN THE INTEGER OUTPUT"
    )

    llm_response = run_llama(model, generation_config, prompt)
    match = re.search(r"^\s*(\d+)\s*$", llm_response.strip(), re.MULTILINE)

    if match:
        return int(match.group(1))
    else:
        return np.nan


if __name__ == "__main__":
    # Initialize model
    model, generation_config = create_vllm_model()

    # Load data
    data = utils.load_csv_from_gdrive_by_id(
        "126QqK18Ej8AFrHoscmLD2a5mBe7EEBw1"
    )
    data["required_steps"] = 0
    data["lines"] = 0

    for index, row in data.iterrows():
        question_text = row["question_text"]
        question_solution = row["response"]

        # Extract lines of code
        lines = extract_lines_of_code(model, generation_config, question_solution)
        data.at[index, "lines"] = lines

        # Extract required steps
        steps = extract_required_steps(
            model, generation_config, question_text, question_solution
        )
        data.at[index, "required_steps"] = steps

        print(f"Processed row {index}: lines={lines}, steps={steps}")

    print(data[["question_text", "lines", "required_steps"]].head())
