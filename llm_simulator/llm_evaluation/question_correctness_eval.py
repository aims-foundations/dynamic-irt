"""
LLM-based code correctness evaluation using Llama 3.1/vLLM.

This module evaluates student code submissions for correctness by using an LLM
to determine if the code produces the expected output for given test inputs.
"""

import os
import pandas as pd
import numpy as np
import torch
import vllm
import gdown
from datasets import load_dataset, Dataset


def load_csv_from_gdrive_by_id(file_id):
    """
    Load a CSV file from Google Drive using its file ID.

    Parameters:
        file_id (str): The Google Drive file ID

    Returns:
        pandas.DataFrame: DataFrame containing the CSV data
    """
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        output = 'temp_csv_file.csv'
        gdown.download(url, output, quiet=False)
        df = pd.read_csv(output)
        return df
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def create_vllm_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Initialize vLLM model for code evaluation.

    Parameters:
        model_name (str): HuggingFace model name

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
        tensor_parallel_size=1,
    )

    generation_config = vllm.SamplingParams(
        n=1,
        best_of=1,
    )

    return model, generation_config


def run_llm(model, generation_config, prompt):
    """
    Run the LLM on a prompt and return the generated text.

    Parameters:
        model: vLLM model instance
        generation_config: vLLM sampling parameters
        prompt (str): Input prompt

    Returns:
        str: Generated text
    """
    outputs = model.generate(
        prompt,
        sampling_params=generation_config,
    )
    return outputs[0].outputs[0].text


def evaluate_code_correctness(unit_df, model, generation_config):
    """
    Evaluate code correctness for all rows in the dataframe.

    Parameters:
        unit_df (pd.DataFrame): DataFrame with question_unittest, question_english,
                                response, input, output columns
        model: vLLM model instance
        generation_config: vLLM sampling parameters

    Returns:
        pd.DataFrame: DataFrame with LLM_response column added
    """
    skipped = []
    unit_df["LLM_scoring"] = 0

    for i in range(len(unit_df)):
        ques_text = unit_df["question_english"][i]
        resp_example = unit_df["response"][i]
        input_ = unit_df["input"][i]
        output = unit_df["output"][i]

        prompt = (
            "Question:\n{0}\n\n"
            "Test Input:\n{1}\n\n"
            "Expected Output:\n{2}\n\n"
            "Student Code:\n{3}\n\n"
            "Let's think step by step to see if the code will produce the expected output:\n"
            "1. [Your first reasoning step here]\n"
            "2. [Your next reasoning step…]\n"
            "3. [...]\n\n"
            "### Final Answer\n"
            "Answer with exactly one digit:\n"
            "- 1 if the code is correct\n"
            "- 0 if the code is incorrect\n\n"
            "Final Answer: "
        ).format(ques_text, input_, output, resp_example)

        try:
            llm_response = run_llm(model, generation_config, prompt).strip()
            unit_df.loc[i, "LLM_response"] = llm_response
            print(f"Finished running row {i}")
        except Exception as e:
            print(f"Skipping row {i} due to error: {e}")
            skipped.append(i)
            unit_df.loc[i, "LLM_response"] = np.nan
            continue

    print(f"Skipped rows: {skipped}")
    return unit_df


def main():
    """Main function to run code correctness evaluation."""
    # Load dataset from HuggingFace
    ds = load_dataset(
        "Kazchoko/my_dataset",
        data_files="evaluated_questions_llm.csv",
        split="train",
    )
    unit_df = ds.to_pandas()

    # Initialize model
    model, generation_config = create_vllm_model()

    # Evaluate
    unit_df = evaluate_code_correctness(unit_df, model, generation_config)

    # Save results to HuggingFace Hub
    ds = Dataset.from_pandas(unit_df)

    # Get HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        ds.push_to_hub(
            repo_id="Kazchoko/my_dataset",
            private=False,
            token=hf_token,
        )
    else:
        print("Warning: HF_TOKEN not set. Results not pushed to HuggingFace Hub.")
        unit_df.to_csv("questions_llm_fullresponse.csv", index=False)
        print("Results saved to questions_llm_fullresponse.csv")


if __name__ == "__main__":
    main()
