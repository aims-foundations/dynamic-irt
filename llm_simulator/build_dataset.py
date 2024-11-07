import argparse
import pickle

import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

SYSTEM_PROMPT = "You are a student of a programming course."
INSTRUCTION = "Question: {question}\nWrite your answer for the above question in C++."
ATTEMPT_INSTRUCTION = (
    "Your score: {score}/1.\nRetry answering the question for higher score."
)


def format_chat_template(tokenizer, questions, responses, scores):
    # questions: List[str]
    # responses: List[List[List[str]]]
    n_student, n_question, n_attempt = scores.shape
    list_text = []
    for sidx in range(n_student):
        for qidx in range(n_question):
            conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

            for aidx in range(n_attempt):
                if scores[sidx, qidx, aidx] == -1:
                    break

                if aidx == 0:
                    conversation.append(
                        {
                            "role": "user",
                            "content": INSTRUCTION.format(question=questions[qidx]),
                        }
                    )
                else:
                    conversation.append(
                        {
                            "role": "user",
                            "content": ATTEMPT_INSTRUCTION.format(
                                score=round(scores[sidx][qidx][aidx], 2)
                            ),
                        }
                    )

                conversation.append(
                    {"role": "assistant", "content": responses[sidx][qidx][aidx]}
                )

            if len(conversation) < 2:
                continue

            list_text.append(
                tokenizer.apply_chat_template(conversation, tokenize=False)
            )

    dataset = Dataset.from_dict({"text": list_text})
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument(
        "--model",
        help="Model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    args = parser.parse_args()

    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Download and load data
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}", repo_type="dataset"
    )

    questions = pickle.load(open(f"{data_folder}/unique_questions.pkl", "rb"))

    responses = pickle.load(open(f"{data_folder}/response_matrix.pkl", "rb"))

    scores = pickle.load(open(f"{data_folder}/correctness_matrix.pkl", "rb"))
    scores = np.array(scores)

    # Split data
    response_train, response_test, score_train, score_test = train_test_split(
        responses, scores, test_size=0.2, random_state=args.seed
    )

    # Format for chat template
    dataset_train = format_chat_template(
        tokenizer, questions, response_train, score_train
    )
    dataset_test = format_chat_template(tokenizer, questions, response_test, score_test)

    dataset_dict = DatasetDict(
        {
            "train": dataset_train,
            "test": dataset_test,
        }
    )

    # Save data to hub
    dataset_dict.push_to_hub(f"stair-lab/{args.course_name}_sft")
