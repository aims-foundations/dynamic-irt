import argparse

import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from tueplots import bundles, constants, cycler, figsizes
from tueplots.constants.color import palettes

plt.rcParams.update(bundles.aaai2024())
plt.rcParams.update({"figure.dpi": 300})
plt.rcParams.update(cycler.cycler(color=palettes.paultol_bright))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="stair-lab/dsa_hk231_wtc_per_student_sft"
    )
    parser.add_argument("--cls", type=str, default="all_cls")
    parser.add_argument(
        "--model",
        help="Model",
        type=str,
        # default="/lfs/local/0/nqduc/Llama-3.1-8B-embedding",
        default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.cls)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Compute maximum number of tokens of samples["output"]
    max_token = 0
    num_tokens = []
    for sample in tqdm(ds):
        max_token = max(max_token, len(tokenizer(sample["output"])["input_ids"]))
        num_tokens.append(len(tokenizer(sample["output"])["input_ids"]))
    print(f"Max number of tokens: {max_token}")

    # Draw histogram of number of tokens
    figsize = figsizes.aaai2024_half(nrows=1, ncols=1)["figure.figsize"]
    plt.figure(figsize=figsize)
    plt.hist(num_tokens, bins=100)
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.savefig("num_tokens.png")
