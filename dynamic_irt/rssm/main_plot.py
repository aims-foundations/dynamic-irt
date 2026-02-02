import json
import os

import matplotlib.pyplot as plt
from tueplots import bundles, constants, cycler, figsizes
from tueplots.constants.color import palettes

plt.rcParams.update(bundles.aaai2024())
plt.rcParams.update({"figure.dpi": 300})
plt.rcParams.update(cycler.cycler(color=palettes.paultol_bright))

RESULT_FILES = {
    # "Naive Prediction": {
    #     "train": "eval_results/all_cls/naive_train_res.json",
    #     "test": "eval_results/all_cls/naive_test_res.json",
    # },
    "Scoring w/o Answers": {
        "train": "eval_results/all_cls/naive_scorer_train_res.json",
        "test": "eval_results/all_cls/naive_scorer_test_res.json",
    },
    "Scoring w/ Answers": {
        "train": "eval_results/all_cls/single_scorer_train_res.json",
        "test": "eval_results/all_cls/single_scorer_test_res.json",
    },
    "Linear Scoring w/ Answers": {
        "train": "eval_results/all_cls/linear_scorer_train_res.json",
        "test": "eval_results/all_cls/linear_scorer_test_res.json",
    },
    "Linear Scoring w/o Answers": {
        "train": "eval_results/all_cls/naive_linear_scorer_train_res.json",
        "test": "eval_results/all_cls/naive_linear_scorer_test_res.json",
    },
    "RSSM": {
        "train": "eval_results/all_cls/rssm_train_res.json",
        "test": "eval_results/all_cls/rssm_test_res.json",
    },
    "Latent RSSM": {
        "train": "eval_results/all_cls/lrssm_train_res.json",
        "test": "eval_results/all_cls/lrssm_test_res.json",
    },
    "Dynamic IRT": {
        "train": "eval_results/all_cls/dirt_train_res.json",
        "test": "eval_results/all_cls/dirt_test_res.json",
    },
}

if __name__ == "__main__":
    # Read the results
    results = {}
    for model, files in RESULT_FILES.items():
        results[model] = {}
        for split, file in files.items():
            with open(file, "r") as f:
                results[model][split] = json.load(f)

    # Plot the results with horizontal columns
    # The number of columns is the number of models
    # Two split (train, test) for each model is plotted over each other in the same column
    figsize = figsizes.aaai2024_half(nrows=1, ncols=1)["figure.figsize"]

    for metric in ["accuracy", "roc_auc"]:
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] // 2))
        for i, (model, split_results) in enumerate(results.items()):
            for split, res in split_results.items():
                if split == "train":
                    color = "#4477aa"
                else:
                    color = "#66ccee"

                if "RSSM" in model:
                    column = f"{metric}_pe"
                else:
                    column = metric
                ax.barh(i, res[column], color=color, label=f"{model} ({split})")

        ax.set_yticks(range(len(results)))
        ax.set_yticklabels(results.keys())
        # ax.set_xlabel("Accuracy")
        plt.savefig(f"eval_results/all_cls/{metric}.png")
