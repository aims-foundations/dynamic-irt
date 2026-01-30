import json
import os
import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from tueplots import bundles, figsizes

plt.rcParams.update(bundles.iclr2024())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cls", type=str, default="CC01")
    args = parser.parse_args()

    all_json_files = os.listdir(f"results/{args.cls}")
    all_json_files = [f for f in all_json_files if f.endswith(".json")]
    all_json_files = sorted(all_json_files)

    final_json = all_json_files[-1]
    student_history = json.load(open(f"results/{args.cls}/{final_json}"))

    exam_scores = []
    exam_flag = False
    ex_ques_idx = None
    # >>> week x questions

    for hist in student_history:
        if "Here are the exam questions." in hist[0]:
            exam_scores.append({})
            exam_flag = True
            end_idx = hist[1].find(":")
            if end_idx > 2:
                start_idx = 20
            else:
                start_idx = 0
            ex_ques_idx = int(hist[1][start_idx:end_idx])

        elif hist[0].startswith("Your score") and exam_flag:
            exc = float(
                re.search(r"Your score:\s*([0-9]*\.?[0-9]+)\/[0-9]+", hist[0]).group(1)
            )
            if ex_ques_idx in exam_scores[-1]:
                if exc > exam_scores[-1][ex_ques_idx]:
                    exam_scores[-1][ex_ques_idx] = exc
            else:
                exam_scores[-1][ex_ques_idx] = exc

            if "Here are the exercise questions" in hist[0]:
                exam_flag = False
                ex_ques_idx = None
            elif hist[1] != "":
                end_idx = hist[1].find(":")
                if end_idx > 2:
                    start_idx = 20
                else:
                    start_idx = 0

                ex_ques_idx = int(hist[1][start_idx:end_idx])

    print(exam_scores)
    mean_exam_scores = [
        5,
    ]

    plt.figure(figsize=figsizes.iclr2024(nrows=1, ncols=1)["figure.figsize"])
