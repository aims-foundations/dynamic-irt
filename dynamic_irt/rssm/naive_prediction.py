import argparse
import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tqdm import tqdm
from utils import ensure_dir, load_data, set_seed


def test(testcase_scores_prob, student_tc_scores):
    # Predict testcase scores using average accuracy probabilities
    mean_acc_testcase_scores = []
    mean_f1_testcase_scores = []
    mean_precision_testcase_scores = []
    mean_recall_testcase_scores = []
    mean_roc_auc_testcase_scores = []

    for _ in range(10):
        pred_tc_scores = []
        gt_scores = []
        for tc_scores in student_tc_scores:
            mask = tc_scores != -1

            gt_score = tc_scores[mask]
            if len(gt_score) == 0:
                continue

            # Random sample from student_tc_scores
            pred_tc_score = (
                torch.bernoulli(torch.ones_like(gt_score) * testcase_scores_prob)
                .float()
                .flatten()
            )
            pred_tc_scores.append(pred_tc_score)
            gt_scores.append(gt_score.flatten())

        pred_tc_scores = torch.concatenate(pred_tc_scores).cpu().numpy()
        gt_scores = torch.concatenate(gt_scores).cpu().numpy()

        # Compute metrics
        test_acc = accuracy_score(gt_scores, pred_tc_scores)
        test_f1 = f1_score(gt_scores, pred_tc_scores)
        test_precision = precision_score(gt_scores, pred_tc_scores)
        test_recall = recall_score(gt_scores, pred_tc_scores)
        test_roc_auc = roc_auc_score(gt_scores, pred_tc_scores)

        mean_acc_testcase_scores.append(test_acc)
        mean_f1_testcase_scores.append(test_f1)
        mean_precision_testcase_scores.append(test_precision)
        mean_recall_testcase_scores.append(test_recall)
        mean_roc_auc_testcase_scores.append(test_roc_auc)

    std_acc_testcase_scores = torch.tensor(mean_acc_testcase_scores).std().item()
    std_f1_testcase_scores = torch.tensor(mean_f1_testcase_scores).std().item()
    std_precision_testcase_scores = (
        torch.tensor(mean_precision_testcase_scores).std().item()
    )
    std_recall_testcase_scores = torch.tensor(mean_recall_testcase_scores).std().item()
    std_roc_auc_testcase_scores = (
        torch.tensor(mean_roc_auc_testcase_scores).std().item()
    )

    mean_acc_testcase_scores = torch.tensor(mean_acc_testcase_scores).mean().item()
    mean_f1_testcase_scores = torch.tensor(mean_f1_testcase_scores).mean().item()
    mean_precision_testcase_scores = (
        torch.tensor(mean_precision_testcase_scores).mean().item()
    )
    mean_recall_testcase_scores = (
        torch.tensor(mean_recall_testcase_scores).mean().item()
    )
    mean_roc_auc_testcase_scores = (
        torch.tensor(mean_roc_auc_testcase_scores).mean().item()
    )

    res = {
        "accuracy": mean_acc_testcase_scores,
        "f1": mean_f1_testcase_scores,
        "precision": mean_precision_testcase_scores,
        "recall": mean_recall_testcase_scores,
        "roc_auc": mean_roc_auc_testcase_scores,
        "std_accuracy": std_acc_testcase_scores,
        "std_f1": std_f1_testcase_scores,
        "std_precision": std_precision_testcase_scores,
        "std_recall": std_recall_testcase_scores,
        "std_roc_auc": std_roc_auc_testcase_scores,
    }
    return res


if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, default="all_cls")
    args = parser.parse_args()

    # Load data
    train_attempts = 1000
    (
        question_embs,
        testcase_embs,
        best_ans_embs,
        train_student_ans_embs,
        test_student_ans_embs,
        train_student_ans_scores,
        test_student_ans_scores,
        train_student_tc_scores,
        test_student_tc_scores,
        train_student_week_idxs,
        test_student_week_idxs,
        data_info,
    ) = load_data(args, device, train_attempts)
    n_students, total_attempts, emb_dim = data_info

    # Compute average accuracy of testcases on training data
    avg_testcase_scores = []
    unique_week_idxs = train_student_week_idxs.unique()
    for wi in unique_week_idxs:
        mask = train_student_week_idxs == wi
        ques_idxs = train_student_week_idxs[mask]
        student_question_scores = train_student_tc_scores[mask]

        for qidx in ques_idxs.unique():
            if qidx == -1:
                continue
            q_mask = ques_idxs == qidx
            testcase_scores = student_question_scores[q_mask]
            # >>> n_attempts_per_question x n_testcases_per_question

            for tidx in range(testcase_scores.shape[1]):
                each_tc_scores = testcase_scores[:, tidx]
                # Filter out -1 scores
                each_tc_scores = each_tc_scores[each_tc_scores != -1]
                if len(each_tc_scores) == 0:
                    continue
                avg_testcase_scores.append(each_tc_scores.mean().item())

    avg_testcase_scores = torch.tensor(avg_testcase_scores)
    avg_testcase_scores = avg_testcase_scores[avg_testcase_scores != -1]
    testcase_scores_prob = avg_testcase_scores.mean()
    print(
        "Average accuracy of testcases on training data:", testcase_scores_prob.item()
    )

    train_eval_res = test(testcase_scores_prob, train_student_tc_scores)
    test_eval_res = test(testcase_scores_prob, test_student_tc_scores)

    # Save evaluation results
    ensure_dir(f"eval_results/{args.cls}")
    with open(f"eval_results/{args.cls}/naive_train_res.json", "w") as f:
        json.dump(train_eval_res, f)
    with open(f"eval_results/{args.cls}/naive_test_res.json", "w") as f:
        json.dump(test_eval_res, f)
