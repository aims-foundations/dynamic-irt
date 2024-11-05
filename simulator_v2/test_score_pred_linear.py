import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from models import LinearScorer, MLP, RSSM
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tqdm import tqdm
from utils import ensure_dir, load_data, set_seed


def train(
    args,
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
):
    # Train RSSM with teacher-forcing
    list_params = list(scorer.parameters())
    print("Number of parameters:", sum([p.numel() for p in list_params]))
    optimizer = torch.optim.Adam(list_params, lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = torch.nn.BCELoss(reduce=False)

    train_losses = []
    test_accs = []

    for epoch in tqdm(range(args.epochs)):
        # Train by batch with batch size 32
        record_loss = None
        for i in range(0, n_students, args.batch_size):
            max_i = min(n_students, i + args.batch_size)
            optimizer.zero_grad()
            ques_embs = question_embs[train_student_week_idxs[i:max_i]]
            tcs_embs = testcase_embs[train_student_week_idxs[i:max_i]]
            bans_embs = best_ans_embs[train_student_week_idxs[i:max_i]]
            list_score_hat = scorer(
                ques_embs, tcs_embs, train_student_ans_embs[i:max_i], bans_embs
            )

            tc_masked_idx = train_student_tc_scores[i:max_i] != -1
            if tc_masked_idx.sum() == 0:
                continue
            loss = loss_fn(
                list_score_hat[tc_masked_idx],
                train_student_tc_scores[i:max_i][tc_masked_idx],
            ).mean()
            record_loss = loss.item()

            loss.backward()
            optimizer.step()
        scheduler.step()

        train_losses.append(record_loss)
        if epoch % 10 == 0:
            print("Train Loss:", record_loss)

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            test_res = test(
                question_embs,
                testcase_embs,
                best_ans_embs,
                test_student_week_idxs,
                test_student_ans_embs,
                test_student_tc_scores,
            )
            print("Test Acc:", test_res["accuracy"])
            test_accs.append(test_res["accuracy"])

    # Draw plot of train loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("score_train_losses.png")
    plt.close()

    plt.figure()
    plt.plot(
        list(range(0, args.epochs, 100)) + [args.epochs - 1],
        test_accs,
        label="Test Acc",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("score_test_acc.png")
    plt.close()

    # Save model
    ensure_dir(f"{args.ckpt_folder}/{args.cls}")
    torch.save(scorer.state_dict(), f"{args.ckpt_folder}/{args.cls}/linear_scorer.pth")


def test(
    question_embs,
    testcase_embs,
    best_ans_embs,
    student_week_idxs,
    student_ans_embs,
    student_tc_scores,
):
    pred_scores = []
    gt_scores = []

    with torch.no_grad():
        # Test by batch with batch size 32
        test_acc = []
        for i in range(0, n_students, args.batch_size):
            max_i = min(n_students, i + args.batch_size)
            ques_embs = question_embs[student_week_idxs[i:max_i]]
            tcs_embs = testcase_embs[student_week_idxs[i:max_i]]
            bans_embs = best_ans_embs[student_week_idxs[i:max_i]]
            list_score_hat = scorer(
                ques_embs, tcs_embs, student_ans_embs[i:max_i], bans_embs
            )
            # >>> n_students

            tc_masked_idx = student_tc_scores[i:max_i] != -1
            pred_tc_score = (list_score_hat[tc_masked_idx] >= 0.5).float()

            # test_acc.append((pred_tc_score == student_tc_scores[i:max_i][tc_masked_idx]).float())
            pred_scores.append(pred_tc_score.float())
            gt_scores.append(student_tc_scores[i:max_i][tc_masked_idx].float())

        pred_scores = torch.concatenate(pred_scores).cpu().numpy()
        gt_scores = torch.concatenate(gt_scores).cpu().numpy()

        test_acc = accuracy_score(gt_scores, pred_scores)
        test_f1 = f1_score(gt_scores, pred_scores)
        test_precision = precision_score(gt_scores, pred_scores)
        test_recall = recall_score(gt_scores, pred_scores)
        test_roc_auc = roc_auc_score(gt_scores, pred_scores)

    return {
        "accuracy": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "roc_auc": test_roc_auc,
    }


if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 128

    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, default="all_cls")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ckpt_folder", type=str, default=f"saves")
    parser.add_argument("--test_only", type=int, default=0)
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

    # Initialize RSSM and MLP
    scorer = LinearScorer(input_dim=emb_dim, hidden_dim=hidden_dim).to(device)

    # Load pre-trained model
    if os.path.exists(f"{args.ckpt_folder}/{args.cls}/linear_scorer.pth"):
        print("Loading pre-trained model")
        scorer.load_state_dict(
            torch.load(f"{args.ckpt_folder}/{args.cls}/linear_scorer.pth")
        )
    print("Scorer:", scorer)

    if args.test_only:
        train_eval_res = test(
            question_embs,
            testcase_embs,
            best_ans_embs,
            train_student_week_idxs,
            train_student_ans_embs,
            train_student_tc_scores,
        )
        test_eval_res = test(
            question_embs,
            testcase_embs,
            best_ans_embs,
            test_student_week_idxs,
            test_student_ans_embs,
            test_student_tc_scores,
        )

        # Save evaluation results
        ensure_dir(f"eval_results/{args.cls}")
        with open(f"eval_results/{args.cls}/linear_scorer_train_res.json", "w") as f:
            json.dump(train_eval_res, f)
        with open(f"eval_results/{args.cls}/linear_scorer_test_res.json", "w") as f:
            json.dump(test_eval_res, f)

    else:
        train(
            args,
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
        )
