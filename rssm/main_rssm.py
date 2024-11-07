import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from models import MLP, RSSM, Scorer
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
    list_params = (
        list(rnn_model.parameters())
        + list(decoder.parameters())
        + list(scorer.parameters())
    )
    print("Number of parameters:", sum([p.numel() for p in list_params]))
    optimizer = torch.optim.Adam(list_params, lr=1e-3)
    loss_fn = torch.nn.MSELoss(reduce=False)
    loss_bce_fn = torch.nn.BCELoss(reduce=False)

    train_losses = []
    test_accs = []
    test_accs_rollout = []

    for epoch in tqdm(range(args.epochs)):
        # Train
        list_emb_hat = []
        list_score_hat = []
        prev_hidden_state = h0

        for aidx in range(train_attempts):
            ques_embs = question_embs[train_student_week_idxs[:, aidx]]
            tcs_embs = testcase_embs[train_student_week_idxs[:, aidx]]
            bans_embs = best_ans_embs[train_student_week_idxs[:, aidx]]
            if aidx == 0:
                prev_emb = e0
            else:
                prev_emb = train_student_ans_embs[:, aidx - 1]

            hidden_state = rnn_model(prev_emb, prev_hidden_state, ques_embs)
            # >>> n_students x hidden_dim

            emb_hat = decoder(hidden_state)
            # >>> n_students x emb_dim

            score_hat = scorer(
                ques_embs, tcs_embs, train_student_ans_embs[:, aidx], bans_embs
            )
            # >>> n_students

            list_emb_hat.append(emb_hat)
            list_score_hat.append(score_hat)
            prev_hidden_state = hidden_state

        list_emb_hat = torch.stack(list_emb_hat)
        list_score_hat = torch.stack(list_score_hat)
        # >>> total_attempts x n_students x emb_dim

        list_emb_hat = list_emb_hat.permute(1, 0, 2)
        list_score_hat = list_score_hat.permute(1, 0, 2).squeeze(-1)
        # >>> n_students x total_attempts x emb_dim

        # Compute masked indexes by week index == -1
        masked_idx = train_student_week_idxs != -1
        tc_masked_idx = train_student_tc_scores != -1

        loss = (
            loss_fn(list_emb_hat, train_student_ans_embs)[masked_idx].sum(-1).mean()
            + loss_bce_fn(
                list_score_hat[tc_masked_idx], train_student_tc_scores[tc_masked_idx]
            ).mean()
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())
        if epoch % 10 == 0:
            print("Train Loss:", loss.item())

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            _, test_res = test(
                prev_hidden_state,
                train_student_ans_embs[:, -1],
                question_embs,
                testcase_embs,
                best_ans_embs,
                test_student_week_idxs,
                test_student_ans_embs,
                test_student_tc_scores,
            )
            test_accs.append(test_res["accuracy"])
            test_accs_rollout.append(test_res["accuracy_pe"])
            print(
                "Test Acc:",
                test_res["accuracy"],
                "\tTest Acc Rollout:",
                test_res["accuracy_pe"],
            )

    # Save model
    ensure_dir(f"saves/{args.cls}")
    torch.save(rnn_model.state_dict(), f"saves/{args.cls}/rssm_rnn_model.pth")
    torch.save(decoder.state_dict(), f"saves/{args.cls}/rssm_decoder.pth")
    torch.save(scorer.state_dict(), f"saves/{args.cls}/rssm_scorer.pth")

    # Draw plot of train loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("rssm_train_loss.png")
    plt.close()

    plt.figure()
    plt.plot(
        list(range(0, args.epochs, 100)) + [args.epochs - 1],
        test_accs,
        label="Real embedding",
    )
    plt.plot(
        list(range(0, args.epochs, 100)) + [args.epochs - 1],
        test_accs_rollout,
        label="Predicted embedding",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("rssm_score_acc.png")
    plt.close()


def test(
    prev_hs_0,
    prev_emb_0,
    question_embs,
    testcase_embs,
    best_ans_embs,
    student_week_idxs,
    student_ans_embs,
    student_tc_scores,
):
    # Test
    with torch.no_grad():
        prev_hidden_state = prev_hs_0
        list_test_emb_hat = []
        list_score_hat = []
        list_score_hat_rollout = []
        for aidx in range(student_ans_embs.shape[1]):
            ques_embs = question_embs[student_week_idxs[:, aidx]]
            tcs_embs = testcase_embs[student_week_idxs[:, aidx]]
            bans_embs = best_ans_embs[student_week_idxs[:, aidx]]

            if aidx == 0:
                prev_emb = prev_emb_0
            else:
                prev_emb = student_ans_embs[:, aidx - 1]

            hidden_state = rnn_model(prev_emb, prev_hidden_state, ques_embs)
            # >>> n_students x hidden_dim

            emb_hat = decoder(hidden_state)
            # >>> n_students x emb_dim

            score_hat = scorer(
                ques_embs, tcs_embs, student_ans_embs[:, aidx], bans_embs
            )
            score_hat_rollout = scorer(ques_embs, tcs_embs, emb_hat, bans_embs)
            # >>> n_students

            list_test_emb_hat.append(emb_hat)
            list_score_hat.append(score_hat)
            list_score_hat_rollout.append(score_hat_rollout)

            prev_hidden_state = hidden_state

        list_test_emb_hat = torch.stack(list_test_emb_hat)
        list_score_hat = torch.stack(list_score_hat)
        list_score_hat_rollout = torch.stack(list_score_hat_rollout)
        # >>> total_attempts x n_students x emb_dim

        list_test_emb_hat = list_test_emb_hat.permute(1, 0, 2)
        list_score_hat = list_score_hat.permute(1, 0, 2).squeeze(-1)
        list_score_hat_rollout = list_score_hat_rollout.permute(1, 0, 2).squeeze(-1)

        # masked_idx = (student_week_idxs != -1)
        tc_masked_idx = student_tc_scores != -1
        gt_scores = student_tc_scores[tc_masked_idx].cpu().numpy()

        pred_tc_score = (list_score_hat[tc_masked_idx] >= 0.5).float().cpu().numpy()
        pred_tc_score_rollout = (
            (list_score_hat_rollout[tc_masked_idx] >= 0.5).float().cpu().numpy()
        )

        # test_acc = (pred_tc_score == student_tc_scores[tc_masked_idx]).float().mean()
        # test_acc_rollout = (pred_tc_score_rollout == student_tc_scores[tc_masked_idx]).float().mean()

        # test_accs.append(test_acc.item())
        # test_accs_rollout.append(test_acc_rollout.item())
        # print("Test Acc:", test_acc.item(),
        #         "\tTest Acc Rollout:", test_acc_rollout.item())

        test_acc = accuracy_score(gt_scores, pred_tc_score)
        test_f1 = f1_score(gt_scores, pred_tc_score)
        test_precision = precision_score(gt_scores, pred_tc_score)
        test_recall = recall_score(gt_scores, pred_tc_score)
        test_roc_auc = roc_auc_score(gt_scores, pred_tc_score)

        test_acc_pe = accuracy_score(gt_scores, pred_tc_score_rollout)
        test_f1_pe = f1_score(gt_scores, pred_tc_score_rollout)
        test_precision_pe = precision_score(gt_scores, pred_tc_score_rollout)
        test_recall_pe = recall_score(gt_scores, pred_tc_score_rollout)
        test_roc_auc_pe = roc_auc_score(gt_scores, pred_tc_score_rollout)

    return prev_hidden_state, {
        "accuracy": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "roc_auc": test_roc_auc,
        "accuracy_pe": test_acc_pe,
        "f1_pe": test_f1_pe,
        "precision_pe": test_precision_pe,
        "recall_pe": test_recall_pe,
        "roc_auc_pe": test_roc_auc_pe,
    }


if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 128

    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, default="all_cls")
    parser.add_argument("--epochs", type=int, default=10000)
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
    rnn_model = RSSM(input_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    decoder = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=emb_dim).to(
        device
    )
    scorer = Scorer(input_dim=emb_dim, hidden_dim=hidden_dim).to(device)

    if os.path.exists(f"saves/{args.cls}/rssm_rnn_model.pth"):
        print("Loading pre-trained RNN model")
        rnn_model.load_state_dict(torch.load(f"saves/{args.cls}/rssm_rnn_model.pth"))
    if os.path.exists(f"saves/{args.cls}/rssm_decoder.pth"):
        print("Loading pre-trained Decoder")
        decoder.load_state_dict(torch.load(f"saves/{args.cls}/rssm_decoder.pth"))
    if os.path.exists(f"saves/{args.cls}/rssm_scorer.pth"):
        print("Loading pre-trained Scorer")
        scorer.load_state_dict(torch.load(f"saves/{args.cls}/rssm_scorer.pth"))

    print("RNN:", rnn_model)
    print("Decoder:", decoder)
    print("Scorer:", scorer)

    h0 = torch.randn((n_students, hidden_dim)).to(device)
    e0 = torch.randn((n_students, emb_dim)).to(device)  # Student info

    if args.test_only:
        hidden_state, train_eval_res = test(
            h0,
            e0,
            question_embs,
            testcase_embs,
            best_ans_embs,
            train_student_week_idxs,
            train_student_ans_embs,
            train_student_tc_scores,
        )
        _, test_eval_res = test(
            hidden_state,
            train_student_ans_embs[:, -1],
            question_embs,
            testcase_embs,
            best_ans_embs,
            test_student_week_idxs,
            test_student_ans_embs,
            test_student_tc_scores,
        )

        # Save evaluation results
        ensure_dir(f"eval_results/{args.cls}")
        with open(f"eval_results/{args.cls}/rssm_train_res.json", "w") as f:
            json.dump(train_eval_res, f)
        with open(f"eval_results/{args.cls}/rssm_test_res.json", "w") as f:
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
