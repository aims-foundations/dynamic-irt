import argparse
import pickle

import matplotlib.pyplot as plt
import torch
from models import MLP, RSSM, Scorer

from tqdm import tqdm
from utils import set_seed

if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_dim = 4096
    hidden_dim = 128
    epochs = 5000

    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, default="CC01")
    args = parser.parse_args()

    # Load question embeddings
    question_embs = pickle.load(open(f"data/{args.cls}/questions.pkl", "rb"))
    question_embs = torch.tensor(question_embs).to(device)
    # >>> n_questions x emb_dim

    # Load testcase embeddings
    testcase_embs = pickle.load(open(f"data/{args.cls}/testcases.pkl", "rb"))
    testcase_embs = torch.tensor(testcase_embs).to(device)
    # >>> n_questions x emb_dim

    # Load answer embeddings
    answer_embs = pickle.load(open(f"data/{args.cls}/answers.pkl", "rb"))
    answer_embs = torch.tensor(answer_embs)
    # >>> (n_students * total_attempts) x emb_dim

    # Load scores
    answer_scores = pickle.load(open(f"data/{args.cls}/scores.pkl", "rb"))
    answer_scores = torch.tensor(answer_scores)
    # >>> (n_students * total_attempts) x emb_dim

    # Load student indexes
    student_idxs = pickle.load(open(f"data/{args.cls}/student_idxs.pkl", "rb"))
    # >>> (n_students * total_attempts)

    # Load week indexes
    week_idxs = pickle.load(open(f"data/{args.cls}/week_idxs.pkl", "rb"))
    # >>> (n_students * total_attempts)

    # Reshape answer embeddings
    new_answer_embs = []
    new_answer_scores = []
    new_week_idxs = []
    prev_si = -1
    for ans_emb, ans_sco, si, wi in zip(
        answer_embs, answer_scores, student_idxs, week_idxs
    ):
        if si != prev_si:
            new_answer_embs.append([])
            new_answer_scores.append([])
            new_week_idxs.append([])

        new_answer_embs[-1].append(ans_emb)
        new_answer_scores[-1].append(ans_sco)
        new_week_idxs[-1].append(wi)
        prev_si = si

    # Find maximum attempts
    max_attempts = max([len(ae) for ae in new_answer_embs])

    # Pad answer embeddings
    student_ans_embs = []
    student_ans_scores = []
    student_week_idxs = []
    for ae, asc, wi in zip(new_answer_embs, new_answer_scores, new_week_idxs):
        if len(ae) < max_attempts:
            ae += [torch.zeros_like(ae[0]) for _ in range(max_attempts - len(ae))]
            asc += [torch.zeros_like(asc[0]) for _ in range(max_attempts - len(asc))]
            wi += [0 for _ in range(max_attempts - len(wi))]

        student_ans_embs.append(torch.stack(ae))
        student_ans_scores.append(torch.stack(asc))
        student_week_idxs.append(wi)

    student_ans_embs = torch.stack(student_ans_embs).to(device)
    # >>> n_students x total_attempts x emb_dim

    student_ans_scores = torch.stack(student_ans_scores).to(device)
    student_week_idxs = torch.tensor(student_week_idxs).to(device) - 1
    # >>> n_students x total_attempts

    n_students, total_attempts, emb_dim = student_ans_embs.shape
    print("n_students:", n_students)
    print("total_attempts:", total_attempts)
    print("emb_dim:", emb_dim)
    print("total_data_points:", (student_week_idxs != -1).sum())

    # Split student_ans_embs into two train and test sets by attempts
    train_attempts = 1000
    train_student_ans_embs = student_ans_embs[:, :train_attempts]
    test_student_ans_embs = student_ans_embs[:, train_attempts:]

    train_student_ans_scores = student_ans_scores[:, :train_attempts]
    test_student_ans_scores = student_ans_scores[:, train_attempts:]

    train_student_week_idxs = student_week_idxs[:, :train_attempts]
    test_student_week_idxs = student_week_idxs[:, train_attempts:]

    # Initialize RSSM and MLP
    scorer = Scorer(input_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    print("Scorer:", scorer)

    # Train RSSM with teacher-forcing
    list_params = list(scorer.parameters())
    print("Number of parameters:", sum([p.numel() for p in list_params]))
    optimizer = torch.optim.Adam(list_params, lr=1e-3)
    loss_fn = torch.nn.MSELoss(reduce=False)
    loss_mae_fn = torch.nn.L1Loss(reduce=False)

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(epochs)):
        # Train
        ques_embs = question_embs[train_student_week_idxs]
        tcs_embs = testcase_embs[train_student_week_idxs]
        list_score_hat = scorer(ques_embs, tcs_embs, train_student_ans_embs)
        list_score_hat = list_score_hat.squeeze(-1)

        # Compute masked indexes by week index == -1
        masked_idx = train_student_week_idxs != -1
        loss = loss_mae_fn(list_score_hat, train_student_ans_scores)[masked_idx].mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())
        if epoch % 10 == 0:
            print("Train Loss:", loss.item())

        if epoch % 100 == 0 or epoch == epochs - 1:
            # Test
            with torch.no_grad():
                ques_embs = question_embs[test_student_week_idxs]
                tcs_embs = testcase_embs[test_student_week_idxs]
                list_score_hat = scorer(ques_embs, tcs_embs, test_student_ans_embs)
                list_score_hat = list_score_hat.squeeze(-1)
                # >>> n_students

                masked_idx = test_student_week_idxs != -1
                test_loss = loss_mae_fn(list_score_hat, test_student_ans_scores)[
                    masked_idx
                ].mean()
                test_losses.append(test_loss.item())
                print("Test Loss:", test_loss.item())

    # Draw plot of train loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(list(range(0, epochs, 100)) + [epochs - 1], test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("score_pred_losses_ba.png")
    plt.close()
