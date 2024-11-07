import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


def parse_time(time_str):
    # Parsing the string into a datetime object
    parsed_datetime = datetime.strptime(time_str, "%d/%m/%y, %H:%M:%S")
    return parsed_datetime


def format_template_testcase(tokenizer, template, testcases, truncate=True):
    output = f"Template: {template}\n"
    num_processed = 0
    for tid, testcase in enumerate(testcases):
        tc_text = (
            f"Testcase {tid+1}:\n"
            f"{testcase['input']}\n"
            f"std input: {testcase['std_input']}\n"
            if testcase["std_input"] != ""
            else "" f"expected result: {testcase['output']}\n"
        )
        if truncate and len(tc_text) > 32768:
            continue
        output += tc_text
        num_processed += 1

    if len(testcases) == 0:
        output += "No testcases\n"
        return output, 0.0

    return output, num_processed / len(testcases)


def get_question_info_by_name(question_name, question_name2idx, question_infos):
    return question_infos[question_name2idx[question_name]]


def parse_question_name(question_names):
    return_names = {}
    for qidx, q in enumerate(question_names):
        if "|" in q:
            names = q.split("|")
            for sub_idx, name in enumerate(names):
                return_names[f"{qidx+1}.{sub_idx+1}"] = name
        else:
            return_names[str(qidx + 1)] = q

    return return_names


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def get_dist(logit, category_size, class_size):
    shape = logit.shape
    logit = torch.reshape(logit, shape=(*shape[:-1], category_size, class_size))
    return torch.distributions.Independent(
        torch.distributions.OneHotCategoricalStraightThrough(logits=logit), 1
    )


def kl_divergence(prior, posterior, category_size, class_size, alpha=0.5):
    prior_dist = get_dist(prior, category_size, class_size)
    post_dist = get_dist(posterior, category_size, class_size)
    kl_lhs = torch.mean(
        torch.distributions.kl.kl_divergence(
            get_dist(posterior.detach(), category_size, class_size), prior_dist
        )
    )
    kl_rhs = torch.mean(
        torch.distributions.kl.kl_divergence(
            post_dist, get_dist(prior.detach(), category_size, class_size)
        )
    )
    kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
    return kl_loss


def load_data(args, device, train_attempts):
    # Load question embeddings
    question_embs = pickle.load(open(f"data/{args.cls}/questions.pkl", "rb"))
    question_embs = torch.tensor(question_embs).to(device)
    # >>> n_questions x emb_dim

    # Load testcase embeddings
    testcase_embs = pickle.load(open(f"data/{args.cls}/testcases.pkl", "rb"))
    testcase_embs = torch.tensor(testcase_embs).to(device)
    # >>> n_questions x emb_dim

    # Load bestans embeddings
    best_ans_embs = pickle.load(open(f"data/{args.cls}/best_answer_by_week.pkl", "rb"))
    best_ans_embs = torch.tensor(best_ans_embs).to(device)

    # Load answer embeddings
    answer_embs = pickle.load(open(f"data/{args.cls}/answers.pkl", "rb"))
    answer_embs = torch.tensor(answer_embs)
    # >>> (n_students * total_attempts) x emb_dim

    # Load scores
    answer_scores = pickle.load(open(f"data/{args.cls}/scores.pkl", "rb"))
    answer_scores = torch.tensor(answer_scores)
    # >>> (n_students * total_attempts) x emb_dim

    # Load testcase scores
    testcase_scores = pickle.load(open(f"data/{args.cls}/testcase_scores.pkl", "rb"))
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
    new_testcase_scores = []
    new_week_idxs = []
    prev_si = -1
    for ans_emb, ans_sco, tc_sco, si, wi in tqdm(
        zip(answer_embs, answer_scores, testcase_scores, student_idxs, week_idxs),
        desc="Spliting data by student",
    ):
        if si != prev_si:
            new_answer_embs.append([])
            new_answer_scores.append([])
            new_testcase_scores.append([])
            new_week_idxs.append([])

        new_answer_embs[-1].append(ans_emb)
        new_answer_scores[-1].append(ans_sco)
        new_testcase_scores[-1].append(torch.tensor(tc_sco))
        new_week_idxs[-1].append(wi)
        prev_si = si

    # Find maximum attempts
    max_attempts = max([len(ae) for ae in new_answer_embs])

    # Pad answer embeddings
    student_ans_embs = []
    student_ans_scores = []
    student_tc_scores = []
    student_week_idxs = []
    for ae, asc, tcs, wi in tqdm(
        zip(new_answer_embs, new_answer_scores, new_testcase_scores, new_week_idxs),
        desc="Padding data",
    ):
        if len(ae) < max_attempts:
            ae += [torch.zeros_like(ae[0]) for _ in range(max_attempts - len(ae))]
            asc += [torch.zeros_like(asc[0]) for _ in range(max_attempts - len(asc))]
            tcs += [
                torch.ones_like(tcs[0]) * -1 for _ in range(max_attempts - len(tcs))
            ]
            wi += [-1 for _ in range(max_attempts - len(wi))]

        student_ans_embs.append(torch.stack(ae))
        student_ans_scores.append(torch.stack(asc))
        student_tc_scores.append(torch.stack(tcs))
        student_week_idxs.append(wi)

    student_ans_embs = torch.stack(student_ans_embs).to(device)
    # >>> n_students x total_attempts x emb_dim

    student_ans_scores = torch.stack(student_ans_scores).to(device)
    # >>> n_students x total_attempts x emb_dim

    student_tc_scores = torch.stack(student_tc_scores).to(device)
    student_tc_scores = student_tc_scores.float()
    # >>> n_students x total_attempts x emb_dim

    student_week_idxs = torch.tensor(student_week_idxs).to(device)
    # >>> n_students x total_attempts

    n_students, total_attempts, emb_dim = student_ans_embs.shape
    print("n_students:", n_students)
    print("total_attempts:", total_attempts)
    print("emb_dim:", emb_dim)
    print("total_data_points:", (student_week_idxs != -1).sum())

    # Split student_ans_embs into two train and test sets by attempts
    train_student_ans_embs = student_ans_embs[:, :train_attempts]
    test_student_ans_embs = student_ans_embs[:, train_attempts:]

    train_student_ans_scores = student_ans_scores[:, :train_attempts]
    test_student_ans_scores = student_ans_scores[:, train_attempts:]

    train_student_tc_scores = student_tc_scores[:, :train_attempts]
    test_student_tc_scores = student_tc_scores[:, train_attempts:]

    train_student_week_idxs = student_week_idxs[:, :train_attempts]
    test_student_week_idxs = student_week_idxs[:, train_attempts:]

    return (
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
        (n_students, total_attempts, emb_dim),
    )
