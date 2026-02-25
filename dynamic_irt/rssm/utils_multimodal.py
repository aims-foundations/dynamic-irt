"""Data loading utilities for Multi-Modal RSSM.

Mirrors utils.py:load_data() but works with the multi-modal feature pickles
produced by process_csv_data.py instead of LLaMA embedding pickles.
"""

import pickle

import numpy as np
import torch
from tqdm import tqdm


def load_multimodal_data(data_dir, device, train_attempts, feature_config):
    """Load multi-modal features and reshape into per-student tensor format.

    Follows the same reshape/pad/split logic as utils.py lines 133-190.

    Args:
        data_dir: Path to directory with pickle files from process_csv_data.py
        device: torch device
        train_attempts: Number of attempts per student to use for training
        feature_config: FeatureConfig instance

    Returns:
        Tuple of:
            question_static: [n_questions, 3] tensor
            n_questions: int
            train_answer_features: [n_students, train_attempts, answer_dim]
            test_answer_features: [n_students, test_attempts, answer_dim]
            train_question_idxs: [n_students, train_attempts]
            test_question_idxs: [n_students, test_attempts]
            train_tc_scores: [n_students, train_attempts, n_testcases]
            test_tc_scores: [n_students, test_attempts, n_testcases]
            train_valid_mask: [n_students, train_attempts]
            test_valid_mask: [n_students, test_attempts]
            data_info: (n_students, total_attempts, answer_dim)
    """
    with open(f"{data_dir}/answer_features.pkl", "rb") as f:
        answer_features = pickle.load(f)
    with open(f"{data_dir}/question_idxs.pkl", "rb") as f:
        question_idxs = pickle.load(f)
    with open(f"{data_dir}/question_static.pkl", "rb") as f:
        question_static = pickle.load(f)
    with open(f"{data_dir}/testcase_scores.pkl", "rb") as f:
        testcase_scores = pickle.load(f)
    with open(f"{data_dir}/student_idxs.pkl", "rb") as f:
        student_idxs = pickle.load(f)
    with open(f"{data_dir}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    answer_dim = feature_config.answer_dim
    n_tc = feature_config.n_testcases
    n_questions = metadata["n_questions"]

    # Reshape into per-student lists (mirrors utils.py lines 133-153)
    per_student_features = []
    per_student_q_idxs = []
    per_student_tc_scores = []
    prev_si = -1

    for feat, qidx, tc, si in tqdm(
        zip(answer_features, question_idxs, testcase_scores, student_idxs),
        desc="Grouping by student",
        total=len(answer_features),
    ):
        if si != prev_si:
            per_student_features.append([])
            per_student_q_idxs.append([])
            per_student_tc_scores.append([])

        per_student_features[-1].append(torch.tensor(feat))
        per_student_q_idxs[-1].append(qidx)
        per_student_tc_scores[-1].append(torch.tensor(tc))
        prev_si = si

    # Find max attempts
    max_attempts = max(len(feats) for feats in per_student_features)

    # Pad to max_attempts (mirrors utils.py lines 159-178)
    padded_features = []
    padded_q_idxs = []
    padded_tc_scores = []
    padded_valid_mask = []

    for feats, qidxs, tcs in tqdm(
        zip(per_student_features, per_student_q_idxs, per_student_tc_scores),
        desc="Padding",
        total=len(per_student_features),
    ):
        n = len(feats)
        pad_n = max_attempts - n

        # Valid mask: 1 for real data, 0 for padding
        mask = [1] * n + [0] * pad_n

        if pad_n > 0:
            feats += [torch.zeros(answer_dim)] * pad_n
            qidxs += [0] * pad_n  # dummy question index
            tcs += [torch.full((n_tc,), -1.0)] * pad_n

        padded_features.append(torch.stack(feats))
        padded_q_idxs.append(qidxs)
        padded_tc_scores.append(torch.stack(tcs))
        padded_valid_mask.append(mask)

    # Stack into tensors
    all_features = torch.stack(padded_features).to(device)
    all_q_idxs = torch.tensor(padded_q_idxs, dtype=torch.long).to(device)
    all_tc_scores = torch.stack(padded_tc_scores).float().to(device)
    all_valid_mask = torch.tensor(padded_valid_mask, dtype=torch.bool).to(device)
    question_static_t = torch.tensor(question_static, dtype=torch.float32).to(device)

    n_students = all_features.shape[0]
    total_attempts = all_features.shape[1]

    print(f"n_students: {n_students}")
    print(f"total_attempts: {total_attempts}")
    print(f"answer_dim: {answer_dim}")
    print(f"n_questions: {n_questions}")
    print(f"valid_data_points: {all_valid_mask.sum().item()}")

    # Split train/test by attempts
    train_answer_features = all_features[:, :train_attempts]
    test_answer_features = all_features[:, train_attempts:]

    train_question_idxs = all_q_idxs[:, :train_attempts]
    test_question_idxs = all_q_idxs[:, train_attempts:]

    train_tc_scores = all_tc_scores[:, :train_attempts]
    test_tc_scores = all_tc_scores[:, train_attempts:]

    train_valid_mask = all_valid_mask[:, :train_attempts]
    test_valid_mask = all_valid_mask[:, train_attempts:]

    return (
        question_static_t,
        n_questions,
        train_answer_features,
        test_answer_features,
        train_question_idxs,
        test_question_idxs,
        train_tc_scores,
        test_tc_scores,
        train_valid_mask,
        test_valid_mask,
        (n_students, total_attempts, answer_dim),
    )
