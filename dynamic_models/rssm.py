"""Recurrent State-Space Model (RSSM) for learning dynamics.

Supports two input modes:
  - features: Handcrafted multi-modal features (32-dim, from featurize.py)
  - embeddings: LLM text embeddings (4096-dim, from featurize.py)

DEPRECATED: The standalone training code below is kept for reference only.
Use the temporal_eval framework instead:
    python -m dynamic_models.temporal_eval.run_student_eval --models RSSM
"""

import argparse
import json
import os
import pickle
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from tueplots import bundles, figsizes

from dynamic_irt.featurize import CONFIGS, EmbeddingConfig, FeatureConfig

plt.rcParams.update(bundles.aaai2024())
COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44"]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class AnswerEncoder(nn.Module):
    """Encodes answer representation (features or embeddings) into fixed dim."""

    def __init__(self, input_dim, enc_dim=64, dropout=0.0):
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, enc_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, enc_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self._encoder(x)


class HandcraftedQuestionEncoder(nn.Module):
    """Learnable embedding + static features for handcrafted feature mode."""

    def __init__(self, n_questions, q_emb_dim=16, static_dim=3, enc_dim=64, dropout=0.0):
        super().__init__()
        self.q_embedding = nn.Embedding(n_questions, q_emb_dim)
        self._encoder = nn.Sequential(
            nn.Linear(q_emb_dim + static_dim, enc_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, enc_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, question_ids, question_static):
        q_emb = self.q_embedding(question_ids)
        return self._encoder(torch.cat([q_emb, question_static], dim=-1))


class EmbeddingQuestionEncoder(nn.Module):
    """Fixed LLM embedding encoder for embedding mode."""

    def __init__(self, emb_dim=4096, enc_dim=128, dropout=0.0):
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(emb_dim, enc_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, enc_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, question_embs):
        return self._encoder(question_embs)


class RSSM(nn.Module):
    """GRU-based Recurrent State-Space Model.

    Mode-agnostic: takes injected encoders and dispatches question kwargs.
    """

    def __init__(self, ans_encoder, ques_encoder, hidden_dim=128, enc_dim=64, dropout=0.0):
        super().__init__()
        self._ans_encoder = ans_encoder
        self._ques_encoder = ques_encoder
        self._cell = nn.GRUCell(enc_dim * 2, hidden_dim)
        self._dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim

    def forward(self, prev_ans, prev_hidden, **ques_kwargs):
        enc_ans = self._ans_encoder(prev_ans)
        enc_ques = self._ques_encoder(**ques_kwargs)
        hidden = self._cell(torch.cat([enc_ans, enc_ques], dim=-1), prev_hidden)
        return self._dropout(hidden)

    def encode_question(self, **ques_kwargs):
        return self._ques_encoder(**ques_kwargs)


class Scorer(nn.Module):
    """Predicts per-testcase pass/fail from hidden state + question encoding."""

    def __init__(self, hidden_dim=128, question_enc_dim=64, n_testcases=15, dropout=0.0):
        super().__init__()
        self._scorer = nn.Sequential(
            nn.Linear(hidden_dim + question_enc_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self._predictor = nn.Sequential(
            nn.Linear(64, n_testcases),
            nn.Sigmoid(),
        )

    def forward(self, hidden_state, question_encoding):
        x = torch.cat([hidden_state, question_encoding], dim=-1)
        return self._predictor(self._scorer(x))


class AuxDecoder(nn.Module):
    """Auxiliary decoder: predicts next timestep's answer representation."""

    def __init__(self, hidden_dim=128, output_dim=32):
        super().__init__()
        self._decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden_state):
        return self._decoder(hidden_state)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_feature_model(config, n_questions, hidden_dim=128, enc_dim=64, dropout=0.0):
    """Build RSSM + Scorer + AuxDecoder for handcrafted feature mode."""
    ans_enc = AnswerEncoder(config.answer_dim, enc_dim, dropout)
    ques_enc = HandcraftedQuestionEncoder(
        n_questions, config.question_emb_dim, config.question_static_dim,
        enc_dim, dropout,
    )
    rssm = RSSM(ans_enc, ques_enc, hidden_dim, enc_dim, dropout)
    scorer = Scorer(hidden_dim, enc_dim, config.n_testcases, dropout)
    aux = AuxDecoder(hidden_dim, config.answer_dim) if config.use_aux_loss else None
    return rssm, scorer, aux


def build_embedding_model(config, hidden_dim=128, enc_dim=128, dropout=0.0):
    """Build RSSM + Scorer + AuxDecoder for LLM embedding mode."""
    ans_enc = AnswerEncoder(config.emb_dim, enc_dim, dropout)
    ques_enc = EmbeddingQuestionEncoder(config.emb_dim, enc_dim, dropout)
    rssm = RSSM(ans_enc, ques_enc, hidden_dim, enc_dim, dropout)
    scorer = Scorer(hidden_dim, enc_dim, config.n_testcases, dropout)
    aux = AuxDecoder(hidden_dim, config.emb_dim) if config.use_aux_loss else None
    return rssm, scorer, aux


# ---------------------------------------------------------------------------
# Helpers

# ---------------------------------------------------------------------------
# DEPRECATED: Standalone training/visualization below.
# Use: python -m dynamic_models.temporal_eval.run_student_eval --models RSSM
# ---------------------------------------------------------------------------

# # ---------------------------------------------------------------------------

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def ensure_dir(dir_path):
#     os.makedirs(dir_path, exist_ok=True)


# # ---------------------------------------------------------------------------
# # Data loading
# # ---------------------------------------------------------------------------

# def load_feature_data(data_dir, device, train_attempts, config):
#     """Load handcrafted feature pickles (from featurize.py).

#     Returns:
#         (question_static, n_questions,
#          train_answer_features, test_answer_features,
#          train_question_idxs, test_question_idxs,
#          train_tc_scores, test_tc_scores,
#          train_valid_mask, test_valid_mask,
#          (n_students, total_attempts, answer_dim))
#     """
#     with open(f"{data_dir}/answer_features.pkl", "rb") as f:
#         answer_features = pickle.load(f)
#     with open(f"{data_dir}/question_idxs.pkl", "rb") as f:
#         question_idxs = pickle.load(f)
#     with open(f"{data_dir}/question_static.pkl", "rb") as f:
#         question_static = pickle.load(f)
#     with open(f"{data_dir}/testcase_scores.pkl", "rb") as f:
#         testcase_scores = pickle.load(f)
#     with open(f"{data_dir}/student_idxs.pkl", "rb") as f:
#         student_idxs = pickle.load(f)
#     with open(f"{data_dir}/metadata.pkl", "rb") as f:
#         metadata = pickle.load(f)

#     answer_dim = config.answer_dim
#     n_tc = config.n_testcases
#     n_questions = metadata["n_questions"]

#     # Group by student
#     per_student = {"features": [], "q_idxs": [], "tc_scores": []}
#     prev_si = -1
#     for feat, qidx, tc, si in tqdm(
#         zip(answer_features, question_idxs, testcase_scores, student_idxs),
#         desc="Grouping by student", total=len(answer_features),
#     ):
#         if si != prev_si:
#             per_student["features"].append([])
#             per_student["q_idxs"].append([])
#             per_student["tc_scores"].append([])
#         per_student["features"][-1].append(torch.tensor(feat))
#         per_student["q_idxs"][-1].append(qidx)
#         per_student["tc_scores"][-1].append(torch.tensor(tc))
#         prev_si = si

#     max_attempts = max(len(f) for f in per_student["features"])

#     # Pad to max_attempts
#     padded_features, padded_q_idxs, padded_tc_scores, padded_masks = [], [], [], []
#     for feats, qidxs, tcs in tqdm(
#         zip(per_student["features"], per_student["q_idxs"], per_student["tc_scores"]),
#         desc="Padding", total=len(per_student["features"]),
#     ):
#         n = len(feats)
#         pad_n = max_attempts - n
#         mask = [1] * n + [0] * pad_n
#         if pad_n > 0:
#             feats += [torch.zeros(answer_dim)] * pad_n
#             qidxs += [0] * pad_n
#             tcs += [torch.full((n_tc,), -1.0)] * pad_n
#         padded_features.append(torch.stack(feats))
#         padded_q_idxs.append(qidxs)
#         padded_tc_scores.append(torch.stack(tcs))
#         padded_masks.append(mask)

#     all_features = torch.stack(padded_features).to(device)
#     all_q_idxs = torch.tensor(padded_q_idxs, dtype=torch.long).to(device)
#     all_tc_scores = torch.stack(padded_tc_scores).float().to(device)
#     all_valid_mask = torch.tensor(padded_masks, dtype=torch.bool).to(device)
#     question_static_t = torch.tensor(question_static, dtype=torch.float32).to(device)

#     n_students = all_features.shape[0]
#     total_attempts = all_features.shape[1]
#     print(f"n_students={n_students}, total_attempts={total_attempts}, "
#           f"answer_dim={answer_dim}, n_questions={n_questions}, "
#           f"valid={all_valid_mask.sum().item()}")

#     return (
#         question_static_t, n_questions,
#         all_features[:, :train_attempts], all_features[:, train_attempts:],
#         all_q_idxs[:, :train_attempts], all_q_idxs[:, train_attempts:],
#         all_tc_scores[:, :train_attempts], all_tc_scores[:, train_attempts:],
#         all_valid_mask[:, :train_attempts], all_valid_mask[:, train_attempts:],
#         (n_students, total_attempts, answer_dim),
#     )


# def load_embedding_data(data_dir, device, train_attempts):
#     """Load LLM embedding pickles (from featurize.py).

#     Returns:
#         (question_embs, testcase_embs, best_ans_embs,
#          train_ans_embs, test_ans_embs,
#          train_tc_scores, test_tc_scores,
#          train_week_idxs, test_week_idxs,
#          (n_students, total_attempts, emb_dim))
#     """
#     question_embs = torch.tensor(
#         pickle.load(open(f"{data_dir}/questions.pkl", "rb"))
#     ).to(device)
#     testcase_embs = torch.tensor(
#         pickle.load(open(f"{data_dir}/testcases.pkl", "rb"))
#     ).to(device)
#     best_ans_embs = torch.tensor(
#         pickle.load(open(f"{data_dir}/best_answer_by_week.pkl", "rb"))
#     ).to(device)

#     answer_embs = torch.tensor(
#         pickle.load(open(f"{data_dir}/answers.pkl", "rb"))
#     )
#     testcase_scores = pickle.load(open(f"{data_dir}/testcase_scores.pkl", "rb"))
#     student_idxs = pickle.load(open(f"{data_dir}/student_idxs.pkl", "rb"))
#     week_idxs = pickle.load(open(f"{data_dir}/week_idxs.pkl", "rb"))

#     # Group by student
#     per_student = {"embs": [], "tc_scores": [], "week_idxs": []}
#     prev_si = -1
#     for emb, tc, si, wi in tqdm(
#         zip(answer_embs, testcase_scores, student_idxs, week_idxs),
#         desc="Grouping by student",
#     ):
#         if si != prev_si:
#             per_student["embs"].append([])
#             per_student["tc_scores"].append([])
#             per_student["week_idxs"].append([])
#         per_student["embs"][-1].append(emb)
#         per_student["tc_scores"][-1].append(torch.tensor(tc))
#         per_student["week_idxs"][-1].append(wi)
#         prev_si = si

#     max_attempts = max(len(e) for e in per_student["embs"])

#     # Pad
#     padded_embs, padded_tcs, padded_weeks = [], [], []
#     for embs, tcs, wis in tqdm(
#         zip(per_student["embs"], per_student["tc_scores"], per_student["week_idxs"]),
#         desc="Padding",
#     ):
#         pad_n = max_attempts - len(embs)
#         if pad_n > 0:
#             embs += [torch.zeros_like(embs[0])] * pad_n
#             tcs += [torch.ones_like(tcs[0]) * -1] * pad_n
#             wis += [-1] * pad_n
#         padded_embs.append(torch.stack(embs))
#         padded_tcs.append(torch.stack(tcs))
#         padded_weeks.append(wis)

#     all_embs = torch.stack(padded_embs).to(device)
#     all_tcs = torch.stack(padded_tcs).float().to(device)
#     all_weeks = torch.tensor(padded_weeks).to(device)

#     n_students, total_attempts, emb_dim = all_embs.shape
#     print(f"n_students={n_students}, total_attempts={total_attempts}, "
#           f"emb_dim={emb_dim}, valid={(all_weeks != -1).sum().item()}")

#     return (
#         question_embs, testcase_embs, best_ans_embs,
#         all_embs[:, :train_attempts], all_embs[:, train_attempts:],
#         all_tcs[:, :train_attempts], all_tcs[:, train_attempts:],
#         all_weeks[:, :train_attempts], all_weeks[:, train_attempts:],
#         (n_students, total_attempts, emb_dim),
#     )


# # ---------------------------------------------------------------------------
# # Training and evaluation
# # ---------------------------------------------------------------------------

# def _compute_metrics(gt, pred, pred_rollout):
#     """Compute accuracy, F1, precision, recall, AUC for teacher-forced and rollout."""
#     if len(gt) == 0 or len(np.unique(gt)) < 2:
#         return {k: 0.0 for k in [
#             "accuracy", "f1", "precision", "recall", "roc_auc",
#             "accuracy_pe", "f1_pe", "precision_pe", "recall_pe", "roc_auc_pe",
#         ]}
#     return {
#         "accuracy": accuracy_score(gt, pred),
#         "f1": f1_score(gt, pred),
#         "precision": precision_score(gt, pred),
#         "recall": recall_score(gt, pred),
#         "roc_auc": roc_auc_score(gt, pred),
#         "accuracy_pe": accuracy_score(gt, pred_rollout),
#         "f1_pe": f1_score(gt, pred_rollout),
#         "precision_pe": precision_score(gt, pred_rollout),
#         "recall_pe": recall_score(gt, pred_rollout),
#         "roc_auc_pe": roc_auc_score(gt, pred_rollout),
#     }


# def evaluate(rssm, scorer, aux, prev_hidden, prev_ans,
#              answer_data, tc_scores, ques_kwargs_fn):
#     """Evaluate model on a sequence. Returns (final_hidden, metrics_dict).

#     Args:
#         ques_kwargs_fn: callable(aidx) -> dict of kwargs for rssm.forward()
#     """
#     rssm.eval()
#     scorer.eval()
#     if aux is not None:
#         aux.eval()

#     with torch.no_grad():
#         list_scores, list_scores_rollout = [], []

#         for aidx in range(answer_data.shape[1]):
#             prev_feat = prev_ans if aidx == 0 else answer_data[:, aidx - 1]
#             qkw = ques_kwargs_fn(aidx)

#             hidden = rssm(prev_feat, prev_hidden, **qkw)
#             q_enc = rssm.encode_question(**qkw)

#             list_scores.append(scorer(hidden, q_enc))

#             if aux is not None:
#                 feat_hat = aux(hidden)
#                 hidden_ro = rssm(
#                     feat_hat if aidx > 0 else prev_ans,
#                     prev_hidden, **qkw,
#                 )
#                 list_scores_rollout.append(scorer(hidden_ro, q_enc))
#             else:
#                 list_scores_rollout.append(list_scores[-1])

#             prev_hidden = hidden

#         all_scores = torch.stack(list_scores).permute(1, 0, 2)
#         all_scores_ro = torch.stack(list_scores_rollout).permute(1, 0, 2)

#         tc_mask = tc_scores != -1
#         gt = tc_scores[tc_mask].cpu().numpy()
#         pred = (all_scores[tc_mask] >= 0.5).float().cpu().numpy()
#         pred_ro = (all_scores_ro[tc_mask] >= 0.5).float().cpu().numpy()

#     return prev_hidden, _compute_metrics(gt, pred, pred_ro)


# def train_feature_mode(args, config, rssm, scorer, aux,
#                        question_static, n_questions,
#                        train_feats, test_feats,
#                        train_qidxs, test_qidxs,
#                        train_tcs, test_tcs,
#                        train_mask, test_mask,
#                        h0, a0):
#     """Training loop for handcrafted feature mode."""

#     def make_ques_kwargs(qidxs, aidx):
#         q_ids = qidxs[:, aidx]
#         return {"question_ids": q_ids, "question_static": question_static[q_ids]}

#     return _train_loop(
#         args, config, rssm, scorer, aux,
#         train_feats, test_feats, train_tcs, test_tcs,
#         train_mask, test_mask, h0, a0,
#         lambda aidx: make_ques_kwargs(train_qidxs, aidx),
#         lambda aidx: make_ques_kwargs(test_qidxs, aidx),
#         question_static=question_static, n_questions=n_questions,
#     )


# def train_embedding_mode(args, config, rssm, scorer, aux,
#                          question_embs, testcase_embs, best_ans_embs,
#                          train_embs, test_embs,
#                          train_tcs, test_tcs,
#                          train_weeks, test_weeks,
#                          h0, e0):
#     """Training loop for LLM embedding mode."""

#     def make_ques_kwargs_train(aidx):
#         return {"question_embs": question_embs[train_weeks[:, aidx]]}

#     def make_ques_kwargs_test(aidx):
#         return {"question_embs": question_embs[test_weeks[:, aidx]]}

#     train_mask = (train_weeks != -1)
#     test_mask = (test_weeks != -1)

#     return _train_loop(
#         args, config, rssm, scorer, aux,
#         train_embs, test_embs, train_tcs, test_tcs,
#         train_mask, test_mask, h0, e0,
#         make_ques_kwargs_train, make_ques_kwargs_test,
#     )


# def _train_loop(args, config, rssm, scorer, aux,
#                 train_ans, test_ans, train_tcs, test_tcs,
#                 train_mask, test_mask, h0, a0,
#                 train_ques_kwargs_fn, test_ques_kwargs_fn,
#                 question_static=None, n_questions=None):
#     """Unified training loop for both modes."""
#     params = list(rssm.parameters()) + list(scorer.parameters())
#     if config.use_aux_loss and aux is not None:
#         params += list(aux.parameters())

#     print(f"Parameters: {sum(p.numel() for p in params)}")
#     optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="max", factor=0.5, patience=200, min_lr=1e-6,
#     )

#     train_attempts = train_ans.shape[1]
#     train_losses = []
#     test_accs = []
#     test_accs_rollout = []

#     best_test_acc = 0.0
#     best_epoch = 0
#     best_state = None

#     pbar = tqdm(range(args.epochs))
#     for epoch in pbar:
#         rssm.train()
#         scorer.train()
#         if aux is not None:
#             aux.train()

#         list_score_hat = []
#         list_feat_hat = []
#         prev_hidden = h0

#         for aidx in range(train_attempts):
#             qkw = train_ques_kwargs_fn(aidx)
#             prev_feat = a0 if aidx == 0 else train_ans[:, aidx - 1]

#             hidden = rssm(prev_feat, prev_hidden, **qkw)
#             q_enc = rssm.encode_question(**qkw)
#             list_score_hat.append(scorer(hidden, q_enc))

#             if config.use_aux_loss and aux is not None:
#                 list_feat_hat.append(aux(hidden))

#             prev_hidden = hidden

#         list_score_hat = torch.stack(list_score_hat).permute(1, 0, 2)

#         tc_mask = train_tcs != -1
#         bce_loss = F.binary_cross_entropy(
#             list_score_hat[tc_mask], train_tcs[tc_mask]
#         )
#         total_loss = bce_loss

#         if config.use_aux_loss and aux is not None and list_feat_hat:
#             list_feat_hat = torch.stack(list_feat_hat).permute(1, 0, 2)
#             valid = train_mask.unsqueeze(-1).expand_as(list_feat_hat)
#             aux_loss = F.mse_loss(list_feat_hat[valid], train_ans[valid])
#             total_loss = bce_loss + args.aux_loss_weight * aux_loss

#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
#         optimizer.step()
#         optimizer.zero_grad()

#         train_losses.append(total_loss.item())
#         if epoch % 10 == 0:
#             pbar.set_postfix(loss=f"{total_loss.item():.4f}")

#         if epoch % 100 == 0 or epoch == args.epochs - 1:
#             _, test_res = evaluate(
#                 rssm, scorer, aux, prev_hidden,
#                 train_ans[:, -1], test_ans, test_tcs,
#                 test_ques_kwargs_fn,
#             )
#             test_accs.append(test_res["accuracy"])
#             test_accs_rollout.append(test_res["accuracy_pe"])

#             scheduler.step(test_res["accuracy"])

#             if test_res["accuracy"] > best_test_acc:
#                 best_test_acc = test_res["accuracy"]
#                 best_epoch = epoch
#                 best_state = {
#                     "rssm": {k: v.cpu().clone() for k, v in rssm.state_dict().items()},
#                     "scorer": {k: v.cpu().clone() for k, v in scorer.state_dict().items()},
#                     "aux": {k: v.cpu().clone() for k, v in aux.state_dict().items()}
#                     if aux is not None else None,
#                 }

#             current_lr = optimizer.param_groups[0]["lr"]
#             tqdm.write(
#                 f"[{epoch}] Test Acc: {test_res['accuracy']:.4f}"
#                 f"  Rollout: {test_res['accuracy_pe']:.4f}"
#                 f"  Best: {best_test_acc:.4f}@{best_epoch}"
#                 f"  LR: {current_lr:.2e}"
#             )

#             if epoch - best_epoch >= args.patience:
#                 tqdm.write(f"\nEarly stopping at epoch {epoch} "
#                            f"(best={best_test_acc:.4f} @ {best_epoch})")
#                 break

#     # Restore best model
#     if best_state is not None:
#         print(f"\nRestoring best model from epoch {best_epoch}")
#         rssm.load_state_dict(best_state["rssm"])
#         scorer.load_state_dict(best_state["scorer"])
#         if aux is not None and best_state["aux"] is not None:
#             aux.load_state_dict(best_state["aux"])

#     # Final evaluation
#     hidden_state, train_res = evaluate(
#         rssm, scorer, aux, h0, a0,
#         train_ans, train_tcs, train_ques_kwargs_fn,
#     )
#     _, test_res = evaluate(
#         rssm, scorer, aux, hidden_state,
#         train_ans[:, -1], test_ans, test_tcs, test_ques_kwargs_fn,
#     )

#     # Save results
#     result_dir = os.path.join(REPO_ROOT, "results", "rssm")
#     ensure_dir(result_dir)

#     with open(os.path.join(result_dir, "fit_metrics.json"), "w") as f:
#         json.dump({"train": train_res, "test": test_res}, f, indent=2)
#     with open(os.path.join(result_dir, "losses.json"), "w") as f:
#         json.dump({"train": train_losses}, f)

#     # Plots
#     print("\nGenerating plots...")
#     plot_losses(train_losses, result_dir)
#     actual_epochs = len(train_losses)
#     eval_epochs = [e for e in range(0, actual_epochs, 100)]
#     if actual_epochs - 1 not in eval_epochs:
#         eval_epochs.append(actual_epochs - 1)
#     eval_epochs = eval_epochs[:len(test_accs)]
#     plot_test_metrics(eval_epochs, test_accs, test_accs_rollout, result_dir)
#     plot_final_metrics_bar(train_res, test_res, result_dir)

#     if question_static is not None:
#         plot_question_difficulty(question_static, result_dir)
#     if n_questions is not None and hasattr(rssm._ques_encoder, "q_embedding"):
#         plot_question_embeddings(rssm, n_questions, result_dir)

#     print(f"\nFinal Train: {json.dumps(train_res, indent=2)}")
#     print(f"\nFinal Test:  {json.dumps(test_res, indent=2)}")


# # ---------------------------------------------------------------------------
# # Plotting
# # ---------------------------------------------------------------------------

# def plot_losses(train_losses, result_dir):
#     fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
#     ax.plot(train_losses, label="Train", alpha=0.7, color=COLORS[0])
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Loss")
#     ax.set_title("RSSM Training")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     fig.savefig(os.path.join(result_dir, "losses.png"), dpi=300, bbox_inches="tight")
#     plt.close(fig)


# def plot_test_metrics(eval_epochs, test_accs, test_accs_rollout, result_dir):
#     fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
#     ax.plot(eval_epochs, test_accs, label="Teacher forcing", alpha=0.7,
#             color=COLORS[0], marker="o", markersize=3)
#     ax.plot(eval_epochs, test_accs_rollout, label="Rollout",
#             alpha=0.7, color=COLORS[1], marker="s", markersize=3)
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Accuracy")
#     ax.set_title("RSSM Test Accuracy")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     fig.savefig(os.path.join(result_dir, "test_accuracy.png"), dpi=300, bbox_inches="tight")
#     plt.close(fig)


# def plot_question_difficulty(question_static, result_dir):
#     difficulties = question_static[:, 0].cpu().numpy()
#     lo, hi = np.percentile(difficulties, (1, 99))
#     clipped = difficulties[(difficulties >= lo) & (difficulties <= hi)]
#     fig, ax = plt.subplots(figsize=figsizes.aaai2024_half()["figure.figsize"])
#     ax.hist(clipped, bins=30, density=True, alpha=0.3, color=COLORS[0])
#     sns.kdeplot(clipped, color=COLORS[0], linewidth=1.5, bw_adjust=0.5, ax=ax)
#     ax.set_xlabel("Empirical Difficulty")
#     ax.set_ylabel("Density")
#     ax.set_title("Question Difficulty Distribution")
#     fig.savefig(os.path.join(result_dir, "question_difficulty.png"), dpi=300, bbox_inches="tight")
#     plt.close(fig)


# def plot_question_embeddings(rssm, n_questions, result_dir):
#     with torch.no_grad():
#         emb_weights = rssm._ques_encoder.q_embedding.weight.cpu().numpy()
#     norms = np.linalg.norm(emb_weights, axis=1)
#     fig, ax = plt.subplots(figsize=figsizes.aaai2024_half()["figure.figsize"])
#     ax.hist(norms, bins=30, density=True, alpha=0.3, color=COLORS[2])
#     sns.kdeplot(norms, color=COLORS[2], linewidth=1.5, bw_adjust=0.5, ax=ax)
#     ax.set_xlabel("Embedding Norm")
#     ax.set_ylabel("Density")
#     ax.set_title("Learned Question Embedding Norms")
#     fig.savefig(os.path.join(result_dir, "question_embedding_norms.png"), dpi=300, bbox_inches="tight")
#     plt.close(fig)


# def plot_final_metrics_bar(train_res, test_res, result_dir):
#     metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
#     labels = ["Accuracy", "F1", "Precision", "Recall", "AUC"]
#     x = np.arange(len(metrics))
#     width = 0.35
#     fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
#     train_vals = [train_res.get(m, 0) for m in metrics]
#     test_vals = [test_res.get(m, 0) for m in metrics]
#     ax.bar(x - width / 2, train_vals, width, label="Train", color=COLORS[0], alpha=0.8)
#     ax.bar(x + width / 2, test_vals, width, label="Test", color=COLORS[1], alpha=0.8)
#     for i, (tv, ttv) in enumerate(zip(train_vals, test_vals)):
#         ax.text(i - width / 2, tv + 0.01, f"{tv:.3f}", ha="center", fontsize=6)
#         ax.text(i + width / 2, ttv + 0.01, f"{ttv:.3f}", ha="center", fontsize=6)
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.set_ylabel("Score")
#     ax.set_title("RSSM Train vs Test Metrics")
#     ax.set_ylim(0, 1.05)
#     ax.legend()
#     ax.grid(True, alpha=0.3, axis="y")
#     fig.savefig(os.path.join(result_dir, "fit_metrics_bar.png"), dpi=300, bbox_inches="tight")
#     plt.close(fig)


# # ---------------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="RSSM for learning dynamics")
#     parser.add_argument("--mode", type=str, default="features",
#                         choices=["features", "embeddings"])
#     parser.add_argument("--cls", type=str, default="dsa_hk231")
#     parser.add_argument("--config", type=str, default="full",
#                         choices=list(CONFIGS.keys()),
#                         help="Feature config (features mode only)")
#     parser.add_argument("--epochs", type=int, default=10000)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--hidden_dim", type=int, default=128)
#     parser.add_argument("--enc_dim", type=int, default=64)
#     parser.add_argument("--train_attempts", type=int, default=1000)
#     parser.add_argument("--aux_loss_weight", type=float, default=0.1)
#     parser.add_argument("--weight_decay", type=float, default=1e-4)
#     parser.add_argument("--dropout", type=float, default=0.2)
#     parser.add_argument("--patience", type=int, default=500)
#     parser.add_argument("--grad_clip", type=float, default=1.0)
#     args = parser.parse_args()

#     set_seed(42)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     if args.mode == "features":
#         config = CONFIGS[args.config]
#         data_dir = os.path.join(
#             os.path.dirname(__file__), "rssm", "data", "multimodal", args.cls
#         )
#         print(f"Mode: features, Config: {args.config} (answer_dim={config.answer_dim})")

#         (question_static, n_questions,
#          train_feats, test_feats,
#          train_qidxs, test_qidxs,
#          train_tcs, test_tcs,
#          train_mask, test_mask,
#          data_info) = load_feature_data(data_dir, device, args.train_attempts, config)

#         n_students, _, answer_dim = data_info
#         rssm, scorer, aux = build_feature_model(
#             config, n_questions, args.hidden_dim, args.enc_dim, args.dropout,
#         )
#         rssm, scorer = rssm.to(device), scorer.to(device)
#         if aux is not None:
#             aux = aux.to(device)

#         h0 = torch.zeros(n_students, args.hidden_dim, device=device)
#         a0 = torch.zeros(n_students, answer_dim, device=device)

#         train_feature_mode(
#             args, config, rssm, scorer, aux,
#             question_static, n_questions,
#             train_feats, test_feats,
#             train_qidxs, test_qidxs,
#             train_tcs, test_tcs,
#             train_mask, test_mask,
#             h0, a0,
#         )

#     else:  # embeddings
#         config = EmbeddingConfig()
#         data_dir = os.path.join(
#             os.path.dirname(__file__), "rssm", "data", args.cls
#         )
#         print(f"Mode: embeddings (emb_dim={config.emb_dim})")

#         (question_embs, testcase_embs, best_ans_embs,
#          train_embs, test_embs,
#          train_tcs, test_tcs,
#          train_weeks, test_weeks,
#          data_info) = load_embedding_data(data_dir, device, args.train_attempts)

#         n_students, _, emb_dim = data_info
#         # Embedding mode uses larger enc_dim to match emb_dim
#         enc_dim = args.enc_dim if args.enc_dim != 64 else 128
#         rssm, scorer, aux = build_embedding_model(
#             config, args.hidden_dim, enc_dim, args.dropout,
#         )
#         rssm, scorer = rssm.to(device), scorer.to(device)
#         if aux is not None:
#             aux = aux.to(device)

#         h0 = torch.randn(n_students, args.hidden_dim, device=device)
#         e0 = torch.randn(n_students, emb_dim, device=device)

#         train_embedding_mode(
#             args, config, rssm, scorer, aux,
#             question_embs, testcase_embs, best_ans_embs,
#             train_embs, test_embs,
#             train_tcs, test_tcs,
#             train_weeks, test_weeks,
#             h0, e0,
#         )
