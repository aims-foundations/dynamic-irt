"""Recurrent State-Space Model (RSSM) for learning dynamics.

GRU-based recurrent model that encodes student answer representations
and question features to predict per-testcase pass/fail outcomes.

Supports two input modes:
  - features: Handcrafted multi-modal features (from featurize.py)
  - embeddings: LLM text embeddings (from featurize.py)

Train via temporal_eval framework:
    python -m dynamic_irt.temporal_eval.run_temporal_eval --models RSSM
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_models.featurize import CONFIGS, EmbeddingConfig, FeatureConfig


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


class PosteriorNet(nn.Module):
    """q(z_t | h_t, e_t): infers discrete latent from hidden state + answer encoding."""

    def __init__(self, hidden_dim, enc_dim, n_vars=16, n_classes=16):
        super().__init__()
        self.n_vars = n_vars
        self.n_classes = n_classes
        self._mlp = nn.Sequential(
            nn.Linear(hidden_dim + enc_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_vars * n_classes),
        )

    def forward(self, h, e):
        x = torch.cat([h, e], dim=-1)
        logits = self._mlp(x).reshape(*h.shape[:-1], self.n_vars, self.n_classes)
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(probs.argmax(dim=-1), self.n_classes).float()
        z = one_hot + probs - probs.detach()
        return z, probs


class PriorNet(nn.Module):
    """p(z_t | h_t): predicts discrete latent from hidden state."""

    def __init__(self, input_dim, n_vars=16, n_classes=16, hidden_dim=128):
        super().__init__()
        self.n_vars = n_vars
        self.n_classes = n_classes
        self._mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_vars * n_classes),
        )

    def forward(self, x):
        logits = self._mlp(x).reshape(*x.shape[:-1], self.n_vars, self.n_classes)
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(probs.argmax(dim=-1), self.n_classes).float()
        z = one_hot + probs - probs.detach()
        return z, probs


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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
