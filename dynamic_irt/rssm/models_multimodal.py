"""Multi-Modal RSSM model components.

Mirrors models.py architecture but uses structured multi-modal features
instead of 4096-dim LLaMA text embeddings.
"""

import torch
import torch.nn as nn

from feature_config import FeatureConfig


class AnswerEncoder(nn.Module):
    """Encodes concatenated answer feature groups into a fixed-dim representation."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

    def forward(self, answer_features):
        return self._encoder(answer_features)


class QuestionEncoder(nn.Module):
    """Encodes question features (learnable embedding + static features)."""

    def __init__(self, n_questions, q_emb_dim=16, static_dim=3, hidden_dim=64):
        super().__init__()
        self.q_embedding = nn.Embedding(n_questions, q_emb_dim)
        total_in = q_emb_dim + static_dim
        self._encoder = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

    def forward(self, question_ids, question_static_features):
        """
        Args:
            question_ids: [batch] long tensor of question indices
            question_static_features: [batch, static_dim] float tensor
        """
        q_emb = self.q_embedding(question_ids)
        combined = torch.cat([q_emb, question_static_features], dim=-1)
        return self._encoder(combined)


class MultiModalRSSM(nn.Module):
    """GRU-based recurrent model using multi-modal features.

    Same core architecture as RSSM in models.py, but with structured
    feature encoders instead of raw embedding encoders.
    """

    def __init__(self, feature_config, n_questions, hidden_dim=128, enc_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self._ans_encoder = AnswerEncoder(feature_config.answer_dim, enc_dim)
        self._ques_encoder = QuestionEncoder(
            n_questions,
            q_emb_dim=feature_config.question_emb_dim,
            static_dim=feature_config.question_static_dim,
            hidden_dim=enc_dim,
        )
        self._cell = nn.GRUCell(enc_dim * 2, hidden_dim)

    def forward(self, prev_ans_features, prev_hidden, question_ids, question_static):
        """
        Args:
            prev_ans_features: [batch, answer_dim] features from previous timestep
            prev_hidden: [batch, hidden_dim] previous GRU hidden state
            question_ids: [batch] long tensor of question indices
            question_static: [batch, static_dim] float tensor

        Returns:
            hidden: [batch, hidden_dim] new GRU hidden state
        """
        enc_ans = self._ans_encoder(prev_ans_features)
        enc_ques = self._ques_encoder(question_ids, question_static)
        gru_input = torch.cat([enc_ans, enc_ques], dim=-1)
        return self._cell(gru_input, prev_hidden)

    def encode_question(self, question_ids, question_static):
        """Encode question features (for use by scorer)."""
        return self._ques_encoder(question_ids, question_static)


class MultiModalScorer(nn.Module):
    """Predicts per-testcase pass/fail from hidden state + question encoding.

    Replaces the original Scorer which compared answer vs best-answer embeddings.
    """

    def __init__(self, hidden_dim=128, question_enc_dim=64, n_testcases=15):
        super().__init__()
        self._scorer = nn.Sequential(
            nn.Linear(hidden_dim + question_enc_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self._predictor = nn.Sequential(
            nn.Linear(64, n_testcases),
            nn.Sigmoid(),
        )

    def forward(self, hidden_state, question_encoding):
        """
        Args:
            hidden_state: [batch, hidden_dim]
            question_encoding: [batch, question_enc_dim]

        Returns:
            [batch, n_testcases] sigmoid predictions
        """
        x = torch.cat([hidden_state, question_encoding], dim=-1)
        return self._predictor(self._scorer(x))


class AnswerFeaturePredictor(nn.Module):
    """Auxiliary decoder: predicts next timestep's answer features from hidden state.

    Provides a self-supervised regularization signal so the GRU learns to
    anticipate the student's next response characteristics.
    """

    def __init__(self, hidden_dim=128, output_dim=32):
        super().__init__()
        self._decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden_state):
        return self._decoder(hidden_state)
