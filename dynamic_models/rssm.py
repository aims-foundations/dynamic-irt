"""Model components for the Recurrent State-Space Model (RSSM).

Provides the building blocks consumed by
dynamic_models/temporal_eval/adapters/rssm_adapter.py:

  - AnswerEncoder: encodes answer embeddings into a fixed dim.
  - EmbeddingQuestionEncoder: encodes LLM question embeddings.
  - Scorer: predicts per-testcase pass/fail from question encoding,
    hidden state, and discrete latent z.
  - _discrete_latent: unimix-smoothed argmax one-hot latent with
    straight-through gradients.
  - PosteriorNet: q(z_t | h_t, e_t).
  - PriorNet: p(z_t | h_t).

Training lives in the temporal_eval framework:
    python -m dynamic_models.temporal_eval.run_student_eval --models RSSM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnswerEncoder(nn.Module):
    """Encodes answer representation (features or embeddings) into fixed dim."""

    def __init__(self, input_dim, enc_dim=64, dropout=0.0):
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, enc_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self._encoder(x)


class EmbeddingQuestionEncoder(nn.Module):
    """Fixed LLM embedding encoder for embedding mode."""

    def __init__(self, emb_dim=4096, enc_dim=128, dropout=0.0):
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(emb_dim, enc_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, question_embs):
        return self._encoder(question_embs)


class Scorer(nn.Module):
    """Predicts per-testcase pass/fail from question encoding + hidden state + latent z.

    DreamerV2 pattern: reward predictor uses p(r_t | q_t, h_t, z_t).
    During training z comes from posterior; during inference z comes from prior.
    """

    def __init__(self, question_enc_dim=64, hidden_dim=128, latent_dim=0,
                 n_testcases=15, dropout=0.0):
        super().__init__()
        input_dim = question_enc_dim + hidden_dim + latent_dim
        self._scorer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self._logit_head = nn.Linear(64, n_testcases)

    def forward_logits(self, *inputs):
        x = torch.cat(inputs, dim=-1)
        return self._logit_head(self._scorer(x))

    def forward(self, *inputs):
        return torch.sigmoid(self.forward_logits(*inputs))


def _discrete_latent(logits, n_latent_vars, n_latent_classes, unimix=0.01):
    """Discrete latent: unimix-smoothed probs, deterministic argmax one-hot
    with straight-through gradients.

    Argmax is load-bearing: it makes z depend sharply on the answer encoding
    from initialization, which is the only path answer information takes into
    the recurrent state. Alternatives were tried and rejected: sampled
    straight-through one-hots inject enough per-step noise over long BPTT
    sequences that the posterior collapses onto the prior, and soft mixture
    latents are near-constant at initialization so the model settles into
    difficulty-only predictions regardless of KL warmup. The cost of argmax
    is rare discontinuous basin flips in the train loss; those are handled by
    validation-based checkpointing and a non-vanishing lr tail.
    """
    logits = logits.reshape(*logits.shape[:-1], n_latent_vars, n_latent_classes)
    probs = F.softmax(logits, dim=-1)
    probs = (1.0 - unimix) * probs + unimix / n_latent_classes
    hard = F.one_hot(probs.argmax(dim=-1), n_latent_classes).float()
    z_sample = hard - probs.detach() + probs
    z_flat = z_sample.reshape(*z_sample.shape[:-2], n_latent_vars * n_latent_classes)
    return z_flat, probs


class PosteriorNet(nn.Module):
    """Posterior q(z_t | h_t, e_t): infers discrete latent from hidden state and answer encoding."""

    def __init__(self, hidden_dim, enc_dim, n_latent_vars, n_latent_classes):
        super().__init__()
        self.n_latent_vars = n_latent_vars
        self.n_latent_classes = n_latent_classes
        self._net = nn.Sequential(
            nn.Linear(hidden_dim + enc_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_latent_vars * n_latent_classes),
        )

    def forward(self, h_t, e_t):
        logits = self._net(torch.cat([h_t, e_t], dim=-1))
        return _discrete_latent(logits, self.n_latent_vars,
                                self.n_latent_classes)


class PriorNet(nn.Module):
    """Prior p(z_t | h_t): predicts discrete latent from hidden state alone."""

    def __init__(self, hidden_dim, n_latent_vars, n_latent_classes):
        super().__init__()
        self.n_latent_vars = n_latent_vars
        self.n_latent_classes = n_latent_classes
        self._net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_latent_vars * n_latent_classes),
        )

    def forward(self, h_t):
        logits = self._net(h_t)
        return _discrete_latent(logits, self.n_latent_vars,
                                self.n_latent_classes)
