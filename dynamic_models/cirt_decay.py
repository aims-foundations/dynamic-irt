"""CIRT-Decay: Constant ability with exponentially decaying question difficulty.

Each student has a constant ability (theta), each item has a base difficulty (z),
and a global decay rate (lambda) reduces effective difficulty over time:

    mean_correct = sigmoid(theta[s] - z[q] * exp(-lambda * t))
    loss = BCE(mean_correct, y)

Within a question, all unit tests share the same difficulty z[q].
As the semester progresses (t increases), effective difficulty decays,
modeling improved performance through familiarity rather than ability growth.
"""

import torch
import torch.nn.functional as F


def negative_log_likelihood(y_obs, student_idx, question_idx, t_flat, theta, z, decay_rate):
    effective_difficulty = z[question_idx] * torch.exp(-decay_rate * t_flat)
    mean_correct = torch.sigmoid(theta[student_idx] - effective_difficulty)
    return F.binary_cross_entropy(mean_correct, y_obs)
