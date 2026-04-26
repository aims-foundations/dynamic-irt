"""Continuous IRT (CIRT) model with sigmoid learning curves.

Each student has a learning rate (theta0) and asymptotic ability (theta1),
each item has a difficulty (z). Correctness is modeled as Beta-distributed:

    mean_correct = theta1[s] * sigmoid(theta0[s] * t - z[q])
    y ~ BetaProportion(mean_correct, concentration)

Train via temporal_eval framework:
    python -m dynamic_irt.temporal_eval.run_temporal_eval --models CIRT
"""

import torch


def negative_log_likelihood(concentration, y_obs, student_idx, question_idx, t_flat, theta0, theta1, z):
    """Compute negative log-likelihood for CIRT model.

    mean_correct = theta1[s] * sigmoid(theta0[s] * t - z[q])
    y ~ BetaProportion(mean_correct, concentration)
    """
    eps = 1e-6
    mean_correct = theta1[student_idx] * torch.sigmoid(
        theta0[student_idx] * t_flat - z[question_idx]
    )
    mean_correct = mean_correct.clamp(eps, 1 - eps)

    alpha = mean_correct * concentration
    beta = (1 - mean_correct) * concentration
    term1 = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    term2 = (alpha - 1) * torch.log(y_obs) + (beta - 1) * torch.log(1 - y_obs)
    nll = -(term1 + term2).mean()

    return nll
