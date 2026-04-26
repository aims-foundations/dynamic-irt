"""Dynamic IRT model with linear student growth.

Each student has an initial ability (theta0) and a growth rate (theta_growth).
Each item has a difficulty (beta). Correctness is modeled as:

    P(correct) = sigmoid(theta0[s] + theta_growth[s] * t - beta[q])

Train via temporal_eval framework:
    python -m dynamic_irt.temporal_eval.run_temporal_eval --models DynamicIRT
"""

import torch.nn as nn


def negative_log_likelihood(response, student_idx, item_idx, time_t,
                            theta0, theta_growth, beta):
    """Compute mean binary cross-entropy for the linear-growth Rasch model.

    logit = theta0[s] + theta_growth[s] * t - beta[q]
    """
    logit = theta0[student_idx] + theta_growth[student_idx] * time_t - beta[item_idx]
    return nn.functional.binary_cross_entropy_with_logits(logit, response)
