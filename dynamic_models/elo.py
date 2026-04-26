"""Elo-based IRT model for sequential ability/difficulty estimation.

Standard Elo update: after each interaction the student ability (theta) and
item difficulty (b) are updated based on the residual (response - predicted):

    theta += K * (response - p)
    b     -= K * (response - p)

where p = sigmoid(theta - b).

Train via temporal_eval framework:
    python -m dynamic_irt.temporal_eval.run_temporal_eval --models Elo
"""

import numpy as np


def basic_update(th, b, day, resp, K=0.4):
    """Standard Elo update for ability and difficulty."""
    p = 1 / (1 + np.exp(-(th - b)))
    return th + K * (resp - p), b - K * (resp - p)
