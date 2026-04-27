"""Dynamic IRT (linear growth) adapter for temporal evaluation.

Model: P(correct) = sigmoid(theta0[s] + theta_growth[s] * t - beta[q])
Trains on train-week items, predicts test-week items with beta_test = 0.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..temporal_split import TemporalSplit


class DynamicIRTAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "DynamicIRT"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        epochs: int = 3000,
        lr: float = 0.001,
        **kwargs,
    ) -> PredictionResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Work with 3D tensors directly (same as CIRT approach)
        N = data.n_students
        T = data.n_max_attempts

        # ---- Extract train observations ----
        train_corr = data.correctness_matrix[:, split.train_item_indices, :]
        Q_train = split.n_train_items

        y_flat = train_corr.reshape(-1)
        valid_mask = (y_flat != -1).numpy()
        valid_indices = np.where(valid_mask)[0]

        t_vals = np.linspace(1, T, T).astype(np.float32)
        # Normalize time to [0, 1]
        t_vals_norm = t_vals / T

        student_idx_np = valid_indices // (Q_train * T)
        question_idx_np = (valid_indices // T) % Q_train
        t_idx_np = valid_indices % T
        t_flat_np = t_vals_norm[t_idx_np]

        y_obs = y_flat[valid_mask].float().to(device)
        t_flat = torch.from_numpy(t_flat_np).to(device).float()
        student_idx = torch.from_numpy(student_idx_np).to(device).long()
        question_idx = torch.from_numpy(question_idx_np).to(device).long()

        # ---- Initialize and train ----
        theta0 = nn.Parameter(torch.zeros(N, device=device))
        theta_growth = nn.Parameter(torch.zeros(N, device=device))
        beta = nn.Parameter(torch.zeros(Q_train, device=device))

        optimizer = optim.Adam([theta0, theta_growth, beta], lr=lr)

        train_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()

            logit = (
                theta0[student_idx]
                + theta_growth[student_idx] * t_flat
                - beta[question_idx]
            )
            prob = torch.sigmoid(logit)
            prob = prob.clamp(1e-6, 1 - 1e-6)

            nll = -(
                y_obs * torch.log(prob)
                + (1 - y_obs) * torch.log(1 - prob)
            ).mean()
            nll.backward()
            optimizer.step()
            train_losses.append(nll.item())

        theta0_np = theta0.detach().cpu().numpy()
        growth_np = theta_growth.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()

        # ---- Predict on test items ----
        test_corr = data.correctness_matrix[:, split.test_item_indices, :]
        Q_test = split.n_test_items

        y_test_flat = test_corr.reshape(-1)
        test_valid_mask = (y_test_flat != -1).numpy()
        test_valid_indices = np.where(test_valid_mask)[0]

        test_student_idx = test_valid_indices // (Q_test * T)
        test_local_q_idx = (test_valid_indices // T) % Q_test
        test_item_idx = split.test_item_indices[test_local_q_idx].numpy()
        test_t_idx = test_valid_indices % T
        test_t_flat = t_vals_norm[test_t_idx]

        y_true = y_test_flat[test_valid_mask].numpy()

        with torch.no_grad():
            s_idx = torch.from_numpy(test_student_idx).to(device).long()
            t_test = torch.from_numpy(test_t_flat).to(device).float()

            # Test difficulty = 0 (prior mean)
            logit = theta0[s_idx] + theta_growth[s_idx] * t_test
            y_pred_prob = torch.sigmoid(logit).cpu().numpy()

        if len(y_true) == 0:
            raise ValueError(
                f"No test observations for cutoff_week={split.cutoff_week}"
            )

        return PredictionResult(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            student_indices=test_student_idx,
            item_indices=test_item_idx,
            losses={"train": train_losses},
            student_params={
                "theta_0 (initial ability)": theta0_np,
                "theta_growth (growth rate)": growth_np,
            },
            item_params={
                "beta (difficulty)": beta_np,
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 3.0
