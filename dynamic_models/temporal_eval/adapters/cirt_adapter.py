"""CIRT model adapter for temporal evaluation.

Trains parametric IRT model (sigmoid learning curves) on train-week items,
predicts test-week items with frozen student params and default difficulty.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..temporal_split import TemporalSplit


class CIRTAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "CIRT"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        epochs: int = 3000,
        lr: float = 0.01,
        concentration: float = 10.0,
        eps: float = 1e-2,
        **kwargs,
    ) -> PredictionResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        N = data.n_students
        T = data.n_max_attempts

        # ---- Extract train observations ----
        train_corr = data.correctness_matrix[:, split.train_item_indices, :]
        Q_train = split.n_train_items

        # Flatten and get valid indices
        y_flat = train_corr.reshape(-1)
        valid_mask = (y_flat != -1).numpy()
        valid_indices = np.where(valid_mask)[0]

        t_vals = np.linspace(1, T, T)
        student_idx_np = valid_indices // (Q_train * T)
        # question_idx is local to train items (0..Q_train-1)
        question_idx_np = (valid_indices // T) % Q_train
        t_idx_np = valid_indices % T
        t_flat_np = t_vals[t_idx_np]

        y_obs = y_flat[valid_mask].float()
        y_obs = y_obs * (1 - 2 * eps) + eps  # smooth away from 0/1

        y_obs = y_obs.to(device)
        t_flat = torch.from_numpy(t_flat_np).to(device).float()
        student_idx = torch.from_numpy(student_idx_np).to(device).long()
        question_idx = torch.from_numpy(question_idx_np).to(device).long()

        # ---- Initialize and train ----
        theta0 = nn.Parameter(torch.abs(torch.randn(N, device=device)))
        theta1 = nn.Parameter(torch.sigmoid(torch.randn(N, device=device)))
        z_train = nn.Parameter(torch.abs(torch.randn(Q_train, device=device)))

        optimizer = optim.Adam([theta0, theta1, z_train], lr=lr)

        train_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()

            mean_correct = theta1[student_idx] * torch.sigmoid(
                theta0[student_idx] * t_flat - z_train[question_idx]
            )
            mean_correct = mean_correct.clamp(1e-6, 1 - 1e-6)

            alpha = mean_correct * concentration
            beta = (1 - mean_correct) * concentration
            term1 = (
                torch.lgamma(alpha + beta)
                - torch.lgamma(alpha)
                - torch.lgamma(beta)
            )
            term2 = (alpha - 1) * torch.log(y_obs) + (beta - 1) * torch.log(
                1 - y_obs
            )
            nll = -(term1 + term2).mean()

            # Regularize theta1 to [0, 1]
            cost = (
                theta1**2 * ((theta1 < 0).float() + (theta1 > 1).float())
            ).mean()
            loss = nll + cost
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---- Extract learned parameters ----
        theta0_np = theta0.detach().cpu().numpy()
        theta1_np = torch.sigmoid(theta1).detach().cpu().numpy()
        z_np = z_train.detach().cpu().numpy()

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
        test_t_flat = t_vals[test_t_idx]

        y_true = y_test_flat[test_valid_mask].numpy()

        with torch.no_grad():
            s_idx = torch.from_numpy(test_student_idx).to(device).long()
            t_test = torch.from_numpy(test_t_flat).to(device).float()

            # Test difficulty = 0 (prior mean)
            z_test = torch.zeros(1, device=device)

            mean_pred = theta1[s_idx] * torch.sigmoid(
                theta0[s_idx] * t_test - z_test
            )
            y_pred_prob = mean_pred.clamp(1e-6, 1 - 1e-6).cpu().numpy()

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
                "theta_0 (learning rate)": theta0_np,
                "theta_1 (asymptotic ability)": theta1_np,
            },
            item_params={
                "z (difficulty)": z_np,
            },
            model_state={
                "theta0": theta0.detach().cpu(),
                "theta1": theta1.detach().cpu(),
                "z_train": z_train.detach().cpu(),
                "concentration": concentration,
                "epochs": epochs,
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 3.0
