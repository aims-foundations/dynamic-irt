"""Basic IRT (1PL Rasch) model adapter for temporal evaluation.

Static student ability and item difficulty, no time component:

    P(correct) = sigmoid(theta[s] - b[q])

Uses binary cross-entropy loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..temporal_split import TemporalSplit


class IRTAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "IRT"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        epochs: int = 3000,
        lr: float = 0.01,
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

        y_flat = train_corr.reshape(-1)
        valid_mask = (y_flat != -1).numpy()
        valid_indices = np.where(valid_mask)[0]

        student_idx_np = valid_indices // (Q_train * T)
        question_idx_np = (valid_indices // T) % Q_train

        y_obs = y_flat[valid_mask].float()
        y_obs = y_obs * (1 - 2 * eps) + eps

        y_obs = y_obs.to(device)
        student_idx = torch.from_numpy(student_idx_np).to(device).long()
        question_idx = torch.from_numpy(question_idx_np).to(device).long()

        # ---- Initialize parameters ----
        theta = nn.Parameter(torch.randn(N, device=device))
        b_train = nn.Parameter(torch.randn(Q_train, device=device))

        optimizer = optim.Adam([theta, b_train], lr=lr)

        print(f"    [IRT] {N} students, {Q_train} train items, "
              f"{len(y_obs)} observations, device={device}", flush=True)

        train_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()

            mean_correct = torch.sigmoid(theta[student_idx] - b_train[question_idx])
            loss = F.binary_cross_entropy(mean_correct, y_obs)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"    [IRT] epoch {epoch+1}/{epochs} "
                      f"loss={loss.item():.4f}", flush=True)

        # ---- Extract learned parameters ----
        theta_np = theta.detach().cpu().numpy()
        b_np = b_train.detach().cpu().numpy()

        # ---- Predict on test items ----
        test_corr = data.correctness_matrix[:, split.test_item_indices, :]
        Q_test = split.n_test_items

        y_test_flat = test_corr.reshape(-1)
        test_valid_mask = (y_test_flat != -1).numpy()
        test_valid_indices = np.where(test_valid_mask)[0]

        test_student_idx = test_valid_indices // (Q_test * T)
        test_local_q_idx = (test_valid_indices // T) % Q_test
        test_item_idx = split.test_item_indices[test_local_q_idx].numpy()

        y_true = y_test_flat[test_valid_mask].numpy()

        with torch.no_grad():
            s_idx = torch.from_numpy(test_student_idx).to(device).long()

            # Test difficulty = 0 (prior mean)
            b_test = torch.zeros(1, device=device)
            mean_pred = torch.sigmoid(theta[s_idx] - b_test)
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
                "theta (ability)": theta_np,
            },
            item_params={
                "b (difficulty)": b_np,
            },
            model_state={
                "theta": theta.detach().cpu(),
                "b_train": b_train.detach().cpu(),
                "epochs": epochs,
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 2.0
