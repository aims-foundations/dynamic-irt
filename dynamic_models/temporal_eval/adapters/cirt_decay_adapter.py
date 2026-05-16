"""CIRT-Decay model adapter for temporal evaluation.

Constant student ability, constant base question difficulty, with exponentially
decaying effective difficulty over time:

    mean_correct = sigmoid(theta[s] - z[q] * exp(-lambda * t))

Within a question, all unit tests share the same difficulty. As the semester
progresses, effective difficulty decays via a learned global decay rate.
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


class CIRTDecayAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "CIRT-Decay"

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

        t_vals = np.linspace(1, T, T, dtype=np.float32)
        student_idx_np = valid_indices // (Q_train * T)
        question_idx_np = (valid_indices // T) % Q_train
        t_idx_np = valid_indices % T
        t_flat_np = t_vals[t_idx_np]

        y_obs = y_flat[valid_mask].float()
        y_obs = y_obs * (1 - 2 * eps) + eps

        y_obs = y_obs.to(device)
        t_flat = torch.from_numpy(t_flat_np).to(device).float()
        student_idx = torch.from_numpy(student_idx_np).to(device).long()
        question_idx = torch.from_numpy(question_idx_np).to(device).long()

        # ---- Initialize parameters ----
        theta = nn.Parameter(torch.randn(N, device=device))
        z_train = nn.Parameter(torch.abs(torch.randn(Q_train, device=device)))
        log_decay = nn.Parameter(torch.tensor(-3.0, device=device))

        optimizer = optim.Adam([theta, z_train, log_decay], lr=lr)

        print(f"    [CIRT-Decay] {N} students, {Q_train} train items, "
              f"{len(y_obs)} observations, device={device}", flush=True)

        train_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()

            decay_rate = torch.exp(log_decay)
            effective_difficulty = z_train[question_idx] * torch.exp(-decay_rate * t_flat)
            mean_correct = torch.sigmoid(theta[student_idx] - effective_difficulty)

            loss = F.binary_cross_entropy(mean_correct, y_obs)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"    [CIRT-Decay] epoch {epoch+1}/{epochs} "
                      f"loss={loss.item():.4f} decay_rate={decay_rate.item():.6f}",
                      flush=True)

        # ---- Extract learned parameters ----
        theta_np = theta.detach().cpu().numpy()
        z_np = z_train.detach().cpu().numpy()
        decay_rate_val = torch.exp(log_decay).detach().cpu().item()

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

            z_test = torch.zeros(1, device=device)
            decay_rate = torch.exp(log_decay)
            effective_difficulty = z_test * torch.exp(-decay_rate * t_test)
            mean_pred = torch.sigmoid(theta[s_idx] - effective_difficulty)
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
                "z (base difficulty)": z_np,
            },
            model_state={
                "theta": theta.detach().cpu(),
                "z_train": z_train.detach().cpu(),
                "log_decay": log_decay.detach().cpu(),
                "decay_rate": decay_rate_val,
                "epochs": epochs,
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 3.0
