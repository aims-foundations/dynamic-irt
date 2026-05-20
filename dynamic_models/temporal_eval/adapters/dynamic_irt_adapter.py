"""Dynamic IRT (linear growth) adapter.

    P(correct) = sigmoid(theta0[s] + theta_growth[s] * t - beta[q])

Supports both temporal splits and student-based splits.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..temporal_split import TemporalSplit


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_valid_obs(corr_3d):
    S, Q, T = corr_3d.shape
    y_flat = corr_3d.reshape(-1)
    valid_mask = (y_flat != -1).numpy()
    valid_indices = np.where(valid_mask)[0]
    student_idx = valid_indices // (Q * T)
    question_idx = (valid_indices // T) % Q
    attempt_idx = valid_indices % T
    y_obs = y_flat[valid_mask].float()
    return y_obs, student_idx, question_idx, attempt_idx


class DynamicIRTAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "DynamicIRT"

    def fit_and_predict(
        self, data, split, seed=42, epochs=3000, lr=0.001, **kwargs,
    ) -> PredictionResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = _get_device()

        N = data.n_students
        T = data.n_max_attempts

        train_corr = data.correctness_matrix[:, split.train_item_indices, :]
        Q_train = split.n_train_items

        y_obs, s_idx_np, q_idx_np, t_idx_np = _extract_valid_obs(train_corr)
        t_vals_norm = np.linspace(1, T, T, dtype=np.float32) / T
        t_flat_np = t_vals_norm[t_idx_np]

        y_d = y_obs.to(device)
        t_d = torch.from_numpy(t_flat_np).to(device).float()
        s_d = torch.from_numpy(s_idx_np).to(device).long()
        q_d = torch.from_numpy(q_idx_np).to(device).long()

        theta0 = nn.Parameter(torch.zeros(N, device=device))
        theta_growth = nn.Parameter(torch.zeros(N, device=device))
        beta = nn.Parameter(torch.zeros(Q_train, device=device))
        optimizer = optim.Adam([theta0, theta_growth, beta], lr=lr)

        train_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            logit = theta0[s_d] + theta_growth[s_d] * t_d - beta[q_d]
            prob = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
            nll = -(y_d * torch.log(prob) + (1 - y_d) * torch.log(1 - prob)).mean()
            nll.backward()
            optimizer.step()
            train_losses.append(nll.item())

        test_corr = data.correctness_matrix[:, split.test_item_indices, :]
        Q_test = split.n_test_items
        y_obs_test, test_s_idx, test_q_idx, test_a_idx = _extract_valid_obs(test_corr)
        test_item_idx = split.test_item_indices[test_q_idx].numpy()
        test_t_flat = t_vals_norm[test_a_idx]
        y_true = y_obs_test.numpy()

        with torch.no_grad():
            s_idx = torch.from_numpy(test_s_idx).to(device).long()
            t_test = torch.from_numpy(test_t_flat).to(device).float()
            logit = theta0[s_idx] + theta_growth[s_idx] * t_test
            y_pred_prob = torch.sigmoid(logit).cpu().numpy()

        return PredictionResult(
            y_true=y_true, y_pred_prob=y_pred_prob,
            student_indices=test_s_idx, item_indices=test_item_idx,
            losses={"train": train_losses},
            student_params={"theta_0": theta0.detach().cpu().numpy(),
                           "theta_growth": theta_growth.detach().cpu().numpy()},
            item_params={"beta (difficulty)": beta.detach().cpu().numpy()},
            model_state={"theta0": theta0.detach().cpu(), "theta_growth": theta_growth.detach().cpu(),
                         "beta": beta.detach().cpu()},
        )

    def fit_and_predict_student_split(
        self, data, split, seed=42, epochs=3000, calib_epochs=1000,
        lr=0.001, **kwargs,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = _get_device()

        T = data.n_max_attempts
        N_train = len(split.train_student_indices)
        N_test = len(split.test_student_indices)
        Q = data.n_items
        t_vals_norm = np.linspace(1, T, T, dtype=np.float32) / T

        # ---- Calibration ----
        calib_corr = data.correctness_matrix[split.train_student_indices, :, :]
        y_obs, s_idx_np, q_idx_np, t_idx_np = _extract_valid_obs(calib_corr)
        t_flat_np = t_vals_norm[t_idx_np]

        y_d = y_obs.to(device)
        s_d = torch.from_numpy(s_idx_np).to(device).long()
        q_d = torch.from_numpy(q_idx_np).to(device).long()
        t_d = torch.from_numpy(t_flat_np).to(device).float()

        theta0_train = nn.Parameter(torch.zeros(N_train, device=device))
        theta_growth_train = nn.Parameter(torch.zeros(N_train, device=device))
        beta = nn.Parameter(torch.zeros(Q, device=device))
        optimizer = optim.Adam([theta0_train, theta_growth_train, beta], lr=lr)

        print(f"    [DynamicIRT] Training: {N_train} students, {Q} items, "
              f"{len(y_obs)} obs", flush=True)

        for epoch in range(epochs):
            optimizer.zero_grad()
            logit = theta0_train[s_d] + theta_growth_train[s_d] * t_d - beta[q_d]
            prob = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
            nll = -(y_d * torch.log(prob) + (1 - y_d) * torch.log(1 - prob)).mean()
            nll.backward()
            optimizer.step()
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"    [DynamicIRT] train{epoch+1}/{epochs} "
                      f"loss={nll.item():.4f}", flush=True)

        beta_np = beta.detach().cpu().numpy()
        beta.requires_grad_(False)

        # ---- Scoring (test students, weeks 1-W) ----
        calib_corr = data.correctness_matrix[split.test_student_indices][:, split.train_item_indices, :]
        y_calib, s_idx_np, q_idx_np, t_idx_np = _extract_valid_obs(calib_corr)
        t_flat_np = t_vals_norm[t_idx_np]

        beta_calib = beta[split.train_item_indices]

        y_d = y_calib.to(device)
        s_d = torch.from_numpy(s_idx_np).to(device).long()
        q_d = torch.from_numpy(q_idx_np).to(device).long()
        t_d = torch.from_numpy(t_flat_np).to(device).float()

        theta0_test = nn.Parameter(torch.zeros(N_test, device=device))
        theta_growth_test = nn.Parameter(torch.zeros(N_test, device=device))
        optimizer = optim.Adam([theta0_test, theta_growth_test], lr=lr)

        print(f"    [DynamicIRT] Calibration: {N_test} students, {len(y_calib)} obs", flush=True)

        for epoch in range(calib_epochs):
            optimizer.zero_grad()
            logit = theta0_test[s_d] + theta_growth_test[s_d] * t_d - beta_calib[q_d]
            prob = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
            nll = -(y_d * torch.log(prob) + (1 - y_d) * torch.log(1 - prob)).mean()
            nll.backward()
            optimizer.step()
            if (epoch + 1) % 200 == 0 or epoch == 0:
                print(f"    [DynamicIRT] calib{epoch+1}/{calib_epochs} "
                      f"loss={nll.item():.4f}", flush=True)

        # ---- Predict (test students, weeks W+1+) ----
        pred_corr = data.correctness_matrix[split.test_student_indices][:, split.test_item_indices, :]
        y_obs_pred, s_idx_np, q_idx_np, a_idx_np = _extract_valid_obs(pred_corr)
        y_true = y_obs_pred.numpy()
        test_item_idx = split.test_item_indices[q_idx_np]
        t_flat_np = t_vals_norm[a_idx_np]

        with torch.no_grad():
            s_d = torch.from_numpy(s_idx_np).to(device).long()
            q_d = torch.from_numpy(q_idx_np).to(device).long()
            t_d = torch.from_numpy(t_flat_np).to(device).float()
            beta_test = beta[split.test_item_indices]
            logit = theta0_test[s_d] + theta_growth_test[s_d] * t_d - beta_test[q_d]
            y_pred_prob = torch.sigmoid(logit).clamp(1e-6, 1-1e-6).cpu().numpy()

        print(f"    [DynamicIRT] Predict: {len(y_true)} test obs", flush=True)

        return PredictionResult(
            y_true=y_true, y_pred_prob=y_pred_prob,
            student_indices=s_idx_np, item_indices=test_item_idx, attempt_indices=a_idx_np,
            student_params={"theta_0": theta0_test.detach().cpu().numpy(),
                           "theta_growth": theta_growth_test.detach().cpu().numpy()},
            item_params={"beta (difficulty)": beta_np},
            model_state={"theta0_test": theta0_test.detach().cpu(),
                         "theta_growth_test": theta_growth_test.detach().cpu(),
                         "beta": beta.detach().cpu()},
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 3.0
