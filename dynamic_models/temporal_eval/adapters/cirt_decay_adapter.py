"""CIRT-Decay model adapter.

    mean_correct = sigmoid(theta[s] - z[q] * exp(-lambda * t))

Supports both temporal splits and student-based splits.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CIRTDecayAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "CIRT-Decay"

    def fit_and_predict(
        self, data, split, seed=42, epochs=3000, lr=0.01, eps=1e-2, **kwargs,
    ) -> PredictionResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = _get_device()

        N = data.n_students
        T = data.n_max_attempts

        train_corr = data.correctness_matrix[:, split.train_item_indices, :]
        Q_train = split.n_train_items

        y_obs, s_idx_np, q_idx_np, t_idx_np = _extract_valid_obs(train_corr)
        t_vals = np.linspace(1, T, T, dtype=np.float32)
        t_flat_np = t_vals[t_idx_np]
        y_obs = y_obs * (1 - 2 * eps) + eps

        y_obs_d = y_obs.to(device)
        t_d = torch.from_numpy(t_flat_np).to(device).float()
        s_d = torch.from_numpy(s_idx_np).to(device).long()
        q_d = torch.from_numpy(q_idx_np).to(device).long()

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
            eff_diff = z_train[q_d] * torch.exp(-decay_rate * t_d)
            pred = torch.sigmoid(theta[s_d] - eff_diff)
            loss = F.binary_cross_entropy(pred, y_obs_d)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"    [CIRT-Decay] epoch {epoch+1}/{epochs} "
                      f"loss={loss.item():.4f} decay_rate={decay_rate.item():.6f}",
                      flush=True)

        theta_np = theta.detach().cpu().numpy()
        z_np = z_train.detach().cpu().numpy()
        decay_rate_val = torch.exp(log_decay).detach().cpu().item()

        test_corr = data.correctness_matrix[:, split.test_item_indices, :]
        Q_test = split.n_test_items
        y_obs_test, test_s_idx, test_q_idx, test_a_idx = _extract_valid_obs(test_corr)
        test_item_idx = split.test_item_indices[test_q_idx].numpy()
        test_t_flat = t_vals[test_a_idx]
        y_true = y_obs_test.numpy()

        with torch.no_grad():
            s_idx = torch.from_numpy(test_s_idx).to(device).long()
            t_test = torch.from_numpy(test_t_flat).to(device).float()
            z_test = torch.zeros(1, device=device)
            decay_rate = torch.exp(log_decay)
            eff_diff = z_test * torch.exp(-decay_rate * t_test)
            y_pred_prob = torch.sigmoid(theta[s_idx] - eff_diff).clamp(1e-6, 1-1e-6).cpu().numpy()

        return PredictionResult(
            y_true=y_true, y_pred_prob=y_pred_prob,
            student_indices=test_s_idx, item_indices=test_item_idx,
            losses={"train": train_losses},
            student_params={"theta (ability)": theta_np},
            item_params={"z (base difficulty)": z_np},
            model_state={"theta": theta.detach().cpu(), "z_train": z_train.detach().cpu(),
                         "log_decay": log_decay.detach().cpu(), "decay_rate": decay_rate_val},
        )

    def fit_and_predict_student_split(
        self, data, split, seed=42, epochs=3000, calib_epochs=1000,
        lr=0.01, eps=1e-2, **kwargs,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = _get_device()

        T = data.n_max_attempts
        N_train = len(split.train_student_indices)
        N_test = len(split.test_student_indices)
        Q = data.n_items
        t_vals = np.linspace(1, T, T, dtype=np.float32)

        # ---- Calibration (train students, all items) ----
        calib_corr = data.correctness_matrix[split.train_student_indices, :, :]
        y_obs, s_idx_np, q_idx_np, t_idx_np = _extract_valid_obs(calib_corr)
        t_flat_np = t_vals[t_idx_np]
        y_obs = y_obs * (1 - 2 * eps) + eps

        y_d = y_obs.to(device)
        s_d = torch.from_numpy(s_idx_np).to(device).long()
        q_d = torch.from_numpy(q_idx_np).to(device).long()
        t_d = torch.from_numpy(t_flat_np).to(device).float()

        theta_train = nn.Parameter(torch.randn(N_train, device=device))
        z = nn.Parameter(torch.abs(torch.randn(Q, device=device)))
        log_decay = nn.Parameter(torch.tensor(-3.0, device=device))
        optimizer = optim.Adam([theta_train, z, log_decay], lr=lr)

        print(f"    [CIRT-Decay] Training: {N_train} students, {Q} items, "
              f"{len(y_obs)} obs", flush=True)

        for epoch in range(epochs):
            optimizer.zero_grad()
            decay = torch.exp(log_decay)
            pred = torch.sigmoid(theta_train[s_d] - z[q_d] * torch.exp(-decay * t_d))
            loss = F.binary_cross_entropy(pred, y_d)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"    [CIRT-Decay] train{epoch+1}/{epochs} "
                      f"loss={loss.item():.4f} decay={decay.item():.6f}", flush=True)

        z_np = z.detach().cpu().numpy()
        z.requires_grad_(False)
        log_decay.requires_grad_(False)

        # ---- Calibration (test students, weeks 1-W items) ----
        calib_corr = data.correctness_matrix[split.test_student_indices][:, split.train_item_indices, :]
        y_calib, s_idx_np, q_idx_np, t_idx_np = _extract_valid_obs(calib_corr)
        t_flat_np = t_vals[t_idx_np]
        y_calib = y_calib * (1 - 2 * eps) + eps

        z_scoring = z[split.train_item_indices]

        y_d = y_calib.to(device)
        s_d = torch.from_numpy(s_idx_np).to(device).long()
        q_d = torch.from_numpy(q_idx_np).to(device).long()
        t_d = torch.from_numpy(t_flat_np).to(device).float()

        theta_test = nn.Parameter(torch.zeros(N_test, device=device))
        optimizer = optim.Adam([theta_test], lr=lr)

        print(f"    [CIRT-Decay] Calibration: {N_test} students, {len(y_calib)} obs", flush=True)

        for epoch in range(calib_epochs):
            optimizer.zero_grad()
            decay = torch.exp(log_decay)
            pred = torch.sigmoid(theta_test[s_d] - z_scoring[q_d] * torch.exp(-decay * t_d))
            loss = F.binary_cross_entropy(pred, y_d)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 200 == 0 or epoch == 0:
                print(f"    [CIRT-Decay] calib{epoch+1}/{calib_epochs} "
                      f"loss={loss.item():.4f}", flush=True)

        # ---- Predict (test students, weeks W+1+ items) ----
        pred_corr = data.correctness_matrix[split.test_student_indices][:, split.test_item_indices, :]
        y_obs_pred, s_idx_np, q_idx_np, a_idx_np = _extract_valid_obs(pred_corr)
        y_true = y_obs_pred.numpy()
        test_item_idx = split.test_item_indices[q_idx_np]
        t_flat_np = t_vals[a_idx_np]

        with torch.no_grad():
            s_d = torch.from_numpy(s_idx_np).to(device).long()
            q_d = torch.from_numpy(q_idx_np).to(device).long()
            t_d = torch.from_numpy(t_flat_np).to(device).float()
            z_test = z[split.test_item_indices]
            decay = torch.exp(log_decay)
            pred = torch.sigmoid(theta_test[s_d] - z_test[q_d] * torch.exp(-decay * t_d))
            y_pred_prob = pred.clamp(1e-6, 1-1e-6).cpu().numpy()

        print(f"    [CIRT-Decay] Predict: {len(y_true)} test obs", flush=True)

        return PredictionResult(
            y_true=y_true, y_pred_prob=y_pred_prob,
            student_indices=s_idx_np, item_indices=test_item_idx, attempt_indices=a_idx_np,
            student_params={"theta (ability)": theta_test.detach().cpu().numpy()},
            item_params={"z (base difficulty)": z_np},
            model_state={"theta_test": theta_test.detach().cpu(), "z": z.detach().cpu(),
                         "log_decay": log_decay.detach().cpu()},
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 3.0
