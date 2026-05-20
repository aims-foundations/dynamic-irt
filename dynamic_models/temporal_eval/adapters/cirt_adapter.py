"""CIRT model adapter.

    P(correct) = sigmoid(theta[s] - z[q] - lambda[q] * t)

Two parameters per item:
  - z[q]: baseline difficulty
  - lambda[q]: temporal difficulty slope (positive = harder over attempts)
One parameter per student:
  - theta[s]: ability
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


class CIRTAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "CIRT"

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
        lam_train = nn.Parameter(torch.zeros(Q_train, device=device))
        optimizer = optim.Adam([theta, z_train, lam_train], lr=lr)

        print(f"    [CIRT] {N} students, {Q_train} train items, "
              f"{len(y_obs)} observations, device={device}", flush=True)

        train_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            logit = theta[s_d] - z_train[q_d] - lam_train[q_d] * t_d
            pred = torch.sigmoid(logit)
            loss = F.binary_cross_entropy(pred, y_obs_d)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"    [CIRT] epoch {epoch+1}/{epochs} "
                      f"loss={loss.item():.4f}", flush=True)

        theta_np = theta.detach().cpu().numpy()
        z_np = z_train.detach().cpu().numpy()
        lam_np = lam_train.detach().cpu().numpy()

        test_corr = data.correctness_matrix[:, split.test_item_indices, :]
        y_obs_test, test_s_idx, test_q_idx, test_a_idx = _extract_valid_obs(test_corr)
        test_item_idx = split.test_item_indices[test_q_idx].numpy()
        test_t_flat = t_vals[test_a_idx]
        y_true = y_obs_test.numpy()

        with torch.no_grad():
            s_idx = torch.from_numpy(test_s_idx).to(device).long()
            t_test = torch.from_numpy(test_t_flat).to(device).float()
            # Test items unseen — use z=0, lambda=0 (prior mean)
            y_pred_prob = torch.sigmoid(theta[s_idx]).clamp(1e-6, 1-1e-6).cpu().numpy()

        return PredictionResult(
            y_true=y_true, y_pred_prob=y_pred_prob,
            student_indices=test_s_idx, item_indices=test_item_idx,
            losses={"train": train_losses},
            student_params={"theta (ability)": theta_np},
            item_params={"z (difficulty)": z_np, "lambda (temporal slope)": lam_np},
            model_state={"theta": theta.detach().cpu(), "z_train": z_train.detach().cpu(),
                         "lam_train": lam_train.detach().cpu()},
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

        # ---- Training (train students, all items) ----
        train_corr = data.correctness_matrix[split.train_student_indices, :, :]
        y_obs, s_idx_np, q_idx_np, t_idx_np = _extract_valid_obs(train_corr)
        t_flat_np = t_vals[t_idx_np]
        y_obs = y_obs * (1 - 2 * eps) + eps

        y_d = y_obs.to(device)
        s_d = torch.from_numpy(s_idx_np).to(device).long()
        q_d = torch.from_numpy(q_idx_np).to(device).long()
        t_d = torch.from_numpy(t_flat_np).to(device).float()

        theta_train = nn.Parameter(torch.randn(N_train, device=device))
        z = nn.Parameter(torch.abs(torch.randn(Q, device=device)))
        lam = nn.Parameter(torch.zeros(Q, device=device))
        optimizer = optim.Adam([theta_train, z, lam], lr=lr)

        print(f"    [CIRT] Training: {N_train} students, {Q} items, "
              f"{len(y_obs)} obs", flush=True)

        train_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            logit = theta_train[s_d] - z[q_d] - lam[q_d] * t_d
            pred = torch.sigmoid(logit)
            loss = F.binary_cross_entropy(pred, y_d)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"    [CIRT] train {epoch+1}/{epochs} "
                      f"loss={loss.item():.4f}", flush=True)

        z_np = z.detach().cpu().numpy()
        lam_np = lam.detach().cpu().numpy()
        z.requires_grad_(False)
        lam.requires_grad_(False)

        # ---- Calibration (test students, weeks 1-W items) ----
        calib_corr = data.correctness_matrix[split.test_student_indices][:, split.train_item_indices, :]
        y_calib, s_idx_np, q_idx_np, t_idx_np = _extract_valid_obs(calib_corr)
        t_flat_np = t_vals[t_idx_np]
        y_calib = y_calib * (1 - 2 * eps) + eps

        z_calib = z[split.train_item_indices]
        lam_calib = lam[split.train_item_indices]

        y_d = y_calib.to(device)
        s_d = torch.from_numpy(s_idx_np).to(device).long()
        q_d = torch.from_numpy(q_idx_np).to(device).long()
        t_d = torch.from_numpy(t_flat_np).to(device).float()

        theta_test = nn.Parameter(torch.zeros(N_test, device=device))
        optimizer = optim.Adam([theta_test], lr=lr)

        print(f"    [CIRT] Calibration: {N_test} students, {len(y_calib)} obs", flush=True)

        for epoch in range(calib_epochs):
            optimizer.zero_grad()
            logit = theta_test[s_d] - z_calib[q_d] - lam_calib[q_d] * t_d
            pred = torch.sigmoid(logit)
            loss = F.binary_cross_entropy(pred, y_d)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 200 == 0 or epoch == 0:
                print(f"    [CIRT] calib {epoch+1}/{calib_epochs} "
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
            z_pred = z[split.test_item_indices]
            lam_pred = lam[split.test_item_indices]
            logit = theta_test[s_d] - z_pred[q_d] - lam_pred[q_d] * t_d
            y_pred_prob = torch.sigmoid(logit).clamp(1e-6, 1-1e-6).cpu().numpy()

        print(f"    [CIRT] Predict: {len(y_true)} test obs", flush=True)

        return PredictionResult(
            y_true=y_true, y_pred_prob=y_pred_prob,
            student_indices=s_idx_np, item_indices=test_item_idx, attempt_indices=a_idx_np,
            losses={"train": train_losses},
            student_params={"theta (ability)": theta_test.detach().cpu().numpy()},
            item_params={"z (difficulty)": z_np, "lambda (temporal slope)": lam_np},
            model_state={
                "theta_test": theta_test.detach().cpu(),
                "z": z.detach().cpu(),
                "lam": lam.detach().cpu(),
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 3.0
