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

        # ---- Training (train students, all items) ----
        calib_corr = data.correctness_matrix[split.train_student_indices, :, :]
        y_obs, s_idx_np, q_idx_np, _ = _extract_valid_obs(calib_corr)
        y_obs = y_obs * (1 - 2 * eps) + eps

        y_obs_d = y_obs.to(device)
        s_idx_d = torch.from_numpy(s_idx_np).to(device).long()
        q_idx_d = torch.from_numpy(q_idx_np).to(device).long()

        theta_train = nn.Parameter(torch.randn(N_train, device=device))
        b = nn.Parameter(torch.randn(Q, device=device))
        optimizer = optim.Adam([theta_train, b], lr=lr)

        print(f"    [IRT] Training: {N_train} students, {Q} items, "
              f"{len(y_obs)} obs, device={device}", flush=True)

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = torch.sigmoid(theta_train[s_idx_d] - b[q_idx_d])
            loss = F.binary_cross_entropy(pred, y_obs_d)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"    [IRT] train epoch {epoch+1}/{epochs} "
                      f"loss={loss.item():.4f}", flush=True)

        b_np = b.detach().cpu().numpy()
        b.requires_grad_(False)

        # ---- Calibration (test students, weeks 1-W items) ----
        calib_corr = data.correctness_matrix[split.test_student_indices][:, split.train_item_indices, :]
        y_calib, s_idx_np, q_idx_np, _ = _extract_valid_obs(calib_corr)
        y_calib = y_calib * (1 - 2 * eps) + eps

        b_calib = b[split.train_item_indices]

        y_calib_d = y_calib.to(device)
        s_idx_d = torch.from_numpy(s_idx_np).to(device).long()
        q_idx_d = torch.from_numpy(q_idx_np).to(device).long()

        theta_test = nn.Parameter(torch.zeros(N_test, device=device))
        optimizer = optim.Adam([theta_test], lr=lr)

        print(f"    [IRT] Calibration: {N_test} test students, "
              f"{len(split.train_item_indices)} items, "
              f"{len(y_calib)} obs", flush=True)

        for epoch in range(calib_epochs):
            optimizer.zero_grad()
            pred = torch.sigmoid(theta_test[s_idx_d] - b_calib[q_idx_d])
            loss = F.binary_cross_entropy(pred, y_calib_d)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 200 == 0 or epoch == 0:
                print(f"    [IRT] calib epoch {epoch+1}/{calib_epochs} "
                      f"loss={loss.item():.4f}", flush=True)

        theta_test_np = theta_test.detach().cpu().numpy()

        # ---- Predict (test students, weeks W+1+ items) ----
        pred_corr = data.correctness_matrix[split.test_student_indices][:, split.test_item_indices, :]
        y_obs_pred, s_idx_np, q_idx_np, attempt_idx = _extract_valid_obs(pred_corr)
        y_true = y_obs_pred.numpy()
        test_item_idx = split.test_item_indices[q_idx_np]

        with torch.no_grad():
            s_d = torch.from_numpy(s_idx_np).to(device).long()
            q_d = torch.from_numpy(q_idx_np).to(device).long()
            b_test = b[split.test_item_indices]
            pred = torch.sigmoid(theta_test[s_d] - b_test[q_d])
            y_pred_prob = pred.clamp(1e-6, 1 - 1e-6).cpu().numpy()

        print(f"    [IRT] Predict: {len(y_true)} test obs", flush=True)

        return PredictionResult(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            student_indices=s_idx_np,
            item_indices=test_item_idx,
            attempt_indices=attempt_idx,
            student_params={"theta (ability)": theta_test_np},
            item_params={"b (difficulty)": b_np},
            model_state={
                "theta_test": theta_test.detach().cpu(),
                "b": b.detach().cpu(),
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 2.0
