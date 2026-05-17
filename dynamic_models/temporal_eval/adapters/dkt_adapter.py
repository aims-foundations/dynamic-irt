"""Deep Knowledge Tracing (DKT) adapter for temporal evaluation.

Follows Piech et al. 2015 (arXiv:1506.05908):

    x_t = one-hot encoding of (q_t, a_t) in {0,1}^{2M}
    h_t = LSTM(x_t, h_{t-1})
    y_t = sigmoid(W_yh * dropout(h_t) + b_y)

    Loss = BCE(y_t[q_{t+1}], a_{t+1})

Output y_t is a vector over all M items; loss is computed only on the
item actually attempted at the next timestep. Dropout applied on the
hidden state when computing the readout (not the recurrence). Gradient
clipping prevents exploding gradients.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..temporal_split import TemporalSplit


class DKTModel(nn.Module):
    def __init__(self, input_dim, n_items, hidden_dim, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, n_items)

    def forward(self, x):
        """x: (batch, seq_len, input_dim).
        Returns: (batch, seq_len, n_items) predicted P(correct)."""
        h, _ = self.lstm(x)
        h = self.dropout(h)
        return torch.sigmoid(self.output(h))


class DKTAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "DKT"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        hidden_dim: int = 200,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 100,
        max_seq_len: int = 0,
        dropout: float = 0.2,
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
        corr = data.correctness_matrix.numpy()
        time_mat = data.time_matrix.numpy()
        M = data.n_items

        train_idx = split.train_item_indices.numpy()
        test_idx = split.test_item_indices.numpy()
        Q_train = len(train_idx)
        Q_test = len(test_idx)

        input_dim = 2 * M

        print(f"    [DKT] {N} students, {M} total items "
              f"({Q_train} train, {Q_test} test), "
              f"hidden={hidden_dim}, device={device}", flush=True)

        # ---- Build per-student sequences from train-week interactions ----
        # Sort by real timestamp (time_matrix) for true chronological order
        student_seqs = []
        for s in range(N):
            events = []
            for item_i in train_idx:
                for t in range(T):
                    c = corr[s, item_i, t]
                    if c != -1:
                        ts = time_mat[s, item_i, t]
                        events.append((ts, int(item_i), int(c)))
            events.sort(key=lambda x: x[0])
            if max_seq_len > 0 and len(events) > max_seq_len:
                events = events[-max_seq_len:]
            student_seqs.append(events)

        seq_lengths = [len(s) for s in student_seqs]
        actual_max = max(seq_lengths) if seq_lengths else 1
        if actual_max == 0:
            raise ValueError("No train interactions found")

        print(f"    [DKT] max_seq={actual_max} (cap={max_seq_len}), "
              f"input_dim={input_dim}", flush=True)

        # ---- Encode as sparse token IDs ----
        token_ids = torch.zeros(N, actual_max, dtype=torch.long)
        next_item = torch.full((N, actual_max), -1, dtype=torch.long)
        next_corr = torch.full((N, actual_max), -1.0)

        for s in range(N):
            events = student_seqs[s]
            for t, (_, item_i, c) in enumerate(events):
                token_ids[s, t] = item_i * 2 + c
                if t > 0:
                    next_item[s, t - 1] = item_i
                    next_corr[s, t - 1] = float(c)

        # ---- Train ----
        model = DKTModel(input_dim, M, hidden_dim, dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        max_grad_norm = 5.0

        train_losses = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(N)
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                B = len(idx)

                tids = token_ids[idx]
                x_b = torch.zeros(B, actual_max, input_dim, device=device)
                for b in range(B):
                    for t in range(seq_lengths[idx[b]]):
                        x_b[b, t, tids[b, t]] = 1.0

                ni_b = next_item[idx].to(device)
                nc_b = next_corr[idx].to(device)

                pred = model(x_b)

                valid = (ni_b >= 0)
                if valid.sum() == 0:
                    continue

                pred_flat = pred.reshape(-1, M)
                ni_flat = ni_b.reshape(-1)
                nc_flat = nc_b.reshape(-1)
                valid_flat = valid.reshape(-1)

                pred_selected = pred_flat[valid_flat, ni_flat[valid_flat]]
                target_selected = nc_flat[valid_flat]

                loss = F.binary_cross_entropy(pred_selected, target_selected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"    [DKT] epoch {epoch+1}/{epochs} "
                      f"loss={avg_loss:.4f}", flush=True)

        # ---- Predict on test items ----
        model.eval()
        test_corr_mat = corr[:, test_idx, :T]

        with torch.no_grad():
            # Use trained train-item predictions as student mastery signal
            student_mastery = np.zeros(N)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                B = end - start

                tids = token_ids[start:end]
                x_b = torch.zeros(B, actual_max, input_dim, device=device)
                for b in range(B):
                    s = start + b
                    for t in range(seq_lengths[s]):
                        x_b[b, t, tids[b, t]] = 1.0

                pred = model(x_b)

                for b in range(B):
                    s = start + b
                    if seq_lengths[s] > 0:
                        last_t = seq_lengths[s] - 1
                        # Average over trained train-item outputs (these weights are learned)
                        student_mastery[s] = pred[b, last_t, train_idx].mean().cpu().item()
                    else:
                        student_mastery[s] = 0.5

        # Build test observation vectors
        y_flat = test_corr_mat.reshape(-1)
        valid_mask = y_flat != -1
        valid_indices = np.where(valid_mask)[0]

        test_student_idx = valid_indices // (Q_test * T)
        test_local_q_idx = (valid_indices // T) % Q_test
        test_attempt_idx = valid_indices % T
        test_item_idx = test_idx[test_local_q_idx]

        y_true = y_flat[valid_mask].astype(float)
        y_pred_prob = np.clip(student_mastery[test_student_idx], 1e-6, 1 - 1e-6)

        if len(y_true) == 0:
            raise ValueError(
                f"No test observations for cutoff_week={split.cutoff_week}"
            )

        print(f"    [DKT] {Q_test} test items, {len(y_true)} test obs", flush=True)

        return PredictionResult(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            student_indices=test_student_idx,
            item_indices=test_item_idx,
            attempt_indices=test_attempt_idx,
            losses={"train": train_losses},
            student_params={
                "mastery": student_mastery,
            },
            model_state={
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "hidden_dim": hidden_dim,
                "n_items": M,
                "student_mastery": student_mastery,
                "epochs": epochs,
            },
        )

    def _build_sequences(self, corr, time_mat, student_indices, item_indices, T):
        """Build chronologically-ordered interaction sequences for given students/items."""
        seqs = []
        for s in student_indices:
            events = []
            for item_i in item_indices:
                for t in range(T):
                    c = corr[s, item_i, t]
                    if c != -1:
                        ts = time_mat[s, item_i, t]
                        events.append((ts, int(item_i), int(c)))
            events.sort(key=lambda x: x[0])
            seqs.append(events)
        return seqs

    def _encode_sequences(self, seqs, input_dim):
        """Encode sequences as token ID tensors."""
        seq_lengths = [len(s) for s in seqs]
        actual_max = max(seq_lengths) if seq_lengths else 1
        N = len(seqs)

        token_ids = torch.zeros(N, actual_max, dtype=torch.long)
        next_item = torch.full((N, actual_max), -1, dtype=torch.long)
        next_corr = torch.full((N, actual_max), -1.0)

        for s in range(N):
            for t, (_, item_i, c) in enumerate(seqs[s]):
                token_ids[s, t] = item_i * 2 + c
                if t > 0:
                    next_item[s, t - 1] = item_i
                    next_corr[s, t - 1] = float(c)

        return token_ids, next_item, next_corr, seq_lengths, actual_max

    def _train_model(self, model, token_ids, next_item, next_corr, seq_lengths,
                     actual_max, input_dim, M, device, optimizer,
                     epochs, batch_size, max_grad_norm=5.0):
        """Run the DKT training loop."""
        N = len(seq_lengths)
        train_losses = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(N)
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                B = len(idx)

                tids = token_ids[idx]
                x_b = torch.zeros(B, actual_max, input_dim, device=device)
                for b in range(B):
                    for t in range(seq_lengths[idx[b]]):
                        x_b[b, t, tids[b, t]] = 1.0

                ni_b = next_item[idx].to(device)
                nc_b = next_corr[idx].to(device)

                pred = model(x_b)

                valid = (ni_b >= 0)
                if valid.sum() == 0:
                    continue

                pred_flat = pred.reshape(-1, M)
                ni_flat = ni_b.reshape(-1)
                nc_flat = nc_b.reshape(-1)
                valid_flat = valid.reshape(-1)

                pred_selected = pred_flat[valid_flat, ni_flat[valid_flat]]
                target_selected = nc_flat[valid_flat]

                loss = F.binary_cross_entropy(pred_selected, target_selected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"    [DKT] epoch {epoch+1}/{epochs} "
                      f"loss={avg_loss:.4f}", flush=True)

        return train_losses

    def _predict_batch(self, model, token_ids, seq_lengths, actual_max,
                       input_dim, target_item_indices, device, batch_size):
        """Get per-item predictions at each student's last timestep."""
        N = len(seq_lengths)
        Q_target = len(target_item_indices)
        all_preds = np.zeros((N, Q_target))

        model.eval()
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                B = end - start

                tids = token_ids[start:end]
                x_b = torch.zeros(B, actual_max, input_dim, device=device)
                for b in range(B):
                    for t in range(seq_lengths[start + b]):
                        x_b[b, t, tids[b, t]] = 1.0

                pred = model(x_b)

                for b in range(B):
                    if seq_lengths[start + b] > 0:
                        last_t = seq_lengths[start + b] - 1
                        all_preds[start + b] = pred[b, last_t, target_item_indices].cpu().numpy()
                    else:
                        all_preds[start + b] = 0.5

        return all_preds

    def fit_and_predict_student_split(
        self, data, split, seed=42, hidden_dim=64, epochs=50,
        lr=0.001, batch_size=100, dropout=0.5, **kwargs,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        T = data.n_max_attempts
        M = data.n_items
        corr = data.correctness_matrix.numpy()
        time_mat = data.time_matrix.numpy()
        input_dim = 2 * M

        train_students = split.train_student_indices
        test_students = split.test_student_indices
        test_items = split.test_item_indices
        scoring_items = split.train_item_indices
        N_train = len(train_students)
        N_test = len(test_students)

        print(f"    [DKT] Train: {N_train} students, Test: {N_test} students, "
              f"{M} items, hidden={hidden_dim}, device={device}", flush=True)

        # ---- Training: train students, ALL items, ALL weeks ----
        all_items = np.arange(M)
        train_seqs = self._build_sequences(corr, time_mat, train_students, all_items, T)
        token_ids, next_item, next_corr, seq_lengths, actual_max = \
            self._encode_sequences(train_seqs, input_dim)

        print(f"    [DKT] Training seqs: max_len={actual_max}", flush=True)

        model = DKTModel(input_dim, M, hidden_dim, dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = self._train_model(
            model, token_ids, next_item, next_corr, seq_lengths,
            actual_max, input_dim, M, device, optimizer, epochs, batch_size,
        )

        # ---- Test: feed test students' weeks 1-W sequences, predict weeks W+1+ ----
        test_seqs = self._build_sequences(corr, time_mat, test_students, scoring_items, T)
        test_tids, _, _, test_seq_lens, test_max = \
            self._encode_sequences(test_seqs, input_dim)

        print(f"    [DKT] Test scoring seqs: max_len={test_max}", flush=True)

        # Get item-specific predictions for test items
        all_test_preds = self._predict_batch(
            model, test_tids, test_seq_lens, test_max,
            input_dim, test_items, device, batch_size,
        )

        # Build test observation vectors
        Q_test = len(test_items)
        pred_corr = corr[np.ix_(test_students, test_items)]
        y_flat = pred_corr.reshape(-1)
        valid_mask = y_flat != -1
        valid_indices = np.where(valid_mask)[0]

        s_idx = valid_indices // (Q_test * T)
        q_idx = (valid_indices // T) % Q_test
        a_idx = valid_indices % T
        item_idx = test_items[q_idx]

        y_true = y_flat[valid_mask].astype(float)
        y_pred_prob = np.array([all_test_preds[s, q] for s, q in zip(s_idx, q_idx)])
        y_pred_prob = np.clip(y_pred_prob, 1e-6, 1 - 1e-6)

        print(f"    [DKT] Predict: {len(y_true)} test obs "
              f"({Q_test} items, {N_test} students)", flush=True)

        return PredictionResult(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            student_indices=s_idx,
            item_indices=item_idx,
            attempt_indices=a_idx,
            losses={"train": train_losses},
            student_params={"mastery": all_test_preds.mean(axis=1)},
            model_state={
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "hidden_dim": hidden_dim,
                "n_items": M,
                "epochs": epochs,
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 5.0
