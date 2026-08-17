"""Code-DKT adapter: DKT with per-submission code embeddings.

Follows Shi et al. 2022 (EDM), which fuses a vector representation of
the submitted code with DKT's interaction encoding at each timestep of
a single recurrent model. The original uses code2vec AST-path features;
this adaptation uses the repository's precomputed LLM code embeddings,
the same features consumed by the RSSM, so the two code-aware models
see identical code signal.

    o_t = one-hot(item_t, corr_t)            in R^{2M}   (as DKT)
    c_t = code embedding of the submission   in R^{emb_dim}
    p_t = dropout(W_proj c_t) * has_emb_t    in R^{proj_dim}
    x_t = concat(o_t, p_t)
    h_t = LSTM(x_t, h_{t-1})
    y_t = sigmoid(W_out dropout(h_t))        in R^{M}

    Loss = BCE(y_t[item_{t+1}], corr_{t+1})  (as DKT)

Missing embeddings contribute a zero vector after projection; the
adapter raises if more than 2% of train and test events lack an
embedding, since that indicates a join bug rather than legitimate gaps.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..embedding_alignment import (
    align_to_universe,
    load_embeddings,
    resolve_emb_dir,
)
from ..temporal_split import TemporalSplit
from .dkt_adapter import (
    build_event_sequences,
    encode_event_sequences,
    flatten_test_observations,
)


class CodeDKTModel(nn.Module):
    def __init__(self, input_dim, n_items, hidden_dim, emb_dim, proj_dim,
                 dropout):
        super().__init__()
        self.n_items = n_items
        self.proj = nn.Linear(emb_dim, proj_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_dim + proj_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, n_items)

    def forward(self, x_onehot, code_emb, has_emb):
        """x_onehot: [B, L, 2M]; code_emb: [B, L, emb_dim];
        has_emb: [B, L] float mask. Returns [B, L, M] P(correct)."""
        p = self.proj_dropout(self.proj(code_emb)) * has_emb.unsqueeze(-1)
        h, _ = self.lstm(torch.cat([x_onehot, p], dim=-1))
        return torch.sigmoid(self.output(self.dropout(h)))


class CodeDKTAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "CodeDKT"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        **kwargs,
    ) -> PredictionResult:
        raise NotImplementedError(
            "CodeDKT supports student splits only. "
            "Use fit_and_predict_student_split() instead."
        )

    def _batch_inputs(self, token_ids, emb_rows, valid, idx, input_dim,
                      emb_matrix, device):
        tids = token_ids[idx]
        v = valid[idx]
        x_onehot = F.one_hot(tids, num_classes=input_dim).float()
        x_onehot = x_onehot * v.unsqueeze(-1)
        code = emb_matrix[emb_rows[idx]].to(device).float()
        has = (emb_rows[idx] > 0).float()
        return x_onehot.to(device), code, has.to(device)

    @staticmethod
    def _check_missing_embeddings(n_missing, n_events, label):
        """Hard-fail if too many events lack an embedding row."""
        miss_frac = n_missing / max(n_events, 1)
        if miss_frac > 0.02:
            raise ValueError(
                f"{100 * miss_frac:.2f}% of {label} events lack an embedding "
                f"row; embedding join is broken (expected < 2%)."
            )
        return miss_frac

    def fit_and_predict_student_split(
        self, data, split, seed=42, hidden_dim=64, epochs=200, lr=0.001,
        # dropout=0.2 deliberately differs from DKT's 0.5; existing results used it
        batch_size=100, dropout=0.2, proj_dim=32, max_seq_len=600,
        # Code-DKT targets the unfiltered superset embeddings (Modal passes this dir explicitly)
        emb_dir="", emb_model_tag="Qwen3-Embedding-8B-unfiltered", strict_universe=True,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        M = data.n_items
        corr = data.correctness_matrix.numpy()
        time_mat = data.time_matrix.numpy()
        input_dim = 2 * M

        emb = load_embeddings(resolve_emb_dir(
            data.course_name, emb_dir=emb_dir, model_tag=emb_model_tag
        ))
        emb_matrix, row_lookup, stats = align_to_universe(
            emb, data, strict=strict_universe
        )
        emb_dim = stats["emb_dim"]

        train_students = split.train_student_indices
        test_students = split.test_student_indices
        test_items = split.test_item_indices
        scoring_items = split.train_item_indices
        N_train = len(train_students)
        N_test = len(test_students)
        item_to_qidx = data.question_infos["qidx"].astype(int).values

        n_params = sum(
            p.numel() for p in CodeDKTModel(
                input_dim, M, hidden_dim, emb_dim, proj_dim, dropout
            ).parameters()
        )
        print(f"    [CodeDKT] Train: {N_train} students, Test: {N_test} students, "
              f"{M} items, hidden={hidden_dim}, proj={proj_dim}, "
              f"emb_dim={emb_dim}, params={n_params:,}, device={device}",
              flush=True)

        # ---- Training: train students, ALL items, ALL weeks ----
        # Test cases of one submission share a timestamp and attempt slot and
        # intentionally share one embedding row.
        all_items = np.arange(M)
        train_seqs, n_ev, n_miss = build_event_sequences(
            corr, time_mat, train_students, all_items, max_seq_len,
            row_lookup=row_lookup, item_to_qidx=item_to_qidx,
        )
        miss_frac = self._check_missing_embeddings(n_miss, n_ev, "train")
        print(f"    [CodeDKT] Train events: {n_ev}, missing embedding: "
              f"{n_miss} ({100 * miss_frac:.2f}%)", flush=True)

        token_ids, emb_rows, valid, next_item, next_corr, _, actual_max = \
            encode_event_sequences(train_seqs)
        print(f"    [CodeDKT] Training seqs: max_len={actual_max} "
              f"(cap={max_seq_len})", flush=True)

        model = CodeDKTModel(input_dim, M, hidden_dim, emb_dim, proj_dim,
                             dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        max_grad_norm = 5.0

        N = len(train_seqs)
        train_losses = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(N)
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                x_onehot, code, has = self._batch_inputs(
                    token_ids, emb_rows, valid, idx, input_dim,
                    emb_matrix, device,
                )
                ni_b = next_item[idx].to(device)
                nc_b = next_corr[idx].to(device)

                pred = model(x_onehot, code, has)

                valid_next = (ni_b >= 0)
                if valid_next.sum() == 0:
                    continue
                pred_flat = pred.reshape(-1, M)
                ni_flat = ni_b.reshape(-1)
                nc_flat = nc_b.reshape(-1)
                vf = valid_next.reshape(-1)
                loss = F.binary_cross_entropy(
                    pred_flat[vf, ni_flat[vf]], nc_flat[vf]
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"    [CodeDKT] epoch {epoch+1}/{epochs} "
                      f"loss={avg_loss:.4f}", flush=True)

        # ---- Test: calibration prefix from weeks 1-W, predict weeks W+1+ ----
        test_seqs, t_ev, t_miss = build_event_sequences(
            corr, time_mat, test_students, scoring_items, max_seq_len,
            row_lookup=row_lookup, item_to_qidx=item_to_qidx,
        )
        self._check_missing_embeddings(t_miss, t_ev, "test")
        print(f"    [CodeDKT] Test scoring seqs: "
              f"max_len={max((len(s) for s in test_seqs), default=0)}, "
              f"missing embedding: {t_miss}/{t_ev}", flush=True)
        t_tids, t_rows, t_valid, _, _, t_lens, _ = \
            encode_event_sequences(test_seqs)

        Q_test = len(test_items)
        all_test_preds = np.zeros((N_test, Q_test))
        target_idx = torch.as_tensor(test_items, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            for start in range(0, N_test, batch_size):
                end = min(start + batch_size, N_test)
                idx = torch.arange(start, end)
                x_onehot, code, has = self._batch_inputs(
                    t_tids, t_rows, t_valid, idx, input_dim,
                    emb_matrix, device,
                )
                pred = model(x_onehot, code, has)
                for b in range(end - start):
                    s = start + b
                    if t_lens[s] > 0:
                        all_test_preds[s] = pred[b, t_lens[s] - 1, target_idx] \
                            .cpu().numpy()
                    else:
                        all_test_preds[s] = 0.5

        y_true, y_pred_prob, s_idx, item_idx, a_idx = flatten_test_observations(
            corr, test_students, test_items, all_test_preds,
        )

        print(f"    [CodeDKT] Predict: {len(y_true)} test obs "
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
                "proj_dim": proj_dim,
                "emb_dim": emb_dim,
                "emb_dir": emb["emb_dir"],
                "max_seq_len": max_seq_len,
                "n_items": M,
                "n_params": n_params,
                "epochs": epochs,
                "missing_event_fraction": miss_frac,
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 20.0
