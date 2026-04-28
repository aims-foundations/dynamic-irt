"""RSSM adapter for temporal evaluation.

Trains a Multi-Modal RSSM from scratch on train-week interactions,
then predicts test-week interactions using the learned model.

The temporal split works at the interaction level:
- Each interaction maps to a question, each question has a week.
- Train: interactions on questions from weeks 1..W
- Test: interactions on questions from weeks W+1+
- Per-student sequences are preserved in time order.

Uses batched GRU forward pass (nn.GRU) for efficiency instead of
step-by-step GRUCell, avoiding Python loop overhead. Processes students
in mini-batches with gradient accumulation for large datasets.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..temporal_split import TemporalSplit


class RSSMAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "RSSM"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        hidden_dim: int = 128,
        enc_dim: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 500,
        patience: int = 100,
        grad_clip: float = 1.0,
        aux_loss_weight: float = 0.1,
        **kwargs,
    ) -> PredictionResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            default_device = "mps"
        else:
            default_device = "cpu"
        device = kwargs.get("device", default_device)

        from dynamic_irt.featurize import FeatureConfig
        from dynamic_irt.rssm import (
            AnswerEncoder,
            AuxDecoder as AnswerFeaturePredictor,
            Scorer as MultiModalScorer,
            HandcraftedQuestionEncoder as QuestionEncoder,
        )

        config = FeatureConfig()

        # Load pre-processed multimodal data
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        data_dir = os.path.join(repo_root, "data", "multimodal", data.course_name)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"RSSM multimodal data not found at {data_dir}. "
                f"Run: python -m dynamic_irt.featurize --mode features --course {data.course_name}"
            )

        with open(f"{data_dir}/answer_features.pkl", "rb") as f:
            answer_features = pickle.load(f)
        with open(f"{data_dir}/question_idxs.pkl", "rb") as f:
            question_idxs = pickle.load(f)
        with open(f"{data_dir}/question_static.pkl", "rb") as f:
            question_static = pickle.load(f)
        with open(f"{data_dir}/testcase_scores.pkl", "rb") as f:
            testcase_scores = pickle.load(f)
        with open(f"{data_dir}/student_idxs.pkl", "rb") as f:
            student_idxs = pickle.load(f)
        with open(f"{data_dir}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        n_questions = metadata["n_questions"]
        n_tc = config.n_testcases
        answer_dim = config.answer_dim

        # Build local_question_idx -> week mapping
        question_to_idx = metadata["question_to_idx"]
        local_idx_to_week = {}
        for qid, local_idx in question_to_idx.items():
            if qid in data.qid_to_week:
                local_idx_to_week[local_idx] = data.qid_to_week[qid]

        # Group interactions by student, tagging each as train/test
        per_student = {}
        for feat, qidx, tc, si in zip(
            answer_features, question_idxs, testcase_scores, student_idxs
        ):
            if si not in per_student:
                per_student[si] = []
            week = local_idx_to_week.get(qidx, 0)
            is_train = week <= split.cutoff_week
            per_student[si].append((feat, qidx, tc, is_train))

        # Split into train and test sequences per student
        train_seqs = []
        test_seqs = []

        for si in sorted(per_student.keys()):
            interactions = per_student[si]
            train_feats, train_qidxs, train_tcs = [], [], []
            test_feats, test_qidxs, test_tcs = [], [], []
            for feat, qidx, tc, is_train in interactions:
                if is_train:
                    train_feats.append(torch.tensor(feat, dtype=torch.float32))
                    train_qidxs.append(qidx)
                    train_tcs.append(torch.tensor(tc, dtype=torch.float32))
                else:
                    test_feats.append(torch.tensor(feat, dtype=torch.float32))
                    test_qidxs.append(qidx)
                    test_tcs.append(torch.tensor(tc, dtype=torch.float32))
            train_seqs.append((train_feats, train_qidxs, train_tcs))
            test_seqs.append((test_feats, test_qidxs, test_tcs))

        n_students = len(train_seqs)
        max_seq_cap = kwargs.get("max_seq_len", 600)
        for i in range(len(train_seqs)):
            feats, qidxs, tcs = train_seqs[i]
            if len(feats) > max_seq_cap:
                train_seqs[i] = (feats[:max_seq_cap], qidxs[:max_seq_cap], tcs[:max_seq_cap])
        for i in range(len(test_seqs)):
            feats, qidxs, tcs = test_seqs[i]
            if len(feats) > max_seq_cap:
                test_seqs[i] = (feats[:max_seq_cap], qidxs[:max_seq_cap], tcs[:max_seq_cap])

        max_train = max(len(s[0]) for s in train_seqs)
        max_test = max((len(s[0]) for s in test_seqs), default=0)

        if max_train == 0:
            raise ValueError(f"No train interactions for cutoff_week={split.cutoff_week}")
        if max_test == 0:
            raise ValueError(f"No test interactions for cutoff_week={split.cutoff_week}")

        # Pad and stack into tensors — keep on CPU, move to GPU in mini-batches
        def pad_and_stack(seqs, max_len):
            all_feats, all_qidxs, all_tcs, all_masks = [], [], [], []
            for feats, qidxs, tcs in seqs:
                n = len(feats)
                pad_n = max_len - n
                mask = [True] * n + [False] * pad_n
                if n > 0:
                    feats_t = torch.stack(feats)
                    tcs_t = torch.stack(tcs)
                else:
                    feats_t = torch.zeros(0, answer_dim)
                    tcs_t = torch.zeros(0, n_tc)
                if pad_n > 0:
                    feats_t = torch.cat([feats_t, torch.zeros(pad_n, answer_dim)])
                    qidxs = list(qidxs) + [0] * pad_n
                    tcs_t = torch.cat([tcs_t, torch.full((pad_n, n_tc), -1.0)])
                all_feats.append(feats_t)
                all_qidxs.append(qidxs)
                all_tcs.append(tcs_t)
                all_masks.append(mask)
            return (
                torch.stack(all_feats),  # CPU
                torch.tensor(all_qidxs, dtype=torch.long),  # CPU
                torch.stack(all_tcs),  # CPU
                torch.tensor(all_masks, dtype=torch.bool),  # CPU
            )

        train_feat_t, train_qidx_t, train_tc_t, train_mask_t = pad_and_stack(
            train_seqs, max_train
        )
        test_feat_t, test_qidx_t, test_tc_t, test_mask_t = pad_and_stack(
            test_seqs, max_test
        )
        question_static_t = torch.tensor(
            question_static, dtype=torch.float32
        ).to(device)

        print(f"    RSSM data: {n_students} students, "
              f"train_len={max_train}, test_len={max_test}, device={device}", flush=True)

        # Build model components separately for batched processing
        ans_encoder = AnswerEncoder(answer_dim, enc_dim, dropout=dropout).to(device)
        q_encoder = QuestionEncoder(
            n_questions,
            q_emb_dim=config.question_emb_dim,
            static_dim=config.question_static_dim,
            enc_dim=enc_dim,
            dropout=dropout,
        ).to(device)
        gru = nn.GRU(
            input_size=enc_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        ).to(device)
        drop = nn.Dropout(dropout).to(device)
        scorer = MultiModalScorer(
            hidden_dim=hidden_dim, question_enc_dim=enc_dim,
            n_testcases=n_tc, dropout=dropout,
        ).to(device)
        aux_predictor = AnswerFeaturePredictor(
            hidden_dim=hidden_dim, output_dim=answer_dim,
        ).to(device)

        params = (
            list(ans_encoder.parameters())
            + list(q_encoder.parameters())
            + list(gru.parameters())
            + list(scorer.parameters())
            + list(aux_predictor.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        # Mini-batch size for students (controls GPU memory usage)
        student_batch_size = kwargs.get("student_batch_size", 512)

        def forward_batch(feat_b, qidx_b, tc_b, mask_b, h_init, T, training=True):
            """Forward pass for a mini-batch of students.

            Args:
                feat_b: [B, T, answer_dim] on GPU
                qidx_b: [B, T] on GPU
                tc_b: [B, T, n_tc] on GPU
                mask_b: [B, T] on GPU
                h_init: [1, B, hidden_dim] on GPU
                T: sequence length
                training: if True, compute losses; if False, return scores

            Returns:
                If training: (bce_loss, aux_loss, n_valid_tc, n_valid_mask, h_final)
                If not training: (scores, h_final)  scores: [B, T, n_tc]
            """
            B = feat_b.shape[0]

            # Build shifted prev_features for teacher forcing
            prev_feats = torch.cat([
                torch.zeros(B, 1, answer_dim, device=device),
                feat_b[:, :-1]
            ], dim=1)

            # Encode answers and questions
            BT = B * T
            ans_enc = ans_encoder(prev_feats.reshape(BT, answer_dim)).reshape(B, T, enc_dim)
            q_ids_flat = qidx_b.reshape(BT)
            q_static_flat = question_static_t[q_ids_flat]
            q_enc = q_encoder(q_ids_flat, q_static_flat).reshape(B, T, enc_dim)

            # GRU forward
            gru_input = torch.cat([ans_enc, q_enc], dim=-1)
            hidden_out, h_final = gru(gru_input, h_init)
            hidden_out = drop(hidden_out)

            # Score
            scores = scorer(
                hidden_out.reshape(BT, hidden_dim),
                q_enc.reshape(BT, enc_dim),
            ).reshape(B, T, n_tc)

            if not training:
                return scores, h_final

            # BCE loss on valid testcases
            tc_valid = tc_b != -1
            n_valid_tc = tc_valid.sum().item()
            if n_valid_tc > 0:
                bce_loss = F.binary_cross_entropy(
                    scores[tc_valid], tc_b[tc_valid], reduction="sum"
                )
            else:
                bce_loss = torch.tensor(0.0, device=device)

            # Auxiliary feature prediction loss
            feat_hat = aux_predictor(
                hidden_out.reshape(BT, hidden_dim)
            ).reshape(B, T, answer_dim)
            valid = mask_b.unsqueeze(-1).expand_as(feat_hat)
            n_valid_mask = valid.sum().item()
            if n_valid_mask > 0:
                aux_loss = F.mse_loss(
                    feat_hat[valid], feat_b[valid], reduction="sum"
                )
            else:
                aux_loss = torch.tensor(0.0, device=device)

            return bce_loss, aux_loss, n_valid_tc, n_valid_mask, h_final

        # Training loop with mini-batches over students
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        n_batches = (n_students + student_batch_size - 1) // student_batch_size

        train_losses = []

        for epoch in range(epochs):
            ans_encoder.train()
            q_encoder.train()
            gru.train()
            scorer.train()
            aux_predictor.train()
            optimizer.zero_grad()

            total_bce = 0.0
            total_aux = 0.0
            total_tc_count = 0
            total_mask_count = 0

            for bi in range(n_batches):
                s_start = bi * student_batch_size
                s_end = min(s_start + student_batch_size, n_students)
                B = s_end - s_start

                feat_b = train_feat_t[s_start:s_end].to(device)
                qidx_b = train_qidx_t[s_start:s_end].to(device)
                tc_b = train_tc_t[s_start:s_end].to(device)
                mask_b = train_mask_t[s_start:s_end].to(device)
                h_init = torch.zeros(1, B, hidden_dim, device=device)

                bce_loss, aux_loss, n_valid_tc, n_valid_mask, _ = forward_batch(
                    feat_b, qidx_b, tc_b, mask_b, h_init, max_train, training=True
                )

                # Scale loss by batch fraction for gradient accumulation
                batch_loss = (bce_loss + aux_loss_weight * aux_loss) / n_batches
                batch_loss.backward()

                total_bce += bce_loss.item()
                total_aux += aux_loss.item()
                total_tc_count += n_valid_tc
                total_mask_count += n_valid_mask

            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            # Compute average losses
            avg_bce = total_bce / max(total_tc_count, 1)
            avg_aux = total_aux / max(total_mask_count, 1)
            avg_loss = avg_bce + aux_loss_weight * avg_aux

            train_losses.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                best_state = {
                    "ans_encoder": {k: v.cpu().clone() for k, v in ans_encoder.state_dict().items()},
                    "q_encoder": {k: v.cpu().clone() for k, v in q_encoder.state_dict().items()},
                    "gru": {k: v.cpu().clone() for k, v in gru.state_dict().items()},
                    "scorer": {k: v.cpu().clone() for k, v in scorer.state_dict().items()},
                    "aux": {k: v.cpu().clone() for k, v in aux_predictor.state_dict().items()},
                }

            if epoch - best_epoch >= patience:
                print(f"    RSSM early stop at epoch {epoch} "
                      f"(best={best_loss:.4f}@{best_epoch})", flush=True)
                break

            if epoch % 50 == 0:
                print(f"    RSSM epoch {epoch}: loss={avg_loss:.4f} "
                      f"(best={best_loss:.4f}@{best_epoch})", flush=True)

        # Restore best model
        if best_state is not None:
            ans_encoder.load_state_dict(best_state["ans_encoder"])
            q_encoder.load_state_dict(best_state["q_encoder"])
            gru.load_state_dict(best_state["gru"])
            scorer.load_state_dict(best_state["scorer"])
            aux_predictor.load_state_dict(best_state["aux"])

        # Extract question embedding norms as item parameter
        with torch.no_grad():
            q_emb_weights = q_encoder.q_embedding.weight.cpu().numpy()
            q_emb_norms = np.linalg.norm(q_emb_weights, axis=1)

        # Inference — mini-batched, no gradients
        ans_encoder.eval()
        q_encoder.eval()
        gru.eval()
        scorer.eval()

        all_test_scores = []
        with torch.no_grad():
            for bi in range(n_batches):
                s_start = bi * student_batch_size
                s_end = min(s_start + student_batch_size, n_students)
                B = s_end - s_start

                # Run through train data to get final hidden state
                train_feat_b = train_feat_t[s_start:s_end].to(device)
                train_qidx_b = train_qidx_t[s_start:s_end].to(device)
                h_init = torch.zeros(1, B, hidden_dim, device=device)

                # Forward through train sequence (no scoring needed)
                BT_train = B * max_train
                prev_feats = torch.cat([
                    torch.zeros(B, 1, answer_dim, device=device),
                    train_feat_b[:, :-1]
                ], dim=1)
                ans_enc = ans_encoder(prev_feats.reshape(BT_train, answer_dim)).reshape(B, max_train, enc_dim)
                q_ids_flat = train_qidx_b.reshape(BT_train)
                q_enc = q_encoder(q_ids_flat, question_static_t[q_ids_flat]).reshape(B, max_train, enc_dim)
                gru_input = torch.cat([ans_enc, q_enc], dim=-1)
                _, h_final = gru(gru_input, h_init)

                # Forward through test sequence
                test_feat_b = test_feat_t[s_start:s_end].to(device)
                test_qidx_b = test_qidx_t[s_start:s_end].to(device)

                scores, _ = forward_batch(
                    test_feat_b, test_qidx_b,
                    test_tc_t[s_start:s_end].to(device),
                    test_mask_t[s_start:s_end].to(device),
                    h_final, max_test, training=False
                )
                all_test_scores.append(scores.cpu())

        test_scores = torch.cat(all_test_scores, dim=0)  # [n_students, max_test, n_tc]

        # Extract predictions on valid test observations
        tc_mask = test_tc_t != -1
        y_true = test_tc_t[tc_mask].numpy()
        y_pred_prob = test_scores[tc_mask].numpy()

        # Build student and item indices for each valid prediction
        # tc_mask shape: [n_students, max_test, n_tc]
        valid_coords = tc_mask.nonzero(as_tuple=False)  # [N_valid, 3]
        test_student_indices = valid_coords[:, 0].numpy()
        # Map test sequence position to question index
        test_seq_pos = valid_coords[:, 1].numpy()
        test_item_indices = test_qidx_t[
            valid_coords[:, 0], valid_coords[:, 1]
        ].numpy()

        if len(y_true) == 0:
            raise ValueError(
                f"No valid test observations for cutoff_week={split.cutoff_week}"
            )

        return PredictionResult(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            student_indices=test_student_indices,
            item_indices=test_item_indices,
            losses={"train": train_losses},
            item_params={
                "question embedding norm": q_emb_norms,
            },
            model_state={
                "ans_encoder": ans_encoder.state_dict(),
                "q_encoder": q_encoder.state_dict(),
                "gru": gru.state_dict(),
                "scorer": scorer.state_dict(),
                "aux_predictor": aux_predictor.state_dict(),
                "hidden_dim": hidden_dim,
                "enc_dim": enc_dim,
                "n_questions": n_questions,
                "best_epoch": best_epoch,
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 30.0
