"""Full RSSM adapter with discrete latent variables and KL balancing.

Architecture:
    AnswerEncoder(answer_t)    -> e_t
    PosteriorNet(e_t)          -> z_t    [discrete categorical, straight-through]
    PriorNet(h_t)              -> z_hat_t
    QuestionEncoder(question)  -> enc_ques
    GRU(concat(z_{t-1}, enc_ques), h_{t-1}) -> h_t
    Scorer(h_t, enc_ques) -> P(pass) per testcase
    EmbeddingPredictor(z_t) -> e_hat_t

Loss = BCE + emb_weight * MSE(e_hat_t, e_t) + beta * KL_balanced(posterior, prior)
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


def kl_categorical(p_probs, q_probs, eps=1e-8):
    """KL(p || q) for categorical distributions. Shape: [..., n_vars, n_classes] -> [..., n_vars]."""
    return (p_probs * (torch.log(p_probs + eps) - torch.log(q_probs + eps))).sum(dim=-1)


class RSSMFullAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "RSSMFull"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        hidden_dim: int = 128,
        enc_dim: int = 64,
        n_latent_vars: int = 16,
        n_latent_classes: int = 16,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 500,
        patience: int = 100,
        grad_clip: float = 1.0,
        emb_weight: float = 0.1,
        beta: float = 0.1,
        alpha: float = 0.8,
        free_nats: float = 1.0,
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

        from dynamic_models.featurize import FeatureConfig
        from dynamic_models.rssm import (
            AnswerEncoder,
            AuxDecoder as EmbeddingPredictor,
            PosteriorNet,
            PriorNet,
            Scorer as MultiModalScorer,
            HandcraftedQuestionEncoder as QuestionEncoder,
        )

        config = FeatureConfig()
        latent_dim = n_latent_vars * n_latent_classes

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

        n_questions = int(metadata["n_questions"])
        n_tc = int(config.n_testcases)
        answer_dim = int(config.answer_dim)

        question_to_idx = metadata["question_to_idx"]
        local_idx_to_week = {}
        for qid, local_idx in question_to_idx.items():
            if qid in data.qid_to_week:
                local_idx_to_week[local_idx] = data.qid_to_week[qid]

        per_student = {}
        for feat, qidx, tc, si in zip(
            answer_features, question_idxs, testcase_scores, student_idxs
        ):
            if si not in per_student:
                per_student[si] = []
            week = local_idx_to_week.get(qidx, 0)
            is_train = week <= split.cutoff_week
            per_student[si].append((feat, qidx, tc, is_train))

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
                torch.stack(all_feats),
                torch.tensor(all_qidxs, dtype=torch.long),
                torch.stack(all_tcs),
                torch.tensor(all_masks, dtype=torch.bool),
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

        print(f"    RSSMFull data: {n_students} students, "
              f"train_len={max_train}, test_len={max_test}, device={device}", flush=True)

        # Build model components
        # GRU backbone: identical to ablated RSSM (dense answer enc + question enc)
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
        # Auxiliary: same as ablated
        aux_predictor = EmbeddingPredictor(
            hidden_dim=hidden_dim, output_dim=answer_dim,
        ).to(device)
        # Discrete latent: posterior q(z|h,e), prior p(z|h)
        posterior_net = PosteriorNet(hidden_dim, enc_dim, n_latent_vars, n_latent_classes).to(device)
        prior_net = PriorNet(hidden_dim, n_latent_vars, n_latent_classes, hidden_dim).to(device)
        emb_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, enc_dim),
        ).to(device)

        params = (
            list(ans_encoder.parameters())
            + list(q_encoder.parameters())
            + list(gru.parameters())
            + list(scorer.parameters())
            + list(aux_predictor.parameters())
            + list(posterior_net.parameters())
            + list(prior_net.parameters())
            + list(emb_predictor.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        student_batch_size = kwargs.get("student_batch_size", 512)

        def forward_batch(feat_b, qidx_b, tc_b, mask_b, h_init, T, training=True):
            """GRU backbone identical to ablated RSSM. Discrete latent is a side computation."""
            B = feat_b.shape[0]
            BT = B * T

            # === GRU backbone (same as ablated RSSM) ===
            # Shifted prev_features for teacher forcing
            prev_feats = torch.cat([
                torch.zeros(B, 1, answer_dim, device=device),
                feat_b[:, :-1]
            ], dim=1)

            ans_enc = ans_encoder(prev_feats.reshape(BT, answer_dim)).reshape(B, T, enc_dim)
            q_ids_flat = qidx_b.reshape(BT)
            q_static_flat = question_static_t[q_ids_flat]
            q_enc = q_encoder(q_ids_flat, q_static_flat).reshape(B, T, enc_dim)

            gru_input = torch.cat([ans_enc, q_enc], dim=-1)
            hidden_out, h_final = gru(gru_input, h_init)  # [B, T, hidden_dim]
            hidden_dropped = drop(hidden_out)

            # Score (same as ablated)
            scores = scorer(
                hidden_dropped.reshape(BT, hidden_dim),
                q_enc.reshape(BT, enc_dim),
            ).reshape(B, T, n_tc)

            if not training:
                return scores, h_final

            # === Losses ===
            # BCE loss (same as ablated)
            tc_valid = (tc_b != -1).float()
            n_valid_tc = tc_valid.sum().item()
            if n_valid_tc > 0:
                bce_per_elem = F.binary_cross_entropy(
                    scores, tc_b.clamp(0, 1), reduction="none"
                )
                bce_loss = (bce_per_elem * tc_valid).sum() / n_valid_tc
            else:
                bce_loss = torch.tensor(0.0, device=device)

            # Aux loss (same as ablated)
            feat_hat = aux_predictor(
                hidden_dropped.reshape(BT, hidden_dim)
            ).reshape(B, T, answer_dim)
            valid_expand = mask_b.unsqueeze(-1).expand_as(feat_hat).float()
            n_valid_mask = valid_expand.sum().item()
            if n_valid_mask > 0:
                aux_loss = ((feat_hat - feat_b) ** 2 * valid_expand).sum() / n_valid_mask
            else:
                aux_loss = torch.tensor(0.0, device=device)

            # === Discrete latent (side computation from h_t) ===
            h_clean = hidden_out.reshape(BT, hidden_dim)
            e_t = ans_encoder(feat_b.reshape(BT, answer_dim))  # current answer encoding

            # Posterior q(z_t | h_t, e_t), Prior p(z_t | h_t)
            z_post, post_probs = posterior_net(h_clean, e_t)
            _, prior_probs = prior_net(h_clean)
            post_probs_seq = post_probs.reshape(B, T, n_latent_vars, n_latent_classes)
            prior_probs_seq = prior_probs.reshape(B, T, n_latent_vars, n_latent_classes)

            # Embedding prediction: z_t -> e_hat_t
            z_flat_all = z_post.reshape(BT, latent_dim)
            e_hat = emb_predictor(z_flat_all).reshape(B, T, enc_dim)
            e_target = e_t.reshape(B, T, enc_dim).detach()
            valid_enc = mask_b.unsqueeze(-1).expand_as(e_hat).float()
            n_valid_enc = valid_enc.sum().item()
            if n_valid_enc > 0:
                emb_loss = ((e_hat - e_target) ** 2 * valid_enc).sum() / n_valid_enc
            else:
                emb_loss = torch.tensor(0.0, device=device)

            # KL balanced loss (mean over valid timesteps)
            kl_mask = mask_b.unsqueeze(-1).float()
            n_valid_steps = mask_b.float().sum()

            kl_prior = kl_categorical(post_probs_seq.detach(), prior_probs_seq)
            kl_post = kl_categorical(post_probs_seq, prior_probs_seq.detach())

            kl_prior_sum = (kl_prior * kl_mask).sum(dim=-1)
            kl_post_sum = (kl_post * kl_mask).sum(dim=-1)

            kl_prior_clamped = torch.clamp(kl_prior_sum, min=free_nats)
            kl_post_clamped = torch.clamp(kl_post_sum, min=free_nats)

            kl_balanced = (alpha * kl_prior_clamped.sum() + (1 - alpha) * kl_post_clamped.sum()) / n_valid_steps
            raw_kl = ((kl_prior_sum + kl_post_sum) / 2).sum().item() / max(n_valid_steps.item(), 1)

            return bce_loss, aux_loss, emb_loss, kl_balanced, raw_kl, h_final

        # No separate forward_inference needed — GRU backbone is the same as
        # ablated RSSM, so we use forward_batch with training=False.

        # Training loop
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
            posterior_net.train()
            prior_net.train()
            emb_predictor.train()
            optimizer.zero_grad()

            # KL warmup: linearly increase beta from 0 to target over first 100 epochs
            kl_warmup_epochs = 100
            beta_t = beta * min(1.0, epoch / max(kl_warmup_epochs, 1))

            aux_loss_weight = emb_weight  # same weight for aux as ablated
            total_bce = 0.0
            total_aux = 0.0
            total_emb = 0.0
            total_kl = 0.0

            for bi in range(n_batches):
                s_start = bi * student_batch_size
                s_end = min(s_start + student_batch_size, n_students)
                B = s_end - s_start

                feat_b = train_feat_t[s_start:s_end].to(device)
                qidx_b = train_qidx_t[s_start:s_end].to(device)
                tc_b = train_tc_t[s_start:s_end].to(device)
                mask_b = train_mask_t[s_start:s_end].to(device)
                h_init = torch.zeros(1, B, hidden_dim, device=device)

                bce_loss, aux_loss, emb_loss, kl_loss, raw_kl, _ = forward_batch(
                    feat_b, qidx_b, tc_b, mask_b, h_init, max_train, training=True
                )

                batch_loss = (
                    bce_loss + aux_loss_weight * aux_loss
                    + emb_weight * emb_loss + beta_t * kl_loss
                ) / n_batches
                batch_loss.backward()

                total_bce += bce_loss.item()
                total_aux += aux_loss.item()
                total_emb += emb_loss.item()
                total_kl += raw_kl

            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            avg_bce = total_bce / n_batches
            avg_aux = total_aux / n_batches
            avg_emb = total_emb / n_batches
            avg_kl = total_kl / n_batches
            avg_loss = avg_bce + aux_loss_weight * avg_aux + emb_weight * avg_emb + beta_t * avg_kl

            train_losses.append(avg_loss)

            # Early stopping on BCE (primary predictive loss)
            if avg_bce < best_loss:
                best_loss = avg_bce
                best_epoch = epoch
                best_state = {
                    "ans_encoder": {k: v.cpu().clone() for k, v in ans_encoder.state_dict().items()},
                    "q_encoder": {k: v.cpu().clone() for k, v in q_encoder.state_dict().items()},
                    "gru": {k: v.cpu().clone() for k, v in gru.state_dict().items()},
                    "scorer": {k: v.cpu().clone() for k, v in scorer.state_dict().items()},
                    "aux_predictor": {k: v.cpu().clone() for k, v in aux_predictor.state_dict().items()},
                    "posterior_net": {k: v.cpu().clone() for k, v in posterior_net.state_dict().items()},
                    "prior_net": {k: v.cpu().clone() for k, v in prior_net.state_dict().items()},
                    "emb_predictor": {k: v.cpu().clone() for k, v in emb_predictor.state_dict().items()},
                }

            if epoch - best_epoch >= patience:
                print(f"    RSSMFull early stop at epoch {epoch} "
                      f"(best={best_loss:.4f}@{best_epoch})", flush=True)
                break

            if epoch % 50 == 0:
                print(f"    RSSMFull epoch {epoch}: bce={avg_bce:.4f} "
                      f"aux={avg_aux:.4f} emb={avg_emb:.4f} kl={avg_kl:.4f} "
                      f"total={avg_loss:.4f} (best={best_loss:.4f}@{best_epoch})", flush=True)

        # Restore best model
        if best_state is not None:
            ans_encoder.load_state_dict(best_state["ans_encoder"])
            q_encoder.load_state_dict(best_state["q_encoder"])
            gru.load_state_dict(best_state["gru"])
            scorer.load_state_dict(best_state["scorer"])
            aux_predictor.load_state_dict(best_state["aux_predictor"])
            posterior_net.load_state_dict(best_state["posterior_net"])
            prior_net.load_state_dict(best_state["prior_net"])
            emb_predictor.load_state_dict(best_state["emb_predictor"])

        # Extract question embedding norms as item parameter
        with torch.no_grad():
            q_emb_weights = q_encoder.q_embedding.weight.cpu().numpy()
            q_emb_norms = np.linalg.norm(q_emb_weights, axis=1)

        # Inference — identical to ablated RSSM (GRU backbone doesn't use z)
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

        test_scores = torch.cat(all_test_scores, dim=0)

        tc_mask = test_tc_t != -1
        y_true = test_tc_t[tc_mask].numpy()
        y_pred_prob = test_scores[tc_mask].numpy()

        valid_coords = tc_mask.nonzero(as_tuple=False)
        test_student_indices = valid_coords[:, 0].numpy()
        test_seq_pos = valid_coords[:, 1].numpy()
        test_tc_indices = valid_coords[:, 2].numpy()
        test_item_indices = test_qidx_t[
            valid_coords[:, 0], valid_coords[:, 1]
        ].numpy()

        test_attempt_indices = np.zeros(len(test_student_indices), dtype=int)
        prev_key = None
        attempt = 0
        prev_seq_pos = -1
        for i in range(len(test_student_indices)):
            key = (test_student_indices[i], test_item_indices[i])
            seq_pos = test_seq_pos[i]
            if key != prev_key:
                attempt = 0
                prev_key = key
                prev_seq_pos = seq_pos
            elif seq_pos != prev_seq_pos:
                attempt += 1
                prev_seq_pos = seq_pos
            test_attempt_indices[i] = attempt

        if len(y_true) == 0:
            raise ValueError(
                f"No valid test observations for cutoff_week={split.cutoff_week}"
            )

        return PredictionResult(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            student_indices=test_student_indices,
            item_indices=test_item_indices,
            testcase_indices=test_tc_indices,
            attempt_indices=test_attempt_indices,
            losses={"train": train_losses},
            item_params={
                "question embedding norm": q_emb_norms,
            },
            model_state={
                "ans_encoder": ans_encoder.state_dict(),
                "posterior_net": posterior_net.state_dict(),
                "prior_net": prior_net.state_dict(),
                "q_encoder": q_encoder.state_dict(),
                "gru": gru.state_dict(),
                "scorer": scorer.state_dict(),
                "emb_predictor": emb_predictor.state_dict(),
                "hidden_dim": hidden_dim,
                "enc_dim": enc_dim,
                "n_latent_vars": n_latent_vars,
                "n_latent_classes": n_latent_classes,
                "n_questions": n_questions,
                "best_epoch": best_epoch,
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 30.0
