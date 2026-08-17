"""Full RSSM adapter with discrete latent variables for LLM embeddings.

Architecture (per the paper's full RSSM):
    Recurrent model:      h_t = GRUCell(concat(z_{t-1}, q_enc_t), h_{t-1})
    Representation model: z_t ~ q(z_t | h_t, e_t)     [posterior, training only]
    Transition predictor: z_t ~ p(z_t | h_t)           [prior, used at inference]
    Embedding predictor:  e_hat_t = MLP(z_t)
    Score predictor:      r_hat_t = Scorer(q_enc_t, h_t, z_t)

The scorer is trained on both the posterior z (grounded in the current answer)
and the prior z (the inference path, which predicts before seeing the answer),
so the distribution it sees at test time is also seen during training.

Loss = BCE(post) + prior_score_weight * BCE(prior) (normalized)
       + emb_weight * MSE(e_hat, sg(enc(e))) + beta * KL_balanced(posterior, prior)
where sg is stop-gradient: the embedding target is the detached encoder
output, so ans_encoder receives no gradient from the MSE term.

Embeddings loaded from data/embeddings/{course}/ (produced by featurize.py).
Pickle student/question indices are remapped into the current filtered
universe via the id maps in the pickle metadata; mismatched universes fail
loudly instead of silently shrinking or misattributing the eval set.
Optimized for A100 GPUs via TF32 matmul.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..embedding_alignment import (
    align_events,
    load_embeddings,
    map_question_indices,
    map_student_indices,
    resolve_emb_dir,
)
from ..temporal_split import TemporalSplit


def kl_categorical(p_probs, q_probs, eps=1e-8):
    """KL(p || q) for categorical distributions. Shape: [..., n_vars, n_classes] -> [..., n_vars]."""
    return (p_probs * (torch.log(p_probs + eps) - torch.log(q_probs + eps))).sum(dim=-1)


class RSSMAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "RSSM"

    @staticmethod
    def _setup_device(device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        **kwargs,
    ) -> PredictionResult:
        raise NotImplementedError(
            "RSSM uses student splits only. "
            "Use fit_and_predict_student_split() instead."
        )

    def fit_and_predict_student_split(
        self,
        data: UnifiedData,
        split,
        seed: int = 42,
        hidden_dim: int = 512,
        enc_dim: int = 512,
        n_latent_vars: int = 16,
        n_latent_classes: int = 16,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 500,
        patience: int = 50,
        grad_clip: float = 1.0,
        emb_weight: float = 0.1,
        beta: float = 0.5,
        alpha: float = 0.8,
        prior_score_weight: float = 1.0,
        pos_weight_mode: str = "none",
        strict_universe: bool = True,
        device: str = None,
        init_difficulty_bias: bool = True,
        difficulty_lr: float = 3e-3,
        student_batch_size: int = 64,
        emb_model_tag: str = "Qwen3-Embedding-8B-unfiltered",
        emb_dir: str = "",
        max_seq_len: int = 600,
        difficulty_reg: float = 0.0,
        use_cosine_lr: bool = True,
        cosine_t_max: int = 0,
        resume_checkpoint: str = None,
        eval_interval: int = 10,
        checkpoint_dir: str = None,
        checkpoint_name: str = "best_checkpoint.pt",
        on_checkpoint=None,
    ) -> PredictionResult:
        """Train on train students (all weeks), calibrate+predict test students."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = self._setup_device(device)

        from dynamic_models.rssm import (
            AnswerEncoder,
            EmbeddingQuestionEncoder as QuestionEncoder,
            PosteriorNet,
            PriorNet,
            Scorer,
        )

        latent_dim = n_latent_vars * n_latent_classes

        emb_data = load_embeddings(resolve_emb_dir(
            data.course_name, emb_dir, model_tag=emb_model_tag))
        answer_features = emb_data["answer_features"]
        question_embeddings = emb_data["question_embeddings"]
        testcase_scores = emb_data["testcase_scores"]
        attempt_idxs = emb_data["attempt_idxs"]
        metadata = emb_data["metadata"]
        if attempt_idxs is None:
            print("    WARNING: pickle lacks attempt_idxs; falling back to "
                  "replayed attempt counters (regenerate embeddings for exact "
                  "attempt alignment)", flush=True)

        answer_dim = int(metadata["answer_dim"])
        n_tc = int(metadata["n_testcases"])

        question_emb_raw = torch.tensor(
            question_embeddings, dtype=torch.float32
        )

        qi = data.question_infos
        qidx_to_week = {}
        for _, row in qi.drop_duplicates("qidx").iterrows():
            qidx_to_week[int(row["qidx"])] = int(row["week"])

        qidx_to_item_range = {}
        for qidx in qi["qidx"].unique():
            items = qi[qi["qidx"] == qidx].index.tolist()
            qidx_to_item_range[int(qidx)] = (min(items), len(items))

        # Map embedding indices to qidx via question_unittest_id. Interactions
        # on questions outside the current filtered universe are dropped (the
        # baselines never see them either); questions in the universe with no
        # embedding are a hard error so zero vectors never enter training.
        emb_to_qidx = map_question_indices(metadata, qi, strict=strict_universe)

        max_qidx = int(qi["qidx"].max())
        n_total_q = max_qidx + 1
        question_emb_t = torch.zeros(n_total_q, question_emb_raw.shape[1])
        for emb_idx, qidx in emb_to_qidx.items():
            question_emb_t[qidx] = question_emb_raw[emb_idx]
        question_emb_t = question_emb_t.to(device)

        # Remap pickle interactions into the current universe by id via the
        # shared alignment (drops out-of-universe rows, validates testcase
        # ordering). Raw fp32 features are kept per event; the fp16 matrix
        # align_to_universe packs is for lookup-style consumers.
        pidx_to_cur = map_student_indices(metadata, data, strict=strict_universe)
        kept, _ = align_events(
            emb_data, data, strict=strict_universe,
            emb_to_qidx=emb_to_qidx, pidx_to_cur=pidx_to_cur,
        )

        train_students_set = set(split.train_student_indices.tolist())
        test_students_set = set(split.test_student_indices.tolist())
        val_students_set = set(split.val_student_indices.tolist())

        per_student = {}
        for orig_i, si, qidx, att in kept:
            per_student.setdefault(si, []).append(
                (answer_features[orig_i], qidx, testcase_scores[orig_i], att)
            )

        missing = test_students_set - set(per_student)
        if missing:
            raise ValueError(
                f"Embedding pickle does not cover {len(missing)} test students "
                f"(e.g. {sorted(missing)[:5]}); metrics would not be comparable "
                f"across models. Regenerate embeddings with the matching filter "
                f"config."
            )

        train_seqs = []
        train_student_order = sorted(s for s in per_student if s in train_students_set)
        for si in train_student_order:
            feats, qidxs, tcs = [], [], []
            for feat, qidx, tc, _ in per_student[si]:
                feats.append(torch.tensor(feat, dtype=torch.float32))
                qidxs.append(qidx)
                tcs.append(torch.tensor(tc, dtype=torch.float32))
            train_seqs.append((feats, qidxs, tcs))

        def build_heldout_seqs(students_set):
            """Per held-out student: calibration rows (weeks <= cutoff) and
            prediction rows (post-cutoff), in chronological order."""
            order = sorted(s for s in per_student if s in students_set)
            calib, pred = [], []
            for si in order:
                c_feats, c_qidxs, c_tcs = [], [], []
                t_feats, t_qidxs, t_tcs, t_atts = [], [], [], []
                for feat, qidx, tc, att in per_student[si]:
                    week = qidx_to_week.get(qidx, 0)
                    ft = torch.tensor(feat, dtype=torch.float32)
                    tct = torch.tensor(tc, dtype=torch.float32)
                    if week <= split.train_week_cutoff:
                        c_feats.append(ft)
                        c_qidxs.append(qidx)
                        c_tcs.append(tct)
                    else:
                        t_feats.append(ft)
                        t_qidxs.append(qidx)
                        t_tcs.append(tct)
                        t_atts.append(att)
                calib.append((c_feats, c_qidxs, c_tcs))
                pred.append((t_feats, t_qidxs, t_tcs, t_atts))
            return order, calib, pred

        test_student_order, calib_seqs, test_seqs = build_heldout_seqs(test_students_set)
        val_student_order, val_calib_seqs, val_seqs = build_heldout_seqs(val_students_set)

        n_train = len(train_seqs)
        n_test = len(test_student_order)
        n_val = len(val_student_order)

        # Keep the most recent interactions when capping: the calibration
        # state feeds directly into test-week prediction, so recent context
        # matters most.
        max_seq_cap = max_seq_len
        for seqs in [train_seqs, calib_seqs, val_calib_seqs]:
            for i in range(len(seqs)):
                feats, qidxs, tcs = seqs[i]
                if len(feats) > max_seq_cap:
                    seqs[i] = (feats[-max_seq_cap:], qidxs[-max_seq_cap:], tcs[-max_seq_cap:])

        max_train_len = max(len(s[0]) for s in train_seqs) if train_seqs else 0
        max_calib_len = max((len(s[0]) for s in calib_seqs), default=0)
        max_val_calib_len = max((len(s[0]) for s in val_calib_seqs), default=0)

        if max_train_len == 0:
            raise ValueError("No training interactions found")

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
            train_seqs, max_train_len
        )

        print(f"    RSSM student split: {n_train} train, {n_val} val, "
              f"{n_test} test students, train_len={max_train_len}, "
              f"calib_len={max_calib_len}, device={device}",
              flush=True)

        ans_encoder = AnswerEncoder(answer_dim, enc_dim, dropout=dropout).to(device)
        q_encoder = QuestionEncoder(
            emb_dim=question_emb_raw.shape[1], enc_dim=enc_dim, dropout=dropout,
        ).to(device)
        gru_cell = nn.GRUCell(latent_dim + enc_dim, hidden_dim).to(device)
        scorer = Scorer(
            question_enc_dim=enc_dim, hidden_dim=hidden_dim,
            latent_dim=latent_dim, n_testcases=n_tc, dropout=dropout,
        ).to(device)
        n_questions = question_emb_t.shape[0]
        difficulty_bias = nn.Embedding(n_questions, n_tc).to(device)
        nn.init.zeros_(difficulty_bias.weight)
        # Start at the item-only solution (logit of train-students' pass rate)
        # so training refines item difficulty instead of spending its step
        # budget crawling toward it from zero.
        if init_difficulty_bias:
            corr_train = data.correctness_matrix[split.train_student_indices]
            with torch.no_grad():
                for qidx, (item_start, n_items_q) in qidx_to_item_range.items():
                    for tc_i in range(min(n_items_q, n_tc)):
                        vals = corr_train[:, item_start + tc_i, :]
                        valid = vals[vals != -1]
                        if len(valid) == 0:
                            continue
                        p = float(np.clip(valid.mean().item(), 0.01, 0.99))
                        difficulty_bias.weight[qidx, tc_i] = float(np.log(p / (1 - p)))
            print(f"    difficulty_bias initialized from train pass rates "
                  f"(std={difficulty_bias.weight.std().item():.3f})", flush=True)
        posterior_net = PosteriorNet(hidden_dim, enc_dim, n_latent_vars, n_latent_classes).to(device)
        prior_net = PriorNet(hidden_dim, n_latent_vars, n_latent_classes).to(device)
        emb_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, enc_dim),
        ).to(device)

        all_modules = [ans_encoder, q_encoder, gru_cell, scorer,
                       posterior_net, prior_net, emb_predictor, difficulty_bias]
        params = []
        for m in all_modules:
            params.extend(m.parameters())
        diff_reg = difficulty_reg
        # The answer pathway (encoder, posterior/prior, emb predictor) gets
        # weak gradients through the discrete latent; weight decay drove it
        # to zero when applied. Exempt it. difficulty_bias moves ~lr per Adam step, so
        # it needs its own larger lr to reach logit-scale values.
        decay_params = [p for m in [q_encoder, gru_cell, scorer]
                        for p in m.parameters()]
        no_decay_params = [p for m in [ans_encoder, posterior_net, prior_net,
                                       emb_predictor] for p in m.parameters()]
        # beta2=0.99: with only ~10 optimizer steps per epoch the default
        # 0.999 gives the second-moment estimate a ~100-epoch memory, turning
        # one bad step into a multi-epoch blow-up. eps stays at the default
        # 1e-8: the answer pathway's gradients are tiny, and eps=1e-5 crushed
        # their normalized updates so hard the posterior never trained (the
        # model froze at difficulty-only predictions).
        optimizer = torch.optim.Adam([
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": list(difficulty_bias.parameters()),
             "lr": difficulty_lr, "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.99), eps=1e-8)
        # Anneal over the realistic run length, not the nominal epoch budget:
        # with patience-based early stopping around epoch 100-150, T_max=epochs
        # kept the lr near its peak for the whole run, producing recurring
        # loss blow-ups.
        cosine_t_max = cosine_t_max or min(epochs, 80)
        # eta_min is a shared absolute value (lr * 0.1), so the base-lr groups
        # anneal to 10% of their base while the difficulty_bias group (its own
        # larger lr) anneals to ~1% of its own base. The retained step budget
        # lets a late argmax basin flip recover instead of becoming a
        # permanent staircase.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=lr * 0.1
        ) if use_cosine_lr else None

        if pos_weight_mode == "auto":
            tc_valid_all = train_tc_t != -1
            n_pos = (train_tc_t[tc_valid_all] == 1).sum().item()
            n_neg = (train_tc_t[tc_valid_all] == 0).sum().item()
            pw = min(n_neg / max(n_pos, 1), 10.0)
            pos_weight_t = torch.tensor(pw, device=device)
            print(f"    pos_weight={pw:.2f} (n_pos={n_pos}, n_neg={n_neg})", flush=True)
        elif pos_weight_mode != "none":
            raise ValueError(f"unknown pos_weight_mode: {pos_weight_mode}")
        else:
            pos_weight_t = None

        def forward_pass(feat_b, qidx_b, T):
            """DreamerV2 forward. Scorer logits are computed for both the
            posterior z (grounded in the current answer) and the prior z
            (the inference path, which predicts before seeing the answer)."""
            B = feat_b.shape[0]
            h_t = torch.zeros(B, hidden_dim, device=device)
            z_t = torch.zeros(B, latent_dim, device=device)

            all_logits_post, all_logits_prior = [], []
            all_post_probs, all_prior_probs = [], []
            all_e_hat, all_e_t = [], []

            for t in range(T):
                q_enc_t = q_encoder(question_emb_t[qidx_b[:, t]])
                h_t = gru_cell(torch.cat([z_t, q_enc_t], dim=-1), h_t)
                e_t = ans_encoder(feat_b[:, t])
                z_prior_t, prior_probs = prior_net(h_t)
                z_t, post_probs = posterior_net(h_t, e_t)
                e_hat_t = emb_predictor(z_t)
                d_bias = difficulty_bias(qidx_b[:, t])
                all_logits_post.append(scorer.forward_logits(q_enc_t, h_t, z_t) + d_bias)
                all_logits_prior.append(scorer.forward_logits(q_enc_t, h_t, z_prior_t) + d_bias)
                all_e_hat.append(e_hat_t)
                all_e_t.append(e_t.detach())
                all_post_probs.append(post_probs)
                all_prior_probs.append(prior_probs)

            return (
                torch.stack(all_logits_post, dim=1),
                torch.stack(all_logits_prior, dim=1),
                torch.stack(all_post_probs, dim=1),
                torch.stack(all_prior_probs, dim=1),
                torch.stack(all_e_hat, dim=1),
                torch.stack(all_e_t, dim=1),
            )

        def compute_losses(logits_post, logits_prior, post_probs, prior_probs,
                           e_hat, e_target, tc_b, mask_b):
            tc_valid = (tc_b != -1).float()
            n_valid_tc = tc_valid.sum().item()
            targets = tc_b.clamp(0, 1)
            bce_post = (F.binary_cross_entropy_with_logits(
                logits_post, targets, pos_weight=pos_weight_t, reduction="none"
            ) * tc_valid).sum() / max(n_valid_tc, 1)
            bce_prior = (F.binary_cross_entropy_with_logits(
                logits_prior, targets, pos_weight=pos_weight_t, reduction="none"
            ) * tc_valid).sum() / max(n_valid_tc, 1)
            bce_loss = (bce_post + prior_score_weight * bce_prior) / (1.0 + prior_score_weight)

            valid_enc = mask_b.unsqueeze(-1).expand_as(e_hat).float()
            n_valid_enc = valid_enc.sum().item()
            emb_loss = ((e_hat - e_target) ** 2 * valid_enc).sum() / max(n_valid_enc, 1)

            kl_mask = mask_b.unsqueeze(-1).float()
            n_valid_steps = mask_b.float().sum()
            kl_prior = kl_categorical(post_probs.detach(), prior_probs)
            kl_post = kl_categorical(post_probs, prior_probs.detach())
            kl_prior_masked = (kl_prior * kl_mask).sum(dim=-1)
            kl_post_masked = (kl_post * kl_mask).sum(dim=-1)
            kl_balanced = (
                alpha * kl_prior_masked.sum()
                + (1 - alpha) * kl_post_masked.sum()
            ) / n_valid_steps
            raw_kl = ((kl_prior_masked + kl_post_masked) / 2).sum().item() / n_valid_steps.item()

            return bce_loss, bce_post, bce_prior, emb_loss, kl_balanced, raw_kl

        # Precompute calibration tensors for the held-out sets. The mask is
        # load-bearing: without it every student shorter than the longest
        # calibration sequence gets phantom (question 0, zero answer) state
        # updates immediately before prediction.
        calib_data = None
        if max_calib_len > 0:
            c_feat_t, c_qidx_t, _, c_mask_t = pad_and_stack(calib_seqs, max_calib_len)
            calib_data = (c_feat_t, c_qidx_t, c_mask_t)
        val_calib_data = None
        if max_val_calib_len > 0:
            c_feat_t, c_qidx_t, _, c_mask_t = pad_and_stack(val_calib_seqs, max_val_calib_len)
            val_calib_data = (c_feat_t, c_qidx_t, c_mask_t)

        test_qidx_set = set(qi.loc[split.test_item_indices, "qidx"].astype(int))
        corr = data.correctness_matrix.numpy()
        test_items = set(split.test_item_indices.tolist())

        test_pos = {int(g): i for i, g in enumerate(split.test_student_indices.tolist())}
        val_pos = {int(g): i for i, g in enumerate(split.val_student_indices.tolist())}

        def run_eval(order, calib, max_cal, pred_seqs, pos):
            """Calibrate on held-out students (weeks <= cutoff), then predict
            their post-cutoff attempts. Returns (y_true, y_pred, indices)."""
            for m in all_modules:
                m.eval()

            n_held = len(order)
            h_finals = torch.zeros(n_held, hidden_dim, device=device)
            z_finals = torch.zeros(n_held, latent_dim, device=device)

            if calib is not None:
                c_feat, c_qidx, c_mask = calib
                calib_batches = (n_held + student_batch_size - 1) // student_batch_size
                with torch.no_grad():
                    for bi in range(calib_batches):
                        s_start = bi * student_batch_size
                        s_end = min(s_start + student_batch_size, n_held)
                        B = s_end - s_start
                        feat_b = c_feat[s_start:s_end].to(device)
                        qidx_b = c_qidx[s_start:s_end].to(device)
                        mask_b = c_mask[s_start:s_end].to(device)
                        h_t = torch.zeros(B, hidden_dim, device=device)
                        z_t = torch.zeros(B, latent_dim, device=device)
                        for t in range(max_cal):
                            m = mask_b[:, t].unsqueeze(-1)
                            if not m.any():
                                break
                            q_enc_t = q_encoder(question_emb_t[qidx_b[:, t]])
                            h_new = gru_cell(torch.cat([z_t, q_enc_t], dim=-1), h_t)
                            e_t = ans_encoder(feat_b[:, t])
                            z_new, _ = posterior_net(h_new, e_t)
                            h_t = torch.where(m, h_new, h_t)
                            z_t = torch.where(m, z_new, z_t)
                        h_finals[s_start:s_end] = h_t
                        z_finals[s_start:s_end] = z_t

            # Sequential prediction on post-cutoff submissions, in order.
            # At each step: predict BEFORE feeding the answer, then update GRU.
            y_true_list, y_pred_list = [], []
            s_idx_list, item_idx_list, attempt_idx_list = [], [], []

            with torch.no_grad():
                for s_local in range(n_held):
                    s_global = order[s_local]
                    h_t = h_finals[s_local:s_local+1]
                    z_t = z_finals[s_local:s_local+1]
                    t_feats, t_qidxs, t_tcs, t_atts = pred_seqs[s_local]

                    for step in range(len(t_feats)):
                        qidx = t_qidxs[step]
                        q_enc = q_encoder(question_emb_t[qidx].unsqueeze(0))

                        # GRU step; the state update runs for every step, but
                        # scoring is only needed on test questions.
                        h_step = gru_cell(torch.cat([z_t, q_enc], dim=-1), h_t)

                        if qidx in test_qidx_set:
                            # Predict BEFORE seeing the answer
                            d_bias = difficulty_bias.weight[qidx].unsqueeze(0)
                            z_pred, _ = prior_net(h_step)
                            logits = scorer.forward_logits(q_enc, h_step, z_pred) + d_bias
                            scores = torch.sigmoid(logits).cpu().numpy()[0]

                            # Record predictions for each test case in this question
                            attempt_num = t_atts[step]
                            item_start, n_items = qidx_to_item_range[qidx]
                            for tc_i in range(min(n_items, n_tc)):
                                item_idx = item_start + tc_i
                                if item_idx not in test_items:
                                    continue
                                c = corr[s_global, item_idx, attempt_num]
                                if c != -1:
                                    y_true_list.append(float(c))
                                    y_pred_list.append(float(np.clip(scores[tc_i], 1e-6, 1 - 1e-6)))
                                    s_idx_list.append(pos[int(s_global)])
                                    item_idx_list.append(item_idx)
                                    attempt_idx_list.append(attempt_num)

                        # Now feed the real answer to update GRU state
                        e_t = ans_encoder(t_feats[step].unsqueeze(0).to(device))
                        z_t, _ = posterior_net(h_step, e_t)
                        h_t = h_step

            # Eval-time tensors fragment the allocator over many epochs on
            # A100-40GB; release them before training resumes.
            if device == "cuda":
                torch.cuda.empty_cache()

            return (np.array(y_true_list), np.array(y_pred_list),
                    np.array(s_idx_list), np.array(item_idx_list), np.array(attempt_idx_list))

        module_names = ["ans_encoder", "q_encoder", "gru_cell", "scorer",
                        "posterior_net", "prior_net", "emb_predictor",
                        "difficulty_bias"]

        # Check for checkpoint to skip training
        if resume_checkpoint:
            print(f"    Loading checkpoint: {resume_checkpoint}", flush=True)
            ckpt = torch.load(resume_checkpoint, map_location=device)
            for name, mod in zip(module_names, all_modules):
                mod.load_state_dict(ckpt["best_state"][name])
            print(f"    Restored epoch {ckpt['epoch']}, val_AUC={ckpt['auc']:.4f}", flush=True)

            y_true, y_pred_prob, s_indices, item_indices, attempt_indices = run_eval(
                test_student_order, calib_data, max_calib_len, test_seqs, test_pos)
            print(f"    RSSM predict: {len(y_true)} test obs", flush=True)

            return PredictionResult(
                y_true=y_true,
                y_pred_prob=y_pred_prob,
                student_indices=s_indices,
                item_indices=item_indices,
                attempt_indices=attempt_indices,
                losses={"train": [], "val_aucs": [(ckpt["epoch"], ckpt["auc"])]},
                model_state={
                    "weights": ckpt["best_state"],
                    "hidden_dim": hidden_dim,
                    "enc_dim": enc_dim,
                    "n_latent_vars": n_latent_vars,
                    "n_latent_classes": n_latent_classes,
                    "best_epoch": ckpt["epoch"],
                },
            )

        # Phase 1: Train on train students. Checkpoint selection and early
        # stopping use the validation students only; the test set is touched
        # exactly once, after training.
        n_batches = (n_train + student_batch_size - 1) // student_batch_size
        best_auc = 0.0
        best_auc_epoch = 0
        best_state = None
        train_losses = []
        comp_hist = {"bce_obj": [], "bce_post": [], "bce_prior": [],
                     "emb": [], "kl": []}
        val_aucs = []
        warned_degenerate = {"empty": False, "one_class": False}
        if n_val == 0:
            print("    WARNING: no validation students (val_frac=0); no "
                  "checkpoint selection or early stopping, last-epoch weights "
                  "are used.", flush=True)

        def _save_state():
            return {
                name: {k: v.cpu().clone() for k, v in mod.state_dict().items()}
                for name, mod in zip(module_names, all_modules)
            }

        print(f"    Training: {n_batches} optimizer steps/epoch, "
              f"up to {epochs * n_batches} total", flush=True)

        for epoch in range(epochs):
            for m in all_modules:
                m.train()

            # Shuffle students each epoch (index-based; no full-tensor copy)
            perm = torch.randperm(n_train)

            total_bce, total_bce_prior, total_emb, total_kl = 0.0, 0.0, 0.0, 0.0
            total_bce_obj = 0.0
            total_kl_bal = 0.0

            for bi in range(n_batches):
                s_start = bi * student_batch_size
                s_end = min(s_start + student_batch_size, n_train)
                idx = perm[s_start:s_end]

                feat_b = train_feat_t[idx].to(device)
                qidx_b = train_qidx_t[idx].to(device)
                tc_b = train_tc_t[idx].to(device)
                mask_b = train_mask_t[idx].to(device)

                logits_post, logits_prior, post_probs, prior_probs, e_hat, e_target = forward_pass(
                    feat_b, qidx_b, max_train_len,
                )
                bce_loss, bce_post, bce_prior, emb_loss, kl_loss, raw_kl = compute_losses(
                    logits_post, logits_prior, post_probs, prior_probs,
                    e_hat, e_target, tc_b, mask_b,
                )
                diff_l2 = diff_reg * (difficulty_bias.weight ** 2).mean() if diff_reg > 0 else 0.0
                batch_loss = bce_loss + emb_weight * emb_loss + beta * kl_loss + diff_l2

                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()

                total_bce += bce_post.item()
                total_bce_prior += bce_prior.item()
                total_bce_obj += bce_loss.item()
                total_emb += emb_loss.item()
                total_kl += raw_kl
                total_kl_bal += kl_loss.item()

            # CosineAnnealingLR is periodic: stepping past T_max raises the lr
            # again, so hold at eta_min once the schedule completes.
            if scheduler and epoch < cosine_t_max:
                scheduler.step()

            avg_bce = total_bce / n_batches
            avg_bce_prior = total_bce_prior / n_batches
            # Plot the optimized objective: the scorer BCE term is
            # (bce_post + w*bce_prior)/(1+w) and the KL term is the balanced
            # KL actually optimized (raw symmetric KL stays in comp_hist).
            avg_loss = (total_bce_obj / n_batches
                        + emb_weight * (total_emb / n_batches)
                        + beta * (total_kl_bal / n_batches))
            train_losses.append(avg_loss)
            comp_hist["bce_obj"].append(total_bce_obj / n_batches)
            comp_hist["bce_post"].append(avg_bce)
            comp_hist["bce_prior"].append(avg_bce_prior)
            comp_hist["emb"].append(total_emb / n_batches)
            comp_hist["kl"].append(total_kl / n_batches)

            if epoch % eval_interval == 0:
                print(f"    RSSM epoch {epoch}: bce_post={avg_bce:.4f} "
                      f"bce_prior={avg_bce_prior:.4f} total={avg_loss:.4f}", flush=True)

            if epoch % eval_interval == 0 and epoch > 0 and n_val > 0:
                from sklearn.metrics import roc_auc_score
                yt, yp, _, _, _ = run_eval(
                    val_student_order, val_calib_data, max_val_calib_len,
                    val_seqs, val_pos)
                if len(yt) == 0:
                    if not warned_degenerate["empty"]:
                        warned_degenerate["empty"] = True
                        print("    WARNING: val eval returned 0 observations", flush=True)
                elif len(np.unique(yt)) < 2:
                    if not warned_degenerate["one_class"]:
                        warned_degenerate["one_class"] = True
                        print(f"    WARNING: val eval has only one class (n={len(yt)}, mean={yt.mean():.4f})", flush=True)
                if len(yt) > 0 and len(np.unique(yt)) == 2:
                    val_auc = roc_auc_score(yt, yp)
                    val_aucs.append((epoch, val_auc))
                    improved = val_auc > best_auc
                    if improved:
                        best_auc = val_auc
                        best_auc_epoch = epoch
                        best_state = _save_state()
                        if checkpoint_dir:
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            ckpt = {"best_state": best_state, "epoch": epoch, "auc": val_auc}
                            ckpt_path = os.path.join(
                                checkpoint_dir,
                                checkpoint_name)
                            torch.save(ckpt, ckpt_path)
                            if on_checkpoint:
                                on_checkpoint()
                            print(f"    Checkpoint saved: {ckpt_path}", flush=True)
                    print(f"    RSSM epoch {epoch}: val_AUC={val_auc:.4f}"
                          f" (best={best_auc:.4f}@{best_auc_epoch})"
                          f"{' *' if improved else ''}", flush=True)
                for mm in all_modules:
                    mm.train()

            if epoch - best_auc_epoch >= patience and best_auc_epoch > 0:
                print(f"    RSSM early stop at epoch {epoch} "
                      f"(best_val_AUC={best_auc:.4f}@{best_auc_epoch})", flush=True)
                break

        if n_val > 0 and best_state is None:
            print("    WARNING: validation students exist but no checkpoint was "
                  "ever selected (no usable val AUC); final-epoch weights are "
                  "used.", flush=True)

        # Restore best checkpoint (by val AUC)
        if best_state is not None:
            print(f"    Restoring best checkpoint: val_AUC={best_auc:.4f}@epoch {best_auc_epoch}", flush=True)
            for name, mod in zip(module_names, all_modules):
                mod.load_state_dict(best_state[name])

        # Final eval: the only time the test set is used
        y_true, y_pred_prob, s_indices, item_indices, attempt_indices = run_eval(
            test_student_order, calib_data, max_calib_len, test_seqs, test_pos)

        if len(y_true) == 0:
            raise ValueError("No valid test observations")

        print(f"    RSSM predict: {len(y_true)} test obs "
              f"({len(test_qidx_set)} questions, {n_test} students)", flush=True)

        return PredictionResult(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            student_indices=s_indices,
            item_indices=item_indices,
            attempt_indices=attempt_indices,
            losses={"train": train_losses, "val_aucs": val_aucs, **comp_hist},
            model_state={
                "weights": best_state,
                "hidden_dim": hidden_dim,
                "enc_dim": enc_dim,
                "n_latent_vars": n_latent_vars,
                "n_latent_classes": n_latent_classes,
                "best_epoch": best_auc_epoch,
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 45.0
