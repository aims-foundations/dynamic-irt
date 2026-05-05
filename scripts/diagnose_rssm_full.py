"""Diagnose full RSSM training dynamics.

Checks:
  1. Loss term scales (sum vs mean, relative magnitudes)
  2. Gradient norms per component
  3. Posterior/prior entropy over training
  4. KL decomposition per variable

Usage:
    python scripts/diagnose_rssm_full.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def kl_categorical(p_probs, q_probs, eps=1e-8):
    return (p_probs * (torch.log(p_probs + eps) - torch.log(q_probs + eps))).sum(dim=-1)


def entropy_categorical(probs, eps=1e-8):
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def main():
    import pickle
    from dynamic_models.featurize import FeatureConfig
    from dynamic_models.rssm import (
        AnswerEncoder, AuxDecoder, PosteriorNet, PriorNet,
        Scorer, HandcraftedQuestionEncoder,
    )
    from dynamic_models.temporal_eval.data_loader import load_unified_data
    from dynamic_models.temporal_eval.temporal_split import generate_temporal_splits

    data = load_unified_data("dsa_hk231")
    splits = generate_temporal_splits(data.item_week, [1])
    split = splits[0]

    config = FeatureConfig()
    hidden_dim = 128
    enc_dim = 64
    n_latent_vars = 16
    n_latent_classes = 16
    latent_dim = n_latent_vars * n_latent_classes
    dropout = 0.2
    n_tc = int(config.n_testcases)
    answer_dim = int(config.answer_dim)
    beta = 0.1
    alpha = 0.8
    free_nats = 1.0
    emb_weight = 0.1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data (abbreviated — just load one batch)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data", "multimodal", data.course_name)

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
    for si in sorted(per_student.keys()):
        interactions = per_student[si]
        feats, qidxs, tcs = [], [], []
        for feat, qidx, tc, is_train in interactions:
            if is_train:
                feats.append(torch.tensor(feat, dtype=torch.float32))
                qidxs.append(qidx)
                tcs.append(torch.tensor(tc, dtype=torch.float32))
        train_seqs.append((feats, qidxs, tcs))

    # Cap sequences
    max_seq = 600
    for i in range(len(train_seqs)):
        f, q, t = train_seqs[i]
        if len(f) > max_seq:
            train_seqs[i] = (f[:max_seq], q[:max_seq], t[:max_seq])

    max_train = max(len(s[0]) for s in train_seqs)
    n_students = len(train_seqs)

    # Pad
    all_feats, all_qidxs, all_tcs, all_masks = [], [], [], []
    for feats, qidxs, tcs in train_seqs:
        n = len(feats)
        pad_n = max_train - n
        mask = [True] * n + [False] * pad_n
        feats_t = torch.stack(feats) if n > 0 else torch.zeros(0, answer_dim)
        tcs_t = torch.stack(tcs) if n > 0 else torch.zeros(0, n_tc)
        if pad_n > 0:
            feats_t = torch.cat([feats_t, torch.zeros(pad_n, answer_dim)])
            qidxs = list(qidxs) + [0] * pad_n
            tcs_t = torch.cat([tcs_t, torch.full((pad_n, n_tc), -1.0)])
        all_feats.append(feats_t)
        all_qidxs.append(qidxs)
        all_tcs.append(tcs_t)
        all_masks.append(mask)

    feat_t = torch.stack(all_feats)
    qidx_t = torch.tensor(all_qidxs, dtype=torch.long)
    tc_t = torch.stack(all_tcs)
    mask_t = torch.tensor(all_masks, dtype=torch.bool)
    question_static_t = torch.tensor(question_static, dtype=torch.float32).to(device)

    T = max_train

    # Build model
    ans_encoder = AnswerEncoder(answer_dim, enc_dim, dropout=dropout).to(device)
    posterior_net = PosteriorNet(enc_dim, n_latent_vars, n_latent_classes, hidden_dim).to(device)
    prior_net = PriorNet(hidden_dim, n_latent_vars, n_latent_classes, hidden_dim).to(device)
    q_encoder = HandcraftedQuestionEncoder(
        n_questions, config.question_emb_dim, config.question_static_dim,
        enc_dim, dropout,
    ).to(device)
    gru = nn.GRU(latent_dim + enc_dim, hidden_dim, 1, batch_first=True).to(device)
    drop = nn.Dropout(dropout)
    scorer = Scorer(hidden_dim, enc_dim, n_tc, dropout).to(device)
    emb_predictor = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, enc_dim),
    ).to(device)

    all_params = (
        list(ans_encoder.parameters()) + list(posterior_net.parameters())
        + list(prior_net.parameters()) + list(q_encoder.parameters())
        + list(gru.parameters()) + list(scorer.parameters())
        + list(emb_predictor.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=1e-3, weight_decay=1e-4)

    components = {
        "ans_encoder": ans_encoder, "posterior_net": posterior_net,
        "prior_net": prior_net, "q_encoder": q_encoder,
        "gru": gru, "scorer": scorer, "emb_predictor": emb_predictor,
    }

    print("=" * 70)
    print("DIAGNOSTIC: Full RSSM Training Dynamics")
    print("=" * 70)
    print(f"Students={n_students}, T={T}, device={device}")
    print(f"latent_dim={latent_dim} ({n_latent_vars}x{n_latent_classes})")
    print()

    # Quick count of elements per loss term
    valid_tc = (tc_t != -1).float().sum().item()
    valid_mask = mask_t.float().sum().item()
    print(f"Valid testcase elements (for BCE): {valid_tc:.0f}")
    print(f"Valid timesteps (for KL): {valid_mask:.0f}")
    print(f"Valid embedding elements (for emb): {valid_mask * enc_dim:.0f}")
    print(f"Ratio BCE/KL elements: {valid_tc / valid_mask:.1f}")
    print()

    print(f"{'Epoch':>5} | {'BCE':>8} {'emb':>8} {'KL':>8} "
          f"| {'postH':>6} {'priorH':>6} "
          f"| {'g_ans':>7} {'g_post':>7} {'g_prior':>7} {'g_gru':>7} {'g_score':>7}")
    print("-" * 110)

    for epoch in range(200):
        ans_encoder.train(); posterior_net.train(); prior_net.train()
        q_encoder.train(); gru.train(); scorer.train()
        optimizer.zero_grad()

        feat_b = feat_t.to(device)
        qidx_b = qidx_t.to(device)
        tc_b = tc_t.to(device)
        mask_b = mask_t.to(device)
        B = n_students
        BT = B * T
        h_init = torch.zeros(1, B, hidden_dim, device=device)

        # Forward
        ans_flat = ans_encoder(feat_b.reshape(BT, answer_dim))
        e_all = ans_flat.reshape(B, T, enc_dim)
        z_post, post_probs = posterior_net(ans_flat)
        z_post_seq = z_post.reshape(B, T, n_latent_vars, n_latent_classes)
        post_probs_seq = post_probs.reshape(B, T, n_latent_vars, n_latent_classes)

        z_flat = z_post_seq.reshape(B, T, latent_dim)
        z_prev = torch.cat([torch.zeros(B, 1, latent_dim, device=device), z_flat[:, :-1]], dim=1)

        q_ids = qidx_b.reshape(BT)
        q_enc = q_encoder(q_ids, question_static_t[q_ids]).reshape(B, T, enc_dim)

        gru_input = torch.cat([z_prev, q_enc], dim=-1)
        hidden_out, _ = gru(gru_input, h_init)
        hidden_out_d = drop(hidden_out)

        _, prior_probs_raw = prior_net(hidden_out_d.reshape(BT, hidden_dim))
        prior_probs_seq = prior_probs_raw.reshape(B, T, n_latent_vars, n_latent_classes)

        scores = scorer(hidden_out_d.reshape(BT, hidden_dim), q_enc.reshape(BT, enc_dim)).reshape(B, T, n_tc)

        # Losses
        tc_valid = (tc_b != -1).float()
        bce_loss = (F.binary_cross_entropy(scores, tc_b.clamp(0, 1), reduction="none") * tc_valid).sum()

        z_flat_all = z_post_seq.reshape(BT, latent_dim)
        e_hat = emb_predictor(z_flat_all).reshape(B, T, enc_dim)
        vmask = mask_b.unsqueeze(-1).expand_as(e_all).float()
        emb_loss = ((e_hat - e_all.detach()) ** 2 * vmask).sum()

        kl_mask = mask_b.unsqueeze(-1).float()
        kl_fwd = kl_categorical(post_probs_seq.detach(), prior_probs_seq)
        kl_rev = kl_categorical(post_probs_seq, prior_probs_seq.detach())
        kl_fwd_sum = (kl_fwd * kl_mask).sum(dim=-1)
        kl_rev_sum = (kl_rev * kl_mask).sum(dim=-1)
        kl_fwd_c = torch.clamp(kl_fwd_sum, min=free_nats)
        kl_rev_c = torch.clamp(kl_rev_sum, min=free_nats)

        beta_t = beta * min(1.0, epoch / 100)
        kl_balanced = alpha * kl_fwd_c.sum() + (1 - alpha) * kl_rev_c.sum()

        total = bce_loss + emb_weight * emb_loss + beta_t * kl_balanced
        total.backward()

        # Gradient norms
        def grad_norm(module):
            norms = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
            return sum(n ** 2 for n in norms) ** 0.5 if norms else 0.0

        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        # Metrics
        avg_bce = bce_loss.item() / max(valid_tc, 1)
        avg_emb = emb_loss.item() / max(valid_mask * enc_dim, 1)
        raw_kl = ((kl_fwd_sum + kl_rev_sum) / 2).sum().item() / max(valid_mask, 1)

        # Entropy
        post_entropy = entropy_categorical(post_probs_seq).mean().item()
        prior_entropy = entropy_categorical(prior_probs_seq).mean().item()

        if epoch % 10 == 0:
            gn = {k: grad_norm(v) for k, v in components.items()}
            print(f"{epoch:5d} | {avg_bce:8.4f} {avg_emb:8.4f} {raw_kl:8.4f} "
                  f"| {post_entropy:6.3f} {prior_entropy:6.3f} "
                  f"| {gn['ans_encoder']:7.1f} {gn['posterior_net']:7.1f} "
                  f"{gn['prior_net']:7.1f} {gn['gru']:7.1f} {gn['scorer']:7.1f}")

    # Final per-variable KL breakdown
    with torch.no_grad():
        ans_flat = ans_encoder(feat_b.reshape(BT, answer_dim))
        _, post_p = posterior_net(ans_flat)
        post_p = post_p.reshape(B, T, n_latent_vars, n_latent_classes)

        hidden_out, _ = gru(
            torch.cat([
                torch.cat([torch.zeros(B, 1, latent_dim, device=device),
                           posterior_net(ans_flat)[0].reshape(B, T, latent_dim)[:, :-1]], dim=1),
                q_encoder(qidx_b.reshape(BT), question_static_t[qidx_b.reshape(BT)]).reshape(B, T, enc_dim)
            ], dim=-1),
            torch.zeros(1, B, hidden_dim, device=device)
        )
        _, prior_p = prior_net(hidden_out.reshape(BT, hidden_dim))
        prior_p = prior_p.reshape(B, T, n_latent_vars, n_latent_classes)

        per_var_kl = kl_categorical(post_p, prior_p)  # [B, T, n_vars]
        mask_expand = mask_b.unsqueeze(-1).float()
        per_var_avg = (per_var_kl * mask_expand).sum(dim=(0, 1)) / mask_b.float().sum()

        print()
        print("Per-variable KL at end of training:")
        for v in range(n_latent_vars):
            print(f"  var {v:2d}: KL={per_var_avg[v].item():.4f}")

        post_ent_per_var = entropy_categorical(post_p).mean(dim=(0, 1))
        prior_ent_per_var = entropy_categorical(prior_p).mean(dim=(0, 1))
        print()
        print("Per-variable entropy:")
        print(f"  {'var':>4} {'post_H':>8} {'prior_H':>8} {'max_H':>8}")
        max_h = np.log(n_latent_classes)
        for v in range(n_latent_vars):
            print(f"  {v:4d} {post_ent_per_var[v].item():8.4f} "
                  f"{prior_ent_per_var[v].item():8.4f} {max_h:8.4f}")


if __name__ == "__main__":
    main()
