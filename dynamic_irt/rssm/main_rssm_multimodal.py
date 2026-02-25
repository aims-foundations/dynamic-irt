"""Training script for Multi-Modal RSSM.

Mirrors main_rssm.py structure (train/test, teacher forcing, same metrics)
but uses structured multi-modal features instead of text embeddings.

Usage:
    python main_rssm_multimodal.py --cls dsa_hk231 --config full --epochs 10000
    python main_rssm_multimodal.py --cls dsa_hk231 --config performance_only --epochs 10000
    python main_rssm_multimodal.py --cls dsa_hk231 --test_only 1
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from tueplots import bundles, figsizes

from feature_config import CONFIGS, FeatureConfig
from models_multimodal import (
    AnswerFeaturePredictor,
    MultiModalRSSM,
    MultiModalScorer,
)
from utils_multimodal import load_multimodal_data

# Match project-wide style (elo.py, cirt.py, dynamic_irt.py)
plt.rcParams.update(bundles.aaai2024())
COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


# ---------------------------------------------------------------------------
# Plotting (matches elo.py / cirt.py / dynamic_irt.py style)
# ---------------------------------------------------------------------------

def plot_losses(train_losses, result_dir):
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    ax.plot(train_losses, label="Train", alpha=0.7, color=COLORS[0])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Multi-Modal RSSM Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(result_dir, "losses.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_test_metrics(eval_epochs, test_accs, test_accs_rollout, result_dir):
    """Plot test accuracy over training (teacher-forcing vs rollout)."""
    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    ax.plot(eval_epochs, test_accs, label="Teacher forcing", alpha=0.7,
            color=COLORS[0], marker="o", markersize=3)
    ax.plot(eval_epochs, test_accs_rollout, label="Rollout (predicted features)",
            alpha=0.7, color=COLORS[1], marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Multi-Modal RSSM — Test Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(result_dir, "test_accuracy.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_question_difficulty(question_static, result_dir):
    """Plot histogram of learned question difficulty (matches param hist style)."""
    difficulties = question_static[:, 0].cpu().numpy()
    lo, hi = np.percentile(difficulties, (1, 99))
    clipped = difficulties[(difficulties >= lo) & (difficulties <= hi)]

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_half()["figure.figsize"])
    ax.hist(clipped, bins=30, density=True, alpha=0.3, color=COLORS[0])
    sns.kdeplot(clipped, color=COLORS[0], linewidth=1.5, bw_adjust=0.5, ax=ax)
    ax.set_xlabel("Empirical Difficulty")
    ax.set_ylabel("Density")
    ax.set_title("Question Difficulty Distribution")
    save_path = os.path.join(result_dir, "question_difficulty.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_question_embeddings(rssm, n_questions, result_dir):
    """Plot learned question embedding norms (proxy for question importance)."""
    with torch.no_grad():
        emb_weights = rssm._ques_encoder.q_embedding.weight.cpu().numpy()
    norms = np.linalg.norm(emb_weights, axis=1)

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_half()["figure.figsize"])
    ax.hist(norms, bins=30, density=True, alpha=0.3, color=COLORS[2])
    sns.kdeplot(norms, color=COLORS[2], linewidth=1.5, bw_adjust=0.5, ax=ax)
    ax.set_xlabel("Embedding Norm")
    ax.set_ylabel("Density")
    ax.set_title("Learned Question Embedding Norms")
    save_path = os.path.join(result_dir, "question_embedding_norms.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_final_metrics_bar(train_res, test_res, result_dir):
    """Bar chart comparing train vs test across all metrics."""
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    labels = ["Accuracy", "F1", "Precision", "Recall", "AUC"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    train_vals = [train_res.get(m, 0) for m in metrics]
    test_vals = [test_res.get(m, 0) for m in metrics]

    ax.bar(x - width / 2, train_vals, width, label="Train", color=COLORS[0], alpha=0.8)
    ax.bar(x + width / 2, test_vals, width, label="Test", color=COLORS[1], alpha=0.8)

    for i, (tv, ttv) in enumerate(zip(train_vals, test_vals)):
        ax.text(i - width / 2, tv + 0.01, f"{tv:.3f}", ha="center", fontsize=6)
        ax.text(i + width / 2, ttv + 0.01, f"{ttv:.3f}", ha="center", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Multi-Modal RSSM — Train vs Test Metrics")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save_path = os.path.join(result_dir, "fit_metrics_bar.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def train(
    args,
    config,
    rssm,
    scorer,
    aux_predictor,
    question_static,
    n_questions,
    train_answer_features,
    test_answer_features,
    train_question_idxs,
    test_question_idxs,
    train_tc_scores,
    test_tc_scores,
    train_valid_mask,
    test_valid_mask,
    h0,
    a0,
):
    params = list(rssm.parameters()) + list(scorer.parameters())
    if config.use_aux_loss and aux_predictor is not None:
        params += list(aux_predictor.parameters())

    print(f"Number of parameters: {sum(p.numel() for p in params)}")
    optimizer = torch.optim.Adam(params, lr=args.lr)

    train_attempts = train_answer_features.shape[1]
    train_losses = []
    test_accs = []
    test_accs_rollout = []

    for epoch in tqdm(range(args.epochs)):
        # --- Train ---
        rssm.train()
        scorer.train()
        if aux_predictor is not None:
            aux_predictor.train()

        list_score_hat = []
        list_feat_hat = []
        prev_hidden = h0

        for aidx in range(train_attempts):
            q_ids = train_question_idxs[:, aidx]
            q_static = question_static[q_ids]

            if aidx == 0:
                prev_feat = a0
            else:
                prev_feat = train_answer_features[:, aidx - 1]

            hidden = rssm(prev_feat, prev_hidden, q_ids, q_static)
            q_enc = rssm.encode_question(q_ids, q_static)
            score_hat = scorer(hidden, q_enc)

            list_score_hat.append(score_hat)

            if config.use_aux_loss and aux_predictor is not None:
                feat_hat = aux_predictor(hidden)
                list_feat_hat.append(feat_hat)

            prev_hidden = hidden

        list_score_hat = torch.stack(list_score_hat).permute(1, 0, 2)
        # [n_students, train_attempts, n_testcases]

        # BCE loss on testcase predictions (masked where tc_score == -1)
        tc_mask = train_tc_scores != -1
        bce_loss = F.binary_cross_entropy(
            list_score_hat[tc_mask], train_tc_scores[tc_mask]
        )

        total_loss = bce_loss

        # Auxiliary feature prediction loss
        if config.use_aux_loss and aux_predictor is not None and list_feat_hat:
            list_feat_hat = torch.stack(list_feat_hat).permute(1, 0, 2)
            # [n_students, train_attempts, answer_dim]
            valid = train_valid_mask.unsqueeze(-1).expand_as(list_feat_hat)
            aux_loss = F.mse_loss(
                list_feat_hat[valid], train_answer_features[valid]
            )
            total_loss = bce_loss + args.aux_loss_weight * aux_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(total_loss.item())
        if epoch % 10 == 0:
            print(f"Train Loss: {total_loss.item():.4f}")

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            _, test_res = test(
                rssm,
                scorer,
                aux_predictor,
                config,
                prev_hidden,
                train_answer_features[:, -1],
                question_static,
                test_question_idxs,
                test_answer_features,
                test_tc_scores,
                test_valid_mask,
            )
            test_accs.append(test_res["accuracy"])
            test_accs_rollout.append(test_res["accuracy_pe"])
            print(
                f"  Test Acc: {test_res['accuracy']:.4f}"
                f"\tTest Acc Rollout: {test_res['accuracy_pe']:.4f}"
            )

    # Save models
    save_dir = f"saves/multimodal/{args.cls}_{args.config}"
    ensure_dir(save_dir)
    torch.save(rssm.state_dict(), f"{save_dir}/rssm.pth")
    torch.save(scorer.state_dict(), f"{save_dir}/scorer.pth")
    if aux_predictor is not None:
        torch.save(aux_predictor.state_dict(), f"{save_dir}/aux_predictor.pth")

    # Final evaluation on both train and test
    hidden_state, train_res = test(
        rssm, scorer, aux_predictor, config,
        h0, a0,
        question_static,
        train_question_idxs, train_answer_features,
        train_tc_scores, train_valid_mask,
    )
    _, test_res = test(
        rssm, scorer, aux_predictor, config,
        hidden_state, train_answer_features[:, -1],
        question_static,
        test_question_idxs, test_answer_features,
        test_tc_scores, test_valid_mask,
    )

    # Save results to standard results/ directory
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "rssm_multimodal",
    )
    ensure_dir(result_dir)

    # Save metrics JSON
    fit_metrics = {"train": train_res, "test": test_res}
    with open(os.path.join(result_dir, "fit_metrics.json"), "w") as f:
        json.dump(fit_metrics, f, indent=2)
    print(f"  {os.path.join(result_dir, 'fit_metrics.json')}")

    # Save losses JSON
    with open(os.path.join(result_dir, "losses.json"), "w") as f:
        json.dump({"train": train_losses}, f)

    # Plots
    print("\nGenerating plots...")
    plot_losses(train_losses, result_dir)
    eval_epochs = list(range(0, args.epochs, 100)) + [args.epochs - 1]
    plot_test_metrics(eval_epochs, test_accs, test_accs_rollout, result_dir)
    plot_question_difficulty(question_static, result_dir)
    plot_question_embeddings(rssm, n_questions, result_dir)
    plot_final_metrics_bar(train_res, test_res, result_dir)

    print(f"\nFinal Train: {json.dumps(train_res, indent=2)}")
    print(f"\nFinal Test:  {json.dumps(test_res, indent=2)}")


def test(
    rssm,
    scorer,
    aux_predictor,
    config,
    prev_hs_0,
    prev_feat_0,
    question_static,
    question_idxs,
    answer_features,
    tc_scores,
    valid_mask,
):
    """Evaluate model. Returns (final_hidden, metrics_dict).

    Two modes (matching original main_rssm.py):
    - Teacher forcing: use real answer features from previous timestep
    - Rollout: use predicted answer features from aux_predictor
    """
    rssm.eval()
    scorer.eval()
    if aux_predictor is not None:
        aux_predictor.eval()

    with torch.no_grad():
        prev_hidden = prev_hs_0
        list_score_hat = []
        list_score_hat_rollout = []

        for aidx in range(answer_features.shape[1]):
            q_ids = question_idxs[:, aidx]
            q_static = question_static[q_ids]

            if aidx == 0:
                prev_feat = prev_feat_0
            else:
                prev_feat = answer_features[:, aidx - 1]

            hidden = rssm(prev_feat, prev_hidden, q_ids, q_static)
            q_enc = rssm.encode_question(q_ids, q_static)

            # Teacher-forced prediction
            score_hat = scorer(hidden, q_enc)
            list_score_hat.append(score_hat)

            # Rollout prediction (use predicted features for next step)
            if aux_predictor is not None:
                feat_hat = aux_predictor(hidden)
                hidden_rollout = rssm(
                    feat_hat if aidx > 0 else prev_feat_0,
                    prev_hidden,
                    q_ids,
                    q_static,
                )
                q_enc_rollout = rssm.encode_question(q_ids, q_static)
                score_hat_rollout = scorer(hidden_rollout, q_enc_rollout)
            else:
                score_hat_rollout = score_hat

            list_score_hat_rollout.append(score_hat_rollout)
            prev_hidden = hidden

        list_score_hat = torch.stack(list_score_hat).permute(1, 0, 2)
        list_score_hat_rollout = torch.stack(list_score_hat_rollout).permute(1, 0, 2)

        # Mask and compute metrics (matching main_rssm.py lines 212-238)
        tc_mask = tc_scores != -1
        gt = tc_scores[tc_mask].cpu().numpy()

        pred = (list_score_hat[tc_mask] >= 0.5).float().cpu().numpy()
        pred_rollout = (list_score_hat_rollout[tc_mask] >= 0.5).float().cpu().numpy()

        if len(gt) == 0 or len(np.unique(gt)) < 2:
            # Not enough data or only one class
            empty = {
                "accuracy": 0.0, "f1": 0.0, "precision": 0.0,
                "recall": 0.0, "roc_auc": 0.0,
                "accuracy_pe": 0.0, "f1_pe": 0.0, "precision_pe": 0.0,
                "recall_pe": 0.0, "roc_auc_pe": 0.0,
            }
            return prev_hidden, empty

        results = {
            "accuracy": accuracy_score(gt, pred),
            "f1": f1_score(gt, pred),
            "precision": precision_score(gt, pred),
            "recall": recall_score(gt, pred),
            "roc_auc": roc_auc_score(gt, pred),
            "accuracy_pe": accuracy_score(gt, pred_rollout),
            "f1_pe": f1_score(gt, pred_rollout),
            "precision_pe": precision_score(gt, pred_rollout),
            "recall_pe": recall_score(gt, pred_rollout),
            "roc_auc_pe": roc_auc_score(gt, pred_rollout),
        }

    return prev_hidden, results


if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, default="dsa_hk231")
    parser.add_argument("--config", type=str, default="full",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--enc_dim", type=int, default=64)
    parser.add_argument("--train_attempts", type=int, default=1000)
    parser.add_argument("--aux_loss_weight", type=float, default=0.1)
    parser.add_argument("--test_only", type=int, default=0)
    args = parser.parse_args()

    config = CONFIGS[args.config]
    data_dir = f"data/multimodal/{args.cls}"

    print(f"Config: {args.config} (answer_dim={config.answer_dim})")
    print(f"Loading data from {data_dir}...")

    (
        question_static,
        n_questions,
        train_answer_features,
        test_answer_features,
        train_question_idxs,
        test_question_idxs,
        train_tc_scores,
        test_tc_scores,
        train_valid_mask,
        test_valid_mask,
        data_info,
    ) = load_multimodal_data(data_dir, device, args.train_attempts, config)

    n_students, total_attempts, answer_dim = data_info

    # Initialize models
    rssm = MultiModalRSSM(
        config, n_questions, hidden_dim=args.hidden_dim, enc_dim=args.enc_dim
    ).to(device)
    scorer = MultiModalScorer(
        hidden_dim=args.hidden_dim,
        question_enc_dim=args.enc_dim,
        n_testcases=config.n_testcases,
    ).to(device)

    aux_predictor = None
    if config.use_aux_loss:
        aux_predictor = AnswerFeaturePredictor(
            hidden_dim=args.hidden_dim, output_dim=config.answer_dim
        ).to(device)

    # Load pre-trained if available
    save_dir = f"saves/multimodal/{args.cls}_{args.config}"
    if os.path.exists(f"{save_dir}/rssm.pth"):
        print("Loading pre-trained models...")
        rssm.load_state_dict(torch.load(f"{save_dir}/rssm.pth", map_location=device))
        scorer.load_state_dict(
            torch.load(f"{save_dir}/scorer.pth", map_location=device)
        )
        if aux_predictor is not None and os.path.exists(f"{save_dir}/aux_predictor.pth"):
            aux_predictor.load_state_dict(
                torch.load(f"{save_dir}/aux_predictor.pth", map_location=device)
            )

    print(f"RSSM: {rssm}")
    print(f"Scorer: {scorer}")
    if aux_predictor:
        print(f"AuxPredictor: {aux_predictor}")

    # Zero-init (structured features have meaningful zero)
    h0 = torch.zeros(n_students, args.hidden_dim, device=device)
    a0 = torch.zeros(n_students, config.answer_dim, device=device)

    # Standard results directory (matches other models)
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "rssm_multimodal",
    )

    if args.test_only:
        # Evaluate on train set to get final hidden state, then on test set
        hidden_state, train_res = test(
            rssm, scorer, aux_predictor, config,
            h0, a0,
            question_static,
            train_question_idxs, train_answer_features,
            train_tc_scores, train_valid_mask,
        )
        _, test_res = test(
            rssm, scorer, aux_predictor, config,
            hidden_state, train_answer_features[:, -1],
            question_static,
            test_question_idxs, test_answer_features,
            test_tc_scores, test_valid_mask,
        )

        ensure_dir(result_dir)
        fit_metrics = {"train": train_res, "test": test_res}
        with open(os.path.join(result_dir, "fit_metrics.json"), "w") as f:
            json.dump(fit_metrics, f, indent=2)

        plot_final_metrics_bar(train_res, test_res, result_dir)
        plot_question_difficulty(question_static, result_dir)
        plot_question_embeddings(rssm, n_questions, result_dir)

        print("\nTrain results:", json.dumps(train_res, indent=2))
        print("\nTest results:", json.dumps(test_res, indent=2))
    else:
        train(
            args, config,
            rssm, scorer, aux_predictor,
            question_static, n_questions,
            train_answer_features, test_answer_features,
            train_question_idxs, test_question_idxs,
            train_tc_scores, test_tc_scores,
            train_valid_mask, test_valid_mask,
            h0, a0,
        )
