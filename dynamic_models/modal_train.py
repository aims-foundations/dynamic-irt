"""Modal training entrypoints for the temporal-eval models.

Shares the Modal app, images, volumes, and secrets defined in
featurize.py, which owns the embedding pipeline.

Usage:
    modal run dynamic_models/modal_train.py::train --course dsa_hk231
    modal run dynamic_models/modal_train.py::train_dkt --course dsa_hk231
    modal run dynamic_models/modal_train.py::train_codedkt --course dsa_hk231
"""

import os
import pickle
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from dynamic_models.featurize import (
    DATA_VOL_PATH,
    EMB_VOL_PATH,
    REPO_ROOT,
    app,
    data_vol,
    emb_tag,
    embeddings_vol,
    hf_secret,
    sync_hf_data,
    torch_image,
)


def _setup_remote_env():
    os.environ["HF_HUB_CACHE"] = f"{DATA_VOL_PATH}/hf_cache"
    os.environ["CODEINSIGHT_CSV_PATH"] = f"{DATA_VOL_PATH}/hf_data"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _run_tag(run_name, emb_model):
    return run_name or (emb_model.split("/")[-1] if emb_model else "default")


def _persist_result(prediction, metrics, course, model_name):
    """Build the result dict and persist it on the volume as
    {model_name}_student_pred.pkl so results survive detached runs."""
    ms = dict(prediction.model_state or {})
    if "model_state_dict" in ms:
        ms["model_state_dict"] = {k: v.cpu() for k, v in ms["model_state_dict"].items()}
    result = {
        "metrics": metrics.to_dict(),
        "y_true": prediction.y_true.tolist(),
        "y_pred_prob": prediction.y_pred_prob.tolist(),
        "attempt_indices": prediction.attempt_indices.tolist(),
        "student_indices": prediction.student_indices.tolist(),
        "item_indices": prediction.item_indices.tolist(),
        "losses": prediction.losses,
        "model_state": ms,
    }
    weights_dir = f"{EMB_VOL_PATH}/{course}/models"
    os.makedirs(weights_dir, exist_ok=True)
    result_path = f"{weights_dir}/{model_name}_student_pred.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    embeddings_vol.commit()
    print(f"Saved to volume: {result_path}")
    return result


def _save_local_result(result, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_student_pred.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    metrics = result["metrics"]
    print(f"\n[{model_name}] Results: AUC={metrics['auc']:.4f}  "
          f"Acc={metrics['accuracy']:.4f}")
    print(f"Saved to {out_path}")


@app.function(
    image=torch_image,
    gpu="A100",
    timeout=14400,
    volumes={EMB_VOL_PATH: embeddings_vol, DATA_VOL_PATH: data_vol},
    secrets=[hf_secret],
)
def train_rssm_remote(
    course: str, seed: int = 42, epochs: int = 200,
    hidden_dim: int = 512, enc_dim: int = 512,
    lr: float = 3e-4, dropout: float = 0.1,
    emb_weight: float = 0.1, beta: float = 0.5,
    n_latent_vars: int = 16, n_latent_classes: int = 16,
    max_seq_len: int = 600,
    emb_model: str = "",
    use_cosine_lr: bool = True,
    run_name: str = "",
    difficulty_reg: float = 0.0,
    prior_score_weight: float = 1.0,
    pos_weight_mode: str = "none",
    patience: int = 50,
    cosine_t_max: int = 0,
    resume_checkpoint: str = "",
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
    min_question_coverage: float = 0.25,
    filter_max_attempts: int = 10,
) -> dict:
    """Train RSSM on Modal A100. All data on volumes."""
    _setup_remote_env()

    from dynamic_models.temporal_eval.adapters.rssm_adapter import RSSMAdapter
    from dynamic_models.temporal_eval.data_loader import load_student_split_data
    from dynamic_models.temporal_eval.metrics import compute_metrics

    tag = _run_tag(run_name, emb_model)
    print(f"[{tag}] Starting training...")

    # Embedding generation only produces -unfiltered superset dirs; the
    # adapter subsets them to the current filter config by id.
    emb_dir = (f"{EMB_VOL_PATH}/{course}/{emb_tag(emb_model)}" if emb_model
               else f"{EMB_VOL_PATH}/{course}")
    data, split = load_student_split_data(course, seed=seed,
                                          max_attempts=filter_max_attempts,
                                          min_pass_rate=min_pass_rate,
                                          max_pass_rate=max_pass_rate,
                                          min_question_coverage=min_question_coverage)
    adapter = RSSMAdapter()

    weights_dir = f"{EMB_VOL_PATH}/{course}/models"
    prediction = adapter.fit_and_predict_student_split(
        data, split, seed=seed, epochs=epochs, emb_dir=emb_dir,
        hidden_dim=hidden_dim, enc_dim=enc_dim, lr=lr, dropout=dropout,
        emb_weight=emb_weight, beta=beta,
        n_latent_vars=n_latent_vars, n_latent_classes=n_latent_classes,
        max_seq_len=max_seq_len,
        use_cosine_lr=use_cosine_lr,
        difficulty_reg=difficulty_reg,
        prior_score_weight=prior_score_weight,
        pos_weight_mode=pos_weight_mode,
        patience=patience,
        cosine_t_max=cosine_t_max or min(epochs, 80),
        resume_checkpoint=resume_checkpoint or None,
        eval_interval=10,
        checkpoint_dir=weights_dir,
        checkpoint_name=f"best_checkpoint_{tag}.pt" if run_name else "best_checkpoint.pt",
        on_checkpoint=lambda: embeddings_vol.commit(),
    )
    metrics = compute_metrics(prediction.y_true, prediction.y_pred_prob)

    print(f"AUC={metrics.auc:.4f}  Acc={metrics.accuracy:.4f}  "
          f"F1={metrics.f1:.4f}  LL={metrics.log_likelihood:.4f}  "
          f"N={metrics.n_test_obs}")

    # Save everything to volume so results survive laptop sleep
    import torch
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = (f"{weights_dir}/rssm_{run_name}.pt" if run_name
                    else f"{weights_dir}/rssm.pt")
    torch.save(prediction.model_state, weights_path)

    result = {
        "metrics": metrics.to_dict(),
        "y_true": prediction.y_true.tolist(),
        "y_pred_prob": prediction.y_pred_prob.tolist(),
        "attempt_indices": prediction.attempt_indices.tolist(),
        "student_indices": prediction.student_indices.tolist(),
        "item_indices": prediction.item_indices.tolist(),
        "losses": prediction.losses,
    }
    result_path = (f"{weights_dir}/RSSM_{run_name}.pkl" if run_name
                   else f"{weights_dir}/RSSM_student_pred.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(result, f)

    embeddings_vol.commit()
    print(f"Saved to volume: {weights_path}, {result_path}")

    return result


@app.function(
    image=torch_image,
    gpu="A100",
    timeout=3600,
    volumes={EMB_VOL_PATH: embeddings_vol, DATA_VOL_PATH: data_vol},
    secrets=[hf_secret],
)
def train_dkt_remote(
    course: str, seed: int = 42, epochs: int = 200,
    hidden_dim: int = 64, lr: float = 0.001,
    dropout: float = 0.5, batch_size: int = 100,
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
    min_question_coverage: float = 0.25,
    filter_max_attempts: int = 10,
) -> dict:
    """Train DKT on Modal A100."""
    _setup_remote_env()

    from dynamic_models.temporal_eval.adapters.dkt_adapter import DKTAdapter
    from dynamic_models.temporal_eval.data_loader import load_student_split_data
    from dynamic_models.temporal_eval.metrics import compute_metrics

    data, split = load_student_split_data(course, seed=seed,
                                          max_attempts=filter_max_attempts,
                                          min_pass_rate=min_pass_rate,
                                          max_pass_rate=max_pass_rate,
                                          min_question_coverage=min_question_coverage)
    adapter = DKTAdapter()
    prediction = adapter.fit_and_predict_student_split(
        data, split, seed=seed, epochs=epochs, hidden_dim=hidden_dim,
        lr=lr, dropout=dropout, batch_size=batch_size,
    )
    metrics = compute_metrics(prediction.y_true, prediction.y_pred_prob)
    print(f"AUC={metrics.auc:.4f}  Acc={metrics.accuracy:.4f}  "
          f"N={metrics.n_test_obs}")

    return _persist_result(prediction, metrics, course, "DKT")


@app.local_entrypoint()
def train_dkt(
    course: str = "dsa_hk231",
    seed: int = 42,
    epochs: int = 200,
    hidden_dim: int = 64,
    lr: float = 0.001,
    dropout: float = 0.5,
    output_dir: str = "",
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
    min_question_coverage: float = 0.25,
    filter_max_attempts: int = 10,
    spawn: bool = False,
):
    if not output_dir:
        output_dir = os.path.join(REPO_ROOT, "results", "student_eval", course)

    print(f"[DKT] Training on Modal A100 (course={course}, epochs={epochs}, "
          f"hidden={hidden_dim})...")

    fn = train_dkt_remote.spawn if spawn else train_dkt_remote.remote
    result = fn(
        course=course,
        seed=seed,
        epochs=epochs,
        hidden_dim=hidden_dim,
        lr=lr,
        dropout=dropout,
        batch_size=100,
        min_pass_rate=min_pass_rate,
        max_pass_rate=max_pass_rate,
        min_question_coverage=min_question_coverage,
        filter_max_attempts=filter_max_attempts,
    )
    if spawn:
        print(f"[DKT] spawned: {result.object_id} "
              f"(result -> volume {course}/models/DKT_student_pred.pkl)")
        return

    _save_local_result(result, output_dir, "DKT")


@app.function(
    image=torch_image,
    gpu="A100",
    timeout=7200,
    volumes={EMB_VOL_PATH: embeddings_vol, DATA_VOL_PATH: data_vol},
    secrets=[hf_secret],
)
def train_codedkt_remote(
    course: str, seed: int = 42, epochs: int = 200,
    hidden_dim: int = 64, lr: float = 0.001,
    emb_model: str = "Qwen/Qwen3-Embedding-8B",
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
    min_question_coverage: float = 0.25,
    filter_max_attempts: int = 10,
) -> dict:
    """Train Code-DKT on Modal A100 using the unfiltered superset embeddings."""
    _setup_remote_env()

    from dynamic_models.temporal_eval.adapters.code_dkt_adapter import CodeDKTAdapter
    from dynamic_models.temporal_eval.data_loader import load_student_split_data
    from dynamic_models.temporal_eval.metrics import compute_metrics

    data, split = load_student_split_data(course, seed=seed,
                                          max_attempts=filter_max_attempts,
                                          min_pass_rate=min_pass_rate,
                                          max_pass_rate=max_pass_rate,
                                          min_question_coverage=min_question_coverage)
    adapter = CodeDKTAdapter()
    prediction = adapter.fit_and_predict_student_split(
        data, split, seed=seed, epochs=epochs, hidden_dim=hidden_dim, lr=lr,
        emb_dir=f"{EMB_VOL_PATH}/{course}/{emb_tag(emb_model)}",
    )
    metrics = compute_metrics(prediction.y_true, prediction.y_pred_prob)
    print(f"AUC={metrics.auc:.4f}  Acc={metrics.accuracy:.4f}  "
          f"N={metrics.n_test_obs}")

    return _persist_result(prediction, metrics, course, "CodeDKT")


@app.local_entrypoint()
def train_codedkt(
    course: str = "dsa_hk231",
    seed: int = 42,
    epochs: int = 200,
    hidden_dim: int = 64,
    lr: float = 0.001,
    emb_model: str = "Qwen/Qwen3-Embedding-8B",
    output_dir: str = "",
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
    min_question_coverage: float = 0.25,
    filter_max_attempts: int = 10,
    spawn: bool = False,
):
    if not output_dir:
        output_dir = os.path.join(REPO_ROOT, "results", "student_eval", course)

    print(f"[CodeDKT] Training on Modal A100 (course={course}, "
          f"epochs={epochs}, hidden={hidden_dim})...")
    fn = train_codedkt_remote.spawn if spawn else train_codedkt_remote.remote
    result = fn(
        course=course,
        seed=seed,
        epochs=epochs,
        hidden_dim=hidden_dim,
        lr=lr,
        emb_model=emb_model,
        min_pass_rate=min_pass_rate,
        max_pass_rate=max_pass_rate,
        min_question_coverage=min_question_coverage,
        filter_max_attempts=filter_max_attempts,
    )
    if spawn:
        print(f"[CodeDKT] spawned: {result.object_id} "
              f"(result -> volume {course}/models/CodeDKT_student_pred.pkl)")
        return

    _save_local_result(result, output_dir, "CodeDKT")


@app.local_entrypoint()
def train(
    course: str = "dsa_hk231",
    seed: int = 42,
    epochs: int = 200,
    hidden_dim: int = 512,
    enc_dim: int = 512,
    lr: float = 3e-4,
    dropout: float = 0.1,
    emb_weight: float = 0.1,
    beta: float = 0.5,
    n_latent_vars: int = 16,
    n_latent_classes: int = 16,
    max_seq_len: int = 600,
    emb_model: str = "",
    use_cosine_lr: bool = True,
    run_name: str = "",
    difficulty_reg: float = 0.0,
    output_dir: str = "",
    sync: bool = False,
    prior_score_weight: float = 1.0,
    pos_weight_mode: str = "none",
    patience: int = 50,
    cosine_t_max: int = 0,
    resume_checkpoint: str = "",
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
    min_question_coverage: float = 0.25,
    filter_max_attempts: int = 10,
    spawn: bool = False,
):
    if not output_dir:
        output_dir = os.path.join(REPO_ROOT, "results", "student_eval", course)

    if sync:
        print("Syncing HF dataset to volume...")
        sync_hf_data.remote()
        print("Done.")
        return

    tag = _run_tag(run_name, emb_model)
    print(f"[{tag}] Training on Modal A100 (epochs={epochs}, hidden={hidden_dim}, "
          f"lr={lr}, beta={beta}, emb_w={emb_weight}, seq={max_seq_len})...")
    fn = train_rssm_remote.spawn if spawn else train_rssm_remote.remote
    result = fn(
        course=course,
        seed=seed,
        epochs=epochs,
        hidden_dim=hidden_dim,
        enc_dim=enc_dim,
        lr=lr,
        dropout=dropout,
        emb_weight=emb_weight,
        beta=beta,
        n_latent_vars=n_latent_vars,
        n_latent_classes=n_latent_classes,
        max_seq_len=max_seq_len,
        emb_model=emb_model,
        use_cosine_lr=use_cosine_lr,
        run_name=run_name,
        difficulty_reg=difficulty_reg,
        prior_score_weight=prior_score_weight,
        pos_weight_mode=pos_weight_mode,
        patience=patience,
        cosine_t_max=cosine_t_max,
        resume_checkpoint=resume_checkpoint,
        min_pass_rate=min_pass_rate,
        max_pass_rate=max_pass_rate,
        min_question_coverage=min_question_coverage,
        filter_max_attempts=filter_max_attempts,
    )
    if spawn:
        # Fire-and-forget: the input survives this client's death and the
        # result lands on the volume as models/RSSM_{tag}.pkl.
        print(f"[{tag}] spawned: {result.object_id}")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"RSSM_{run_name}.pkl" if run_name else "RSSM_student_pred.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    metrics = result["metrics"]
    print(f"\n[{tag}] Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"\nSaved to {out_path}")
