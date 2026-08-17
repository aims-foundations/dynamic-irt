"""Generate LLM text embeddings for learning dynamics models.

Reads from CodeInsightTeam/code_insights_csv, prepares texts, then
embeds via vLLM on a Modal A100 GPU. Also owns the shared Modal app,
images, and volumes; training entrypoints live in modal_train.py.

Usage:
    modal run dynamic_models/featurize.py --course dsa_hk231
    modal run dynamic_models/featurize.py --course dsa_hk231 --model deepseek-ai/deepseek-coder-1.3b-base
"""

import os
import pickle

import modal
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def emb_tag(model):
    """Volume directory tag for a model's unfiltered-superset embeddings."""
    return f"{model.split('/')[-1]}-unfiltered"


def _resolve_csv_path():
    """Resolve the CSV dataset dir: CODEINSIGHT_CSV_PATH, else HF snapshot."""
    csv_path = os.environ.get("CODEINSIGHT_CSV_PATH")
    if csv_path:
        return csv_path
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv", repo_type="dataset",
    )


# ---------------------------------------------------------------------------
# Data preparation (runs locally)
# ---------------------------------------------------------------------------


def _parse_pass_string(pass_str):
    """Parse the binary pass string from main_data.csv into a list of ints."""
    s = str(pass_str).strip()
    if "." in s:
        try:
            s = str(int(float(s)))
        except ValueError:
            return []
    return [int(c) for c in s if c in ("0", "1")]


def prepare_texts(course, max_students=None, max_attempts=15):
    """Load CSV data and extract texts to embed + interaction metadata.

    Embeds the whole course (all students/questions, attempts capped at
    max_attempts), producing a superset pickle the adapters subset for any
    downstream filter config.

    Args:
        course: Course name.
        max_students: If set, only process the first N students (for testing).
        max_attempts: Cap on attempts per student-question pair.

    Returns (answer_texts, question_texts, interaction_metadata, metadata).
    """
    import sys

    import pandas as pd
    from tqdm import tqdm

    sys.path.insert(0, REPO_ROOT)
    from dynamic_models.temporal_eval.data_loader import load_unified_data

    data = load_unified_data(course)
    main_data = data.main_data.copy()
    student_ids = data.student_ids

    if max_students is not None:
        student_ids = student_ids[:max_students]
        main_data = main_data[main_data["student_id"].isin(student_ids)].copy()

    student_to_idx = {sid: idx for idx, sid in enumerate(student_ids)}
    question_ids = main_data["question_unittest_id"].unique()
    question_to_idx = {int(qid): idx for idx, qid in enumerate(question_ids)}

    main_data["n_testcases"] = main_data["pass"].apply(
        lambda s: len(_parse_pass_string(s))
    )
    max_tc_per_q = main_data.groupby("question_unittest_id")["n_testcases"].max().to_dict()
    n_tc_global = max(max_tc_per_q.values()) if max_tc_per_q else 15

    # Attempt numbering must match csv2matrices exactly (same sort, same tie
    # order), so compute it under the matrix sort and store it per row.
    main_data = main_data.sort_values(
        ["student_id", "question_unittest_id", "timestamp"], kind="stable"
    )
    main_data["_attempt"] = main_data.groupby(
        ["student_id", "question_unittest_id"]
    ).cumcount()
    main_data = main_data[main_data["_attempt"] < max_attempts].copy()

    # Chronological order per student: the RSSM consumes these rows as the
    # student's trajectory, so sequence position must follow time, not
    # question grouping. The stored _attempt keeps ground-truth alignment
    # independent of this ordering.
    main_data = main_data.sort_values(["student_id", "timestamp"], kind="stable")

    # Per-interaction data
    answer_texts = []
    interaction_student_idxs = []
    interaction_question_idxs = []
    interaction_tc_scores = []
    interaction_attempt_idxs = []

    for _, row in tqdm(main_data.iterrows(), total=len(main_data), desc="Processing"):
        sid = row["student_id"]
        qid = int(row["question_unittest_id"])
        if sid not in student_to_idx or qid not in question_to_idx:
            continue
        tc = _parse_pass_string(row["pass"])
        if not tc:
            continue
        tc_padded = np.full(n_tc_global, -1.0)
        tc_padded[:len(tc)] = tc
        answer_texts.append(str(row["response"]) if pd.notna(row["response"]) else "")
        interaction_student_idxs.append(student_to_idx[sid])
        interaction_question_idxs.append(question_to_idx[qid])
        interaction_tc_scores.append(tc_padded)
        interaction_attempt_idxs.append(int(row["_attempt"]))

    # Question texts: UnifiedData's question_infos lacks question_text and
    # question_template, so read the raw CSVs (same resolution path as
    # data_loader).
    csv_path = _resolve_csv_path()
    question_infos_csv = pd.read_csv(f"{csv_path}/question_infos.csv")
    course_infos = pd.read_csv(f"{csv_path}/course_infos.csv")
    course_id = course_infos[course_infos["course_name"] == course]["course_id"].values[0]
    course_qi = question_infos_csv[question_infos_csv["course_id"] == course_id]

    question_texts = []
    for qid in question_ids:
        q_rows = course_qi[course_qi["question_id"] == qid]
        if len(q_rows) > 0:
            row = q_rows.iloc[0]
            parts = []
            if pd.notna(row.get("question_text", None)):
                parts.append(str(row["question_text"]))
            if pd.notna(row.get("question_template", None)):
                parts.append(f"Template:\n{row['question_template']}")
            question_texts.append("\n\n".join(parts))
        else:
            question_texts.append(f"Question {qid}")

    metadata = {
        "n_students": len(student_to_idx),
        "n_questions": len(question_to_idx),
        "n_interactions": len(answer_texts),
        "n_testcases": n_tc_global,
        "question_to_idx": question_to_idx,
        "student_to_idx": {int(k): v for k, v in student_to_idx.items()},
        "course": course,
        # Provenance: lets consumers verify the pickle universe matches the
        # data config of the current run instead of trusting raw indices.
        "filter": "none",
        "max_attempts": max_attempts,
        "n_items": data.n_items,
        "interaction_order": "chronological",
        "question_unittest_ids": sorted(int(q) for q in question_ids),
    }
    interaction_data = {
        "question_idxs": interaction_question_idxs,
        "testcase_scores": interaction_tc_scores,
        "student_idxs": interaction_student_idxs,
        "attempt_idxs": interaction_attempt_idxs,
    }

    print(f"Prepared {len(answer_texts)} answer texts, "
          f"{len(question_texts)} question texts")

    return answer_texts, question_texts, interaction_data, metadata


# ---------------------------------------------------------------------------
# Modal GPU embedding
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "numpy", "pandas", "huggingface_hub", "tqdm", "torch")
    .env({"VLLM_USE_DEEP_GEMM": "0"})
    .add_local_python_source("dynamic_models", "data_collection")
)

torch_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "numpy", "pandas", "scikit-learn",
        "huggingface_hub", "tqdm", "matplotlib", "tueplots", "seaborn",
    )
    .add_local_python_source("dynamic_models", "data_collection")
)

embeddings_vol = modal.Volume.from_name("codeinsight-embeddings", create_if_missing=True)
data_vol = modal.Volume.from_name("codeinsight-data", create_if_missing=True)
EMB_VOL_PATH = "/vol/embeddings"
DATA_VOL_PATH = "/vol/data"
hf_secret = modal.Secret.from_name("huggingface-secret")

app = modal.App("codeinsight")


@app.function(
    image=vllm_image,
    gpu="A100",
    timeout=7200,
    volumes={EMB_VOL_PATH: embeddings_vol, DATA_VOL_PATH: data_vol},
    secrets=[hf_secret],
)
def embed_and_save(
    course: str,
    model: str,
    batch_size: int = 256,
    max_students: int = 0,
):
    """Prepare texts + embed on GPU + save to volume. All on Modal.

    Filters are a training/eval concern, not an embedding one: pickles
    cover the whole course and the adapters subset them to whatever
    filter config a run uses.
    """
    import sys
    sys.path.insert(0, "/root")
    os.environ["HF_HUB_CACHE"] = f"{DATA_VOL_PATH}/hf_cache"
    os.environ["CODEINSIGHT_CSV_PATH"] = f"{DATA_VOL_PATH}/hf_data"

    import numpy as np
    from dynamic_models.featurize import prepare_texts

    n_students = max_students if max_students > 0 else None
    print(f"Preparing texts for {course}...")
    answer_texts, question_texts, interaction_data, metadata = prepare_texts(
        course, max_students=n_students,
    )
    # vLLM rejects empty prompts
    answer_texts = [t if t.strip() else "# empty" for t in answer_texts]
    question_texts = [t if t.strip() else "# empty" for t in question_texts]

    from vllm import LLM
    llm = LLM(model=model, runner="pooling", enforce_eager=True)
    max_chars = 4096 * 4

    print(f"Embedding {len(answer_texts)} answers...")
    answer_embs = []
    for i in range(0, len(answer_texts), batch_size):
        batch_texts = [t[:max_chars] for t in answer_texts[i:i + batch_size]]
        outputs = llm.embed(batch_texts)
        answer_embs.extend([o.outputs.embedding for o in outputs])
        print(f"  {min(i + batch_size, len(answer_texts))}/{len(answer_texts)}")

    print(f"Embedding {len(question_texts)} questions...")
    q_outputs = llm.embed([t[:max_chars] for t in question_texts])
    q_embs = [o.outputs.embedding for o in q_outputs]

    answer_features = [np.array(e, dtype=np.float32) for e in answer_embs]
    question_embeddings = np.array(q_embs, dtype=np.float32)
    emb_dim = question_embeddings.shape[1]
    metadata["answer_dim"] = emb_dim

    # The -unfiltered suffix is the established artifact name on the volume;
    # adapters resolve their emb dirs against it.
    data_dir = f"{EMB_VOL_PATH}/{course}/{emb_tag(model)}"
    os.makedirs(data_dir, exist_ok=True)
    for name, obj in [
        ("answer_features", answer_features),
        ("question_embeddings", question_embeddings),
        ("question_idxs", interaction_data["question_idxs"]),
        ("testcase_scores", interaction_data["testcase_scores"]),
        ("student_idxs", interaction_data["student_idxs"]),
        ("attempt_idxs", interaction_data["attempt_idxs"]),
        ("metadata", metadata),
    ]:
        with open(os.path.join(data_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)

    embeddings_vol.commit()
    print(f"\nSaved to volume {data_dir}/")
    print(f"  answer_features: {len(answer_features)} x {emb_dim}")
    print(f"  question_embeddings: {question_embeddings.shape}")


ALL_EMBED_MODELS = [
    "Qwen/Qwen3-Embedding-8B",
    "deepseek-ai/deepseek-coder-6.7b-base",
    "nomic-ai/nomic-embed-code",
]


@app.local_entrypoint()
def embed(
    course: str = "dsa_hk231",
    model: str = "",
    batch_size: int = 256,
    max_students: int = 0,
    all_models: bool = False,
):
    models = ALL_EMBED_MODELS if all_models else [model or ALL_EMBED_MODELS[0]]

    print(f"Launching {len(models)} embedding job(s) on Modal...")
    handles = {}
    for m in models:
        tag = m.split("/")[-1]
        print(f"  [{tag}] spawning...")
        handles[tag] = embed_and_save.spawn(course, m, batch_size, max_students)
    failed = []
    for tag, handle in handles.items():
        try:
            handle.get()
            print(f"  [{tag}] done.")
        except Exception as e:
            print(f"  [{tag}] FAILED: {e}")
            failed.append(tag)
    if failed:
        raise SystemExit(f"Embedding jobs failed: {', '.join(failed)}")
    print("All embedding jobs complete.")


# ---------------------------------------------------------------------------
# Modal data sync + GPU training
# ---------------------------------------------------------------------------

@app.function(
    image=torch_image,
    volumes={DATA_VOL_PATH: data_vol},
    secrets=[hf_secret],
    timeout=600,
)
def sync_hf_data():
    """Download HuggingFace dataset to volume, resolving symlinks."""
    import shutil
    from huggingface_hub import snapshot_download
    cache_dir = f"{DATA_VOL_PATH}/hf_cache"
    os.environ["HF_HUB_CACHE"] = cache_dir
    csv_path = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv",
        repo_type="dataset",
        cache_dir=cache_dir,
        force_download=True,
    )
    # Copy resolved files to a flat dir (Modal volumes don't handle symlinks)
    flat_dir = f"{DATA_VOL_PATH}/hf_data"
    os.makedirs(flat_dir, exist_ok=True)
    for f in os.listdir(csv_path):
        src = os.path.realpath(os.path.join(csv_path, f))
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(flat_dir, f))
    print(f"HF dataset synced to {flat_dir} ({len(os.listdir(flat_dir))} files)")
    data_vol.commit()
    print("HF dataset synced to volume.")
