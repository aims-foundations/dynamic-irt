"""Shared loading/alignment of precomputed code embeddings.

Single implementation used by every embedding consumer (RSSM adapter,
Code-DKT adapter). Embeddings are stored per
interaction (student, question, attempt) by dynamic_models/featurize.py.
Pickle indices are remapped onto the current filtered data universe by
id, and testcase/attempt alignment is validated against the correctness
matrix; mismatches are a hard error when strict.
"""

import os
import pickle

import numpy as np
import torch

EMBED_FILES = ["answer_features", "question_idxs", "question_embeddings",
               "testcase_scores", "student_idxs", "metadata"]


def resolve_emb_dir(course_name, emb_dir="", model_tag="Qwen3-Embedding-8B"):
    """Resolution order: explicit dir, then the model-tag subdir, then the
    legacy course root (which may itself contain a single pickle subdir)."""
    if emb_dir:
        return emb_dir
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    base = os.path.join(repo_root, "data", "embeddings", course_name)
    tagged = os.path.join(base, model_tag)
    if os.path.exists(os.path.join(tagged, "answer_features.pkl")):
        return tagged
    return base


def load_embeddings(emb_dir):
    """Load the embedding pickle set; attempt_idxs is optional (legacy)."""
    if not os.path.exists(emb_dir):
        raise FileNotFoundError(
            f"Embedding data not found at {emb_dir}. "
            f"Run: modal run dynamic_models/featurize.py::embed"
        )
    data_dir = emb_dir
    subdirs = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))
               and os.path.exists(os.path.join(data_dir, d, "answer_features.pkl"))]
    if os.path.exists(os.path.join(data_dir, "answer_features.pkl")):
        if subdirs:
            print(f"  WARNING: using legacy root pickle at {data_dir} even "
                  f"though tagged subdirs with pickles exist: {sorted(subdirs)}. "
                  f"Pass emb_dir or model_tag explicitly to pick one.", flush=True)
    else:
        if len(subdirs) == 1:
            data_dir = os.path.join(data_dir, subdirs[0])
            print(f"  Using embedding subdir: {data_dir}")
        elif len(subdirs) > 1:
            raise ValueError(
                f"Multiple embedding subdirs under {emb_dir}: {sorted(subdirs)}. "
                f"Pass emb_dir or model_tag explicitly to pick one."
            )
    loaded = {}
    for name in EMBED_FILES:
        with open(os.path.join(data_dir, f"{name}.pkl"), "rb") as f:
            loaded[name] = pickle.load(f)
    attempt_path = os.path.join(data_dir, "attempt_idxs.pkl")
    if os.path.exists(attempt_path):
        with open(attempt_path, "rb") as f:
            loaded["attempt_idxs"] = pickle.load(f)
    else:
        loaded["attempt_idxs"] = None
    loaded["emb_dir"] = data_dir
    return loaded


def _sid_key(sid):
    try:
        return int(sid)
    except (TypeError, ValueError):
        return sid


def map_question_indices(metadata, question_infos, strict=True):
    """Map pickle question indices onto current-universe qidx by id.

    Questions in the current universe with no embeddings are a hard error
    when strict, so zero vectors never enter training.
    """
    qi = question_infos
    if "question_to_idx" not in metadata or "question_unittest_id" not in qi.columns:
        raise ValueError(
            "Embedding metadata lacks question_to_idx or question_infos "
            "lacks question_unittest_id; regenerate embeddings with "
            "dynamic_models/featurize.py"
        )
    emb_to_quid = {v: int(k) for k, v in metadata["question_to_idx"].items()}
    quid_to_qidx = dict(zip(
        qi["question_unittest_id"].astype(int),
        qi["qidx"].astype(int),
    ))
    emb_to_qidx = {
        emb_idx: quid_to_qidx[quid]
        for emb_idx, quid in emb_to_quid.items()
        if quid in quid_to_qidx
    }
    missing_questions = set(quid_to_qidx) - set(emb_to_quid.values())
    print(f"    Mapped {len(emb_to_qidx)}/{len(emb_to_quid)} embedded questions "
          f"to filtered set; {len(missing_questions)} filtered-set questions "
          f"missing from embeddings", flush=True)
    if missing_questions:
        msg = (f"{len(missing_questions)} questions in the current universe "
               f"have no embeddings (e.g. question_unittest_ids "
               f"{sorted(missing_questions)[:5]}). Regenerate embeddings "
               f"with the matching filter config.")
        if strict:
            raise ValueError(msg)
        print(f"    WARNING: {msg}", flush=True)
    return emb_to_qidx


def map_student_indices(metadata, data, strict=True):
    """Map pickle student indices onto the current universe by student id.

    Raw positional indices are only valid if the pickle was built under the
    identical filter config; ids make that explicit.
    """
    cur_sid_to_idx = {_sid_key(sid): i for i, sid in enumerate(data.student_ids)}
    pickle_s2i = metadata.get("student_to_idx")
    if pickle_s2i is None:
        raise ValueError(
            "Embedding metadata lacks student_to_idx; regenerate embeddings "
            "with dynamic_models/featurize.py"
        )
    pidx_to_cur = {
        pidx: cur_sid_to_idx[_sid_key(sid)]
        for sid, pidx in pickle_s2i.items()
        if _sid_key(sid) in cur_sid_to_idx
    }
    n_cur_matched = len(set(pidx_to_cur.values()))
    print(f"    Matched {len(pidx_to_cur)}/{len(pickle_s2i)} embedding students; "
          f"{n_cur_matched}/{data.n_students} current students covered "
          f"(0.99 coverage threshold: up to 1% of students may be silently "
          f"absent here; strict full coverage of test students is enforced by "
          f"the RSSM adapter)", flush=True)
    if n_cur_matched < 0.99 * data.n_students:
        msg = (f"Embedding pickle covers only {n_cur_matched}/{data.n_students} "
               f"current students; the pickle was built under a different "
               f"filter config. Regenerate embeddings.")
        if strict:
            raise ValueError(msg)
        print(f"    WARNING: {msg}", flush=True)
    return pidx_to_cur


def check_testcase_alignment(per_student, data, strict=True):
    """Validate testcase ordering and attempt indexing against the
    correctness matrix on a sample of students.

    per_student: {student_idx: [(qidx, attempt, testcase_scores), ...]}
    Returns (n_checked, n_mismatch).
    """
    qi = data.question_infos
    qidx_to_item_range = {}
    for qidx in qi["qidx"].unique():
        items = qi[qi["qidx"] == qidx].index.tolist()
        qidx_to_item_range[int(qidx)] = (min(items), len(items))

    corr_check = data.correctness_matrix.numpy()
    rng = np.random.RandomState(0)
    check_students = rng.choice(
        sorted(per_student), size=min(50, len(per_student)), replace=False
    )
    n_checked, n_mismatch = 0, 0
    for si in check_students:
        for qidx, att, tc in per_student[si]:
            if att >= data.n_max_attempts or qidx not in qidx_to_item_range:
                continue
            item_start, n_items_q = qidx_to_item_range[qidx]
            tc_arr = np.asarray(tc)
            n_cmp = min(n_items_q, len(tc_arr))
            gt = corr_check[si, item_start:item_start + n_cmp, att]
            pk = tc_arr[:n_cmp]
            valid = (gt != -1) & (pk != -1)
            n_checked += int(valid.sum())
            n_mismatch += int((gt[valid] != pk[valid]).sum())
    print(f"    Testcase alignment check: {n_checked} entries, "
          f"{n_mismatch} mismatches", flush=True)
    if n_checked == 0 or n_mismatch > 0:
        msg = (f"Testcase/attempt alignment failed ({n_mismatch}/{n_checked} "
               f"mismatches): embedding pickle order does not match the "
               f"correctness matrix.")
        if strict:
            raise ValueError(msg)
        print(f"    WARNING: {msg}", flush=True)
    return n_checked, n_mismatch


def align_events(emb, data, strict=True, emb_to_qidx=None, pidx_to_cur=None):
    """Remap pickle interactions onto the current filtered universe.

    The single shared remap: every consumer's view of "which pickle rows
    belong to the current run" comes from here. Rows outside the universe
    (unknown student/question, attempt >= data.n_max_attempts) are dropped
    with counts, and testcase/attempt ordering is validated against the
    correctness matrix.

    Pass precomputed emb_to_qidx / pidx_to_cur to reuse maps a caller
    already built; otherwise they are derived here.

    Returns (kept, stats): kept is [(pickle_row, student_idx, qidx, attempt)]
    in pickle (chronological) order.
    """
    metadata = emb["metadata"]
    if emb_to_qidx is None:
        emb_to_qidx = map_question_indices(metadata, data.question_infos, strict=strict)
    if pidx_to_cur is None:
        pidx_to_cur = map_student_indices(metadata, data, strict=strict)

    student_idxs = emb["student_idxs"]
    question_idxs = emb["question_idxs"]
    attempt_idxs = emb["attempt_idxs"]

    kept = []
    replay_counts = {}
    n_dropped_student, n_dropped_question, n_dropped_attempt = 0, 0, 0
    for i in range(len(student_idxs)):
        si = pidx_to_cur.get(student_idxs[i])
        if si is None:
            n_dropped_student += 1
            continue
        qidx = emb_to_qidx.get(question_idxs[i])
        if qidx is None:
            n_dropped_question += 1
            continue
        if attempt_idxs is not None:
            att = int(attempt_idxs[i])
        else:
            att = replay_counts.get((si, qidx), 0)
            replay_counts[(si, qidx)] = att + 1
        if att >= data.n_max_attempts:
            n_dropped_attempt += 1
            continue
        kept.append((i, si, qidx, att))
    if n_dropped_student or n_dropped_question or n_dropped_attempt:
        print(f"    Dropped interactions outside current universe: "
              f"{n_dropped_student} by student, {n_dropped_question} by question, "
              f"{n_dropped_attempt} by attempt cap",
              flush=True)

    testcase_scores = emb["testcase_scores"]
    per_student = {}
    for orig_i, si, qidx, att in kept:
        per_student.setdefault(si, []).append((qidx, att, testcase_scores[orig_i]))
    n_checked, n_mismatch = check_testcase_alignment(per_student, data, strict=strict)

    stats = {
        "n_interactions": len(student_idxs),
        "n_matched": len(kept),
        "n_dropped_student": n_dropped_student,
        "n_dropped_question": n_dropped_question,
        "n_dropped_attempt": n_dropped_attempt,
        "n_checked": n_checked,
        "n_mismatch": n_mismatch,
        "has_attempt_idxs": attempt_idxs is not None,
    }
    return kept, stats


def align_to_universe(emb, data, strict=True):
    """align_events packed for lookup-style consumers (Code-DKT, TIKTOC).

    Returns (emb_matrix, row_lookup, stats):
      emb_matrix: fp16 CPU tensor [n_matched + 1, emb_dim]; row 0 is zeros.
      row_lookup: {(student_idx, qidx, attempt): row >= 1}
      stats: counts for logging and gating.
    """
    kept, stats = align_events(emb, data, strict=strict)

    answer_features = emb["answer_features"]
    emb_dim = int(np.asarray(answer_features[0]).shape[0])
    emb_matrix = torch.zeros(len(kept) + 1, emb_dim, dtype=torch.float16)
    row_lookup = {}
    for row, (orig_i, si, qidx, att) in enumerate(kept):
        row_lookup[(si, qidx, att)] = row + 1
        emb_matrix[row + 1] = torch.from_numpy(
            np.asarray(answer_features[orig_i], dtype=np.float32)
        ).to(torch.float16)

    stats["emb_dim"] = emb_dim
    return emb_matrix, row_lookup, stats
