"""Unified data loading for temporal evaluation.

Loads data once from HuggingFace in both CSV and tensor formats,
so all model adapters can consume data in their preferred format.

For student-based evaluation, use load_student_split_data() which
applies quality filtering, caps attempts, and splits students —
returning a single (data, split) pair that all adapters share.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download


@dataclass
class UnifiedData:
    """All data loaded once, served in multiple formats."""

    # Raw CSV
    main_data: pd.DataFrame

    # 3D Tensors
    correctness_matrix: torch.Tensor  # [n_students, n_items, n_max_attempts]
    time_matrix: torch.Tensor  # [n_students, n_items, n_max_attempts]

    # Question metadata
    question_infos: pd.DataFrame  # qidx, qname, week, topic

    # Item-to-week mapping
    item_week: torch.Tensor  # [n_items], int: week number for each item

    # question_unittest_id (int) -> week (int) mapping for CSV-based models
    qid_to_week: Dict[int, int] = field(default_factory=dict)

    # Student info
    student_ids: List[str] = field(default_factory=list)

    # Dimensions
    n_students: int = 0
    n_items: int = 0
    n_max_attempts: int = 0
    course_name: str = ""


def load_unified_data(course_name: str = "all") -> UnifiedData:
    """Load all data from HuggingFace.

    Args:
        course_name: Course to load. Use a specific course name
            (e.g., "dsa_hk231") to load pre-built tensors, or "all"
            to build combined matrices from raw CSV.

    Returns:
        UnifiedData with both CSV and tensor representations.
    """
    # Load raw CSV
    csv_path = os.environ.get("CODEINSIGHT_CSV_PATH")
    if csv_path and not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CODEINSIGHT_CSV_PATH is set to '{csv_path}' but that path "
            f"does not exist."
        )
    if not csv_path:
        csv_path = snapshot_download(
            repo_id="CodeInsightTeam/code_insights_csv", repo_type="dataset",
        )
    main_data = pd.read_csv(
        f"{csv_path}/main_data.csv", low_memory=False, on_bad_lines="skip"
    )
    question_infos_csv = pd.read_csv(f"{csv_path}/question_infos.csv")
    course_infos = pd.read_csv(f"{csv_path}/course_infos.csv")

    # Filter to submissions
    main_data = main_data[
        main_data["response_type"].isin(["Submit", "Prechecked"])
    ].copy()
    main_data = main_data.dropna(subset=["pass"])

    if course_name != "all":
        course_row = course_infos[course_infos["course_name"] == course_name]
        if len(course_row) > 0:
            course_id = course_row["course_id"].values[0]
            main_data = main_data[main_data["course_id"] == course_id].copy()

    cache_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", ".cache", "matrices", course_name
    )
    cache_dir = os.path.abspath(cache_dir)

    # Also check for matrices from HF repo (fagunpatel98/code_insights_matrices)
    if not os.path.exists(os.path.join(cache_dir, "correctness_matrix.pt")):
        try:
            hf_token = os.environ.get("HF_TOKEN")
            hf_matrices_path = snapshot_download(
                repo_id="fagunpatel98/code_insights_matrices",
                repo_type="dataset",
                allow_patterns=[f"{course_name}/*"],
                token=hf_token,
            )
            hf_course_dir = os.path.join(hf_matrices_path, course_name)
            if os.path.exists(os.path.join(hf_course_dir, "correctness_matrix.pt")):
                cache_dir = hf_course_dir
                print(f"Using matrices from fagunpatel98/code_insights_matrices/{course_name}")
        except Exception as e:
            print(f"Could not fetch matrices from HF: {e}")

    if os.path.exists(os.path.join(cache_dir, "correctness_matrix.pt")):
        print(f"Loading cached matrices from {cache_dir}")
        correctness = torch.load(
            f"{cache_dir}/correctness_matrix.pt", map_location="cpu"
        )
        time_mat = torch.load(
            f"{cache_dir}/time_matrix.pt", map_location="cpu"
        )
        question_infos = pd.read_csv(f"{cache_dir}/question_infos.csv")
        student_info_path = f"{cache_dir}/student_info.csv"
        if os.path.exists(student_info_path):
            student_ids = pd.read_csv(student_info_path)["student_id"].tolist()
        else:
            student_ids = list(range(correctness.shape[0]))
    else:
        print(f"Building matrices from CSV for {course_name}...")
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        sys.path.insert(0, os.path.join(repo_root, "data_collection"))
        from csv2matrices import build_matrices

        result = build_matrices(main_data, question_infos_csv, course_name, "cpu")
        student_info, question_info_list, correctness, time_mat, _ = result
        question_infos = pd.DataFrame(question_info_list)
        student_ids = [s["student_id"] for s in student_info]

        os.makedirs(cache_dir, exist_ok=True)
        torch.save(correctness, f"{cache_dir}/correctness_matrix.pt")
        torch.save(time_mat, f"{cache_dir}/time_matrix.pt")
        question_infos.to_csv(f"{cache_dir}/question_infos.csv", index=False)
        pd.DataFrame({"student_id": student_ids}).to_csv(
            f"{cache_dir}/student_info.csv", index=False
        )
        print(f"Cached matrices to {cache_dir}")

    item_week = torch.tensor(
        question_infos["week"].fillna(0).astype(int).values, dtype=torch.long
    )

    # Build question_unittest_id -> week mapping
    # Prefer corrected per-item data (has question_unittest_id + week)
    if "question_unittest_id" in question_infos.columns:
        qid_week_df = question_infos[["question_unittest_id", "week"]].dropna().drop_duplicates("question_unittest_id")
        qid_to_week = dict(
            zip(qid_week_df["question_unittest_id"].astype(int), qid_week_df["week"].astype(int))
        )
    else:
        if course_name == "all":
            qid_week_df = question_infos_csv[["question_id", "week"]].dropna()
        else:
            course_row = course_infos[course_infos["course_name"] == course_name]
            if len(course_row) > 0:
                cid = course_row["course_id"].values[0]
                qid_week_df = question_infos_csv[
                    question_infos_csv["course_id"] == cid
                ][["question_id", "week"]].dropna()
            else:
                qid_week_df = question_infos_csv[["question_id", "week"]].dropna()
        qid_to_week = dict(
            zip(qid_week_df["question_id"].astype(int), qid_week_df["week"].astype(int))
        )

    n_students, n_items, n_max_attempts = correctness.shape
    print(f"Loaded {course_name}: {n_students} students, {n_items} items, "
          f"{n_max_attempts} max attempts")
    print(f"  Weeks: {sorted(item_week.unique().tolist())}")
    print(f"  qid_to_week entries: {len(qid_to_week)}")

    return UnifiedData(
        main_data=main_data,
        correctness_matrix=correctness.float(),
        time_matrix=time_mat.float(),
        question_infos=question_infos,
        item_week=item_week,
        qid_to_week=qid_to_week,
        student_ids=student_ids,
        n_students=n_students,
        n_items=n_items,
        n_max_attempts=n_max_attempts,
        course_name=course_name,
    )


def load_student_split_data(
    course_name: str = "dsa_hk231",
    max_attempts: int = 10,
    test_frac: float = 0.3,
    val_frac: float = 0.15,
    train_week_cutoff: int = 3,
    seed: int = 42,
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
    min_question_coverage: float = 0.25,
) -> Tuple["UnifiedData", "StudentSplit"]:
    """Load data, apply quality filter, cap attempts, split students.

    Returns a single (data, split) pair shared by all adapters.
    """
    from .data_filter import DataFilterConfig, filter_data
    from .student_split import generate_student_split

    data = load_unified_data(course_name)

    max_week = int(data.question_infos["week"].max())
    config = DataFilterConfig(max_week=max_week, max_attempts=max_attempts,
                              min_pass_rate=min_pass_rate,
                              max_pass_rate=max_pass_rate,
                              min_question_coverage=min_question_coverage)
    data = filter_data(data, config)

    split = generate_student_split(
        n_students=data.n_students,
        item_week=data.item_week,
        train_week_cutoff=train_week_cutoff,
        test_frac=test_frac,
        val_frac=val_frac,
        seed=seed,
    )

    print(f"  Filtered: {data.n_students} students, {data.n_items} items, "
          f"{data.n_max_attempts} attempts")
    print(f"  {split}")

    return data, split
