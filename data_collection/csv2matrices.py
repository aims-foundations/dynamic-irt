"""
Convert CSV data to PyTorch tensor matrices for IRT models.

This script reads from the CSV dataset (CodeInsightTeam/code_insights_csv) and creates
3D PyTorch tensors for use with GPIRT and other learning dynamics models.

Input: CSV files from HuggingFace (main_data.csv, question_infos.csv, etc.)
Output: PyTorch tensors uploaded to CodeInsightTeam/code_insights_matrices
    - correctness_matrix.pt: [n_students, n_questions*n_testcases, n_attempts]
    - time_matrix.pt: [n_students, n_questions*n_testcases, n_attempts]
    - is_exam_matrix.pt: [n_students, n_questions*n_testcases, n_attempts]

Usage:
    python csv2matrices.py --course_name dsa_hk231
"""

import io
import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm


def parse_time(time_str, course_name):
    """Parse timestamp string to days since course start."""
    if pd.isna(time_str) or time_str == "":
        return -1

    try:
        # Format: "01/09/23, 10:30:00"
        dt = datetime.strptime(time_str, "%d/%m/%y, %H:%M:%S")

        # Course start dates (approximate)
        course_starts = {
            "dsa_hk231": datetime(2023, 9, 1),
            "dsa_hk221": datetime(2022, 9, 1),
            "pf_hk232": datetime(2023, 9, 1),
            "pf_hk222": datetime(2022, 9, 1),
        }
        start = course_starts.get(course_name, datetime(2023, 1, 1))
        days = (dt - start).total_seconds() / 86400
        return max(0, days)
    except (ValueError, TypeError):
        return -1


def load_csv_data(course_name):
    """Load CSV data from HuggingFace."""
    # Try local cache first
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--CodeInsightTeam--code_insights_csv/"
        "snapshots/99d53fe7c11f6302fb28b82fab5ebd77c00e5d12"
    )
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        path = cache_path
    else:
        path = snapshot_download(
            repo_id="CodeInsightTeam/code_insights_csv", repo_type="dataset"
        )

    main_data = pd.read_csv(f"{path}/main_data.csv", low_memory=False)
    question_infos = pd.read_csv(f"{path}/question_infos.csv")
    course_infos = pd.read_csv(f"{path}/course_infos.csv")

    # Get course_id for the specified course
    course_row = course_infos[course_infos["course_name"] == course_name]
    if len(course_row) == 0:
        raise ValueError(f"Course {course_name} not found. Available: {course_infos['course_name'].tolist()}")
    course_id = course_row["course_id"].values[0]

    # Filter to specified course
    main_data = main_data[main_data["course_id"] == course_id].copy()

    # Filter to actual submissions (not Started/Finished/Saved)
    main_data = main_data[main_data["response_type"].isin(["Submit", "Prechecked"])].copy()

    # Drop rows with missing pass data
    main_data = main_data.dropna(subset=["pass"])

    return main_data, question_infos, course_name


def build_matrices(main_data, question_infos, course_name, device):
    """Build 3D matrices from CSV data."""
    # Filter question_infos to this course (only valid for single-course data)
    single_course = (
        main_data["course_id"].nunique() == 1
        if "course_id" in main_data.columns else True
    )
    if not single_course:
        print("WARNING: main_data spans multiple courses; skipping question_infos course filter and derived-week fallback (per-course metadata requires a single course).")
    if single_course and "course_id" in main_data.columns and "course_id" in question_infos.columns:
        course_id = int(main_data["course_id"].iloc[0])
        question_infos = question_infos[question_infos["course_id"].astype(int) == course_id]

    # Create mappings
    student_ids = main_data["student_id"].unique()
    student_to_idx = {sid: idx for idx, sid in enumerate(student_ids)}

    question_ids = main_data["question_unittest_id"].unique()
    question_to_idx = {qid: idx for idx, qid in enumerate(question_ids)}

    # Compute max testcases per question
    def count_testcases(pass_str):
        try:
            s = str(pass_str).strip()
            if "." in s:
                s = str(int(float(s)))
            return len(s)
        except (ValueError, TypeError):
            return 0

    main_data["n_testcases"] = main_data["pass"].apply(count_testcases)
    max_testcases_per_q = main_data.groupby("question_unittest_id")["n_testcases"].max().to_dict()

    # Create flattened question-testcase indices
    # Each question expands to multiple items (one per testcase)
    qidx_to_tc_range = {}
    tc_idx = 0
    for qid in question_ids:
        n_tc = max_testcases_per_q.get(qid, 1)
        qidx_to_tc_range[qid] = (tc_idx, tc_idx + n_tc)
        tc_idx += n_tc

    n_students = len(student_ids)
    n_items = tc_idx  # Total testcase items

    # Compute max attempts per student-question pair
    attempt_counts = main_data.groupby(["student_id", "question_unittest_id"]).size()
    max_attempts = attempt_counts.max() if len(attempt_counts) > 0 else 1

    print(f"Building matrices: {n_students} students x {n_items} items x {max_attempts} attempts")

    # Initialize matrices with -1 (missing)
    correctness_matrix = np.full((n_students, n_items, max_attempts), -1, dtype=np.int8)
    time_matrix = np.full((n_students, n_items, max_attempts), -1, dtype=np.float32)
    is_exam_matrix = np.full((n_students, n_items, max_attempts), -1, dtype=np.int8)

    # Sort by student, question, timestamp to get attempt order
    main_data = main_data.sort_values(
        ["student_id", "question_unittest_id", "timestamp"], kind="stable"
    )

    # Track attempt number per student-question pair
    main_data["attempt_num"] = main_data.groupby(["student_id", "question_unittest_id"]).cumcount()

    # Fill matrices
    for _, row in tqdm(main_data.iterrows(), total=len(main_data), desc="Building matrices"):
        s_idx = student_to_idx[row["student_id"]]
        q_id = row["question_unittest_id"]
        t_idx = row["attempt_num"]

        if t_idx >= max_attempts:
            continue

        tc_start, tc_end = qidx_to_tc_range[q_id]

        # Parse pass string to get per-testcase results
        pass_str = str(row["pass"]).strip()
        if "." in pass_str:
            try:
                pass_str = str(int(float(pass_str)))
            except ValueError:
                continue

        # Parse time
        time_val = parse_time(row["timestamp"], course_name)
        is_exam_val = int(row["is_exam"]) if pd.notna(row["is_exam"]) else 0

        for tc_offset, char in enumerate(pass_str):
            if tc_start + tc_offset < tc_end:
                item_idx = tc_start + tc_offset
                correctness_matrix[s_idx, item_idx, t_idx] = int(char)
                time_matrix[s_idx, item_idx, t_idx] = time_val
                is_exam_matrix[s_idx, item_idx, t_idx] = is_exam_val

    # Convert to tensors
    correctness_matrix = torch.tensor(correctness_matrix, device=device, dtype=torch.int8)
    time_matrix = torch.tensor(time_matrix, device=device, dtype=torch.float32)
    is_exam_matrix = torch.tensor(is_exam_matrix, device=device, dtype=torch.int8)

    # Filter out students with no valid data
    valid_students = []
    for s_idx in range(n_students):
        if (correctness_matrix[s_idx] != -1).any():
            valid_students.append(s_idx)

    if len(valid_students) < n_students:
        print(f"Filtering: {n_students} -> {len(valid_students)} students with data")
        valid_students = torch.tensor(valid_students, device=device)
        correctness_matrix = correctness_matrix[valid_students]
        time_matrix = time_matrix[valid_students]
        is_exam_matrix = is_exam_matrix[valid_students]
        student_ids = [student_ids[i] for i in valid_students.cpu().numpy()]

    # Build question info for all items
    qidx_list = []
    for item_idx in range(n_items):
        for qid, (start, end) in qidx_to_tc_range.items():
            if start <= item_idx < end:
                qidx_list.append(question_to_idx[qid])
                break

    # Derive weeks from timestamps for questions missing from question_infos
    qi_qids = set(question_infos["question_id"].values)
    missing_qids = [qid for qid in question_ids if qid not in qi_qids]
    derived_weeks = {}
    if missing_qids and single_course:
        ts_data = main_data[["question_unittest_id", "timestamp"]].copy()
        ts_data["ts"] = pd.to_datetime(ts_data["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce")
        first_ts = ts_data.dropna(subset=["ts"]).groupby("question_unittest_id")["ts"].min()
        if len(first_ts) > 0:
            course_start = first_ts.min()
            for qid in missing_qids:
                if qid in first_ts.index:
                    derived_weeks[qid] = (first_ts[qid] - course_start).days // 7 + 1
        print(f"Derived weeks for {len(derived_weeks)}/{len(missing_qids)} qids "
              f"missing from question_infos; {len(missing_qids) - len(derived_weeks)} "
              f"fall back to week 0 (dropped from train/test splits downstream)")

    # Build student info
    student_info = [{"student_id": sid} for sid in student_ids]

    # Build question info dataframe
    question_info_list = []
    for i, qidx in enumerate(qidx_list):
        qid = list(question_to_idx.keys())[list(question_to_idx.values()).index(qidx)]
        q_row = question_infos[question_infos["question_id"] == qid]
        if len(q_row) > 0:
            question_info_list.append({
                "qidx": qidx,
                "question_unittest_id": int(qid),
                "qname": q_row["question_name"].values[0] if "question_name" in q_row.columns else str(qid),
                "week": q_row["week"].values[0] if "week" in q_row.columns else 0,
                "topic": q_row["topic"].values[0] if "topic" in q_row.columns else "",
            })
        else:
            question_info_list.append({
                "qidx": qidx,
                "question_unittest_id": int(qid),
                "qname": str(qid),
                "week": derived_weeks.get(qid, 0),
                "topic": "",
            })

    return (
        student_info,
        question_info_list,
        correctness_matrix,
        time_matrix,
        is_exam_matrix,
    )


def upload_matrices(course_name, student_info, question_info, correctness_matrix, time_matrix, is_exam_matrix):
    """Upload matrices to HuggingFace."""
    upload_api = HfApi()
    repo_id = "CodeInsightTeam/code_insights_matrices"

    print(f"Uploading matrices for {course_name}...")
    print(f"  Correctness: {correctness_matrix.shape}")
    print(f"  Time: {time_matrix.shape}")
    print(f"  Is Exam: {is_exam_matrix.shape}")

    # Question info
    question_info_df = pd.DataFrame(question_info)
    question_info_file = io.BytesIO()
    question_info_df.to_csv(question_info_file, index=False)
    question_info_file.seek(0)
    upload_api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"{course_name}/question_infos.csv",
        path_or_fileobj=question_info_file,
    )

    # Student info
    student_info_df = pd.DataFrame(student_info)
    student_info_file = io.BytesIO()
    student_info_df.to_csv(student_info_file, index=False)
    student_info_file.seek(0)
    upload_api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"{course_name}/student_info.csv",
        path_or_fileobj=student_info_file,
    )

    # Correctness matrix
    correctness_file = io.BytesIO()
    torch.save(correctness_matrix.cpu(), correctness_file)
    correctness_file.seek(0)
    upload_api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"{course_name}/correctness_matrix.pt",
        path_or_fileobj=correctness_file,
    )

    # Is exam matrix
    is_exam_file = io.BytesIO()
    torch.save(is_exam_matrix.cpu(), is_exam_file)
    is_exam_file.seek(0)
    upload_api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"{course_name}/is_exam_matrix.pt",
        path_or_fileobj=is_exam_file,
    )

    # Time matrix
    time_file = io.BytesIO()
    torch.save(time_matrix.cpu(), time_file)
    time_file.seek(0)
    upload_api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"{course_name}/time_matrix.pt",
        path_or_fileobj=time_file,
    )

    print("Upload complete!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--course_name",
        type=str,
        default="dsa_hk231",
        help="Course name (dsa_hk231, dsa_hk221, pf_hk232, pf_hk222)"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace (default: just build locally)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Local output directory (if not uploading)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading CSV data for {args.course_name}...")
    main_data, question_infos, course_name = load_csv_data(args.course_name)
    print(f"Loaded {len(main_data)} submissions")

    # Build matrices
    student_info, question_info, correctness_matrix, time_matrix, is_exam_matrix = build_matrices(
        main_data, question_infos, course_name, device
    )

    print(f"\nFinal matrices:")
    print(f"  Students: {len(student_info)}")
    print(f"  Items: {correctness_matrix.shape[1]}")
    print(f"  Max attempts: {correctness_matrix.shape[2]}")

    if args.upload:
        upload_matrices(
            course_name, student_info, question_info,
            correctness_matrix, time_matrix, is_exam_matrix
        )
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(correctness_matrix.cpu(), f"{args.output_dir}/correctness_matrix.pt")
        torch.save(time_matrix.cpu(), f"{args.output_dir}/time_matrix.pt")
        torch.save(is_exam_matrix.cpu(), f"{args.output_dir}/is_exam_matrix.pt")
        pd.DataFrame(student_info).to_csv(f"{args.output_dir}/student_info.csv", index=False)
        pd.DataFrame(question_info).to_csv(f"{args.output_dir}/question_infos.csv", index=False)
        print(f"Saved to {args.output_dir}/")
    else:
        print("\nUse --upload to upload to HuggingFace or --output_dir to save locally")
