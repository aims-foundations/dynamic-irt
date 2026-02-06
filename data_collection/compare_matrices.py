"""
Compare matrices by matching question names rather than positions.

This script validates that csv2matrices.py produces the same data as the
existing matrices on HuggingFace, regardless of question ordering differences.
"""

import os
import pandas as pd
import torch
import numpy as np
from huggingface_hub import snapshot_download

def main():
    hf_token = "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"
    course_name = "dsa_hk231"

    # Load existing matrices from HuggingFace
    print("Loading existing matrices from HuggingFace...")
    existing_path = snapshot_download(
        repo_id="stair-lab/code_insights_matrices",
        repo_type="dataset",
        token=hf_token
    )

    existing_correctness = torch.load(f"{existing_path}/{course_name}/correctness_matrix.pt")
    existing_question_info = pd.read_csv(f"{existing_path}/{course_name}/question_infos.csv")
    existing_student_info = pd.read_csv(f"{existing_path}/{course_name}/student_info.csv")

    print(f"Existing path: {existing_path}")
    print(f"Existing matrices shape: {existing_correctness.shape}")
    print(f"Existing student_info columns: {existing_student_info.columns.tolist()}")
    print(f"First 5 existing student rows:\n{existing_student_info.head()}")
    print(f"Existing questions: {len(existing_question_info)}")
    print(f"Existing students: {len(existing_student_info)}")

    # Load CSV data to build new matrices
    print("\nLoading CSV data...")
    csv_cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--stair-lab--code_insights_csv/"
        "snapshots/99d53fe7c11f6302fb28b82fab5ebd77c00e5d12"
    )
    if os.path.exists(csv_cache_path):
        csv_path = csv_cache_path
    else:
        csv_path = snapshot_download(
            repo_id="stair-lab/code_insights_csv",
            repo_type="dataset",
            token=hf_token
        )

    main_data = pd.read_csv(f"{csv_path}/main_data.csv", low_memory=False)
    question_infos_csv = pd.read_csv(f"{csv_path}/question_infos.csv")
    course_infos = pd.read_csv(f"{csv_path}/course_infos.csv")
    student_infos_csv = pd.read_csv(f"{csv_path}/student_infos.csv")

    print(f"CSV student_infos columns: {student_infos_csv.columns.tolist()}")
    print(f"First 5 CSV student_infos rows:\n{student_infos_csv.head()}")

    # Get course_id
    course_row = course_infos[course_infos["course_name"] == course_name]
    course_id = course_row["course_id"].values[0]

    # Filter to course and submission types
    main_data = main_data[main_data["course_id"] == course_id].copy()
    main_data = main_data[main_data["response_type"].isin(["Submit", "Prechecked"])].copy()
    main_data = main_data.dropna(subset=["pass"])

    print(f"Filtered submissions: {len(main_data)}")

    # Build question name mapping from CSV
    # question_unittest_id in main_data maps to question_id in question_infos_csv
    qid_to_name = dict(zip(question_infos_csv["question_id"], question_infos_csv["question_name"]))

    # Get unique questions in the CSV data
    csv_question_ids = main_data["question_unittest_id"].unique()
    csv_question_names = [qid_to_name.get(qid, f"unknown_{qid}") for qid in csv_question_ids]

    print(f"\nCSV has {len(csv_question_ids)} unique questions")
    print(f"First 5 CSV questions: {csv_question_names[:5]}")

    # Get unique questions in existing matrices
    existing_qnames = existing_question_info["qname"].unique()
    print(f"Existing has {len(existing_qnames)} unique questions")
    print(f"First 5 existing questions: {list(existing_qnames[:5])}")

    # Find overlapping questions
    csv_qnames_set = set(csv_question_names)
    existing_qnames_set = set(existing_qnames)

    overlap = csv_qnames_set & existing_qnames_set
    only_csv = csv_qnames_set - existing_qnames_set
    only_existing = existing_qnames_set - csv_qnames_set

    print(f"\nQuestion overlap analysis:")
    print(f"  Overlap: {len(overlap)}")
    print(f"  Only in CSV: {len(only_csv)}")
    print(f"  Only in existing: {len(only_existing)}")

    if only_csv:
        print(f"  Sample only in CSV: {list(only_csv)[:5]}")
    if only_existing:
        print(f"  Sample only in existing: {list(only_existing)[:5]}")

    # Now let's check if the existing matrices have item-level or question-level data
    # by looking at the relationship between existing_question_info rows and n_items
    print(f"\nExisting question_info rows: {len(existing_question_info)}")
    print(f"Existing matrix items (dim 1): {existing_correctness.shape[1]}")

    # Check if there are duplicate qnames (indicating testcase expansion)
    qname_counts = existing_question_info["qname"].value_counts()
    print(f"Max testcases per question (existing): {qname_counts.max()}")
    print(f"Questions with multiple testcases: {(qname_counts > 1).sum()}")

    # For comparison, let's aggregate at the question level
    # and compare pass rates per student-question pair
    print("\n" + "="*60)
    print("Comparing data by question name...")
    print("="*60)

    # Build student mapping
    # Existing uses student_uid directly, CSV uses sequential student_id with student_uid in student_infos
    existing_students = existing_student_info["student_id"].tolist()  # These are actually UIDs

    # Map CSV student_id -> student_uid
    csv_sid_to_uid = dict(zip(student_infos_csv["student_id"], student_infos_csv["student_uid"]))

    # Get UIDs for students in this course's main_data
    csv_sids_in_data = main_data["student_id"].unique()
    csv_student_uids = [csv_sid_to_uid.get(sid, -1) for sid in csv_sids_in_data]
    csv_student_uids = [uid for uid in csv_student_uids if uid != -1]

    print(f"\nExisting students (UIDs): {len(existing_students)}")
    print(f"CSV students (UIDs): {len(csv_student_uids)}")
    print(f"Sample existing student UIDs: {existing_students[:5]}")
    print(f"Sample CSV student UIDs: {csv_student_uids[:5]}")

    # Check student overlap
    existing_students_set = set(existing_students)
    csv_students_set = set(csv_student_uids)
    student_overlap = existing_students_set & csv_students_set
    print(f"Student overlap: {len(student_overlap)}")

    # For a comprehensive comparison, use all overlapping questions and more students
    sample_questions = list(overlap)  # All overlapping questions
    sample_student_uids = list(student_overlap)[:100]  # 100 students for deeper analysis

    print(f"\nSampling {len(sample_questions)} questions and {len(sample_student_uids)} students for detailed comparison")

    # Build mappings for existing matrices
    # existing_question_info has qname -> row indices (which map to item indices)
    existing_qname_to_items = {}
    for idx, row in existing_question_info.iterrows():
        qname = row["qname"]
        if qname not in existing_qname_to_items:
            existing_qname_to_items[qname] = []
        existing_qname_to_items[qname].append(idx)

    # Build student UID to index mapping for existing matrices
    existing_student_to_idx = {sid: idx for idx, sid in enumerate(existing_students)}

    # Build CSV student UID to student_id (sequential) mapping for querying main_data
    csv_uid_to_sid = dict(zip(student_infos_csv["student_uid"], student_infos_csv["student_id"]))

    # For CSV data, get pass data per student-question
    csv_qid_to_name = {qid: qid_to_name.get(qid, f"unknown_{qid}") for qid in csv_question_ids}
    csv_name_to_qid = {}
    for qid, qname in csv_qid_to_name.items():
        if qname not in csv_name_to_qid:
            csv_name_to_qid[qname] = qid

    matches = 0
    mismatches = 0
    skipped = 0
    value_mismatches = 0  # When both have data but values differ
    existing_only = 0     # When existing has data but CSV doesn't
    csv_only = 0          # When CSV has data but existing doesn't

    for qname in sample_questions:
        if qname not in existing_qname_to_items or qname not in csv_name_to_qid:
            skipped += 1
            continue

        existing_items = existing_qname_to_items[qname]
        csv_qid = csv_name_to_qid[qname]

        for student_uid in sample_student_uids:
            if student_uid not in existing_student_to_idx:
                skipped += 1
                continue

            existing_sidx = existing_student_to_idx[student_uid]

            # Get existing data for this student-question (all testcases, first attempt)
            existing_vals = []
            for item_idx in existing_items:
                val = existing_correctness[existing_sidx, item_idx, 0].item()
                if val != -1:
                    existing_vals.append(val)

            # Get CSV student_id (sequential) from UID
            csv_sid = csv_uid_to_sid.get(student_uid)
            if csv_sid is None:
                skipped += 1
                continue

            # Get CSV data for this student-question (first attempt)
            csv_subset = main_data[
                (main_data["student_id"] == csv_sid) &
                (main_data["question_unittest_id"] == csv_qid)
            ].sort_values("timestamp")

            if len(csv_subset) > 0:
                first_row = csv_subset.iloc[0]
                pass_str = str(first_row["pass"]).strip()
                if "." in pass_str:
                    try:
                        pass_str = str(int(float(pass_str)))
                    except ValueError:
                        pass_str = ""
                csv_vals = [int(c) for c in pass_str if c in "01"]
            else:
                csv_vals = []

            # Compare
            if len(existing_vals) == 0 and len(csv_vals) == 0:
                matches += 1
            elif len(existing_vals) > 0 and len(csv_vals) > 0:
                # Compare the values
                min_len = min(len(existing_vals), len(csv_vals))
                if existing_vals[:min_len] == csv_vals[:min_len]:
                    matches += 1
                else:
                    mismatches += 1
                    value_mismatches += 1
                    if value_mismatches <= 3:
                        print(f"\nValue mismatch for student_uid={student_uid}, question={qname}:")
                        print(f"  Existing (first {min_len}): {existing_vals[:min_len]}")
                        print(f"  CSV (first {min_len}): {csv_vals[:min_len]}")
            elif len(existing_vals) > 0:
                # Existing has data, CSV doesn't
                mismatches += 1
                existing_only += 1
                if existing_only <= 2:
                    print(f"\nExisting-only for student_uid={student_uid}, question={qname}:")
                    print(f"  Existing vals: {existing_vals}")
            else:
                # CSV has data, existing doesn't
                mismatches += 1
                csv_only += 1
                if csv_only <= 2:
                    print(f"\nCSV-only for student_uid={student_uid}, question={qname}:")
                    print(f"  CSV vals: {csv_vals}")

    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Total comparisons: {matches + mismatches}")
    print(f"Matches (both empty or same values): {matches}")
    print(f"Mismatches breakdown:")
    print(f"  - Value mismatches (both have data, differ): {value_mismatches}")
    print(f"  - Existing-only (existing has data, CSV empty): {existing_only}")
    print(f"  - CSV-only (CSV has data, existing empty): {csv_only}")
    print(f"Skipped: {skipped}")

    if matches + mismatches > 0:
        match_rate = matches / (matches + mismatches) * 100
        print(f"\nOverall match rate: {match_rate:.1f}%")
        if value_mismatches > 0:
            print(f"Value match rate (when both have data): {(matches - (existing_only + csv_only)) / ((matches - (existing_only + csv_only)) + value_mismatches) * 100:.1f}%")

if __name__ == "__main__":
    main()
