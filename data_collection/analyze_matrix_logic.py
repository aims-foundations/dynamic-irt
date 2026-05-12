"""Analyze what attempt the existing matrices store."""

import os
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from collections import Counter

def main():
    hf_token = "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"
    course_name = "dsa_hk231"

    # Load existing matrices
    existing_path = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_matrices",
        repo_type="dataset",
        token=hf_token
    )
    existing_correctness = torch.load(f"{existing_path}/{course_name}/correctness_matrix.pt")
    existing_question_info = pd.read_csv(f"{existing_path}/{course_name}/question_infos.csv")
    existing_student_info = pd.read_csv(f"{existing_path}/{course_name}/student_info.csv")

    # Load CSV data
    csv_cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--CodeInsightTeam--code_insights_csv/"
        "snapshots/99d53fe7c11f6302fb28b82fab5ebd77c00e5d12"
    )
    csv_path = csv_cache_path if os.path.exists(csv_cache_path) else snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv", repo_type="dataset", token=hf_token
    )

    main_data = pd.read_csv(f"{csv_path}/main_data.csv", low_memory=False)
    question_infos_csv = pd.read_csv(f"{csv_path}/question_infos.csv")
    course_infos = pd.read_csv(f"{csv_path}/course_infos.csv")
    student_infos_csv = pd.read_csv(f"{csv_path}/student_infos.csv")

    course_id = course_infos[course_infos["course_name"] == course_name]["course_id"].values[0]

    # Build mappings
    csv_uid_to_sid = dict(zip(student_infos_csv["student_uid"], student_infos_csv["student_id"]))
    qname_to_qid = dict(zip(question_infos_csv["question_name"], question_infos_csv["question_id"]))

    # For each existing student-question with data, find which attempt matches
    match_types = Counter()
    sample_count = 0
    max_samples = 200  # Check 200 cases

    existing_students = existing_student_info["student_id"].tolist()

    for sidx, student_uid in enumerate(existing_students[:50]):  # Check 50 students
        csv_sid = csv_uid_to_sid.get(student_uid)
        if csv_sid is None:
            continue

        for qidx, row in existing_question_info.iterrows():
            if sample_count >= max_samples:
                break

            qname = row["qname"]
            qid = qname_to_qid.get(qname)
            if qid is None:
                continue

            # Get existing value for first attempt
            existing_val = existing_correctness[sidx, qidx, 0].item()
            if existing_val == -1:
                continue

            # Get all CSV submissions for this student-question
            csv_subs = main_data[
                (main_data["student_id"] == csv_sid) &
                (main_data["question_unittest_id"] == qid) &
                (main_data["course_id"] == course_id) &
                (main_data["response_type"].isin(["Submit", "Prechecked"]))
            ].sort_values("timestamp")

            if len(csv_subs) == 0:
                match_types["no_csv_data"] += 1
                continue

            # Get the testcase index within the question
            # existing_question_info has one row per testcase, qidx is the item index
            # We need to find which testcase this is
            same_q_items = existing_question_info[existing_question_info["qname"] == qname].index.tolist()
            tc_offset = same_q_items.index(qidx) if qidx in same_q_items else 0

            found_match = False
            for sub_idx, (_, sub_row) in enumerate(csv_subs.iterrows()):
                pass_str = str(sub_row["pass"]).strip()
                if not pass_str or not all(c in "01" for c in pass_str):
                    continue
                if tc_offset < len(pass_str):
                    csv_val = int(pass_str[tc_offset])
                    if csv_val == existing_val:
                        if sub_idx == 0:
                            match_types["first_attempt"] += 1
                        elif sub_idx == len(csv_subs) - 1:
                            match_types["last_attempt"] += 1
                        else:
                            match_types[f"middle_attempt_{sub_idx}"] += 1
                        found_match = True
                        break

            if not found_match:
                match_types["no_match"] += 1

            sample_count += 1

        if sample_count >= max_samples:
            break

    print("=" * 60)
    print("ANALYSIS: Which attempt does existing matrix store?")
    print("=" * 60)
    print(f"Total samples analyzed: {sample_count}")
    print(f"\nMatch type distribution:")
    for key, count in sorted(match_types.items(), key=lambda x: -x[1]):
        pct = count / sample_count * 100 if sample_count > 0 else 0
        print(f"  {key}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
