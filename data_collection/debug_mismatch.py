"""Debug specific mismatches between existing matrices and CSV data."""

import os
import pandas as pd
import torch
from huggingface_hub import snapshot_download

def main():
    hf_token = "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"
    course_name = "dsa_hk231"

    # Load existing matrices
    existing_path = snapshot_download(
        repo_id="stair-lab/code_insights_matrices",
        repo_type="dataset",
        token=hf_token
    )
    existing_correctness = torch.load(f"{existing_path}/{course_name}/correctness_matrix.pt")
    existing_question_info = pd.read_csv(f"{existing_path}/{course_name}/question_infos.csv")
    existing_student_info = pd.read_csv(f"{existing_path}/{course_name}/student_info.csv")

    # Load CSV data
    csv_cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--stair-lab--code_insights_csv/"
        "snapshots/99d53fe7c11f6302fb28b82fab5ebd77c00e5d12"
    )
    csv_path = csv_cache_path if os.path.exists(csv_cache_path) else snapshot_download(
        repo_id="stair-lab/code_insights_csv", repo_type="dataset", token=hf_token
    )

    main_data = pd.read_csv(f"{csv_path}/main_data.csv", low_memory=False)
    question_infos_csv = pd.read_csv(f"{csv_path}/question_infos.csv")
    course_infos = pd.read_csv(f"{csv_path}/course_infos.csv")
    student_infos_csv = pd.read_csv(f"{csv_path}/student_infos.csv")

    # Get course_id
    course_id = course_infos[course_infos["course_name"] == course_name]["course_id"].values[0]

    # Debug case 1: Value mismatch
    # student_uid=2211873, question="10 - Book management software (3)"
    student_uid = 2211873
    question_name = "10 - Book management software (3)"

    print("=" * 70)
    print(f"DEBUGGING: student_uid={student_uid}, question='{question_name}'")
    print("=" * 70)

    # Get existing data
    existing_sidx = list(existing_student_info["student_id"]).index(student_uid)
    existing_items = existing_question_info[existing_question_info["qname"] == question_name].index.tolist()
    print(f"\nExisting matrix:")
    print(f"  Student index: {existing_sidx}")
    print(f"  Item indices for this question: {existing_items}")
    existing_vals = [existing_correctness[existing_sidx, idx, 0].item() for idx in existing_items]
    print(f"  First attempt values: {existing_vals}")

    # Get CSV data
    csv_sid = student_infos_csv[student_infos_csv["student_uid"] == student_uid]["student_id"].values[0]
    qid = question_infos_csv[question_infos_csv["question_name"] == question_name]["question_id"].values
    if len(qid) == 0:
        print(f"\nQuestion '{question_name}' not found in question_infos_csv")
        # Try partial match
        matches = question_infos_csv[question_infos_csv["question_name"].str.contains("Book management", na=False)]
        print(f"Partial matches: {matches[['question_id', 'question_name']].to_string()}")
        return
    qid = qid[0]

    print(f"\nCSV data:")
    print(f"  CSV student_id: {csv_sid}")
    print(f"  Question ID: {qid}")

    # Get all submissions for this student-question (unfiltered)
    all_submissions = main_data[
        (main_data["student_id"] == csv_sid) &
        (main_data["question_unittest_id"] == qid) &
        (main_data["course_id"] == course_id)
    ].sort_values("timestamp")

    print(f"\n  ALL submissions for this student-question:")
    print(f"  Total: {len(all_submissions)}")
    if len(all_submissions) > 0:
        print(all_submissions[["timestamp", "response_type", "pass", "attempt_id"]].head(10).to_string())

    # Get filtered submissions (Submit/Prechecked only)
    filtered = all_submissions[all_submissions["response_type"].isin(["Submit", "Prechecked"])]
    print(f"\n  FILTERED submissions (Submit/Prechecked only):")
    print(f"  Total: {len(filtered)}")
    if len(filtered) > 0:
        print(f"  FIRST 5:")
        print(filtered[["timestamp", "response_type", "pass", "attempt_id"]].head(5).to_string())
        print(f"\n  LAST 5:")
        print(filtered[["timestamp", "response_type", "pass", "attempt_id"]].tail(5).to_string())

        # Check if any attempt matches the existing values
        print(f"\n  Looking for attempt matching existing {existing_vals}:")
        for idx, row in filtered.iterrows():
            pass_str = str(row["pass"]).strip()
            if pass_str and all(c in "01" for c in pass_str):
                pass_vals = [int(c) for c in pass_str]
                if pass_vals == existing_vals:
                    print(f"    MATCH at attempt_id={row['attempt_id']}: {pass_str}")
                    break
        else:
            print(f"    No matching attempt found")

    # Check what response_types exist
    print(f"\n  Response types in all submissions: {all_submissions['response_type'].unique().tolist()}")

    # Debug case 2: CSV-only (CSV has data, existing empty)
    print("\n" + "=" * 70)
    print("DEBUGGING CSV-ONLY CASE")
    print("=" * 70)

    # Find a question where CSV has data but existing doesn't for some student
    # Let's check "P2-06-Application-Find Closest Pair" from the comparison output
    question_name2 = "P2-06-Application-Find Closest Pair"
    student_uid2 = 2252808

    existing_items2 = existing_question_info[existing_question_info["qname"] == question_name2].index.tolist()
    if not existing_items2:
        print(f"Question '{question_name2}' not found in existing")
    else:
        existing_sidx2 = list(existing_student_info["student_id"]).index(student_uid2) if student_uid2 in existing_student_info["student_id"].values else -1
        if existing_sidx2 >= 0:
            existing_vals2 = [existing_correctness[existing_sidx2, idx, 0].item() for idx in existing_items2]
            print(f"Existing values for student {student_uid2}: {existing_vals2}")

    csv_sid2 = student_infos_csv[student_infos_csv["student_uid"] == student_uid2]["student_id"].values
    if len(csv_sid2) > 0:
        csv_sid2 = csv_sid2[0]
        qid2 = question_infos_csv[question_infos_csv["question_name"] == question_name2]["question_id"].values
        if len(qid2) > 0:
            qid2 = qid2[0]
            all_sub2 = main_data[
                (main_data["student_id"] == csv_sid2) &
                (main_data["question_unittest_id"] == qid2) &
                (main_data["course_id"] == course_id)
            ]
            print(f"\nCSV submissions for student {student_uid2}, question '{question_name2}':")
            print(f"Total: {len(all_sub2)}")
            if len(all_sub2) > 0:
                print(all_sub2[["timestamp", "response_type", "pass", "attempt_id"]].to_string())

if __name__ == "__main__":
    main()
