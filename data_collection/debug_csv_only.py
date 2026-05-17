"""Debug CSV-only cases - where CSV has data but existing doesn't."""

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

    # Filter to course
    main_data_course = main_data[main_data["course_id"] == course_id].copy()

    # Build mappings
    csv_sid_to_uid = dict(zip(student_infos_csv["student_id"], student_infos_csv["student_uid"]))
    existing_student_to_idx = {uid: idx for idx, uid in enumerate(existing_student_info["student_id"])}
    qid_to_qname = dict(zip(question_infos_csv["question_id"], question_infos_csv["question_name"]))

    # Build existing qname to items mapping
    existing_qname_to_items = {}
    for idx, row in existing_question_info.iterrows():
        qname = row["qname"]
        if qname not in existing_qname_to_items:
            existing_qname_to_items[qname] = []
        existing_qname_to_items[qname].append(idx)

    csv_only_cases = []
    reasons = Counter()

    # Get all student-question pairs with Submit/Prechecked in CSV
    submissions = main_data_course[main_data_course["response_type"].isin(["Submit", "Prechecked"])]
    submissions = submissions.dropna(subset=["pass"])

    # Group by student-question
    student_questions = submissions.groupby(["student_id", "question_unittest_id"]).first().reset_index()

    print(f"Total student-question pairs in CSV: {len(student_questions)}")

    for _, row in student_questions.iterrows():
        csv_sid = row["student_id"]
        qid = row["question_unittest_id"]

        student_uid = csv_sid_to_uid.get(csv_sid)
        qname = qid_to_qname.get(qid)

        if student_uid is None or qname is None:
            continue

        # Check if in existing
        existing_sidx = existing_student_to_idx.get(student_uid)
        existing_items = existing_qname_to_items.get(qname)

        if existing_sidx is None:
            reasons["student_not_in_existing"] += 1
            if len(csv_only_cases) < 5:
                csv_only_cases.append({
                    "csv_sid": csv_sid,
                    "student_uid": student_uid,
                    "qid": qid,
                    "qname": qname,
                    "reason": "student_not_in_existing"
                })
            continue

        if existing_items is None:
            reasons["question_not_in_existing"] += 1
            if len(csv_only_cases) < 10:
                csv_only_cases.append({
                    "csv_sid": csv_sid,
                    "student_uid": student_uid,
                    "qid": qid,
                    "qname": qname,
                    "reason": "question_not_in_existing"
                })
            continue

        # Check if existing has data
        existing_has_data = any(
            existing_correctness[existing_sidx, item_idx, 0].item() != -1
            for item_idx in existing_items
        )

        if not existing_has_data:
            reasons["existing_has_no_data"] += 1
            if len(csv_only_cases) < 15:
                csv_only_cases.append({
                    "csv_sid": csv_sid,
                    "student_uid": student_uid,
                    "qid": qid,
                    "qname": qname,
                    "reason": "existing_has_no_data"
                })

    print("\n" + "=" * 70)
    print("CSV-ONLY CASES ANALYSIS")
    print("=" * 70)
    print(f"\nReason distribution:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")

    print(f"\nSample cases:")
    for case in csv_only_cases:
        print(f"  {case['reason']}: student_uid={case['student_uid']}, qname='{case['qname'][:50]}'")

    # Investigate "existing_has_no_data" cases more
    if reasons["existing_has_no_data"] > 0:
        print("\n" + "-" * 40)
        print("Investigating 'existing_has_no_data' cases:")
        case = [c for c in csv_only_cases if c["reason"] == "existing_has_no_data"][0]
        print(f"\nCase: student_uid={case['student_uid']}, qname='{case['qname']}'")

        # Get CSV submissions
        csv_subs = submissions[
            (submissions["student_id"] == case["csv_sid"]) &
            (submissions["question_unittest_id"] == case["qid"])
        ].sort_values("timestamp")
        print(f"CSV submissions: {len(csv_subs)}")
        print(csv_subs[["timestamp", "response_type", "pass", "attempt_id"]].head(5).to_string())

        # Get existing matrix values
        existing_sidx = existing_student_to_idx[case["student_uid"]]
        existing_items = existing_qname_to_items[case["qname"]]
        print(f"\nExisting items for this question: {existing_items}")
        for item_idx in existing_items[:5]:
            val = existing_correctness[existing_sidx, item_idx, 0].item()
            print(f"  Item {item_idx}: {val}")

if __name__ == "__main__":
    main()
