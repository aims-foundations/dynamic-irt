"""Debug existing-only cases - where existing has data but CSV doesn't."""

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

    course_id = course_infos[course_infos["course_name"] == course_name]["course_id"].values[0]

    # Build mappings
    csv_uid_to_sid = dict(zip(student_infos_csv["student_uid"], student_infos_csv["student_id"]))
    qname_to_qid = dict(zip(question_infos_csv["question_name"], question_infos_csv["question_id"]))

    # Find existing-only cases
    existing_students = existing_student_info["student_id"].tolist()
    existing_only_cases = []

    for sidx, student_uid in enumerate(existing_students[:100]):  # Check first 100 students
        csv_sid = csv_uid_to_sid.get(student_uid)

        for qname in existing_question_info["qname"].unique():
            qid = qname_to_qid.get(qname)
            same_q_items = existing_question_info[existing_question_info["qname"] == qname].index.tolist()

            # Get existing data
            existing_has_data = any(
                existing_correctness[sidx, item_idx, 0].item() != -1
                for item_idx in same_q_items
            )

            if not existing_has_data:
                continue

            # Check CSV data
            if csv_sid is None:
                existing_only_cases.append({
                    "student_uid": student_uid,
                    "qname": qname,
                    "reason": "student_not_in_csv"
                })
                continue

            if qid is None:
                existing_only_cases.append({
                    "student_uid": student_uid,
                    "qname": qname,
                    "reason": "question_not_in_csv"
                })
                continue

            # Get CSV submissions
            csv_subs = main_data[
                (main_data["student_id"] == csv_sid) &
                (main_data["question_unittest_id"] == qid) &
                (main_data["course_id"] == course_id) &
                (main_data["response_type"].isin(["Submit", "Prechecked"]))
            ]

            if len(csv_subs) == 0:
                # Check if there are ANY submissions (not just Submit/Prechecked)
                all_subs = main_data[
                    (main_data["student_id"] == csv_sid) &
                    (main_data["question_unittest_id"] == qid) &
                    (main_data["course_id"] == course_id)
                ]
                if len(all_subs) == 0:
                    existing_only_cases.append({
                        "student_uid": student_uid,
                        "qname": qname,
                        "reason": "no_submissions_in_csv"
                    })
                else:
                    response_types = all_subs["response_type"].unique().tolist()
                    existing_only_cases.append({
                        "student_uid": student_uid,
                        "qname": qname,
                        "reason": f"filtered_out:response_types={response_types}"
                    })

            if len(existing_only_cases) >= 20:
                break

        if len(existing_only_cases) >= 20:
            break

    print("=" * 70)
    print("EXISTING-ONLY CASES (existing has data, CSV doesn't)")
    print("=" * 70)
    for case in existing_only_cases[:20]:
        print(f"  Student {case['student_uid']}, Q='{case['qname'][:40]}': {case['reason']}")

    # Count reasons
    from collections import Counter
    reasons = Counter(c["reason"].split(":")[0] for c in existing_only_cases)
    print(f"\nReason distribution:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")

if __name__ == "__main__":
    main()
