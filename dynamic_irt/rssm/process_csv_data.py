"""Download CodeInsights CSV from HuggingFace and extract multi-modal features.

Usage:
    python process_csv_data.py --course dsa_hk231 --output_dir data/multimodal/dsa_hk231
    python process_csv_data.py --course all       --output_dir data/multimodal/all
"""

import argparse
import math
import os
import pickle

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

from feature_config import FeatureConfig


def remove_decimal_if_whole(val):
    """Clean pass column values (e.g., '111.0' -> '111'). Matches elo.py."""
    try:
        val_str = str(val)
        if "." in val_str:
            num = float(val_str)
            if num.is_integer():
                return str(int(num))
            return val_str
        return val_str
    except ValueError:
        return str(val)


def load_csv_data(course):
    """Download and load CSV data from HuggingFace.

    Args:
        course: Course name (e.g., 'dsa_hk231') or 'all' for all courses.

    Returns:
        main_data (DataFrame), question_infos (DataFrame)
    """
    path = snapshot_download(
        repo_id="stair-lab/code_insights_csv", repo_type="dataset"
    )
    main_data = pd.read_csv(f"{path}/main_data.csv", low_memory=False)
    question_infos = pd.read_csv(f"{path}/question_infos.csv")

    if course != "all":
        course_infos = pd.read_csv(f"{path}/course_infos.csv")
        course_row = course_infos[course_infos["course_name"] == course]
        if len(course_row) == 0:
            available = course_infos["course_name"].tolist()
            raise ValueError(f"Course '{course}' not found. Available: {available}")
        course_id = course_row["course_id"].values[0]
        main_data = main_data[main_data["course_id"] == course_id].copy()

    # Filter to actual submissions
    main_data = main_data[
        main_data["response_type"].isin(["Submit", "Prechecked"])
    ].copy()

    # Clean pass column (matches elo.py lines 204-218)
    main_data["pass"] = main_data["pass"].apply(remove_decimal_if_whole)
    main_data["pass"] = main_data["pass"].replace("nan", np.nan)
    main_data = main_data.dropna(subset=["pass"])

    # Parse timestamps (matches elo.py lines 221-222)
    main_data["timestamp"] = pd.to_datetime(
        main_data["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
    )
    main_data = main_data.dropna(subset=["timestamp"])

    # Sort by student and time
    main_data = main_data.sort_values(["student_id", "timestamp"]).reset_index(
        drop=True
    )

    return main_data, question_infos


def build_question_lookup(main_data, question_infos, n_testcases=15):
    """Build question feature lookup from data.

    Returns:
        question_to_idx: dict mapping question_unittest_id -> integer index
        question_static: np.array of shape [n_questions, 3] with
            (empirical_difficulty, n_testcases_norm, week_norm)
    """
    question_ids = sorted(main_data["question_unittest_id"].unique())
    question_to_idx = {qid: idx for idx, qid in enumerate(question_ids)}

    # Compute empirical difficulty per question
    question_stats = {}
    for qid, group in main_data.groupby("question_unittest_id"):
        pass_rates = group["pass"].apply(
            lambda s: sum(c == "1" for c in str(s)) / max(len(str(s)), 1)
        )
        n_tc = group["pass"].apply(lambda s: len(str(s))).median()
        question_stats[qid] = {
            "difficulty": 1.0 - pass_rates.mean(),
            "n_testcases": n_tc,
        }

    # Try to get week info from question_infos
    week_lookup = {}
    if "week" in question_infos.columns and "question_id" in question_infos.columns:
        for _, row in question_infos.iterrows():
            week_lookup[row["question_id"]] = row.get("week", 0)

    max_week = max(week_lookup.values()) if week_lookup else 1.0
    if max_week == 0:
        max_week = 1.0

    # Build static feature array
    n_questions = len(question_ids)
    question_static = np.zeros((n_questions, 3), dtype=np.float32)
    for qid, idx in question_to_idx.items():
        stats = question_stats.get(qid, {"difficulty": 0.5, "n_testcases": 5})
        week = week_lookup.get(qid, 0)
        question_static[idx, 0] = stats["difficulty"]
        question_static[idx, 1] = stats["n_testcases"] / n_testcases  # normalize
        question_static[idx, 2] = week / max_week  # normalize

    return question_to_idx, question_static


def extract_features(main_data, question_to_idx, config):
    """Extract per-interaction multi-modal features.

    Returns:
        answer_features: list of np.array, each shape [answer_dim]
        question_idxs: list of int
        testcase_scores: list of np.array, each shape [n_testcases]
        student_idxs: list of int
    """
    n_tc = config.n_testcases
    answer_dim = config.answer_dim

    answer_features = []
    question_idxs = []
    testcase_scores = []
    student_idxs = []

    # Build student index mapping
    unique_students = sorted(main_data["student_id"].unique())
    student_to_idx = {sid: idx for idx, sid in enumerate(unique_students)}

    # Per-student state tracking
    student_state = {}

    for _, row in tqdm(
        main_data.iterrows(), total=len(main_data), desc="Extracting features"
    ):
        sid = row["student_id"]
        qid = row["question_unittest_id"]

        if qid not in question_to_idx:
            continue

        # Initialize student state if needed
        if sid not in student_state:
            student_state[sid] = {
                "first_timestamp": row["timestamp"],
                "last_timestamp": None,
                "cumulative_attempts": 0,
                "recent_scores": [],  # last 10 pass rates
                "cumulative_correct": 0.0,
                "cumulative_total": 0,
                "question_attempts": {},  # qid -> count
                "unique_questions": set(),
                "last_response": {},  # qid -> last response text
            }

        state = student_state[sid]

        # --- Parse pass string ---
        pass_str = str(row["pass"])
        tc_vector = np.full(n_tc, -1.0, dtype=np.float32)
        for i, ch in enumerate(pass_str):
            if i >= n_tc:
                break
            tc_vector[i] = float(ch == "1")

        valid_tc = [v for v in tc_vector if v >= 0]
        pass_rate = np.mean(valid_tc) if valid_tc else 0.0
        is_perfect = 1.0 if all(v == 1.0 for v in valid_tc) else 0.0
        actual_n_tc = len(valid_tc) / n_tc  # normalized

        # --- Build feature vector ---
        features = []

        # Group A: Performance
        if config.use_performance:
            features.extend(tc_vector.tolist())
            features.append(pass_rate)
            features.append(is_perfect)
            features.append(actual_n_tc)

        # Group B: Temporal
        if config.use_temporal:
            if state["last_timestamp"] is not None:
                time_since_last = (
                    row["timestamp"] - state["last_timestamp"]
                ).total_seconds()
                log_time = math.log1p(max(time_since_last, 0))
            else:
                log_time = 0.0

            q_attempt_num = state["question_attempts"].get(qid, 0)
            cum_attempts = state["cumulative_attempts"]
            is_exam = float(row.get("is_exam", 0) or 0)

            days_since_start = (
                row["timestamp"] - state["first_timestamp"]
            ).total_seconds() / 86400.0

            # Week: extract from question_infos join or estimate from days
            week_num = days_since_start / 7.0

            features.append(log_time)
            features.append(math.log1p(q_attempt_num))
            features.append(math.log1p(cum_attempts))
            features.append(is_exam)
            features.append(min(week_num / 20.0, 1.0))  # normalize by ~20 weeks
            features.append(math.log1p(days_since_start))

        # Group C: Code Structural
        if config.use_code_struct:
            response = str(row.get("response", ""))
            code_length = len(response)
            line_count = response.count("\n") + 1

            # Edit distance from previous attempt on same question
            prev_response = state["last_response"].get(qid)
            if prev_response is not None and code_length > 0:
                # Truncate for efficiency
                trunc_curr = response[:2000]
                trunc_prev = prev_response[:2000]
                edit_dist = levenshtein_distance(trunc_curr, trunc_prev)
                code_ratio = len(response) / max(len(prev_response), 1)
            else:
                edit_dist = 0
                code_ratio = 1.0

            features.append(math.log1p(code_length))
            features.append(math.log1p(line_count))
            features.append(math.log1p(edit_dist))
            features.append(min(code_ratio, 5.0) / 5.0)  # normalize to [0, 1]

        # Group D: Student State
        if config.use_student_state:
            recent = state["recent_scores"][-10:]
            running_avg = np.mean(recent) if recent else 0.0

            cum_ratio = (
                state["cumulative_correct"] / max(state["cumulative_total"], 1)
            )

            # Improvement trend: last 3 avg > prior 3 avg
            if len(recent) >= 6:
                trend = 1.0 if np.mean(recent[-3:]) > np.mean(recent[-6:-3]) else 0.0
            else:
                trend = 0.0

            n_unique_q = len(state["unique_questions"])

            features.append(running_avg)
            features.append(cum_ratio)
            features.append(trend)
            features.append(math.log1p(n_unique_q))

        # --- Store ---
        answer_features.append(np.array(features, dtype=np.float32))
        question_idxs.append(question_to_idx[qid])
        testcase_scores.append(tc_vector)
        student_idxs.append(student_to_idx[sid])

        # --- Update student state ---
        state["last_timestamp"] = row["timestamp"]
        state["cumulative_attempts"] += 1
        state["recent_scores"].append(pass_rate)
        state["cumulative_correct"] += pass_rate
        state["cumulative_total"] += 1
        state["question_attempts"][qid] = (
            state["question_attempts"].get(qid, 0) + 1
        )
        state["unique_questions"].add(qid)
        if config.use_code_struct:
            state["last_response"][qid] = str(row.get("response", ""))

    return answer_features, question_idxs, testcase_scores, student_idxs


def main():
    parser = argparse.ArgumentParser(description="Process CSV data for Multi-Modal RSSM")
    parser.add_argument(
        "--course", type=str, default="dsa_hk231",
        help="Course name or 'all' for all courses",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: data/multimodal/{course})",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"data/multimodal/{args.course}"

    config = FeatureConfig()

    print(f"Loading CSV data for course: {args.course}")
    main_data, question_infos = load_csv_data(args.course)
    print(f"  Loaded {len(main_data)} submissions")

    print("Building question lookup...")
    question_to_idx, question_static = build_question_lookup(
        main_data, question_infos, config.n_testcases
    )
    n_questions = len(question_to_idx)
    print(f"  {n_questions} unique questions")

    print(f"Extracting features (answer_dim={config.answer_dim})...")
    answer_features, question_idxs, testcase_scores, student_idxs = extract_features(
        main_data, question_to_idx, config
    )
    print(f"  {len(answer_features)} interactions from "
          f"{len(set(student_idxs))} students")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    with open(f"{args.output_dir}/answer_features.pkl", "wb") as f:
        pickle.dump(answer_features, f)
    with open(f"{args.output_dir}/question_idxs.pkl", "wb") as f:
        pickle.dump(question_idxs, f)
    with open(f"{args.output_dir}/question_static.pkl", "wb") as f:
        pickle.dump(question_static, f)
    with open(f"{args.output_dir}/testcase_scores.pkl", "wb") as f:
        pickle.dump(testcase_scores, f)
    with open(f"{args.output_dir}/student_idxs.pkl", "wb") as f:
        pickle.dump(student_idxs, f)

    metadata = {
        "n_students": len(set(student_idxs)),
        "n_questions": n_questions,
        "n_interactions": len(answer_features),
        "answer_dim": config.answer_dim,
        "n_testcases": config.n_testcases,
        "question_to_idx": question_to_idx,
        "course": args.course,
    }
    with open(f"{args.output_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nSaved to {args.output_dir}/")
    print(f"  answer_features: {len(answer_features)} x {config.answer_dim}")
    print(f"  question_static: {question_static.shape}")
    print(f"  n_students: {metadata['n_students']}")
    print(f"  n_questions: {metadata['n_questions']}")


if __name__ == "__main__":
    main()
