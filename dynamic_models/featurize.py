"""Featurize student interaction data for learning dynamics models.

Two modes:
  - features: Extract handcrafted multi-modal features (32-dim) from CSV
  - embeddings: Generate LLM text embeddings (4096-dim) via vLLM

Usage:
    cd CodeInsights
    # Handcrafted features
    python -m dynamic_irt.featurize --mode features --course dsa_hk231
    python -m dynamic_irt.featurize --mode features --course all --output_dir data/multimodal/all

    # LLM embeddings
    python -m dynamic_irt.featurize --mode embeddings \
        --dataset stair-lab/dsa_hk231_wtc_per_student_sft_lf_splited \
        --cls train --model /path/to/Llama-3.1-8B-embedding
"""

import argparse
import math
import os
import pickle
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configs (model-agnostic — any learning dynamics model can use these)
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Configuration for handcrafted multi-modal features.

    Feature Groups:
        A - Performance (18 dims): testcase pass/fail vector, pass_rate, is_perfect, n_testcases
        B - Temporal (6 dims): time_since_last, attempt_num, cumulative_attempts, is_exam, week, days
        C - Code Structural (4 dims): code_length, line_count, edit_distance, code_length_ratio
        D - Student State (4 dims): running_avg, cumulative_ratio, improvement_trend, unique_questions
        E - Question (always on, 19 dims): embedding(16) + difficulty + n_testcases + week
    """
    use_performance: bool = True
    use_temporal: bool = True
    use_code_struct: bool = True
    use_student_state: bool = True
    use_aux_loss: bool = True

    n_testcases: int = 15
    question_emb_dim: int = 16
    question_static_dim: int = 3

    @property
    def performance_dim(self) -> int:
        return self.n_testcases + 3

    @property
    def temporal_dim(self) -> int:
        return 6

    @property
    def code_struct_dim(self) -> int:
        return 4

    @property
    def student_state_dim(self) -> int:
        return 4

    @property
    def answer_dim(self) -> int:
        dim = 0
        if self.use_performance:
            dim += self.performance_dim
        if self.use_temporal:
            dim += self.temporal_dim
        if self.use_code_struct:
            dim += self.code_struct_dim
        if self.use_student_state:
            dim += self.student_state_dim
        return dim

    @property
    def question_dim(self) -> int:
        return self.question_emb_dim + self.question_static_dim


@dataclass
class EmbeddingConfig:
    """Configuration for LLM embedding mode."""
    emb_dim: int = 4096
    n_testcases: int = 15
    use_aux_loss: bool = True

    @property
    def answer_dim(self) -> int:
        return self.emb_dim


CONFIGS = {
    "full": FeatureConfig(),
    "performance_only": FeatureConfig(
        use_temporal=False, use_code_struct=False, use_student_state=False
    ),
    "no_code": FeatureConfig(use_code_struct=False),
    "minimal": FeatureConfig(use_code_struct=False, use_student_state=False),
    "no_aux": FeatureConfig(use_aux_loss=False),
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def remove_decimal_if_whole(val):
    """Clean pass column values (e.g., '111.0' -> '111')."""
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


# ---------------------------------------------------------------------------
# Feature mode: handcrafted multi-modal features from CSV
# ---------------------------------------------------------------------------

def load_csv_data(course):
    """Download and load CSV data from HuggingFace."""
    path = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_csv", repo_type="dataset"
    )
    main_data = pd.read_csv(f"{path}/main_data.csv", low_memory=False, on_bad_lines="skip")
    question_infos = pd.read_csv(f"{path}/question_infos.csv")

    if course != "all":
        course_infos = pd.read_csv(f"{path}/course_infos.csv")
        course_row = course_infos[course_infos["course_name"] == course]
        if len(course_row) == 0:
            available = course_infos["course_name"].tolist()
            raise ValueError(f"Course '{course}' not found. Available: {available}")
        course_id = course_row["course_id"].values[0]
        main_data = main_data[main_data["course_id"] == course_id].copy()

    main_data = main_data[
        main_data["response_type"].isin(["Submit", "Prechecked"])
    ].copy()

    main_data["pass"] = main_data["pass"].apply(remove_decimal_if_whole)
    main_data["pass"] = main_data["pass"].replace("nan", np.nan)
    main_data = main_data.dropna(subset=["pass"])

    main_data["timestamp"] = pd.to_datetime(
        main_data["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
    )
    main_data = main_data.dropna(subset=["timestamp"])
    main_data = main_data.sort_values(["student_id", "timestamp"]).reset_index(drop=True)

    return main_data, question_infos


def build_question_lookup(main_data, question_infos, n_testcases=15):
    """Build question feature lookup from data."""
    question_ids = sorted(main_data["question_unittest_id"].unique())
    question_to_idx = {qid: idx for idx, qid in enumerate(question_ids)}

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

    week_lookup = {}
    if "week" in question_infos.columns and "question_id" in question_infos.columns:
        for _, row in question_infos.iterrows():
            week_lookup[row["question_id"]] = row.get("week", 0)

    max_week = max(week_lookup.values()) if week_lookup else 1.0
    if max_week == 0:
        max_week = 1.0

    n_questions = len(question_ids)
    question_static = np.zeros((n_questions, 3), dtype=np.float32)
    for qid, idx in question_to_idx.items():
        stats = question_stats.get(qid, {"difficulty": 0.5, "n_testcases": 5})
        week = week_lookup.get(qid, 0)
        question_static[idx, 0] = stats["difficulty"]
        question_static[idx, 1] = stats["n_testcases"] / n_testcases
        question_static[idx, 2] = week / max_week

    return question_to_idx, question_static


def extract_features(main_data, question_to_idx, config):
    """Extract per-interaction multi-modal features."""
    from Levenshtein import distance as levenshtein_distance

    n_tc = config.n_testcases
    answer_features = []
    question_idxs = []
    testcase_scores = []
    student_idxs = []

    unique_students = sorted(main_data["student_id"].unique())
    student_to_idx = {sid: idx for idx, sid in enumerate(unique_students)}
    student_state = {}

    for _, row in tqdm(
        main_data.iterrows(), total=len(main_data), desc="Extracting features"
    ):
        sid = row["student_id"]
        qid = row["question_unittest_id"]

        if qid not in question_to_idx:
            continue

        if sid not in student_state:
            student_state[sid] = {
                "first_timestamp": row["timestamp"],
                "last_timestamp": None,
                "cumulative_attempts": 0,
                "recent_scores": [],
                "cumulative_correct": 0.0,
                "cumulative_total": 0,
                "question_attempts": {},
                "unique_questions": set(),
                "last_response": {},
            }

        state = student_state[sid]

        pass_str = str(row["pass"])
        tc_vector = np.full(n_tc, -1.0, dtype=np.float32)
        for i, ch in enumerate(pass_str):
            if i >= n_tc:
                break
            tc_vector[i] = float(ch == "1")

        valid_tc = [v for v in tc_vector if v >= 0]
        pass_rate = np.mean(valid_tc) if valid_tc else 0.0
        is_perfect = 1.0 if all(v == 1.0 for v in valid_tc) else 0.0
        actual_n_tc = len(valid_tc) / n_tc

        features = []

        if config.use_performance:
            features.extend(tc_vector.tolist())
            features.append(pass_rate)
            features.append(is_perfect)
            features.append(actual_n_tc)

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
            week_num = days_since_start / 7.0

            features.append(log_time)
            features.append(math.log1p(q_attempt_num))
            features.append(math.log1p(cum_attempts))
            features.append(is_exam)
            features.append(min(week_num / 20.0, 1.0))
            features.append(math.log1p(days_since_start))

        if config.use_code_struct:
            response = str(row.get("response", ""))
            code_length = len(response)
            line_count = response.count("\n") + 1

            prev_response = state["last_response"].get(qid)
            if prev_response is not None and code_length > 0:
                edit_dist = levenshtein_distance(response[:2000], prev_response[:2000])
                code_ratio = len(response) / max(len(prev_response), 1)
            else:
                edit_dist = 0
                code_ratio = 1.0

            features.append(math.log1p(code_length))
            features.append(math.log1p(line_count))
            features.append(math.log1p(edit_dist))
            features.append(min(code_ratio, 5.0) / 5.0)

        if config.use_student_state:
            recent = state["recent_scores"][-10:]
            running_avg = np.mean(recent) if recent else 0.0
            cum_ratio = state["cumulative_correct"] / max(state["cumulative_total"], 1)

            if len(recent) >= 6:
                trend = 1.0 if np.mean(recent[-3:]) > np.mean(recent[-6:-3]) else 0.0
            else:
                trend = 0.0

            features.append(running_avg)
            features.append(cum_ratio)
            features.append(trend)
            features.append(math.log1p(len(state["unique_questions"])))

        answer_features.append(np.array(features, dtype=np.float32))
        question_idxs.append(question_to_idx[qid])
        testcase_scores.append(tc_vector)
        student_idxs.append(student_to_idx[sid])

        state["last_timestamp"] = row["timestamp"]
        state["cumulative_attempts"] += 1
        state["recent_scores"].append(pass_rate)
        state["cumulative_correct"] += pass_rate
        state["cumulative_total"] += 1
        state["question_attempts"][qid] = state["question_attempts"].get(qid, 0) + 1
        state["unique_questions"].add(qid)
        if config.use_code_struct:
            state["last_response"][qid] = str(row.get("response", ""))

    return answer_features, question_idxs, testcase_scores, student_idxs


ALL_COURSES = ["dsa_hk231", "dsa_hk221", "pf_hk232", "pf_hk222"]


def run_feature_mode(args):
    """Extract handcrafted multi-modal features from CSV."""
    if args.course == "all":
        for course in ALL_COURSES:
            args_copy = argparse.Namespace(**vars(args))
            args_copy.course = course
            args_copy.output_dir = None
            run_feature_mode(args_copy)
        return

    config = FeatureConfig()
    output_dir = args.output_dir or f"data/multimodal/{args.course}"

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

    os.makedirs(output_dir, exist_ok=True)
    for name, data in [
        ("answer_features", answer_features),
        ("question_idxs", question_idxs),
        ("question_static", question_static),
        ("testcase_scores", testcase_scores),
        ("student_idxs", student_idxs),
    ]:
        with open(f"{output_dir}/{name}.pkl", "wb") as f:
            pickle.dump(data, f)

    metadata = {
        "n_students": len(set(student_idxs)),
        "n_questions": n_questions,
        "n_interactions": len(answer_features),
        "answer_dim": config.answer_dim,
        "n_testcases": config.n_testcases,
        "question_to_idx": question_to_idx,
        "course": args.course,
    }
    with open(f"{output_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nSaved to {output_dir}/")
    print(f"  answer_features: {len(answer_features)} x {config.answer_dim}")
    print(f"  question_static: {question_static.shape}")
    print(f"  n_students: {metadata['n_students']}")


# ---------------------------------------------------------------------------
# Embedding mode: LLM text embeddings via vLLM
# ---------------------------------------------------------------------------

def parse_score_from_feedback(sample):
    return float(re.search(r"Your score:\s*([0-9]*\.?[0-9]+)\/[0-9]+", sample).group(1))


def parse_question_name(question_names):
    return_names = {}
    for qidx, q in enumerate(question_names):
        if "|" in q:
            names = q.split("|")
            for sub_idx, name in enumerate(names):
                return_names[f"{qidx+1}.{sub_idx+1}"] = name
        else:
            return_names[str(qidx + 1)] = q
    return return_names


def get_question_info_by_name(question_name, question_name2idx, question_infos):
    return question_infos[question_name2idx[question_name]]


def format_template_testcase(tokenizer, template, testcases, truncate=True):
    output = f"Template: {template}\n"
    num_processed = 0
    for tid, testcase in enumerate(testcases):
        tc_text = (
            f"Testcase {tid+1}:\n"
            f"{testcase['input']}\n"
            f"std input: {testcase['std_input']}\n"
            if testcase["std_input"] != ""
            else "" f"expected result: {testcase['output']}\n"
        )
        if truncate and len(tc_text) > 32768:
            continue
        output += tc_text
        num_processed += 1

    if len(testcases) == 0:
        output += "No testcases\n"
        return output, 0.0

    return output, num_processed / len(testcases)


def run_embedding_mode(args):
    """Generate LLM embeddings using vLLM."""
    from datasets import Dataset, load_dataset
    from embed_text_package.embed_text_v2 import Embedder
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    output_dir = args.output_dir or f"data/{args.cls}"

    ds = load_dataset(args.dataset, split=args.cls)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data_folder = snapshot_download(
        repo_id="stair-lab/dsa_hk231_wtc", repo_type="dataset"
    )
    question_infos = pickle.load(open(f"{data_folder}/unique_questions.pkl", "rb"))
    question_name2idx = pickle.load(open(f"{data_folder}/question_name2idx.pkl", "rb"))
    best_answers = pickle.load(open(f"{output_dir}/best_answers.pkl", "rb"))

    list_questions_by_week = []
    list_tcs_by_week = []
    list_best_ans_by_week = []
    list_student_attempts = []
    list_student_scores = []
    student_idxs = []
    week_idxs = []

    student_idx = -1
    is_practice = False
    total_rows = len(ds)
    for ri, row in enumerate(tqdm(ds)):
        if len(row["history"]) == 0:
            student_idx += 1

        if "Here are the exercise questions for practice." in row["instruction"]:
            ques = (
                row["instruction"]
                .replace("Here are the exercise questions for practice.", "")
                .strip()
            )
            list_questions_by_week.append(ques)
            ques_names = parse_question_name(row["question_name"])
            tcs_by_week = ""
            best_ans_by_week = ""
            for qid, q_name in ques_names.items():
                content, template, testcases = get_question_info_by_name(
                    q_name, question_name2idx, question_infos
                )
                tcs_by_week += f"Question {qid}:\n" + format_template_testcase(
                    tokenizer, template, testcases
                )
                best_ans_by_week += (
                    f"Solution for question {qid}:\n{best_answers[q_name]}\n"
                )

            list_tcs_by_week.append(tcs_by_week)
            list_best_ans_by_week.append(best_ans_by_week)
            is_practice = True

        if "Here are the exam questions." in row["instruction"]:
            is_practice = False

        if ri == total_rows - 1:
            break

        if is_practice:
            if ds[ri + 1]["instruction"].startswith("Your score"):
                week_idxs.append(len(list_questions_by_week) - 1)
                student_idxs.append(student_idx)
                list_student_attempts.append(row["output"])
                score = parse_score_from_feedback(ds[ri + 1]["instruction"])
                list_student_scores.append(score)

    # Load embedding model
    embedder = Embedder()
    embedder.load(
        args.model,
        tensor_parallel_size=args.num_gpu,
        enable_chunked_prefill=False,
        enforce_eager=True,
    )

    # Embed all text
    def embed(texts, label):
        print(f"Embedding {label}")
        ds_tmp = Dataset.from_dict({"text": texts})
        return (
            embedder.get_embeddings(
                DataLoader(ds_tmp, batch_size=args.batch_size),
                embedder.which_model, ["text"],
            ).data["text"].to_pylist()
        )

    ques_emb = embed(list_questions_by_week, "questions")
    tcs_emb = embed(list_tcs_by_week, "testcases")
    bans_emb = embed(list_best_ans_by_week, "best answers")
    answer_emb = embed(list_student_attempts, "answers")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    for name, data in [
        ("questions", ques_emb),
        ("testcases", tcs_emb),
        ("best_answer_by_week", bans_emb),
        ("answers", answer_emb),
        ("scores", list_student_scores),
        ("student_idxs", student_idxs),
        ("week_idxs", week_idxs),
    ]:
        with open(f"{output_dir}/{name}.pkl", "wb") as f:
            pickle.dump(data, f)

    print(f"\nSaved to {output_dir}/")
    print(f"  {len(ques_emb)} question embeddings")
    print(f"  {len(answer_emb)} answer embeddings")
    print(f"  {len(set(student_idxs))} students")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for RSSM")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["features", "embeddings"])
    parser.add_argument("--output_dir", type=str, default=None)

    # Feature mode args
    parser.add_argument("--course", type=str, default="all",
                        help="Course name or 'all' (features mode)")

    # Embedding mode args
    parser.add_argument("--dataset", type=str,
                        default="stair-lab/dsa_hk231_wtc_per_student_sft_lf_splited",
                        help="HuggingFace dataset (embeddings mode)")
    parser.add_argument("--cls", type=str, default="train",
                        help="Dataset split (embeddings mode)")
    parser.add_argument("--model", type=str,
                        default="/lfs/local/0/nqduc/Llama-3.1-8B-embedding",
                        help="Embedding model path (embeddings mode)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_gpu", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "features":
        run_feature_mode(args)
    else:
        run_embedding_mode(args)
