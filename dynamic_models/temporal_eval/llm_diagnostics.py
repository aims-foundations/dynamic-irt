"""LLM-only diagnostics for real vs simulated best-submit alignment.

Builds a cached joined table of overlapping real/simulated student-question
pairs, then generates:

1. An aggregate alignment plot:
   - hexbin of human_best_score vs sim_best_score
   - calibration curve: mean human_best_score by sim_best_score bin
2. A student-level plot:
   - one subplot per high-overlap student
   - human_best_score vs sim_best_score across question sequence

Usage:
    python -m dynamic_models.temporal_eval.llm_diagnostics
    python -m dynamic_models.temporal_eval.llm_diagnostics --refresh_cache
"""

from __future__ import annotations

import argparse
import json
import math
import os
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from scipy import stats

matplotlib.use("Agg")


HF_REAL_REPO = "CodeInsightTeam/code_insights_csv"
HF_SIM_REPO = "CodeInsightTeam/simulation_output"
SIM_SUBDIR = "v4_profile_mindiff"

PAIR_CSV_NAME = "llm_best_vs_real_pairs.csv"
AGGREGATE_PLOT_NAME = "llm_best_vs_real_aggregate.png"
STUDENT_PLOT_NAME = "llm_best_vs_real_students.png"

REAL_COLOR = "#4477aa"
SIM_COLOR = "#ee6677"


def _normalize_pass_string(value) -> str:
    value = str(value).strip()
    if not value or value == "nan":
        return ""
    if "." in value:
        try:
            value = str(int(float(value)))
        except ValueError:
            return ""
    return value


def _pass_fraction(value) -> float:
    value = _normalize_pass_string(value)
    if not value:
        return np.nan
    return sum(ch == "1" for ch in value) / len(value)


def _configure_style() -> None:
    try:
        from tueplots import bundles

        plt.rcParams.update(bundles.neurips2024())
    except ImportError:
        pass

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def _safe_corr(x: Iterable[float], y: Iterable[float], method: str) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)

    if len(x_arr) < 2:
        return np.nan
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return np.nan

    if method == "pearson":
        return float(np.corrcoef(x_arr, y_arr)[0, 1])
    if method == "spearman":
        return float(stats.spearmanr(x_arr, y_arr).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def _load_real_best_rows() -> pd.DataFrame:
    repo_dir = snapshot_download(
        repo_id=HF_REAL_REPO,
        repo_type="dataset",
        local_files_only=True,
    )

    main_data = pd.read_csv(
        os.path.join(repo_dir, "main_data.csv"),
        low_memory=False,
        on_bad_lines="skip",
        usecols=[
            "student_id",
            "course_id",
            "question_unittest_id",
            "response_type",
            "pass",
            "timestamp",
        ],
    )
    course_infos = pd.read_csv(os.path.join(repo_dir, "course_infos.csv"))
    question_infos = pd.read_csv(os.path.join(repo_dir, "question_infos.csv"))

    main_data = main_data[main_data["response_type"] == "Submit"].copy()
    main_data["student_id"] = main_data["student_id"].astype(str)
    main_data["course_id"] = pd.to_numeric(main_data["course_id"], errors="coerce")
    main_data["question_unittest_id"] = pd.to_numeric(
        main_data["question_unittest_id"], errors="coerce"
    )
    main_data = main_data.dropna(
        subset=["student_id", "course_id", "question_unittest_id", "pass"]
    )
    main_data["course_id"] = main_data["course_id"].astype(int)
    main_data["question_unittest_id"] = main_data["question_unittest_id"].astype(int)
    main_data["timestamp_dt"] = pd.to_datetime(
        main_data["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
    )
    main_data = main_data.dropna(subset=["timestamp_dt"])
    main_data["pass"] = main_data["pass"].apply(_normalize_pass_string)
    main_data["score"] = main_data["pass"].apply(_pass_fraction)
    main_data = main_data.dropna(subset=["score"])

    key_cols = ["course_id", "student_id", "question_unittest_id"]

    # Question sequence is based on when the student first encountered the
    # question in real data, not on score rank.
    first_question_rows = (
        main_data.sort_values(key_cols + ["timestamp_dt"])
        .groupby(key_cols, as_index=False)
        .first()[key_cols + ["timestamp_dt"]]
        .rename(columns={"timestamp_dt": "human_first_submit_ts"})
    )
    first_question_rows = first_question_rows.sort_values(
        ["course_id", "student_id", "human_first_submit_ts", "question_unittest_id"]
    )
    first_question_rows["question_sequence"] = (
        first_question_rows.groupby(["course_id", "student_id"]).cumcount() + 1
    )

    best_rows = (
        main_data.sort_values(
            key_cols + ["score", "timestamp_dt"],
            ascending=[True, True, True, False, True],
        )
        .groupby(key_cols, as_index=False)
        .first()[key_cols + ["pass", "score"]]
        .rename(columns={"pass": "human_best_pass", "score": "human_best_score"})
    )

    course_names = course_infos[["course_id", "course_name"]].drop_duplicates()
    question_meta = question_infos[
        ["question_id", "week", "topic", "question_name"]
    ].drop_duplicates()
    question_meta = question_meta.rename(columns={"question_id": "question_unittest_id"})

    real_rows = best_rows.merge(first_question_rows, on=key_cols, how="inner")
    real_rows = real_rows.merge(course_names, on="course_id", how="left")
    real_rows = real_rows.merge(question_meta, on="question_unittest_id", how="left")
    return real_rows


def _iter_simulation_files(sim_data_path: Optional[str] = None) -> List[str]:
    if sim_data_path:
        return [sim_data_path]

    repo_dir = snapshot_download(
        repo_id=HF_SIM_REPO,
        repo_type="dataset",
        local_files_only=True,
    )
    shard_pattern = os.path.join(
        repo_dir, SIM_SUBDIR, "glm_server_n10_attempts50_shard*of*.jsonl"
    )
    files = sorted(glob(shard_pattern))
    if files:
        return files

    merged_file = os.path.join(repo_dir, SIM_SUBDIR, "glm_v4_merged.jsonl")
    if os.path.exists(merged_file):
        return [merged_file]

    raise FileNotFoundError(
        f"No simulation JSONL found under {repo_dir}/{SIM_SUBDIR}"
    )


def _load_sim_best_rows(sim_data_path: Optional[str] = None) -> pd.DataFrame:
    key_cols = ["course_id", "student_id", "question_unittest_id"]
    best_rows: Dict[Tuple[int, str, int], Tuple[float, int, str]] = {}

    for fpath in _iter_simulation_files(sim_data_path):
        with open(fpath) as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("response_type") != "Submit":
                    continue

                try:
                    key = (
                        int(record["course_id"]),
                        str(record["student_id"]),
                        int(record["question_unittest_id"]),
                    )
                    attempt_id = int(record["attempt_id"])
                except (TypeError, ValueError, KeyError):
                    continue

                pass_str = _normalize_pass_string(record.get("pass"))
                score = _pass_fraction(pass_str)
                if math.isnan(score):
                    continue

                prev = best_rows.get(key)
                if prev is None or score > prev[0] or (
                    score == prev[0] and attempt_id < prev[1]
                ):
                    best_rows[key] = (score, attempt_id, pass_str)

    rows = [
        {
            "course_id": key[0],
            "student_id": key[1],
            "question_unittest_id": key[2],
            "sim_best_score": score,
            "sim_best_attempt": attempt_id,
            "sim_best_pass": pass_str,
        }
        for key, (score, attempt_id, pass_str) in best_rows.items()
    ]

    sim_rows = pd.DataFrame(rows)
    if sim_rows.empty:
        raise ValueError("No simulated Submit rows were found.")
    return sim_rows


def build_or_load_pairs(
    output_dir: str,
    sim_data_path: Optional[str] = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, PAIR_CSV_NAME)

    if os.path.exists(cache_path) and not refresh_cache:
        pairs = pd.read_csv(cache_path)
        pairs["student_id"] = pairs["student_id"].astype(str)
        return pairs

    real_rows = _load_real_best_rows()
    sim_rows = _load_sim_best_rows(sim_data_path)

    key_cols = ["course_id", "student_id", "question_unittest_id"]
    pairs = real_rows.merge(sim_rows, on=key_cols, how="inner")

    if pairs.empty:
        raise ValueError("No overlapping real/simulated student-question rows found.")

    pairs["human_best_full"] = (pairs["human_best_score"] >= 1.0).astype(int)
    pairs["sim_best_full"] = (pairs["sim_best_score"] >= 1.0).astype(int)
    pairs["exact_best_pass_match"] = (
        pairs["human_best_pass"] == pairs["sim_best_pass"]
    ).astype(int)
    pairs["full_best_pass_agreement"] = (
        pairs["human_best_full"] == pairs["sim_best_full"]
    ).astype(int)
    pairs["score_abs_diff"] = (
        pairs["human_best_score"] - pairs["sim_best_score"]
    ).abs()
    pairs = pairs.sort_values(
        ["course_id", "student_id", "question_sequence", "question_unittest_id"]
    ).reset_index(drop=True)

    pairs.to_csv(cache_path, index=False)
    return pairs


def plot_aggregate_alignment(pairs: pd.DataFrame, output_dir: str) -> str:
    _configure_style()

    pearson = _safe_corr(pairs["human_best_score"], pairs["sim_best_score"], "pearson")
    spearman = _safe_corr(
        pairs["human_best_score"], pairs["sim_best_score"], "spearman"
    )

    fig, (ax_main, ax_cal) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.6),
        gridspec_kw={"width_ratios": [1.8, 1.0]},
        constrained_layout=True,
    )

    hb = ax_main.hexbin(
        pairs["human_best_score"],
        pairs["sim_best_score"],
        gridsize=35,
        cmap="Blues",
        mincnt=1,
        linewidths=0.0,
    )
    ax_main.plot([0, 1], [0, 1], linestyle="--", color="0.35", linewidth=1.0)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("Real Best Score")
    ax_main.set_ylabel("Simulated Best Score")
    ax_main.set_title("Pairwise Score Alignment")
    fig.colorbar(hb, ax=ax_main, label="Matched rows")

    summary_text = (
        f"N = {len(pairs):,}\n"
        f"Pearson = {pearson:.3f}\n"
        f"Spearman = {spearman:.3f}\n"
        f"Exact pass match = {pairs['exact_best_pass_match'].mean():.1%}\n"
        f"Full-pass agreement = {pairs['full_best_pass_agreement'].mean():.1%}"
    )
    ax_main.text(
        0.03,
        0.97,
        summary_text,
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        fontsize=8,
    )

    # Calibration-like view: if sim score increases, does human score increase?
    bins = np.linspace(0.0, 1.0, 11)
    cal = pairs.copy()
    cal["sim_bin"] = pd.cut(
        cal["sim_best_score"],
        bins=bins,
        include_lowest=True,
        right=True,
        duplicates="drop",
    )
    cal_grouped = (
        cal.groupby("sim_bin", observed=False)
        .agg(
            sim_best_mean=("sim_best_score", "mean"),
            human_best_mean=("human_best_score", "mean"),
            count=("human_best_score", "size"),
        )
        .dropna()
        .reset_index(drop=True)
    )

    ax_cal.plot(
        cal_grouped["sim_best_mean"],
        cal_grouped["human_best_mean"],
        color=SIM_COLOR,
        marker="o",
        linewidth=1.8,
        label="Observed mean",
    )
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="0.35", linewidth=1.0)
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel("Simulated Best Score Bin Mean")
    ax_cal.set_ylabel("Mean Real Best Score")
    ax_cal.set_title("Calibration by Simulated Score")

    if not cal_grouped.empty:
        for row in cal_grouped.itertuples(index=False):
            ax_cal.annotate(
                str(int(row.count)),
                (row.sim_best_mean, row.human_best_mean),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=7,
                color="0.35",
            )

    fig.suptitle("Simulated Best Score vs Real Best Score")

    save_path = os.path.join(output_dir, AGGREGATE_PLOT_NAME)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_student_alignment(
    pairs: pd.DataFrame, output_dir: str, n_students: int = 6, min_questions: int = 8
) -> str:
    _configure_style()

    student_counts = (
        pairs.groupby(["course_id", "course_name", "student_id"], as_index=False)
        .agg(n_questions=("question_unittest_id", "size"))
        .query("n_questions >= @min_questions")
        .sort_values(["n_questions", "course_id", "student_id"], ascending=[False, True, True])
    )
    chosen = student_counts.head(n_students)

    if chosen.empty:
        raise ValueError(
            f"No students with at least {min_questions} matched questions were found."
        )

    n_panels = len(chosen)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 2.7 * n_rows),
        sharey=True,
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    for ax, row in zip(axes.flat, chosen.itertuples(index=False)):
        student_rows = pairs[
            (pairs["course_id"] == row.course_id)
            & (pairs["student_id"] == row.student_id)
        ].sort_values(["question_sequence", "question_unittest_id"])

        x_vals = np.arange(1, len(student_rows) + 1)
        human = student_rows["human_best_score"].to_numpy()
        sim = student_rows["sim_best_score"].to_numpy()

        ax.plot(
            x_vals,
            human,
            color=REAL_COLOR,
            marker="o",
            linewidth=1.6,
            markersize=3,
            label="Real best",
        )
        ax.plot(
            x_vals,
            sim,
            color=SIM_COLOR,
            marker="s",
            linewidth=1.6,
            markersize=3,
            label="Simulated best",
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Matched Question Sequence")
        ax.set_ylabel("Best Score")

        pearson = _safe_corr(human, sim, "pearson")
        title = (
            f"{row.course_name} | Student {row.student_id}\n"
            f"n={row.n_questions}, r={pearson:.2f}"
            if not math.isnan(pearson)
            else f"{row.course_name} | Student {row.student_id}\n"
            f"n={row.n_questions}, r=NA"
        )
        ax.set_title(title, fontsize=9)

    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Student-Level Real vs Simulated Best Scores")

    save_path = os.path.join(output_dir, STUDENT_PLOT_NAME)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM-only real-vs-simulated score diagnostics."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("results", "temporal_eval"),
        help="Directory for cached pairs and plots.",
    )
    parser.add_argument(
        "--sim_data_path",
        type=str,
        default=None,
        help="Optional local JSONL path overriding the default HF simulation data.",
    )
    parser.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Rebuild the cached joined table instead of reusing it.",
    )
    parser.add_argument(
        "--n_students",
        type=int,
        default=6,
        help="Number of high-overlap students to show in the student plot.",
    )
    parser.add_argument(
        "--min_questions",
        type=int,
        default=8,
        help="Minimum matched questions required for a student subplot.",
    )
    args = parser.parse_args()

    pairs = build_or_load_pairs(
        output_dir=args.output_dir,
        sim_data_path=args.sim_data_path,
        refresh_cache=args.refresh_cache,
    )

    aggregate_path = plot_aggregate_alignment(pairs, args.output_dir)
    student_path = plot_student_alignment(
        pairs,
        args.output_dir,
        n_students=args.n_students,
        min_questions=args.min_questions,
    )

    print(f"Matched rows: {len(pairs):,}")
    print(f"Cache saved to: {os.path.join(args.output_dir, PAIR_CSV_NAME)}")
    print(f"Aggregate plot saved to: {aggregate_path}")
    print(f"Student plot saved to: {student_path}")


if __name__ == "__main__":
    main()
