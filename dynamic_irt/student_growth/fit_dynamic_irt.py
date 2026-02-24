import os
import json
from pathlib import Path
import urllib.request
from getpass import getpass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel


# ── Configuration ────────────────────────────────────────────────────────────

STAN_FILE_STAGE1 = "dynamic_rasch.stan"
STAN_FILE_STAGE2 = "dynamic_rasch_stage2.stan"
REMOTE_URL = "https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv/resolve/main/codeinsights_student_response.csv"
LOCAL_CACHE = Path("codeinsights_student_response.csv")
OUTPUT_DIR = Path("output_dynamic_rasch_two_stage")

STAGE1_SAMPLE_SIZE = 250
STAGE2_BATCH_SIZE = 200

STAGE1_SAMPLING = dict(
    seed=42,
    chains=2,
    parallel_chains=2,
    iter_warmup=200,
    iter_sampling=200,
    adapt_delta=0.8,
    save_warmup=False,
    show_console=True,
)

STAGE2_SAMPLING = dict(
    seed=42,
    chains=2,
    parallel_chains=2,
    iter_warmup=100,
    iter_sampling=100,
    adapt_delta=0.8,
    save_warmup=False,
    show_console=True,
)

# Data Preparation

def get_data_path() -> str:
    if not LOCAL_CACHE.exists():
        hf_token = getpass("Enter your Hugging Face token: ")
        print(f"Downloading from {REMOTE_URL}...")
        req = urllib.request.Request(REMOTE_URL)
        req.add_header("Authorization", f"Bearer {hf_token}")
        with urllib.request.urlopen(req) as response:
            LOCAL_CACHE.write_bytes(response.read())
        print(f"  Saved to {LOCAL_CACHE}")
    else:
        print(f"Using cached file: {LOCAL_CACHE}")
    return str(LOCAL_CACHE)


def load_full_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CSV, create indices over the FULL dataset. Returns (df, person_map, item_map)."""
    df = pd.read_csv(csv_path)

    df["item_key"] = (
        df["question_unittest_id"].astype(str) + "_" + df["unit_test_id"].astype(str)
    )
    student_ids = sorted(df["student_id"].unique())
    item_keys = sorted(df["item_key"].unique())
    student_to_idx = {sid: i + 1 for i, sid in enumerate(student_ids)}
    item_to_idx = {ik: i + 1 for i, ik in enumerate(item_keys)}

    df["person_idx"] = df["student_id"].map(student_to_idx)
    df["item_idx"] = df["item_key"].map(item_to_idx)
    t_min = df["time_index"].min()
    if t_min == 0:
        df["time_idx"] = df["time_index"] + 1
        print("Note: time_index was 0-based; shifted to 1-based for Stan.")
    elif t_min == 1:
        df["time_idx"] = df["time_index"]
    else:
        raise ValueError(f"Unexpected minimum time_index value: {t_min}")

    person_map = (
        df[["student_id", "person_idx"]]
        .drop_duplicates()
        .sort_values("person_idx")
        .reset_index(drop=True)
    )
    item_map = (
        df[["item_key", "item_idx"]]
        .drop_duplicates()
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    N_items = df["item_idx"].nunique()
    N_students = df["person_idx"].nunique()
    print(f"Full dataset: {N_students} students, {N_items} items, {len(df)} obs")

    return df, person_map, item_map

# Fit Model

def run_stage1(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Fit full dynamic IRT on a subsample to estimate item parameters.
    Returns dict with item parameters beta and sigma
    """
    
    stage1_dir = output_dir / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    result_path = stage1_dir / "item_calibration.json"

    # Check for cached Stage 1 results
    if result_path.exists():
        print("Stage 1: Loading cached item calibration")
        with open(result_path) as f:
            calibration = json.load(f)
        print(f"  Loaded {len(calibration['beta'])} item parameters.")
        return calibration

    print("Stage 1: Item calibration on subsample")

    # Subsample students
    all_students = sorted(df["person_idx"].unique())
    rng = np.random.default_rng(seed=42)
    if len(all_students) <= STAGE1_SAMPLE_SIZE:
        sample_students = all_students
    else:
        sample_students = rng.choice(all_students, size=STAGE1_SAMPLE_SIZE, replace=False).tolist()
    sub_df = df[df["person_idx"].isin(sample_students)].copy()
    local_students = sorted(sub_df["person_idx"].unique())
    global_to_local = {g: i + 1 for i, g in enumerate(local_students)}
    sub_df["local_person_idx"] = sub_df["person_idx"].map(global_to_local)

    N_persons_local = len(local_students)
    N_items = df["item_idx"].max()
    T_max = int(sub_df["time_idx"].max())

    print(f"Subsample: {N_persons_local} students, {N_items} items, {len(sub_df)} obs, T_max={T_max}")

    stan_data = {
        "N": len(sub_df),
        "N_persons": N_persons_local,
        "N_items": N_items,
        "T_max": T_max,
        "person": sub_df["local_person_idx"].values.astype(int).tolist(),
        "item": sub_df["item_idx"].values.astype(int).tolist(),
        "time": sub_df["time_idx"].values.astype(int).tolist(),
        "response": sub_df["response"].values.astype(int).tolist(),
    }

    # Fit
    cmdstan_dir = str(stage1_dir / "cmdstan_output")
    os.makedirs(cmdstan_dir, exist_ok=True)

    print("  Compiling Stage 1 model...")
    model = CmdStanModel(stan_file=STAN_FILE_STAGE1)

    print("  Sampling Stage 1...")
    fit = model.sample(
        data=stan_data,
        output_dir=cmdstan_dir,
        **STAGE1_SAMPLING,
    )

    fit.runset._csv_files = [str(f) for f in fit.runset.csv_files]
    print(fit.diagnose())

    # Extract item parameters
    full_summary = fit.summary()

    beta_rows = full_summary.loc[full_summary.index.str.startswith("beta[")]
    beta_values = beta_rows["Mean"].values.tolist()

    sigma_row = full_summary.loc["sigma"]
    sigma_value = float(sigma_row["Mean"])

    calibration = {
        "beta": beta_values,
        "sigma": sigma_value,
    }

    # Save
    with open(result_path, "w") as f:
        json.dump(calibration, f, indent=2)
    beta_summary = beta_rows.reset_index()
    beta_summary["item_idx"] = range(1, len(beta_values) + 1)
    beta_summary.to_csv(stage1_dir / "item_parameters.csv", index=False)
    print(f"Stage 1 done: {len(beta_values)} betas, sigma={sigma_value:.4f}")
    print(f"Saved to {result_path}")

    return calibration

def run_stage2(df: pd.DataFrame, calibration: dict, person_map: pd.DataFrame,
               output_dir: Path) -> pd.DataFrame:
    """
    Fix item params, estimate person params (theta_0, theta_growth) in batches.
    Returns combined student_params DataFrame.
    """
    stage2_dir = output_dir / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    combined_path = stage2_dir / "all_student_params.csv"

    # Check for cached final result
    if combined_path.exists():
        print("Stage 2: Loading cached person estimates")
        return pd.read_csv(combined_path)

    print("Stage 2: Person estimation in batches")

    beta_fixed = calibration["beta"]
    sigma_fixed = calibration["sigma"]
    N_items = len(beta_fixed)

    # Compile Stage 2 model once
    model = CmdStanModel(stan_file=STAN_FILE_STAGE2)
    # Split students into batches
    all_students = sorted(df["person_idx"].unique())
    batches = [
        all_students[i:i + STAGE2_BATCH_SIZE]
        for i in range(0, len(all_students), STAGE2_BATCH_SIZE)
    ]
    print(f"{len(all_students)} students → {len(batches)} batches of ≤{STAGE2_BATCH_SIZE}")

    all_results = []

    for batch_idx, batch_students in enumerate(batches):
        batch_dir = stage2_dir / f"batch_{batch_idx:03d}"
        batch_result_path = batch_dir / "batch_params.csv"

        # Skip if this batch is already done
        if batch_result_path.exists():
            print(f"  Batch {batch_idx + 1}/{len(batches)}: cached ✓")
            all_results.append(pd.read_csv(batch_result_path))
            continue

        batch_dir.mkdir(parents=True, exist_ok=True)
        print(f"Batch {batch_idx + 1}/{len(batches)}: {len(batch_students)} students")

        batch_df = df[df["person_idx"].isin(batch_students)].copy()
        local_students = sorted(batch_df["person_idx"].unique())
        global_to_local = {g: i + 1 for i, g in enumerate(local_students)}
        batch_df["local_person_idx"] = batch_df["person_idx"].map(global_to_local)

        N_persons_local = len(local_students)
        T_max_local = int(batch_df["time_idx"].max())

        stan_data = {
            "N": len(batch_df),
            "N_persons": N_persons_local,
            "N_items": N_items,
            "T_max": T_max_local,
            "person": batch_df["local_person_idx"].values.astype(int).tolist(),
            "item": batch_df["item_idx"].values.astype(int).tolist(),
            "time": batch_df["time_idx"].values.astype(int).tolist(),
            "response": batch_df["response"].values.astype(int).tolist(),
            "beta_fixed": beta_fixed,
            "sigma_fixed": sigma_fixed,
        }

        cmdstan_dir = str(batch_dir / "cmdstan_output")
        os.makedirs(cmdstan_dir, exist_ok=True)

        fit = model.sample(
            data=stan_data,
            output_dir=cmdstan_dir,
            **STAGE2_SAMPLING,
        )

        fit.runset._csv_files = [str(f) for f in fit.runset.csv_files]
        full_summary = fit.summary()

        # Extract person parameters
        theta0_rows = full_summary.loc[full_summary.index.str.startswith("theta_0[")].reset_index()
        growth_rows = full_summary.loc[full_summary.index.str.startswith("theta_growth[")].reset_index()

        batch_params = pd.DataFrame({
            "local_person_idx": range(1, N_persons_local + 1),
            "person_idx": local_students,
            "theta_0_mean": theta0_rows["Mean"].values,
            "theta_0_sd": theta0_rows["StdDev"].values,
            "theta_growth_mean": growth_rows["Mean"].values,
            "theta_growth_sd": growth_rows["StdDev"].values,
        })

        batch_params.to_csv(batch_result_path, index=False)
        all_results.append(batch_params)
        print(f"    → saved {batch_result_path}")

    # Combine all batches
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.drop(columns=["local_person_idx"])
    combined = combined.merge(person_map, on="person_idx")
    combined = combined.sort_values("person_idx").reset_index(drop=True)
    combined.to_csv(combined_path, index=False)
    print(f"\n  Combined results: {len(combined)} students → {combined_path}")

    return combined

# Result Analysis

def plot_top_growth(student_params: pd.DataFrame, output_dir: Path) -> None:
    """Plot top 5 students by positive growth rate."""
    positive = student_params[student_params["theta_growth_mean"] > 0].sort_values(
        "theta_growth_mean", ascending=False
    )
    print(f"\nStudents with positive growth: {len(positive)} / {len(student_params)}")

    if positive.empty:
        print("  No students with positive growth.")
        return

    top5 = positive.head(5)
    colors = ["steelblue", "coral", "seagreen", "orchid", "goldenrod"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for (_, row), color in zip(top5.iterrows(), colors):
        # Reconstruct trajectory from theta_0 + growth * t
        t_vals = np.arange(1, 101)
        theta_vals = row["theta_0_mean"] + row["theta_growth_mean"] * t_vals
        ax.plot(
            t_vals, theta_vals,
            color=color,
            label=f"Student {row['student_id']} (g={row['theta_growth_mean']:.3f})",
        )

    ax.set_xlabel("Time index")
    ax.set_ylabel("θ (deterministic trajectory)")
    ax.set_title("Top 5 students by growth rate (two-stage estimates)")
    ax.legend()
    plt.tight_layout()
    plot_path = output_dir / "top_growth_trajectories.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {plot_path}")
    
# Run

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Load data
    data_path = get_data_path()
    df, person_map, item_map = load_full_data(data_path)
    person_map.to_csv(OUTPUT_DIR / "person_map.csv", index=False)
    item_map.to_csv(OUTPUT_DIR / "item_map.csv", index=False)

    # Stage 1
    calibration = run_stage1(df, OUTPUT_DIR)
    # Stage 2
    student_params = run_stage2(df, calibration, person_map, OUTPUT_DIR)

    # Plot
    plot_top_growth(student_params, OUTPUT_DIR)
    print("All outputs in:", OUTPUT_DIR)
    for f in sorted(OUTPUT_DIR.rglob("*.csv")):
        print(f"{f.relative_to(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
