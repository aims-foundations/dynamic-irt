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

STAN_FILE = "dynamic_rasch.stan"
REMOTE_URL = "https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv/resolve/main/codeinsights_student_response.csv"
LOCAL_CACHE = Path("codeinsights_student_response.csv")
OUTPUT_DIR = Path("output_dynamic_rasch")

SAMPLING_CONFIG = dict(
    seed=42,
    chains=2,
    parallel_chains=2,
    iter_warmup=300,
    iter_sampling=300,
    adapt_delta=0.9,
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
        print(f"Saved to {LOCAL_CACHE}")
    else:
        print(f"Using cached file: {LOCAL_CACHE}")
    return str(LOCAL_CACHE)


def prepare_data(csv_path: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CSV, subsample, and return (stan_data, df, person_map, item_map)."""
    df = pd.read_csv(csv_path)

    # Composite item key
    df["item_key"] = (
        df["question_unittest_id"].astype(str) + "_" + df["unit_test_id"].astype(str)
    )

    # Stable 1-based indices
    student_ids = sorted(df["student_id"].unique())
    item_keys = sorted(df["item_key"].unique())
    student_to_idx = {sid: i + 1 for i, sid in enumerate(student_ids)}
    item_to_idx = {ik: i + 1 for i, ik in enumerate(item_keys)}

    df["person_idx"] = df["student_id"].map(student_to_idx)
    df["item_idx"] = df["item_key"].map(item_to_idx)

    # Ensure time_index is 1-based
    t_min = df["time_index"].min()
    if t_min == 0:
        df["time_idx"] = df["time_index"] + 1
        print("Note: time_index was 0-based; shifted to 1-based for Stan.")
    elif t_min == 1:
        df["time_idx"] = df["time_index"]
    else:
        raise ValueError(f"Unexpected minimum time_index value: {t_min}")

    N_persons = df["person_idx"].nunique()
    N_items = df["item_idx"].nunique()
    N_obs = len(df)
    T_max = int(df["time_idx"].max())

    print(f"Persons: {N_persons}")
    print(f"Items:   {N_items}")
    print(f"Obs:     {N_obs}")
    print(f"T_max:   {T_max}")

    stan_data = {
        "N": N_obs,
        "N_persons": N_persons,
        "N_items": N_items,
        "T_max": T_max,
        "person": df["person_idx"].values.astype(int).tolist(),
        "item": df["item_idx"].values.astype(int).tolist(),
        "time": df["time_idx"].values.astype(int).tolist(),
        "response": df["response"].values.astype(int).tolist(),
    }

    # Lookup tables
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

    return stan_data, df, person_map, item_map

# Fit Model

def fit_model(stan_data: dict, output_dir: Path):
    """Compile and sample, or reload from checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmdstan_output_dir = output_dir / "cmdstan_output"
    cmdstan_output_dir.mkdir(exist_ok=True)
    fit_done_marker = output_dir / "fit_complete.flag"

    if fit_done_marker.exists():
        print("\nFound completed fit. Reloading from saved CSV output...")
        from cmdstanpy import from_csv
        csv_files = sorted(str(f) for f in cmdstan_output_dir.glob("*.csv"))
        if csv_files:
            fit = from_csv(csv_files)
            print(f"  Loaded {len(csv_files)} chain file(s).")
            return fit
        else:
            print("Warning: flag exists but no CSV files found. Re-fitting...")
            fit_done_marker.unlink()

    print("\nCompiling Stan model...")
    model = CmdStanModel(stan_file=STAN_FILE)

    # Save stan_data to JSON
    data_json_path = output_dir / "stan_data.json"
    with open(data_json_path, "w") as f:
        json.dump(stan_data, f)
    print(f"  Stan data saved to {data_json_path}")

    print("\nStarting sampling (this may take a long time)...")
    fit = model.sample(
        data=stan_data,
        output_dir=str(cmdstan_output_dir),
        **SAMPLING_CONFIG,
    )

    fit_done_marker.touch()
    print(f"\nSampling complete. Chain CSVs saved in {cmdstan_output_dir}")

    return fit


# Result Analysis

def save_results(fit, stan_data: dict, person_map: pd.DataFrame,
                 item_map: pd.DataFrame, output_dir: Path) -> None:
    """Extract summaries, save CSVs and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    N_persons = stan_data["N_persons"]
    N_items = stan_data["N_items"]
    T_max = stan_data["T_max"]

    fit.runset._csv_files = [str(f) for f in fit.runset.csv_files]

    # ── Diagnostics ─────────────────────────────────────────────────────
    diag = fit.diagnose()
    print("\n── Diagnostics ──")
    print(diag)
    with open(output_dir / "diagnostics.txt", "w") as f:
        f.write(diag)

    # ── Full summary ────────────────────────────────────────────────────
    full_summary = fit.summary()
    full_summary.to_csv(output_dir / "full_parameter_summary.csv")
    print(f"Full summary shape: {full_summary.shape}")

    # ── Item parameters (beta) ──────────────────────────────────────────
    beta_rows = full_summary.loc[full_summary.index.str.startswith("beta[")].copy()
    beta_summary = beta_rows.reset_index()
    beta_summary["item_idx"] = range(1, N_items + 1)
    beta_summary = beta_summary.merge(item_map, on="item_idx")
    beta_path = output_dir / "item_parameters_dynamic.csv"
    beta_summary.to_csv(beta_path, index=False)
    print(f"  Saved {beta_path}")

    # ── Student parameters (theta_0 + growth) ───────────────────────────
    theta0_rows = full_summary.loc[full_summary.index.str.startswith("theta_0[")].reset_index()
    growth_rows = full_summary.loc[full_summary.index.str.startswith("theta_growth[")].reset_index()

    student_params = pd.DataFrame({
        "person_idx": range(1, N_persons + 1),
        "theta_0_mean": theta0_rows["Mean"].values,
        "theta_0_sd": theta0_rows["StdDev"].values,
        "theta_growth_mean": growth_rows["Mean"].values,
        "theta_growth_sd": growth_rows["StdDev"].values,
    })
    student_params = student_params.merge(person_map, on="person_idx")
    student_path = output_dir / "student_dynamic_irt.csv"
    student_params.to_csv(student_path, index=False)
    print(f"  Saved {student_path}")

    # ── Trajectory plot: top 5 students by positive growth ──────────────
    growth_for_plot = full_summary.loc[full_summary.index.str.startswith("theta_growth[")].copy()
    growth_for_plot["person_idx"] = range(1, N_persons + 1)
    positive_growth = growth_for_plot[growth_for_plot["Mean"] > 0].sort_values("Mean", ascending=False)

    print(f"\nStudents with positive growth: {len(positive_growth)} / {N_persons}")

    if positive_growth.empty:
        print("  Warning: No students with positive growth found — skipping trajectory plot.")
    else:
        colors = ["steelblue", "coral", "seagreen", "orchid", "goldenrod"]
        top5 = positive_growth["person_idx"].values[:5]

        fig, ax = plt.subplots(figsize=(10, 5))
        for person_idx, color in zip(top5, colors):
            person_vars = [f"theta[{person_idx},{t}]" for t in range(1, T_max + 1)]
            psummary = full_summary.loc[full_summary.index.isin(person_vars)].copy()
            if psummary.empty:
                continue
            psummary["time"] = range(1, len(psummary) + 1)
            student_id = person_map.loc[
                person_map["person_idx"] == person_idx, "student_id"
            ].values[0]
            g = growth_for_plot.loc[
                growth_for_plot["person_idx"] == person_idx, "Mean"
            ].values[0]
            ax.plot(
                psummary["time"], psummary["Mean"],
                color=color, label=f"Student {student_id} (g={g:.3f})",
            )
            ax.fill_between(
                psummary["time"], psummary["5%"], psummary["95%"],
                alpha=0.12, color=color,
            )

        ax.set_xlabel("Time index")
        ax.set_ylabel("θ")
        ax.set_title("Ability trajectories: Top 5 students by growth rate")
        ax.legend()
        plt.tight_layout()
        plot_path = output_dir / "positive_growth_trajectories.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {plot_path}")


# Run

def main():
    print("=" * 60)
    print("Dynamic Rasch Model Fitting")
    print("=" * 60)

    # Load data
    data_path = get_data_path()

    # Prepare data
    stan_data, df, person_map, item_map = prepare_data(data_path)

    # Save mappings
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    person_map.to_csv(OUTPUT_DIR / "person_map.csv", index=False)
    item_map.to_csv(OUTPUT_DIR / "item_map.csv", index=False)

    # Fit model (with checkpoint / reload)
    fit = fit_model(stan_data, OUTPUT_DIR)

    # Extract and save results
    save_results(fit, stan_data, person_map, item_map, OUTPUT_DIR)

    print("\nDone! All outputs in:", OUTPUT_DIR)
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
