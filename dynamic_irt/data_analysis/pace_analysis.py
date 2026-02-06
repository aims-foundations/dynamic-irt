"""
Pace Analysis
Converted from R to Python

This script analyzes student submission pacing/timing data:
- Time deltas between submissions
- Statistical summaries of submission timing
- Scatter plots of submits vs pause duration and final scores
- Top submitters analysis and repeated ID detection

Usage:
    python pace_analysis.py --course_name dsa_hk231
    python pace_analysis.py --course_name dsa_hk231 --output_dir ./results
"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from huggingface_hub import snapshot_download

# Set plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
sns.set_style("whitegrid")


# -----------------------------------
# Data Loading Functions
# -----------------------------------

def load_data_from_huggingface(course_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from HuggingFace dataset stair-lab/code_insights_csv.

    Returns:
        main_data: Submission data filtered by course
        question_infos: Question metadata
        course_infos: Course metadata
    """
    # Download or use cached data
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--stair-lab--code_insights_csv/"
        "snapshots/99d53fe7c11f6302fb28b82fab5ebd77c00e5d12"
    )

    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        path = cache_path
    else:
        print("Downloading from HuggingFace...")
        path = snapshot_download(
            repo_id="stair-lab/code_insights_csv", repo_type="dataset"
        )

    main_data = pd.read_csv(f"{path}/main_data.csv", low_memory=False)
    question_infos = pd.read_csv(f"{path}/question_infos.csv")
    course_infos = pd.read_csv(f"{path}/course_infos.csv")

    # Get course_id for the specified course
    course_row = course_infos[course_infos["course_name"] == course_name]
    if len(course_row) == 0:
        available = course_infos['course_name'].tolist()
        raise ValueError(f"Course '{course_name}' not found. Available: {available}")

    course_id = course_row["course_id"].values[0]
    print(f"Filtering data for course: {course_name} (id={course_id})")

    # Filter to specified course
    main_data = main_data[main_data["course_id"] == course_id].copy()

    return main_data, question_infos, course_infos


def compute_marks_from_pass(pass_str) -> float:
    """
    Compute marks from pass string (e.g., "111011" -> 0.833).

    The pass column contains binary strings where each character represents
    a testcase result (1=pass, 0=fail).
    """
    if pd.isna(pass_str) or pass_str == '':
        return np.nan

    try:
        s = str(pass_str).strip()
        # Handle float strings like "111.0"
        if '.' in s:
            s = str(int(float(s)))

        if len(s) == 0:
            return np.nan

        passed = sum(1 for c in s if c == '1')
        total = len(s)
        return passed / total if total > 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def prepare_data_for_analysis(main_data: pd.DataFrame, question_infos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Prepare data for analysis by computing time deltas and grouping by question.

    Maps columns:
        - id: student_id
        - action: response_type
        - marks: computed from pass column (proportion of tests passed)
        - time_delta_seconds: computed from timestamps

    Returns:
        Dictionary with 'all_data' and 'filtered_data' (submissions only)
    """
    # Compute marks from pass column
    main_data["marks"] = main_data["pass"].apply(compute_marks_from_pass)

    # Sort by student, question, timestamp
    main_data = main_data.sort_values(["student_id", "question_unittest_id", "timestamp"])

    # Parse timestamps and compute time deltas
    main_data["timestamp_dt"] = pd.to_datetime(main_data["timestamp"], format="%d/%m/%y, %H:%M:%S", errors='coerce')

    # Compute time delta within each student-question group
    main_data["time_delta_seconds"] = main_data.groupby(
        ["student_id", "question_unittest_id"]
    )["timestamp_dt"].diff().dt.total_seconds()

    # Rename columns for compatibility
    main_data = main_data.rename(columns={
        "student_id": "id",
        "response_type": "action"
    })

    # Create filtered version (submissions only)
    filtered_data = main_data[main_data["action"].isin(["Submit", "Prechecked"])].copy()

    # Group by question for detailed analysis
    data_frames = {}
    filtered_data_frames = {}

    for question_id in main_data["question_unittest_id"].unique():
        question_data = main_data[main_data["question_unittest_id"] == question_id].copy()
        filtered_question_data = filtered_data[filtered_data["question_unittest_id"] == question_id].copy()

        # Get question name
        q_info = question_infos[question_infos["question_id"] == question_id]
        if len(q_info) > 0 and "question_name" in q_info.columns:
            q_name = q_info["question_name"].values[0]
        else:
            q_name = str(question_id)

        clean_name = q_name.replace(" ", "_").replace("(", "").replace(")", "")

        if len(question_data) > 0:
            data_frames[clean_name] = question_data
        if len(filtered_question_data) > 0:
            filtered_data_frames[clean_name] = filtered_question_data

    return {
        'all_data': main_data,
        'filtered_data': filtered_data,
        'data_frames': data_frames,
        'filtered_data_frames': filtered_data_frames
    }


# -----------------------------------
# Analysis Functions
# -----------------------------------

def calculate_percentage_over_hours(data: pd.DataFrame, threshold_hours: float) -> float:
    """Calculate percentage of time deltas over a threshold (in hours)."""
    threshold_seconds = threshold_hours * 60 * 60
    valid_data = data[data['time_delta_seconds'].notna()]

    if len(valid_data) == 0:
        return 0.0

    over_threshold = valid_data[valid_data['time_delta_seconds'] > threshold_seconds]
    percentage = 100 * len(over_threshold) / len(valid_data)
    return percentage


def create_scatterplot(df: pd.DataFrame, x_var: str, y_var: str,
                       color_var: str = None, title: str = "",
                       output_path: str = None):
    """Create scatter plots with optional color coding and regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter valid data
    plot_df = df[[x_var, y_var]].dropna()
    if color_var and color_var in df.columns:
        plot_df = df[[x_var, y_var, color_var]].dropna()

    if len(plot_df) < 3:
        print(f"Skipping scatterplot - insufficient data points")
        plt.close()
        return

    # Create scatter plot
    if color_var and color_var in plot_df.columns:
        scatter = ax.scatter(plot_df[x_var], plot_df[y_var], c=plot_df[color_var],
                            cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, ax=ax, label=color_var.replace('_', ' ').title())
    else:
        ax.scatter(plot_df[x_var], plot_df[y_var], alpha=0.6, color='steelblue', s=50)

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        plot_df[x_var], plot_df[y_var]
    )
    line_x = np.linspace(plot_df[x_var].min(), plot_df[x_var].max(), 100)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)

    ax.set_xlabel(x_var.replace('_', ' ').title())
    ax.set_ylabel(y_var.replace('_', ' ').title())
    ax.set_title(title)

    # Log scale for median_time_delta_seconds
    if y_var == "median_time_delta_seconds":
        ax.set_yscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

    plt.close()
    return fig


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze a dataframe to compute per-student statistics.

    Returns DataFrame with:
    - id: Student ID
    - median_time_delta_seconds: Median time between submissions
    - total_submits: Total number of submit actions
    - final_score: Final marks achieved
    """
    grouped = df.groupby('id').agg(
        median_time_delta_seconds=('time_delta_seconds', 'median'),
        total_submits=('action', lambda x: (x.isin(['Submit', 'Prechecked'])).sum()),
        final_score=('marks', 'max')
    ).reset_index()

    return grouped


def get_top_submitters(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Get top n submitters by total submit count."""
    return df.nlargest(n, 'total_submits')[['id', 'total_submits', 'final_score']]


def main():
    """Main function to run the pace analysis."""
    parser = argparse.ArgumentParser(description="Pace analysis of student submissions")
    parser.add_argument(
        "--course_name",
        type=str,
        default="dsa_hk231",
        help="Course name (e.g., dsa_hk231, dsa_hk221, pf_hk232)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for plots and CSV files"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------
    # Data Import
    # -----------------------------------
    print("Loading data from HuggingFace...")
    main_data, question_infos, course_infos = load_data_from_huggingface(args.course_name)
    print(f"Loaded {len(main_data)} submissions")

    # Prepare data
    print("\nPreparing data for analysis...")
    data = prepare_data_for_analysis(main_data, question_infos)
    all_data = data['all_data']
    filtered_data = data['filtered_data']
    data_frames = data['data_frames']
    filtered_data_frames = data['filtered_data_frames']

    print(f"Created {len(data_frames)} question-specific dataframes")
    print(f"Created {len(filtered_data_frames)} filtered dataframes (submissions only)")

    # -----------------------------------
    # Data Analysis
    # -----------------------------------
    print("\n--- Statistical Summary (All Data) ---")

    # Convert to minutes for reporting
    all_data['time_delta_minutes'] = all_data['time_delta_seconds'] / 60
    valid_minutes = all_data['time_delta_minutes'].dropna()

    if len(valid_minutes) > 0:
        time_delta_stats = {
            'Mean': valid_minutes.mean(),
            'Median': valid_minutes.median(),
            'Min': valid_minutes.min(),
            'Max': valid_minutes.max(),
            '95th Percentile': valid_minutes.quantile(0.95)
        }
        print(pd.DataFrame([time_delta_stats]))

        # Percentage calculations
        print(f"\nPercentage longer than 1 hour: {calculate_percentage_over_hours(all_data, 1):.2f}%")
        print(f"Percentage longer than 2 hours: {calculate_percentage_over_hours(all_data, 2):.2f}%")
        print(f"Percentage longer than 3 hours: {calculate_percentage_over_hours(all_data, 3):.2f}%")

        # Density plot of time delta
        median_minutes = valid_minutes.median()
        mean_minutes = valid_minutes.mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        positive_minutes = valid_minutes[valid_minutes > 0]

        if len(positive_minutes) > 0:
            ax.hist(positive_minutes, bins=50, density=True, alpha=0.7,
                    color='skyblue', edgecolor='darkblue')

            ax.set_xscale('log')
            ax.axvline(x=median_minutes, color='red', linestyle='--',
                       label=f'Median ({median_minutes:.1f} mins)')
            ax.axvline(x=mean_minutes, color='blue', linestyle='--',
                       label=f'Mean ({mean_minutes:.1f} mins)')

        ax.set_xlabel('Minutes (log scale)')
        ax.set_ylabel('Density')
        ax.set_title('Density Plot of log(time_delta)')
        ax.legend()
        plt.tight_layout()
        output_path = os.path.join(args.output_dir, "log_time_delta_density_plot.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\nSaved: {output_path}")

    # Action count analysis
    print("\n--- Action Counts (All Data) ---")
    if 'action' in all_data.columns:
        print(all_data['action'].value_counts())

    # -----------------------------------
    # Filtered Data Analysis
    # -----------------------------------
    print("\n--- Statistical Summary (Filtered Data - Submissions Only) ---")

    filtered_data['time_delta_minutes'] = filtered_data['time_delta_seconds'] / 60
    valid_filtered = filtered_data['time_delta_minutes'].dropna()

    if len(valid_filtered) > 0:
        filtered_stats = {
            'Mean': valid_filtered.mean(),
            'Median': valid_filtered.median(),
            'Min': valid_filtered.min(),
            'Max': valid_filtered.max(),
            '95th Percentile': valid_filtered.quantile(0.95)
        }
        print(pd.DataFrame([filtered_stats]))

        print(f"\nPercentage longer than 1 hour: {calculate_percentage_over_hours(filtered_data, 1):.2f}%")
        print(f"Percentage longer than 2 hours: {calculate_percentage_over_hours(filtered_data, 2):.2f}%")
        print(f"Percentage longer than 3 hours: {calculate_percentage_over_hours(filtered_data, 3):.2f}%")

    # -----------------------------------
    # Research Questions Analysis
    # -----------------------------------
    print("\n--- Research Questions Analysis ---")

    for name, df in filtered_data_frames.items():
        df_analysis = analyze_dataframe(df)

        if len(df_analysis) < 3:
            continue

        # Submits vs Pause Duration plot
        create_scatterplot(
            df_analysis, "total_submits", "median_time_delta_seconds", "final_score",
            f"Submits vs Pause Duration for {name}",
            os.path.join(args.output_dir, f"scatterplot_submits_pause_{name}.png")
        )

        # Submits vs Final Score plot
        create_scatterplot(
            df_analysis, "total_submits", "final_score", None,
            f"Submits vs Final Score for {name}",
            os.path.join(args.output_dir, f"scatterplot_submits_score_{name}.png")
        )

    # -----------------------------------
    # Top Submitters Analysis
    # -----------------------------------
    print("\n--- Top Submitters Analysis ---")

    all_top_submitters = []

    for name, df in filtered_data_frames.items():
        df_analysis = analyze_dataframe(df)

        if len(df_analysis) < 3:
            continue

        top_submitters = get_top_submitters(df_analysis, n=3)
        top_submitters['dataframe'] = name
        all_top_submitters.append(top_submitters)
        print(f"\n{name}:")
        print(top_submitters.to_string(index=False))

    if all_top_submitters:
        all_top_df = pd.concat(all_top_submitters, ignore_index=True)

        # Analyze repeated IDs among top submitters
        repeated_ids = all_top_df.groupby('id').agg(
            frequency=('id', 'count'),
            dataframes=('dataframe', lambda x: ', '.join(x)),
            total_submits=('total_submits', lambda x: ', '.join(map(str, x))),
            final_scores=('final_score', lambda x: ', '.join(map(str, x)))
        ).reset_index()

        repeated_ids = repeated_ids[repeated_ids['frequency'] > 1].sort_values(
            'frequency', ascending=False
        )

        print("\n--- IDs appearing in top submitters more than once ---")
        if len(repeated_ids) > 0:
            print(repeated_ids.to_string(index=False))
            output_path = os.path.join(args.output_dir, "repeated_top_submitters.csv")
            repeated_ids.to_csv(output_path, index=False)
            print(f"\nSaved: {output_path}")
        else:
            print("No repeated IDs found among top submitters")


if __name__ == "__main__":
    main()
