"""
Problem-by-problem Analysis
Converted from R Markdown to Python

Author: Karen D. Wang (original R), converted to Python
Date: 2024-12-03

This script analyzes student submission data to examine:
- Marks achieved and number of steps taken by individual students
- Correlation between steps, edit distance, and marks
- Visualizations of student progress patterns

Usage:
    python problem_by_problem_analysis.py --course_name dsa_hk231
    python problem_by_problem_analysis.py --course_name dsa_hk231 --output_dir ./results
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from huggingface_hub import snapshot_download

# Set plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
sns.set_style("whitegrid")


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
    Prepare data for analysis by grouping by question and computing step numbers.

    Maps columns:
        - id: student_id
        - step: attempt number (computed)
        - marks: computed from pass column (proportion of tests passed)
        - edit_distance: edit_distance (if available)
        - action: response_type

    Returns:
        Dictionary mapping question names to DataFrames
    """
    # Compute marks from pass column
    main_data["marks"] = main_data["pass"].apply(compute_marks_from_pass)

    # Filter to submissions only
    main_data = main_data[main_data["response_type"].isin(["Submit", "Prechecked"])].copy()

    # Sort by student, question, timestamp
    main_data = main_data.sort_values(["student_id", "question_unittest_id", "timestamp"])

    # Compute step (attempt number) per student-question pair
    main_data["step"] = main_data.groupby(["student_id", "question_unittest_id"]).cumcount() + 1

    # Rename columns for compatibility with analysis functions
    main_data = main_data.rename(columns={
        "student_id": "id",
        "response_type": "action"
    })

    # Ensure edit_distance column exists (may not be in all datasets)
    if "edit_distance" not in main_data.columns:
        main_data["edit_distance"] = np.nan

    # Group by question
    data_frames = {}
    for question_id in main_data["question_unittest_id"].unique():
        question_data = main_data[main_data["question_unittest_id"] == question_id].copy()

        # Get question name from question_infos
        q_info = question_infos[question_infos["question_id"] == question_id]
        if len(q_info) > 0 and "question_name" in q_info.columns:
            q_name = q_info["question_name"].values[0]
        else:
            q_name = str(question_id)

        # Clean name for use as key
        clean_name = q_name.replace(" ", "_").replace("(", "").replace(")", "")
        data_frames[f"df_data_{clean_name}"] = question_data

    return data_frames


def process_dataframe(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """
    Process a dataframe to compute summary statistics per student.

    For each student (id), compute:
    - max_step: Maximum step number reached
    - max_marks: Maximum marks achieved

    Then aggregate across all students to get:
    - mean_step: Average of max steps
    - mean_marks: Average of max marks
    - max_marks: Maximum marks across all students
    - n: Number of students with marks > 0
    """
    # Filter out NA ids
    df_filtered = df[df['id'].notna()].copy()

    # Group by id and summarize
    summary = df_filtered.groupby('id').agg(
        max_step=('step', lambda x: x.max() if x.notna().any() else np.nan),
        max_marks=('marks', lambda x: x.max() if x.notna().any() else np.nan)
    ).reset_index()

    # Filter to students with marks > 0
    summary = summary[summary['max_marks'] > 0]

    if len(summary) == 0:
        return pd.DataFrame({
            'dataframe': [df_name],
            'mean_step': [np.nan],
            'mean_marks': [np.nan],
            'max_marks': [np.nan],
            'n': [0]
        })

    # Calculate overall statistics
    result = pd.DataFrame({
        'dataframe': [df_name],
        'mean_step': [round(summary['max_step'].mean(), 0)],
        'mean_marks': [round(summary['max_marks'].mean(), 3)],
        'max_marks': [summary['max_marks'].max()],
        'n': [len(summary)]
    })

    return result


def create_edit_distance_bar_plot(df: pd.DataFrame, output_path: str = "edit_distance_by_step.png"):
    """Create overlaid bar plot of edit distance by step for each student."""
    # Skip if no edit_distance data
    if df['edit_distance'].isna().all():
        print(f"Skipping edit distance plot - no edit_distance data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique IDs and create a color palette
    unique_ids = df['id'].dropna().unique()
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_ids), 20)))

    for idx, student_id in enumerate(unique_ids[:20]):  # Limit to 20 students for readability
        student_data = df[df['id'] == student_id]
        ax.bar(student_data['step'], student_data['edit_distance'],
               alpha=0.3, color=colors[idx % len(colors)])

    ax.set_xlabel('Step')
    ax.set_ylabel('Edit Distance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_dotplot(data: pd.Series, output_path: str, xlabel: str = "Total Number of Steps"):
    """Create a dot plot (histogram-like) showing distribution of values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use histogram with appropriate binwidth
    data_clean = data.dropna()
    if len(data_clean) == 0:
        print(f"Skipping dotplot - no data available")
        plt.close()
        return

    ax.hist(data_clean, bins=range(int(data_clean.min()), int(data_clean.max()) + 2),
            alpha=0.5, color='steelblue', edgecolor='white')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_correlation_scatterplot(x: pd.Series, y: pd.Series,
                                   xlabel: str, ylabel: str,
                                   output_path: str):
    """Create scatterplot with correlation annotation and regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        print(f"Skipping scatterplot - insufficient data points")
        plt.close()
        return

    x_clean, y_clean = x[mask], y[mask]

    # Scatter plot
    ax.scatter(x_clean, y_clean, alpha=0.6, color='steelblue', s=50)

    # Linear regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    line_x = np.linspace(x_clean.min(), x_clean.max(), 100)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r-', alpha=0.8, linewidth=2)

    # Add correlation annotation
    ax.annotate(f'r = {r_value:.2f}',
               xy=(0.95, 0.95), xycoords='axes fraction',
               ha='right', va='top', fontsize=12)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def analyze_and_plot_dataframe(df: pd.DataFrame, df_name: str, output_dir: str = ".") -> pd.DataFrame:
    """
    Analyze a dataframe and generate plots.

    Returns a DataFrame with correlations:
    - cor_step_edit: Correlation between max steps and mean edit distance
    - cor_edit_marks: Correlation between mean edit distance and marks
    - cor_step_marks: Correlation between max steps and marks
    """
    # Filter and process data
    df_filtered = df[df['id'].notna()].copy()

    # Group by student ID
    agg_dict = {
        'step': 'max',
        'marks': 'max'
    }
    if 'edit_distance' in df_filtered.columns and not df_filtered['edit_distance'].isna().all():
        agg_dict['edit_distance'] = 'mean'

    summarized = df_filtered.groupby('id').agg(
        max_step=('step', 'max'),
        mean_edit_distance=('edit_distance', 'mean') if 'edit_distance' in df_filtered.columns else ('step', lambda x: np.nan),
        marks=('marks', 'max')
    ).reset_index()

    # Filter to students with marks > 0
    summarized = summarized[summarized['marks'] > 0]

    if len(summarized) < 3:
        return pd.DataFrame({
            'dataframe': [df_name],
            'cor_step_edit': [np.nan],
            'cor_edit_marks': [np.nan],
            'cor_step_marks': [np.nan]
        })

    # Calculate correlations
    cor_step_edit = summarized['max_step'].corr(summarized['mean_edit_distance'])
    cor_edit_marks = summarized['mean_edit_distance'].corr(summarized['marks'])
    cor_step_marks = summarized['max_step'].corr(summarized['marks'])

    # Create dotplot for steps
    create_dotplot(
        summarized['max_step'],
        os.path.join(output_dir, f"total_steps_{df_name}.png"),
        "Total Number of Steps"
    )

    # Create scatterplot for steps vs edit distance (if edit_distance available)
    if not summarized['mean_edit_distance'].isna().all():
        create_correlation_scatterplot(
            summarized['max_step'],
            summarized['mean_edit_distance'],
            "Total Number of Steps",
            "Mean Edit Distance",
            os.path.join(output_dir, f"total_steps_vs_edit_distance_{df_name}.png")
        )

    return pd.DataFrame({
        'dataframe': [df_name],
        'cor_step_edit': [cor_step_edit],
        'cor_edit_marks': [cor_edit_marks],
        'cor_step_marks': [cor_step_marks]
    })


def main():
    """Main function to run the problem-by-problem analysis."""
    parser = argparse.ArgumentParser(description="Problem-by-problem analysis of student submissions")
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

    # Load data from HuggingFace
    print("Loading data from HuggingFace...")
    main_data, question_infos, course_infos = load_data_from_huggingface(args.course_name)
    print(f"Loaded {len(main_data)} submissions")

    # Prepare data for analysis
    print("\nPreparing data for analysis...")
    data_frames = prepare_data_for_analysis(main_data, question_infos)
    print(f"Created {len(data_frames)} question-specific dataframes")

    # Process each dataframe and get summary statistics
    print("\n--- Processing Summary Statistics ---")
    all_summaries = []
    for df_name, df in data_frames.items():
        summary = process_dataframe(df, df_name)
        all_summaries.append(summary)

    final_summary = pd.concat(all_summaries, ignore_index=True)
    output_path = os.path.join(args.output_dir, "individual_questions_steps_and_scores.csv")
    final_summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(final_summary.to_string())

    # Analyze and plot all dataframes
    print("\n--- Analyzing Correlations and Creating Plots ---")
    correlations = []
    for df_name, df in data_frames.items():
        # Remove 'df_data_' prefix for cleaner names
        clean_name = df_name.replace("df_data_", "")
        result = analyze_and_plot_dataframe(df, clean_name, args.output_dir)
        correlations.append(result)

    all_correlations = pd.concat(correlations, ignore_index=True)
    output_path = os.path.join(args.output_dir, "cor_step_edit.csv")
    all_correlations.to_csv(output_path, index=False)
    print("\nCorrelation Summary:")
    print(all_correlations.to_string())
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
