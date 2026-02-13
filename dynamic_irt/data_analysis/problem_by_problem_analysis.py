"""
Problem-by-problem Analysis
Converted from R Markdown to Python

This script analyzes student submission data to examine:
- Marks achieved and number of steps taken by individual students
- Correlation between steps, edit distance, and marks
- Aggregate visualizations of patterns across all problems

Usage:
    python problem_by_problem_analysis.py --course_name dsa_hk231
    python problem_by_problem_analysis.py --all_courses

Output:
    Files saved to problem_outputs/ directory:
    - individual_questions_steps_and_scores.csv: Per-problem statistics
    - cor_step_edit.csv: Correlation analysis
    - aggregate_problem_patterns.png: Summary visualizations (300 DPI)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from huggingface_hub import snapshot_download, login
from tueplots import bundles

# Set style for all plots with LaTeX fonts and tueplots
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# Output directory for problem analysis results
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "problem_outputs")


def load_data_from_huggingface(course_name: str = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from HuggingFace dataset stair-lab/code_insights_csv.

    Args:
        course_name: Course name to filter (e.g., dsa_hk231). If None, loads all courses.

    Returns:
        main_data: Submission data (filtered by course if specified)
        question_infos: Question metadata
        course_infos: Course metadata
    """
    # Try local cache first
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--stair-lab--code_insights_csv/"
        "snapshots/99d53fe7c11f6302fb28b82fab5ebd77c00e5d12"
    )

    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        path = cache_path
    else:
        print("Downloading from HuggingFace...")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        path = snapshot_download(
            repo_id="stair-lab/code_insights_csv", repo_type="dataset"
        )

    main_data = pd.read_csv(f"{path}/main_data.csv", low_memory=False)
    question_infos = pd.read_csv(f"{path}/question_infos.csv")
    course_infos = pd.read_csv(f"{path}/course_infos.csv")

    # Merge course names into main data
    main_data = main_data.merge(course_infos, on="course_id", how="left")

    # Filter by course if specified
    if course_name:
        if course_name not in course_infos["course_name"].values:
            available = course_infos['course_name'].tolist()
            raise ValueError(f"Course '{course_name}' not found. Available: {available}")

        main_data = main_data[main_data["course_name"] == course_name].copy()
        print(f"Filtered to course: {course_name}")
        print(f"Total submissions in course: {len(main_data):,}")
    else:
        print(f"Analyzing all courses")
        print(f"Total submissions: {len(main_data):,}")

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
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
        default=None,
        help="Course name (e.g., dsa_hk231, dsa_hk221, pf_hk232). If not specified, analyzes all courses."
    )
    parser.add_argument(
        "--all_courses",
        action="store_true",
        help="Analyze all courses (same as not specifying --course_name)"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine course to analyze
    if args.all_courses:
        course_name = None
        output_prefix = "all_courses"
    elif args.course_name:
        course_name = args.course_name
        output_prefix = args.course_name
    else:
        # Default to all courses
        course_name = None
        output_prefix = "all_courses"

    # Load data from HuggingFace
    print("Loading data from HuggingFace...")
    main_data, question_infos, course_infos = load_data_from_huggingface(course_name)
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
    output_path = os.path.join(OUTPUT_DIR, f"individual_questions_steps_and_scores_{output_prefix}.csv")
    final_summary.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Print aggregate statistics
    print("\n--- Aggregate Statistics Across All Problems ---")
    print(f"Number of problems analyzed: {len(final_summary)}")
    print(f"Average steps per problem: {final_summary['mean_step'].mean():.1f}")
    print(f"Average marks per problem: {final_summary['mean_marks'].mean():.3f}")
    print(f"Average students per problem: {final_summary['n'].mean():.1f}")

    # Analyze correlations (skip individual plots)
    print("\n--- Analyzing Correlations ---")
    correlations = []
    for df_name, df in data_frames.items():
        # Remove 'df_data_' prefix for cleaner names
        clean_name = df_name.replace("df_data_", "")

        # Compute correlations without creating individual plots
        df_filtered = df[df['id'].notna()].copy()
        summarized = df_filtered.groupby('id').agg(
            max_step=('step', 'max'),
            mean_edit_distance=('edit_distance', 'mean') if 'edit_distance' in df_filtered.columns else ('step', lambda x: np.nan),
            marks=('marks', 'max')
        ).reset_index()
        summarized = summarized[summarized['marks'] > 0]

        if len(summarized) >= 3:
            cor_step_edit = summarized['max_step'].corr(summarized['mean_edit_distance'])
            cor_edit_marks = summarized['mean_edit_distance'].corr(summarized['marks'])
            cor_step_marks = summarized['max_step'].corr(summarized['marks'])
        else:
            cor_step_edit = cor_edit_marks = cor_step_marks = np.nan

        correlations.append(pd.DataFrame({
            'dataframe': [clean_name],
            'cor_step_edit': [cor_step_edit],
            'cor_edit_marks': [cor_edit_marks],
            'cor_step_marks': [cor_step_marks]
        }))

    all_correlations = pd.concat(correlations, ignore_index=True)
    output_path = os.path.join(OUTPUT_DIR, f"cor_step_edit_{output_prefix}.csv")
    all_correlations.to_csv(output_path, index=False)

    # Print correlation summary
    print(f"\nCorrelation Summary (averaged across {len(all_correlations)} problems):")
    print(f"  Steps ↔ Edit Distance: r = {all_correlations['cor_step_edit'].mean():.3f}")
    print(f"  Edit Distance ↔ Marks: r = {all_correlations['cor_edit_marks'].mean():.3f}")
    print(f"  Steps ↔ Marks: r = {all_correlations['cor_step_marks'].mean():.3f}")
    print(f"\nSaved: {output_path}")

    # Create aggregate visualization
    print("\n--- Creating Aggregate Visualization ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Distribution of mean steps across problems
    axes[0, 0].hist(final_summary['mean_step'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(final_summary['mean_step'].mean(), color='red', linestyle='--',
                       label=f"Mean: {final_summary['mean_step'].mean():.1f}")
    axes[0, 0].set_xlabel('Average Steps per Problem')
    axes[0, 0].set_ylabel('Number of Problems')
    axes[0, 0].set_title('Distribution of Problem Difficulty (Steps)')
    axes[0, 0].legend()

    # Top-right: Distribution of mean marks across problems
    axes[0, 1].hist(final_summary['mean_marks'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(final_summary['mean_marks'].mean(), color='red', linestyle='--',
                       label=f"Mean: {final_summary['mean_marks'].mean():.3f}")
    axes[0, 1].set_xlabel('Average Score per Problem')
    axes[0, 1].set_ylabel('Number of Problems')
    axes[0, 1].set_title('Distribution of Problem Success Rates')
    axes[0, 1].legend()

    # Bottom-left: Steps vs Marks relationship
    axes[1, 0].scatter(final_summary['mean_step'], final_summary['mean_marks'], alpha=0.6, s=50)
    if len(final_summary.dropna(subset=['mean_step', 'mean_marks'])) > 2:
        valid_data = final_summary.dropna(subset=['mean_step', 'mean_marks'])
        slope, intercept, r_value, _, _ = stats.linregress(valid_data['mean_step'], valid_data['mean_marks'])
        line_x = np.linspace(valid_data['mean_step'].min(), valid_data['mean_step'].max(), 100)
        line_y = slope * line_x + intercept
        axes[1, 0].plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2,
                       label=f'$r={r_value:.3f}$')
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Average Steps')
    axes[1, 0].set_ylabel('Average Marks')
    axes[1, 0].set_title('Problem Difficulty vs Success Rate')
    axes[1, 0].set_ylim(0, 1)

    # Bottom-right: Correlation heatmap
    cor_data = all_correlations[['cor_step_edit', 'cor_edit_marks', 'cor_step_marks']].mean()
    cor_matrix = np.array([[1.0, cor_data['cor_step_edit'], cor_data['cor_step_marks']],
                           [cor_data['cor_step_edit'], 1.0, cor_data['cor_edit_marks']],
                           [cor_data['cor_step_marks'], cor_data['cor_edit_marks'], 1.0]])

    im = axes[1, 1].imshow(cor_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1, 1].set_xticks([0, 1, 2])
    axes[1, 1].set_yticks([0, 1, 2])
    axes[1, 1].set_xticklabels(['Steps', 'Edit Dist', 'Marks'], fontsize=8)
    axes[1, 1].set_yticklabels(['Steps', 'Edit Dist', 'Marks'], fontsize=8)
    axes[1, 1].set_title('Average Correlations')

    # Add correlation values to heatmap
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{cor_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)

    plt.suptitle(f'Problem-by-Problem Analysis Summary - {output_prefix.upper()}',
                fontsize=14, y=0.995)

    # Add colorbar AFTER suptitle
    fig.colorbar(im, ax=axes[1, 1], label='Correlation')

    # Don't use tight_layout with colorbar - use bbox_inches='tight' instead
    output_path = os.path.join(OUTPUT_DIR, f"aggregate_problem_patterns_{output_prefix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
