"""
Pace Analysis
Converted from R to Python

This script analyzes student submission pacing/timing data:
- Time deltas between submissions
- Statistical summaries of submission timing
- Aggregate visualizations of submission patterns across all questions
- Top submitters analysis and repeated ID detection

Usage:
    python pace_analysis.py --course_name dsa_hk231
    python pace_analysis.py --all_courses

Output:
    Files saved to pace_outputs/ directory:
    - log_time_delta_density_{course}.png: Distribution of pause times
    - aggregate_submission_patterns_{course}.png: Summary plots (4-panel)
    - repeated_top_submitters_{course}.csv: Academic integrity analysis
"""

import os
import argparse
import warnings
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

# Output directory for pace analysis results
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "pace_outputs")


# -----------------------------------
# Data Loading Functions
# -----------------------------------

def sanitize_latex(text):
    """Sanitize text for LaTeX rendering by escaping special characters."""
    if pd.isna(text):
        return text
    text = str(text)
    # Escape LaTeX special characters
    replacements = {
        '_': '\\_',
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def beautify_course_name(course_name):
    """Convert course codes to readable names.

    Examples:
        pf_hk232 -> Programming Fundamentals (Spring 2023)
        dsa_hk231 -> Data Structures \\& Algorithms (Fall 2023)
    """
    if pd.isna(course_name):
        return course_name

    course_name = str(course_name).lower()

    # Parse course type
    if course_name.startswith("pf"):
        course_type = "Programming Fundamentals"
    elif course_name.startswith("dsa"):
        course_type = "Data Structures \\& Algorithms"  # Escape & for LaTeX
    else:
        course_type = course_name.split("_")[0].upper()

    # Parse semester (HK format: HKXYZ where X=year, Y=semester, Z=unused)
    if "_hk" in course_name:
        semester_code = course_name.split("_hk")[1][:3]
        year = "20" + semester_code[0:2]
        semester = "Fall" if semester_code[2] == "1" else "Spring"
        return f"{course_type} ({semester} {year})"

    return course_type


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

    # Create scatter plot (without colorbar first)
    if color_var and color_var in plot_df.columns:
        scatter = ax.scatter(plot_df[x_var], plot_df[y_var], c=plot_df[color_var],
                            cmap='viridis', alpha=0.6, s=50)
    else:
        ax.scatter(plot_df[x_var], plot_df[y_var], alpha=0.6, color='steelblue', s=50)

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        plot_df[x_var], plot_df[y_var]
    )
    line_x = np.linspace(plot_df[x_var].min(), plot_df[x_var].max(), 100)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)

    ax.set_xlabel(sanitize_latex(x_var.replace('_', ' ').title()))
    ax.set_ylabel(sanitize_latex(y_var.replace('_', ' ').title()))
    ax.set_title(sanitize_latex(title))

    # Log scale for median_time_delta_seconds
    if y_var == "median_time_delta_seconds":
        ax.set_yscale('log')

    # Add colorbar AFTER setting labels and title
    if color_var and color_var in plot_df.columns:
        fig.colorbar(scatter, ax=ax, label=sanitize_latex(color_var.replace('_', ' ').title()))

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
        # Default to a specific course if neither flag is set
        course_name = "dsa_hk231"
        output_prefix = "dsa_hk231"

    # -----------------------------------
    # Data Import
    # -----------------------------------
    print("Loading data from HuggingFace...")
    main_data, question_infos, course_infos = load_data_from_huggingface(course_name)
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
        ax.set_title(f'Density Plot of log(time\\_delta) - {beautify_course_name(output_prefix)}')
        ax.legend()
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"log_time_delta_density_{output_prefix}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
    # Aggregate Analysis (All Questions Combined)
    # -----------------------------------
    print("\n--- Aggregate Submission Pattern Analysis ---")

    # Combine all filtered data into one aggregate analysis
    all_student_data = analyze_dataframe(filtered_data)

    if len(all_student_data) >= 3:
        # Create a 2x2 grid of aggregate summary plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: Submits vs Pause Duration (colored by final score)
        ax = axes[0, 0]
        scatter = ax.scatter(
            all_student_data["total_submits"],
            all_student_data["median_time_delta_seconds"],
            c=all_student_data["final_score"],
            cmap='viridis', alpha=0.5, s=30
        )
        ax.set_xlabel(sanitize_latex("Total Submissions"))
        ax.set_ylabel(sanitize_latex("Median Pause Time (seconds)"))
        ax.set_title(sanitize_latex("Submission Count vs Pause Duration"))
        ax.set_yscale('log')
        fig.colorbar(scatter, ax=ax, label=sanitize_latex("Final Score"))

        # Top-right: Submits vs Final Score
        ax = axes[0, 1]
        ax.scatter(
            all_student_data["total_submits"],
            all_student_data["final_score"],
            alpha=0.5, color='steelblue', s=30
        )
        # Add regression line
        if len(all_student_data) > 2:
            # Drop NaN values from both columns together
            regression_data = all_student_data[["total_submits", "final_score"]].dropna()
            if len(regression_data) > 2:
                slope, intercept, r_value, _, _ = stats.linregress(
                    regression_data["total_submits"],
                    regression_data["final_score"]
                )
                line_x = np.linspace(
                    regression_data["total_submits"].min(),
                    regression_data["total_submits"].max(), 100
                )
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2,
                       label=f'$R^2={r_value**2:.3f}$')
                ax.legend()
        ax.set_xlabel(sanitize_latex("Total Submissions"))
        ax.set_ylabel(sanitize_latex("Final Score"))
        ax.set_title(sanitize_latex("Submission Count vs Performance"))
        ax.set_ylim(0, 1)

        # Bottom-left: Histogram of submission counts
        ax = axes[1, 0]
        ax.hist(all_student_data["total_submits"], bins=50,
                edgecolor='black', alpha=0.7)
        ax.axvline(all_student_data["total_submits"].median(),
                  color='red', linestyle='--',
                  label=f'Median: {all_student_data["total_submits"].median():.0f}')
        ax.set_xlabel(sanitize_latex("Total Submissions"))
        ax.set_ylabel(sanitize_latex("Number of Students"))
        ax.set_title(sanitize_latex("Distribution of Submission Counts"))
        ax.legend()

        # Bottom-right: Histogram of pause times
        ax = axes[1, 1]
        pause_times = all_student_data["median_time_delta_seconds"].dropna()
        pause_times_filtered = pause_times[pause_times > 0]
        if len(pause_times_filtered) > 0:
            ax.hist(pause_times_filtered, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(pause_times_filtered.median(), color='red', linestyle='--',
                      label=f'Median: {pause_times_filtered.median():.1f}s')
            ax.set_xscale('log')
            ax.set_xlabel(sanitize_latex("Median Pause Time (seconds, log scale)"))
            ax.set_ylabel(sanitize_latex("Number of Students"))
            ax.set_title(sanitize_latex("Distribution of Pause Times"))
            ax.legend()

        plt.suptitle(f'Aggregate Submission Patterns - {beautify_course_name(output_prefix)}',
                    fontsize=14, y=0.995)

        output_path = os.path.join(OUTPUT_DIR, f"aggregate_submission_patterns_{output_prefix}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

        # Print summary statistics
        print(f"\nAggregate Statistics:")
        print(f"  Total students analyzed: {len(all_student_data):,}")
        print(f"  Median submissions per student: {all_student_data['total_submits'].median():.0f}")
        print(f"  Mean submissions per student: {all_student_data['total_submits'].mean():.1f}")
        print(f"  Median pause time: {pause_times_filtered.median():.1f} seconds")
        print(f"  Mean pause time: {pause_times_filtered.mean():.1f} seconds")

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
            output_path = os.path.join(OUTPUT_DIR, f"repeated_top_submitters_{output_prefix}.csv")
            repeated_ids.to_csv(output_path, index=False)
            print(f"\nSaved: {output_path}")
        else:
            print("No repeated IDs found among top submitters")

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
