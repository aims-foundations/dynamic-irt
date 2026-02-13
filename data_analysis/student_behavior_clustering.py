"""Student Behavior Clustering Analysis for CodeInsight Dataset.

This module performs K-Means clustering on student coding behavior patterns
to identify different types of problem-solving strategies. Students are grouped
based on their submission patterns for specific coding questions.

The analysis uses three behavioral features:
1. Average time between consecutive submissions (seconds)
2. Average edit distance between consecutive code submissions (Levenshtein)
3. Total number of submissions made

Typical clusters might include:
- Quick Iterators: Many submissions with small edits and short intervals
- Careful Planners: Few submissions with large edits and long intervals
- Struggling Students: Many submissions with inconsistent patterns
- Potential Anomalies: Unusual patterns that may indicate cheating

Usage:
    # Analyze all courses
    python student_behavior_clustering.py

    # Analyze specific course
    python student_behavior_clustering.py --course_name dsa_hk231

    # Analyze specific question (optional)
    python student_behavior_clustering.py --question_pattern "52CD+"

Output:
    Files saved to clustering_outputs/ directory:
    - student_metrics_{course}.csv: Student-level metrics with cluster assignments
    - centroids_{course}.csv: Normalized cluster centers
    - student_behavior_clusters_{course}.png: Multi-panel 2D visualization of clusters

Dependencies:
    - python-Levenshtein: For computing edit distances between code submissions
    - scikit-learn: For K-Means clustering
    - tueplots: For publication-quality plots with LaTeX fonts
"""

import os
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import login, snapshot_download
from Levenshtein import distance as levenshtein_distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
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

# Output directory for clustering results
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "clustering_outputs")


def load_codeinsight_data():
    """Load CodeInsight data from HuggingFace or local cache.

    Uses the same centralized CSV dataset as the EDA script.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - main_data: Student submissions with course info
            - questions: Question metadata
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
        # Fall back to HuggingFace download
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        path = snapshot_download(
            repo_id="stair-lab/code_insights_csv", repo_type="dataset"
        )

    main_data = pd.read_csv(f"{path}/main_data.csv", low_memory=False)
    questions = pd.read_csv(f"{path}/question_infos.csv")
    courses = pd.read_csv(f"{path}/course_infos.csv")

    # Merge course names into main data
    main_data = main_data.merge(courses, on="course_id", how="left")

    return main_data, questions


def convert_to_seconds(timestamp_str):
    """Convert timestamp string to Unix timestamp in seconds.

    Args:
        timestamp_str (str): Timestamp in format "DD/MM/YY, HH:MM:SS".

    Returns:
        float: Unix timestamp in seconds since epoch.
    """
    timestamp_format = "%d/%m/%y, %H:%M:%S"
    datetime_obj = datetime.strptime(timestamp_str, timestamp_format)
    return datetime_obj.timestamp()


def extract_student_metrics(df, question_pattern=None, min_submissions=2):
    """Extract behavioral metrics for students from CSV data.

    Analyzes student coding behavior by computing three key metrics
    for each student based on their submission history.

    Args:
        df (pd.DataFrame): Main data with student submissions.
        question_pattern (str, optional): Filter questions containing this text.
            If None, aggregates across all questions per student.
        min_submissions (int): Minimum submissions required to include student.

    Returns:
        list[dict]: List of dictionaries containing student metrics:
            - student_id: Unique student identifier
            - average_time_diff: Mean seconds between submissions
            - total_submissions: Count of valid submissions
            - average_edit_distance: Mean Levenshtein distance between submissions
    """
    # Filter to actual submissions (not just saves)
    submit_df = df[df["response_type"].isin(["Submit", "Prechecked"])].copy()
    submit_df = submit_df.dropna(subset=["response"])

    # Filter by question pattern if specified
    if question_pattern:
        questions_with_pattern = df[
            df["question_unittest_id"].astype(str).str.contains(question_pattern, na=False)
        ]["question_unittest_id"].unique()
        submit_df = submit_df[submit_df["question_unittest_id"].isin(questions_with_pattern)]
        print(f"Found {len(questions_with_pattern)} questions matching '{question_pattern}'")

    # Parse timestamps
    submit_df["timestamp_dt"] = pd.to_datetime(
        submit_df["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
    )
    submit_df = submit_df.dropna(subset=["timestamp_dt"])

    # Sort by student and timestamp
    submit_df = submit_df.sort_values(["student_id", "timestamp_dt"])

    student_metrics = []

    # Group by student and calculate metrics
    for student_id, student_data in submit_df.groupby("student_id"):
        if len(student_data) < min_submissions:
            continue

        responses = student_data["response"].tolist()
        timestamps = student_data["timestamp_dt"].tolist()

        total_submissions = len(responses)

        # Calculate average time difference between consecutive submissions
        if len(timestamps) > 1:
            time_diffs = [
                (timestamps[i] - timestamps[i - 1]).total_seconds()
                for i in range(1, len(timestamps))
            ]
            average_time_diff = np.mean(time_diffs)
        else:
            average_time_diff = 0

        # Calculate average edit distance between consecutive submissions
        if len(responses) > 1:
            edit_distances = [
                levenshtein_distance(str(responses[i]), str(responses[i - 1]))
                for i in range(1, len(responses))
            ]
            average_edit_distance = np.mean(edit_distances)
        else:
            average_edit_distance = 0

        student_metrics.append({
            "student_id": student_id,
            "average_time_diff": average_time_diff,
            "total_submissions": total_submissions,
            "average_edit_distance": average_edit_distance,
        })

    print(f"Analyzed {len(student_metrics)} students with >= {min_submissions} submissions")
    return student_metrics


def cluster_students(student_metrics, output_prefix="all_courses"):
    """Perform K-Means clustering on student behavioral metrics.

    Groups students into 4 distinct clusters based on their coding behavior
    patterns. Outputs CSV files with metrics and cluster assignments, plus
    a 3D visualization of the clusters.

    Args:
        student_metrics (list[dict]): Student behavioral metrics.
        output_prefix (str): Prefix for output filenames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - df: DataFrame with student metrics and cluster assignments
            - centroids: DataFrame with cluster centers for each feature

    Outputs:
        Creates three files in clustering_outputs/ directory:
        - student_metrics_{prefix}.csv: Student data with cluster labels
        - centroids_{prefix}.csv: Normalized cluster centers
        - student_behavior_clusters_{prefix}.png: Multi-panel 2D visualization (300 DPI)

    Clustering Method:
        - Algorithm: K-Means with k=4 clusters
        - Features: [average_time_diff, average_edit_distance, total_submissions]
        - Random seed: 42 (for reproducibility)
    """
    df = pd.DataFrame(student_metrics)

    if len(df) == 0:
        print("No data to cluster!")
        return None, None

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Perform K-Means clustering
    features = df[["average_time_diff", "average_edit_distance", "total_submissions"]]
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(features)

    # Extract and normalize cluster centroids
    centroids = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=[
            "Average time difference",
            "Average edit distance",
            "Total submissions",
        ],
    )
    scaler = MinMaxScaler()
    centroids_normalized = scaler.fit_transform(centroids)
    normalized_centroids_df = pd.DataFrame(
        centroids_normalized, columns=centroids.columns
    )
    centroids_file = os.path.join(OUTPUT_DIR, f"centroids_{output_prefix}.csv")
    normalized_centroids_df.to_csv(centroids_file, index=False)

    # Save student metrics with cluster assignments
    metrics_file = os.path.join(OUTPUT_DIR, f"student_metrics_{output_prefix}.csv")
    df.to_csv(metrics_file, index=False)
    print(f"Data saved to {metrics_file}")

    # Create multi-panel 2D visualization (much better than 3D!)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]  # Professional color palette
    cluster_labels = [f"Cluster {i+1}" for i in range(4)]

    # Top-left: Time vs Edit Distance
    for i in range(4):
        cluster_data = df[df["cluster"] == i]
        axes[0, 0].scatter(
            cluster_data["average_time_diff"],
            cluster_data["average_edit_distance"],
            color=colors[i],
            label=cluster_labels[i],
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5
        )
    axes[0, 0].set_xlabel("Average Time Between Submissions (s)")
    axes[0, 0].set_ylabel("Average Edit Distance")
    axes[0, 0].set_title("Time vs Edit Distance")
    axes[0, 0].legend(loc="best")
    axes[0, 0].set_xscale("log")

    # Top-middle: Time vs Total Submissions
    for i in range(4):
        cluster_data = df[df["cluster"] == i]
        axes[0, 1].scatter(
            cluster_data["average_time_diff"],
            cluster_data["total_submissions"],
            color=colors[i],
            label=cluster_labels[i],
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5
        )
    axes[0, 1].set_xlabel("Average Time Between Submissions (s)")
    axes[0, 1].set_ylabel("Total Submissions")
    axes[0, 1].set_title("Time vs Total Submissions")
    axes[0, 1].set_xscale("log")

    # Top-right: Edit Distance vs Total Submissions
    for i in range(4):
        cluster_data = df[df["cluster"] == i]
        axes[0, 2].scatter(
            cluster_data["average_edit_distance"],
            cluster_data["total_submissions"],
            color=colors[i],
            label=cluster_labels[i],
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5
        )
    axes[0, 2].set_xlabel("Average Edit Distance")
    axes[0, 2].set_ylabel("Total Submissions")
    axes[0, 2].set_title("Edit Distance vs Total Submissions")

    # Bottom-left: Box plots for Time
    time_data = [df[df["cluster"] == i]["average_time_diff"] for i in range(4)]
    bp1 = axes[1, 0].boxplot(time_data, tick_labels=cluster_labels, patch_artist=True)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel("Average Time Between Submissions (s)")
    axes[1, 0].set_title("Time Distribution per Cluster")
    axes[1, 0].set_yscale("log")
    axes[1, 0].tick_params(axis="x", rotation=15)

    # Bottom-middle: Box plots for Edit Distance
    edit_data = [df[df["cluster"] == i]["average_edit_distance"] for i in range(4)]
    bp2 = axes[1, 1].boxplot(edit_data, tick_labels=cluster_labels, patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].set_ylabel("Average Edit Distance")
    axes[1, 1].set_title("Edit Distance Distribution per Cluster")
    axes[1, 1].tick_params(axis="x", rotation=15)

    # Bottom-right: Cluster profile heatmap
    cluster_summary = df.groupby("cluster")[
        ["average_time_diff", "average_edit_distance", "total_submissions"]
    ].mean()

    # Normalize for heatmap
    cluster_summary_norm = (cluster_summary - cluster_summary.min()) / (cluster_summary.max() - cluster_summary.min())

    im = axes[1, 2].imshow(cluster_summary_norm.T, cmap="YlOrRd", aspect="auto")
    axes[1, 2].set_xticks(range(4))
    axes[1, 2].set_xticklabels(cluster_labels)
    axes[1, 2].set_yticks(range(3))
    axes[1, 2].set_yticklabels(["Avg Time", "Avg Edit Dist", "Total Subs"], fontsize=8)
    axes[1, 2].set_title("Normalized Cluster Profiles")

    # Add text annotations to heatmap
    for i in range(4):
        for j in range(3):
            text = axes[1, 2].text(i, j, f"{cluster_summary_norm.iloc[i, j]:.2f}",
                                   ha="center", va="center", color="black", fontsize=8)

    plt.suptitle(f"Student Behavior Clusters - {output_prefix.upper()}", fontsize=14, y=0.995)

    # Add colorbar AFTER setting the title (before tight_layout would cause conflict)
    fig.colorbar(im, ax=axes[1, 2], label="Normalized Value")

    # Save the figure with high resolution
    output_file = os.path.join(OUTPUT_DIR, f"student_behavior_clusters_{output_prefix}.png")
    plt.savefig(output_file, format="png", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_file}")

    # Close to prevent display in headless environments
    plt.close()

    return df, centroids


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Cluster students by coding behavior patterns."
    )
    parser.add_argument(
        "--course_name",
        help="Course name to filter (e.g., dsa_hk231, pf_hk222). If not specified, analyzes all courses.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--question_pattern",
        help="Filter questions containing this text (e.g., '52CD+'). If not specified, analyzes all questions.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--min_submissions",
        help="Minimum number of submissions required per student (default: 2)",
        type=int,
        default=2
    )
    args = parser.parse_args()

    # Load centralized CSV data
    print("Loading CodeInsight data...")
    main_data, questions = load_codeinsight_data()

    # Filter by course if specified
    if args.course_name:
        main_data = main_data[main_data["course_name"] == args.course_name]
        output_prefix = args.course_name
        print(f"Filtered to course: {args.course_name}")
        print(f"Total submissions in course: {len(main_data):,}")
    else:
        output_prefix = "all_courses"
        print(f"Analyzing all courses")
        print(f"Total submissions: {len(main_data):,}")

    # Extract student metrics
    print("\nExtracting behavioral metrics...")
    student_metrics = extract_student_metrics(
        main_data,
        question_pattern=args.question_pattern,
        min_submissions=args.min_submissions
    )

    if len(student_metrics) == 0:
        print("No students found matching criteria!")
        exit(1)

    # Perform clustering
    print("\nPerforming K-Means clustering...")
    df, centroids = cluster_students(student_metrics, output_prefix)

    if df is not None:
        # Print cluster distribution
        cluster_counts = df["cluster"].value_counts().sort_index()
        print("\nCluster Distribution:")
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id + 1}: {count} students ({100*count/len(df):.1f}%)")
        print(f"\nTotal students analyzed: {len(df)}")

        # Print cluster characteristics
        print("\nCluster Characteristics (mean values):")
        cluster_summary = df.groupby("cluster")[
            ["average_time_diff", "average_edit_distance", "total_submissions"]
        ].mean()
        print(cluster_summary.to_string())
