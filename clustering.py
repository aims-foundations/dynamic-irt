import os
import pickle
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from Levenshtein import distance as levenshtein_distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def convert_to_seconds(timestamp_str):
    timestamp_format = "%d/%m/%y, %H:%M:%S"
    datetime_obj = datetime.strptime(timestamp_str, timestamp_format)
    return datetime_obj.timestamp()


def group_students(main_dir):
    response_matrix = load_pickle(f"{main_dir}/response_matrix.pkl")
    time_matrix = response_matrix = load_pickle(f"{main_dir}/time_matrix.pkl")
    student_ids = load_pickle(f"{main_dir}/student_ids.pkl")
    unique_questions = load_pickle(f"{main_dir}/unique_questions.pkl")
    # print(student_ids)

    question_text = "52CD+"

    question_index = None
    for idx, question in enumerate(unique_questions):
        if question_text in question:
            question_index = idx
            break

    if question_index is None:
        print("No question like that.")
        return

    students_who_attempted = []
    student_metrics = []

    for s_idx, questions in enumerate(response_matrix):
        if len(questions) > question_index:
            attempts = questions[question_index]
            if any(attempts):
                students_who_attempted.append(s_idx)

    for s_idx in students_who_attempted:
        responses = response_matrix[s_idx][question_index]
        times = time_matrix[s_idx][question_index]

        valid_indices = [
            i
            for i, resp in enumerate(responses)
            if resp and not resp.startswith("Saved:")
        ]
        valid_responses = [responses[i] for i in valid_indices]
        valid_times = [times[i] for i in valid_indices]

        # Calculate total submissions
        total_submissions = len(valid_responses)

        # Calculate average time difference
        if len(valid_times) > 1:
            time_diffs = [
                abs(
                    convert_to_seconds(valid_times[i])
                    - convert_to_seconds(valid_times[i - 1])
                )
                for i in range(1, len(valid_times))
            ]
            average_time_diff = np.mean(time_diffs)
        else:
            average_time_diff = 0

        # Calculate average edit distance
        if len(valid_responses) > 1:
            edit_distances = [
                levenshtein_distance(valid_responses[i], valid_responses[i - 1])
                for i in range(1, len(valid_responses))
            ]
            average_edit_distance = np.mean(edit_distances)
        else:
            average_edit_distance = 0

        student_metrics.append(
            {
                "student_id": student_ids[s_idx]["student_id"],
                "average_time_diff": average_time_diff,
                "total_submissions": total_submissions,
                "average_edit_distance": average_edit_distance,
            }
        )

    return student_metrics


def cluster_students(main_dir):
    directory_name = os.path.basename(main_dir)
    student_metrics = group_students(main_dir)
    df = pd.DataFrame(student_metrics)

    features = df[["average_time_diff", "average_edit_distance", "total_submissions"]]
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(features)

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
    normalized_centroids_df.to_csv(f"centroids_{directory_name}.csv", index=False)

    df.to_csv(f"student_metrics_{directory_name}.csv", index=False)
    print("Data saved!")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["blue", "green", "red", "purple"]
    for i in range(4):
        clustered_data = df[df["cluster"] == i]
        ax.scatter(
            clustered_data["average_time_diff"],
            clustered_data["average_edit_distance"],
            clustered_data["total_submissions"],
            color=colors[i],
            label=f"Cluster {i+1}",
            alpha=0.6,
            edgecolors="w",
            s=100,
        )

    # Plot centroids
    ax.scatter(
        centroids["Average time difference"],
        centroids["Average edit distance"],
        centroids["Total submissions"],
        color="black",
        marker="x",
        s=200,
        label="Centroids",
    )

    ax.set_title(f"Students cluster in {directory_name}")
    ax.set_xlabel("Average time difference")
    ax.set_ylabel("Average edit distance")
    ax.set_zlabel("Total submissions")
    ax.legend()

    # Save the figure
    plt.savefig(f"students_cluster_{directory_name}.png", format="png", dpi=300)

    # Display the plot
    plt.close()

    return df, centroids


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    args = parser.parse_args()

    main_dir = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}", repo_type="dataset"
    )

    df, centroids = cluster_students(main_dir)
    cluster_counts = df["cluster"].value_counts().sort_index()
    print(cluster_counts)