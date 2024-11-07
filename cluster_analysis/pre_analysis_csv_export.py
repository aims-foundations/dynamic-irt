#!/usr/bin/env python
# coding: utf-8


import json
import os

import pandas as pd

# from collections import Counter
# import matplotlib.pyplot as plt
# import ast #handle and process abstract syntax trees


def extract_action_category(action):
    return action.split(":", 1)[0].strip()


def create_flat_dataframe(df):
    flat_data = []
    for _, row in df.iterrows():
        id_value = row["id"]
        results = row["results"]

        for result in results:
            flat_data.append(
                {
                    "id": id_value,
                    "step": result.get("step", ""),
                    "time": result.get("time", ""),
                    "action": extract_action_category(result.get("action", "")),
                    "state": result.get("state", ""),
                    "marks": result.get("marks", ""),
                }
            )
    return pd.DataFrame(flat_data)


def create_filtered_flat_dataframe(df):
    # New function to create filtered dataframe with only "Started" and "Submit" action
    flat_data = []
    for _, row in df.iterrows():
        id_value = row["id"]
        results = row["results"]

        for result in results:
            action = extract_action_category(result.get("action", ""))
            if action in ["Started", "Submit"]:
                flat_data.append(
                    {
                        "id": id_value,
                        "step": result.get("step", ""),
                        "time": result.get("time", ""),
                        "action": action,
                        "state": result.get("state", ""),
                        "marks": result.get("marks", ""),
                    }
                )
    return pd.DataFrame(flat_data)


directory = "data/week_2"

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
            df = pd.DataFrame(data["student_answers"])
            flat_df = create_flat_dataframe(df)
            flat_df["time"] = pd.to_datetime(
                flat_df["time"], format="%d/%m/%y, %H:%M:%S"
            )
            flat_df = flat_df.sort_values(["id", "time"])
            flat_df["time_delta"] = flat_df.groupby("id")["time"].diff()

            filtered_flat_df = create_filtered_flat_dataframe(df)
            filtered_flat_df["time"] = pd.to_datetime(
                filtered_flat_df["time"], format="%d/%m/%y, %H:%M:%S"
            )
            filtered_flat_df = filtered_flat_df.sort_values(["id", "time"])
            filtered_flat_df["time_delta"] = filtered_flat_df.groupby("id")[
                "time"
            ].diff()

            # Export to .csv
            df_name = f"df_{filename.replace('.json', '').replace('-', '_')}"
            csv_filename = f"{df_name}.csv"
            flat_df.to_csv(csv_filename, index=False)

            filtered_df_name = (
                f"filtered_df_{filename.replace('.json', '').replace('-', '_')}"
            )
            filtered_csv_filename = f"{filtered_df_name}.csv"
            filtered_flat_df.to_csv(filtered_csv_filename, index=False)

            print(f"Processed: {filename}")
            print("---")
