import os
import json
import pandas as pd
from utils import (
    response_to_df,
    get_unittest_infos,
    compile_and_execute,
    compute_scenario_metrics,
    get_scenario_student_df,
)

LIST_OS_LLMS = [
    "google_gemma-3-27b-it",
    "meta_llama-3.1-8b-instruct",
    "qwen_qwen2.5-14b-instruct",
]

LIST_LLMS = [
    "google_gemma-3-27b-it",
    "meta_llama-3.1-8b-instruct",
    "qwen_qwen2.5-14b-instruct",
    "gpt-4o",
    "claude-3-5",
    "gemini-2.5-pro",
    "mistral"
]

LIST_SCENARIOS = ["S1", "S2", "S3", "S4"]

DATA_FOLDER = "https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv/codeinsights_llm_simulation/resolve/main/"

if __name__ == "__main__":
    # Convert outputs to dataframe
    print("Converting results to df...")
    for scenario in LIST_SCENARIOS:
        for llm in LIST_OS_LLMS:
            output_path = os.path.join(
                DATA_FOLDER,
                f"opensource_llm_output/{llm}/{scenario}/scenario_state.json",
            )
            # Pre-processing model outputs
            output_df = response_to_df(path=output_path, scenario=scenario)

            # Save output dataframe
            scenario_id = scenario[1:]
            llm = "_".join(llm.split("_")[1:])
            os.makedirs(
                os.path.join(
                    DATA_FOLDER,
                    f"scenario_results/{llm}",
                ),
                exist_ok=True,
            )
            output_df.to_csv(
                os.path.join(
                    DATA_FOLDER,
                    f"scenario_results/{llm}/{llm}_scenario{scenario_id}.csv",
                )
            )
            print(f"Saved {scenario} - {llm}")

    # Read question data
    question_data = pd.read_csv(
        os.path.join(DATA_FOLDER, "data/codeinsights_question.csv")
    )

    # Compute metrics
    all_results = {}
    for scenario in LIST_SCENARIOS:
        print(f"Computing metrics for scenario {scenario}...")
        scenario_id = scenario[1:]
        scenario_student_df = get_scenario_student_df(
            scenario=scenario, data_folder=DATA_FOLDER
        )

        scenario_testcase_json = get_unittest_infos(
            scenario=scenario, data_folder=DATA_FOLDER
        )

        all_results[scenario] = {}

        for llm in LIST_LLMS:
            print(f"    + LLM: {llm}...")
            # Load results
            if llm in LIST_OS_LLMS:
                llm = "_".join(llm.split("_")[1:])
            scenario_output_df = pd.read_csv(
                os.path.join(
                    DATA_FOLDER,
                    f"scenario_results/{llm}/{llm}_scenario{scenario_id}.csv",
                )
            )
            scenario_output_df["question_id"] = scenario_output_df[
                "question_id"
            ].astype(str)
            if scenario != "S1":
                scenario_output_df["student_id"] = scenario_output_df[
                    "student_id"
                ].astype(str)

            # Run LLM-generated code
            scenario_result_df = compile_and_execute(
                question_data=question_data,
                scenario_output_df=scenario_output_df,
                scenario_testcase_json=scenario_testcase_json,
                has_stid=(scenario != "S1"),
            )
            kwargs = {}

            if scenario == "S4":
                # Filter student data to only those which have LLM results
                scenario_student_df[["student_id", "question_id"]] = (
                    scenario_student_df[["student_id", "question_id"]].astype(str)
                )

                # Now do the inner‐join on those string keys
                scenario_student_subset = scenario_student_df.merge(
                    scenario_result_df[["student_id", "question_id"]].drop_duplicates(),
                    on=["student_id", "question_id"],
                    how="inner",
                )

                # Run student code
                scenario_student_result_df = compile_and_execute(
                    question_data=question_data,
                    scenario_output_df=scenario_student_subset,
                    scenario_testcase_json=scenario_testcase_json,
                )
                kwargs["scenario_student_result_df"] = scenario_student_result_df

            # Compute metrics
            all_results[scenario][llm] = compute_scenario_metrics(
                scenario=scenario,
                scenario_df=scenario_output_df,
                scenario_result_df=scenario_result_df,
                scenario_student_df=scenario_student_df,
                **kwargs,
            )

    # Save all results
    with open("all_results.json", "w", encoding="utf8") as f:
        json.dump(all_results, f, indent=2)
