import requests
import pandas as pd
import re
import sys
import json
import os
import tempfile
import subprocess
import numpy as np
import time
from jinja2 import Template
from typing import List, Dict
from cossim_calculator import CodeSimilarityCalculator


def response_to_df(path, scenario):
    if "http" in path:
        response = requests.get(path)
        raw = response.json()
    else:
        raw = json.load(open(path, "r", encoding="utf8"))

    # Figure out where the state‐objects live
    if isinstance(raw, dict) and "request_states" in raw:
        states = raw["request_states"]
    elif isinstance(raw, list):
        states = raw
    else:
        raise ValueError(
            "JSON must be either a dict with 'request_states' or a list of state‐objects"
        )

    # Extract completion text
    texts = []
    for state in states:
        for comp in state.get("result", {}).get("completions", []):
            txt = comp.get("text")
            texts.append(txt)

    # Extract instance IDs
    ids = []
    for state in states:
        id_val = state.get("instance", {}).get("id")
        ids.append(str(id_val))

    # Create DataFrame
    if scenario == "S1":
        df = pd.DataFrame({"question_id": [int(x) for x in ids], "text": texts})
    else:
        parsed = [
            (int(sid), int(qid), text)
            for s, text in zip(ids, texts)
            if "_" in s
            for sid, qid in [s.split("_")]
        ]
        df = pd.DataFrame(parsed, columns=["student_id", "question_id", "text"])

    return df


# Define Functions


def extract_student_code(model_code: str) -> str:
    """
    Extracts clean C++ code from model output:
    - Trims preambles
    - Removes student's main()
    """
    code_blocks = re.findall(r"```(?:c\+\+)?\n(.*?)```", model_code, flags=re.DOTALL)
    if code_blocks:
        model_code = code_blocks[0].strip()  # Use the first code block
        print("[Markdown extraction] Used fenced code blocks.")

    # Post-processing
    # Comment out as a testing - 7/3/2025
    lines = model_code.strip().splitlines()
    start_keywords = ("#include", "using namespace")
    for i, line in enumerate(lines):
        if any(line.strip().startswith(k) for k in start_keywords):
            lines[i] = ""
    code = "\n".join(lines).strip()
    if "int main" in code:
        code = code.split("int main")[0].strip()

    # --- Final touch ---
    if "print(" in code and "void print()" not in code and "print()" not in code:
        print("⚠️ WARNING: `print()` is called in test input but not defined.")

    return code


def parse_unittests(block: str):
    pattern = re.compile(
        r"Unittest\s*(\d+):\s*"
        r"Input:\s*(.*?)\s*"
        r"Output:\s*(.*?)(?=Unittest\s*\d+:|$)",
        flags=re.DOTALL,
    )
    tests = []
    for m in pattern.finditer(block):
        tests.append(
            {
                "unittest": m.group(1).strip(),
                "input": m.group(2).strip(),
                "output": m.group(3).strip(),
            }
        )
    return tests


def format_testcases(test_case):
    """Formats the test cases into the required format for the grading engine.

    Returns:
        Tuple[List[Dict[str]], List[str]]: A tuple containing the formatted test cases and standard inputs.
    """
    formatted_testcases = []
    std_inputs = []
    for testcase in test_case:
        formatted_testcases.append(
            {
                "testcode": testcase["input"],
                "expected_output": testcase["output"],
            }
        )
        if "std_in" not in testcase:
            std_inputs.append("")
        else:
            std_inputs.append(testcase["std_in"])
    return formatted_testcases, std_inputs


def generate_code(
    template: str,
    student_answer: str,
    formatted_testcases: List[Dict[str, str]],
) -> List[str]:
    """
    Generates one C++ file per test case by rendering the Jinja2 template.

    Args:
        template:     Your question_template (with Jinja2 tags).
        student_answer:  The raw LLM output, including ```cpp fences.
        formatted_testcases: A list of dicts, where each dict provides the keys
            'extra' (pre‑test code) and 'testcode' (the call or check).

    Returns:
        A list of fully rendered C++ source strings, one per testcase.
    """
    # Strip any ```cpp ... ``` markdown fences
    student_answer = re.sub(r"^```cpp\s*|\s*```$", "", student_answer)

    # Compile the Jinja2 template once
    j2 = Template(template)

    rendered_codes: List[str] = []
    error_flags: List[bool] = []

    rendered_codes: List[str] = []
    for tc in formatted_testcases:
        # clean up the testcase
        tc["testcode"] = tc["testcode"].replace("STD input:", "").strip()

        try:
            code = j2.render(STUDENT_ANSWER=student_answer, TESTCASES=[tc])
        except TypeError as e:
            # record that this testcase failed, log if you like
            error_flags.append(True)
            print(
                f"[Warning] TypeError rendering testcase {tc!r}: {e}", file=sys.stderr
            )
            continue
        else:
            error_flags.append(False)
            rendered_codes.append(code)

    return rendered_codes


def get_scenario_student_df(scenario_id, data_folder):
    scenario_student_df = pd.read_csv(
        os.path.join(
            data_folder,
            f"Scenario{scenario_id}_full_data.csv",
        ),
        dtype={"pass": str}
    )
    return scenario_student_df


def get_unittest_infos(scenario, data_folder):
    scenario_student_df = get_scenario_student_df(scenario, data_folder)
    scenario_student_df["tests"] = scenario_student_df["question_unittests"].apply(
        parse_unittests
    )

    result = {
        str(int(qid)): tests
        for qid, tests in zip(
            scenario_student_df["question_id"], scenario_student_df["tests"]
        )
    }
    jsonfile = json.dumps(result, indent=2)
    scenario_testcase_json = json.loads(jsonfile)
    return scenario_testcase_json


def compile_and_execute(
    question_data, scenario_output_df, scenario_testcase_json, has_stid=True
):
    # Run LLM generated codes and make result dataframe
    scenario_results = []

    for i in range(len(scenario_output_df)):
        test_result = extract_student_code(scenario_output_df.iloc[i]["text"])
        id_val = int(scenario_output_df.iloc[i]["question_id"])
        if has_stid:
            student_id = scenario_output_df.iloc[i]["student_id"]
        else:
            student_id = None
        sub_question = question_data[question_data["question_id"] == id_val]

        # format testcases
        try:
            formatted_testcases, std_inputs = format_testcases(
                scenario_testcase_json[str(id_val)]
            )
            expected_output = [tc["expected_output"] for tc in formatted_testcases]
        except KeyError:
            # record the failure with no specific testcase
            scenario_results.append(
                {
                    "student_id": student_id,
                    "question_id": id_val,
                    "test_case_id": None,
                    "cpp_code": None,
                    "stdout": None,
                    "expected_output": None,
                    "run_time": None,
                }
            )
            continue
        # generate code
        try:
            codes = generate_code(
                sub_question["question_template"].iloc[0],
                test_result,
                formatted_testcases,
            )
        except Exception as e:
            scenario_results.append(
                {
                    "student_id": student_id,
                    "question_id": id_val,
                    "test_case_id": None,
                    "cpp_code": None,
                    "stdout": None,
                    "expected_output": None,
                    "run_time": None,
                }
            )
            continue

        # compile & run each snippet
        for j, cpp_code in enumerate(codes):
            stdout_val = None
            expected_output_val = expected_output[j]
            # default row template
            row = {
                "student_id": student_id,
                "question_id": id_val,
                "test_case_id": j,
                "cpp_code": cpp_code,
                "stdout": None,
                "expected_output": expected_output_val,
                "run_time": None,
            }

            with tempfile.TemporaryDirectory() as tmpdir:
                cpp_path = os.path.join(tmpdir, "test.cpp")
                exe_path = os.path.join(tmpdir, "test")

                # write out the code
                with open(cpp_path, "w") as f:
                    f.write(cpp_code)

                # Compile
                compile_proc = subprocess.run(
                    ["g++", "-std=c++17", cpp_path, "-o", exe_path],
                    capture_output=True,
                    text=True,
                )
                if compile_proc.returncode != 0:
                    # record failure, leave stdout as None (or store compile_proc.stderr if you like)
                    scenario_results.append(row)
                    continue

                start = time.perf_counter()
                try:
                    run_proc = subprocess.run(
                        [exe_path],
                        capture_output=True,
                        text=True,
                        timeout=5,  # ← kill the process if it runs longer than 10s
                    )
                    end = time.perf_counter()
                except subprocess.TimeoutExpired as e:
                    end = time.perf_counter()
                    print(
                        f"Testcase {j} for question {id_val} timed out after {e.timeout}s",
                        file=sys.stderr,
                    )
                    # record the timeout as a failure
                    row["run_time"] = "fail"
                    scenario_results.append(row)
                    continue
                row["run_time"] = end - start

                if run_proc.returncode != 0:
                    print("Runtime error:\n", run_proc.stderr, file=sys.stderr)
                    scenario_results.append(row)
                    continue
                row["stdout"] = run_proc.stdout
                scenario_results.append(row)
                print(f"Processed question {id_val}, test case {j}")

    # Store result in dataframe
    scenario_result_df = pd.DataFrame(scenario_results)
    scenario_result_df["question_id"] = scenario_result_df["question_id"].astype(str)
    scenario_result_df["test_case_id"] = scenario_result_df["test_case_id"].astype(str)
    scenario_result_df["correctness"] = pd.Series(dtype="Int64")
    scenario_result_df["stdout"] = scenario_result_df["stdout"].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    return scenario_result_df


def compute_function_correctness(scenario_result_df, **kwargs):
    # Evaluate correctness of LLM generated code
    # Where stdout == expected_output → 1, else → 0
    matches = scenario_result_df["stdout"] == scenario_result_df["expected_output"]
    scenario_result_df.loc[matches, "correctness"] = 1
    scenario_result_df.loc[~matches, "correctness"] = 0
    # If test_case_id is missing, reset correctness to <NA>
    mask_missing = scenario_result_df["test_case_id"].isna()
    scenario_result_df.loc[mask_missing, "correctness"] = pd.NA
    # Where stdout == expected_output → 1, else → 0
    scenario_result_df["stdout"] = scenario_result_df["stdout"].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    scenario1_result_df = scenario_result_df.rename(
        columns={
            "correctness": "LLM_correctness",
        }
    )
    functional_correctnes = scenario1_result_df["LLM_correctness"].mean()
    print(f"Functional correctness: {functional_correctnes:.2%}")

    return {"functional_correctnes": functional_correctnes}


def verify_results(scenario_result_df, scenario_student_df):
    # Unit-test Correctness Alignment
    # Student Pass Patterns
    scenario_student_df["num_unittest"] = (
        scenario_student_df["question_unittests"]
        .astype(str)
        .str.count(r"Unittest\s+\d+:")
    )

    # 2) pad on the right with “0” to match num_unittest
    scenario_student_df["pass_padded"] = scenario_student_df.apply(
        lambda r: r["pass"].ljust(int(r["num_unittest"]), "0"), axis=1
    )
    # 3) explode into one digit per row
    df_exploded = (
        scenario_student_df.assign(
            pass_list=scenario_student_df["pass_padded"].apply(
                lambda s: [int(ch) for ch in s]
            )
        )
        .explode("pass_list")
        .reset_index()  # keep original row‑index for grouping
    )

    # 4) get your unittest_id and rename columns
    df_result = (
        df_exploded.assign(unittest_id=df_exploded.groupby("index").cumcount())
        .loc[:, ["student_id", "question_id", "unittest_id", "pass_list"]]
        .rename(columns={"pass_list": "pass_unittest"})
        .reset_index(drop=True)
    )
    df_result["question_id"] = df_result["question_id"].astype(int).astype(str)
    df_result["student_id"] = df_result["student_id"].astype(str)
    df_result["unittest_id"] = df_result["unittest_id"].astype(str)
    df_result = df_result.rename(
        columns={
            "unittest_id": "test_case_id",
            "pass_unittest": "real_student_correctness",
        }
    )
    scenario_merged_df = scenario_result_df.merge(
        df_result,
        on=["student_id", "question_id", "test_case_id"],
        how="inner",
        # optional, to disambiguate any overlapping column names
        suffixes=("_sc2", "_sc1"),
    )
    mask = scenario_merged_df["LLM_correctness"].isin([0, 1]) & scenario_merged_df[
        "real_student_correctness"
    ].isin([0, 1])
    valid = scenario_merged_df[mask]
    return scenario_merged_df, valid


def compute_ed_and_similarity(
    scenario_df, scenario_result_df, scenario_student_df, valid, **kwargs
):
    # 2) check equality
    matches = valid["LLM_correctness"] == valid["real_student_correctness"]
    unit_test_correctness_alignment = matches.mean()
    print(f"Unit Test Correctness Alignment: {unit_test_correctness_alignment:.2%}")

    # AST Edit Distance and CodeBERT Similarity
    student_response_subset = scenario_student_df[
        ["student_id", "question_id", "response"]
    ]
    student_response_subset = student_response_subset.rename(
        columns={"response": "student_response"}
    )
    scenario_df = scenario_df.rename(columns={"text": "LLM_response"})
    scenario_response_full = scenario_df.merge(
        student_response_subset, on=["student_id", "question_id"], how="inner"
    )
    calculator = CodeSimilarityCalculator()
    result_df = calculator.process_dataframe(scenario_response_full)
    ast_edit_distance = result_df["ast_edit_distance"].mean()
    codebert_cosine_similarity = result_df["codebert_cosine_similarity"].mean()
    print(f"AST Edit Distance: {ast_edit_distance}")
    print(f"CodeBERT Cosine Similarity: {codebert_cosine_similarity}")

    return {
        "correctness_alignment": unit_test_correctness_alignment,
        "ast_edit_distance": ast_edit_distance,
        "codebert_cosine_similarity": codebert_cosine_similarity,
    }


# Question-level mistake alignment
def compute_mistake_alignment(scenario_merged_df, **kwargs):
    question_props = (
        scenario_merged_df.groupby("question_id")
        .agg(
            LLM_question_mistake_prop=("LLM_correctness", lambda x: (x == 0).mean()),
            student_question_mistake_prop=(
                "real_student_correctness",
                lambda x: (x == 0).mean(),
            ),
        )
        .reset_index()
    )

    # 2) Compute squared difference per question
    question_props["squared_diff"] = (
        question_props["LLM_question_mistake_prop"]
        - question_props["student_question_mistake_prop"]
    ) ** 2

    # 3) RMSE across all questions
    question_level_mistake_alignment_score = np.sqrt(
        question_props["squared_diff"].mean()
    )
    print(
        "Question‑level mistake alignment score:",
        question_level_mistake_alignment_score,
    )
    return {"mistake_alignment": question_level_mistake_alignment_score}


# generate dataframe for runtime analysis


def compute_efficiency_alignment(
    scenario_result_df, scenario_student_result_df, **kwargs
):
    scenario_student_result_df = scenario_student_result_df.rename(
        columns={"run_time": "student_run_time", "cpp_code": "student_cpp_code"}
    )
    scenario_student_result_df["student_id"] = scenario_student_result_df[
        "student_id"
    ].astype(str)
    scenario_student_result_df["question_id"] = scenario_student_result_df[
        "question_id"
    ].astype(str)
    scenario_student_result_df["test_case_id"] = scenario_student_result_df[
        "test_case_id"
    ].astype(str)
    scenario_result_df = scenario_result_df.rename(
        columns={"run_time": "LLM_run_time", "cpp_code": "LLM_cpp_code"}
    )
    student_runtime_subset = scenario_student_result_df[
        [
            "student_id",
            "question_id",
            "test_case_id",
            "student_cpp_code",
            "student_run_time",
        ]
    ]
    LLM_runtime_subset = scenario_result_df[
        ["student_id", "question_id", "test_case_id", "LLM_cpp_code", "LLM_run_time"]
    ]
    runtime_data = student_runtime_subset.merge(
        LLM_runtime_subset,
        on=["student_id", "question_id", "test_case_id"],
        how="inner",
    )
    # 1) coerce to float, turning non‑numbers into NaN
    runtime_data["student_rt_num"] = pd.to_numeric(
        runtime_data["student_run_time"], errors="coerce"
    )
    runtime_data["LLM_rt_num"] = pd.to_numeric(
        runtime_data["LLM_run_time"], errors="coerce"
    )
    mask = runtime_data["student_rt_num"].notna() & runtime_data["LLM_rt_num"].notna()
    # 3) Compute the efficiency ratio: student_time / LLM_time
    #    (a ratio >1 means student is slower; <1 means LLM is slower)
    runtime_data.loc[mask, "efficiency_ratio"] = (
        runtime_data.loc[mask, "student_rt_num"] / runtime_data.loc[mask, "LLM_rt_num"]
    )
    efficiency_alignment_score = runtime_data["efficiency_ratio"].mean()
    print(f"Efficiency alignment score: {efficiency_alignment_score}")
    return {"efficiency_alignment": efficiency_alignment_score}


def compute_scenario_metrics(scenario, **kwargs):
    if scenario == "S1":
        return compute_function_correctness(**kwargs)

    elif scenario == "S2":
        _, valid = verify_results(
            kwargs["scenario_result_df"], kwargs["scenario_student_df"]
        )
        metrics = compute_function_correctness(**kwargs)
        metrics.update(compute_ed_and_similarity(valid=valid, **kwargs))
        return metrics

    elif scenario == "S3":
        scenario_merged_df, valid = verify_results(
            kwargs["scenario_result_df"], kwargs["scenario_student_df"]
        )
        metrics = compute_function_correctness(**kwargs)
        metrics.update(compute_ed_and_similarity(valid=valid, **kwargs))
        metrics.update(
            compute_mistake_alignment(scenario_merged_df=scenario_merged_df, **kwargs)
        )
        return metrics

    elif scenario == "S4":
        scenario_merged_df, valid = verify_results(
            kwargs["scenario_result_df"], kwargs["scenario_student_df"]
        )
        metrics = compute_function_correctness(**kwargs)
        metrics.update(compute_ed_and_similarity(valid=valid, **kwargs))
        metrics.update(compute_efficiency_alignment(**kwargs))
        return metrics

    else:
        raise NotImplementedError(f"Scenario {scenario} is not implemented.")
