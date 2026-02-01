from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, VALID_SPLIT
from huggingface_hub import hf_hub_download
import pandas as pd


class CodeInsightsCorrectCodeScenario(Scenario):
    name = "codeinsights_correct_code"
    description = "Generate correct response code for C++ programming questions"
    tags = ["codeinsights", "c++", "correct_code"]

    def __init__(self, num_testcases: int = 1):
        super().__init__()
        self.num_testcases = num_testcases

    def get_instances(self, output_path: str):
        data_file = hf_hub_download(
            repo_id="CodeInsightTeam/code_insights_csv",
            repo_type="dataset",
            filename="codeinsights_llm_simulation/data/Scenario1_full_data.csv",
            revision="b2ed07387d109af257089734a14fd7beee273bd9",
        )

        df = pd.read_csv(data_file, dtype={"pass": "str"})

        # Load test cases (unit tests)
        instances = []
        for question_id, question_df in df.groupby("question_unittest_id"):
            target = question_df.iloc[0]
            question_test_cases = []
            tc_parsing_success = True

            for testcase_str in target["question_unittests"].split("Unittest")[1:]:
                testcase_str = testcase_str[testcase_str.find(":") + 1 :]
                input_idx = testcase_str.find("Input:")
                std_in_idx = testcase_str.find("STD input:")
                output_idx = testcase_str.find("Output:")
                if input_idx == -1 or std_in_idx == -1 or output_idx == -1:
                    tc_parsing_success = False
                    break

                testcase = {
                    "input": testcase_str[input_idx + 6 : std_in_idx].strip(),
                    "std_in": testcase_str[std_in_idx + 10 : output_idx].strip(),
                    "output": testcase_str[output_idx + 7 :].strip(),
                }
                question_test_cases.append(testcase)

            if not tc_parsing_success or len(question_test_cases) < self.num_testcases:
                # If not enough test cases, skip this question
                continue

            if self.num_testcases >= 0:
                # If more than one test case is requested, only take the first ones
                question_test_cases = question_test_cases[: self.num_testcases]

            prompt = (
                f"Question: {target['question_name']} — {target['question_text']}\n\n"
                "Template:\n"
                f"{target['question_template']}\n\n"
                "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template."
                "– Do NOT reproduce any part of the template"
                "– Do NOT emit `int main()` (it’s already declared)"
                "– Ensure your code is correct, efficient, handles all edge cases, and includes any needed class definitions"
                "IMPORTANT:"
                "Your entire response must be exactly one Markdown C++ code-block."
                "1. The first line of your output must be:"
                "```cpp"
                "2. The last line of your output must be:"
                "```"
                "3. No extra characters, whitespace, or text may appear before the opening ```cpp or after the closing ```."
                "Your output will therefore match this regex exactly:"
                "^```cpp\n([\s\S]+)\n```$"
            )
            instances.append(
                Instance(
                    id=f"{question_id}",
                    input=Input(text=prompt),
                    references=[],
                    extra_data={
                        "question_template": target["question_template"],
                        "test_cases": question_test_cases,
                        "question_id": str(question_id) if question_id else None,
                        "question_name": target.get("question_name", ""),
                    },
                    split=VALID_SPLIT,
                )
            )
        return instances
