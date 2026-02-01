from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, Output, Reference, VALID_SPLIT, CORRECT_TAG
from huggingface_hub import hf_hub_download
import pandas as pd


class CodeInsightsStudentCodingScenario(Scenario):
    name = "codeinsights_student_coding"
    description = "Mimic student C++ style on foundational questions"
    tags = ["codeinsights", "c++", "student_coding"]

    def __init__(self, num_testcases: int = 1):
        super().__init__()
        self.num_testcases = num_testcases

    def get_instances(self, output_path: str):
        data_file = hf_hub_download(
            repo_id="CodeInsightTeam/code_insights_csv",
            repo_type="dataset",
            filename="codeinsights_llm_simulation/data/Scenario2_full_data.csv",
            revision="b2ed07387d109af257089734a14fd7beee273bd9",
        )
        df = pd.read_csv(
            data_file,
            dtype={"pass": "str"},
        )

        student_topic_file = hf_hub_download(
            repo_id="CodeInsightTeam/code_insights_csv",
            repo_type="dataset",
            filename="codeinsights_llm_simulation/data/student_performace_by_topic.csv",
            revision="b2ed07387d109af257089734a14fd7beee273bd9",
        )
        student_topic = pd.read_csv(student_topic_file)

        instances = []
        for student_id, student_df in df.groupby("student_id"):
            student_df = student_df.sort_values("timestamp")
            if len(student_df) < 4:
                continue

            # only look at the first 4 attempts
            attempts = student_df.iloc[:4]

            # build the student‐level profile once per student
            student_level_prompt = f"Student {student_id} has the following performance across topics:\n"
            topic_performance = student_topic[student_topic["student_id"] == student_id]
            for _, row in topic_performance.iterrows():
                student_level_prompt += (
                    f"- For topic '{row['topic']}', the unit test pass rate is "
                    f"{row['pass_rate']:.2f}, and the rate of passing all tests is {row['perfect']:.2f}.\n"
                )

            # now rotate each of the 4 attempts in turn as the 'target'
            for idx in range(4):
                # examples = all but the idx-th attempt
                example_rows = [attempts.iloc[i] for i in range(4) if i != idx]
                target = attempts.iloc[idx]
                question_id = target.get("question_unittest_id", None)
                # parse the target's test cases
                question_test_cases = []
                tc_parsing_success = True
                for testcase_str in target["question_unittests"].split("Unittest")[1:]:
                    body = testcase_str[testcase_str.find(":") + 1 :]
                    i1 = body.find("Input:")
                    i2 = body.find("STD input:")
                    i3 = body.find("Output:")
                    if -1 in (i1, i2, i3):
                        tc_parsing_success = False
                        break
                    question_test_cases.append(
                        {
                            "input": body[i1 + 6 : i2].strip(),
                            "std_in": body[i2 + 10 : i3].strip(),
                            "output": body[i3 + 7 :].strip(),
                        }
                    )

                # parse the target's pass/fail pattern
                correctness = target.get("pass", "")
                if "." in correctness:
                    raise RuntimeError("The type of `pass` column should be string!")
                student_correctness_list = [int(ch) for ch in correctness]

                if (
                    not tc_parsing_success
                    or len(question_test_cases) < self.num_testcases
                    or len(student_correctness_list) < self.num_testcases
                ):
                    continue

                if self.num_testcases >= 0:
                    question_test_cases = question_test_cases[: self.num_testcases]
                    student_correctness_list = student_correctness_list[: self.num_testcases]

                # Ensure that the number of testcase matches with the student results
                if len(question_test_cases) != len(student_correctness_list) or len(question_test_cases) == 0:
                    continue

                # build the few‐shot prompt
                prompt = (
                    "=== Student Profile ===\n"
                    f"{student_level_prompt}\n"
                    f"Week: {target['week']}\n"
                    f"Topic: {target['topic']}\n\n"
                )
                for n, ex in enumerate(example_rows, start=1):
                    prompt += (
                        f"Example {n}:\n"
                        f"Question: {ex['question_name']} — {ex['question_text']}\n"
                        f"Template:\n{ex['question_template']}\n"
                        f"Your Code:\n{ex['response']}\n\n"
                    )
                prompt += (
                    "Now, using that same student style, attempt this:\n"
                    f"Question: {target['question_name']} — {target['question_text']}\n"
                    f"Template:\n{target['question_template']}\n\n"
                    "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template.  "
                    "– Do NOT reproduce any part of the template  "
                    "– Do NOT emit `int main()` (it’s already declared)  "
                    "– Ensure your code mirrors the style of the previous examples and includes any necessary class definitions  "
                    "IMPORTANT: your entire response must be exactly one Markdown C++ code‐block:\n"
                    "1. First line: ```cpp\n"
                    "2. Last line: ```\n"
                    "Your output will therefore match this regex exactly:"
                    "^```cpp\n([\s\S]+)\n```$"
                )
                instances.append(
                    Instance(
                        id=f"{student_id}_{target['question_unittest_id']}",
                        input=Input(text=prompt),
                        references=[Reference(output=Output(text=target["response"]), tags=[CORRECT_TAG])],
                        extra_data={
                            "question_template": target["question_template"],
                            "test_cases": question_test_cases,
                            "question_id": str(question_id) if question_id else None,
                            "question_name": target.get("question_name", ""),
                            "student_id": str(student_id),
                            "student_correctness_pattern": student_correctness_list,
                        },
                        split=VALID_SPLIT,
                    )
                )
        return instances
