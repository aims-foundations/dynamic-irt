from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, Output, Reference, VALID_SPLIT, CORRECT_TAG
from huggingface_hub import hf_hub_download
import pandas as pd


class CodeInsightsStudentMistakeScenario(Scenario):
    name = "codeinsights_student_mistake"
    description = "Mimic how students mistake their C++ codes on foundational questions"
    tags = ["codeinsights", "c++", "student_mistake"]

    def __init__(self, num_testcases: int = 1):
        super().__init__()
        self.num_testcases = num_testcases

    def get_instances(self, output_path: str):
        data_file = hf_hub_download(
            repo_id="CodeInsightTeam/code_insights_csv",
            repo_type="dataset",
            filename="codeinsights_llm_simulation/data/Scenario3_full_data.csv",
            revision="b2ed07387d109af257089734a14fd7beee273bd9",
        )
        df = pd.read_csv(data_file, dtype={"pass": "str"})
        student_topic_file = hf_hub_download(
            repo_id="CodeInsightTeam/code_insights_csv",
            repo_type="dataset",
            filename="codeinsights_llm_simulation/data/student_performace_by_topic.csv",
            revision="b2ed07387d109af257089734a14fd7beee273bd9",
        )
        student_topic = pd.read_csv(student_topic_file)

        instances = []
        for student_id, student_df in df.groupby("student_id"):
            # sort by student, question, then time
            student_df = student_df.sort_values(by=["student_id", "question_unittest_id", "timestamp"])
            if len(student_df) < 4:
                continue

            # grab exactly the first 4 attempts
            attempts = student_df.iloc[:4]

            # build the student‐level summary once
            student_level_prompt = f"Student {student_id} has the following performance across topics:\n"
            topic_performance = student_topic[student_topic["student_id"] == student_id]
            for _, row in topic_performance.iterrows():
                student_level_prompt += (
                    f"- For topic '{row['topic']}', the unit test pass rate is "
                    f"{row['pass_rate']:.2f}, and the rate of passing all unit tests is {row['perfect']:.2f}.\n"
                )

            # rotate so each of the 4 becomes the “target” once
            for idx in range(4):
                target = attempts.iloc[idx]
                examples = [attempts.iloc[i] for i in range(4) if i != idx]

                # skip if no test‐cases available
                question_id = target.get("question_unittest_id", None)

                # parse test cases
                question_test_cases = []
                tc_parsing_success = True
                for s in target["question_unittests"].split("Unittest")[1:]:
                    body = s[s.find(":") + 1 :]
                    i1, i2, i3 = body.find("Input:"), body.find("STD input:"), body.find("Output:")
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

                # build the few‐shot + target prompt
                prompt = (
                    "=== Student Profile ===\n"
                    f"{student_level_prompt}\n"
                    "When students submit a code to the platform, it will be tested by number of unit tests, where\n"
                    "- Unit test pass rate = proportion of unit tests passed with the code\n"
                    "- Full pass rate   = proportion of code passing all unit tests\n\n"
                    "=== Past Mistake Examples ===\n"
                )
                for n, ex in enumerate(examples, start=1):
                    prompt += (
                        f"Example {n} (Week {ex['week']}, Topic: {ex['topic']}):\n"
                        f"Question: {ex['question_name']} — {ex['question_text']}\n"
                        "Template:\n"
                        f"{ex['question_template']}\n"
                        "Student's Response Code with Error:\n"
                        f"{ex['response_mistake']}\n\n"
                    )
                prompt += (
                    "=== New Target Problem ===\n"
                    f"Week: {target['week']}, Topic: {target['topic']}\n"
                    f"Question: {target['question_name']} — {target['question_text']}\n" + "Template:\n"
                    f"{target['question_template']}\n\n"
                    "⚠**Instructions:**\n"
                    "1. Mimic your own coding style, naming conventions, indentation, and typical error patterns from the examples.\n"
                    "2. Introduce a mistake you are likely to make (e.g., off‑by‑one index, wrong initialization, missing edge case).\n"
                    "3. Do **not** produce a fully correct solution or add unfamiliar optimizations.\n\n"
                    "4. Include any needed class definitions, and make sure the code is compatible with the Unit Test Input.\n"
                    "5. Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template.\n"
                    "6. Do NOT reproduce any part of the template.\n"
                    "7. Do NOT emit `int main()` (it’s already declared).\n\n"
                    "IMPORTANT: your entire response must be exactly one Markdown C++ code‑block:\n"
                    "1. First line: ```cpp\n"
                    "2. Last line: ```\n"
                    "No extra characters, whitespace, or text before/after.\n"
                )

                print(f"\n=== DEBUG INFO FOR STUDENT {student_id}, QUESTION {question_id} ===")
                print(f"Test cases loaded: {len(question_test_cases)}")
                print(f"Student correctness pattern: {student_correctness_list}")
                print(f"Original pass field: {target.get('pass', 'MISSING')}")
                print(f"Question template exists: {'question_template' in target}")
                print(f"Question name: {target.get('question_name', 'MISSING')}")

                # Also add this validation in your UnitTestAlignmentMetric evaluate_generation method:
                def evaluate_generation(self, adapter_spec, request_state, metric_service, eval_cache_path):
                    print("\n=== UNIT TEST METRIC DEBUG ===")
                    print(f"Has extra_data: {hasattr(request_state.instance, 'extra_data')}")
                    if hasattr(request_state.instance, "extra_data"):
                        extra_data = request_state.instance.extra_data
                        print(f"Extra data keys: {list(extra_data.keys())}")
                        print(f"Test cases: {len(extra_data.get('test_cases', []))}")
                        print(f"Student pattern: {extra_data.get('student_correctness_pattern', 'MISSING')}")

                instances.append(
                    Instance(
                        id=f"{student_id}_{target['question_unittest_id']}",
                        input=Input(text=prompt),
                        references=[Reference(output=Output(text=target["response_mistake"]), tags=[CORRECT_TAG])],
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
