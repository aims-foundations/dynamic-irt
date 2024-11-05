import json
import os
import warnings
from argparse import ArgumentParser
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait

from config import CLASSES, WEEK_FILES
from grading_engine.engine import CPPEvaluator
from tqdm import tqdm


def preprocess_answer(answer):
    for prefix in ["Saved: ", "Prechecked: ", "Submit: "]:
        answer = answer.replace(prefix, "")
    return answer


def process_one_student(student_answer):
    # for si, student_answer in enumerate(tqdm(student_answers, desc="Testing student")):
    list_errors = []
    for qi, question in enumerate(student_answer["response_history"]):
        if "." in question["question"]:
            dotidx = question["question"].rfind(".")
            qidx = int(float(question["question"].split(" ")[-1])) - 1
            sub_qidx = int(question["question"][dotidx + 1 :]) - 1
        else:
            qidx = int(question["question"].split(" ")[-1]) - 1
            sub_qidx = None

        for ai, attempt in enumerate(question["results"]):
            run_test = False
            if attempt["action"].startswith("Started") or attempt["action"].startswith(
                "Attempt finished"
            ):
                pass
            elif attempt["action"].startswith("Prechecked") or attempt[
                "action"
            ].startswith("Saved"):
                if attempt["score"] != "":
                    run_test = True
            elif attempt["action"].startswith("Submit"):
                run_test = True

            if run_test:
                answer = preprocess_answer(attempt["action"])
                evaluator = list_evaluators[wi][topic_file][qidx]
                if sub_qidx is not None:
                    evaluator = evaluator[sub_qidx]

                student_results = evaluator.evaluate(answer)
                if attempt["score"] == "":
                    attempt["score"] = "0.0"

                if student_results["score"] != float(attempt["score"]):
                    # Raise warning
                    warnings.warn(
                        f"Score mismatch for {student_answer['id']}, question {qidx}. Expected: {attempt['score']}, Got: {student_results['score']}."
                    )
                    list_errors.append(
                        abs(float(attempt["score"]) - student_results["score"])
                    )
                else:
                    list_errors.append(0)

                student_answer["response_history"][qi]["results"][ai]["testcases"] = (
                    student_results["testcases"]
                )
            else:
                student_answer["response_history"][qi]["results"][ai]["testcases"] = []

    if len(list_errors) == 0:
        return 0

    print("error:", sum(list_errors) / len(list_errors))
    return sum(list_errors) / len(list_errors)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    parser.add_argument("--class_name", type=str, default="CC01")
    parser.add_argument("--start_week", type=int, default=1)
    args = parser.parse_args()

    class_name = args.class_name
    course_name = args.course_name

    list_files = os.listdir(os.path.join("data", course_name.upper(), class_name))
    list_files = [x for x in list_files if x.endswith("json")]
    list_files = sorted(list_files)

    # Check if list file has type WEEK_FILES[args.course_name][0] or WEEK_FILES[args.course_name][1]
    if list_files[0] in WEEK_FILES[course_name][1][0]:
        weeks = WEEK_FILES[course_name][1]
    else:
        weeks = WEEK_FILES[course_name][0]

    # Do the same for exam
    weeks.append(WEEK_FILES[course_name]["exam"])

    # Loop on all weeks to create set of CPPEvaluator
    # One CPPEvaluator corresponds to one question in one week
    list_evaluators = []

    # >>> weeks x questions
    list_res = []

    for wi, week in enumerate(weeks[args.start_week - 1 :]):
        print("Running week", wi)
        list_evaluators.append({})
        for topic_file in week:
            print("Running topic:", topic_file)

            if not os.path.exists(
                os.path.join("data", course_name.upper(), class_name, topic_file)
            ):
                print(f"Topic {topic_file} not found!")
                continue

            # Load the questions and testcases
            week_data = json.load(
                open(
                    os.path.join("data", course_name.upper(), class_name, topic_file),
                    "r",
                )
            )

            list_questions = week_data["list_questions"]
            week_evaluators = []
            for question_data in list_questions:
                if isinstance(question_data, list):
                    # The case of random question set
                    question_set = []
                    for subquestion in question_data:
                        template = subquestion["template"]
                        testcases = subquestion["testcases"]

                        # Create the CPPEvaluator
                        evaluator = CPPEvaluator(template, testcases, max_workers=4)
                        question_set.append(evaluator)

                    week_evaluators.append(question_set)
                else:
                    template = question_data["template"]
                    testcases = question_data["testcases"]

                    # Create the CPPEvaluator
                    evaluator = CPPEvaluator(template, testcases, max_workers=4)
                    week_evaluators.append(evaluator)

            list_evaluators[wi][topic_file] = week_evaluators

            # Load the student answers
            student_answers = week_data["student_answers"]

            executables = []
            with ThreadPoolExecutor(max_workers=16) as executor:
                # Submit all compilation tasks
                futures = [
                    executor.submit(process_one_student, sa) for sa in student_answers
                ]

                # Retrieve results as they complete
                for future in tqdm(futures, desc="Testing student"):
                    result = future.result()
                    list_res.append(result)

                wait(futures, return_when=ALL_COMPLETED)

            # Save the updated student answers
            week_data["student_answers"] = student_answers
            json.dump(
                week_data,
                open(
                    os.path.join("data", course_name.upper(), class_name, topic_file),
                    "w",
                ),
            )

            print("Mean error:", sum(list_res) / len(list_res))
