import json
import os
import re
import time
from urllib.parse import parse_qs, unquote, urlparse, urlsplit

from selenium import webdriver

from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def run_crawler(crawler, url):
    trial = 0
    success = False
    while trial < 10:
        try:
            crawler.driver.get(url)
            success = True
            break
        except:
            trial += 1
            time.sleep(3)

    return success


def parse_score(header_text):
    search_result = re.search(r"/(\d+\.\d+)", header_text)
    if search_result:
        return float(search_result.group(1))
    else:
        return None


def filter_class_group(details, sid, class_name):
    for detail in details:
        if detail["Class Group"] == "No groups":
            if detail["ID"] == sid:
                return True

        if detail["ID"] == sid and (
            detail["Class Group"] == class_name or detail["Class Group"] == "All"
        ):
            return True

    return False


def safe_find_element(driver, by, value):
    try:
        element = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except TimeoutException:
        print(f"Timeout while trying to find element {value}")
        return None


def safe_navigate(driver, service, chrome_options, url):
    try:
        driver.get(url)
    except WebDriverException:
        print(f"Error navigating to {url}. Attempting to recover...")
        driver.quit()
        driver = webdriver.Chrome(
            service=service, options=chrome_options
        )  # Reinitialize the driver
        driver.get(url)
    return driver


def get_test_cases(driver):
    i = 0
    test_cases = []
    while True:
        test_code_id = f"id_testcode_{i}"
        expected_id = f"id_expected_{i}"
        try:
            test_code_element = driver.find_element(By.ID, test_code_id)
            expected_output_element = driver.find_element(By.ID, expected_id)
            if test_code_element and expected_output_element:
                test_code = test_code_element.get_attribute("value").strip()
                expected_output = expected_output_element.get_attribute("value").strip()
                test_cases.append(
                    {
                        "test_case_no": i + 1,
                        "test_code": test_code,
                        "expected_output": expected_output,
                    }
                )
                i += 1
            else:
                break
        except NoSuchElementException:
            break  # Break if either element is not found
    return test_cases


def find_global_max(repo_id, course_name, class_name):
    global_max = 0

    for root, dirs, files in os.walk(repo_id):
        if class_name in dirs:
            class_dir_path = os.path.join(root, class_name)

            for subdir_root, subdir_dirs, subdir_files in os.walk(class_dir_path):
                for file in subdir_files:
                    if file.endswith(".json"):
                        file_path = os.path.join(subdir_root, file)
                        try:
                            with open(file_path, "r") as json_file:
                                data = json.load(json_file)
                                ids = []
                                for answers in data["student_answers"]:
                                    ids.append(answers["id"])

                                for idx in range(len(data["list_questions"])):
                                    max_score = data["list_questions"][idx][
                                        "max_scores"
                                    ]
                                    q_index = idx + 1

                                    records = []
                                    for answers in data["student_answers"]:
                                        for answer in answers["response_history"]:
                                            marks = []
                                            if (
                                                answer["question"]
                                                == f"Question {q_index}"
                                            ):
                                                mark_per_attempt = []
                                                for score_idx in range(
                                                    len(answer["results"])
                                                ):
                                                    if (
                                                        answer["results"][score_idx][
                                                            "marks"
                                                        ]
                                                        != ""
                                                    ):
                                                        mark_per_attempt.append(
                                                            answer["results"][
                                                                score_idx
                                                            ]["marks"]
                                                        )

                                            marks.append(mark_per_attempt)
                                        records.extend(marks)

                                    try:
                                        records = [
                                            [
                                                float(mark) * 10 / max_score
                                                for mark in student_marks
                                            ]
                                            for student_marks in records
                                        ]
                                    except:
                                        continue

                                    max_attempts = [
                                        len(student_marks) for student_marks in records
                                    ]
                                    global_max = max(global_max, max(max_attempts))

                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from file: {file_path}: {e}")
    return global_max
