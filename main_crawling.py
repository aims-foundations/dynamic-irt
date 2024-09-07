"""
Run this file to crawl students' scores.
"""

import argparse
import json
import os
import pathlib
import re
import string
from urllib.parse import urlparse

import wandb
from configs import DATA_LINKS, LOGIN_PASSWD, LOGIN_USER
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--course_name", help="Class Name", type=str, default="DSA-HK231")
parser.add_argument("--class_name", help="Class Name", type=str, default="L09")
args = parser.parse_args()

if __name__ == "__main__":
    # Chrome setup
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 20)

    course_name = args.course_name
    class_name = args.class_name

    wandb.init(project="student-score-crawler")

    target_link = DATA_LINKS[course_name][class_name]
    parsed_url = urlparse(target_link)
    # Extract the domain
    domain = parsed_url.netloc
    # Login
    driver.get(
        "https://sso.hcmut.edu.vn/cas/login?service=https%3A%2F%2F"
        f"{domain}/login/index.php%3FauthCAS%3DCAS"
    )
    username = driver.find_element(By.ID, "username")
    password = driver.find_element(By.ID, "password")
    username.send_keys(LOGIN_USER)
    password.send_keys(LOGIN_PASSWD)
    password.send_keys(Keys.RETURN)

    # Get course homepage
    driver.get(target_link)
    xpath_expression = (
        f"//a[starts-with(@href, 'https://{domain}/mod/quiz/view.php?id=')]"
    )
    filtered_links = driver.find_elements(By.XPATH, xpath_expression)
    QUIZZES_RESULT_LINKS = [
        link.get_attribute("href").replace("view.php", "report.php") + "&mode=overview"
        for link in filtered_links
    ]
    current_path = pathlib.Path().resolve()
    data_path = f"{current_path}/data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # Create folders
    os.makedirs(f"{data_path}/{course_name}/{class_name}", exist_ok=True)

    # print(driver.current_url)
    all_data = []
    for quiz_link in tqdm(QUIZZES_RESULT_LINKS, desc="Crawling"):
        print(quiz_link)
        driver.get(quiz_link)

        data = {
            "lab_name": driver.title.split(":")[0].strip(),
            "list_questions": [],
            "student_answers": [],
        }

        student_rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

        for index, row in enumerate(student_rows):
            if "emptyrow" in row.get_attribute("class"):
                print("Skipping empty row.")
                continue

            try:
                student_link = f"mod-quiz-report-overview-report_r{index}_c2"
                link = wait.until(EC.presence_of_element_located((By.ID, student_link)))

                try:
                    student_name = link.find_element(By.TAG_NAME, "a").text.strip()
                    review_link = link.find_element(
                        By.CSS_SELECTOR, "a.reviewlink"
                    ).get_attribute("href")
                    student_id = row.find_element(
                        By.CSS_SELECTOR, "td.cell.c3"
                    ).text.strip()
                    print(student_name, review_link, student_id)

                    data["student_answers"].append(
                        {
                            "name": student_name,
                            "id": student_id,
                            "review_link": review_link,
                        }
                    )
                except NoSuchElementException:
                    print(
                        f"Missing expected elements within the link ID {student_link}"
                    )
                    continue

            except TimeoutException:
                print(f"Element with ID {student_link} did not appear in time.")
                continue

        if not data["student_answers"]:
            print("There are no students in this quiz link.")
            continue

        driver.get(data["student_answers"][0]["review_link"])
        try:
            questions = wait.until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, ".que.coderunner")
                )
            )
            list_questions = []

            for question in questions:
                question_text = " ".join(
                    question.find_element(
                        By.CSS_SELECTOR, "div.content div.formulation"
                    ).text.split()
                )

                coderunner_examples_div = wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div.coderunner-examples")
                    )
                )
                expected_output_table = coderunner_examples_div.find_element(
                    By.CSS_SELECTOR, "table.coderunnerexamples"
                )

                rows = expected_output_table.find_elements(By.CSS_SELECTOR, "tbody tr")
                expected_outputs = []
                for row in rows:
                    test_cell = row.find_element(
                        By.CSS_SELECTOR, "td.cell.c0 pre.tablecell"
                    ).text
                    result_cell = row.find_element(
                        By.CSS_SELECTOR, "td.cell.c1 pre.tablecell"
                    ).text
                    expected_outputs.append({"test": test_cell, "result": result_cell})

                list_questions.append(
                    {"question": question_text, "expected_outputs": expected_outputs}
                )

            data["list_questions"] = list_questions
        except TimeoutException:
            print("Timeout.")
            continue

        for record in data["student_answers"]:
            driver.get(record["review_link"])
            attempt_data = []

            history_headers = driver.find_elements(
                By.XPATH, "//h4[contains(text(), 'Response history')]"
            )

            for index, header in enumerate(history_headers):
                table = header.find_element(
                    By.XPATH, "following-sibling::div//table[@class='generaltable']"
                )
                rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
                table_data = {"question": f"Question {index+1}", "results": []}
                for row in rows:
                    cells = row.find_elements(By.CSS_SELECTOR, "td")
                    if len(cells) >= 5:
                        step = cells[0].text
                        time = cells[1].text
                        action = cells[2].text
                        state = cells[3].text
                        marks = cells[4].text

                        table_data["results"].append(
                            {
                                "step": step,
                                "time": time,
                                "action": action,
                                "state": state,
                                "marks": marks,
                            }
                        )

                attempt_data.append(table_data)

            record["response_history"] = attempt_data

        all_data.append(data)

        first_part = driver.title.split(":")[0]
        filename = re.sub(f"[{string.punctuation}]", "_", first_part) + ".json"
        with open(
            f"{data_path}/{course_name}/{class_name}/{filename}", "w", encoding="utf-8"
        ) as json_file:
            json.dump(data, json_file)

    metrics = {
        "total_quiz_links": len(QUIZZES_RESULT_LINKS),
        "total_students": sum(len(data["student_answers"]) for data in all_data),
    }
    wandb.log(metrics)

    wandb.finish()
    driver.quit()

    # Get course homepage
    driver.get(target_link)
    xpath_expression = (
        f"//a[starts-with(@href, 'https://{domain}/mod/quiz/view.php?id=')]"
    )
    filtered_links = driver.find_elements(By.XPATH, xpath_expression)
    QUIZZES_RESULT_LINKS = [
        link.get_attribute("href").replace("view.php", "report.php") + "&mode=overview"
        for link in filtered_links
    ]
    current_path = pathlib.Path().resolve()
    data_path = f"{current_path}/data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # Create folders
    os.makedirs(f"{data_path}/{course_name}/{class_name}", exist_ok=True)

    # print(QUIZZES_RESULT_LINKS)
    all_data = []
    for quiz_link in tqdm(QUIZZES_RESULT_LINKS, desc="Crawling"):
        print(quiz_link)
        driver.get(quiz_link)

        data = {
            "lab_name": driver.title.split(":")[0].strip(),
            "list_questions": [],
            "student_answers": [],
        }

        input_field = driver.find_elements(By.ID, "id_pagesize")
        input_field[0].clear()

        # Set the new value to '100'
        input_field[0].send_keys("100")

        # Simulate pressing the Enter key
        input_field[0].send_keys(Keys.ENTER)

        table = driver.find_elements(By.CSS_SELECTOR, "table.generaltable")
        if len(table) > 0:
            headers = table[0].find_elements(By.TAG_NAME, "th")
        else:
            continue

        column_names = [header.text for header in headers[9:]]
        max_scores = [parse_score(column_name) for column_name in column_names]
        print(max_scores)

        student_rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

        for index, row in enumerate(student_rows):
            if "emptyrow" in row.get_attribute("class"):
                print("Skipping empty row.")
                continue

            try:
                student_link = f"mod-quiz-report-overview-report_r{index}_c2"
                link = wait.until(EC.presence_of_element_located((By.ID, student_link)))

                try:
                    student_name = link.find_element(By.TAG_NAME, "a").text.strip()
                    review_link = link.find_element(
                        By.CSS_SELECTOR, "a.reviewlink"
                    ).get_attribute("href")
                    student_id = row.find_element(
                        By.CSS_SELECTOR, "td.cell.c3"
                    ).text.strip()
                    print(student_name, review_link, student_id)

                    data["student_answers"].append(
                        {
                            "name": student_name,
                            "id": student_id,
                            "review_link": review_link,
                        }
                    )
                except NoSuchElementException:
                    print(
                        f"Missing expected elements within the link ID {student_link}"
                    )
                    continue

            except TimeoutException:
                print(f"Element with ID {student_link} did not appear in time.")
                continue

        if not data["student_answers"]:
            print("There are no students in this quiz link.")
            continue

        driver.get(data["student_answers"][0]["review_link"])
        try:
            questions = wait.until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, ".que.coderunner")
                )
            )
            list_questions = []

            for q_idx, question in enumerate(questions):
                question_text = " ".join(
                    question.find_element(
                        By.CSS_SELECTOR, "div.content div.formulation"
                    ).text.split()
                )

                coderunner_examples_div = wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div.coderunner-examples")
                    )
                )
                expected_output_table = coderunner_examples_div.find_element(
                    By.CSS_SELECTOR, "table.coderunnerexamples"
                )

                rows = expected_output_table.find_elements(By.CSS_SELECTOR, "tbody tr")
                expected_outputs = []
                for row in rows:
                    test_cell = row.find_element(
                        By.CSS_SELECTOR, "td.cell.c0 pre.tablecell"
                    ).text
                    result_cell = row.find_element(
                        By.CSS_SELECTOR, "td.cell.c1 pre.tablecell"
                    ).text
                    expected_outputs.append({"test": test_cell, "result": result_cell})

                list_questions.append(
                    {
                        "question": question_text,
                        "expected_outputs": expected_outputs,
                        "max_scores": max_scores[q_idx],
                    }
                )

            data["list_questions"] = list_questions
        except TimeoutException:
            print("Timeout.")
            continue

        for record in data["student_answers"]:
            driver.get(record["review_link"])
            attempt_data = []

            history_headers = driver.find_elements(
                By.XPATH, "//h4[contains(text(), 'Response history')]"
            )

            for index, header in enumerate(history_headers):
                table = header.find_element(
                    By.XPATH, "following-sibling::div//table[@class='generaltable']"
                )
                rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
                table_data = {"question": f"Question {index+1}", "results": []}
                for row in rows:
                    cells = row.find_elements(By.CSS_SELECTOR, "td")
                    if len(cells) >= 5:
                        step = cells[0].text
                        time = cells[1].text
                        action = cells[2].text
                        state = cells[3].text
                        marks = cells[4].text

                        table_data["results"].append(
                            {
                                "step": step,
                                "time": time,
                                "action": action,
                                "state": state,
                                "marks": marks,
                            }
                        )

                attempt_data.append(table_data)

            record["response_history"] = attempt_data

        all_data.append(data)

        first_part = driver.title.split(":")[0]
        filename = re.sub(f"[{string.punctuation}]", "_", first_part) + ".json"
        with open(
            f"{data_path}/{course_name}/{class_name}/{filename}", "w", encoding="utf-8"
        ) as json_file:
            json.dump(data, json_file)

    metrics = {
        "total_quiz_links": len(QUIZZES_RESULT_LINKS),
        "total_students": sum(len(data["student_answers"]) for data in all_data),
    }
    wandb.log(metrics)

    wandb.finish()
    driver.quit()
