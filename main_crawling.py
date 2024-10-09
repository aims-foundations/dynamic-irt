import argparse
import json
import os
import re
import string
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import html2text

import wandb
from configs import DATA_LINKS, LOGIN_PASSWD, LOGIN_USER
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from utils import (
    filter_class_group,
    get_test_cases,
    parse_score,
    run_crawler,
    safe_find_element,
    safe_navigate,
)
from webdriver_manager.chrome import ChromeDriverManager


class CrawlData:
    def __init__(
        self, course_name, class_name, chromedriver_path, chrome_binary_path, timeout=60
    ):
        self.course_name = course_name
        self.class_name = class_name
        self.chromedriver_path = chromedriver_path
        self.chrome_binary_path = chrome_binary_path
        self.timeout = timeout
        self.driver = None
        self.initialize_driver()

    def initialize_driver(self):
        chrome_options = Options()
        chrome_options.binary_location = chrome_binary_path
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--ignore-ssl-errors=yes")
        chrome_options.add_argument("--ignore-certificate-errors")

        service = Service(executable_path=self.chromedriver_path)
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def extract_data(self):
        student_details = []
        rows = WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "table#participants tbody tr")
            )
        )
        for row in tqdm(rows, desc="Extracting student infos"):
            if row.get_attribute("class") == "emptyrow":
                break

            id_number = row.find_element(By.CSS_SELECTOR, "td.cell.c2").text.strip()
            class_group = row.find_element(By.CSS_SELECTOR, "td.cell.c4").text.strip()

            if id_number and class_group:
                student_details.append({"ID": id_number, "Class Group": class_group})

        return student_details

    def get_question(self, domain, course_id):
        sucess = run_crawler(
            crawler, f"https://{domain}/mod/quiz/edit.php?cmid={course_id}"
        )
        xpath_editquestion = f"//a[starts-with(@href, 'https://{domain}/question/bank/editquestion/question.php?returnurl=')]"
        filtered_editquestion_links = WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_all_elements_located((By.XPATH, xpath_editquestion))
        )
        max_marks_elements = WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".instancemaxmark"))
        )

        # Questions tab
        preview_links = [
            str(link.get_attribute("href")) for link in filtered_editquestion_links
        ]

        # Get max score for each question
        max_scores = [float(elem.text.strip()) for elem in max_marks_elements]

        list_questions = []
        for q_idx, question_link in enumerate(
            tqdm(preview_links, desc="Crawling questions")
        ):
            run_crawler(self, question_link)

            # Get question template
            question_template = (
                WebDriverWait(self.driver, self.timeout)
                .until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "#id_template")
                    )
                )[0]
                .get_attribute("innerText")
            )

            # Get question text
            question_text = (
                WebDriverWait(self.driver, self.timeout)
                .until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "#id_questiontext")
                    )
                )[0]
                .get_attribute("innerText")
            )
            question_text = html2text.html2text(question_text)

            # Get test cases
            list_testcases = []
            list_testcase_input = WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_all_elements_located(
                    (
                        By.CSS_SELECTOR,
                        "#id_testcasehdrcontainer .testcaseexpression div textarea[name^='testcode']",
                    )
                )
            )
            list_testcase_output = WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_all_elements_located(
                    (
                        By.CSS_SELECTOR,
                        "#id_testcasehdrcontainer .testcaseresult div textarea[name^='expected']",
                    )
                )
            )
            for tc_in, tc_out in zip(list_testcase_input, list_testcase_output):
                tc_in_text = tc_in.text
                tc_out_text = tc_out.text
                list_testcases.append(
                    {
                        "input": tc_in_text,
                        "ouput": tc_out_text,
                    }
                )

            list_questions.append(
                {
                    "question": question_text,
                    "max_score": max_scores[q_idx],
                    "template": question_template,
                    "testcases": list_testcases,
                }
            )

        return list_questions

    def get_student_answers(self, student_record):
        run_crawler(self, student_record["review_link"])
        attempt_data = []

        history_headers = WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//h4[contains(text(), 'Response history')]")
            )
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

                    result_entry = {
                        "step": step,
                        "time": time,
                        "action": action,
                        "state": state,
                        "score": marks,
                    }

                    table_data["results"].append(result_entry)

            attempt_data.append(table_data)
        return attempt_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Class Name", type=str, default="DSA-HK231"
    )
    parser.add_argument("--class_name", help="Class Name", type=str, default="CC01")
    parser.add_argument("--timeout", help="Timeout for waiting", type=int, default=90)
    args = parser.parse_args()

    # Chrome setup
    chromedriver_path = "/afs/cs.stanford.edu/u/nqduc/chromedriver-linux64/chromedriver"
    chrome_binary_path = "/afs/cs.stanford.edu/u/nqduc/chrome-linux64/chrome"
    crawler = CrawlData(
        args.course_name,
        args.class_name,
        chromedriver_path,
        chrome_binary_path,
        timeout=args.timeout,
    )

    course_name = args.course_name
    class_name = args.class_name

    wandb.init(project="student-score-crawler")

    target_link = DATA_LINKS[course_name][class_name]
    parsed_url = urlparse(target_link)

    # Extract the domain
    domain = parsed_url.netloc

    # Login
    success = run_crawler(
        crawler,
        (
            "https://sso.hcmut.edu.vn/cas/login?service=https%3A%2F%2F"
            f"{domain}/login/index.php%3FauthCAS%3DCAS"
        ),
    )
    assert success, "Login problem!"
    username = crawler.driver.find_element(By.ID, "username")
    password = crawler.driver.find_element(By.ID, "password")
    username.send_keys(LOGIN_USER)
    password.send_keys(LOGIN_PASSWD)
    password.send_keys(Keys.RETURN)
    print("Login successfully")

    # Find correct students in specified class
    parsed_url = urlparse(target_link)
    query_params = parse_qs(parsed_url.query)
    course_id = query_params.get("id", [None])[0]
    assert course_id is not None, "No course id found!"

    student_list_url = (
        f"https://{domain}/user/index.php?page=0&perpage=200&contextid=0&id={course_id}"
    )
    sucess = run_crawler(crawler, student_list_url)
    assert success, "Failed to get student list!"
    print("Access success!")

    # Start extraction student info from the first page
    student_details = crawler.extract_data()

    # ------------------------------------------------

    # Get all quiz links
    xpath_expression = (
        f"//a[starts-with(@href, 'https://{domain}/mod/quiz/view.php?id=')]"
    )
    filtered_links = crawler.driver.find_elements(By.XPATH, xpath_expression)
    QUIZZES_RESULT_LINKS = [
        str(
            link.get_attribute("href").replace("view.php", "report.php")
            + "&mode=overview"
        )
        for link in filtered_links
    ]

    # Create data folder
    os.makedirs(f"data/{course_name}/{class_name}", exist_ok=True)

    all_data = []
    for quiz_link in tqdm(QUIZZES_RESULT_LINKS, desc="Crawling"):
        print("Running: ", quiz_link)
        sucess = run_crawler(crawler, quiz_link)
        if not sucess:
            print("Failed to get", quiz_link)
            continue

        topic_data = {
            "lab_name": crawler.driver.title.split(":")[0].strip(),
            "list_questions": [],
            "student_answers": [],
        }

        student_rows = WebDriverWait(crawler.driver, args.timeout).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "table#attempts tbody tr")
            )
        )

        # Filter students in the specified class
        for index, row in enumerate(tqdm(student_rows, desc="Filtering students")):
            if row.get_attribute("class") == "emptyrow":
                break

            if row.find_elements(By.CSS_SELECTOR, "td.cell.c2 a"):
                try:
                    student_link = f"mod-quiz-report-overview-report_r{index}_c2"
                    link = crawler.driver.find_element(By.ID, student_link)

                    # Parse student info
                    student_name = None
                    review_link = None
                    student_id = row.find_element(
                        By.CSS_SELECTOR, "td.cell.c3"
                    ).text.strip()

                    if filter_class_group(student_details, student_id, class_name):
                        student_name = link.find_element(By.TAG_NAME, "a").text.strip()
                        review_link = row.find_element(
                            By.CSS_SELECTOR, "a.reviewlink"
                        ).get_attribute("href")

                    if student_name and review_link:
                        topic_data["student_answers"].append(
                            {
                                "name": student_name,
                                "id": student_id,
                                "review_link": review_link,
                            }
                        )

                except NoSuchElementException as e:
                    print(f"Element not found in row {index}: {str(e)}")

                except TimeoutException:
                    print(f"Element with ID {student_link} did not appear in time.")

        if not topic_data["student_answers"]:
            print("There are no students in this quiz link in this class.")
            continue

        parsed_url = urlparse(quiz_link)
        query_params = parse_qs(parsed_url.query)

        # Extract id
        course_id = query_params.get("id", [None])[0]
        if course_id is None:
            raise ValueError("ID parameter not found in URL")

        # Crawl questions
        topic_data["list_questions"] = crawler.get_question(domain, course_id)

        # Crawl answers
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(crawler.get_student_answers, record)
                for record in topic_data["student_answers"]
            ]
            for fi, future in enumerate(tqdm(futures, desc="Crawling student answers")):
                res = future.result()
                topic_data["student_answers"][fi]["response_history"] = res

        # Append to data
        all_data.append(topic_data)

        first_part = crawler.driver.title.split(":")[0]
        filename = re.sub(f"[{string.punctuation}]", "_", first_part) + ".json"
        with open(
            f"data/{course_name}/{class_name}/{filename}", "w", encoding="utf-8"
        ) as json_file:
            json.dump(topic_data, json_file)

    metrics = {
        "total_quiz_links": len(QUIZZES_RESULT_LINKS),
        "total_students": sum(len(data["student_answers"]) for data in all_data),
    }
    wandb.log(metrics)

    crawler.driver.quit()
    wandb.finish()
