import argparse
import json
import os
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import html2text

import wandb
from configs import (
    chrome_binary_path,
    chromedriver_path,
    DATA_LINKS,
    LOGIN_PASSWD,
    LOGIN_USER,
)
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from utils import compute_ed, filter_class_group, run_crawler, safe_find_element


class CrawlData:
    def __init__(
        self,
        course_name,
        class_name,
        chromedriver_path,
        chrome_binary_path,
        domain,
        timeout=60,
    ):
        self.course_name = course_name
        self.class_name = class_name
        self.chromedriver_path = chromedriver_path
        self.chrome_binary_path = chrome_binary_path
        self.timeout = timeout
        self.driver = None
        self.domain = domain
        self.initialize_driver()

    def initialize_driver(self, reinitializing=False):
        if reinitializing:
            try:
                self.driver.quit()
            except Exception as e:
                print("No session to quit:", str(e))
        chrome_options = Options()
        chrome_options.binary_location = chrome_binary_path
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--ignore-ssl-errors=yes")
        chrome_options.add_argument("--ignore-certificate-errors")

        service = Service(executable_path=self.chromedriver_path)
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.login()

    def login(self):
        success = run_crawler(
            self,
            (
                "https://sso.hcmut.edu.vn/cas/login?service=https%3A%2F%2F"
                f"{self.domain}/login/index.php%3FauthCAS%3DCAS"
            ),
        )
        assert success, "Login problem!"
        username = WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_all_elements_located((By.ID, "username"))
        )[0]
        password = self.driver.find_element(By.ID, "password")
        username.send_keys(LOGIN_USER)
        password.send_keys(LOGIN_PASSWD)
        password.send_keys(Keys.RETURN)
        print("Login successfully")

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

    def get_questions(self, domain, course_id):
        # Ensure the crawler is alive
        success = run_crawler(
            self, f"https://{domain}/mod/quiz/edit.php?cmid={course_id}"
        )
        while not success:
            time.sleep(5)
            self.initialize_driver(reinitializing=True)
            success = run_crawler(
                self, f"https://{domain}/mod/quiz/edit.php?cmid={course_id}"
            )

        print("Question URL:", f"https://{domain}/mod/quiz/edit.php?cmid={course_id}")
        xpath_editquestion = f"//a[starts-with(@href, 'https://{domain}/question/bank/editquestion/question.php?returnurl=')]"

        try:
            filtered_editquestion_links = WebDriverWait(
                self.driver, self.timeout
            ).until(EC.presence_of_all_elements_located((By.XPATH, xpath_editquestion)))
        except:
            # This case is a list of random questions
            return self.get_list_random_question(domain, course_id)

        max_marks_elements = WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".instancemaxmark"))
        )

        # Questions tab links
        preview_links = [
            str(link.get_attribute("href")) for link in filtered_editquestion_links
        ]

        # Get max score for each question
        max_scores = [float(elem.text.strip()) for elem in max_marks_elements]

        list_questions = []
        for q_idx, question_link in enumerate(
            tqdm(preview_links, desc="Crawling questions")
        ):
            q_info = self.get_single_question(question_link)
            q_info["max_score"] = max_scores[q_idx]
            list_questions.append(q_info)

        return list_questions

    def get_single_question(self, question_link):
        success = run_crawler(self, question_link)
        while not success:
            time.sleep(5)
            self.initialize_driver(reinitializing=True)
            success = run_crawler(self, question_link)

        # Get question template
        try:
            question_template = (
                WebDriverWait(self.driver, self.timeout)
                .until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "#id_template")
                    )
                )[0]
                .get_attribute("innerText")
            )
        except:
            question_template = ""

        # Get question name
        question_name = (
            WebDriverWait(self.driver, self.timeout)
            .until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#id_name")))[
                0
            ]
            .get_attribute("value")
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
        try:
            list_testcase_input = WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_all_elements_located(
                    (
                        By.CSS_SELECTOR,
                        "#id_testcasehdrcontainer .testcaseexpression div textarea[name^='testcode']",
                    )
                )
            )
            list_std_input = WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_all_elements_located(
                    (
                        By.CSS_SELECTOR,
                        "#id_testcasehdrcontainer .testcasestdin div textarea[name^='stdin']",
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
            for tc_in, std_in, tc_out in zip(
                list_testcase_input, list_std_input, list_testcase_output
            ):
                list_testcases.append(
                    {
                        "input": tc_in.text,
                        "std_input": std_in.text,
                        "output": tc_out.text,
                    }
                )
        except:
            list_testcases = []

        return {
            "question": question_text,
            "name": question_name,
            "template": question_template,
            "testcases": list_testcases,
        }

    def get_list_random_question(self, domain, course_id):
        filtered_rdquestion_links = WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, ".mod_quiz_random_qbank_link")
            )
        )
        filtered_rdquestion_links = [
            str(link.get_attribute("href")) for link in filtered_rdquestion_links
        ]

        max_marks_elements = WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".instancemaxmark"))
        )

        # Get max score for each question
        max_scores = [float(elem.text.strip()) for elem in max_marks_elements]

        # Loop on each question set
        list_questions = []
        for q_idx, question_link in enumerate(
            tqdm(filtered_rdquestion_links, desc="Crawling questions")
        ):
            success = run_crawler(self, question_link)
            while not success:
                time.sleep(5)
                self.initialize_driver(reinitializing=True)
                success = run_crawler(self, question_link)

            all_question_rows = WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "table#categoryquestions tbody tr")
                )
            )
            list_sub_question_links = []

            # Get all question links in set
            for q_sub_idx, row in enumerate(all_question_rows):
                q_link = row.find_element(
                    By.CSS_SELECTOR, f"#action-menu-{q_sub_idx+1}-menu > a:nth-child(1)"
                ).get_attribute("href")
                list_sub_question_links.append(q_link)

            # Get all question in set
            list_sub_questions = []
            for q_link in tqdm(list_sub_question_links, desc="Crawling sub-questions"):
                q_info = self.get_single_question(q_link)
                q_info["max_score"] = max_scores[q_idx]
                list_sub_questions.append(q_info)

            list_questions.append(list_sub_questions)

        return list_questions

    def get_student_answers(
        self, student_record, list_question, question_randomized=False
    ):
        success = run_crawler(self, student_record["review_link"])
        while not success:
            time.sleep(5)
            self.initialize_driver(reinitializing=True)
            success = run_crawler(self, student_record["review_link"])

        # Pick question sub-index in case of randomized questions
        student_question_subidxs = []
        if question_randomized:
            student_questions = WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, ".que.coderunner")
                )
            )
            for qidx, question in enumerate(student_questions):
                question_text = safe_find_element(
                    question, By.CSS_SELECTOR, "div.content div.formulation"
                ).text
                question_text = question_text.replace("Question text\n", "")
                end_idx = question_text.rfind("\nFor example:")
                question_text = question_text[:end_idx]

                min_idx = -1
                min_ed = 1e5

                # Find index of the most similar question, assumming that elements in eds can be duplicated
                for subqidx, q in enumerate(list_question[qidx]):
                    ed = compute_ed(question_text, q["question"])
                    if min_idx == -1 or ed < min_ed:
                        min_idx = subqidx
                        min_ed = ed
                student_question_subidxs.append(min_idx + 1)

        attempt_data = []
        history_headers = WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//h4[contains(text(), 'Response history')]")
            )
        )

        # Loop through each question in student submission
        for index, header in enumerate(history_headers):
            table = header.find_element(
                By.XPATH, "following-sibling::div//table[@class='generaltable']"
            )

            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
            table_data = {"question": f"Question {index+1}", "results": []}
            if question_randomized:
                table_data["question"] += f".{student_question_subidxs[index]}"

            for row in rows:
                cells = row.find_elements(By.CSS_SELECTOR, "td")
                if len(cells) >= 5:
                    step = cells[0].text
                    time = cells[1].text
                    action = html2text.html2text(cells[2].get_attribute("innerHTML"))
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


def parallel_get_student_answer(
    args, domain, student_record, list_question, question_randomized
):
    crawler = CrawlData(
        args.course_name,
        args.class_name,
        chromedriver_path,
        chrome_binary_path,
        domain=domain,
        timeout=args.timeout,
    )
    result = crawler.get_student_answers(
        student_record, list_question, question_randomized
    )
    crawler.driver.quit()
    return result


if __name__ == "__main__":
    wandb.init(project="student-score-crawler")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--class_name", help="Class Name", type=str, default="CC01")
    parser.add_argument("--timeout", help="Timeout for waiting", type=int, default=60)
    # parser.add_argument("--max_workers", help="Max Workers", type=int, default=4)
    args = parser.parse_args()

    course_name = args.course_name
    class_name = args.class_name

    target_link = DATA_LINKS[course_name][class_name]
    parsed_url = urlparse(target_link)

    # Extract the domain
    domain = parsed_url.netloc

    crawler = CrawlData(
        args.course_name,
        args.class_name,
        chromedriver_path,
        chrome_binary_path,
        domain=domain,
        timeout=args.timeout,
    )

    # Find correct students in specified class
    parsed_url = urlparse(target_link)
    query_params = parse_qs(parsed_url.query)
    course_id = query_params.get("id", [None])[0]
    assert course_id is not None, "No course id found!"

    student_list_url = f"https://{domain}/user/index.php?page=0&perpage=1650&contextid=0&id={course_id}"
    success = run_crawler(crawler, student_list_url)
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
    os.makedirs(f"../data/{course_name}/{class_name}", exist_ok=True)

    all_data = []
    for quiz_link in tqdm(QUIZZES_RESULT_LINKS, desc="Crawling"):
        print("Running: ", quiz_link)
        success = run_crawler(crawler, quiz_link)
        if not success:
            print("Failed to get", quiz_link)
            continue

        try:
            # Set max number of students to 1650
            print("Setting max number of students")
            num_max_student = crawler.driver.find_element(By.ID, "id_pagesize")
            n_max_student = num_max_student.get_attribute("value")
            if n_max_student != "1650":
                for _ in range(10):
                    num_max_student.send_keys(Keys.BACKSPACE)
                num_max_student.send_keys("1650")
                num_max_student.send_keys(Keys.RETURN)
                WebDriverWait(crawler.driver, args.timeout).until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "table#attempts tbody tr")
                    )
                )
            else:
                print("Max number of students is already set to 1650")
        except:
            print("Failed to set max number of students")
            continue

        first_part = crawler.driver.title.split(":")[0]
        filename = re.sub(f"[{string.punctuation}]", "_", first_part)
        if os.path.exists(f"../data/{course_name}/{class_name}/{filename}.json"):
            print("Skipping: ", quiz_link)
            continue

        topic_data = {
            "lab_name": crawler.driver.title.split(":")[0].strip(),
            "list_questions": [],
            "student_answers": [],
        }

        try:
            student_rows = WebDriverWait(crawler.driver, args.timeout).until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "table#attempts tbody tr")
                )
            )
        except:
            student_rows = []

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
        topic_data["list_questions"] = crawler.get_questions(domain, course_id)

        # Crawl answers
        question_randomized = False
        if isinstance(topic_data["list_questions"][0], list):
            question_randomized = True

        # with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        #     futures = [
        #         executor.submit(
        #             parallel_get_student_answer,
        #             args,
        #             domain,
        #             record,
        #             topic_data["list_questions"],
        #             question_randomized,
        #         )
        #         for record in topic_data["student_answers"]
        #     ]
        #     for fi, future in enumerate(tqdm(futures, desc="Crawling student answers")):
        #         res = future.result()
        #         topic_data["student_answers"][fi]["response_history"] = res

        for fi, record in enumerate(
            tqdm(topic_data["student_answers"], desc="Crawling student answers")
        ):
            res = crawler.get_student_answers(
                record, topic_data["list_questions"], question_randomized
            )
            topic_data["student_answers"][fi]["response_history"] = res

        # Append to data
        all_data.append(topic_data)

        # Save data
        with open(
            f"../data/{course_name}/{class_name}/{filename}.json", "w", encoding="utf-8"
        ) as json_file:
            json.dump(topic_data, json_file)

    metrics = {
        "total_quiz_links": len(QUIZZES_RESULT_LINKS),
        "total_students": sum(len(data["student_answers"]) for data in all_data),
    }
    wandb.log(metrics)

    crawler.driver.quit()
    wandb.finish()
