import argparse
import json
import os
import pathlib
import re
import string
from urllib.parse import parse_qs, urlparse

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
from utils import filter_class_group, get_test_cases, parse_score, safe_navigate, safe_find_element
from webdriver_manager.chrome import ChromeDriverManager

class CrawlData:
    def __init__(self, course_name, class_name, chromedriver_path, chrome_binary_path):
        self.course_name = course_name
        self.class_name = class_name
        self.chromedriver_path = chromedriver_path
        self.chrome_binary_path = chrome_binary_path
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
        rows = WebDriverWait(self.driver, 30).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "table#participants tbody tr")
            )
        )
        for row in rows:
            id_number = row.find_element(By.CSS_SELECTOR, "td.cell.c2").text.strip()
            class_group = row.find_element(By.CSS_SELECTOR, "td.cell.c4").text.strip()

            if id_number and class_group:
                print(f"Extracted ID: {id_number}, Class Group: {class_group}")
                student_details.append({"ID": id_number, "Class Group": class_group})

        return student_details

    def navigate_pagination(self):
        try:
            pagination_links = self.driver.find_elements(
                By.CSS_SELECTOR, "nav.pagination-centered ul.pagination li.page-item a"
            )
            total_pages = len(pagination_links) - 1
            for i in range(1, total_pages):
                next_page_link = WebDriverWait(self.driver, 30).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, f"li.page-item[data-page-number='{i + 1}'] a")
                    )
                )
                self.driver.execute_script("arguments[0].click();", next_page_link)

                WebDriverWait(self.driver, 30).until(
                    EC.staleness_of(next_page_link)
                )  # Ensure the old page link is stale
                self.extract_data()
        except TimeoutException:
            print(
                f"Timeout occurred when trying to navigate to page {i + 1}. Check if the page exists and is accessible."
            )
    
    def get_question(self):
        xpath_editquestion = f"//a[starts-with(@href, 'https://{domain}/question/bank/editquestion/question.php?returnurl=')]"
        filtered_editquestion_links = self.driver.find_elements(By.XPATH, xpath_editquestion)
        preview_links = [
            link.get_attribute("href") for link in filtered_editquestion_links
        ]
    
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".instancemaxmark"))
        )
    
        max_marks_elements = self.driver.find_elements(By.CSS_SELECTOR, ".instancemaxmark")
        max_marks = [elem.text.strip() for elem in max_marks_elements]
    
        self.driver.get(data["student_answers"][0]["review_link"])
        # print(driver.current_url)
        try:
            questions = WebDriverWait(self.driver, 30).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".que.coderunner"))
            )
            list_questions = []
    
            print(len(questions), len(preview_links))
            for q_idx, question in enumerate(questions):
                print(self.driver.current_url)
                question_text = " ".join(
                    safe_find_element(self.driver, By.CSS_SELECTOR, "div.content div.formulation").text.split()
                )
    
                coderunner_examples_div = WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div.coderunner-examples")
                    )
                )
                expected_output_table = safe_find_element(self.driver, By.CSS_SELECTOR, "table.coderunnerexamples")
    
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
    
                template_content = "No link found for this question"
                test_cases = []
                if q_idx < len(preview_links):
                    print(preview_links[q_idx])
                    crawler.driver.get(preview_links[q_idx])
                    try:
                        WebDriverWait(self.driver, 30).until(
                            EC.presence_of_element_located((By.ID, "id_template"))
                        )
                        template_content = self.driver.find_element(
                            By.ID, "id_template"
                        ).get_attribute("value")
                        test_cases = get_test_cases(self.driver)
                    except TimeoutException:
                        print("Failed to load template or test cases due to timeout.")
                    finally:
                        # driver.get(data["student_answers"][0]["review_link"])
                        self.driver.get(data["student_answers"][0]["review_link"])
    
                list_questions.append(
                    {
                        "question": question_text,
                        "expected_outputs": expected_outputs,
                        "max_scores": max_marks[q_idx],
                        "template": template_content,
                        "test_cases": test_cases,
                    }
                )
    
            # print(list_questions)
            return list_questions
        except TimeoutException:
            print("Timeout.")

    def get_student_answers(self, data):
        for record in data["student_answers"]:
            self.driver.get(record["review_link"])
            attempt_data = []

            history_headers = self.driver.find_elements(
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

                        result_entry = {
                            "step": step,
                            "time": time,
                            "action": action,
                            "state": state,
                            "marks": marks,
                        }

                        table_data["results"].append(result_entry)

                attempt_data.append(table_data)

            record["response_history"] = attempt_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course_name", help="Class Name", type=str, default="DSA-HK231")
    parser.add_argument("--class_name", help="Class Name", type=str, default="L09")
    args = parser.parse_args()
    
    # Chrome setup
    chromedriver_path = "/afs/cs.stanford.edu/u/nqduc/chromedriver-linux64/chromedriver"
    chrome_binary_path = "/afs/cs.stanford.edu/u/nqduc/chrome-linux64/chrome"
    crawler = CrawlData(args.course_name, args.class_name, chromedriver_path, chrome_binary_path)

    course_name = args.course_name
    class_name = args.class_name

    wandb.init(project="student-score-crawler")

    target_link = DATA_LINKS[course_name][class_name]
    parsed_url = urlparse(target_link)
    # Extract the domain
    domain = parsed_url.netloc
    # Login
    crawler.driver.get(
        "https://sso.hcmut.edu.vn/cas/login?service=https%3A%2F%2F"
        f"{domain}/login/index.php%3FauthCAS%3DCAS"
    )
    username = crawler.driver.find_element(By.ID, "username")
    password = crawler.driver.find_element(By.ID, "password")
    username.send_keys(LOGIN_USER)
    password.send_keys(LOGIN_PASSWD)
    password.send_keys(Keys.RETURN)
    print("Login successfully")

    # find student group
    parsed_url = urlparse(target_link)
    query_params = parse_qs(parsed_url.query)
    course_id = query_params.get("id", [None])[0]
    if course_id:
        student_list_url = f"https://{domain}/user/index.php?id={course_id}"
        crawler.driver.get(student_list_url)
        print("Access success!")
    else:
        print("No course ID found in the URL.")

    # Start extraction from the first page
    student_details = crawler.extract_data()

    # Function to find and click pagination links
    crawler.navigate_pagination()

    xpath_expression = (
        f"//a[starts-with(@href, 'https://{domain}/mod/quiz/view.php?id=')]"
    )
    filtered_links = crawler.driver.find_elements(By.XPATH, xpath_expression)
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
        crawler.driver.get(quiz_link)

        data = {
            "lab_name": crawler.driver.title.split(":")[0].strip(),
            "list_questions": [],
            "student_answers": [],
        }

        student_rows = crawler.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

        for index, row in enumerate(student_rows):
            if row.find_elements(By.CSS_SELECTOR, "td.cell.c2 a"):
                try:
                    student_link = f"mod-quiz-report-overview-report_r{index}_c2"
                    link = WebDriverWait(crawler.driver, 30).until(
                        EC.presence_of_element_located((By.ID, student_link))
                    )

                    student_name = None
                    review_link = None
                    student_id = row.find_element(
                        By.CSS_SELECTOR, "td.cell.c3"
                    ).text.strip()
                    print(f"Student ID: {student_id}")
                    if filter_class_group(student_details, student_id, class_name):
                        student_name = link.find_element(
                            By.TAG_NAME, "a"
                        ).text.strip()
                        review_link = row.find_element(
                            By.CSS_SELECTOR, "a.reviewlink"
                        ).get_attribute("href")

                    if student_name and review_link:
                        print(f"Student Name: {student_name}, {review_link}")
                        data["student_answers"].append(
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
            
            else:
                print("Skipping empty row.")

        if not data["student_answers"]:
            print("There are no students in this quiz link.")
            continue

        parsed_url = urlparse(quiz_link)
        query_params = parse_qs(parsed_url.query)

        # Extract id
        course_id = query_params.get("id", [None])[0]
        if course_id is None:
            raise ValueError("ID parameter not found in URL")

        new_url = f"https://{domain}/mod/quiz/edit.php?cmid={course_id}"
        crawler.driver.get(new_url)

        data["list_questions"] = crawler.get_question()

        crawler.get_student_answers(data)

        all_data.append(data)

        first_part = crawler.driver.title.split(":")[0]
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
    crawler.driver.quit()
