"""
This is the main process for crawling.
"""

import os
import json
import argparse
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from configs import LOGIN_URL, LOGIN_USER, LOGIN_PASSWD, DATA_LINKS
from utils import parse_score, check

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

    # Login
    driver.get(LOGIN_URL)
    username = driver.find_element(By.ID, "username")
    password = driver.find_element(By.ID, "password")
    username.send_keys(LOGIN_USER)
    password.send_keys(LOGIN_PASSWD)
    password.send_keys(Keys.RETURN)

    # Get course homepage
    course_name = args.course_name
    class_name = args.class_name
    driver.get(DATA_LINKS[course_name][class_name])

    # Filter quizzes only
    links = driver.find_elements(By.CSS_SELECTOR, "a.courseindex-link")
    link_urls = [link.get_attribute("href") for link in links]
    filtered_links = [
        link
        for link in link_urls
        if link.startswith("https://e-learning.hcmut.edu.vn/mod/quiz/view.php?id=")
    ]
    QUIZZES_RESULT_LINKS = [
        filtered_link.replace("view.php", "report.php") + "&mode=overview"
        for filtered_link in filtered_links
    ]

    # Create folders
    os.makedirs(f"data/{course_name}/{class_name}", exist_ok=True)

    for quiz_link in tqdm(QUIZZES_RESULT_LINKS[21:], desc="Crawling"):
        driver.get(quiz_link)

        input_field = driver.find_elements(By.ID, "id_pagesize")

        # Clear the current value
        input_field[0].clear()

        # Set the new value to '100'
        input_field[0].send_keys("100")

        # Simulate pressing the Enter key
        input_field[0].send_keys(Keys.ENTER)

        if "Backup" in driver.title or "Bù" in driver.title or "SEB" in driver.title:
            continue

        records = []

        table = driver.find_elements(By.CSS_SELECTOR, "table.generaltable")
        headers = table[0].find_elements(By.TAG_NAME, "th")

        students = table[0].find_elements(By.TAG_NAME, "tr")

        for student in students[1:-2]:
            try:
                cells = student.find_elements(By.TAG_NAME, "td")
                review_link = (
                    cells[2]
                    .find_elements(By.CSS_SELECTOR, "a.reviewlink")[0]
                    .get_attribute("href")
                )
                records.append((cells[3].text, review_link))
            except RuntimeError:
                continue

        column_names = [header.text for header in headers[9:]]
        max_scores = [parse_score(column_name) for column_name in column_names]

        student_attemps = []

        i = 1

        for rid, review_link in records:
            print(f"\r{i}/{len(records)}", end="")
            driver.get(review_link)

            wait.until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, ".que .history table")
                )
            )

            tables = driver.find_elements(By.CSS_SELECTOR, ".que .history table")

            record_on_questions = []

            for table in tables:
                record_on_question = []
                rows = table.find_elements(By.TAG_NAME, "tr")
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 4 and cells[2].text.startswith("Submit"):
                        record_on_question.append(cells[4].text)
                record_on_questions.append(record_on_question)

            student_attemps.append({"id": rid, "records": record_on_questions})
            i += 1

        data = {"max_scores": max_scores, "attemps": student_attemps}

        check(data)

        with open(
            f"data/{course_name}/{class_name}/{driver.title.replace(' ', '_').split(':')[0]}.json",
            "w",
            encoding="utf8",
        ) as json_file:
            json.dump(data, json_file)
