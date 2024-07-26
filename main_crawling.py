"""
Run this file to crawl students' scores.
"""
import argparse
import json
import os
from urllib.parse import urlparse
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from configs import LOGIN_USER, LOGIN_PASSWD, DATA_LINKS
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

    course_name = args.course_name
    class_name = args.class_name
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
    xpath_expression = f"//a[starts-with(@href, 'https://{domain}/mod/quiz/view.php?id=')]"
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, xpath_expression)))
        filtered_links = driver.find_elements(By.XPATH, xpath_expression)
    except NoSuchElementException as e:
        print("Element not found:", e)
    except TimeoutException as e:
        print("Request timed out:", e)
    QUIZZES_RESULT_LINKS = [
        link.get_attribute('href').replace("view.php", "report.php") + "&mode=overview"
        for link in filtered_links
    ]

    # Create folders
    os.makedirs(f"data/{course_name}/{class_name}", exist_ok=True)

    for quiz_link in tqdm(QUIZZES_RESULT_LINKS, desc="Crawling"):
        print(quiz_link)
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
        try:
            table = driver.find_elements(By.CSS_SELECTOR, "table.generaltable")
            headers = table[0].find_elements(By.TAG_NAME, "th")
        except NoSuchElementException as e:
            continue
        except TimeoutException as e:
            continue

        students = table[0].find_elements(By.TAG_NAME, "tr")
        for student in students[1:-2]:
            try:
                cells = student.find_elements(By.TAG_NAME, "td")
                review_link = (
                    cells[2]
                    .find_elements(By.CSS_SELECTOR, "a.reviewlink")[0]
                    .get_attribute("href")
                )
                student_id = cells[3].text  # Renamed 'id' to 'student_id'
                records.append((student_id, review_link))
            except NoSuchElementException as e:
                continue
            except TimeoutException as e:
                continue
        column_names = [header.text for header in headers[9:]]
        max_scores = [parse_score(column_name) for column_name in column_names]
        student_attemps = []
        N_STUDENTS = len(records)
        i = 1
        for student_id, review_link in records:
            print(f"\r{i}/{N_STUDENTS}", end="")
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

            student_attemps.append({"student_id": student_id, "records": record_on_questions})
            i += 1

        data = {"max_scores": max_scores, "attemps": student_attemps}
        check(data)
        with open(
            f"data/{course_name}/{class_name}/{driver.title.replace(' ', '_').split(':')[0]}.json",
            "w",
            encoding='utf-8') as json_file:
            json.dump(data, json_file)
