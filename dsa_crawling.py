import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
import time
import argparse

def login(driver, login_url, userid, pwd):
    driver.get(login_url)
    username = driver.find_element(By.ID, 'username')
    password = driver.find_element(By.ID, 'password')
    username.send_keys(userid)  # '010344'
    password.send_keys(pwd)  # '010344'
    password.send_keys(Keys.RETURN)
    
def navigate_link(driver, tested_url):
    try:
        link = link = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, f"//a[@href='{tested_url}']")))
        link.click()
        print("Link exists in the DOM and is clickable.")
    except NoSuchElementException:
        try:
            # Attempt to find the link even if not clickable
            link = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, f"//a[@href='{tested_url}']")))
            driver.execute_script("arguments[0].click();", link)
            print("Link was not initially clickable but does exist in the DOM.")
        except NoSuchElementException:
            print("Link does not exist in the DOM.")

def parse_score(name):
    return float(name.split('\n')[1][1:])

def check(data):
    data = data["attempts"]
    print(f"Checking: n={len(data)}, sample: {data[0]}")
    
def save_json(data_file, data):
    data_dir = './data'
    if os.path.exists(data_dir) is False:
        os.mkdir(data_dir)
    
    with open(f'{data_dir}/{data_file}', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main(args):
    login_url = 'https://sso.hcmut.edu.vn/cas/login?service=https%3A%2F%2Fe-learning.hcmut.edu.vn%2Flogin%2Findex.php%3FauthCAS%3DCAS'
    
    course_links = ["https://e-learning.hcmut.edu.vn/my/courses.php", "https://e-learning.hcmut.edu.vn/course/view.php?id=108885", 
                    "https://e-learning.hcmut.edu.vn/mod/quiz/view.php?id=188779", "https://e-learning.hcmut.edu.vn/mod/quiz/report.php?id=188779&mode=overview"]
    
    # chrome driver setup
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Runs Chrome in headless mode.
    chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
    chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 20)  # wait for 20s
    
    login(driver, login_url, '010344', '010344')
    
    driver.get("https://e-learning.hcmut.edu.vn")
    
    for course_link in course_links:
        navigate_link(driver, course_link)
        
    wait.until(EC.visibility_of_element_located((By.ID, 'region-main')))
    lab_name = driver.title.split(":")[0].strip()

    data = {
        'lab_name': lab_name,
        'list_questions': [],
        'student_answers': []
    }

    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'table')))
    student_rows = driver.find_elements(By.CSS_SELECTOR, 'table tbody tr')

    for index, row in enumerate(student_rows):
        if 'emptyrow' in row.get_attribute('class'):
                print("Skipping empty row.")
                continue  # Skip this row and move to the next one

        try:
            student_link = f'mod-quiz-report-overview-report_r{index}_c2'
            link = wait.until(EC.presence_of_element_located((By.ID, student_link)))

            try:
                student_name = link.find_element(By.TAG_NAME, 'a').text.strip()
                review_link = link.find_element(By.CSS_SELECTOR, 'a.reviewlink').get_attribute('href')
                student_id = row.find_element(By.CSS_SELECTOR, 'td.cell.c3').text.strip()
                print(student_name, review_link, student_id)

                data['student_answers'].append({
                    'name': student_name,
                    'id': student_id,
                    'review_link': review_link
                })
            except NoSuchElementException:
                print(f"Missing expected elements within the link ID {student_link}")
                continue
        
        except TimeoutException:
            print(f"Element with ID {student_link} did not appear in time.")
            continue
    
    # extract code and questions from each student's revision
    driver.get(data['student_answers'][0]['review_link'])
    questions = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.que.coderunner')))
    list_questions = []

    for question in questions:
        question_text = " ".join(question.find_element(By.CSS_SELECTOR, 'div.content div.formulation').text.split())
        coderunner_examples_div = question.find_element(By.CSS_SELECTOR, 'div.coderunner-examples')
        expected_output_table = coderunner_examples_div.find_element(By.CSS_SELECTOR, 'table.coderunnerexamples')

        rows = expected_output_table.find_elements(By.CSS_SELECTOR, 'tbody tr')
        expected_outputs = []
        for row in rows:
            test_cell = row.find_element(By.CSS_SELECTOR, 'td.cell.c0 pre.tablecell').text
            result_cell = row.find_element(By.CSS_SELECTOR, 'td.cell.c1 pre.tablecell').text
            expected_outputs.append({'test': test_cell, 'result': result_cell})

        list_questions.append({
            'question': question_text,
            'expected_outputs': expected_outputs
        })
    data['list_questions'] = list_questions
    
    # extract other metrics
    for record in data['student_answers']:
        driver.get(record['review_link'])
        attempt_data = []

        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'responsehistoryheader')))
        history_headers = driver.find_elements(By.XPATH, "//h4[contains(text(), 'Response history')]")

        for index, header in enumerate(history_headers):
            table = header.find_element(By.XPATH, "following-sibling::div//table[@class='generaltable']")
            rows = table.find_elements(By.CSS_SELECTOR, 'tbody tr')
            table_data = {
                'question': f'Question {index+1}',
                'results': []
            }
            for row in rows:
                cells = row.find_elements(By.CSS_SELECTOR, 'td')
                if len(cells) >= 5:
                    step = cells[0].text
                    time = cells[1].text
                    action = cells[2].text
                    state = cells[3].text
                    marks = cells[4].text

                    table_data['results'].append({
                        'step': step,
                        'time': time,
                        'action': action,
                        'state': state,
                        'marks': marks
                    })

            attempt_data.append(table_data)

        record['response_history'] = attempt_data
        
    save_json("dt03_oop.json", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--plot_task", action="all_questions", help="Plot as required")
    
    args = parser.parse_args()

    main(args)