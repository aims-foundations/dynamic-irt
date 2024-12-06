import time
from datetime import datetime

from Levenshtein import distance
from selenium import webdriver

from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

STARTING_TIME = datetime.strptime("1/9/23, 00:00:00", "%d/%m/%y, %H:%M:%S")


def parse_time(time_str):
    # Parsing the string into a datetime object
    parsed_datetime = datetime.strptime(time_str, "%d/%m/%y, %H:%M:%S") - STARTING_TIME
    # Convert to days
    return parsed_datetime.total_seconds() / 86400


def parse_score(header_text):
    search_result = re.search(r"/(\d+\.\d+)", header_text)
    if search_result:
        return float(search_result.group(1))
    else:
        return None


def compute_ed(original, list_str):
    return [distance(original, x) for x in list_str]


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


def filter_class_group(details, sid, class_name):
    if class_name.lower() == "all":
        return True

    for detail in details:
        if detail["Class Group"] == "No groups":
            if detail["ID"] == sid:
                return True

        if detail["ID"] == sid and (
            detail["Class Group"] == class_name or detail["Class Group"] == "All"
        ):
            return True

    return False
