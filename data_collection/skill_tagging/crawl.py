import json
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from tqdm import tqdm

chromedriver_path = "/afs/cs.stanford.edu/u/nqduc/chromedriver-linux64/chromedriver"
chrome_binary_path = "/afs/cs.stanford.edu/u/nqduc/chrome-linux64/chrome"

URLs = {
    "math": "https://www.ixl.com/standards/common-core/math",
    "language_arts": "https://www.ixl.com/standards/common-core/ela",
    "science": "https://www.ixl.com/standards/common-core/science",
    "social_studies": "https://www.ixl.com/standards/common-core/social-studies",
}

chrome_options = Options()
chrome_options.binary_location = chrome_binary_path
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--ignore-ssl-errors=yes")
chrome_options.add_argument("--ignore-certificate-errors")


def crawl_elenent(parent):
    if parent is None:
        return []

    list_elements = parent.find_elements(By.CSS_SELECTOR, "li.each-alignment > a")

    element_data = []
    for element in list_elements:
        text = element.text
        open_p_idx = text.rfind("(")
        close_p_idx = text.rfind(")")
        element_data.append(
            {
                "id": text[open_p_idx + 1 : close_p_idx],
                "content": text[:open_p_idx].strip(),
            }
        )

    return element_data


def get_name_id(level, standard_content):
    standard_names_lv2 = []
    all_li_h4 = standard_content.find_elements(
        By.CSS_SELECTOR, f"{level} > li.each-category > h4"
    )
    return all_li_h4
    num_checked_li_h4 = 0
    n_lv2 = 1

    while num_checked_li_h4 < len(all_li_h4):
        next_lv2 = standard_content.find_elements(
            By.CSS_SELECTOR, f"li:nth-child({n_lv2}) > h4"
        )
        n_lv2 += 1
        if len(next_lv2) == 0:
            continue
        standard_names_lv2.append(next_lv2[0])
        num_checked_li_h4 += len(next_lv2)
    return standard_names_lv2


def algin_name_content(standard_names, standard_contents):
    if len(standard_names) == len(standard_contents):
        return standard_names, standard_contents

    if len(standard_names) > len(standard_contents):
        i = len(standard_contents) - 1
        list_new_contents = []

        for name in standard_names[::-1]:
            if i >= 0 and name.location["y"] < standard_contents[i].location["y"]:
                list_new_contents.append(standard_contents[i])
                i = i - 1
            else:
                list_new_contents.append(None)

        return standard_names, list_new_contents[::-1]

    if len(standard_names) < len(standard_contents):
        raise NotImplementedError("Not implemented yet")


def crawl_standards(driver):
    standard_tree = []
    standards = driver.find_element(By.CSS_SELECTOR, "#dv-listing-standards-alignment")
    standard_names = standards.find_elements(By.TAG_NAME, "h3")
    standard_contents = standards.find_elements(By.CSS_SELECTOR, ".listing-category")

    assert len(standard_names) == len(standard_contents)
    for standard_name, standard_content in tqdm(
        zip(standard_names, standard_contents), total=len(standard_names)
    ):
        if standard_content is not None:
            standard_names_lv2 = get_name_id(".listing-category", standard_content)
            standard_contents_lv2 = standard_content.find_elements(
                By.CSS_SELECTOR, "li > ul.listing-level2"
            )
            standard_names_lv2, standard_contents_lv2 = algin_name_content(
                standard_names_lv2, standard_contents_lv2
            )

        if standard_content is None or len(standard_names_lv2) == 0:
            element_data = crawl_elenent(standard_content)
            standard_tree.append(
                {"lv1_id": standard_name.text, "content": element_data}
            )
            continue

        lv2_content = []

        assert len(standard_names_lv2) == len(standard_contents_lv2)
        for standard_name_lv2, standard_content_lv2 in zip(
            standard_names_lv2, standard_contents_lv2
        ):
            if standard_content_lv2 is not None:
                standard_names_lv3 = get_name_id(
                    ".listing-level2", standard_content_lv2
                )
                standard_contents_lv3 = standard_content_lv2.find_elements(
                    By.CSS_SELECTOR, "li > ul.listing-level3"
                )
                standard_names_lv3, standard_contents_lv3 = algin_name_content(
                    standard_names_lv3, standard_contents_lv3
                )

            if standard_content_lv2 is None or len(standard_names_lv3) == 0:
                element_data = crawl_elenent(standard_content_lv2)
                lv2_content.append(
                    {"lv2_id": standard_name_lv2.text, "content": element_data}
                )
                continue

            lv3_content = []
            assert len(standard_names_lv3) == len(standard_contents_lv3)
            for standard_name_lv3, standard_content_lv3 in zip(
                standard_names_lv3, standard_contents_lv3
            ):
                if standard_content_lv3 is not None:
                    standard_names_lv4 = get_name_id(
                        ".listing-level3", standard_content_lv3
                    )
                    standard_contents_lv4 = standard_content_lv3.find_elements(
                        By.CSS_SELECTOR, "li > ul.listing-level4"
                    )
                    standard_names_lv4, standard_contents_lv4 = algin_name_content(
                        standard_names_lv4, standard_contents_lv4
                    )

                if standard_content_lv3 is None or len(standard_names_lv4) == 0:
                    element_data = crawl_elenent(standard_content_lv3)
                    lv3_content.append(
                        {"lv3_id": standard_name_lv3.text, "content": element_data}
                    )
                    continue

                lv4_content = []
                assert len(standard_names_lv4) == len(standard_contents_lv4)
                for standard_name_lv4, standard_content_lv4 in zip(
                    standard_names_lv4, standard_contents_lv4
                ):
                    if standard_content_lv4 is not None:
                        standard_names_lv5 = get_name_id(
                            ".listing-level4", standard_content_lv4
                        )
                        standard_contents_lv5 = standard_content_lv4.find_elements(
                            By.CSS_SELECTOR, "li > ul.listing-level5"
                        )
                        standard_names_lv5, standard_contents_lv5 = algin_name_content(
                            standard_names_lv5, standard_contents_lv5
                        )

                    if standard_content_lv4 is None or len(standard_names_lv5) == 0:
                        element_data = crawl_elenent(standard_content_lv4)
                        lv4_content.append(
                            {"lv4_id": standard_name_lv4.text, "content": element_data}
                        )
                        continue

                    lv5_content = []
                    assert len(standard_names_lv5) == len(standard_contents_lv5)
                    for standard_name_lv5, standard_content_lv5 in zip(
                        standard_names_lv5, standard_contents_lv5
                    ):
                        element_data = crawl_elenent(standard_content_lv5)
                        lv5_content.append(
                            {"lv5_id": standard_name_lv5.text, "content": element_data}
                        )

                    lv4_content.append(
                        {"lv4_id": standard_name_lv4.text, "content": lv5_content}
                    )

                lv3_content.append(
                    {"lv3_id": standard_name_lv3.text, "content": lv4_content}
                )

            lv2_content.append(
                {"lv2_id": standard_name_lv2.text, "content": lv3_content}
            )

        standard_tree.append({"lv1_id": standard_name.text, "content": lv2_content})

    return standard_tree


def run_crawl(driver, URL):
    driver.get(URL)
    grade_box = WebDriverWait(driver, 60).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#lstStdsLks"))
    )

    list_grades = grade_box[0].find_elements(By.TAG_NAME, "a")
    grade_urls = []
    grade_strs = []
    for grade in list_grades:
        grade_url = grade.get_attribute("href")
        grade_urls.append(grade_url)
        grade_strs.append(grade.text)

    all_data = {}
    for grade_url, grade_str in zip(grade_urls, grade_strs):
        print(f"Running crawl on {grade_url}")
        driver.get(grade_url)

        standard_trees = []
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".each-standards"))
        )
        list_standards = driver.find_elements(By.CSS_SELECTOR, ".each-standards")

        standard_name = (
            list_standards[0].find_element(By.CSS_SELECTOR, ".name-standards").text
        )
        standard_data = crawl_standards(driver)
        standard_trees.append({"standard": standard_name, "data": standard_data})

        standard_urls = []
        for standard in list_standards[1:]:
            standard_url = standard.find_element(By.CSS_SELECTOR, "a").get_attribute(
                "href"
            )
            standard_urls.append(standard_url)

        for standard_url in standard_urls:
            driver.get(standard_url)
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#stateStdsDetails"))
            )
            standard = driver.find_element(By.CSS_SELECTOR, ".each-standards.selected")
            standard_name = standard.find_element(
                By.CSS_SELECTOR, ".name-standards"
            ).text
            standard_data = crawl_standards(driver)
            standard_trees.append({"standard": standard_name, "data": standard_data})

        all_data[grade_str] = standard_trees

    return all_data


if __name__ == "__main__":
    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    for name, URL in URLs.items():
        print(f"Running crawl on {URL}")
        subject_data = run_crawl(driver, URL)

        # Save data to file
        os.makedirs("data", exist_ok=True)
        with open(f"data/{name}.json", "w") as f:
            json.dump(subject_data, f, indent=4)

    driver.quit()
