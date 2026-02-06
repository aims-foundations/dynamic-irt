import argparse
import json
import os

import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download
from openai import OpenAI
from together import Together
from tqdm import tqdm
from utils import (
    GRADE_MAP,
    LIST_SUBJETCS,
    PROMPT_ELEMENT,
    PROMPT_GRADE,
    PROMPT_LV,
    PROMPT_SUBJECT,
    REVERSE_GRADE_MAP,
)


def generate(client, model, messages, **kwargs):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    
    return completion.choices


def get_subjects(question, **kwargs):
    messages = [{"role": "user", "content": PROMPT_SUBJECT.format(question=question)}]
    list_subjects = []
    is_success = False
    for _ in range(5):
        try:
            completions = generate(messages=messages, **kwargs)
            list_subjects = eval(completions[0].message.content)
            if len(list_subjects) > 0 and all(
                subject in LIST_SUBJETCS for subject in list_subjects
            ):
                is_success = True
                break
            else:
                raise Exception("Invalid subjects")
        except Exception as e:
            continue

    if not is_success:
        try:
            list_subjects = [x for x in list_subjects if x in LIST_SUBJETCS]
        except Exception as e:
            list_subjects = []

    return list_subjects


def get_grades(question, subject_skills, **kwargs):
    list_possible_grades = list(subject_skills.keys())
    list_possible_grades = [GRADE_MAP[grade] for grade in list_possible_grades]
    grade_desc = ", ".join(list_possible_grades)
    messages = [
        {
            "role": "user",
            "content": PROMPT_GRADE.format(question=question, grade_desc=grade_desc),
        }
    ]
    list_grades = []
    is_success = False
    for _ in range(5):
        try:
            completions = generate(messages=messages, **kwargs)
            list_grades = eval(completions[0].message.content)
            if len(list_grades) > 0 and all(
                grade in list_possible_grades for grade in list_grades
            ):
                is_success = True
                break
            else:
                raise Exception("Invalid grades")
        except Exception as e:
            continue

    if not is_success:
        try:
            list_grades = [x for x in list_grades if x in list_possible_grades]
        except Exception as e:
            list_grades = []

    return list_grades


def get_each_lv(question, skills, prompt, **kwargs):
    standards = [
        "{idx}. {standard}".format(idx=idx + 1, standard=standard)
        for idx, standard in enumerate(skills)
    ]
    standard_desc = "\n".join(standards)
    messages = [
        {
            "role": "user",
            "content": prompt.format(question=question, desc=standard_desc),
        }
    ]

    is_success = False
    for _ in range(5):
        try:
            completions = generate(messages=messages, **kwargs)
            list_standards = eval(completions[0].message.content)

            list_standards = [
                int(standard.split(".")[0]) for standard in list_standards
            ]
            if len(list_standards) > 0 and all(
                standard in range(1, len(standards) + 1) for standard in list_standards
            ):
                is_success = True
                break
            else:
                raise Exception("Invalid levels")
        except Exception as e:
            continue

    if not is_success:
        try:
            list_standards = [
                x for x in list_standards if x in range(1, len(standards) + 1)
            ]
        except Exception as e:
            list_standards = []

    return list_standards


def get_lvs_recursive(question, skills, lv_id=1, **kwargs):
    list_lv_skills = [s[f"lv{lv_id}_id"] for s in skills]
    list_lv_skills = get_each_lv(question, list_lv_skills, PROMPT_LV, **kwargs)
    outputs = []

    print(list_lv_skills)
    for lv_idx in list_lv_skills:
        lv = skills[lv_idx - 1][f"lv{lv_id}_id"]
        elements = skills[lv_idx - 1]["content"]

        if len(elements) > 0:
            if "id" in elements[0]:
                # Last level
                skill_elements = [element["content"] for element in elements]
                element_idxs = get_each_lv(
                    question, skill_elements, PROMPT_ELEMENT, **kwargs
                )
                output_elements = []
                for element_idx in element_idxs:
                    output_elements.append(elements[element_idx - 1])
                elements = output_elements
            else:
                elements = get_lvs_recursive(
                    question, elements, lv_id=lv_id + 1, **kwargs
                )

            outputs.append(
                {
                    "level": lv,
                    "skills": elements,
                }
            )
        else:
            outputs.append(
                {
                    "level": lv,
                    "skills": [],
                }
            )

    return outputs


def get_standard_lvs(question, grade_skills, **kwargs):
    # Standard
    standards = [standard["standard"] for standard in grade_skills]
    list_standards = get_each_lv(question, standards, PROMPT_LV, **kwargs)
    print(list_standards)

    outputs = []
    for standard_idx in list_standards:
        standard = grade_skills[standard_idx - 1]["standard"]
        standard_skills = grade_skills[standard_idx - 1]["data"]

        standard_skills = get_lvs_recursive(question, standard_skills, **kwargs)
        outputs.append(
            {
                "standard": standard,
                "skills": standard_skills,
            }
        )

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_url", type=str, default="http://hyperturing1:8080/v1")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct"
    )
    args = parser.parse_args()
    num_gpus = torch.cuda.device_count()
    upload_api = HfApi()
    os.makedirs("results", exist_ok=True)

    if args.model_url == "together":
        client = Together()
    else:
        client = OpenAI(
            base_url=args.model_url,
            api_key="token-abc123",
        )

    data_folder = snapshot_download(
        repo_id="stair-lab/reeval_matrices", repo_type="dataset"
    )

    skills_by_subject = {
        "Mathematics": json.load(open("data/math.json")),
        "Language & Arts": json.load(open("data/language_arts.json")),
        "Science": json.load(open("data/science.json")),
        "Social Studies": json.load(open("data/social_studies.json")),
    }

    dataset_name = args.dataset
    print(f"Processing {dataset_name}")
    ds_short_name = dataset_name.replace("/", "_")
    dataset = pd.read_csv(f"{data_folder}/{dataset_name}/question_keys.csv")
    question_infos = []

    if os.path.exists(f"results/{ds_short_name}_skills.json"):
        with open(f"results/{ds_short_name}_skills.json", "r") as f:
            question_infos = json.load(f)

        start_idx = len(question_infos)
    else:
        start_idx = 0

    for qi, question in enumerate(tqdm(dataset["raw_question"][start_idx:])):
        print("Getting subjects...")
        subjects = get_subjects(
            question,
            model=args.model,
            client=client,
        )
        print(subjects)

        if len(subjects) == 0:
            question_infos.append(
                {
                    "instance_id": int(dataset["instance_id"][qi]),
                    "prompt": question,
                    "skills": [],
                }
            )

        for subject in subjects:
            print(f"Getting grades for {subject}...")
            grades = get_grades(
                question,
                subject_skills=skills_by_subject[subject],
                model=args.model,
                client=client,
            )
            if subject == "Mathematics" and (
                "10" in grades or "11" in grades or "12" in grades
            ):
                if "10" in grades:
                    grades.remove("10")
                if "11" in grades:
                    grades.remove("11")
                if "12" in grades:
                    grades.remove("12")
                grades.append("10, 11, and 12")

            if len(grades) == 0:
                question_infos.append(
                    {
                        "instance_id": int(dataset["instance_id"][qi]),
                        "prompt": question,
                        "skills": [
                            {
                                "subject": subject,
                                "skills": [],
                            }
                        ],
                    }
                )

            list_skills = []
            for grade in grades:
                print("Getting lvs...")
                mapped_grade = REVERSE_GRADE_MAP[grade]
                skills = get_standard_lvs(
                    question,
                    grade_skills=skills_by_subject[subject][mapped_grade],
                    model=args.model,
                    client=client,
                )
                list_skills.append(
                    {
                        "grade": grade,
                        "skills": skills,
                    }
                )

            question_infos.append(
                {
                    "instance_id": int(dataset["instance_id"][qi]),
                    "prompt": question,
                    "skills": [
                        {
                            "subject": subject,
                            "skills": list_skills,
                        }
                    ],
                }
            )

        if qi % 10 == 0:
            with open(f"results/{ds_short_name}_skills.json", "w") as f:
                json.dump(question_infos, f, indent=4)
