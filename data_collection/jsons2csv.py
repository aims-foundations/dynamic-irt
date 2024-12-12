import json
import os
from argparse import ArgumentParser

import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--course_name", help="Class Name", type=str, default="dsa_hk231"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download and load data
    data_folder = snapshot_download(
        repo_id=f"stair-lab/code_insights_jsons", repo_type="dataset"
    )
    data_folder = os.path.join(data_folder, args.course_name)
    directory_json_files = {}
    
    for folder in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, folder)):
            list_jsons = os.listdir(os.path.join(data_folder, folder))
            list_jsons = [x for x in list_jsons if x.endswith(".json")]
            directory_json_files[folder] = list_jsons

    # Load data
    data_dict = {
        "student_id": [],
        "course_id": [],
        "section_id": [],
        "question_id": [],
        "question_info": [],
        "attempt_id": [],
        "response": [],
        "score": [],
        "testcase_scores": [],
    }
    
    directory_json_files = dict(sorted(directory_json_files.items()))
    
    for section_id, json_files in tqdm(directory_json_files.items()):
        for json_file in json_files:
            print(f"Processing {section_id}/{json_file}")
            with open(os.path.join(data_folder, section_id, json_file), "r") as f:
                data = json.load(f)
            
            data_q = data["list_questions"]
            data_s = data["student_answers"]

            question_content_map = {}
            for idx, q in enumerate(data_q):
                if isinstance(q, list):
                    for sub_idx, sq in enumerate(q):
                        question_content_map[f"Question {idx + 1}.{sub_idx + 1}"] = (
                            sq["name"],
                            sq["max_score"],
                            sq["question"],
                            sq["template"],
                            sq["testcases"]
                        )
                else:
                    question_content_map[f"Question {idx + 1}"] = (
                        q["name"],
                        q["max_score"],
                        q["question"],
                        q["template"],
                        q["testcases"]
                    )
            
            for answer in data_s:
                student_id = answer["id"]
                
                for response_history in answer["response_history"]:
                    current_question_name, question_max_score, \
                    current_question_text, current_question_template, current_question_testcases = (
                        question_content_map.get(response_history["question"], ("", 0, "", "", []))
                    )
                    if not current_question_name or float(question_max_score) == 0:
                        continue
                    
                    question_info = current_question_text + "\n" + current_question_template
                    for tc in current_question_testcases:
                        question_info += f"\n\nInput: {tc['input']}\nSTD input: {tc['std_input']}\nOutput: {tc['output']}"

                    for t, result in enumerate(response_history["results"][1:-1]):
                        if result["score"] == "":
                            score = 0
                        elif float(result["score"]) > float(question_max_score):
                            print("Student score exceeds max score!")
                        else:
                            score = float(result["score"]) / float(question_max_score)

                        response = result["action"]
                        for prefix in ["Prechecked: ", "Saved: ", "Submit: "]:
                            if result["action"].startswith(prefix):
                                response = response.replace(prefix, "")

                        if "testcases" not in result:
                            result["testcases"] = []

                        
                        
                        data_dict["student_id"].append(student_id)
                        data_dict["course_id"].append(args.course_name)
                        data_dict["section_id"].append(section_id)
                        data_dict["question_id"].append(current_question_name)
                        data_dict["question_info"].append(question_info)
                        data_dict["attempt_id"].append(t+1)
                        data_dict["response"].append(response)
                        data_dict["score"].append(score)
                        data_dict["testcase_scores"].append(result["testcases"])
    
    df = pd.DataFrame(data_dict)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/{args.course_name}.csv", index=False)