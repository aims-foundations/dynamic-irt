import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import find_global_max

def plot_per_question(dirpath):
    for data_file in os.listdir(dirpath):
        if data_file == ".ipynb_checkpoints":
            continue
        
        data_file = os.path.join(dirpath, data_file)
        with open(data_file, "r") as f:
            data = json.load(f)

        parts = dirpath.split('/')

        ids = []
        for answers in data['student_answers']:
            ids.append(answers['id'])

        for idx in range(len(data['list_questions'])):
            plt.clf()
    
            max_score = data['list_questions'][idx]['max_scores']
            q_index = idx + 1

            records = []
            for answers in data['student_answers']:
                for answer in answers['response_history']:
                    marks = []
                    if answer['question'] == f"Question {q_index}":
                        mark_per_attempt = []
                        for score_idx in range(len(answer['results'])):
                            mark_per_attempt.append(answer['results'][score_idx]['marks'])

                        mark_per_attempt = ['0' if mark == '' else mark for mark in mark_per_attempt]

                    marks.append(mark_per_attempt)
                records.extend(marks)
            
            try:
                records = [[float(mark) * 10 / max_score for mark in student_marks] for student_marks in records]
            except:
                continue

            print("PLOT")

            max_attempts = max(len(student_marks) for student_marks in records)
            
            # Generate x values (attempts)
            x = list(range(1, max_attempts + 1))
    
            # Find the average score for each number of attemps
            # Pad the marks of each student with highest marks
            padded_records = []
            for student_marks in records:
                if student_marks:
                    padded_records.append(student_marks + [max(student_marks)] * (max_attempts - len(student_marks)))
                else:
                    padded_records.append([0] * max_attempts)
            # Convert the list of lists to a NumPy array
            padded_records = np.array(padded_records)
            # Calculate the average marks for each attempt number across all students
            average_marks = np.nanmean(padded_records, axis=0)
    
            # Plot each student's attempts
            for i, student_marks in enumerate(padded_records):
                plt.plot(x, student_marks, label=f"{ids[i]}", color='blue', alpha=0.3)
    
            # Plot the average marks as a bold line
            plt.plot(x, average_marks, label='Average Marks', linewidth=3, color='red')

            # Add labels and legend
            plt.xlabel('Attempts')
            plt.ylabel('Marks')
            plt.title(f"{data['lab_name']} - Q{q_index}")
            plt.savefig(f"plots/{parts[1]}/{parts[2]}/{data['lab_name']}-Q{q_index}.png")


def plot_all_questions(dirpath):
    plt.clf()

    all_padded_records = []
    x = None
    global_max = find_global_max(dirpath)
    
    for data_file in os.listdir(dirpath):
        if data_file == ".ipynb_checkpoints":
            continue
        
        data_file = os.path.join(dirpath, data_file)
        with open(data_file, "r") as f:
            data = json.load(f)
    
        parts = dirpath.split('/')

        ids = []
        for answers in data['student_answers']:
            ids.append(answers['id'])

        for idx in range(len(data['list_questions'])):
            plt.clf()
    
            max_score = data['list_questions'][idx]['max_scores']
            q_index = idx + 1

            records = []
            for answers in data['student_answers']:
                for answer in answers['response_history']:
                    marks = []
                    if answer['question'] == f"Question {q_index}":
                        mark_per_attempt = []
                        for score_idx in range(len(answer['results'])):
                            mark_per_attempt.append(answer['results'][score_idx]['marks'])

                        mark_per_attempt = ['0' if mark == '' else mark for mark in mark_per_attempt]

                    marks.append(mark_per_attempt)
                records.extend(marks)

            try:
                records = [[float(mark) * 10 / max_score for mark in student_marks] for student_marks in records]
            except:
                continue
            print(f"PLOT {idx}")

            # Generate x values (attempts)
            x = list(range(1, global_max + 1))

            padded_records = []
            for student_marks in records:
                if student_marks:
                    padded_records.append(student_marks + [max(student_marks)] * (global_max - len(student_marks)))
                    all_padded_records.append(student_marks + [max(student_marks)] * (global_max - len(student_marks)))
                else:
                    padded_records.append([0] * global_max)
                    all_padded_records.append([0] * global_max)
                
            padded_records = np.array(padded_records)
            
            # Plot each student's attempts
            for i, student_marks in enumerate(padded_records):
                plt.plot(x, student_marks, label=f"{ids[i]}", color='blue', alpha=0.2)
        
    all_padded_records = np.array(all_padded_records)
    all_average_marks = np.nanmean(all_padded_records, axis=0)

    plt.plot(x, all_average_marks, label='All Average Marks', linewidth=3, color='Red')
    
    # Add labels and legend
    plt.xlabel('Attempts')
    plt.ylabel('Marks')
    plt.title(f'All questions')
    plt.savefig(f'plots/{parts[1]}/{parts[2]}/all-questions.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="DSA-HK231"
    )
    parser.add_argument("--class_name", help="Class Name", type=str, default="L01")
    args = parser.parse_args()
    course_name = args.course_name
    class_name = args.class_name
    os.makedirs("plots", exist_ok=True)
    os.makedirs(f"plots/{course_name}", exist_ok=True)
    os.makedirs(f"plots/{course_name}/{class_name}", exist_ok=True)

    wandb.init(project="student-score-crawler")

    plot_per_question(f"data/{course_name}/{class_name}")
    plot_all_questions(f"data/{course_name}/{class_name}")

    wandb.finish()


if __name__ == "__main__":
    main()
