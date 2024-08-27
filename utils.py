import json

def parse_score(name):
    return float(name.split('\n')[1][1:])

def find_global_max(dirpath):
    global_max = 0
    for data_file in os.listdir(dirpath):
        if data_file == ".ipynb_checkpoints":
            continue
    
        data_file = os.path.join(dirpath, data_file)
        with open(data_file, "r") as f:
            data = json.load(f)
    
        ids = []
        for answers in data['student_answers']:
            ids.append(answers['id'])

        for idx in range(len(data['list_questions'])):
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
    
            max_attempts = [len(student_marks) for student_marks in records]
            global_max = max(global_max, max(max_attempts))

    return global_max
