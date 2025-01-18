import pandas as pd
import numpy as np
from Levenshtein import distance

# Read data files
df_q = pd.read_csv("question_infos.csv")
df = pd.read_csv("main_data.csv", low_memory=False)
course_info = pd.read_csv("course_infos.csv")
section_info = pd.read_csv("section_infos.csv")

# Print total number of unique questions
num_questions = df_q['question_id'].nunique()
print(f"Total number of unique questions: {num_questions}")

# Print dataframe info
print(df.columns)
print(df.shape)

# Convert pass column to string type
df['pass'] = df['pass'].astype('string')

def calculate_ratio(value):
    if pd.isna(value):
        return np.nan
    elif value == '0.0':
        return 0.0
    elif value == '' or value.split('.')[0] == '':
        return np.nan
    else:
        clean_string = value.split('.')[0]
        denominator = len(clean_string)
        numerator = clean_string.count('1')
        return numerator / denominator

# Calculate pass ratio
df['pass_ratio'] = df['pass'].apply(calculate_ratio)

def calculate_edit_distance(group):
    responses = group['response'].tolist()
    distances = [None]  # First response has no previous one to compare
    for i in range(1, len(responses)):
        distances.append(distance(str(responses[i-1]), str(responses[i])))
    group['edit_distance'] = distances
    return group

# Calculate edit distances
df_with_distances = df.groupby(['student_id', 'question_unittest_id']).apply(calculate_edit_distance)

# Define section categories
cc_sections = [1, 2, 3, 4, 5, 6, 7, 8, 28, 29, 30, 31, 32]
l_sections = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 34, 35, 36, 37, 38, 39, 40, 41]

# Create section categories
df_with_distances['section'] = np.select(
    [
        df_with_distances['section_id'].isin(cc_sections),
        df_with_distances['section_id'].isin(l_sections)
    ],
    ['cc', 'l'],
    default='other'
)

# Remove 'other' sections
df_with_distances = df_with_distances[df_with_distances['section'] != 'other']

# Create nested dataframe
nested_df = df_with_distances.groupby(['student_id', 'question_unittest_id']).agg({
    'attempt_id': 'count',
    'edit_distance': lambda x: x.dropna().mean(),
    'pass_ratio': 'max',
    'course_id': 'first',
    'section': 'first',
    'is_exam': 'first'
}).reset_index()

# Rename columns
nested_df = nested_df.rename(columns={
    'attempt_id': 'num_attempts',
    'edit_distance': 'avg_edit_distance',
    'pass_ratio': 'max_pass_ratio'
})

# Export to CSV
nested_df.to_csv('nested_analysis.csv', index=False)