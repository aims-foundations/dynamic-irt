# Data Analysis

Exploratory data analysis and behavioral analysis of the CodeInsights dataset.

## Overview

This directory contains scripts for analyzing student coding behavior, submission patterns, and problem characteristics from the CodeInsights dataset. The analyses provide insights into:

- **Student behavior patterns** - Clustering students by coding strategies
- **Temporal dynamics** - Submission pacing and iteration patterns
- **Problem characteristics** - Difficulty analysis and attempt patterns
- **Learning trajectories** - Pass rates and skill development over time

## Quick Start

### Reproduce All Results

To regenerate all figures and CSVs:

```bash
bash reproduce.sh
```

This runs all 4 analysis scripts and generates outputs in ~10-15 minutes (depending on system).

### Run Individual Analyses

```bash
# Exploratory data analysis (EDA)
python eda_codeinsight.py

# Student clustering analysis
python student_behavior_clustering.py

# Submission pacing analysis
python pace_analysis.py --all_courses

# Problem-level analysis
python problem_by_problem_analysis.py --all_courses
```

### Filter by Course

Analyze a specific course:

```bash
bash reproduce.sh --course dsa_hk231
```

Available courses: `dsa_hk231`, `dsa_hk232`, `dsa_hk222`, `pf_hk232`, `pf_hk222`

## Scripts

### 1. `eda_codeinsight.py` - Exploratory Data Analysis

Generates comprehensive dataset overview and visualizations.

**Outputs** (in `eda_outputs/`):
- `pass_rate_distribution.png` - Distribution of pass rates across questions
- `attempts_distribution.png` - Distribution of attempts per student
- `temporal_patterns.png` - Submission patterns by hour/day of week
- `question_difficulty.png` - Question difficulty rankings
- `learning_curves.png` - Pass rates over student progress deciles
- `course_comparison.png` - Comparison across different courses
- `code_length_distribution.png` - Distribution of code lengths

**Usage**:
```bash
python eda_codeinsight.py
```

### 2. `student_behavior_clustering.py` - Student Clustering

K-means clustering to identify distinct coding behavior patterns based on:
- Average time between submissions
- Average edit distance (Levenshtein) between submissions
- Total submission count

**Outputs** (in `clustering_outputs/`):
- `student_behavior_clusters_all_courses.png` - 2D projections of behavioral clusters ⭐ **Used in paper**
- `student_metrics_all_courses.csv` - Student-level metrics with cluster assignments
- `centroids_all_courses.csv` - Normalized cluster centers

**Usage**:
```bash
# All courses
python student_behavior_clustering.py

# Specific course
python student_behavior_clustering.py --course_name dsa_hk231

# Filter by question
python student_behavior_clustering.py --question_pattern "Binary_search"
```

**Key Findings**:
- **Cluster 1 (84.7%)**: Rapid iterators - many submissions, short intervals
- **Cluster 2 (13.9%)**: Deliberate planners - medium submissions, larger edits
- **Cluster 3 (1.3%)**: Careful thinkers - few submissions, long intervals
- **Cluster 4 (0.1%)**: Minimal engagement - very few submissions

### 3. `pace_analysis.py` - Submission Pacing

Analyzes temporal patterns in student submissions including pause times, submission volume, and performance correlations.

**Outputs** (in `pace_outputs/`):
- `aggregate_submission_patterns_all_courses.png` - 4-panel summary ⭐ **Used in paper**
- `log_time_delta_density_all_courses.png` - Distribution of pause times
- `repeated_top_submitters_all_courses.csv` - Academic integrity analysis
- `scatterplot_submits_pause_*.png` - Per-problem scatter plots

**Usage**:
```bash
# All courses
python pace_analysis.py --all_courses

# Specific course
python pace_analysis.py --course_name dsa_hk231
```

**Key Findings**:
- Median pause time: 78 seconds (1.3 minutes)
- Mean submissions per student: 610
- Positive correlation between submission count and final score
- Students exhibit diverse pacing strategies

### 4. `problem_by_problem_analysis.py` - Problem-Level Analysis

Examines problem characteristics and relationships between attempts, edit distances, and final scores.

**Outputs** (in `problem_outputs/`):
- `aggregate_problem_patterns_all_courses.png` - 4-panel problem summary ⭐ **Used in paper**
- `individual_questions_steps_and_scores_all_courses.csv` - Per-problem statistics
- `cor_step_edit_all_courses.csv` - Correlation analysis

**Usage**:
```bash
# All courses
python problem_by_problem_analysis.py --all_courses

# Specific course
python problem_by_problem_analysis.py --course_name dsa_hk231
```

**Key Findings**:
- Substantial heterogeneity in problem difficulty
- Negative correlation between average steps and final scores
- Difficult problems require both more attempts and larger code changes

## Output Directories

```
data_analysis/
├── eda_outputs/          # 7 EDA visualizations
├── clustering_outputs/   # Student clustering results (1 figure, 2 CSVs)
├── pace_outputs/         # Pacing analysis (~100+ per-problem plots + aggregates)
└── problem_outputs/      # Problem-level analysis (1 figure, 2 CSVs)
```

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install huggingface-hub python-Levenshtein tueplots
```

## Data Source

All scripts load data from HuggingFace:
- **Repository**: `stair-lab/code_insights_csv`
- **Files**: `main_data.csv`, `question_infos.csv`, `course_infos.csv`

Data is cached locally at:
```
~/.cache/huggingface/hub/datasets--stair-lab--code_insights_csv/
```

## Integration with Paper

Three key figures are used in the paper (Section 3: Dataset Collection):

1. **Figure: Student Clusters** → `clustering_outputs/student_behavior_clusters_all_courses.png`
2. **Figure: Submission Patterns** → `pace_outputs/aggregate_submission_patterns_all_courses.png`
3. **Figure: Problem Patterns** → `problem_outputs/aggregate_problem_patterns_all_courses.png`

To copy to overleaf:
```bash
cp clustering_outputs/student_behavior_clusters_all_courses.png ../../codeinsight-overleaf/figures/
cp pace_outputs/aggregate_submission_patterns_all_courses.png ../../codeinsight-overleaf/figures/
cp problem_outputs/aggregate_problem_patterns_all_courses.png ../../codeinsight-overleaf/figures/
```

These commands are automatically shown by `reproduce.sh` upon completion.

## Notes

- All visualizations use LaTeX fonts (tueplots + Computer Modern) for publication quality
- Figures are generated at 300 DPI for print quality
- Scripts use the same HuggingFace snapshot for consistency
- Clustering uses normalized features (MinMax scaling) before K-means
- Pace analysis filters outliers (>24 hours pause time) in some visualizations

## Troubleshooting

**Error: Module not found**
```bash
pip install -r ../requirements.txt
```

**Error: HuggingFace authentication**
```bash
export HF_TOKEN="your_token_here"
```

**Memory issues**
- Run analyses per-course using `--course_name` flag
- Close other applications to free RAM
- Consider running on a machine with 16GB+ RAM for all-courses analysis
