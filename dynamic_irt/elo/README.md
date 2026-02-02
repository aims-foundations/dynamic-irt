# Elo-based IRT Analysis

This module contains Elo rating system implementations for analyzing learning dynamics
in the Edmentum (K-12 math/reading) and CodeInsights (university coding) datasets.

## Location

This module is part of the CodeInsights repository at `dynamic_irt/elo/`.

## Overview

This codebase provides tools for:
- Analyzing student performance data on online learning platforms (Edmentum and university programming course)
- Modeling difficulty of coding tasks using Elo rating systems
- Exploring correlations between problem features and estimated difficulty
- Generating descriptive statistics and visualizations of learning patterns
- Data pre-processing for the CodeInsights dataset

## Running the Analysis

From the CodeInsights root directory:

```bash
# Main Elo experiments
python -m dynamic_irt.elo.main_elo

# Difficulty analysis
python -m dynamic_irt.elo.difficulty_analysis

# Descriptive statistics (optional)
python -m dynamic_irt.elo.descriptive_stats
```

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy gdown
# For feature extraction (GPU required):
pip install torch vllm huggingface_hub
```

## Data Files

This framework uses data that will be automatically downloaded from Google Drive using the `gdown` library. The data IDs are already embedded in the code, so you don't need to download files manually.

## Main Analysis Files

1. **main_elo.py** - Implements and evaluates various Elo rating models:
   - Dynamic updates with forgetting mechanisms
   - Student ability tracking based on performance
   - Difficulty estimation for coding problems
   - Performance evaluation using AUC metrics with plots

2. **difficulty_analysis.py** - Analyzes relationships between:
   - Estimated difficulty and problem features
   - Test-level difficulty distributions
   - Correlations between difficulty, code length, and required steps

## Support Files

- **utils.py** - Contains all utility functions used by the main scripts:
  - Data loading helpers
  - Elo update functions (basic, time-based, attempt-based)
  - Evaluation metrics
  - Visualization tools

- **descriptive_stats.py** - Generates descriptive statistics and visualizations:
  - Student attempt counts
  - Score distributions
  - Item difficulty distributions

## Do Not Run

The following files are not recommended to run as they are mainly for transparency in data preprocessing:

- **data_preprocessing.py** - Data preprocessing script that has already been executed to generate the required datasets.
- **feature_extraction.py** - Uses LLM models to extract problem features and requires specific GPU configurations.

## Key Functions

### Elo Rating Functions

- `basic_update()` - Standard Elo updates for ability and difficulty
- `update_with_time()` - Elo updates accounting for time-based forgetting
- `run_update()` - Applies specified update function to the entire dataset
- `run_update_with_new_attempt()` - Handles updates considering previous attempts

### Evaluation Functions

- `calculate_auc()` - Calculates AUC on masked data for model evaluation
- `compute_difficulty()` - Estimates average difficulty for each item
- `plot_trajectory()` - Visualizes ability and difficulty updates for students

### Visualization Functions

- `plot_distribution()` - Creates histograms for various metrics
- `evaluate_edmentum()` / `evaluate_codeinsights()` - Run standard evaluation

## Workflow

1. Start by running `main_elo.py` to generate the main performance models
2. Run `difficulty_analysis.py` to analyze relationships between difficulty and problem features
3. Optionally run `descriptive_stats.py` if you need additional visualizations

## Model Descriptions

The framework implements several Elo-based models:

1. **Basic Elo** - Standard ability/difficulty updates based on binary correctness
2. **Time-based Elo** - Incorporates forgetting with time intervals between attempts
3. **Attempt-aware Elo** - Adjusts updates based on whether items are new or repeated

## Results Interpretation

- Higher AUC values indicate better predictive performance
- Ability trajectories show updates in Elo modeling
- Difficulty correlations reveal which problem features contribute most to challenge level

## Growth Model - R Analysis

The repository also includes an R Markdown file for running linear growth models on the Edmentum dataset:

- **growth_model.Rmd** - Implements hierarchical linear growth models using R:
  - Tracks student growth over time with mixed-effects models
  - Accounts for individual learning trajectories
  - Visualizes ability growth distributions
  - Evaluates model performance with ROC/AUC metrics

To run this analysis:

1. Make sure R and RStudio are installed
2. Install required R packages:
   ```R
   install.packages(c("dplyr", "ggplot2", "lme4", "pROC", "googledrive"))
   ```
3. Open the .Rmd file in RStudio and use "Knit" to generate the HTML report
4. When loading the Edmentum data with googledrive library, it asks you for credentials and to login to your Google account

This analysis complements the Python-based Elo models by providing a linear growth perspective on student learning trajectories.

## Notes

- Each experiment masks 20% of the data for evaluation
- Default learning rate (K) is 0.4
- The code automatically downloads required datasets from Google Drive
