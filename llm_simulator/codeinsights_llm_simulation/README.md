# CodeInsights LLM Simulation

A comprehensive evaluation framework for assessing Large Language Models (LLMs) on coding tasks through multiple scenarios, measuring functional correctness, behavioral alignment, and psychometric properties.

## Overview

This project evaluates LLMs across four distinct scenarios:
- **Scenario 1**: Basic coding problem solving
- **Scenario 2**: Student behavior simulation with profiling
- **Scenario 3**: Mistake pattern replication
- **Scenario 4**: Correct solution generation with efficiency alignment

## Features

- **Multi-Model Support**: Claude, GPT-4, Gemini, Mistral, and open-source models
- **Comprehensive Metrics**: Functional correctness, AST edit distance, CodeBERT similarity, psychometric correlations
- **Parallel Processing**: Efficient API calls with rate limiting
- **Code Similarity Analysis**: Advanced AST parsing and semantic similarity measurements
- **Psychometric Analysis**: IRT-based ability and difficulty parameter estimation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd codeinsights-llm-simulation

# Install required dependencies
pip install pandas numpy requests jinja2 anthropic openai google-generativeai mistralai
pip install transformers torch scikit-learn apted tree-sitter tree-sitter-cpp
pip install scipy matplotlib seaborn tueplots huggingface-hub
```

## Quick Start

### 1. Data Preprocessing

First, generate the analysis datasets from raw student submissions:

```bash
python data_preprocessing.py
```

This script:
- Downloads student coding submission data from HuggingFace
- Processes submissions to extract mistake-fix patterns
- Creates scenario-specific datasets
- Saves processed data for downstream analysis

**Output**: Creates `Scenario{1-4}_full_data.csv` files in the data directory.

### 2. LLM Evaluation

Run commercial LLMs on the generated scenarios:

```bash
# Set your API keys as environment variables
export ANTHROPIC_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
export GOOGLE_API_KEY="your_key_here"
export MISTRAL_API_KEY="your_key_here"

python run_commercial_model.py
```

This script:
- Processes all four scenarios with multiple LLMs
- Generates tailored prompts for each scenario
- Handles parallel API calls with rate limiting
- Saves model responses for analysis

**Output**: Creates model-specific CSV files in `scenario_results/` directory.

### 3. Metrics Computation

Calculate comprehensive evaluation metrics:

```bash
python compute_metrics.py
```

This script:
- Compiles and executes generated code against test cases
- Computes functional correctness scores
- Calculates behavioral alignment metrics (AST edit distance, CodeBERT similarity)
- Evaluates mistake pattern alignment and efficiency metrics

**Dependencies**: Uses `utils.py` and `cossim_calculator.py` for core functionality.

**Output**: Generates `all_results.json` with comprehensive metrics.

### 4. Psychometric Analysis

Generate correlation-based psychometric evaluations:

```bash
python psychometrics_metrics.py
```

This script:
- Applies Item Response Theory (IRT) modeling
- Estimates student ability and item difficulty parameters
- Computes correlations between LLM and human parameters
- Provides psychometric lens for model evaluation

**Output**: Creates correlation matrices in `correlations/all_correlations.json`.

## Project Structure

```
├── data_preprocessing.py      # Data preparation and scenario generation
├── run_commercial_model.py    # LLM API integration and prompt execution
├── compute_metrics.py         # Main metrics calculation pipeline
├── utils.py                   # Core utilities for code execution and metrics
├── cossim_calculator.py       # AST and semantic similarity computation
├── psychometrics_metrics.py   # IRT-based psychometric analysis
├── data/                      # Processed datasets
├── scenario_results/          # LLM outputs by model and scenario
├── ability/                   # Student ability parameter estimates
├── difficulty/                # Item difficulty parameter estimates
└── correlations/              # Psychometric correlation results
```

## Scenarios Explained

### Scenario 1: Basic Problem Solving
- **Goal**: Measure fundamental coding ability
- **Input**: Question description and template
- **Metric**: Functional correctness

### Scenario 2: Student Behavior Simulation
- **Goal**: Replicate individual student coding patterns
- **Input**: Student profile + example submissions + target problem
- **Metrics**: Functional correctness + AST similarity + CodeBERT similarity

### Scenario 3: Mistake Pattern Replication
- **Goal**: Generate realistic coding mistakes
- **Input**: Student profile + past mistakes + target problem
- **Metrics**: All Scenario 2 metrics + mistake pattern alignment

### Scenario 4: Efficiency-Aware Correct Solutions
- **Goal**: Generate correct solutions matching student efficiency
- **Input**: Student profile + correct examples + target problem
- **Metrics**: All previous metrics + runtime efficiency alignment

## Key Metrics

### Functional Metrics
- **Functional Correctness**: Percentage of test cases passed
- **Unit Test Alignment**: Agreement between LLM and student test outcomes

### Behavioral Metrics
- **AST Edit Distance**: Structural similarity between code trees
- **CodeBERT Similarity**: Semantic similarity via code embeddings
- **Mistake Alignment**: RMSE of question-level error rates
- **Efficiency Alignment**: Runtime correlation between LLM and student code

### Psychometric Metrics
- **Ability Correlation**: Pearson correlation of estimated student abilities
- **Difficulty Correlation**: Pearson/Spearman correlation of item difficulties

## Supported Models

### Commercial Models
- **Claude**: Sonnet 4 (claude-sonnet-4-20250514)
- **GPT-4**: o1-mini
- **Gemini**: 1.5-flash
- **Mistral**: mistral-large-latest

### Open Source Models
- **Gemma**: 3-27b-it
- **Llama**: 3.1-8b-instruct  
- **Qwen**: 2.5-14b-instruct

## Configuration

### API Rate Limits
- Adjust `max_workers` and `delay` parameters in `run_commercial_model.py`
- Default settings are conservative to avoid rate limiting

### Data Sources
- Base URL: `https://huggingface.co/datasets/Kazchoko/codeinsights_llm_simulation/resolve/main/`
- Student data: `stair-lab/code_insights_csv`

### Code Execution Environment
- Compiler: `g++ -std=c++17`
- Timeout: 5 seconds per test case
- Supports C++ with standard libraries

## Output Files

### Primary Results
- `all_results.json`: Complete metrics across all scenarios and models
- `all_correlations.json`: Psychometric correlation analysis

### Intermediate Data
- `scenario_results/{model}/{model}_scenario{1-4}.csv`: Raw LLM responses
- `ability/{model}_student_ability.csv`: Estimated student abilities
- `difficulty/{model}_difficulty.csv`: Estimated item difficulties

## Error Handling

The framework includes robust error handling for:
- API timeouts and rate limits
- Code compilation failures
- Runtime errors and infinite loops
- Missing or malformed data
- AST parsing failures with fallback mechanisms

## Contributing

When extending the framework:
1. Follow the existing modular structure
2. Add comprehensive error handling
3. Update documentation for new metrics or models
4. Test with small datasets before full runs