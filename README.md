# CodeInsights: Models of Human Learning Dynamics

A research project focused on understanding and predicting learning dynamics over time.

## Motivation

Traditional psychometrics models learning from binary response data (pass/fail). This project extends that paradigm: because we have **raw student code submissions**, we can pursue two complementary approaches:

- **Dynamic IRT** (`dynamic_irt/`) - Classic psychometric models (Elo, GPIRT, RSSM) for learning curves from response data
- **LLM Simulation** (`llm_simulator/`) - Fine-tuned LLMs that mimic student coding behavior from raw text

## Datasets

1. **CodeInsight** - University students learning C++ (includes both pass/fail and raw code)
2. **Edmentum** - K-12 students learning math and reading (pass/fail only)

## Getting Started

```bash
git clone https://github.com/sangttruong/CodeInsights.git
pip install -r requirements.txt
```

## Project Structure

```
CodeInsights/
├── data_collection/        # Web scraping + data preprocessing
│   ├── collect_data.py     # Selenium-based scraping
│   ├── convert_matrices.py # Data ETL for matrix format
│   └── wnb_configs/        # Weights & Biases configs
│
├── data_analysis/          # Analysis of student behavior and LLM outputs
│   ├── eda_codeinsight.py              # Exploratory data analysis
│   ├── student_behavior_clustering.py  # K-means clustering on submissions
│   ├── pace_analysis.py                # Timing and pacing analysis
│   ├── compute_metrics.py              # Functional correctness, AST, CodeBERT metrics
│   ├── psychometrics_metrics.py        # IRT ability/difficulty estimation
│   ├── cossim_calculator.py            # AST edit distance + CodeBERT similarity
│   └── generate_plots.py              # Publication figures
│
├── dynamic_irt/            # All learning dynamics models
│   ├── cirt/               # Continuous IRT models
│   ├── gpirt/              # Gaussian Process IRT
│   ├── elo/                # Elo-based rating (Edmentum + CodeInsight)
│   ├── rssm/               # Recurrent State-Space Models
│   ├── simulated_learner/  # Synthetic student behavior
│   └── codeinsights_testlet_analysis.R  # Bayesian Testlet Response Theory (R)
│
├── llm_simulator/          # LLM input→output engine
│   ├── config.py           # Prompt templates and model configs
│   ├── grading_engine.py   # C++ compilation and test execution
│   ├── utils.py            # Inference utilities (vLLM, prompt formatting)
│   ├── data_preprocessing.py           # Generates scenario datasets from HF
│   ├── run_single_turn.py              # Single-turn runners for all models (API + vLLM)
│   ├── run_iterative_model.py          # Multi-attempt S1 simulation with test feedback
│   └── training/           # SFT fine-tuning pipeline
│
├── reproducibility_results/ # Pipeline outputs and report
│   └── generate_report.py  # Compares results against paper values
│
├── reproduce.sh            # Full reproducibility pipeline (Steps 1–7)
└── docs/                   # Hugo-based presentation slides
```

## Data Collection

```bash
# Scrape student submissions
python data_collection/collect_data.py --course_name DSA-HK231 --class_name L09

# Convert to matrix format
python data_collection/convert_matrices.py --course_name dsa_hk231
```

## Data Analysis

Exploratory data analysis, clustering, and behavioral analysis of student submissions.

```bash
# Reproduce all analyses (EDA, clustering, pacing, problem-level)
cd data_analysis
bash reproduce.sh

# Or run individual scripts
python data_analysis/eda_codeinsight.py
python data_analysis/student_behavior_clustering.py
python data_analysis/pace_analysis.py --all_courses
python data_analysis/problem_by_problem_analysis.py --all_courses
```

See [data_analysis/README.md](data_analysis/README.md) for detailed documentation.

## Learning Dynamics Models

### RSSM (Recurrent State-Space Models)
```bash
python dynamic_irt/rssm/process_data.py
python dynamic_irt/rssm/main_rssm.py
```

### Elo-based IRT (supports Edmentum + CodeInsight datasets)
```bash
cd CodeInsights
python -m dynamic_irt.elo.main_elo
python -m dynamic_irt.elo.difficulty_analysis
```

## LLM Simulator

The LLM simulator treats language models as student behavior generators: questions go in, code responses come out. It evaluates LLMs across four scenarios that mirror real student interactions with an online judge.

### Evaluation Scenarios

| Scenario | Goal | Key Metrics |
|----------|------|-------------|
| **S1 – Correct Code** | Write a correct C++ solution | Functional correctness |
| **S2 – Behavior Imitation** | Replicate a specific student's coding style | Correctness + AST + CodeBERT similarity |
| **S3 – Error Replication** | Generate realistic student mistakes | All S2 metrics + mistake alignment |
| **S4 – Efficiency Alignment** | Match a student's runtime efficiency | All S3 metrics + runtime correlation |

### Supported Models

**Commercial**: `claude-sonnet-4`, `gpt-4.1-nano`, `gemini-2.0-flash`, `mistral-large-latest`

**Open-source** (via vLLM): `gemma-3-27b-it`, `llama-3.1-8b-instruct`, `qwen2.5-14b-instruct`

### Installation

```bash
pip install pandas numpy requests jinja2 anthropic openai google-generativeai mistralai
pip install transformers torch scikit-learn apted tree-sitter tree-sitter-cpp
pip install scipy matplotlib seaborn tueplots huggingface-hub vllm

# R dependencies (for testlet analysis)
Rscript -e 'install.packages(c("dplyr", "tidyr", "ggplot2", "brms", "pROC"))'
```

### Running the Full Pipeline

```bash
# Set API keys
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."
export HF_TOKEN="..."

# Run all steps (data → LLM eval → metrics → psychometrics → plots → report)
./reproduce.sh

# Skip LLM evaluation (use existing results)
./reproduce.sh --skip-llm
```

### Running Individual Steps

```bash
# Step 1: Generate scenario datasets from HuggingFace
(cd llm_simulator && python data_preprocessing.py)

# Step 2: Run LLMs (single-shot, all 4 scenarios)
(cd llm_simulator && python run_single_turn.py --models claude gpt gemini mistral)   # commercial
(cd llm_simulator && python run_single_turn.py --models llama gemma qwen)            # open-source (GPU)

# Step 2b: Run iterative S1 evaluation (multi-attempt with test feedback)
(cd llm_simulator && python run_iterative_model.py --models claude-sonnet-4 gpt-4.1-nano)

# Step 3–4: Compute metrics and psychometrics
(cd data_analysis && python compute_metrics.py)
python data_analysis/psychometrics_metrics.py --data reproducibility_results --output reproducibility_results/psychometrics

# Step 5: Testlet analysis (R)
Rscript dynamic_irt/codeinsights_testlet_analysis.R
```

### Training (SFT with LoRA)

```bash
trl sft --config llm_simulator/training/configs/sft_dsa_hk231.yaml \
    --use_peft --lora_r 256 --lora_alpha 512 --lora_dropout 0.1
```

## Documentation

Run the presentation slides locally:
```bash
cd docs && hugo server
```

## Supported Courses

- DSA-HK231 (L09, DT01)
- DSA-HK222
- DSA-HK232
- PF courses

## Data Sources

- HuggingFace: `CodeInsightTeam/code_insights_csv`, `stair-lab/code_insights_csv`
