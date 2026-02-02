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
├── dynamic_irt/            # All learning dynamics models
│   ├── cirt/               # Continuous IRT models
│   ├── gpirt/              # Gaussian Process IRT
│   ├── elo/                # Elo-based rating (Edmentum + CodeInsight)
│   ├── rssm/               # Recurrent State-Space Models
│   ├── data_analysis/      # Student clustering, cheating detection
│   └── simulated_learner/  # Synthetic student behavior
│
├── llm_simulator/          # All LLM-related functionality
│   ├── configs/            # Training YAML configs
│   ├── grading_engine/     # C++ code compilation/execution
│   ├── llm_evaluation/     # vLLM-based code evaluation
│   ├── evaluation/         # Multi-model LLM evaluation framework
│   ├── helm_codeinsights/  # Custom HELM scenarios/metrics
│   └── helm/               # Stanford CRFM HELM (submodule)
│
└── docs/                   # Hugo-based presentation slides
```

## Data Collection

```bash
# Scrape student submissions
python data_collection/collect_data.py --course_name DSA-HK231 --class_name L09

# Convert to matrix format
python data_collection/convert_matrices.py --course_name dsa_hk231
```

## Learning Dynamics Models

### Data Analysis
```bash
python dynamic_irt/data_analysis/main_analyzing.py --course_name DSA-HK231 --class_name L09
```

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

### Training (SFT with LoRA)
```bash
trl sft --config llm_simulator/configs/sft_dsa_hk231.yaml \
    --use_peft --lora_r 256 --lora_alpha 512 --lora_dropout 0.1
```

### Merge and Push to HuggingFace
```bash
python llm_simulator/merge_push.py --config configs/sft_dsa_hk231.yaml \
    --use_peft --lora_r 256 --lora_alpha 512 --lora_dropout 0.1
```

### LLM Evaluation
```bash
# Run evaluation pipeline
python llm_simulator/evaluation/data_preprocessing.py
python llm_simulator/evaluation/run_commercial_model.py
python llm_simulator/evaluation/compute_metrics.py
python llm_simulator/evaluation/psychometrics_metrics.py
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
