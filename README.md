# CodeInsights: Models of Human Learning Dynamics

A research project focused on understanding and predicting learning dynamics over time.

## Motivation

Traditional psychometrics models learning from binary response data (pass/fail). This project extends that paradigm: because we have **raw student code submissions**, we can pursue two complementary approaches:

- **Dynamic IRT** (`dynamic_models/`) — Psychometric models (Elo, CIRT, GPIRT, RSSM) for learning curves from response data
- **LLM Simulation** (`llm_simulator/`) — Fine-tuned LLMs that mimic student coding behavior from raw text

## Datasets

1. **CodeInsight** — University students learning C++ (includes both pass/fail and raw code)
2. **Edmentum** — K-12 students learning math and reading (pass/fail only)

Data sources: HuggingFace [`stair-lab/code_insights_csv`](https://huggingface.co/datasets/stair-lab/code_insights_csv), [`CodeInsightTeam/code_insights_csv`](https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv)

Supported courses: DSA-HK231 (L09, DT01), DSA-HK222, DSA-HK232, PF courses

## Getting Started

```bash
pip install -r requirements.txt
```

## Project Structure

```
CodeInsights/
├── data_collection/           # Web scraping + data preprocessing
│   ├── collect_data.py        # Selenium-based scraping from online judge
│   ├── csv2matrices.py        # CSV -> 3D tensor matrices
│   ├── convert_matrices.py    # Data ETL for matrix format
│   └── skill_tagging/         # LLM-based skill labeling for questions
│
├── data_analysis/             # Analysis of student behavior
│   ├── eda_codeinsight.py     # Exploratory data analysis
│   ├── student_behavior_clustering.py  # K-means on submission patterns
│   ├── pace_analysis.py       # Timing and pacing analysis
│   ├── compute_metrics.py     # Functional correctness, AST, CodeBERT
│   ├── psychometrics_metrics.py        # IRT ability/difficulty estimation
│   └── generate_plots.py      # Publication figures
│
├── dynamic_models/               # All learning dynamics models (flat layout)
│   ├── cirt.py                # Continuous IRT (parametric growth curves)
│   ├── dynamic_models.py         # Dynamic IRT (time-varying latent traits)
│   ├── elo.py                 # Elo-based rating (Edmentum + CodeInsight)
│   ├── gpirt.py               # Gaussian Process IRT (ESS inference)
│   ├── rssm.py                # Recurrent State-Space Models
│   ├── featurize.py           # Model-agnostic feature extraction
│   ├── mcmc_diagnostics.py    # GPIRT chain convergence diagnostics
│   └── temporal_eval/         # Temporal evaluation framework
│       ├── harness.py         # Run all models across temporal horizons
│       ├── data_loader.py     # Unified data loading
│       ├── temporal_split.py  # Week-based train/test splits
│       ├── metrics.py         # AUC, accuracy, F1, RMSE, log-likelihood
│       ├── plot_results.py    # Metrics, trajectories, concept scatter
│       ├── run_temporal_eval.py  # CLI entry point
│       └── adapters/          # Per-model adapters (Elo, CIRT, GPIRT, ...)
│
├── llm_simulator/             # LLM student simulation
│   ├── runners.py             # Unified model runners + registry
│   ├── prompts.py             # Prompt builder (zero-shot / few-shot / feedback)
│   ├── data_loader.py         # Data loading + example selection
│   ├── run.py                 # Unified CLI entry point
│   ├── grading_engine.py      # C++ compilation and test execution
│   ├── data_preprocessing.py  # Generate scenario datasets from HF
│   ├── config.py              # Legacy prompt templates (used by training)
│   ├── utils.py               # Inference utilities (used by training)
│   └── training/              # SFT fine-tuning pipeline
│       ├── build_dataset.py   # Build SFT training datasets
│       ├── sft.py             # SFT training wrapper (TRL)
│       ├── main_optimize.py   # Hyperparameter optimization
│       ├── merge_model.py     # LoRA merge + HF push
│       └── configs/           # YAML training configs
│
├── script/                    # Reproducibility pipeline
│   ├── reproduce.sh           # Full pipeline (Steps 1-7)
│   ├── generate_report.py     # Compare results against paper
│   ├── inspect_hf_run.py      # Pull and inspect HELM runs from HF
│   └── upload_to_hf.py        # Push results to HuggingFace
│
├── archived/                  # Legacy code (HELM benchmark components)
└── docs/                      # Hugo-based presentation slides
```

## Dynamic IRT Models

### Temporal Evaluation (primary comparison framework)

Week-based temporal splits: train on weeks 1..W, predict weeks W+1+.

```bash
cd CodeInsights

# Run all models
python -m dynamic_models.temporal_eval.run_temporal_eval --course_name all

# Specific models
python -m dynamic_models.temporal_eval.run_temporal_eval \
    --course_name dsa_hk231 --models Elo CIRT DynamicIRT

# Skip slow models (GPIRT)
python -m dynamic_models.temporal_eval.run_temporal_eval --skip_slow
```

Results: `results/temporal_eval/temporal_eval_{course}.csv`

Plots generated: metrics vs horizon, student trajectories, concept pair scatter.

### Individual Models

```bash
# Elo
python -m dynamic_models.elo

# RSSM (requires pre-processed features)
python -m dynamic_models.featurize --mode features --course all
python -m dynamic_models.rssm --mode features --cls all

# GPIRT (multi-chain MCMC)
CUDA_VISIBLE_DEVICES=0 python -m dynamic_models.gpirt \
    --course_name all --n_samples 15000 --warmup 500 \
    --blocked --testlet --thin 10 --seed 62

# MCMC diagnostics (ONLY after all chains finish)
python -m dynamic_models.mcmc_diagnostics --seeds 62 63 64 65 --warmup 500 --course all
```

### GPIRT Operational Notes

- **Never run diagnostics while chains are running.** Loading 4 checkpoints (~24 GB each) can OOM and corrupt in-progress `torch.save()` writes.
- Check `free -h` before loading multiple chains (~96 GB for 4 chains on `all`).
- Checkpoints resume automatically if interrupted (saves every 500 iterations).

## LLM Simulator

Treats language models as student behavior generators: questions in, code responses out.

One unified scenario — "imitate this student" — parameterized by:
- `--n_examples N`: number of in-context examples (0 = zero-shot, N = few-shot with student code)
- `--max_attempts N`: retry budget (1 = single-shot, >1 = iterative with compile/test feedback)

All metrics (functional correctness, AST similarity, CodeBERT, mistake alignment, runtime) are computed on every output.

### Supported Models

**Commercial**: `claude` (claude-sonnet-4), `gpt` (gpt-4.1-nano), `gemini` (gemini-2.0-flash), `mistral` (mistral-large-latest)

**Open-source** (via vLLM): `llama` (Llama-3.1-8B), `gemma` (Gemma-3-27B), `qwen` (Qwen2.5-14B), `glm` (GLM-4.7-AWQ)

### Running

```bash
cd CodeInsights

# Generate scenario datasets
python -m llm_simulator.data_preprocessing

# Zero-shot (no student examples)
python -m llm_simulator.run --models claude gpt --n_examples 0

# Few-shot student simulation (3 examples of the student's code)
python -m llm_simulator.run --models claude gpt --n_examples 3

# Iterative with test feedback (zero-shot + retry on failure)
python -m llm_simulator.run --models claude --max_attempts 100

# Few-shot + iterative
python -m llm_simulator.run --models claude --n_examples 3 --max_attempts 5

# Quick test (dry run)
python -m llm_simulator.run --models claude --n_examples 0 --max_samples 2 --dry_run
```

### SFT Training (LoRA)

```bash
trl sft --config llm_simulator/training/configs/sft_dsa_hk231.yaml \
    --use_peft --lora_r 256 --lora_alpha 512 --lora_dropout 0.1
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."
export HF_TOKEN="..."
```

## Reproducibility

```bash
cd CodeInsights/script

# Full pipeline (data -> LLM eval -> metrics -> psychometrics -> plots -> report)
./reproduce.sh

# Skip LLM evaluation (use existing results)
./reproduce.sh --skip-llm
```

---

## D1/D2 Pipeline: Comparing Human vs LLM Learning Dynamics

**Goal**: Create a unified pipeline so that human data (D1) and LLM-simulated data (D2) share a canonical format. Any IRT model can consume either dataset identically, enabling direct comparison of latent statistics (ability trajectories, difficulty estimates, learning curves).

### Canonical CSV Schema

```
student_id      : str    -- learner identifier
item_id         : str    -- "question_id_testcase_index" (atomic binary item)
attempt_index   : int    -- 0-indexed attempt number per (student, item)
correctness     : int    -- 0 or 1
timestamp_days  : float  -- days since course start (real for D1, ordinal for D2)
response        : str    -- raw code (optional, for RSSM embeddings)
source          : str    -- "human" or LLM model name
course_id       : str    -- course identifier
```

### Stage 1: Data Export

**D1 (Human Data)**
- Source: HuggingFace `stair-lab/code_insights_csv` (`main_data.csv`)
- Scale: 3,286 students, 396 problems, 3M+ submissions
- Processing: explode binary pass strings into per-testcase rows, compute timestamps and attempt indices

**D2 (LLM-Simulated Data)**
- Two input formats: HELM (open-source models from HF) and CSV (commercial models from `run_single_turn.py`)
- Timestamps: ordinal indexing (avoids fabricating temporal data)
- Multi-attempt: D2 has single attempts per question (sufficient for Elo; noted limitation for CIRT/GPIRT)

### Stage 2: Unified Model Fitting

Each model produces standardized outputs:
- `ability_trajectories.csv`: student ability estimates over time
- `difficulty_estimates.csv`: per-item difficulty estimates
- `fit_metrics.json`: AUC, accuracy, F1, log-likelihood

Model priority: Elo (works naturally with single-attempt D2) > CIRT > GPIRT > RSSM (deferred, requires CodeBERT embeddings)

### Stage 3: Comparative Analysis

- **Ability distributions**: KS test, KL divergence, Wasserstein distance
- **Difficulty alignment**: Pearson/Spearman correlation on shared items
- **Learning curves**: Compare growth rates and asymptotic abilities
- **Archetype clustering**: K-means on IRT-derived features, compare D1 vs D2 cluster profiles

### D2 Data Inventory

| Source | Models | Temps | Status |
|--------|--------|-------|--------|
| HF `CodeInsightTeam/evaluation_results` | gemma, llama, deepseek | 0.0 | S1/S2/S3 only |
| Local `codeinsights_Dec8` | gemma, llama, qwen | 0.3/0.6/0.9 | 35/36 complete |
| Local `codeinsights_Oct3` | llama, qwen, gpt-4o | 0.0/0.5/1.0 | Mostly incomplete |
| Commercial (from `run_single_turn.py`) | claude, gpt, gemini, mistral | -- | Needs locating |

### Key Findings So Far

- Claude Sonnet 4 achieves highest UTSR (0.668) and best ability recovery (rho=0.359)
- Open-source models have better structural alignment (lower ASTED) but worse functional correctness
- All models over-optimize (EAS > 1.0) -- they write faster code than real students
- Psychometric correlations are modest (0.15-0.36), indicating LLMs don't yet faithfully recover latent parameters
