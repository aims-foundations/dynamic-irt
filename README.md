# Dynamic Measurement Model

A research project focused on measuring an unobserved quantitive that can change overtime, such as latent ability during learning. Toward this end, the center of the project is a set of probabilistic latent dynamic models (ncluding Elo, CIRT, GPIRT, and RSSM) that infers ability trajectories from data. Using this model, we can accomplish various goals. For example, we can predict future performance of learners with limited measure. In addition, we can use the latent trajectories as a way to compare two learners. This is particularly useful to compare the realism of virtual students, which has recently gained poplarity in the research community for the promise of enabling highly custimized learning curriculum for human through the mean of simulation. 

Here is a related paper: https://aclanthology.org/2025.aimecon-main.43.pdf

We use CodeInsight data from first year university students learning C++ (includes both pass/fail and raw code)
Data sources: [`CodeInsightTeam/code_insights_csv`](https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv). Supported courses: DSA-HK231 (L09, DT01), DSA-HK222, DSA-HK232, PF courses. 3,286 students, 396 problems, 3M+ submissions. Canonical CSV Schema

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

## Getting Started

```bash
pip install -r requirements.txt
export HF_TOKEN="..."
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
├── dynamic_models/            # All learning dynamics models (flat layout)
│   ├── cirt.py                # Continuous IRT (parametric growth curves)
│   ├── dynamic_models.py      # Dynamic IRT (time-varying latent traits)
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
├── llm_simulator/             # LLM student simulation (grounded evaluation)
│   ├── eval_student_split.py  # Entry point: orchestrates the full pipeline
│   ├── run.py                 # Core attempt loop: prompt → LLM → grade → repeat
│   ├── persona.py             # Builds behavioral profiles from weeks 1-3
│   ├── rag.py                 # TF-IDF retrieval of similar prior submissions
│   ├── summarize.py           # Compresses submission history via Haiku
│   ├── prompts.py             # Prompt construction and response parsing
│   ├── runners.py             # LLM API wrappers (Claude, GPT, Gemini, Mistral, vLLM)
│   └── data_loader.py         # Data structures, student-split loading, item difficulty
│
├── script/                    # Reproducibility pipeline [TODO: I think this is very outdated]
│   ├── reproduce.sh           # Full pipeline (Steps 1-7)
│   ├── generate_report.py     # Compare results against paper
│   ├── inspect_hf_run.py      # Pull and inspect HELM runs from HF
│   └── upload_to_hf.py        # Push results to HuggingFace
│
└── docs/                      # Hugo-based presentation slides
```

## Dynamic Models

We use temporal evaluation as the primary comparison framework. Week-based temporal splits: train on weeks 1..W, predict weeks W+1+.

[TODO] We first featurize the trajectory (one time computation)
```bash
python -m dynamic_models.featurize --mode features --course all
```

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

Results: https://huggingface.co/datasets/CodeInsightTeam/simulation_output 
[TODO] Plots generated for each models: metrics vs horizon, student trajectories, concept pair scatter.

### Individual Models

```bash
# Elo
python -m dynamic_models.elo

# RSSM (requires pre-processed features)
python -m dynamic_models.rssm --mode features --cls all

# GPIRT (multi-chain MCMC)
[TODO] Can we allow specification of the number of chain? 

CUDA_VISIBLE_DEVICES=0 python -m dynamic_models.gpirt \
    --course_name all --n_samples 15000 --warmup 500 \
    --blocked --testlet --thin 10 --seed 62

[TODO] Can we make this automatically run after the inference is done for MCMC
# MCMC diagnostics (ONLY after all chains finish)
python -m dynamic_models.mcmc_diagnostics --seeds 62 63 64 65 --warmup 500 --course all
```

For GPIRT Operational note:
- **Never run diagnostics while chains are running.** Loading 4 checkpoints (~24 GB each) can OOM and corrupt in-progress `torch.save()` writes.
- Check `free -h` before loading multiple chains (~96 GB for 4 chains on `all`).
- Checkpoints resume automatically if interrupted (saves every 500 iterations).

## LLM Simulator

Evaluates whether LLMs can predict real student behavior on programming problems. Uses a **grounded evaluation** approach: the LLM follows each student's real attempt trajectory step-by-step and predicts the next submission, rather than generating code freely.

### Data Split
- **Train students (70%)**: compute item difficulty metrics
- **Test students (30%)**: evaluation targets
  - Weeks 1-3: build persona + RAG context
  - Weeks 4-6: prediction targets (attempt-by-attempt)

The simulator uses the same student split as the psychometric models (IRT, CIRT, BKT, DKT), enabling direct comparison.

### Supported Models
Commercial: `opus` (Claude Opus), `haiku` (Claude Haiku), `gpt` (GPT-4.1-nano), `gemini` (Gemini 2.0 Flash), `mistral` (Mistral Large).
Open-source (via vLLM): `llama` (Llama-3.1-8B), `gemma` (Gemma-3-27B), `qwen` (Qwen2.5-14B), `glm` (GLM-4.7-AWQ).

```bash
# Full evaluation on a course
python -m llm_simulator.eval_student_split --course dsa_hk231 --models haiku

# Quick test with subset
python -m llm_simulator.eval_student_split --course dsa_hk231 --models haiku --max_students 5 --max_questions 3 --dry_run

# Multiple models
python -m llm_simulator.eval_student_split --course dsa_hk231 --models haiku opus gpt
```

Output: `results/llm_student_eval/{course}/{model}_attempts{N}.jsonl`

## Comparative Analysis
[TODO] Largely not done
- **Ability distributions**: KS test, KL divergence, Wasserstein distance
- **Difficulty alignment**: Pearson/Spearman correlation on shared items
- **Learning curves**: Compare growth rates and asymptotic abilities
- **Archetype clustering**: K-means on IRT-derived features, compare D1 vs D2 cluster profiles

## Reproducibility

```bash
cd CodeInsights/script

# Full pipeline (data -> LLM eval -> metrics -> psychometrics -> plots -> report)
./reproduce.sh

# Skip LLM evaluation (use existing results)
./reproduce.sh --skip-llm
```