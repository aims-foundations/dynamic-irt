# Dynamic Measurement Model

A research project focused on measuring unobserved quantities that change over time, such as latent ability during learning. The project combines probabilistic latent dynamic models (IRT, CIRT, BKT, DKT) with LLM-based prediction to model student learning trajectories from programming submissions. All models share a common evaluation framework, enabling direct comparison on identical data splits.

Related paper: https://aclanthology.org/2025.aimecon-main.43.pdf

## Dataset

We use CodeInsight data from first-year university students learning C++ (includes both pass/fail and raw code).
Data source: [`CodeInsightTeam/code_insights_csv`](https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv). 4 courses (DSA-HK231, DSA-HK221, PF-HK232, PF-HK222), 3,286 students, 396 problems, 3M+ submissions.

```
student_id      : str    -- learner identifier
item_id         : str    -- "question_id_testcase_index" (atomic binary item)
attempt_index   : int    -- 0-indexed attempt number per (student, item)
correctness     : int    -- 0 or 1
timestamp_days  : float  -- days since course start
response        : str    -- raw C++ code
course_id       : str    -- course identifier
```

## Getting Started

```bash
pip install -r requirements.txt
export HF_TOKEN="..."
```

## Project Structure

```
├── data_collection/           # Web scraping + data preprocessing
│   ├── collect_data.py        # Selenium-based scraping from online judge
│   ├── csv2matrices.py        # CSV -> 3D tensor matrices
│   └── skill_tagging/         # LLM-based skill labeling for questions
│
├── data_analysis/             # Analysis scripts + paper figure generation
│   ├── plot_filtered_accuracy.py          # Per-attempt accuracy
│   ├── kendall_tau_decomposition.py       # Kendall tau decomposition
│   ├── llm_behavioral_comparison.py       # Behavioral comparison
│   ├── llm_bug_comparison.py              # Error type flow
│   ├── visualize_response_matrix.py       # Attempt progression heatmaps
│   ├── learning_curves_by_year.py         # Learning curves
│   ├── dataset_summary_table.py           # Dataset summary table
│   ├── student_behavior_clustering.py     # Student behavior clustering
│   ├── problem_by_problem_analysis.py     # Problem-level patterns
│   └── pace_analysis.py                   # Submission pacing
│
├── dynamic_models/            # Probabilistic latent dynamic models
│   └── temporal_eval/         # Evaluation framework
│       ├── run_student_eval.py    # Entry point: 
│       ├── harness.py             # Orchestrates model fitting + prediction
│       ├── student_split.py       # data split logic
│       ├── data_filter.py         # Quality filtering (coverage, pass rates)
│       ├── data_loader.py         # Unified data loading + caching
│       ├── metrics.py             # AUC, accuracy, F1, log-likelihood
│       ├── plot_results.py        # Loss curves, diagnostics
│       └── adapters/              # Per-model adapters (IRT, CIRT, BKT, DKT, ...)
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
├── scripts/
│   └── build_figures.sh       # Generate all paper figures → overleaf/figures/
│
└── overleaf/                  # LaTeX paper (EMNLP 2026, ACL format)
```

## Student-Split Evaluation

All models are evaluated using a shared student-split protocol:

1. **Quality filtering** — Remove students and items below coverage/pass-rate thresholds
2. **Student split** — Randomly partition into 70% train / 30% test students (seed-controlled)
3. **Train phase** — Train students' responses estimate item difficulty parameters across all weeks
4. **Calibration** — Test students' weeks 1-3 responses calibrate their ability estimates
5. **Prediction** — Predict test students' performance on weeks 4-6

This shared split ensures all models (knowledge-tracing and LLM) are evaluated on identical student-item pairs.

```bash
# Run all models (IRT, CIRT, BKT, DKT) on all courses
python -m dynamic_models.temporal_eval.run_student_eval

# Specific models on a specific course
python -m dynamic_models.temporal_eval.run_student_eval --models IRT BKT DKT --courses dsa_hk231
```

Results saved to `results/student_eval/{course}/student_eval.csv`.

## LLM Simulator

Evaluates whether LLMs can predict real student behavior on programming problems. Uses a **grounded evaluation** approach: the LLM follows each student's real attempt trajectory step-by-step and predicts the next submission, rather than generating code freely.

The simulator uses the same student split as the knowledge-tracing models. For each test student on each target question (weeks 4-6):

1. Build a behavioral **persona** from weeks 1-3 (submission pacing, precheck usage, topic pass rates, code complexity)
2. Retrieve similar prior problem trajectories via **TF-IDF** (RAG)
3. Compress retrieved trajectories into behavioral **summaries** (via Claude Haiku)
4. At each attempt, the LLM sees the student's full trajectory so far and predicts the next submission
5. The predicted code is compiled and **graded** against unit tests

### Supported Models
Commercial: `opus` (Claude Opus 4.6), `haiku` (Claude Haiku), `gpt` (GPT-4.1-nano), `gemini` (Gemini 2.0 Flash), `mistral` (Mistral Large).
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

## Figure Generation

All paper figures are generated via `scripts/build_figures.sh`, which runs individual analysis scripts and copies outputs to `overleaf/figures/`.

```bash
bash scripts/build_figures.sh

# Regenerate with fresh grading cache
bash scripts/build_figures.sh --regrade
```

## Reproducibility

```bash
# 1. Run knowledge-tracing models
python -m dynamic_models.temporal_eval.run_student_eval

# 2. Run LLM simulator
python -m llm_simulator.eval_student_split --course dsa_hk231 --models haiku

# 3. Generate all paper figures
bash scripts/build_figures.sh
```
