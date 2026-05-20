# CodeInsight: Modeling Student Learning Dynamics

Predicting how students improve over time is a fundamental challenge in educational assessment. This project compares classical knowledge-tracing models (IRT, CIRT, BKT, DKT) with an LLM-based predictor on longitudinal programming submissions, using a shared student-split evaluation framework that ensures identical data splits across all approaches.

Paper: [EMNLP 2026 submission](https://aclanthology.org/2025.aimecon-main.43.pdf)

## Dataset

[`CodeInsightTeam/code_insights_csv`](https://huggingface.co/datasets/CodeInsightTeam/code_insights_csv) contains C++ submissions from first-year university students. 4 courses (DSA-HK231, DSA-HK221, PF-HK232, PF-HK222), 3,286 students, 396 problems, 3M+ submissions.

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
├── data_analysis/             # Paper figure scripts + dataset analysis
│   ├── plot_filtered_accuracy.py          # Per-attempt accuracy (Fig. 3)
│   ├── kendall_tau_decomposition.py       # Kendall tau decomposition (Fig. 5)
│   ├── llm_behavioral_comparison.py       # Behavioral comparison (Fig. 6)
│   ├── llm_bug_comparison.py              # Error type flow (Fig. 4)
│   ├── visualize_response_matrix.py       # Attempt progression (Fig. 1-2)
│   ├── learning_curves_by_year.py         # Learning curves (Fig. data)
│   ├── dataset_summary_table.py           # Dataset summary table
│   ├── student_behavior_clustering.py     # Student behavior (Appendix)
│   ├── problem_by_problem_analysis.py     # Problem patterns (Appendix)
│   └── pace_analysis.py                   # Submission pacing (Appendix)
│
├── dynamic_models/            # Knowledge-tracing models
│   ├── cirt.py                # Continuous IRT (temporal decay)
│   ├── dynamic_irt.py         # Dynamic IRT (time-varying traits)
│   ├── elo.py                 # Elo rating system
│   ├── gpirt.py               # Gaussian Process IRT
│   ├── rssm.py                # Recurrent State-Space Model
│   ├── featurize.py           # Feature extraction
│   └── temporal_eval/         # Evaluation framework
│       ├── run_student_eval.py    # Entry point: student-split evaluation
│       ├── harness.py             # Orchestrates model fitting + prediction
│       ├── student_split.py       # 70/30 student split logic
│       ├── data_filter.py         # Quality filtering (coverage, pass rates)
│       ├── data_loader.py         # Unified data loading + caching
│       ├── metrics.py             # AUC, accuracy, F1, log-likelihood
│       ├── plot_results.py        # Loss curves, diagnostics
│       └── adapters/              # Per-model adapters
│           ├── irt_adapter.py
│           ├── cirt_adapter.py
│           ├── bkt_adapter.py
│           ├── dkt_adapter.py
│           └── ...
│
├── llm_simulator/             # LLM-as-Predictor (grounded evaluation)
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

## Evaluation Framework

All models are evaluated using the same student-split protocol:

1. **Quality filtering**: Remove students and items below coverage/pass-rate thresholds (configurable via `DataFilterConfig`)
2. **Student split**: Randomly partition students into 70% train / 30% test (seed-controlled for reproducibility)
3. **Train phase**: Train students' responses across all weeks estimate item difficulty parameters
4. **Calibration phase**: Test students' weeks 1-3 responses calibrate their ability estimates
5. **Prediction phase**: Predict test students' performance on weeks 4-6

This shared split ensures knowledge-tracing models and the LLM predictor are evaluated on identical student-item pairs.

### Running the evaluation

```bash
# Run all models (IRT, CIRT, BKT, DKT) on all courses
python -m dynamic_models.temporal_eval.run_student_eval

# Specific models on a specific course
python -m dynamic_models.temporal_eval.run_student_eval --models IRT BKT DKT --courses dsa_hk231

# With loss curve plots
python -m dynamic_models.temporal_eval.run_student_eval --plot_losses
```

Results are saved to `results/student_eval/{course}/student_eval.csv` with per-model metrics (AUC, accuracy, F1, log-likelihood). Trained model predictions are serialized as `{model}_student_pred.pkl`.

## LLM Simulator

Evaluates whether LLMs can predict real student behavior on programming problems. Uses a **grounded evaluation** approach: the LLM follows each student's real attempt trajectory step-by-step and predicts the next submission, rather than generating code freely.

The simulator uses the same student split as the knowledge-tracing models, enabling direct comparison. For each test student on each target question (weeks 4-6):

1. **Persona**: Build a behavioral profile from weeks 1-3 (submission pacing, precheck usage, topic pass rates, code complexity)
2. **RAG**: Retrieve the student's most similar prior problem trajectories via TF-IDF
3. **Summarize**: Compress retrieved trajectories into behavioral summaries (via Claude Haiku)
4. **Predict**: At each attempt, the LLM sees the student's full trajectory so far and predicts the next submission
5. **Grade**: The predicted code is compiled and graded against unit tests

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

## Figure Generation

All paper figures are generated via `scripts/build_figures.sh`, which runs individual analysis scripts and copies outputs to `overleaf/figures/`.

```bash
bash scripts/build_figures.sh

# Regenerate with fresh grading cache
bash scripts/build_figures.sh --regrade

# Rebuild trajectory metrics
bash scripts/build_figures.sh --rebuild-trajectories
```

## Reproducibility

Full pipeline from evaluation to paper figures:

```bash
# 1. Run knowledge-tracing models
python -m dynamic_models.temporal_eval.run_student_eval

# 2. Run LLM simulator
python -m llm_simulator.eval_student_split --course dsa_hk231 --models haiku

# 3. Generate all paper figures
bash scripts/build_figures.sh
```
