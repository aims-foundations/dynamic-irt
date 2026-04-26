# LLM as a Dynamic Predictive Model

## Hypothesis

An LLM conditioned on a student's submission history is a competitive generative model of student coding performance, producing calibrated pass/fail predictions under the same temporal evaluation protocol as parametric and neural models.

When given N prior question-attempt trajectories from weeks 1..W (including the student's actual C++ code, Precheck/Submit actions, and per-test pass patterns), an LLM can generate code on unseen week W+1..end problems whose graded outcomes achieve metrics comparable to the five existing models (Elo, CIRT, DynamicIRT, GPIRT, RSSM).

## Approach: LLM as 6th Dynamic Model

Create an `LLMAdapter` that plugs into the temporal evaluation framework (`temporal_eval/`), using **pre-generated simulation data** from `CodeInsightTeam/simulation_output` on HuggingFace — no new LLM calls needed.

1. Load pre-generated simulation JSONL (graded LLM code outputs for each (student, question) pair)
2. Match simulation results to real student outcomes using temporal splits
3. Use the simulated pass/fail pattern as the model's prediction
4. Evaluate on the same metrics as all other models

The LLM follows the same `ModelAdapter` contract as all other models: receives `UnifiedData` + `TemporalSplit`, returns `PredictionResult`.

### Available Simulation Data

Source: `CodeInsightTeam/simulation_output` on HuggingFace (57.4 GB total)

| Version | Directory | Size | Description |
|---------|-----------|------|-------------|
| Original | `glm/` | 1.3 GB | Single-attempt, no prompt stored |
| v1 | `v1_rewrite_prompt/` | 1.7 GB | 50 attempts, rewritten prompts |
| **v4** | **`v4_profile_mindiff/`** | **27.2 GB (328K rows)** | **50 attempts, student profile + 10 few-shot examples** |

The **v4** data (`glm_v4_merged.jsonl`) is the primary dataset. Schema matches `main_data.csv` exactly:
```
student_id, course_id, section_id, question_unittest_id, attempt_id,
timestamp, is_exam, response_type, response, pass,
model, n_examples, prompt, raw_response
```

### What the LLM sees (prompt context in v4)

The v4 prompt has three sections:

1. **Student Profile** (aggregate stats — new in v4):
   ```
   === Student Profile ===
   Questions attempted: 67
   Avg attempts per question: 3.9
   Overall pass rate: 8%
   Partial credit rate: 18%
   Precheck/Submit ratio: 5% / 95%
   Avg code length: 413 chars
   Topics solved: C-String, OOP 1, Pointer 2, ...
   Typical strategy: moderate iterator, a few attempts per problem
   Match this student's skill level and behavioral patterns.
   ```

2. **Student Submission History** (10 prior questions with ALL attempts):
   ```
   --- Problem 1: ClockType: setTime ---
   [Full question text + template]
   Attempt 1 [Submit] → Result: 1101:
   [Student's actual C++ code]
   Attempt 2 [Submit] → Result: 1111:
   [Student's revised code]
   ```

3. **Target Question** + instructions to choose [Precheck] or [Submit]

On iterative retries (attempts 1+): previous code + failed test feedback (input/expected/actual).

### Simulation flow per (student, question)

Each pair gets up to 50 iterative attempts:
- Attempt 0: Initial prompt (profile + history + target question)
- Attempt 1+: Same context + feedback from previous failed attempt
- Stops when: all tests pass, submit budget exhausted, or 5 consecutive stalled submits
- The LLM decides [Precheck] vs [Submit] on each attempt

### What the LLM captures that parametric models cannot

The five existing models operate on summary statistics: Elo tracks a scalar ability, CIRT fits a sigmoid curve, DynamicIRT assumes linear growth, GPIRT models a GP trajectory, RSSM encodes handcrafted answer features into a GRU hidden state. None reads actual source code.

The LLM processes raw code text — variable naming conventions, algorithmic strategies, recurring bug patterns, the student's Precheck-vs-Submit decision policy, and how their code evolves across attempts. It can recognize that a student consistently uses brute-force nested loops and fails edge cases involving empty inputs. This is a qualitatively different information channel.

---

## Adversarial Critique

### 1. Output Space Mismatch (Critical)

The 5 models output P(correct) ∈ [0,1]. The LLM produces binary pass/fail from code execution. This breaks metrics:

- **AUC becomes identical to accuracy** — with binary predictions, the ROC curve is a single point
- **RMSE is structurally unfair** — LLM pays no penalty for overconfidence when correct, maximal penalty when wrong
- **Log-likelihood is catastrophic** — a single wrong prediction at p=1e-7 gives log(1e-7) = -16.1, dominating the metric

**Fix**: Use the **test-case pass fraction** from the simulation's `pass` field as `y_pred_prob`. E.g., `"0110100000"` → 3/10 = 0.3. This gives a continuous signal in [0,1] aligned with IRT probability semantics. The all-or-nothing binary is used only for `y_true` (real student outcome).

### 2. Information Asymmetry

The IRT models receive a sparse (student x item x time) tensor of binary outcomes. The LLM receives: full problem description, C++ template, and N prior complete student submissions with all code and results.

This is not apples-to-apples. It is information-theoretically impossible for IRT models to match an approach that reads problem text and student code.

**Fix**: The comparison should be framed not as "which model is better" but "does the LLM use this extra information effectively?" The ablation controls (Section below) isolate this.

### 3. Conflation of LLM Coding Ability vs Student Modeling

When the LLM passes a test, is it because it accurately modeled a weak student and produced code that student would write? Or because the LLM can solve most introductory C++ problems regardless of context?

The prompt says "use the same student's coding style" but there's no enforcement mechanism. The LLM has a strong prior for writing correct C++ that overwhelms any attempt to imitate a struggling student.

**Systematic bias**: Over-predicts success for weak students (LLM writes better code), potentially under-predicts for domain-specific problems the LLM finds hard.

**Fix**: Ablation controls (see below). Also: stratified calibration plot of LLM pass rate vs real student pass rate by ability quartile — slope should be ~1.0 if genuinely modeling students.

### 4. Multi-Attempt Confound

The iterative loop allows up to 50 attempts with feedback. The existing models each produce one P(correct). These are fundamentally different tasks.

- First attempt: no feedback advantage, clean comparison
- Final Submit: measures debugging ability, not prediction
- Best attempt: cherry-picking

**Fix**: Primary evaluation uses first-attempt, Submit-only. Report iterative results separately.

### 5. Cost and Scalability

Generating simulation data from scratch is expensive: at 50 attempts per (student, item) with ~100K observations = ~5M API calls per horizon.

**Fix (already resolved)**: We use pre-generated data from `CodeInsightTeam/simulation_output` (328K rows already computed). The `LLMAdapter` loads this data — no new LLM calls at evaluation time. Runtime is dominated by I/O (loading 27 GB JSONL) not inference. Report the original simulation compute cost for context.

### 6. Problem Modeling vs Student Modeling

The few-shot examples may serve as worked examples of the problem *genre*, not as evidence of student ability. If history includes 3 linked-list problems with all code, the LLM learns "how to solve linked-list problems" more than "what this student typically gets wrong."

**Fix**: Content-disjoint splits (ensure test problems share no topic with training examples). If LLM performance drops on out-of-topic problems but IRT models remain stable, the LLM is leveraging problem content, not student state.

---

## Required Ablation Controls

Three conditions to disentangle coding ability from student modeling:

| Condition | Student Context | What it Tests |
|-----------|----------------|---------------|
| **(a) Few-shot** | Real student's history | Full model (proposed approach) |
| **(b) Zero-shot** | None | LLM coding ability baseline |
| **(c) Shuffled** | Random student's history | Whether student-specific context matters |

**Interpretation**:
- If (a) >> (b) and (a) >> (c): LLM genuinely models the student
- If (a) ≈ (b): LLM is just solving problems, ignoring student context
- If (a) ≈ (c): Student context matters, but not student-*specific* context (genre learning)

---

## Related Work

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| Aher et al. (ICML 2023) "Using LLMs to Simulate Multiple Humans" | LLMs replicate aggregate behavioral trends but compress individual variance | Validates framing; warns about variance compression |
| "Take Out Your Calculators" (arXiv 2601.09953, 2026) | LLM student simulations achieve AUC 0.78-0.90 for item difficulty; *weaker* LLMs produce more realistic errors | Directly validates approach; proficiency calibration is core challenge |
| "Faster, Cheaper, More Accurate" (arXiv 2603.02830) | Traditional KT models beat LLMs on accuracy, speed, cost (600-12,000x cheaper) | LLMs not drop-in replacements for IRT |
| LLM-KT (arXiv 2502.02945) | Hybrid LLM embeddings + sequence models show gains in cold-start | LLMs add value as feature extractors, not standalone |
| Gui & Toubia (arXiv 2312.15524) | Prompt ambiguity violates unconfoundedness in LLM behavioral experiments | Methodological warning for prompt design |

**Consensus**: LLMs as student simulators is validated, but they compress human variance, and the framing should be "complementary signal" not "LLM replaces IRT."

---

## Implementation Design

### `LLMAdapter` (new file: `temporal_eval/adapters/llm_adapter.py`)

Loads pre-generated simulation data and matches against real outcomes per temporal split.

```python
class LLMAdapter(ModelAdapter):
    name = "LLM"
    
    def fit_and_predict(self, data, split, seed=42, **kwargs):
        # 1. Load simulation JSONL from HuggingFace
        #    (CodeInsightTeam/simulation_output, v4_profile_mindiff/glm_v4_merged.jsonl)
        #    Cache the parsed DataFrame across temporal splits
        
        # 2. Filter simulation to test-week questions:
        #    sim_df["week"] = sim_df["question_unittest_id"].map(data.qid_to_week)
        #    test_sim = sim_df[sim_df["week"] > split.cutoff_week]
        
        # 3. Take first Submit per (student, question):
        #    first_submits = test_sim[test_sim["response_type"] == "Submit"]
        #                    .groupby(["student_id", "question_unittest_id"]).first()
        
        # 4. Compute y_pred_prob from simulation pass fraction:
        #    y_pred_prob = first_submits["pass"].apply(pass_fraction)
        
        # 5. Compute y_true from real data (same logic as elo_adapter.py:104-105):
        #    For each (student, question) in test set, binarize real outcome
        
        # 6. Inner join: only pairs in BOTH simulation and real data
        
        # 7. Return PredictionResult(y_true, y_pred_prob)
```

**Data source**: `CodeInsightTeam/simulation_output` on HuggingFace
- Primary file: `v4_profile_mindiff/glm_v4_merged.jsonl` (27.2 GB, 328K rows)
- Alternative: 3 shards at ~6 GB each for parallel loading

**Reuses existing patterns from**:
- `elo_adapter.py:49-56`: `pass_fraction()` logic for converting pass strings to [0,1]
- `elo_adapter.py:104-105`: binarization of real outcomes (all pass → 1.0, else → 0.0)
- `data_loader.py:115-132`: `qid_to_week` mapping (already in `UnifiedData`)
- `temporal_split.py:68-69`: train/test split by week

**Parameters** (via `**kwargs`):
- `sim_data_path`: path to simulation JSONL (default: download from HF)
- `attempt_mode`: "first_submit" (primary) | "best_submit" | "all" for secondary analysis

### Files to create/modify

| File | Action | Purpose |
|------|--------|---------|
| `temporal_eval/adapters/llm_adapter.py` | Create | Main adapter implementation |
| `temporal_eval/harness.py` | Modify | Add LLMAdapter to registry |
| `temporal_eval/adapters/__init__.py` | Modify | Add import |
| `temporal_eval/run_temporal_eval.py` | Modify | Add `--sim_data_path` CLI arg |

---

## Expected Outcomes

The most informative outcome is a **mixed one**:
- LLM excels on students with distinctive, consistent coding styles (where code context provides signal beyond binary outcomes)
- IRT models win on population statistics (regression to the mean, difficulty calibration)
- This motivates hybrid approaches: LLM features + parametric models

This decomposition of predictive signal into a **style-specific component** (where LLMs dominate) and a **population-statistical component** (where parametric models dominate) would be the core contribution.
