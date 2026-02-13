# CIRT vs GPIRT Model Comparison

This folder contains scripts to compare CIRT and GPIRT models on the same dataset using quantitative metrics.

## Quick Start

Run the full comparison pipeline (trains both models if needed, then compares):

```bash
cd CodeInsights/dynamic_irt
./run_comparison.sh
```

## Manual Usage

### 1. Train CIRT Model

```bash
python cirt/continuous_irt.py \
    --course_name dsa_hk231 \
    --seed 42 \
    --concentration 10.0 \
    --epochs 10000
```

Output: `results/dsa_hk231_10.0/model.pkl`

### 2. Train GPIRT Model

```bash
python gpirt/inference.py \
    --course_name dsa_hk231 \
    --seed 42 \
    --kernel RBF \
    --length_scale 1.0 \
    --fitting_method hmc \
    --max_epoch 1000 \
    --warmup_steps 200
```

Output: `results/dsa_hk231_s42_D1_PL1_hmc_kernelRBF_ls1.0/{ability.pt, difficulty.pt}`

### 3. Run Comparison

```bash
python compare_cirt_gpirt.py \
    --course_name dsa_hk231 \
    --seed 42 \
    --cirt_concentration 10.0 \
    --gpirt_kernel RBF \
    --gpirt_length_scale 1.0
```

## Comparison Metrics

The comparison script evaluates both models on:

### Predictive Performance
- **Log-Likelihood**: How well the model predicts observed responses (higher is better)
- **AUC (Area Under ROC Curve)**: Classification performance (higher is better, max 1.0)
- **Accuracy**: Proportion of correct predictions (higher is better)
- **F1 Score**: Harmonic mean of precision and recall (higher is better)
- **Precision & Recall**: Classification quality metrics

### Model Calibration
- **Goodness-of-Fit (GoF)**: Compares theoretical vs empirical success rates across ability bins
  - Bins students by ability level
  - Computes difference between model predictions and actual performance
  - Reports mean ± std (closer to 1.0 is better)

## Output

Results are saved in `comparison_results/<course_name>_<timestamp>/`:

- `comparison.csv` - Metrics in tabular format
- `comparison.json` - Full results with configuration
- `metrics_comparison.png` - Visual comparison bar plots

### Example Output

```
================================================================================
COMPARISON SUMMARY
================================================================================
  model  log_likelihood    auc  accuracy     f1  precision  recall  gof_mean  gof_std
   CIRT         -0.4523  0.892     0.847  0.839      0.851   0.828     0.912    0.043
  GPIRT         -0.3891  0.908     0.865  0.859      0.863   0.855     0.931    0.037
```

## Interpretation Guide

### When CIRT is Better:
- Faster training time
- Simpler interpretation (parametric sigmoid curves)
- Similar or better log-likelihood
- Good for exploratory analysis

### When GPIRT is Better:
- Higher AUC/accuracy
- Better model calibration (higher GoF)
- Captures complex temporal patterns
- Better uncertainty quantification
- More flexible for research

## Configuration Options

```bash
python compare_cirt_gpirt.py \
    --course_name dsa_hk231 \          # Dataset name
    --seed 42 \                         # Random seed
    --cirt_concentration 10.0 \         # CIRT Beta concentration parameter
    --gpirt_kernel RBF \                # GPIRT kernel (RBF or Matern)
    --gpirt_length_scale 1.0 \          # GPIRT temporal correlation length
    --output_folder comparison_results  # Output directory
```

## Dataset Requirements

Both models expect data from HuggingFace in the format:
- Repository: `stair-lab/{course_name}`
- Files: `correctness_matrix.pkl` (shape: N × Q × T)
  - N = number of students
  - Q = number of questions
  - T = number of time points

## Troubleshooting

### "CIRT model not found"
Train CIRT first:
```bash
python cirt/continuous_irt.py --course_name dsa_hk231
```

### "GPIRT model not found"
Train GPIRT first (takes longer, ~30min - 2 hours):
```bash
python gpirt/inference.py --course_name dsa_hk231 --max_epoch 1000
```

### CUDA out of memory
Reduce batch size or use CPU:
```bash
# Training will automatically use CPU if CUDA unavailable
```

## Citation

If you use this comparison in your research:

```bibtex
@article{codeinsights2024,
  title={CodeInsights: Understanding Learning Dynamics with Psychometric Models},
  author={...},
  year={2024}
}
```
