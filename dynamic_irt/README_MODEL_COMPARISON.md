
# Dynamic IRT Models Comparison Framework

This framework provides comprehensive comparison tools for all dynamic IRT models in the CodeInsights project.

## Available Models

### 1. **CIRT** (Continuous IRT)
- **Type**: Parametric, deterministic
- **Approach**: Sigmoid learning curves with Beta distribution
- **Parameters**: θ₀ (learning rate), θ₁ (asymptotic ability), z (difficulty)
- **Training**: Fast (~10 minutes for 10K epochs)
- **Best for**: Quick exploratory analysis, interpretable parameters

### 2. **GPIRT** (Gaussian Process IRT)
- **Type**: Non-parametric, Bayesian
- **Approach**: GP priors with MCMC inference (HMC/NUTS)
- **Parameters**: Ability distributions, difficulty distributions
- **Training**: Slow (~1-2 hours for 1000 samples)
- **Best for**: Uncertainty quantification, flexible temporal patterns

### 3. **Elo** (Elo Rating System)
- **Type**: Online learning, dynamic updates
- **Approach**: Iterative rating updates based on performance
- **Parameters**: Student abilities, item difficulties (dynamically updated)
- **Training**: Very fast (linear in data size)
- **Best for**: Online/streaming settings, real-time updates

### 4. **RSSM** (Recurrent State-Space Model)
- **Type**: Deep learning, sequence modeling
- **Approach**: RNN-based with code embeddings
- **Parameters**: Neural network weights
- **Training**: Moderate (~30 minutes with GPU)
- **Best for**: Code-aware predictions, representation learning

## Quick Start: Compare All Models

```bash
cd CodeInsights/dynamic_irt

# Compare all models (will use existing trained models)
python compare_all_models.py --course_name dsa_hk231

# Compare specific models only
python compare_all_models.py --models cirt gpirt

# Compare with custom hyperparameters
python compare_all_models.py \
    --course_name dsa_hk231 \
    --cirt_concentration 10.0 \
    --gpirt_kernel RBF \
    --gpirt_length_scale 1.0
```

## Step-by-Step: Train and Compare

### Option 1: Automatic (Recommended)

Use the provided script to train all models and compare:

```bash
# This will train all models if needed, then compare them
./run_all_models_comparison.sh
```

### Option 2: Manual Training

Train each model individually, then compare:

```bash
# 1. Train CIRT (~10 minutes)
python cirt/continuous_irt.py \
    --course_name dsa_hk231 \
    --concentration 10.0 \
    --epochs 10000

# 2. Train GPIRT (~1-2 hours)
python gpirt/inference.py \
    --course_name dsa_hk231 \
    --kernel RBF \
    --length_scale 1.0 \
    --fitting_method hmc \
    --max_epoch 1000 \
    --warmup_steps 200

# 3. Train Elo (~5 minutes)
python -m dynamic_irt.elo.main_elo

# 4. Train RSSM (~30 minutes)
python rssm/main_rssm.py \
    --course_name dsa_hk231 \
    --epochs 100

# 5. Compare all models
python compare_all_models.py --course_name dsa_hk231
```

## Comparison Metrics

All models are evaluated on the same test set (20% holdout) using:

### Predictive Performance
- **Log-Likelihood**: Higher is better (measures probability calibration)
- **AUC**: Higher is better (0.5 = random, 1.0 = perfect)
- **Accuracy**: Proportion of correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / predicted positives
- **Recall**: True positives / actual positives

### Model-Specific
- **Goodness-of-Fit** (CIRT, GPIRT): Alignment between predicted and observed probabilities
- **Training Time**: Computational efficiency

## Output Files

Results are saved in `comparison_results/all_models_<course>_<timestamp>/`:

```
comparison_results/
└── all_models_dsa_hk231_20260213_120000/
    ├── comparison.csv               # Metrics table
    ├── comparison.json              # Full results with config
    ├── model_comparison_bars.png    # Bar chart comparison
    ├── model_comparison_radar.png   # Radar chart
    └── model_comparison_heatmap.png # Heatmap visualization
```

### Example Output

```
================================================================================
COMPARISON SUMMARY
================================================================================
  model  auc  accuracy    f1  precision  recall  log_likelihood
   CIRT  0.892     0.847  0.839      0.851   0.828         -0.452
  GPIRT  0.908     0.865  0.859      0.863   0.855         -0.389
    Elo  0.875     0.831  0.824      0.819   0.829         -0.498
   RSSM  0.901     0.854  0.847      0.856   0.838         -0.421

🏆 Best Model (by AUC): GPIRT (AUC=0.908)
```

## Model Selection Guide

### Choose **CIRT** when:
- ✅ You need fast training and inference
- ✅ You want interpretable parametric curves
- ✅ You're doing exploratory analysis
- ✅ Your data follows sigmoid learning patterns

### Choose **GPIRT** when:
- ✅ You need uncertainty quantification
- ✅ You have complex temporal patterns
- ✅ You want the most flexible model
- ✅ Computational cost is not a concern
- ✅ You're doing research requiring Bayesian inference

### Choose **Elo** when:
- ✅ You need online/real-time updates
- ✅ You have streaming data
- ✅ You want the fastest training
- ✅ You need simple, interpretable ratings
- ✅ You're building a production system

### Choose **RSSM** when:
- ✅ You have access to code embeddings
- ✅ You want to leverage code semantics
- ✅ You need sequence modeling capabilities
- ✅ You have GPU resources
- ✅ You're interested in representation learning

## Performance Comparison Summary

| Model | Training Speed | Inference Speed | Accuracy | Flexibility | Interpretability | Uncertainty |
|-------|---------------|-----------------|----------|-------------|------------------|-------------|
| CIRT  | ⚡⚡⚡ Fast    | ⚡⚡⚡ Fast     | 🟡 Good  | 🔴 Low      | ✅ High          | ❌ No       |
| GPIRT | 🔴 Slow       | 🟡 Medium       | ✅ Best  | ✅ High     | 🟡 Medium        | ✅ Yes      |
| Elo   | ⚡⚡⚡ Fastest | ⚡⚡⚡ Fastest  | 🟡 Good  | 🟡 Medium   | ✅ High          | ❌ No       |
| RSSM  | 🟡 Medium     | ⚡⚡ Fast       | 🟢 Great | 🟢 High     | 🔴 Low           | ❌ No       |

## Advanced Usage

### Pairwise Comparison (CIRT vs GPIRT only)

```bash
python compare_cirt_gpirt.py --course_name dsa_hk231
```

### Custom Model Paths

```bash
python compare_all_models.py \
    --course_name dsa_hk231 \
    --cirt_folder custom_results/cirt_model \
    --gpirt_folder custom_results/gpirt_model
```

### Statistical Significance Testing

The comparison framework computes metrics on a holdout test set. For statistical testing:

```python
# In Python after running comparison
import pandas as pd
from scipy import stats

results = pd.read_csv('comparison_results/.../comparison.csv')

# Bootstrap confidence intervals
# (Add custom statistical testing as needed)
```

## Visualization Examples

### 1. Bar Chart
Shows absolute metric values for easy comparison across models.

### 2. Radar Chart
Displays multiple metrics simultaneously on a circular plot, making trade-offs visible.

### 3. Heatmap
Color-coded performance matrix showing which models excel at which metrics.

## Troubleshooting

### "Model not found" errors

Train the model first:
```bash
# See training commands above for each model
```

### Different data formats

Some models require specific preprocessing:
- **CIRT/GPIRT**: Use correctness_matrix.pkl
- **Elo**: May use different CSV format
- **RSSM**: Requires code embeddings

### Memory issues

```bash
# Use smaller batch sizes or reduce data
# For GPIRT, reduce max_epoch or use thinning
python gpirt/compute_metrics.py --thinning 5
```

## Citation

If you use this comparison framework in your research:

```bibtex
@article{codeinsights2024,
  title={CodeInsights: Comparative Analysis of Dynamic IRT Models for Learning Prediction},
  author={...},
  journal={...},
  year={2024}
}
```

## Further Reading

- [CIRT Model Details](cirt/README.md)
- [GPIRT Model Details](gpirt/README.md)
- [Elo Model Details](elo/README.md)
- [RSSM Model Details](rssm/README.md)

## Contributing

To add a new model to the comparison framework:

1. Create a new evaluator class inheriting from `ModelEvaluator`
2. Implement `load_model()` and `predict()` methods
3. Add the model to the comparison script
4. Update this README

See `compare_all_models.py` for implementation examples.
