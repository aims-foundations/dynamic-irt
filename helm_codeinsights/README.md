# HELM CodeInsights Integration

Custom scenarios, metrics, and run specs for evaluating LLMs on CodeInsights benchmarks using [HELM](https://github.com/stanford-crfm/helm).

## Setup

1. Install HELM in editable mode:
```bash
git clone https://github.com/stanford-crfm/helm.git
cd helm
pip install -e .
```

2. Copy the custom CodeInsights files into HELM:
```bash
cp helm_codeinsights/scenarios/*.py helm/src/helm/benchmark/scenarios/
cp helm_codeinsights/metrics/*.py helm/src/helm/benchmark/metrics/
cp helm_codeinsights/run_specs/*.py helm/src/helm/benchmark/run_specs/
cp helm_codeinsights/presentation/*.conf helm/src/helm/benchmark/presentation/
```

3. Register the scenarios and metrics in HELM's `__init__.py` files as needed.

## Contents

- **scenarios/** - Custom scenario classes for student coding evaluation
- **metrics/** - Custom metrics for code correctness, efficiency, and edge cases
- **run_specs/** - Run specifications for CodeInsights benchmarks
- **presentation/** - Configuration for result presentation
