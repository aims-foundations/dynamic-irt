#!/bin/bash
# Setup script for HELM with CodeInsights custom scenarios and metrics
# Usage: ./setup.sh [HELM_DIR]
#   HELM_DIR: Directory to clone HELM into (default: ./helm)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELM_DIR="${1:-$SCRIPT_DIR/../helm}"

echo "==> Cloning HELM from stanford-crfm/helm..."
if [ -d "$HELM_DIR" ]; then
    echo "    HELM directory already exists at $HELM_DIR, skipping clone"
else
    git clone https://github.com/stanford-crfm/helm.git "$HELM_DIR"
fi

echo "==> Installing HELM in editable mode..."
pip install -e "$HELM_DIR"

echo "==> Copying CodeInsights scenarios..."
cp "$SCRIPT_DIR"/scenarios/*.py "$HELM_DIR/src/helm/benchmark/scenarios/"

echo "==> Copying CodeInsights metrics..."
cp "$SCRIPT_DIR"/metrics/*.py "$HELM_DIR/src/helm/benchmark/metrics/"

echo "==> Copying CodeInsights run specs..."
cp "$SCRIPT_DIR"/run_specs/*.py "$HELM_DIR/src/helm/benchmark/run_specs/"

echo "==> Copying CodeInsights presentation config..."
cp "$SCRIPT_DIR"/presentation/*.conf "$HELM_DIR/src/helm/benchmark/presentation/"

echo "==> Done! HELM is installed at: $HELM_DIR"
echo "    Run 'helm-run' to start evaluations."
