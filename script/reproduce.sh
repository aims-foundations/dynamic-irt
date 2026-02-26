#!/bin/bash
# reproduce.sh - Full reproducibility pipeline for EduCodeSim paper
# Generates all CSV data and plots for open-source models (no API costs)
#
# Usage:
#   ./reproduce.sh              # Run full pipeline
#   ./reproduce.sh --skip-llm   # Skip LLM evaluation (use existing results)
#   ./reproduce.sh --help       # Show help

set -e  # Exit on error

# ─── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"
OUTPUT_DIR="${BASE_DIR}/reproducibility_results"
CONDA_ENV="codeinsight"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^${CONDA_ENV} "; then
        echo "[INFO] Activating conda environment: ${CONDA_ENV}"
        conda activate ${CONDA_ENV}
    fi
fi

# Set HuggingFace cache to avoid permission issues
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" 2>/dev/null || true

# Open-source models (no API costs)
MODELS=("google_gemma-3-27b-it" "meta_llama-3.1-8b-instruct" "qwen_qwen2.5-14b-instruct")
SCENARIOS=("S1" "S2" "S3" "S4")

# Parse arguments
SKIP_LLM=false
SKIP_PLOTS=false
MODELS_TO_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-llm)
            SKIP_LLM=true
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS=true
            shift
            ;;
        --models)
            MODELS_TO_RUN="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./reproduce.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-llm     Skip LLM evaluation step (use existing results)"
            echo "  --skip-plots   Skip plot generation step"
            echo "  --models LIST  Comma-separated list of models to run"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./reproduce.sh                          # Run full pipeline"
            echo "  ./reproduce.sh --skip-llm               # Skip LLM evaluation"
            echo "  ./reproduce.sh --models llama,qwen      # Run specific models"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ─── Helper Functions ──────────────────────────────────────────────────────────
log_step() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  STEP $1: $2"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
}

log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[SUCCESS] $1"
}

log_warning() {
    echo "[WARNING] $1"
}

log_error() {
    echo "[ERROR] $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed or not in PATH"
        return 1
    fi
    return 0
}

# ─── Environment Checks ────────────────────────────────────────────────────────
log_step "0" "Environment Checks"

log_info "Checking Python..."
check_command python || exit 1
PYTHON_VERSION=$(python --version 2>&1)
log_info "Found: $PYTHON_VERSION"

log_info "Checking R (optional for TRT analysis)..."
if check_command Rscript; then
    R_VERSION=$(Rscript --version 2>&1 | head -1)
    log_info "Found: R available"
else
    log_warning "R not found. TRT analysis will be skipped."
fi

log_info "Checking for GPU (optional for LLM evaluation)..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    log_info "Found GPU: $GPU_INFO"
else
    log_warning "No GPU detected. LLM evaluation may be slow."
fi

# ─── Create Output Directories ─────────────────────────────────────────────────
log_info "Creating output directories..."
mkdir -p "${OUTPUT_DIR}/scenario_data"
mkdir -p "${OUTPUT_DIR}/scenario_results"
mkdir -p "${OUTPUT_DIR}/metrics"
mkdir -p "${OUTPUT_DIR}/psychometrics/ability"
mkdir -p "${OUTPUT_DIR}/psychometrics/difficulty"
mkdir -p "${OUTPUT_DIR}/psychometrics/correlations"
mkdir -p "${OUTPUT_DIR}/plots"
log_success "Output directories created at: ${OUTPUT_DIR}"

# ─── Step 1: Data Preprocessing ────────────────────────────────────────────────
log_step "1" "Data Preprocessing"
log_info "Downloading data from HuggingFace and generating scenario datasets..."

(cd "${BASE_DIR}/llm_simulator" && python data_preprocessing.py)

# Copy generated files to output directory
if [ -d "${BASE_DIR}/llm_simulator/data" ]; then
    cp -r "${BASE_DIR}/llm_simulator/data/"* "${OUTPUT_DIR}/scenario_data/" 2>/dev/null || true
fi

log_success "Data preprocessing complete. Scenario CSVs generated."

# ─── Step 2: Run Open-Source Models ────────────────────────────────────────────
if [ "$SKIP_LLM" = false ]; then
    log_step "2" "Open-Source Model Evaluation (vLLM)"
    log_info "Running LLM evaluation on GPU..."
    log_info "This step may take 10-15 hours per model."

    if [ -f "${BASE_DIR}/llm_simulator/run_single_turn.py" ]; then
        if [ -n "$MODELS_TO_RUN" ]; then
            (cd "${BASE_DIR}/llm_simulator" && python run_single_turn.py --models "$MODELS_TO_RUN" --output "${OUTPUT_DIR}")
        else
            # Default: open-source models only (no API keys required)
            (cd "${BASE_DIR}/llm_simulator" && python run_single_turn.py --models "llama,gemma,qwen" --output "${OUTPUT_DIR}")
        fi
        log_success "LLM evaluation complete."
    else
        log_warning "llm_simulator/run_single_turn.py not found. Skipping LLM evaluation."
        log_info "Checking for existing results..."
    fi
else
    log_step "2" "Open-Source Model Evaluation (SKIPPED)"
    log_info "Using existing LLM results from scenario_results/"
fi

# ─── Step 3: Compute Metrics ───────────────────────────────────────────────────
log_step "3" "Compute Metrics"

# Check if any LLM results exist before computing metrics
if [ -d "${OUTPUT_DIR}/scenario_results" ] && [ "$(ls -A ${OUTPUT_DIR}/scenario_results 2>/dev/null)" ]; then
    log_info "Computing functional correctness, AST similarity, CodeBERT similarity..."
    (cd "${BASE_DIR}/data_analysis" && python compute_metrics.py) || {
        log_warning "Metrics computation failed. This may be due to missing LLM results."
    }

    # Copy results to output directory
    if [ -f "${BASE_DIR}/data_analysis/all_results.json" ]; then
        cp "${BASE_DIR}/data_analysis/all_results.json" "${OUTPUT_DIR}/metrics/"
    fi
    log_success "Metrics computation complete."
else
    log_warning "No LLM results found in ${OUTPUT_DIR}/scenario_results"
    log_info "Skipping metrics computation. Run LLM evaluation first."
fi

# ─── Step 4: Psychometric Analysis ─────────────────────────────────────────────
log_step "4" "Psychometric Analysis (IRT)"

# Check if any LLM results exist before running psychometric analysis
if [ -d "${OUTPUT_DIR}/scenario_results" ] && [ "$(ls -A ${OUTPUT_DIR}/scenario_results 2>/dev/null)" ]; then
    log_info "Computing ability and difficulty parameters using Item Response Theory..."
    python "${BASE_DIR}/data_analysis/psychometrics_metrics.py" --data "${OUTPUT_DIR}" --output "${OUTPUT_DIR}/psychometrics" || {
        log_warning "Psychometric analysis failed. This may be due to missing LLM results."
    }
    log_success "Psychometric analysis complete."
else
    log_warning "No LLM results found. Skipping psychometric analysis."
fi

# ─── Step 5: Testlet Analysis (R) ──────────────────────────────────────────────
log_step "5" "Testlet Response Theory Analysis (R)"

if command -v Rscript &> /dev/null; then
    if [ -f "${BASE_DIR}/dynamic_irt/codeinsights_testlet_analysis.R" ]; then
        log_info "Running Bayesian TRT analysis..."
        log_info "This may take 1-2 hours for MCMC sampling."
        Rscript "${BASE_DIR}/dynamic_irt/codeinsights_testlet_analysis.R" || {
            log_warning "TRT analysis failed. Check R dependencies (brms, dplyr, tidyr, ggplot2, pROC)."
        }
        log_success "TRT analysis complete."
    else
        log_warning "codeinsights_testlet_analysis.R not found. Skipping."
    fi
else
    log_warning "R not available. Skipping TRT analysis."
fi

# ─── Step 6: Generate Plots ────────────────────────────────────────────────────
if [ "$SKIP_PLOTS" = false ]; then
    log_step "6" "Generate Publication Figures"

    if [ -f "${BASE_DIR}/data_analysis/generate_plots.py" ]; then
        log_info "Generating scenario result plots and psychometric figures..."
        python "${BASE_DIR}/data_analysis/generate_plots.py" --output "${OUTPUT_DIR}/plots" --metrics "${OUTPUT_DIR}/metrics"
        log_success "Plots generated in ${OUTPUT_DIR}/plots/"
    else
        log_warning "generate_plots.py not found. Skipping plot generation."
    fi
else
    log_step "6" "Generate Publication Figures (SKIPPED)"
fi

# ─── Step 7: Generate Reproducibility Report ───────────────────────────────────
log_step "7" "Generate Reproducibility Report"

if [ -f "${OUTPUT_DIR}/generate_report.py" ]; then
    log_info "Comparing results against paper values..."
    python "${OUTPUT_DIR}/generate_report.py" \
        --metrics "${OUTPUT_DIR}/metrics/all_results.json" \
        --correlations "${OUTPUT_DIR}/psychometrics/correlations/all_correlations.json" \
        --output "${OUTPUT_DIR}/reproducibility_report.md"
    log_success "Reproducibility report generated."
else
    log_warning "generate_report.py not found. Skipping report generation."
fi

# ─── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  REPRODUCIBILITY PIPELINE COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  - Scenario data:    ${OUTPUT_DIR}/scenario_data/"
echo "  - Metrics:          ${OUTPUT_DIR}/metrics/all_results.json"
echo "  - Psychometrics:    ${OUTPUT_DIR}/psychometrics/"
echo "  - Plots:            ${OUTPUT_DIR}/plots/"
echo "  - Report:           ${OUTPUT_DIR}/reproducibility_report.md"
echo ""

if [ -f "${OUTPUT_DIR}/reproducibility_report.md" ]; then
    echo "To view the reproducibility report:"
    echo "  cat ${OUTPUT_DIR}/reproducibility_report.md"
fi

echo ""
log_success "Done!"
