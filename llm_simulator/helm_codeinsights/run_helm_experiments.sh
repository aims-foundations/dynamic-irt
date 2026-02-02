#!/bin/bash
# run_helm_experiments.sh - Run CodeInsights HELM experiments with vLLM
#
# This script:
# 1. Starts vLLM servers for open-source models
# 2. Runs HELM benchmarks with CodeInsights scenarios
# 3. Collects and exports results
#
# Usage:
#   ./run_helm_experiments.sh              # Run all models
#   ./run_helm_experiments.sh --model llama  # Run specific model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELM_DIR="${SCRIPT_DIR}/../helm"
OUTPUT_DIR="${SCRIPT_DIR}/helm_results"
LOG_DIR="${OUTPUT_DIR}/logs"

# Model configurations
# Format: model_name:port:gpus:tensor_parallel_size
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct:8001:4:1"
    "Qwen/Qwen2.5-14B-Instruct:8003:5:1"
    "google/gemma-2-27b-it:8002:6,7:2"
)

# HuggingFace cache
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" 2>/dev/null || true

# Parse arguments
SELECTED_MODEL=""
SKIP_SERVER=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            SELECTED_MODEL="$2"
            shift 2
            ;;
        --skip-server)
            SKIP_SERVER=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_helm_experiments.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model NAME     Run only specific model (llama, qwen, gemma)"
            echo "  --skip-server    Skip starting vLLM servers (use existing)"
            echo "  --skip-eval      Skip HELM evaluation (only start servers)"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  CodeInsights HELM Experiments"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Function to start a vLLM server
start_vllm_server() {
    local model_spec="$1"
    IFS=':' read -r model_name port gpus tp_size <<< "$model_spec"

    local short_name=$(basename "$model_name" | tr '[:upper:]' '[:lower:]')
    local log_file="${LOG_DIR}/vllm_${short_name}.log"

    echo "[INFO] Starting vLLM server for ${model_name} on GPU(s) ${gpus}..."

    # Check if server is already running
    if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "[INFO] Server already running on port ${port}"
        return 0
    fi

    # Start vLLM server in background
    CUDA_VISIBLE_DEVICES="${gpus}" python -m vllm.entrypoints.openai.api_server \
        --model "${model_name}" \
        --port "${port}" \
        --tensor-parallel-size "${tp_size}" \
        --max-model-len 8192 \
        --trust-remote-code \
        --dtype auto \
        > "${log_file}" 2>&1 &

    local pid=$!
    echo "${pid}" > "${LOG_DIR}/vllm_${short_name}.pid"

    echo "[INFO] Server PID: ${pid}, log: ${log_file}"

    # Wait for server to be ready
    echo "[INFO] Waiting for server to be ready..."
    local max_wait=300
    local waited=0
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            echo "[ERROR] Server failed to start within ${max_wait} seconds"
            cat "${log_file}"
            return 1
        fi
        echo "[INFO] Waiting... (${waited}s)"
    done

    echo "[SUCCESS] Server ready on port ${port}"
    return 0
}

# Function to stop vLLM servers
stop_vllm_servers() {
    echo "[INFO] Stopping vLLM servers..."
    for pid_file in "${LOG_DIR}"/vllm_*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                echo "[INFO] Stopped server PID: ${pid}"
            fi
            rm -f "$pid_file"
        fi
    done
}

# Trap to clean up on exit
trap stop_vllm_servers EXIT

# Start vLLM servers if not skipped
if [ "$SKIP_SERVER" = false ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  Starting vLLM Servers"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""

    for model_spec in "${MODELS[@]}"; do
        model_name=$(echo "$model_spec" | cut -d: -f1)
        short_name=$(basename "$model_name" | tr '[:upper:]' '[:lower:]')

        # Skip if not selected model
        if [ -n "$SELECTED_MODEL" ] && [[ ! "$short_name" =~ "$SELECTED_MODEL" ]]; then
            continue
        fi

        start_vllm_server "$model_spec" || {
            echo "[ERROR] Failed to start ${model_name}"
            continue
        }
    done
fi

# Run HELM evaluation if not skipped
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  Running HELM Evaluation"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""

    # Copy vLLM model deployments to HELM
    if [ -f "${SCRIPT_DIR}/vllm_model_deployments.yaml" ]; then
        mkdir -p "${HELM_DIR}/prod_env"
        cp "${SCRIPT_DIR}/vllm_model_deployments.yaml" "${HELM_DIR}/prod_env/model_deployments.yaml"
        echo "[INFO] Copied vLLM model deployments to HELM"
    fi

    # Create run configuration
    cat > "${OUTPUT_DIR}/run_specs.conf" << 'EOF'
entries: [
    # LLaMA-3.1-8B
    {description: "codeinsights_correct_code:model=meta-llama/Llama-3.1-8B-Instruct,tpr=0.0,num_testcases=-1", priority: 1},
    {description: "codeinsights_student_coding:model=meta-llama/Llama-3.1-8B-Instruct,tpr=0.0,num_testcases=-1", priority: 1},
    {description: "codeinsights_student_mistake:model=meta-llama/Llama-3.1-8B-Instruct,tpr=0.0,num_testcases=-1", priority: 1},
    {description: "codeinsights_code_efficiency:model=meta-llama/Llama-3.1-8B-Instruct,tpr=0.0,num_testcases=-1", priority: 1},

    # Qwen-2.5-14B
    {description: "codeinsights_correct_code:model=Qwen/Qwen2.5-14B-Instruct,tpr=0.0,num_testcases=-1", priority: 1},
    {description: "codeinsights_student_coding:model=Qwen/Qwen2.5-14B-Instruct,tpr=0.0,num_testcases=-1", priority: 1},
    {description: "codeinsights_student_mistake:model=Qwen/Qwen2.5-14B-Instruct,tpr=0.0,num_testcases=-1", priority: 1},
    {description: "codeinsights_code_efficiency:model=Qwen/Qwen2.5-14B-Instruct,tpr=0.0,num_testcases=-1", priority: 1},

    # Gemma-2-27B
    {description: "codeinsights_correct_code:model=google/gemma-2-27b-it,tpr=0.0,num_testcases=-1", priority: 1},
    {description: "codeinsights_student_coding:model=google/gemma-2-27b-it,tpr=0.0,num_testcases=-1", priority: 1},
    {description: "codeinsights_student_mistake:model=google/gemma-2-27b-it,tpr=0.0,num_testcases=-1", priority: 1},
    {description: "codeinsights_code_efficiency:model=google/gemma-2-27b-it,tpr=0.0,num_testcases=-1", priority: 1},
]
EOF

    echo "[INFO] Running HELM benchmark..."

    cd "${HELM_DIR}"
    helm-run \
        --conf-paths "${OUTPUT_DIR}/run_specs.conf" \
        --suite codeinsights \
        --output-path "${OUTPUT_DIR}/benchmark_output" \
        --num-threads 1 \
        --max-eval-instances 100 \
        2>&1 | tee "${LOG_DIR}/helm_run.log"

    echo "[INFO] Summarizing results..."
    helm-summarize \
        --suite codeinsights \
        --output-path "${OUTPUT_DIR}/benchmark_output" \
        2>&1 | tee "${LOG_DIR}/helm_summarize.log"

    echo ""
    echo "[SUCCESS] HELM evaluation complete!"
    echo "Results saved to: ${OUTPUT_DIR}/benchmark_output"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Experiment Complete"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"
