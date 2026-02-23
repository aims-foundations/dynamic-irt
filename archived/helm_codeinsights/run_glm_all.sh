#!/bin/bash
# Start vLLM server, wait for ready, then run S3 and S4
set -e

PYTHON="/lfs/local/0/sttruong/miniconda3/envs/codeinsight/bin/python"
SCRIPT="/lfs/skampere1/0/sttruong/support/codeinsight/CodeInsights/llm_simulator/helm_codeinsights/run_evaluation.py"
OUTPUT="/lfs/skampere1/0/sttruong/support/codeinsight/CodeInsights/llm_simulator/helm_codeinsights/helm_results/evaluation"
export HF_HUB_CACHE="/lfs/skampere1/0/shared_hf_cache"

echo "$(date): Starting vLLM server on 4 GPUs (0-3)..."
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model QuantTrio/GLM-4.7-AWQ \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8240 \
    --served-model-name glm-4.7-awq \
    --trust-remote-code \
    --quantization awq &
VLLM_PID=$!
echo "$(date): vLLM PID: $VLLM_PID"

# Wait for server ready
echo "$(date): Waiting for vLLM to be ready..."
while ! curl -s http://localhost:8240/health > /dev/null 2>&1; do
    sleep 10
    echo "$(date): Still waiting..."
    # Check if vLLM crashed
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "$(date): vLLM process died!"
        exit 1
    fi
done
echo "$(date): vLLM ready!"

echo ""
echo "$(date): Starting S4 (Code Efficiency, 1000 instances)..."
$PYTHON $SCRIPT --model glm --scenario S4 --max-instances 1000 --output $OUTPUT 2>&1

echo ""
echo "$(date): ALL SCENARIOS COMPLETE!"
echo "$(date): Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "$(date): Done. GPUs freed."
