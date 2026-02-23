#!/bin/bash
# Wait for S2 to finish, then run S3 and S4 sequentially
set -e

PYTHON="/lfs/local/0/sttruong/miniconda3/envs/codeinsight/bin/python"
SCRIPT="/lfs/skampere1/0/sttruong/support/codeinsight/CodeInsights/llm_simulator/helm_codeinsights/run_evaluation.py"
OUTPUT="/lfs/skampere1/0/sttruong/support/codeinsight/CodeInsights/llm_simulator/helm_codeinsights/helm_results/evaluation"
export HF_HUB_CACHE="/lfs/skampere1/0/shared_hf_cache"

S2_RESULTS="$OUTPUT/glm/S2/results.json"

echo "$(date): Waiting for S2 to finish (target: 1000 instances)..."

while true; do
    if [ -f "$S2_RESULTS" ]; then
        COUNT=$($PYTHON -c "import json; print(len(json.load(open('$S2_RESULTS'))))" 2>/dev/null || echo "0")
        echo "$(date): S2 progress: $COUNT/1000"
        if [ "$COUNT" -ge 1000 ]; then
            echo "$(date): S2 complete!"
            break
        fi
    fi
    sleep 300  # check every 5 minutes
done

echo ""
echo "$(date): Starting S3 (Student Mistake, 1000 instances)..."
$PYTHON $SCRIPT --model glm --scenario S3 --max-instances 1000 --output $OUTPUT 2>&1

echo ""
echo "$(date): Starting S4 (Code Efficiency, 1000 instances)..."
$PYTHON $SCRIPT --model glm --scenario S4 --max-instances 1000 --output $OUTPUT 2>&1

echo ""
echo "$(date): ALL SCENARIOS COMPLETE!"
