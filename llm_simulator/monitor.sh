#!/bin/bash
# Monitor shard progress every 15 minutes.
# Run in tmux: tmux new -d -s monitor 'bash llm_simulator/monitor.sh'

LOG_DIR="$HOME/support/codeinsight/llm_simulator/results/llm_eval"
STATUS="$LOG_DIR/monitor_status.log"

while true; do
    echo "========== $(date) ==========" >> "$STATUS"

    for i in 0 1; do
        LOG="$LOG_DIR/shard${i}.log"
        echo "--- Shard $i ---" >> "$STATUS"
        if [ ! -f "$LOG" ]; then
            echo "  No log file" >> "$STATUS"
            continue
        fi

        # Check for errors
        ERRORS=$(grep -c 'ERROR' "$LOG" 2>/dev/null)
        if [ "$ERRORS" -gt 0 ]; then
            echo "  *** $ERRORS ERROR(s) detected! ***" >> "$STATUS"
            grep 'ERROR' "$LOG" | tail -3 >> "$STATUS"
        fi

        # Latest progress
        grep -E '(Chunk.*items|Attempt.*result|sub-batch|Saved|Done)' "$LOG" | tail -5 >> "$STATUS"
        echo "" >> "$STATUS"
    done

    # Check servers are alive
    for port in 8000 8001; do
        RESP=$(curl -s -H 'Authorization: Bearer EMPTY' "http://localhost:${port}/v1/models" 2>/dev/null | head -1)
        if echo "$RESP" | grep -q 'GLM'; then
            echo "Server :${port} OK" >> "$STATUS"
        else
            echo "*** Server :${port} DOWN ***" >> "$STATUS"
        fi
    done

    echo "" >> "$STATUS"

    # Also print to stdout for tmux viewing
    tail -30 "$STATUS"
    echo ""
    echo "Next check in 15 minutes..."
    sleep 900
done
