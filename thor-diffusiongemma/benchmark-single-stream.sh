#!/bin/bash
# benchmark-single-stream.sh
# Measures single-stream tok/s + wall-clock on DiffusionGemma. Uses /v1/chat/completions
# (raw /v1/completions is incoherent on this thinking chat model). Model name "/model"
# (server does not set --served-model-name).
#
# Usage:
#   ./benchmark-single-stream.sh baseline    # measure server on :8004
#   ./benchmark-single-stream.sh thinking    # thinking on vs off
#   ./benchmark-single-stream.sh prefixcache # TTFT with/without prefix hit

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/thor-diffusiongemma/results"
PORT="${DG_PORT:-8004}"
RUNS="${DG_RUNS:-4}"
mkdir -p "$RESULTS_DIR"
SUBCMD="${1:-baseline}"

wait_ready() {
    local W=0
    until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
        sleep 10; W=$((W+10))
        [ $W -ge 900 ] && { echo "TIMEOUT waiting for server on :$PORT"; exit 1; }
        [ $((W % 60)) -eq 0 ] && echo "  ...waiting ${W}s..."
    done
    echo "  server ready"
}

chat_request() {  # thinking content maxtok
    curl -sf "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"/model\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":$3,\"temperature\":0.0,\"chat_template_kwargs\":{\"enable_thinking\":$1}}" 2>/dev/null
}

measure() {  # label thinking prompt maxtok
    local LABEL="$1" THINKING="$2" PROMPT="$3" MAXTOK="$4"
    local TOTAL_TOK=0 TOTAL_SEC=0
    echo "  [$LABEL] thinking=$THINKING max_tokens=$MAXTOK"
    for i in $(seq 1 "$RUNS"); do
        local S E R TOK THINK_TOK WALL_MS
        S=$(date +%s%3N); R=$(chat_request "$THINKING" "$PROMPT" "$MAXTOK"); E=$(date +%s%3N)
        WALL_MS=$((E - S))
        TOK=$(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo 0)
        THINK_TOK=$(echo "$R" | python3 -c "import json,sys;m=json.load(sys.stdin)['choices'][0]['message'];print(len((m.get('reasoning_content') or '').split()))" 2>/dev/null || echo 0)
        if [ "$TOK" -gt 0 ]; then
            local TPS; TPS=$(awk "BEGIN{printf \"%.1f\", $TOK/($WALL_MS/1000)}")
            echo "    run $i: ${TOK} tok (${THINK_TOK} reasoning words) in ${WALL_MS}ms = ${TPS} tok/s"
            TOTAL_TOK=$((TOTAL_TOK+TOK)); TOTAL_SEC=$(awk "BEGIN{printf \"%.3f\", $TOTAL_SEC+$WALL_MS/1000}")
        else echo "    run $i: FAILED"; fi
        sleep 3
    done
    local AVG; AVG=$(awk "BEGIN{if($TOTAL_SEC>0)printf \"%.1f\", $TOTAL_TOK/$TOTAL_SEC; else print \"NA\"}")
    echo "  [$LABEL] AVG: ${AVG} tok/s over ${RUNS} runs"
    echo "${LABEL},${THINKING},${MAXTOK},${AVG}" >> "$RESULTS_DIR/benchmark.csv"
}

wait_ready

if [ "$SUBCMD" = "baseline" ]; then
    echo "=== Baseline single-stream benchmark ==="
    echo "label,thinking,max_tokens,avg_tok_s" > "$RESULTS_DIR/benchmark.csv"
    measure "short"     "false" "What is 17 times 23?" 64
    measure "medium"    "false" "Write a Python implementation of binary search with type hints and docstring." 256
    measure "long"      "false" "Write a detailed technical explanation of how mixture-of-experts routing works in transformer language models, covering router networks, top-k gating, expert capacity, and load balancing. Include code examples." 768
    measure "reasoning" "true"  "Solve step by step: A train leaves city A at 60mph. Another leaves city B (300 miles away) at 80mph toward city A. When do they meet?" 512
    echo ""; echo "=== Results: $RESULTS_DIR/benchmark.csv ==="
    column -t -s, "$RESULTS_DIR/benchmark.csv" 2>/dev/null || cat "$RESULTS_DIR/benchmark.csv"

elif [ "$SUBCMD" = "thinking" ]; then
    echo "=== Thinking ablation ==="
    P="Explain the tradeoffs between transformer attention complexity and sequence length, covering sparse attention, linear attention, and sliding window approaches."
    [ -f "$RESULTS_DIR/benchmark.csv" ] || echo "label,thinking,max_tokens,avg_tok_s" > "$RESULTS_DIR/benchmark.csv"
    measure "thinking-off" "false" "$P" 512
    measure "thinking-on"  "true"  "$P" 512

elif [ "$SUBCMD" = "prefixcache" ]; then
    echo "=== Prefix cache TTFT ablation ==="
    SYS="You are an expert software engineering assistant with deep knowledge of distributed systems, algorithms, data structures, and software architecture. You provide precise, actionable technical guidance with code examples. You have access to tools for code execution, file management, and web search. Always reason carefully before responding."
    for label in cold warm; do
        if [ "$label" = cold ]; then Q="Implement a thread-safe LRU cache in Python."; else Q="Implement a min-heap in Python."; fi
        S=$(date +%s%3N)
        curl -sf "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
          -d "{\"model\":\"/model\",\"messages\":[{\"role\":\"system\",\"content\":\"$SYS\"},{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":256,\"chat_template_kwargs\":{\"enable_thinking\":false}}" >/tmp/dg_pc_$label.json 2>/dev/null
        E=$(date +%s%3N)
        TOK=$(python3 -c "import json;print(json.load(open('/tmp/dg_pc_$label.json'))['usage']['completion_tokens'])" 2>/dev/null || echo 0)
        echo "  [$label] wall=$((E-S))ms tokens=$TOK"
        sleep 3
    done
    echo "If warm wall << cold, APC is working on the diffusion path."
fi
