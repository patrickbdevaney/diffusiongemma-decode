#!/bin/bash
# phase4-prefix-cache-verify.sh
# Verifies vLLM automatic prefix caching (APC) on the DiffusionGemma diffusion path.
# Measures wall time on: cold (full prefill) vs warm (shared system prefix -> APC hit).
# If warm << cold, APC works on the diffusion path. Model name "/model".

set -euo pipefail
PORT="${DG_PORT:-8004}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/thor-diffusiongemma/results"
mkdir -p "$RESULTS_DIR"

echo "=== Phase 4: Prefix cache verification (server :$PORT) ==="

SYS_PREFIX="You are an expert software engineering assistant with deep knowledge of distributed systems, algorithms, data structures, compiler design, and software architecture. You provide precise, actionable technical guidance. You have access to tools for code execution, web search, and file management. When given a task, you reason carefully and produce complete, working implementations."

wait_ready() { local W=0; until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do sleep 5; W=$((W+5)); [ $W -ge 300 ] && { echo "TIMEOUT"; exit 1; }; done; }

timed_request() {  # label content
    local S E WALL_MS TOK
    S=$(date +%s%3N)
    R=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
        -d "{\"model\":\"/model\",\"messages\":[{\"role\":\"system\",\"content\":\"$SYS_PREFIX\"},{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":128,\"chat_template_kwargs\":{\"enable_thinking\":false}}" 2>/dev/null)
    E=$(date +%s%3N); WALL_MS=$((E-S))
    TOK=$(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo 0)
    echo "  [$1] wall=${WALL_MS}ms tokens=${TOK}"
    echo "$1,${WALL_MS},${TOK}" >> "$RESULTS_DIR/prefix-cache.csv"
}

wait_ready
echo "label,wall_ms,tokens" > "$RESULTS_DIR/prefix-cache.csv"
echo "Turn 1 (cold — system prompt not cached):"; timed_request "cold" "Implement a thread-safe LRU cache in Python."; sleep 3
echo "Turn 2 (warm — same system prefix, should hit APC):"; timed_request "warm-hit-1" "Implement a min-heap in Python."; sleep 3
echo "Turn 3 (warm again):"; timed_request "warm-hit-2" "Implement a binary search tree in Python."; sleep 3
echo ""; echo "=== Results ==="
column -t -s, "$RESULTS_DIR/prefix-cache.csv" 2>/dev/null || cat "$RESULTS_DIR/prefix-cache.csv"
echo ""
echo "PASS: warm-hit wall_ms >20% lower than cold = APC working."
echo "FAIL: all similar = APC not active (confirm --enable-prefix-caching) or needs image update."
