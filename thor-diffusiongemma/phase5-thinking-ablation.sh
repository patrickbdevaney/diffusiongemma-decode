#!/bin/bash
# phase5-thinking-ablation.sh
# Measures wall-time cost of enable_thinking=true vs false per agentic turn type.
# Identifies which turns should route with thinking=false. Model name "/model".

set -euo pipefail
PORT="${DG_PORT:-8004}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/thor-diffusiongemma/results"
mkdir -p "$RESULTS_DIR"

echo "=== Phase 5: Thinking ablation (server :$PORT) ==="

req() {  # thinking prompt maxtok
    local S E TOK THINK_TOK
    S=$(date +%s%3N)
    R=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
        -d "{\"model\":\"/model\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":$3,\"chat_template_kwargs\":{\"enable_thinking\":$1}}" 2>/dev/null)
    E=$(date +%s%3N)
    TOK=$(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo 0)
    THINK_TOK=$(echo "$R" | python3 -c "import json,sys;m=json.load(sys.stdin)['choices'][0]['message'];print(len((m.get('reasoning_content') or '').split()))" 2>/dev/null || echo 0)
    echo "  thinking=$1: wall=$((E-S))ms completion_tok=$TOK reasoning_words=$THINK_TOK"
    echo "thinking_$1,${2:0:30},$((E-S)),$TOK,$THINK_TOK" >> "$RESULTS_DIR/thinking-ablation.csv"
}

echo "label,prompt_prefix,wall_ms,completion_tok,reasoning_words" > "$RESULTS_DIR/thinking-ablation.csv"
TURNS=(
    "List the files in /etc/nginx/:64"
    "What is 144 divided by 12?:32"
    "Write a Python function to flatten a nested list:256"
    "Explain why TCP uses a three-way handshake:256"
)
for TURN in "${TURNS[@]}"; do
    PROMPT="${TURN%:*}"; MAXTOK="${TURN#*:}"
    echo "Prompt: '${PROMPT:0:50}...'"
    req "false" "$PROMPT" "$MAXTOK"; sleep 2
    req "true"  "$PROMPT" "$MAXTOK"; sleep 2
    echo ""
done
echo "=== Results: $RESULTS_DIR/thinking-ablation.csv ==="
column -t -s, "$RESULTS_DIR/thinking-ablation.csv" 2>/dev/null || cat "$RESULTS_DIR/thinking-ablation.csv"
echo ""
echo "ACTION: route turns where thinking=true wall >> thinking=false AND reasoning_words>50 with enable_thinking=false."
