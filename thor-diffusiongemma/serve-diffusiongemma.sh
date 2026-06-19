#!/bin/bash
# serve-diffusiongemma.sh
# Production single-stream DiffusionGemma server on Jetson AGX Thor.
# Every serving flag is grounded in the official NVIDIA model card and dg-repro findings.
# Do NOT add canvas_length / max_denoising_steps / commit_threshold — they do not exist.
#
# FIX vs directive: this image's ENTRYPOINT is ["vllm","serve"], so the container command
# is `/model <flags>`, NOT `vllm serve /model` (which doubles -> "unrecognized arguments:
# serve /model" and crashes instantly). Verified on this Thor unit.

set -euo pipefail
MODE="${1:-default}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${DG_MODEL:-$REPO_ROOT/models/DiffusionGemma-26B-A4B-NVFP4}"
SCRIPT_DIR="$REPO_ROOT/thor-diffusiongemma"
CONFIG_DIR="$SCRIPT_DIR/config"
RESULTS_DIR="$SCRIPT_DIR/results"
PORT="${DG_PORT:-8004}"
CACHE_ROOT="$HOME/thor-vllm-cache/diffusiongemma"
IMAGE="${DG_IMAGE:-vllm/vllm-openai:gemma-aarch64-cu130}"
mkdir -p "$CONFIG_DIR" "$RESULTS_DIR" "$CACHE_ROOT"

[ ! -d "$MODEL_DIR" ] && {
    echo "ERROR: model not found at $MODEL_DIR"
    echo "  Set DG_MODEL env or run: hf download nvidia/diffusiongemma-26B-A4B-it-NVFP4 --local-dir $MODEL_DIR"
    exit 1
}

# Mode controls ONLY levers that actually exist.
case "$MODE" in
default)
    MAX_SEQS=1; GPU_UTIL=0.80; PREFIX_CACHE="--enable-prefix-caching"
    echo "Mode: default | single-stream | prefix-caching ON | util=0.80"
    ;;
prefetch-test)
    MAX_SEQS=1; GPU_UTIL=0.80; PREFIX_CACHE=""
    echo "Mode: prefetch-test | prefix-caching OFF | measure TTFT delta vs default"
    ;;
fp4experts)
    MODEL_DIR="${DG_FP4_MODEL:-$REPO_ROOT/models/DiffusionGemma-26B-A4B-FP4experts}"
    MAX_SEQS=1; GPU_UTIL=0.60; PREFIX_CACHE="--enable-prefix-caching"
    [ ! -d "$MODEL_DIR" ] && { echo "ERROR: FP4-experts checkpoint not found — run phase3-quantize-experts.sh first"; exit 1; }
    echo "Mode: fp4experts | FP4-quantized experts | util=0.60"
    ;;
*)
    echo "Usage: $0 [default|prefetch-test|fp4experts]"; exit 1 ;;
esac

# ── Thor perf + memory mitigations ────────────────────────────────────────────
echo "=== Thor performance mode ==="
sudo nvpmodel -m 1 2>/dev/null && echo "nvpmodel -m 1 (120W sustained)" || echo "WARN: nvpmodel failed"
sudo jetson_clocks 2>/dev/null && echo "jetson_clocks locked" || echo "WARN: jetson_clocks failed"
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1 || true
echo 0 | sudo tee /proc/sys/vm/nr_hugepages >/dev/null 2>&1 || true
sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 && echo "page cache dropped" || true
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
echo "host MemAvailable: ${AVAIL_GB} GiB"
[ "${AVAIL_GB:-0}" -lt 40 ] 2>/dev/null && echo "!! LOW MEMORY — sudo reboot if launch sticks in Created"

# ── Kill prior instance ────────────────────────────────────────────────────────
docker ps --format '{{.Names}}' | grep -E '^diffusiongemma-|^dg-repro$' | xargs -r docker rm -f 2>/dev/null || true
sudo fuser -k ${PORT}/tcp 2>/dev/null || true
sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true; sleep 1

echo ""
echo "=== Docker pull: $IMAGE ==="
docker pull "$IMAGE" || docker image inspect "$IMAGE" &>/dev/null || { echo "ERROR: no cached image"; exit 1; }

CONTAINER_NAME="diffusiongemma-$(date +%s)"
echo "$CONTAINER_NAME" > "$CACHE_ROOT/current-container-name"

echo ""
echo "=== Launch (foreground) — container: $CONTAINER_NAME on :$PORT ==="
echo "  WANT in logs: TRITON_ATTN, V2 model runner, diffusion_gemma init, NVFP4 (no marlin)"
echo "  FAIL: FLASHINFER selected | 'unrecognized arguments: serve /model'"
echo ""

docker run --rm -it \
    --name "$CONTAINER_NAME" \
    --runtime nvidia --gpus all \
    --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e USE_FASTSAFETENSOR=true \
    -e HF_HUB_DISABLE_XET=1 \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    -v "${MODEL_DIR}:/model:ro" \
    -v "${CACHE_ROOT}:/root/.cache/vllm" \
    "$IMAGE" \
    /model \
        --trust-remote-code \
        --max-num-seqs "$MAX_SEQS" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-model-len 65536 \
        --attention-backend TRITON_ATTN \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4 \
        --override-generation-config '{"max_new_tokens": null}' \
        --default-chat-template-kwargs '{"enable_thinking":true}' \
        ${PREFIX_CACHE} \
        --port "$PORT" \
        --host 0.0.0.0
