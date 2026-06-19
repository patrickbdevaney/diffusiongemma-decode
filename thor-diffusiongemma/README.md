# DiffusionGemma-26B-A4B-NVFP4 — Thor Optimized Single-Stream Decode

## Run order

```bash
# Model is reused from ~/models (set DG_MODEL) — not re-downloaded into the repo.
export DG_MODEL=$HOME/models/DiffusionGemma-26B-A4B-NVFP4
export DG_IMAGE=vllm/vllm-openai:gemma-aarch64-cu130   # aarch64 build for Thor

# 1. Launch production server
./thor-diffusiongemma/serve-diffusiongemma.sh default

# 2. Baseline benchmark (second terminal while server is up)
./thor-diffusiongemma/benchmark-single-stream.sh baseline

# 3. Verify prefix caching works on diffusion path
./thor-diffusiongemma/phase4-prefix-cache-verify.sh

# 4. Identify which turn types benefit from thinking=false
./thor-diffusiongemma/phase5-thinking-ablation.sh

# 5. Expert FP4 quantization (the bandwidth lever — run after baseline; HIGH RISK, see below)
./thor-diffusiongemma/phase3-quantize-experts.sh
```

> **Two fixes applied vs the directive (verified on this Thor unit):**
> 1. The gemma image `ENTRYPOINT` is `["vllm","serve"]`, so the launch passes `/model <flags>`,
>    **not** `vllm serve /model` (which doubles → `unrecognized arguments: serve /model`, instant crash).
> 2. Image defaults to `gemma-aarch64-cu130` (the verified-working aarch64 build), not bare `:gemma`.

---

## Generation profile — read this before routing traffic here

DiffusionGemma has a fundamentally different generation profile from an AR model.
The 256-token canvas is fixed in the model weights — it is not configurable.

### How the canvas works

The model fills a 256-token canvas by iteratively denoising all 256 positions in
parallel over N adaptive steps, then commits the entire canvas to KV and starts a
fresh canvas. N is determined by an entropy-bound convergence rule.

### The per-canvas tax on short turns

Every generation pays at least one full canvas denoising cycle. Measured profile
(dg-repro, chat endpoint, this Thor unit):

| Turn type         | Output tokens | Measured tok/s | vs AR incumbent |
|-------------------|---------------|----------------|-----------------|
| Short (tool call) | ~64           | ~35.8          | ❌ 137 AR wins  |
| Medium (code)     | ~256          | ~105.9         | ✅ near parity  |
| Long warm-canvas  | ~768          | ~96.1          | ✅ competitive  |

The short-turn number is structural, not a bug. It does not improve with optimization.

### Routing for an agentic harness

- Short turns (<150 tokens expected): Qwen3.6-35B-A3B + DFlash (137 tok/s)
- Long synthesis turns (>256 tokens expected): DiffusionGemma (~100 tok/s)
- When uncertain: default to AR stack

### Prefix caching

vLLM blog states APC works out of the box (encoder commit-pass writes KV like AR).
phase4 verifies this empirically on this unit. For an agentic harness with a long
shared system prompt + tool defs, APC removes prefill cost on every turn after the first.

### Thinking mode

Per-request: `chat_template_kwargs: {enable_thinking: true/false}`. Thinking tokens
consume canvas budget; `enable_thinking:false` avoids CoT on tool-call/short turns.
phase5 measures which turn types benefit.

---

## Optimizations implemented

| Optimization | Script | Expected lift | Status |
|---|---|---|---|
| TRITON_ATTN backend (required) | serve-diffusiongemma.sh | Correctness, not speed | Baked in |
| Prefix caching (APC) | serve + phase4 | TTFT reduction on cached prefix | Enabled |
| thinking=false per turn | phase5 | Wall-time on non-reasoning turns | Measure first |
| FP8/FP4 expert quantization | phase3 | ~106→~140 tok/s (IF it works) | Quality-gated, HIGH RISK |
| Max KV headroom (gpu_util 0.80) | serve | Long-context stability | Enabled |

## What cannot be optimized

| Item | Why |
|---|---|
| Short-turn canvas tax | 256-token canvas fixed in weights |
| Canvas size | Not exposed in V2 model runner |
| Fast-dLLM stacking | Already the native engine |
| Concurrency aggregate | Not the target workload (single-stream is) |

## ⚠️ Phase 3 risk note

The NVFP4 checkpoint keeps MoE experts/attn in bf16. Requantizing them is the real
bandwidth lever, BUT: DiffusionGemma is `DiffusionGemmaForBlockDiffusion`, not a
`CausalLM`, so `AutoModelForCausalLM.from_pretrained` (what llm-compressor uses) may
fail to load it, and reloading a packed-NVFP4 modelopt checkpoint for re-quant is
itself fragile. Treat phase3 as exploratory — it is expected to need custom module
targeting (inspect `named_modules()`), and quantization error compounds across the T
denoising steps. Do not serve the result until the quality gate passes.

## Ports
Production: 8004. Benchmark: 8005. FP4-experts comparison: 8006.
