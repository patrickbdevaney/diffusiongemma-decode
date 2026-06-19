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

# 5. Weight-quant headroom probe (reproduces: experts already FP4; attention is the only
#    bf16 lever and has no tractable path — see REQUANT-ANALYSIS.md)
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

| Optimization | Script | Result (measured on Thor) | Status |
|---|---|---|---|
| TRITON_ATTN backend (required) | serve-diffusiongemma.sh | Correctness, not speed | ✅ baked in |
| Prefix caching (APC) | serve + phase4 | Confirmed active — hit rate 17.6%→33.9% in logs | ✅ verified |
| thinking=false per turn | phase5 | ~1.4s saved on short turns (nginx 2561→1097ms) | ✅ real lever |
| Max KV headroom (gpu_util 0.80) | serve | Long-context stability | ✅ enabled |
| ~~Expert FP4 quantization~~ | phase3 | **Premise wrong — experts already FP4** | ❌ see REQUANT-ANALYSIS.md |

Baseline single-stream (chat): short 50.7 / medium 105.7 / long 90.3 tok/s.

## What cannot be optimized (verified)

| Item | Why |
|---|---|
| Short-turn canvas tax | 256-token canvas fixed in weights |
| Canvas size | Not exposed in V2 model runner |
| Fast-dLLM stacking | Already the native engine |
| Concurrency aggregate | Not the target workload (single-stream is) |
| **Expert requant** | **Experts are ALREADY NVFP4** (verified — `experts.*_proj.weight` are U8/FP4) |
| **Attention requant** | Only bf16 weight + the real bottleneck, but no tractable path: W4A4 needs calibration (no loadable DiffusionGemma framework); W4A16 needs modelopt packing (won't install) + Marlin; plus diffusion quality risk NVIDIA already declined. See REQUANT-ANALYSIS.md |

## Weight-quant headroom — the short version

The directive's "primary bandwidth lever" (requant the bf16 experts) does not exist: the
experts are already FP4-cutlass-quantized. The only bf16 weights are attention (1.25 B) +
norms/router/embeddings. Attention is actually the single-stream bottleneck (dense → ~88% of
active bytes; experts are sparse), but FP4-ing it has no off-the-shelf path and real quality
risk on the diffusion denoising trajectory. Full evidence + decision tree: **REQUANT-ANALYSIS.md**.
The shipped checkpoint (FP4 experts + bf16 attention) is near the practical optimum; the real
wins are serve-time (prefix caching, thinking=false, concurrency).

## Ports
Production: 8004. Benchmark: 8005. FP4-experts comparison: 8006.
