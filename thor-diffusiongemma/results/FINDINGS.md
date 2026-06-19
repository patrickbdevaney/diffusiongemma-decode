# DiffusionGemma single-stream optimization — results (Thor SM110, 2026-06-19)

Server: gemma-aarch64-cu130, V2 runner, TRITON_ATTN, modelopt_fp4, FP8 KV,
prefix-caching ON, max-num-seqs 1, gpu_util 0.80. (Entrypoint fix applied: pass /model.)

## Baseline single-stream (chat endpoint)
| turn | thinking | max_tok | tok/s |
|---|---|---|---|
| short | false | 64 | 50.7 |
| medium | false | 256 | 105.7 |
| long | false | 768 | 90.3 |
| reasoning | true | 512 | 115.8 |
Consistent with dg-repro. Note short=50.7 with thinking=false vs 35.8 with thinking=on.

## Prefix caching (phase4) — WORKS
Wall-time TTFT delta inconclusive (cold 1969 / warm 2264 / 1566 ms) because the test's
shared system prefix (~70 words) is tiny vs the 128-tok generation. BUT server logs prove
APC active on the diffusion path: prefix-cache hit rate climbed 17.6% -> 22.9% -> 26.8% -> 33.9%.
=> Confirmed (vLLM blog claim holds on Thor). For a real agentic harness with a long shared
system prompt + tool defs, the prefill saving would be material; this micro-benchmark under-measures it.

## Thinking ablation (phase5) — real lever on short turns
| turn | thinking=false | thinking=true | delta |
|---|---|---|---|
| "list files /etc/nginx" (64) | 1097ms (59 tok) | 2561ms (64 tok) | -1464ms |
| "144/12" (32) | 602ms (14 tok) | 958ms (32 tok) | -356ms |
| flatten list (256) | 2804ms | 2083ms | mixed (capped) |
| TCP handshake (256) | 2759ms | 2938ms | mixed (capped) |
=> thinking=false saves ~1.4s on short tool/factual turns. Route those with enable_thinking=false.
reasoning_content not separated by gemma4 parser (reasoning_words=0) but thinking still costs tokens.

## Phase 3 (expert requant) — INFEASIBLE as written (verified)
The FP4-experts bandwidth lever is real (NVFP4 checkpoint keeps mlp/router/self_attn in bf16),
but llm-compressor cannot reach it: it wraps AutoModelForCausalLM, and
`AutoModelForCausalLM.from_pretrained('/model')` fails with:
  ValueError: model type `diffusion_gemma` ... Transformers does not recognize this architecture.
The DiffusionGemma model code lives in the vLLM image (not as a transformers trust_remote_code
class); config has architectures=['DiffusionGemmaForBlockDiffusion'], auto_map=NONE, modelopt-packed
weights. So requantizing experts requires a modelopt-based flow against the vLLM model definition
(a research effort), not this llm-compressor script. Did NOT run the install/quant (would fail at load).
Bandwidth lever remains open but out of reach with off-the-shelf tooling.
