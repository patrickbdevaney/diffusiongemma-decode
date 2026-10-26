# DiffusionGemma-26B-A4B on Jetson AGX Thor: Single-Stream vLLM Reference

**Serving configuration, decode-speed measurements and quality checks for NVIDIA's block-diffusion Gemma on one
Jetson AGX Thor, at concurrency 1**

| | |
|---|---|
| **Model** | [`nvidia/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) (`DiffusionGemmaForBlockDiffusion`): 26B MoE, ~4B active, 4 of 128 experts routed |
| **Precision** | Experts NVFP4 (W4A4, CUTLASS); attention, norms, router and embeddings BF16 (2.99B parameters) |
| **Generation** | Block diffusion: a fixed 256-token canvas is denoised in parallel over an adaptive number of steps, then committed to the KV cache |
| **Runtime** | vLLM `vllm/vllm-openai:gemma-aarch64-cu130`, V2 model runner, `TRITON_ATTN` |
| **Hardware** | NVIDIA Jetson AGX Thor (`sm_110`), 128 GB unified memory, `nvpmodel -m 1` (120 W) |
| **Workload** | Single stream (`--max-num-seqs 1`), chat completions with tool calling |
| **Validated** | 2026-06-19 |

---

## Summary

- **Fast on long outputs, slow on short ones.** Every request pays at least one full 256-token canvas. Short answers
  therefore run at 35.8–50.7 tok/s, while 256–768-token answers run at 90–116 tok/s.
- **Prefix caching is the largest lever measured.** It works on the diffusion path, and with a 6,182-token shared
  system prompt it cuts request wall time by **79–83%** (5.4 s → 0.95–1.14 s). That is worth more to an agent harness
  than any decode-rate change.
- **Leave thinking on.** With `enable_thinking: false`, terse prompts return a single token and stop. An earlier
  finding that disabling thinking saved time on short turns is retracted.
- **Quantization headroom is exhausted.** The experts are already NVFP4. The remaining BF16 weights are attention,
  which dominates single-stream weight traffic but has no tractable requantization path
  ([`docs/REQUANT-ANALYSIS.md`](docs/REQUANT-ANALYSIS.md)).

---

## Performance

### Decode throughput by turn length

Chat endpoint, temperature 0, 4 runs per row. Throughput is completion tokens divided by request wall time, so
prefill and the first canvas are included.

| Turn | `max_tokens` | Thinking | tok/s | Source |
|---|--:|---|--:|---|
| Short (factual / tool call) | 64 | on | **35.8** | earlier reproduction run on this device |
| Short | 64 | off | 50.7 | [`results/benchmark.csv`](results/benchmark.csv) |
| Medium (code) | 256 | on | 105.9 | earlier reproduction run |
| Medium | 256 | off | 105.7 | `results/benchmark.csv` |
| Long, warm canvas | 768 | on | 96.1 | earlier reproduction run |
| Long | 768 | off | 90.3 | `results/benchmark.csv` |
| Reasoning | 512 | on | **115.8** | `results/benchmark.csv` |

The thinking-off rows are kept as measurements. Thinking-off is not a recommended configuration (see
[Quality](#quality)).

**Why short turns are slow.** The 256-token canvas is fixed in the model and not exposed by the V2 model runner.
A 64-token answer costs at least the same denoising cycle as a 256-token one. This is structural and does not
respond to tuning.

**For comparison, on the same Thor:** Qwen3.6-35B-A3B with DFlash speculative decoding (an autoregressive target)
runs at 116.5 tok/s on a 4-task mean and 139.1 tok/s at peak
([dflash-vllm-resources](https://github.com/patrickbdevaney/dflash-vllm-resources)). DiffusionGemma is close to it on
256-token and longer outputs and well behind it on short turns.

### Prefix caching

Automatic prefix caching works on the diffusion path: the commit pass writes KV entries the way an autoregressive
model does.

| Request (6,182-token shared prefix, 8,161 prompt tokens) | Wall time |
|---|--:|
| Cold, prefix not cached | 5,447 ms |
| Warm, run 1 | 1,135 ms |
| Warm, run 2 | 947 ms |

That is a **79–83% reduction**, about 5× faster time to first token on every turn after the first. A shorter
micro-test with a ~70-word prefix ([`scripts/phase4-prefix-cache-verify.sh`](scripts/phase4-prefix-cache-verify.sh))
was too small to show a wall-time difference. Its server log did show the cache-hit rate climbing from 17.6% to
33.9%.

### Quality

Run by [`eval/quality-eval.py`](eval/quality-eval.py) with thinking on. Detail is in
[`docs/QUALITY-FINDINGS.md`](docs/QUALITY-FINDINGS.md).

| Check | Result |
|---|---|
| Generated code executed: `fib`, `flatten`, `is_prime` | Correct, given enough token budget |
| Factual: 17 × 23, capital of Japan, letters in "strawberry", end of WWII | Correct |
| Reasoning: bat-and-ball | Correct ($0.05) |
| Instruction following: exact "PONG", exact JSON | Correct |
| Tool call (`gemma4` parser): `get_weather(Paris)` | Valid `tool_call` |

These are functional checks, not a benchmark score. No difference from the BF16 model was measured.

---

## Operating guidance

| Topic | Guidance |
|---|---|
| Thinking | Keep `enable_thinking: true`, which is the server default here. Off returns empty or one-token answers on terse prompts |
| `max_tokens` | Budget generously. Thinking and the answer share it, and the `gemma4` parser leaves the thinking trace inline in `content` (`reasoning_content` stays empty). `is_prime` needs ~1,200 tokens; at 400 it is cut off |
| Runaway thinking | Some analytic prompts ("Is 2027 prime?") do not conclude within 1,024 tokens. Cap and retry, or route elsewhere |
| Prompt layout | Put the stable system prompt and tool definitions first, so the prefix cache reuses them |
| Routing in an agent harness | Short turns (under ~150 expected tokens) are better served by an autoregressive model with speculative decoding. Long synthesis turns (over ~256 tokens) suit DiffusionGemma |

---

## Deployment

### Quick start

```bash
hf download nvidia/diffusiongemma-26B-A4B-it-NVFP4 --local-dir ~/models/DiffusionGemma-26B-A4B-NVFP4
export DG_MODEL=$HOME/models/DiffusionGemma-26B-A4B-NVFP4
export DG_IMAGE=vllm/vllm-openai:gemma-aarch64-cu130        # default; aarch64 build for Thor

./scripts/serve-diffusiongemma.sh default                   # OpenAI-compatible server on :8004
```

**Verify:** the startup log should show `TRITON_ATTN`, the V2 model runner, `diffusion_gemma` initialization and
NVFP4 without Marlin. Failure signs are `FLASHINFER` being selected, or `unrecognized arguments: serve /model`.

### Serving configuration

| Setting | Value | Reason |
|---|---|---|
| Container command | `/model <flags>` | The image's `ENTRYPOINT` is already `["vllm","serve"]`. Passing `vllm serve /model` doubles it and the server exits at once |
| `VLLM_USE_V2_MODEL_RUNNER=1` | set | The diffusion model runs on the V2 runner |
| `--attention-backend` | `TRITON_ATTN` | Required for correctness on this model and device |
| `--max-num-seqs` | 1 | Single-stream target |
| `--gpu-memory-utilization` | 0.80 | KV headroom for long context |
| `--max-model-len` | 65,536 | |
| `--enable-prefix-caching` | on (`default` mode) | The main agentic win |
| `--tool-call-parser`, `--reasoning-parser` | `gemma4` | |
| `--default-chat-template-kwargs` | `{"enable_thinking": true}` | See [Operating guidance](#operating-guidance) |
| `--override-generation-config` | `{"max_new_tokens": null}` | Lets the request's `max_tokens` govern |

Do not pass `canvas_length`, `max_denoising_steps` or `commit_threshold`. These options do not exist in this
runtime.

The script also:
- sets `nvpmodel -m 1` and `jetson_clocks`;
- drops the page cache;
- removes any previous instance before launch.

Modes:
- `prefetch-test` turns prefix caching off, for A/B runs;
- `fp4experts` targets a requantized checkpoint, which turned out to be unnecessary (see below).

### Reproduce the measurements

| Step | Command | Output |
|---|---|---|
| Decode benchmark | `./scripts/benchmark-single-stream.sh baseline` | `results/benchmark.csv` |
| Prefix-cache micro-test | `./scripts/phase4-prefix-cache-verify.sh` | `results/prefix-cache.csv` |
| Thinking ablation | `./scripts/phase5-thinking-ablation.sh` | `results/thinking-ablation.csv` (conclusion retracted; see Quality) |
| Quality, long-prefix cache and tool calls | `python3 eval/quality-eval.py 8004` | stdout |
| Requantization probe | `./scripts/phase3-quantize-experts.sh` | Reproduces "experts are already FP4" |

Run the benchmarks from a second terminal while the server is up. `DG_PORT` and `DG_RUNS` override the port and the
run count.

> **Note:** `eval/quality-eval.py` runs model-generated Python with `exec()` to check the code tasks. Run it only in
> an environment where that is acceptable.

---

## What cannot be optimized here

| Lever | Finding |
|---|---|
| Short-turn canvas cost | The 256-token canvas is fixed in the model weights |
| Canvas size or step count | Not exposed by the V2 model runner |
| Fast-dLLM-style caching | Already the native engine |
| Requantizing the experts | **Already NVFP4.** All 11,520 `experts.*_proj` weights are packed FP4 with FP8 group scales. The checkpoint's `*mlp*` exclusion never matched them |
| Requantizing attention | The real single-stream bottleneck: dense attention is about 88% of active weight bytes. W4A4 needs calibration, and DiffusionGemma cannot be loaded through transformers or llm-compressor. W4A16 needs `nvidia-modelopt` packing, which did not install here, and the Marlin kernel. Bidirectional attention over many denoising steps is also the regime most sensitive to FP4 error |
| `W4A16_NVFP4` config flip | No-op: all FP4 weights sit in FusedMoE, which kept `VLLM_CUTLASS` |

The shipped checkpoint (FP4 experts, BF16 attention) is close to the practical optimum on this stack. The remaining
wins are at serve time, chiefly prefix caching. The full investigation is in
[`docs/REQUANT-ANALYSIS.md`](docs/REQUANT-ANALYSIS.md).

---

## Limitations

- **Single stream only.** Aggregate throughput under concurrency was not measured.
- **Prompt-specific numbers.** One prompt per turn class and 4 runs per row; this is not a benchmark suite.
- **Short and long thinking-on rows come from an earlier reproduction run** on the same device (chat endpoint), not
  from `results/benchmark.csv`.
- **Quality is checked, not scored.** Functional checks only; no comparison against BF16 or other models.
- **Thinking output is not separated** from the answer by the `gemma4` parser on this image.

## Repository layout

| Path | Contents |
|---|---|
| [`scripts/serve-diffusiongemma.sh`](scripts/serve-diffusiongemma.sh) | Production server (`default`, `prefetch-test`, `fp4experts`) |
| [`scripts/benchmark-single-stream.sh`](scripts/benchmark-single-stream.sh) | Decode, thinking and prefix-cache benchmarks |
| [`scripts/phase3-quantize-experts.sh`](scripts/phase3-quantize-experts.sh), [`phase4-prefix-cache-verify.sh`](scripts/phase4-prefix-cache-verify.sh), [`phase5-thinking-ablation.sh`](scripts/phase5-thinking-ablation.sh) | Requantization probe, prefix-cache test, thinking ablation |
| [`eval/quality-eval.py`](eval/quality-eval.py) | Quality, long-prefix cache and tool-call checks |
| [`results/`](results/) | Raw CSVs |
| [`docs/RESULTS-LOG.md`](docs/RESULTS-LOG.md) | Dated run log, including the retracted thinking-off finding |
| [`docs/QUALITY-FINDINGS.md`](docs/QUALITY-FINDINGS.md), [`docs/REQUANT-ANALYSIS.md`](docs/REQUANT-ANALYSIS.md) | Quality findings; quantization headroom analysis |

## License

- **This repository:** MIT ([`LICENSE`](LICENSE)).
- **DiffusionGemma-26B-A4B-it-NVFP4:** Apache-2.0 (NVIDIA). No weights are distributed here.
- **vLLM:** Apache-2.0.

This is an independent community project, not affiliated with NVIDIA or Google.
