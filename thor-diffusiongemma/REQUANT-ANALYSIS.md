# Weight-quantization headroom analysis (DiffusionGemma-26B-A4B-NVFP4)

Investigation of the "requant the experts for bandwidth" lever. Verdict: **the premise was
wrong, and the real remaining lever (attention) has no tractable positive-EV path.** Evidence below.

## 1. The experts are ALREADY NVFP4 — there is nothing to requant there

Checkpoint tensor inspection:
```
experts.N.{gate,up,down}_proj.weight        -> U8       (packed FP4, 2 nibbles/byte)
experts.N.{gate,up,down}_proj.weight_scale  -> F8_E4M3  (per-16 group scale)
experts.N.{gate,up,down}_proj.weight_scale_2-> F32      (per-tensor global scale)
experts.N.{gate,up,down}_proj.input_scale   -> F32      (activation scale -> W4A4)
```
11,520 expert layers are FP4. The directive assumed the `*mlp*` exclude pattern in
hf_quant_config kept experts in bf16 — but experts are named `experts.*_proj`, which `*mlp*`
never matches. So "MoE experts stay bf16" was false.

## 2. Only 2.99 B params are bf16 — and attention, not experts, is the bandwidth bottleneck

BF16 tensors: self_attn (1.25 B), norms (0.x B), router, embeddings, dense/self-conditioning.
Norms/router/embeddings must stay high-precision (correctness). The only requant candidate is
**attention (1.25 B)**.

Active-weight bytes per forward (single stream) — the directive had this BACKWARDS:
- Attention is **dense** (100% active every token): 1.25 B bf16 ≈ **2.5 GB / forward**
- Active experts are **sparse** (4 of 128 routed): ~22 B FP4 × 4/128 ≈ 0.69 B × 0.5 B ≈ **0.34 GB**

Attention dominates the active stream (~88%). FP4 attention (2.5 -> ~0.6 GB) is the real lever,
worth a rough ~1.3-1.5x IF it could run on the fast path AND quality held.

## 3. Why it can't be done cheaply — the W4A4(cutlass)/W4A16(marlin) kernel split

vLLM's modelopt_fp4 loader has two FP4 linear kernels, and the choice is GLOBAL:
- `ModelOptNvFp4LinearMethod` — **W4A4, cutlass** (fast). Requires a calibrated `input_scale`
  (activation amax) for every quantized layer.
- `ModelOptNvFp4W4A16LinearMethod` — **W4A16, Marlin** (slower GEMM). No activation quant
  (placeholder input_scale), so no calibration needed.

To make attention FP4 you must pick one:
- **W4A4 path**: needs a calibrated attention `input_scale`. Calibration requires running data
  through the model's attention layers — but DiffusionGemma is `DiffusionGemmaForBlockDiffusion`,
  has no transformers class (`AutoModelForCausalLM` fails: "Transformers does not recognize
  diffusion_gemma"), no `auto_map`, and the modeling code lives only inside the vLLM image. So
  there is no off-the-shelf way to calibrate. (llm-compressor, which wraps AutoModelForCausalLM,
  is dead for the same reason.)
- **W4A16 path**: no calibration, but you must repack attention weights to NVFP4 and accept Marlin
  for the attention GEMM. `nvidia-modelopt` (for correct packing) won't pip-install on this host
  (externally-managed-environment; aarch64 wheels uncertain), leaving hand-rolled e2m1/e4m3
  packing — high bug risk for a layout that, if subtly wrong, silently produces garbage.

Tested: flipping the config to `W4A16_NVFP4` on the existing checkpoint is a NO-OP — all FP4
weights are in the FusedMoE (which selects its backend independently and stayed `VLLM_CUTLASS`),
so there are no quantized Linear layers for the W4A16 method to touch. Measured ~baseline tok/s.

## 4. Why NVIDIA almost certainly kept attention bf16 on purpose

DiffusionGemma refines a 256-token canvas with **bidirectional attention over T denoising steps**.
Quantization error in attention compounds across every step and every canvas position — the
worst-case regime for FP4 attention. So bf16 attention is most likely a deliberate quality choice,
not an oversight. Any requant must pass a hard quality gate, and there is real risk it fails.

## Verdict

- Experts: already FP4. No action.
- Attention: the only bf16 lever, but both paths are blocked off-the-shelf (W4A4 needs a
  framework-level calibration that doesn't exist; W4A16 needs modelopt packing that won't install
  + accepts Marlin) AND carries genuine diffusion-specific quality risk.
- **Net: no tractable, positive-EV requant remains.** The shipped checkpoint (FP4-cutlass experts +
  bf16 attention) is close to the practical optimum on this stack. The real wins on this model are
  the serve-time levers (prefix caching, thinking=false on short turns, concurrency), not requant.

To pursue it anyway would require: a loadable DiffusionGemma module + calibration data (for W4A4),
or a modelopt environment for W4A16 packing — then a strict A/B quality gate across full denoising
trajectories before serving. That is a research effort, not a script.
