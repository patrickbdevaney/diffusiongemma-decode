#!/bin/bash
# phase3-quantize-experts.sh  (EXPLORATORY — expected to need iteration)
# The NVIDIA NVFP4 checkpoint excludes mlp/router/self_attn (MoE experts stay bf16),
# so active-weight bytes/forward are ~2x full-FP4. Requantizing those layers is the
# remaining single-stream bandwidth lever. This uses llm-compressor.
#
# RISK (read README phase-3 note): DiffusionGemma is DiffusionGemmaForBlockDiffusion,
# NOT a CausalLM, so AutoModelForCausalLM.from_pretrained may not load it; and reloading
# a packed-NVFP4 modelopt checkpoint for re-quant is fragile. Quantization error also
# compounds across the T denoising steps. Quality-gate before serving the result.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${DG_MODEL:-$REPO_ROOT/models/DiffusionGemma-26B-A4B-NVFP4}"
DST="${DG_FP4_MODEL:-$REPO_ROOT/models/DiffusionGemma-26B-A4B-FP4experts}"
RESULTS_DIR="$REPO_ROOT/thor-diffusiongemma/results"
mkdir -p "$DST" "$RESULTS_DIR"
export SRC DST

[ ! -d "$SRC" ] && { echo "ERROR: source model not found: $SRC"; exit 1; }

echo "=== Phase 3: Quantize MoE experts (EXPLORATORY) ==="
echo "Source:      $SRC"
echo "Destination: $DST"

if ! python3 -c "import llmcompressor" 2>/dev/null; then
    echo "Installing llm-compressor..."
    pip install --quiet llmcompressor || { echo "ERROR: llm-compressor install failed"; exit 1; }
fi

echo "=== Step 1: inspect source — can it even load as a quantizable model? ==="
python3 << 'PYEOF'
import os, json, glob
src = os.environ['SRC']
# what's quantized vs bf16 in the checkpoint?
for name in ('hf_quant_config.json','quantization_config.json'):
    p = os.path.join(src, name)
    if os.path.exists(p):
        q = json.load(open(p))
        print(f"{name}: quant_algo={q.get('quantization',{}).get('quant_algo')}  "
              f"excludes={q.get('quantization',{}).get('exclude_modules')}")
# arch
cfg = json.load(open(os.path.join(src,'config.json')))
print("architectures:", cfg.get('architectures'), "| model_type:", cfg.get('model_type'))
print(">> If arch is DiffusionGemmaForBlockDiffusion, AutoModelForCausalLM will likely NOT load it.")
print(">> In that case the expert layer names must be discovered from the custom model class.")
PYEOF

echo ""
echo "=== Step 2: attempt load + quantize (FP8_DYNAMIC first — safer than FP4 on diffusion) ==="
python3 << PYEOF || { echo ""; echo "QUANTIZATION FAILED — see error above."; echo "Most likely: DiffusionGemmaForBlockDiffusion is not an AutoModelForCausalLM."; echo "Next step: load via the model's custom class (trust_remote_code) and inspect named_modules()"; echo "to find the real expert layer names, then target them. This is exploratory."; exit 1; }
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SRC, DST = os.environ['SRC'], os.environ['DST']
print(f"Loading {SRC} (bf16 for calibration)...")
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True, device_map="auto")
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

# discover actual expert/attn module names rather than guessing
names = [n for n,_ in model.named_modules()]
print("sample modules:", names[:8], "...")
targets = [r"re:.*\.mlp\..*", r"re:.*block_sparse_moe.*", r"re:.*\.self_attn\..*"]

recipe = QuantizationModifier(targets=targets, scheme="FP8_DYNAMIC", ignore=["lm_head"])
print("Applying FP8_DYNAMIC to experts/attn (weight-only, no calibration data)...")
oneshot(model=model, recipe=recipe, output_dir=DST)
tok.save_pretrained(DST)
print(f"Saved: {DST}")
print("NEXT: serve on :8006 and run the quality gate vs the bf16 baseline before promoting.")
print("If FP8 quality holds, re-run with scheme='NVFP4' for the full bandwidth win.")
PYEOF

echo ""
echo "=== Phase 3 complete (FP8-experts checkpoint written) ==="
echo "Quality gate: serve both and compare before promoting:"
echo "  DG_MODEL=$SRC DG_PORT=8004 ./serve-diffusiongemma.sh default"
echo "  DG_FP4_MODEL=$DST DG_PORT=8006 ./serve-diffusiongemma.sh fp4experts"
echo "  DG_PORT=8006 ./benchmark-single-stream.sh baseline   # compare tok/s + coherence"
echo "Kill criterion: any incoherent/factually-collapsed FP8 output the bf16 handles => revert."
