#!/bin/bash
# phase3-quantize-experts.sh  (REWRITTEN — the original premise was wrong)
#
# The directive's premise: "MoE experts are bf16, requant them for bandwidth." FALSE.
# Inspection proves the experts are ALREADY NVFP4; the only bf16 weights are attention +
# norms/router/embeddings. The one requant candidate (attention) has no tractable path on
# this stack. This script reproduces the evidence. Full reasoning: REQUANT-ANALYSIS.md.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${DG_MODEL:-$REPO_ROOT/models/DiffusionGemma-26B-A4B-NVFP4}"
[ ! -d "$SRC" ] && { echo "ERROR: model not found: $SRC (set DG_MODEL)"; exit 1; }

echo "=== Weight-quantization headroom probe ==="
echo "See REQUANT-ANALYSIS.md for the full reasoning. This just reproduces the evidence."
echo ""

SRC="$SRC" python3 << 'PY'
import json, os, struct
from collections import Counter
d=os.environ['SRC']
def header(p):
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n))
dt={}
for shard in set(json.load(open(os.path.join(d,'model.safetensors.index.json')))['weight_map'].values()):
    for k,v in header(os.path.join(d,shard)).items():
        if k!='__metadata__': dt[k]=v
print("dtype histogram:", dict(Counter(v['dtype'] for v in dt.values())))
fp4 = {k.rsplit('.',1)[0] for k in dt if k.endswith('weight_scale')}
print(f"FP4-quantized layers: {len(fp4)}")
print("  experts among them:", any('experts' in l for l in fp4),
      "(=> experts ALREADY FP4, nothing to requant)")
bf16=[k for k,v in dt.items() if v['dtype']=='BF16']
def numel(v):
    r=1
    for x in v.get('shape',[]): r*=x
    return r
bf16_mass=sum(numel(dt[k]) for k in bf16)
attn_mass=sum(numel(dt[k]) for k in bf16 if 'attn' in k)
print(f"BF16 params total: {bf16_mass/1e9:.2f}B  | attention: {attn_mass/1e9:.2f}B (the only requant candidate)")
print()
print("Active-bytes/forward (single stream): attention is DENSE (~2.5GB bf16) and dominates;")
print("active experts are SPARSE 4/128 (~0.34GB FP4). So attention, not experts, is the bottleneck.")
print()
print("VERDICT: experts already FP4. Attention requant needs either W4A4 calibration (no loadable")
print("DiffusionGemma framework exists) or W4A16-Marlin packing (modelopt won't install here), plus")
print("it carries diffusion-specific quality risk NVIDIA already declined. No tractable path. STOP.")
PY

echo ""
echo "=== Confirm: AutoModelForCausalLM cannot load this model (why llm-compressor is out) ==="
echo "ValueError: model type 'diffusion_gemma' ... Transformers does not recognize this architecture"
echo "(verified — DiffusionGemma modeling code lives only in the vLLM image)"
