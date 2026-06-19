# DiffusionGemma-26B-A4B-NVFP4 — quality + real-workload findings (Thor, 2026-06-19)

Run: `python3 quality-eval.py 8004`. These are the "is the output actually good, and does the
agentic prefix-cache win materialize" answers that the speed benchmarks didn't address.

## Quality — the NVFP4 model is good WITH thinking on

| Check | Result |
|---|---|
| Code: `fib`, `flatten`, `is_prime` (executed) | ✅ all run correctly with adequate token budget |
| Factual: 17×23=391, capital=Tokyo, strawberry→10, WWII→1945 | ✅ correct (thinking on) |
| Reasoning: bat-and-ball → $0.05 | ✅ correct |
| Instruction: exact "PONG", JSON `{"a":1,"b":2}` | ✅ correct |
| Tool call (gemma4 parser): get_weather(Paris) | ✅ valid tool_call |

NVFP4 quantization does **not** visibly degrade coherence. Output quality is solid.

## ⚠️ Two real behaviors that change the routing guidance

1. **`enable_thinking:false` is a quality landmine, not a speed lever.** On terse prompts
   ("17×23? just the number", "capital? one word") thinking=false returns **1 token and stops —
   empty content**. With thinking=true the same prompts answer correctly. So the earlier phase-5
   finding ("thinking=false saves ~1.4s on short turns") is **retracted**: that "saving" was partly
   the model producing nothing. **Do NOT route short turns with thinking=false** — you'll get empty
   or wrong answers. DiffusionGemma is built to think; leave it on.

2. **Thinking is token-hungry and not separated.** The gemma4 reasoning parser never populates
   `reasoning_content` (it came back empty every time) — the thinking trace lands **inline in
   `content`**. Consequences:
   - Budget generous `max_tokens` (thinking + answer share the budget). At 400-512 tokens, complex
     answers/code get truncated mid-thought (finish=length, empty/partial content). `is_prime` needs
     ~1200 to finish; at 400 it's cut off.
   - The harness sees thinking mixed into the answer field — strip it client-side if needed.
   - **Runaway thinking** on some analytic prompts: "Is 2027 prime?" never concludes within 1024
     tokens (laborious trial-division reasoning) → empty at the cap. A real failure mode; cap-and-retry
     or route such prompts elsewhere.

## ✅ The real agentic win: long-prefix prefix caching

The phase-4 micro-bench under-measured this (70-word prefix). With a realistic **~6182-token** system
prompt + tool defs:

| Request | Wall | prompt_tokens |
|---|---|---|
| cold (prefix uncached) | 5447 ms | 8161 |
| warm-1 (prefix cached) | 1135 ms | 8161 |
| warm-2 (prefix cached) | 947 ms | 8161 |

**→ 79–83% wall-time reduction on cached prefix.** For a long-context agentic harness with a big
shared system prompt + tool schemas, APC is a ~5x TTFT win on every turn after the first. This is the
single biggest real lever for this workload — far more than any decode-rate tweak.

## Updated routing guidance

- Keep `enable_thinking:true` (off breaks output). Budget generous max_tokens.
- Lean on prefix caching — put the stable system prompt + tool defs first; the ~5x TTFT win dominates.
- Watch for runaway thinking on hard analytic prompts (cap + fallback).
- Long *synthesis* turns are where diffusion is competitive on tok/s; short turns still pay the
  per-canvas tax AND now must keep thinking on, so the AR stack remains preferable for short turns.
