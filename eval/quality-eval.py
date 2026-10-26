#!/usr/bin/env python3
# quality-eval.py — does DiffusionGemma actually produce GOOD output (not just fast)?
# Three checks against a running server (default :8004):
#   1. Quality: code-execution (run the generated code), factual, instruction-following
#   2. Long-prefix prefix-cache quantification (the real agentic APC win)
#   3. Tool-call + reasoning functional verification
# Usage: python3 quality-eval.py [port]
import json, sys, time, urllib.request, re

PORT = sys.argv[1] if len(sys.argv) > 1 else "8004"
URL = f"http://localhost:{PORT}/v1/chat/completions"

def chat(messages, max_tokens=256, thinking=False, tools=None):
    body = {"model": "/model", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools: body["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=300)); dt = time.time()-t
    return r["choices"][0]["message"], r["usage"], dt

def ask(prompt, **kw):
    m, u, dt = chat([{"role": "user", "content": prompt}], **kw)
    return (m.get("content") or ""), m, u, dt

def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()

print("="*64); print("1. QUALITY — code execution (gold standard)"); print("="*64)
code_tests = [
    ("is_prime", "Write a Python function `is_prime(n)` returning True iff n is prime. Output ONLY the function code.",
     lambda ns: ns["is_prime"](7) is True and ns["is_prime"](10) is False and ns["is_prime"](2027) is True),
    ("fib", "Write a Python function `fib(n)` returning the nth Fibonacci number with fib(0)=0, fib(1)=1. Output ONLY code.",
     lambda ns: ns["fib"](10) == 55 and ns["fib"](1) == 1),
    ("flatten", "Write a Python function `flatten(lst)` that flattens an arbitrarily nested list into a flat list. Output ONLY code.",
     lambda ns: ns["flatten"]([1,[2,[3,4]],5]) == [1,2,3,4,5]),
]
code_pass = 0
for name, prompt, check in code_tests:
    content, _, _, dt = ask(prompt, max_tokens=400)
    code = extract_code(content)
    try:
        ns = {}; exec(code, ns)
        ok = check(ns)
    except Exception as e:
        ok = False; print(f"  {name}: EXEC ERROR {type(e).__name__}: {str(e)[:60]}")
    code_pass += ok
    print(f"  {name}: {'PASS' if ok else 'FAIL'} ({dt:.1f}s)")

print("\n" + "="*64); print("2. QUALITY — factual / instruction-following"); print("="*64)
fact_tests = [
    ("17*23", "What is 17 times 23? Answer with just the number.", lambda c: "391" in c),
    ("capital", "What is the capital of Japan? One word.", lambda c: "tokyo" in c.lower()),
    ("prime", "Is 2027 a prime number? Answer yes or no.", lambda c: "yes" in c.lower()),
    ("ww2", "What year did World War II end? Just the year.", lambda c: "1945" in c),
    ("exact", "Reply with exactly this and nothing else: PONG", lambda c: c.strip().upper().strip(".!") == "PONG"),
    ("json", 'Output a JSON object with keys "a" set to 1 and "b" set to 2. Output only the JSON.',
     lambda c: (lambda j: j.get("a")==1 and j.get("b")==2)(json.loads(re.search(r"\{.*\}", c, re.DOTALL).group(0)))),
]
fact_pass = 0
for name, prompt, check in fact_tests:
    content, _, _, dt = ask(prompt, max_tokens=128)
    try: ok = check(content)
    except Exception: ok = False
    fact_pass += ok
    print(f"  {name}: {'PASS' if ok else 'FAIL'} -> {content.strip()[:60]!r}")

print("\n" + "="*64); print("3. LONG-PREFIX prefix-cache quantification"); print("="*64)
# realistic ~large agentic system prompt (instructions + fake tool schemas), repeated to bulk up
TOOLDEFS = "\n".join(
    f'- tool_{i}(arg_a: string, arg_b: int): performs operation {i} on the given arguments and returns a structured result with status, payload, and metadata fields for downstream processing.'
    for i in range(60))
BIG_SYS = ("You are an expert software engineering agent operating in a long-running session. "
           "Follow instructions precisely, reason about tool use, and produce complete working code. "
           "Available tools:\n" + TOOLDEFS) * 3
sys_tok = len(BIG_SYS.split())
print(f"  system prefix ~{sys_tok} words (~{int(sys_tok*1.3)} tokens)")
def timed(sys, q):
    m,u,dt = chat([{"role":"system","content":sys},{"role":"user","content":q}], max_tokens=32)
    return dt, u["completion_tokens"], u.get("prompt_tokens")
d1,t1,p1 = timed(BIG_SYS, "Say READY.")              # cold: prefix not cached
d2,t2,p2 = timed(BIG_SYS, "Say GO.")                 # warm: same big prefix -> APC hit
d3,t3,p3 = timed(BIG_SYS, "Say DONE.")               # warm again
print(f"  cold  (prefix uncached): {d1*1000:.0f}ms  prompt_tok={p1}")
print(f"  warm1 (prefix cached):   {d2*1000:.0f}ms  prompt_tok={p2}")
print(f"  warm2 (prefix cached):   {d3*1000:.0f}ms  prompt_tok={p3}")
if d1 > 0:
    print(f"  => warm vs cold: {(1-d2/d1)*100:.0f}% / {(1-d3/d1)*100:.0f}% wall reduction")

print("\n" + "="*64); print("4. TOOL CALL + reasoning"); print("="*64)
tools = [{"type":"function","function":{"name":"get_weather","description":"Get weather for a city",
         "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
m,u,dt = chat([{"role":"user","content":"What's the weather in Paris? Use the get_weather tool."}], max_tokens=128, tools=tools)
tc = m.get("tool_calls")
if tc:
    fn = tc[0]["function"]
    try: args_ok = "paris" in json.loads(fn["arguments"]).get("city","").lower()
    except Exception: args_ok = False
    print(f"  tool call: {'PASS' if fn['name']=='get_weather' and args_ok else 'PARTIAL'} -> {fn['name']}({fn['arguments']})")
else:
    print(f"  tool call: FAIL (no tool_calls) content={ (m.get('content') or '')[:80]!r}")
# reasoning
m2,u2,dt2 = chat([{"role":"user","content":"A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball? Think carefully."}], max_tokens=600, thinking=True)
rc = m2.get("reasoning_content") or ""; c2 = m2.get("content") or ""
correct = "0.05" in (rc+c2) or "5 cent" in (rc+c2).lower() or "$.05" in (rc+c2)
print(f"  reasoning: answer_correct(0.05)={correct}  reasoning_content_chars={len(rc)}  [THINK]_in_content={'[THINK]' in c2}")
print(f"    content: {c2.strip()[:120]!r}")

print("\n" + "="*64)
print(f"SUMMARY: code {code_pass}/{len(code_tests)}  factual/instruct {fact_pass}/{len(fact_tests)}")
print("="*64)
