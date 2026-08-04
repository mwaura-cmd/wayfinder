"""
verify.py — Quick health-check for all Wayfinder components.
Run: python verify.py
"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

PASS = "  PASS"
FAIL = "  FAIL"
WARN = "  WARN"

def sep():
    print("-" * 60)

print("=" * 60)
print("WAYFINDER — COMPONENT VERIFICATION")
print("=" * 60)

# ── 1. Load config (auto-reads env file) ──────────────────────
sep()
print("[1/4] API Keys from env file")
import config

okey = config.OPENROUTER_API_KEY
tkey = config.TAVILY_API_KEY

if okey:
    print(f"  OpenRouter key : {okey[:8]}...{okey[-4:]}  ({len(okey)} chars)")
else:
    print("  OpenRouter key : MISSING")

if tkey:
    print(f"  Tavily key     : {tkey[:8]}...{tkey[-4:]}  ({len(tkey)} chars)")
else:
    print("  Tavily key     : MISSING")

if not okey or not tkey:
    print(f"{FAIL}: One or more keys are empty — check your `.env` file")
    sys.exit(1)

print(f"{PASS}: Both keys loaded from env file")

# ── 2. Test OpenRouter API ────────────────────────────────────
sep()
print(f"[2/4] OpenRouter ({config.OPENROUTER_MODEL}) — live API call")
try:
    import httpx
    headers = {
        "Authorization": f"Bearer {okey}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Wayfinder Verification",
    }
    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: WAYFINDER_OK"}],
        "max_tokens": 20,
    }
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    reply = (msg.get("content") or msg.get("reasoning") or choice.get("finish_reason") or "OK")
    print(f"  Model reply: {reply!r}")
    print(f"{PASS}: OpenRouter API is live and responding")
except Exception as exc:
    print(f"{FAIL}: {exc}")
    sys.exit(1)

# ── 3. Test Tavily API ────────────────────────────────────────
sep()
print("[3/4] Tavily Search — live API call")
try:
    import httpx
    payload = {
        "api_key": tkey,
        "query": "what is the speed of light",
        "max_results": 2,
    }
    r = httpx.post("https://api.tavily.com/search", json=payload, timeout=20.0)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    print(f"  Results    : {len(results)} source(s) returned")
    if results:
        url = results[0].get("url", "N/A")
        title = results[0].get("title", "N/A")
        print(f"  First hit  : {title}")
        print(f"  URL        : {url}")
        print(f"{PASS}: Tavily API is live and returning results")
    else:
        print(f"{WARN}: Tavily responded but returned 0 results for the test query")
except Exception as exc:
    print(f"{FAIL}: {exc}")
    sys.exit(1)

# ── 4. Test Core Memory structures ────────────────────────────
sep()
print("[4/4] Core Memory — WorkingMemory & EpisodicMemory")
try:
    from core import memory as mem
    wm = mem.WorkingMemory()
    wm.add_message("user", "What is Python?")
    wm.add_observation("Python is a high-level programming language.")
    ctx = wm.get_context()
    assert len(ctx) == 2, f"Expected 2 messages in WorkingMemory context, got {len(ctx)}"

    em = mem.EpisodicMemory()
    em.log_thought("Thinking about the query...", step=1)
    em.log_observation("Found relevant search results", step=1)
    trace = em.get_trace()
    assert len(trace) == 1, f"Expected 1 trace step, got {len(trace)}"
    assert trace[0].thought == "Thinking about the query..."

    print(f"  WorkingMemory  : {len(ctx)} messages recorded")
    print(f"  EpisodicMemory : {len(trace)} step(s) logged")
    print(f"{PASS}: Core Memory structures verified successfully")
except Exception as exc:
    print(f"{FAIL}: {exc}")
    sys.exit(1)

# ── Summary ────────────────────────────────────────────────────
print("=" * 60)
print("ALL CHECKS PASSED")
print()
print("Start the server:")
print("  python main.py")
print()
print("Then open your browser at:")
print("  http://localhost:8000")
print("=" * 60)
