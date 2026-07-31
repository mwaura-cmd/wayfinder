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

gkey = config.GEMINI_API_KEY
tkey = config.TAVILY_API_KEY

if gkey:
    print(f"  Gemini key : {gkey[:8]}...{gkey[-4:]}  ({len(gkey)} chars)")
else:
    print("  Gemini key : MISSING")

if tkey:
    print(f"  Tavily key : {tkey[:8]}...{tkey[-4:]}  ({len(tkey)} chars)")
else:
    print("  Tavily key : MISSING")

if not gkey or not tkey:
    print(f"{FAIL}: One or more keys are empty — check your `env` file")
    sys.exit(1)

print(f"{PASS}: Both keys loaded from env file")

# ── 2. Test Gemini API ────────────────────────────────────────
sep()
print("[2/4] Gemini 2.5 Flash — live API call (google-genai SDK)")
try:
    from google import genai
    client = genai.Client(api_key=gkey)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with exactly: WAYFINDER_OK",
    )
    text = resp.text.strip()
    print(f"  Model reply: {text!r}")
    print(f"{PASS}: Gemini API is live and responding")
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

# ── 4. Test SQLite memory ──────────────────────────────────────
sep()
print("[4/4] SQLite Memory — init, write, read, cleanup")
try:
    import memory as mem
    import tempfile
    import pathlib

    tmp = pathlib.Path(tempfile.gettempdir()) / "wayfinder_verify_test.db"
    conn = mem.init_db(tmp)
    assert conn is not None, "init_db returned None"

    mem.save_session(
        conn,
        question="What is Python used for?",
        answer="Python is a general-purpose programming language.",
        sources=["https://python.org"],
    )
    mem.save_session(
        conn,
        question="How does FastAPI handle requests?",
        answer="FastAPI uses async I/O and Pydantic models.",
        sources=["https://fastapi.tiangolo.com"],
    )

    hits = mem.lookup_memory(conn, ["Python"])
    assert len(hits) > 0, "lookup_memory returned empty for keyword 'Python'"

    print(f"  Stored     : 2 test sessions")
    print(f"  Lookup hit : {hits[0]['question']!r}")

    conn.close()
    tmp.unlink(missing_ok=True)
    print(f"{PASS}: SQLite memory init, write, read, and cleanup all work")
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
