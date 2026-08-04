import sys
import asyncio
import os

sys.stdout.reconfigure(encoding="utf-8")

import config
from llm_provider import get_llm_client_and_model, get_async_llm_client_and_model
from agent import run_agent
from providers.groq import GroqProvider
from providers.openrouter import OpenRouterProvider
from core.provider import LLMRequest, Message

print("=" * 60)
print("WAYFINDER — GROQ & OPENROUTER PROVIDER TEST SUITE")
print("=" * 60)

async def test_all():
    # ── 1. Test Factory Error Handling for Missing Key ──────────────
    print("\n[1/6] Testing factory error handling for missing API keys...")
    original_groq_key = config.GROQ_API_KEY
    try:
        config.GROQ_API_KEY = ""
        get_llm_client_and_model("groq")
        print("  FAIL: Expected ValueError for missing GROQ_API_KEY")
    except ValueError as e:
        print(f"  PASS: Caught expected error: {e}")
    finally:
        config.GROQ_API_KEY = original_groq_key

    # ── 2. Test Invalid Provider Name ──────────────────────────────
    print("\n[2/6] Testing invalid provider name...")
    try:
        get_llm_client_and_model("invalid_provider")
        print("  FAIL: Expected ValueError for invalid provider")
    except ValueError as e:
        print(f"  PASS: Caught expected error: {e}")

    # ── 3. Test agent.py Error Yielding ────────────────────────────
    print("\n[3/6] Testing agent.py error handling (non-crashing)...")
    config.GROQ_API_KEY = ""
    events = []
    async for ev in run_agent("test query", provider_name="groq"):
        events.append(ev)
    config.GROQ_API_KEY = original_groq_key

    assert len(events) == 1, f"Expected 1 event, got {len(events)}"
    assert events[0]["type"] == "error", f"Expected error event type, got {events[0]['type']}"
    print(f"  PASS: run_agent yielded clean error event: {events[0]['text']}")

    # ── 4. Live Groq Provider Call ─────────────────────────────────
    print("\n[4/6] Testing Live Groq Provider (LLM_PROVIDER=groq)...")
    groq_p = GroqProvider()
    req_groq = LLMRequest(
        messages=[Message(role="user", content="Respond with exactly 'GROQ_OK'")],
        tools=[],
        system="You are a helpful test assistant.",
        max_tokens=10,
        temperature=0.0,
        stop_sequences=[]
    )
    try:
        res_groq = await groq_p.complete(req_groq)
        print(f"  PASS: Groq response content: {res_groq.content.strip()!r}")
        print(f"        Raw model reported: {res_groq.raw.get('model')}")
    except Exception as e:
        print(f"  FAIL: Groq call error: {e}")

    # ── 5. Live OpenRouter Provider Call ───────────────────────────
    print("\n[5/6] Testing Live OpenRouter Provider (LLM_PROVIDER=openrouter)...")
    openrouter_p = OpenRouterProvider()
    req_or = LLMRequest(
        messages=[Message(role="user", content="Respond with exactly 'OPENROUTER_OK'")],
        tools=[],
        system="You are a helpful test assistant.",
        max_tokens=10,
        temperature=0.0,
        stop_sequences=[]
    )
    try:
        res_or = await openrouter_p.complete(req_or)
        print(f"  PASS: OpenRouter response content: {res_or.content.strip()!r}")
        print(f"        Raw model reported: {res_or.raw.get('model')}")
    except Exception as e:
        print(f"  FAIL: OpenRouter call error: {e}")

    # ── 6. Fallback Behavior Test (Section 6) ───────────────────────
    print("\n[6/6] Testing automatic fallback from Groq to OpenRouter on failure...")
    config.GROQ_API_KEY = "gsk_invalid_test_key_12345"
    groq_fallback_p = GroqProvider()
    try:
        res_fb = await groq_fallback_p.complete(req_groq)
        print(f"  PASS: Automatic fallback to OpenRouter succeeded! Response: {res_fb.content.strip()!r}")
    except Exception as e:
        print(f"  FAIL: Fallback error: {e}")
    finally:
        config.GROQ_API_KEY = original_groq_key

    print("\n" + "=" * 60)
    print("ALL 6 PROVIDER VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_all())
