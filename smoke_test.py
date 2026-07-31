"""smoke_test.py — Live end-to-end test against the running Wayfinder server."""
import httpx
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000"
QUESTION = "What year was Python first released?"

print("Live smoke test: POST /research + SSE stream")
print("-" * 55)

# 1. Start task
r = httpx.post(f"{BASE}/research", json={"question": QUESTION}, timeout=10)
r.raise_for_status()
task_id = r.json()["task_id"]
print(f"Task ID: {task_id}")
print()

# 2. Stream events
event_types = []
with httpx.stream("GET", f"{BASE}/stream/{task_id}", timeout=180) as s:
    for line in s.iter_lines():
        if not line.startswith("data:"):
            continue
        event = json.loads(line[5:].strip())
        etype = event.get("type", "?")
        event_types.append(etype)

        text = event.get("text", "")
        if etype == "narrative":
            print(f"  NARRATIVE : {text[:90]}")
        elif etype == "work":
            print(f"  WORK      : {text[:90]}")
        elif etype == "answer":
            meta = event.get("meta", {})
            print(f"  ANSWER    : {text[:120]}")
            print(f"  Sources   : {meta.get('source_count', 0)} | Turns: {meta.get('turn_count', 0)} | Time: {meta.get('elapsed_seconds', '?')}s")
        elif etype == "error":
            print(f"  ERROR     : {text}")
        elif etype == "done":
            break

print()
print("Events seen:", event_types)
print()

if "answer" in event_types and "work" in event_types:
    print("PASS: Full agent pipeline works end-to-end")
else:
    print("FAIL: Missing expected events")
    sys.exit(1)
