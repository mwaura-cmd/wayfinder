# Wayfinder — Web Research Agent

A **real ReAct agent** that runs a plan → search → observe → decide loop,
streams its reasoning live to a browser UI, and remembers what it has
researched across sessions.

## Design notes (confirming §2–§10)

> **Security scope (§1):** The FastAPI server binds to `127.0.0.1` only and
> has **no authentication layer**. This is a deliberate design choice for a
> single-user personal tool running locally. Do NOT deploy this as-is to a
> public host or multi-user environment — add an auth middleware layer and
> change the bind address before doing so.

---

### §2 — Agent loop
Implemented in `agent.py`. The model is called in a `while inner_turn <= MAX_TURNS`
loop. On each iteration the model either emits tool calls (→ execute + continue)
or produces text (→ parse as final answer). Multi-part questions are handled by
the system prompt rule: *"call `web_search` once per sub-question with a focused
query."* If the cap is hit, the model is explicitly asked to produce its best
partial answer and must say so — a finished-looking fabricated answer is never
produced.

### §3 — Tool design
Two tools only: `web_search` and `check_memory`. Descriptions in
`tools.py::TOOL_DECLARATIONS` are written to be unambiguous. All errors
(timeout, 429, empty results, unknown DB) are caught inside the tool functions
and returned as descriptive plain-text strings — no raw exceptions ever reach
the model or the user.

### §4 — Context management
`web_search` returns only `title + URL + snippet (≤600 chars)` per result,
never the full page dump. Additionally, `agent.py::_compact_history` replaces
raw tool-result content from turns older than `_COMPACTION_THRESHOLD` with a
one-line placeholder. The model's understanding extracted from those results is
already encoded in its subsequent reasoning turns, so the raw text is no longer
needed. This keeps token usage bounded across long multi-search sessions.

### §5 — Guardrails
No write/send/delete-capable external tool exists. `web_search` is read-only
(Tavily GET). `check_memory` is read-only (SQLite SELECT). The only write
action is `memory.save_session()`, called in `agent.py` after task completion —
this is the agent writing to its own local store, not an external service.
**If any future tool is added with write/send/delete capability, a
confirm-before-acting step is mandatory before that tool call executes.**
Goal drift is handled by the system prompt: *"periodically remind yourself of
the ORIGINAL question"* and *"each search turn should connect back to answering
that original question."*

### §6 — Failure modes
- **Loop without converging:** `MAX_TURNS` cap (default 8) in `agent.py`.
- **Hallucinated search result:** System prompt explicitly forbids filling gaps
  from training data; empty-result tool responses surface a clear "no results"
  message so the model cannot pretend it found something.
- **Silent goal drift:** System prompt instructs the model to periodically
  restate the original goal internally.
- **Rate limits:** `tools.py::_with_retry` and `agent.py::_call_gemini_with_retry`
  implement exponential backoff (1s → 2s → 4s, 3 attempts max) for both
  Tavily and Gemini. HTTP 429 produces a specific human-readable message:
  *"Tavily's free-tier limit was hit — try again in ~1 minute."*
- **Memory store errors:** `memory.init_db()` catches `sqlite3.DatabaseError`
  and returns `None`. All memory functions (`save_session`, `lookup_memory`,
  `check_memory`) check for `conn is None` and degrade gracefully — the agent
  runs with no past-session context rather than crashing.

### §7 — Architecture
Single agent, two tools. No multi-agent design. No LangGraph/CrewAI.

### §8 — Execution UI
- **Backend:** FastAPI + SSE (`/stream/{task_id}`). The agent runs in a daemon
  thread; events are queued and streamed to the browser in real time — not
  buffered and sent all at once.
- **Frontend:** `frontend/index.html` — single file, no build step.
  - **Layer 1 (narrative nodes):** first-person conversational statements,
    each rendered as an expandable timeline node with a pulsing dot.
  - **Layer 2 (work panels):** search queries, source counts, memory results —
    evidence of work, never raw chain-of-thought.
  - **Active/inactive state:** only one node active at a time; previous nodes
    collapse and remain clickable as an audit trail.
  - **Completion summary:** commit-message style (e.g. "Reviewed 6 sources and
    answered in 38s"), never generic "Done". The full execution trail is
    preserved client-side in `allEvents[]` and toggleable via the summary bar.
  - No execution info is ever discarded — all events accumulate in `allEvents`.

### §9 — Persistent memory
SQLite (`wayfinder_memory.db`), Python stdlib only. Schema: `sessions` table
with `question, answer, sources (JSON), created_at`. `check_memory` tool uses
keyword LIKE matching — no embeddings (deliberate; would over-engineer a
hobby-scale tool). Retention cap: `MAX_MEMORY_SESSIONS=200`; pruning is logged.
Memory is append-only from the agent's perspective — no autonomous delete.
Memory hits surface as their own narrative node in the UI.

### §10 — Evaluation scenarios

| # | Query | Pass condition | Fail condition |
|---|-------|---------------|----------------|
| 1 | *"What is the capital of Australia?"* | Single `check_memory` call + ≤2 `web_search` calls; clear answer with source | Multiple redundant searches; hallucinated answer without search |
| 2 | *"Who founded SpaceX and when did NASA first partner with them?"* | ≥2 separate web searches (founding vs. NASA partnership); both sub-questions answered with distinct sources | Single combined query; one sub-question left unanswered |
| 3 | *"Is Python 3.12 or 3.13 faster for data processing?"* | Agent re-searches when sources conflict (e.g. different benchmark results); explicitly notes disagreement | Picks one conflicting source and presents it as definitive fact |
| 4 | Run query #1 again, or ask a closely related factual question | `check_memory` hits on the first session; narrative node says *"I've researched something similar before…"*; search credits saved | Memory is never checked; agent starts web search from zero |

---

## Stack

| Component | Choice | Free tier |
|-----------|--------|-----------|
| LLM | Gemini 2.5 Flash | 15 req/min, 1500 req/day — no card |
| Search | Tavily | 1,000 credits/month — no card |
| Orchestration | Plain Python loop | — |
| Backend | FastAPI + uvicorn | — |
| Memory | SQLite (stdlib) | — |
| Frontend | Vanilla HTML/JS/CSS | — |

---

## Setup

### 1. Get free API keys

**Gemini API key (Google AI Studio)**
1. Go to → https://aistudio.google.com/app/apikey
2. Click **Create API key** → copy it.

**Tavily API key**
1. Go to → https://app.tavily.com/sign-up (no credit card required)
2. After sign-up, your API key is on the dashboard → copy it.

### 2. Set environment variables

**Windows (PowerShell — current session)**
```powershell
$env:GEMINI_API_KEY = "AIza..."
$env:TAVILY_API_KEY = "tvly-..."
```

**Windows (permanent, user-level)**
```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY","AIza...","User")
[System.Environment]::SetEnvironmentVariable("TAVILY_API_KEY","tvly-...","User")
```

**Linux / macOS**
```bash
export GEMINI_API_KEY="AIza..."
export TAVILY_API_KEY="tvly-..."
```

Or create a `.env` file and use `python-dotenv` (optional).

### 3. Install dependencies

```bash
pip install fastapi uvicorn google-generativeai httpx
```

Full pinned requirements (optional):
```
fastapi>=0.111
uvicorn[standard]>=0.29
google-generativeai>=0.8
httpx>=0.27
```

### 4. Run the server

```bash
cd wayfinder
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 5. Open the UI

Navigate to → **http://localhost:8000**

---

## Configuration

Edit `config.py` to change:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Primary LLM |
| `MAX_TURNS` | `8` | Loop cap per task |
| `MAX_RETRIES` | `3` | Retry attempts on API errors |
| `TAVILY_MAX_RESULTS` | `5` | Results per search |
| `SNIPPET_MAX_CHARS` | `600` | Chars kept per result snippet |
| `MAX_MEMORY_SESSIONS` | `200` | SQLite retention cap |
| `DB_PATH` | `./wayfinder_memory.db` | Memory DB location |

---

## Project structure

```
wayfinder/
├── main.py               # FastAPI backend + SSE endpoints
├── agent.py              # ReAct loop — system prompt, Gemini dispatch
├── tools.py              # web_search + check_memory (with retry logic)
├── memory.py             # SQLite persistent memory (§9)
├── config.py             # All tunables in one place
├── wayfinder_memory.db   # Created automatically on first run
└── frontend/
    └── index.html        # Complete browser UI (no build step)
```

---

## Clearing memory manually

The agent never deletes memory autonomously. To clear it manually:

```bash
# Option 1 — delete the file (agent recreates it)
del wayfinder_memory.db

# Option 2 — keep the file, wipe the table
python -c "import sqlite3; c=sqlite3.connect('wayfinder_memory.db'); c.execute('DELETE FROM sessions'); c.commit()"
```
