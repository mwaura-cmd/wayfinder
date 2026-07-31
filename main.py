"""
main.py — FastAPI backend for Wayfinder Research Agent.

Serves:
  GET  /          → frontend/index.html  (public — no auth required)
  POST /research  → starts agent loop, returns task_id  [auth required]
  GET  /stream/{task_id} → SSE stream of agent events   [auth required]
  GET  /history   → past research sessions               [auth required]
  GET  /health    → liveness probe for Render            [public]

  If deployed to a public server, please ensure you add an authentication layer.
  (Previously this used a WAYFINDER_API_KEY, which was removed per user request).

Deployment note:
  Binds to 0.0.0.0 so Render's router can reach the process.
  PORT is read from $PORT env var (Render sets it automatically).

  SQLite note: Render's free tier has ephemeral disk — the memory DB
  resets on each redeploy/restart. This is acceptable for a personal tool;
  the agent still works fully, just without cross-session memory after
  a restart. Paid Render disk or an external DB (e.g. Turso) would fix this.

SSE event format (one JSON object per data: line):
  {"type": "narrative", "text": "..."}
  {"type": "work",      "text": "..."}
  {"type": "answer",    "text": "...", "meta": {...}}
  {"type": "error",     "text": "..."}
  {"type": "done"}   ← sentinel — stream closes after this
"""
import asyncio
import json
import logging
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import memory as mem_module
import agent as agent_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── Startup warnings ────────────────────────────────────────────────────────────
if not config.GEMINI_API_KEY:
    log.warning("GEMINI_API_KEY is not set — the agent will not be able to call Gemini.")
if not config.TAVILY_API_KEY:
    log.warning("TAVILY_API_KEY is not set — web search will not work.")

# ── App init ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Wayfinder Research Agent",
    version="1.0.0",
    docs_url=None,   # disable Swagger UI in production
    redoc_url=None,
)

# Allow the browser to make cross-origin requests (needed when the frontend
# is opened locally while the API is on Render during development handoff)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your domain once deployed
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global SQLite connection
_db_conn = mem_module.init_db()

# In-memory task registry: task_id → queue
_task_queues: dict[str, queue.Queue] = {}
_SENTINEL = object()


# ── Public endpoints ───────────────────────────────────────────────────────────

_FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page UI — no auth required (it prompts for the key itself)."""
    if not _FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return HTMLResponse(content=_FRONTEND_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    """Liveness probe — Render pings this to confirm the service is up."""
    return {"status": "ok", "memory_available": _db_conn is not None}


# ── Protected endpoints ────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    question: str


@app.post("/research")
async def start_research(req: ResearchRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    task_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    _task_queues[task_id] = q

    def run_in_thread():
        try:
            for event in agent_module.run_agent(question, _db_conn):
                q.put(event)
        except Exception as exc:  # noqa: BLE001
            log.exception("Unhandled error in agent loop for task %s", task_id)
            q.put({"type": "error", "text": f"Internal agent error: {exc}"})
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=run_in_thread, daemon=True).start()
    return {"task_id": task_id}


@app.get("/stream/{task_id}")
async def stream_events(task_id: str, request: Request):
    if task_id not in _task_queues:
        raise HTTPException(status_code=404, detail="Task not found.")

    q = _task_queues[task_id]

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    log.info("Client disconnected from task %s", task_id)
                    break
                try:
                    event = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue

                if event is _SENTINEL:
                    yield "data: " + json.dumps({"type": "done"}) + "\n\n"
                    break

                yield "data: " + json.dumps(event) + "\n\n"

                if event.get("type") in ("answer", "error"):
                    await asyncio.sleep(0.1)
                    yield "data: " + json.dumps({"type": "done"}) + "\n\n"
                    break
        finally:
            _task_queues.pop(task_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/history")
async def get_history(limit: int = 10):
    if _db_conn is None:
        return {"sessions": [], "memory_available": False}
    try:
        rows = _db_conn.execute(
            "SELECT question, answer, sources, created_at FROM sessions "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        sessions = [
            {
                "question": r["question"],
                "answer": r["answer"],
                "sources": json.loads(r["sources"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    except Exception:
        sessions = []
    return {"sessions": sessions, "memory_available": True}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
    )
