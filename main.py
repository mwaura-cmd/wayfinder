import asyncio
import logging
import uuid
import json
import base64
import io
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import memory
from core.provider import ProviderRegistry
from core.tools import ToolRegistry, ToolDefinition
from core.skills import SkillRegistry
from observability.telemetry import TelemetryHub, TelemetryEvent, Payload
from orchestration.topologies import run_sequential_agent
from core.auth import get_current_user
from core.firebase import init_firebase, get_db
from providers.openrouter import OpenRouterProvider
from providers.groq import GroqProvider
from tools.tavily import TavilySearchExecutor
from llm_provider import get_llm_client_and_model

# Optional: Gemini provider (only registered when key is configured)
try:
    if config.GEMINI_API_KEY:
        from providers.gemini import GeminiProvider
        _GEMINI_AVAILABLE = True
    else:
        _GEMINI_AVAILABLE = False
except Exception:
    _GEMINI_AVAILABLE = False

# Optional: Firestore server timestamp (graceful import)
try:
    from google.cloud import firestore as _firestore
    _FIRESTORE_TIMESTAMP = _firestore.SERVER_TIMESTAMP
except Exception:
    _FIRESTORE_TIMESTAMP = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

# Initialize memory database and Firebase on startup
memory_conn = memory.init_db()
firebase_app = init_firebase()

app = FastAPI(title="Wayfinder Research Agent v2", version="2.0.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=Path(__file__).parent / "frontend" / "assets"), name="assets")

# ── Register Components ────────────────────────────────────────────────────────
ProviderRegistry.register("openrouter", OpenRouterProvider())
ProviderRegistry.register("groq", GroqProvider())
if _GEMINI_AVAILABLE:
    ProviderRegistry.register("gemini", GeminiProvider())
    log.info("Gemini provider registered (GEMINI_API_KEY is set).")

ToolRegistry.register(ToolDefinition(
    name="web_search",
    description="Search the web for up-to-date information.",
    category="search",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."}
        },
        "required": ["query"]
    },
    output_schema={"type": "string"},
    timeout_seconds=20,
    executor=TavilySearchExecutor()
))

from core.tools import ToolRegistry, ToolDefinition, BaseToolExecutor, ToolResult, ToolError

class MemorySearchExecutor(BaseToolExecutor):
    async def execute(self, inputs: dict, telemetry: Any, track_id: str) -> ToolResult:
        query = inputs.get("query", "")
        keywords = inputs.get("keywords")
        conn = memory.init_db()
        words = keywords or ([query] if isinstance(query, str) and query else [])
        if isinstance(query, str) and not keywords:
            words = [w for w in query.split() if len(w) > 3]
        results = memory.lookup_memory(conn, words, limit=config.MEMORY_LOOKUP_LIMIT)
        if not results:
            output = "No previous research found matching those keywords."
        else:
            lines = []
            for r in results:
                srcs = ", ".join(r.get("sources", []))
                lines.append(f"Past Research on '{r['question']}':\n{r['answer']}\nSources: {srcs}")
            output = "\n\n".join(lines)
        return ToolResult(
            call_id="",
            tool_name="check_memory",
            output=output,
            raw={"results": results},
            success=True
        )


ToolRegistry.register(ToolDefinition(
    name="check_memory",
    description="Check internal past research memory for previously answered topics and verified facts.",
    category="search",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords or topic to look up in memory."}
        },
        "required": ["query"]
    },
    output_schema={"type": "string"},
    timeout_seconds=5,
    executor=MemorySearchExecutor()
))

telemetry_hub = TelemetryHub()
_FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if not _FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return HTMLResponse(content=_FRONTEND_PATH.read_text(encoding="utf-8"))

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0"}

def extract_text_from_attachment(name: str, file_type: str, data: str) -> str:
    try:
        if "," in data:
            data = data.split(",", 1)[1]
        raw_bytes = base64.b64decode(data)
        if name.lower().endswith('.pdf') or 'pdf' in file_type.lower():
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
                return f"\n--- Attached Document: {name} ---\n{text.strip()}\n"
            except Exception as pdf_err:
                return f"\n--- Attached Document: {name} (PDF parsing error: {pdf_err}) ---\n"
        elif name.lower().endswith('.docx') or 'word' in file_type.lower():
            try:
                import docx
                doc = docx.Document(io.BytesIO(raw_bytes))
                text = "\n".join([p.text for p in doc.paragraphs])
                return f"\n--- Attached Document: {name} ---\n{text.strip()}\n"
            except Exception as docx_err:
                return f"\n--- Attached Document: {name} (DOCX parsing error: {docx_err}) ---\n"
        else:
            text = raw_bytes.decode('utf-8', errors='replace')
            return f"\n--- Attached Document: {name} ---\n{text.strip()}\n"
    except Exception as e:
        log.error(f"Error parsing attachment {name}: {e}")
        return f"\n--- Attached Document: {name} (Failed to parse: {e}) ---\n"

class FileAttachment(BaseModel):
    name: str
    type: str
    data: str

class ResearchRequest(BaseModel):
    question: str
    thread_id: Optional[int] = None
    level: Optional[str] = "standard"
    provider: Optional[str] = None  # override LLM provider per-request
    attachments: Optional[List[FileAttachment]] = None

class FeedbackRequest(BaseModel):
    feedback: Optional[str] = None  # "up" | "down" | None
    feedback_note: Optional[str] = None

@app.post("/research")
async def start_research(req: ResearchRequest, uid: str = Depends(get_current_user)):
    question = req.question.strip()
    if not question and not req.attachments:
        raise HTTPException(status_code=400, detail="Question or attachments required.")

    if not question:
        question = "Analyze and summarize the attached document(s)."

    full_prompt = question
    if req.attachments:
        attached_text = "".join([extract_text_from_attachment(att.name, att.type, att.data) for att in req.attachments])
        full_prompt = f"{question}\n\n[USER ATTACHMENTS / CONTEXT]\n{attached_text}"

    task_id = str(uuid.uuid4())
    req_level = req.level if req.level in config.RESEARCH_LEVELS else config.DEFAULT_RESEARCH_LEVEL

    # Resolve or create thread upfront scoped to user UID
    conn = memory.init_db()
    active_thread_id = req.thread_id
    prior_messages = []
    if conn:
        active_thread_id = memory.get_or_create_thread(conn, req.thread_id, question, user_id=uid)
        if req.thread_id:
            all_msgs = memory.get_thread_messages(conn, req.thread_id, user_id=uid)
            # Pass up to the last 6 messages to keep prompt focused
            prior_messages = all_msgs[-6:] if len(all_msgs) > 6 else all_msgs

    # Resolve provider: use request override if valid, else fall back to config default
    requested_provider = req.provider if req.provider else config.LLM_PROVIDER
    try:
        ProviderRegistry.get(requested_provider)
        active_provider = requested_provider
    except ValueError:
        log.warning(f"Unknown provider '{requested_provider}' requested — falling back to '{config.LLM_PROVIDER}'.")
        active_provider = config.LLM_PROVIDER

    # Tool categories: only include memory search for initial topic queries, not follow-ups
    tool_cats = ["search"]

    async def bg_task():
        try:
            # Validate provider configuration upfront (yields clean error if key missing or invalid provider)
            get_llm_client_and_model(config.LLM_PROVIDER)
        except Exception as e:
            err_msg = str(e)
            log.error(f"Task {task_id} configuration error: {err_msg}")
            await telemetry_hub.emit(TelemetryEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                track_id=task_id,
                parent_track_id=None,
                source_type="agent",
                payload=Payload(
                    type="error",
                    text=err_msg
                )
            ))
            return

        try:
            result = await run_sequential_agent(
                prompt=full_prompt,
                provider_id=active_provider,
                tool_categories=tool_cats,
                skill_domain="",
                role_prompt=(
                    "You are Wayfinder, an elite, deep web research engine.\n"
                    "Your mission is to rigorously research the user's question, gather corroborated evidence across high-quality sources, and synthesize an insightful, comprehensive, expert-level response.\n\n"
                    "Research Guidelines:\n"
                    "- Always search the web for up-to-date and accurate information before answering.\n"
                    "- If documents are attached, thoroughly incorporate their context.\n"
                    "- Cross-verify facts and key claims across multiple reputable sources.\n"
                    "- When you have gathered sufficient evidence, output your Final Answer directly without narrating intent beforehand.\n\n"
                    "Synthesis & Final Answer Formatting Rules:\n"
                    "1. OPENING EXECUTIVE SYNTHESIS: Always open your final answer directly with an engaging, high-level executive summary paragraph (1–3 sentences) that answers the core question, frames the macro context, or highlights the defining shift/takeaway (e.g. \"2025 was the year fusion moved from 'is it possible?' to 'how fast can we build it?' — the sector's defining shift was from research physics to industrial deployment. Here's what mattered most.\"). Do NOT include label headers like 'Executive Summary:' or 'Opening Synthesis:' — begin directly with the introductory text itself. NEVER jump directly into numbered lists, raw bullet points, or dry breakdowns without this opening lead paragraph.\n"
                    "2. STRUCTURED THEMATIC DEEP-DIVE: Follow the executive lead with structured thematic sections using markdown headers (###), bolded concept titles (e.g. 1. **Sector Milestones & Deployment Race**: ...), concrete data points, metrics, dates, and key organization/project names.\n"
                    "3. COMPARISONS & TABLES: Use clean markdown tables where comparisons or chronological data make findings easier to digest.\n"
                    "4. TONE & WRITING STYLE: Write with the depth, clarity, and authority of a top-tier analyst report. Avoid robotic meta-statements like 'Based on my search results' or 'Here is what I found'."
                ),
                telemetry=telemetry_hub,
                track_id=task_id,   # ← same ID the frontend subscribes to
                level=req_level,
                prior_messages=prior_messages,
            )
            log.info(f"Task {task_id} completed. Success: {result.success}, Model: {result.model_used}, Level: {result.level}")

            # Safety check: ensure answer text is never empty or placeholder
            persisted_answer = (result.final_output or "").strip()
            if not persisted_answer:
                log.warning(f"Task {task_id}: result.final_output is empty; falling back to episodic summary.")
                persisted_answer = (result.episodic_summary or "").strip()
            if not persisted_answer:
                persisted_answer = f"Research concluded for '{question}'."

            # 1. Save to SQLite memory store (threads & messages) scoped to user UID
            db_conn = memory.init_db()
            saved_msg_id = 0
            if db_conn:
                saved_msg_id = memory.save_message(
                    conn=db_conn,
                    thread_id=active_thread_id,
                    question=question,
                    answer=persisted_answer,
                    sources=result.sources,
                    model_used=result.model_used,
                    level=result.level,
                    user_id=uid,
                )
                log.info(f"Saved message {saved_msg_id} to thread {active_thread_id} for user {uid}")

            # 2. Save session to Firestore with full telemetry events, sources, and metrics
            db = get_db()
            if db:
                try:
                    trace = telemetry_hub.get_trace(task_id)
                    serialized_events = [ev.model_dump() for ev in trace] if trace else []

                    # Normalize sources
                    formatted_sources = []
                    for s in (result.sources or []):
                        if isinstance(s, dict):
                            formatted_sources.append(s)
                        elif isinstance(s, str):
                            formatted_sources.append({"url": s, "title": s})

                    doc_data = {
                        "task_id": task_id,
                        "thread_id": active_thread_id,
                        "message_id": saved_msg_id,
                        "question": question,
                        "success": result.success,
                        "final_answer": persisted_answer,
                        "sources": formatted_sources,
                        "model_used": result.model_used,
                        "level": result.level,
                        "steps_taken": result.steps_taken,
                        "search_count": result.search_count,
                        "elapsed_seconds": result.elapsed_seconds,
                        "events": serialized_events,
                        "uid": uid,
                    }
                    # Use SERVER_TIMESTAMP only when the firestore module imported correctly
                    if _FIRESTORE_TIMESTAMP is not None:
                        doc_data["created_at"] = _FIRESTORE_TIMESTAMP
                    db.collection("sessions").document(task_id).set(doc_data)
                    log.info(f"Task {task_id} saved to Firestore sessions collection for user {uid}.")
                except Exception as db_err:
                    log.error(f"Failed to save session {task_id} to Firestore: {db_err}")

        except Exception as e:
            log.exception(f"Error in agent execution for task {task_id}")
            asyncio.create_task(telemetry_hub.emit(TelemetryEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                track_id=task_id,
                parent_track_id=None,
                track_type="root",
                source_type="system",
                payload=Payload(
                    type="error_event",
                    error={"message": f"Execution failed: {str(e)}", "code": "system_error", "recoverable": False}
                )
            )))

    asyncio.create_task(bg_task())
    return {
        "task_id": task_id,
        "thread_id": active_thread_id,
        "level": req_level,
        "provider": active_provider,
    }

@app.get("/stream/{task_id}")
async def stream_events(task_id: str, request: Request):
    return StreamingResponse(
        telemetry_hub.get_stream(task_id),
        media_type="text/event-stream"
    )

@app.get("/providers")
async def list_providers():
    """Return all registered LLM providers and the current default."""
    registered = ProviderRegistry.list()
    return {
        "providers": registered,
        "default": config.LLM_PROVIDER,
        "available": {
            "openrouter": bool(config.OPENROUTER_API_KEY),
            "groq": bool(config.GROQ_API_KEY),
            "gemini": _GEMINI_AVAILABLE,
        }
    }

@app.get("/history")
async def get_history(limit: int = 30, uid: str = Depends(get_current_user)):
    conn = memory.init_db()
    threads = memory.get_threads(conn, user_id=uid, limit=limit) if conn else []
    
    # Return threads both under "threads" and "sessions" for full backward compatibility
    return {
        "threads": threads,
        "sessions": threads,
        "message": "Success"
    }

@app.get("/thread/{thread_id}")
async def get_thread_detail(thread_id: int, uid: str = Depends(get_current_user)):
    conn = memory.init_db()
    if not conn:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    thread = memory.get_thread(conn, thread_id, user_id=uid)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return {"thread": thread, "message": "Success"}

@app.get("/session/{identifier}")
async def get_session_or_thread(identifier: str, uid: str = Depends(get_current_user)):
    conn = memory.init_db()
    
    # Check if identifier is an integer thread ID
    if identifier.isdigit() and conn:
        thread = memory.get_thread(conn, int(identifier), user_id=uid)
        if thread:
            return {"thread": thread, "session": thread, "message": "Success"}

    # Fallback to Firestore session lookup
    db = get_db()
    if db:
        try:
            doc = db.collection("sessions").document(identifier).get()
            if doc.exists:
                data = doc.to_dict()
                if data.get("uid") == uid:
                    created_at = data.get("created_at")
                    if created_at and hasattr(created_at, "isoformat"):
                        data["created_at"] = created_at.isoformat()
                    return {"session": data, "message": "Success"}
        except Exception as e:
            log.error(f"Error fetching session {identifier} from Firestore: {e}")

    # Fallback to thread check
    if conn and identifier.startswith("thread_"):
        try:
            tid = int(identifier.split("_")[1])
            thread = memory.get_thread(conn, tid, user_id=uid)
            if thread:
                return {"thread": thread, "session": thread, "message": "Success"}
        except Exception:
            pass

    raise HTTPException(status_code=404, detail="Session or thread not found.")

@app.post("/messages/{message_id}/feedback")
async def update_message_feedback(message_id: int, req: FeedbackRequest, uid: str = Depends(get_current_user)):
    conn = memory.init_db()
    if not conn:
        raise HTTPException(status_code=500, detail="Database unavailable")

    success = memory.update_feedback(conn, message_id, req.feedback, req.feedback_note, user_id=uid)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found or update failed")

    return {
        "status": "ok",
        "message_id": message_id,
        "feedback": req.feedback,
        "feedback_note": req.feedback_note
    }
