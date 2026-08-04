import asyncio
import logging
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import datetime
# Assuming concrete implementations for the new architecture are written in future steps
from core.provider import ProviderRegistry
from core.tools import ToolRegistry, ToolDefinition
from core.skills import SkillRegistry
from observability.telemetry import TelemetryHub, TelemetryEvent, Payload
from orchestration.topologies import run_sequential_agent
from core.firebase import get_db
from firebase_admin import firestore
from core.memory import FirebaseSemanticMemoryStore
from core.auth import get_current_user

from providers.openrouter import OpenRouterProvider
from tools.tavily import TavilySearchExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

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

import base64
import io
from typing import Optional, List

def extract_text_from_attachment(name: str, file_type: str, data: str) -> str:
    try:
        # If it's a data URL, strip the prefix
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
    attachments: Optional[List[FileAttachment]] = None

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
    
    async def bg_task():
        try:
            result = await run_sequential_agent(
                prompt=full_prompt,
                provider_id="openrouter",
                tool_categories=["search"],
                skill_domain="",
                role_prompt=(
                    "You are Wayfinder, a precise and diligent web research agent.\n"
                    "Your job is to answer the user's question by searching the web methodically\n"
                    "and synthesising what you find — not by recalling training data.\n\n"
                    "If the user provides attached documents, incorporate their content into your analysis.\n"
                    "Always use web_search to look up current information before answering.\n"
                    "When you have enough evidence, produce your Final Answer directly without narrating intent beforehand."
                ),
                telemetry=telemetry_hub,
                track_id=task_id,   # ← same ID the frontend subscribes to
            )
            log.info(f"Task {task_id} completed. Success: {result.success}")
            
            # Save session to Firestore
            db = get_db()
            if db:
                try:
                    db.collection("sessions").document(task_id).set({
                        "task_id": task_id,
                        "question": question,
                        "success": result.success,
                        "final_answer": result.final_output,
                        "created_at": firestore.SERVER_TIMESTAMP,
                        "uid": uid,
                    })
                    log.info(f"Task {task_id} saved to Firestore sessions collection for user {uid}.")
                    
                    # Also save to semantic memory for future retrieval
                    semantic_store = FirebaseSemanticMemoryStore()
                    await semantic_store.store(
                        key=task_id,
                        content=result.final_output,
                        metadata={"question": question, "task_id": task_id, "uid": uid}
                    )
                    log.info(f"Task {task_id} saved to semantic memory.")
                except Exception as db_err:
                    log.error(f"Failed to save session {task_id} to Firestore: {db_err}")
                    
        except Exception as e:
            log.exception(f"Error in agent execution for task {task_id}")
            # Emit an error event so the frontend UI doesn't hang indefinitely waiting for completion
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
    return {"task_id": task_id}

@app.get("/stream/{task_id}")
async def stream_events(task_id: str, request: Request):
    return StreamingResponse(
        telemetry_hub.get_stream(task_id),
        media_type="text/event-stream"
    )

@app.get("/history")
async def get_history(limit: int = 10, uid: str = Depends(get_current_user)):
    db = get_db()
    if not db:
        return {"sessions": [], "message": "Firebase not initialized. Semantic memory offline."}
    
    try:
        # Filter by uid and order by created_at descending
        docs = db.collection("sessions")\
            .where("uid", "==", uid)\
            .order_by("created_at", direction=firestore.Query.DESCENDING)\
            .limit(limit).stream()
        sessions = []
        for doc in docs:
            data = doc.to_dict()
            # Convert timestamp to ISO string if present
            created_at = data.get("created_at")
            if created_at and hasattr(created_at, "isoformat"):
                data["created_at"] = created_at.isoformat()
            sessions.append(data)
            
        return {"sessions": sessions, "message": "Success"}
    except Exception as e:
        log.error(f"Error fetching history from Firestore: {e}")
        return {"sessions": [], "message": f"Error fetching history: {e}"}
