import asyncio
import logging
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import datetime
# Assuming concrete implementations for the new architecture are written in future steps
from core.provider import ProviderRegistry
from core.tools import ToolRegistry, ToolDefinition
from core.skills import SkillRegistry
from observability.telemetry import TelemetryHub, TelemetryEvent, Payload
from orchestration.topologies import run_sequential_agent

from providers.gemini import GeminiProvider
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

# ── Register Components ────────────────────────────────────────────────────────
ProviderRegistry.register("gemini", GeminiProvider(model_name="gemini-1.5-flash"))

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

class ResearchRequest(BaseModel):
    question: str

@app.post("/research")
async def start_research(req: ResearchRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    task_id = str(uuid.uuid4())
    
    async def bg_task():
        try:
            result = await run_sequential_agent(
                prompt=question,
                provider_id="gemini",
                tool_categories=["search"],
                skill_domain="",
                role_prompt=(
                    "You are Wayfinder, a precise and diligent web research agent.\n"
                    "Your job is to answer the user's question by searching the web methodically\n"
                    "and synthesising what you find — not by recalling training data.\n\n"
                    "Always use web_search to look up current information before answering.\n"
                    "When you have enough evidence, produce your Final Answer."
                ),
                telemetry=telemetry_hub,
                track_id=task_id,   # ← same ID the frontend subscribes to
            )
            log.info(f"Task {task_id} completed. Success: {result.success}")
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
async def get_history(limit: int = 10):
    return {"sessions": [], "message": "Semantic memory not yet fully wired for global history"}
