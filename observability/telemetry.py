import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncIterator, Literal, Optional, Union
from pydantic import BaseModel, Field

# Matches the spec's Telemetry Error payload
class TelemetryError(BaseModel):
    code: str
    message: str
    recoverable: bool

# Payload schema per spec
class Payload(BaseModel):
    type: Literal["narrative_start", "work_delta", "track_complete", "run_complete", "error_event"]
    node_id: Optional[str] = None
    narrative: Optional[str] = None
    label: Optional[str] = None
    detail: Optional[str] = None
    summary: Optional[str] = None
    final_output: Optional[str] = None
    error: Optional[TelemetryError] = None

# Unified Event Schema per spec
class TelemetryEvent(BaseModel):
    event_id: str
    timestamp: str
    track_id: str
    parent_track_id: Optional[str] = None
    track_type: Literal["root", "subagent", "parallel_worker"]
    source_type: Literal["agent", "tool", "skill", "system"]
    payload: Payload

class TelemetryHub:
    def __init__(self):
        # We need a way to broadcast events to SSE subscribers.
        # Simple implementation: queues per track_id.
        self._listeners: dict[str, list[asyncio.Queue]] = {}
        self._history: dict[str, list[TelemetryEvent]] = {}

    async def emit(self, event: TelemetryEvent) -> None:
        track_id = event.track_id
        
        # Store in history
        if track_id not in self._history:
            self._history[track_id] = []
        self._history[track_id].append(event)
        
        # Also map to root parent if it's a child (for unified streaming of a whole run)
        root_id = event.parent_track_id if event.parent_track_id else track_id
        if root_id != track_id:
            if root_id not in self._history:
                self._history[root_id] = []
            self._history[root_id].append(event)
            
        # Broadcast to listeners of the root
        if root_id in self._listeners:
            for q in self._listeners[root_id]:
                await q.put(event)

    async def get_stream(self, track_id: str) -> AsyncIterator[str]:
        q = asyncio.Queue()
        if track_id not in self._listeners:
            self._listeners[track_id] = []
        self._listeners[track_id].append(q)
        
        try:
            # Yield any historical events for this track first
            if track_id in self._history:
                for ev in self._history[track_id]:
                    yield f"data: {ev.model_dump_json()}\n\n"
            
            while True:
                event = await q.get()
                yield f"data: {event.model_dump_json()}\n\n"
                if event.payload.type == "run_complete" and event.track_id == track_id:
                    break
        finally:
            self._listeners[track_id].remove(q)
            if not self._listeners[track_id]:
                del self._listeners[track_id]

    def get_trace(self, track_id: str) -> list[TelemetryEvent]:
        return self._history.get(track_id, [])

class QueuedTelemetryHub(TelemetryHub):
    """Used by parallel workers to pipe events into a multiplexer queue."""
    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self.queue = queue
        
    async def emit(self, event: TelemetryEvent) -> None:
        await self.queue.put(event)
