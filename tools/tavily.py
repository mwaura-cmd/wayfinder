import os
import httpx
from typing import Any
from core.tools import BaseToolExecutor, ToolResult, ToolError
import config

class TavilySearchExecutor(BaseToolExecutor):
    async def execute(self, inputs: dict, telemetry: Any, track_id: str) -> ToolResult:
        query = inputs.get("query")
        if not query:
            return ToolResult(
                call_id="", tool_name="web_search", output="", raw={}, success=False,
                error=ToolError(code="validation_failed", message="Missing 'query'", recoverable=True)
            )

        if not config.TAVILY_API_KEY:
            return ToolResult(
                call_id="", tool_name="web_search", output="", raw={}, success=False,
                error=ToolError(code="execution_failed", message="TAVILY_API_KEY is not set.", recoverable=False)
            )

        # Emit work_delta per spec
        import datetime
        import uuid
        from observability.telemetry import TelemetryEvent, Payload
        await telemetry.emit(TelemetryEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            track_id=track_id,
            parent_track_id=None,
            track_type="root",
            source_type="tool",
            payload=Payload(
                type="work_delta",
                detail=f"Searching for: '{query}'"
            )
        ))

        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": config.TAVILY_API_KEY,
                        "query": query,
                        "search_depth": "basic",
                        "include_answer": False,
                        "max_results": 5,
                    },
                    timeout=15.0
                )
                res.raise_for_status()
                data = res.json()

            results = data.get("results", [])
            if not results:
                formatted_output = "No results found."
            else:
                snippets = []
                for idx, r in enumerate(results, 1):
                    snippets.append(f"[{idx}] {r.get('title', 'Untitled')}\nURL: {r.get('url', '')}\n{r.get('content', '')}")
                formatted_output = "\n\n".join(snippets)

            await telemetry.emit(TelemetryEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                track_id=track_id,
                parent_track_id=None,
                track_type="root",
                source_type="tool",
                payload=Payload(
                    type="work_delta",
                    detail=f"Found {len(results)} results."
                )
            ))

            return ToolResult(
                call_id=inputs.get("call_id", ""),
                tool_name="web_search",
                output=formatted_output,
                raw=data,
                success=True,
                error=None
            )
        except Exception as e:
            return ToolResult(
                call_id="", tool_name="web_search", output="", raw={}, success=False,
                error=ToolError(code="execution_failed", message=str(e), recoverable=True)
            )
