"""
agent.py — Thin convenience wrapper around run_sequential_agent.

NOTE: The real agent execution path is:
  main.py::start_research() → orchestration/topologies.py::run_sequential_agent()
  → execution/engine.py::LoopEngine.run()

This module provides a standalone run_agent() helper that can be used
for testing or scripting without the full FastAPI stack.
"""

import logging
from typing import Dict, Any, List, Optional

import config
from llm_provider import get_llm_client_and_model
from core.provider import ProviderRegistry
from orchestration.topologies import run_sequential_agent
from observability.telemetry import TelemetryHub
import uuid

logger = logging.getLogger(__name__)


async def run_agent(
    prompt: str,
    provider_name: Optional[str] = None,
    level: str = "standard",
    prior_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Standalone agent runner — validates provider, runs the sequential agent loop,
    and returns the final AgentRunResult as a dict.

    Use this for CLI scripts or tests. In production, main.py calls
    run_sequential_agent() directly for full SSE streaming support.
    """
    target_provider = provider_name or config.LLM_PROVIDER

    # Validate provider credentials upfront
    try:
        get_llm_client_and_model(target_provider)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Provider configuration error for '{target_provider}': {error_msg}")
        return {"type": "error", "text": error_msg}

    # Validate provider is registered
    try:
        ProviderRegistry.get(target_provider)
    except ValueError:
        return {
            "type": "error",
            "text": f"Unknown provider '{target_provider}' — expected one of: openrouter, groq, gemini.",
        }

    telemetry = TelemetryHub()
    track_id = str(uuid.uuid4())

    logger.info(f"Running agent with provider={target_provider}, level={level}, track_id={track_id}")

    result = await run_sequential_agent(
        prompt=prompt,
        provider_id=target_provider,
        tool_categories=["search"],
        skill_domain="",
        role_prompt=(
            "You are Wayfinder, an elite web research engine. "
            "Research the user's question thoroughly using web search, "
            "then synthesize a comprehensive, expert-level final answer."
        ),
        telemetry=telemetry,
        track_id=track_id,
        level=level,
        prior_messages=prior_messages or [],
    )

    return {
        "type": "complete",
        "final_output": result.final_output,
        "sources": result.sources,
        "model_used": result.model_used,
        "steps_taken": result.steps_taken,
        "search_count": result.search_count,
        "elapsed_seconds": result.elapsed_seconds,
        "success": result.success,
    }
