"""
agent.py — Agent runner interface wrapping provider selection and agent execution loop.
"""

import logging
from typing import AsyncIterator, Dict, Any, List, Optional

import config
from llm_provider import get_llm_client_and_model, get_async_llm_client_and_model
from core.provider import ProviderRegistry

logger = logging.getLogger(__name__)


async def run_agent(
    prompt: str,
    provider_name: Optional[str] = None,
    level: str = "standard",
    prior_messages: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Runs the agent loop for the requested provider (defaulting to config.LLM_PROVIDER).
    Validates provider credentials at the top before starting the loop and yields a clean
    error event on missing configuration or invalid provider names.
    """
    target_provider = provider_name or config.LLM_PROVIDER

    # 1. Validate provider selection and API key configuration upfront
    try:
        client, model = get_llm_client_and_model(target_provider)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Provider configuration error for '{target_provider}': {error_msg}")
        yield {"type": "error", "text": error_msg}
        return

    # 2. Get provider instance from registry
    try:
        provider_inst = ProviderRegistry.get(target_provider)
    except ValueError:
        yield {
            "type": "error",
            "text": f"Unknown LLM_PROVIDER value '{target_provider}' — expected 'groq' or 'openrouter'.",
        }
        return

    # Yield initialization status event
    yield {
        "type": "status",
        "text": f"Initialized agent using {target_provider} ({model})",
        "model": model,
        "provider": target_provider,
    }
