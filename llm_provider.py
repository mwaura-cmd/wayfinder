"""
llm_provider.py — Central client factory for Wayfinder LLM calls.

Supports Groq ("groq") and OpenRouter ("openrouter") providers via standard OpenAI SDK.
Both providers remain fully configured and switchable via config.LLM_PROVIDER.
"""

import logging
from openai import OpenAI, AsyncOpenAI
import config

logger = logging.getLogger(__name__)


def get_llm_client_and_model(provider: str | None = None) -> tuple[OpenAI, str]:
    """
    Returns (client, model_name) for the requested provider, defaulting to
    config.LLM_PROVIDER. Both providers use the OpenAI-compatible client —
    only base_url, api_key, and model differ.
    """
    provider_name = (provider or getattr(config, "LLM_PROVIDER", "openrouter")).lower()

    if provider_name == "groq":
        key = getattr(config, "GROQ_API_KEY", "") or ""
        if not key.strip():
            raise ValueError("GROQ_API_KEY is not set, but LLM_PROVIDER is 'groq'. Add it to your .env file.")
        model = getattr(config, "GROQ_MODEL", "llama-3.3-70b-versatile")
        return (
            OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key.strip()),
            model,
        )
    elif provider_name == "openrouter":
        key = getattr(config, "OPENROUTER_API_KEY", "") or ""
        if not key.strip():
            raise ValueError("OPENROUTER_API_KEY is not set, but LLM_PROVIDER is 'openrouter'. Add it to your .env file.")
        model = getattr(config, "OPENROUTER_MODEL", "openrouter/free")
        return (
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key.strip(),
                default_headers={
                    "HTTP-Referer": "https://github.com/mwaura-cmd/wayfinder",
                    "X-Title": "Wayfinder Research Agent",
                },
            ),
            model,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER value '{provider_name}' — expected 'groq' or 'openrouter'.")


def get_async_llm_client_and_model(provider: str | None = None) -> tuple[AsyncOpenAI, str]:
    """
    Returns (async_client, model_name) for the requested provider, defaulting to
    config.LLM_PROVIDER.
    """
    provider_name = (provider or getattr(config, "LLM_PROVIDER", "openrouter")).lower()

    if provider_name == "groq":
        key = getattr(config, "GROQ_API_KEY", "") or ""
        if not key.strip():
            raise ValueError("GROQ_API_KEY is not set, but LLM_PROVIDER is 'groq'. Add it to your .env file.")
        model = getattr(config, "GROQ_MODEL", "llama-3.3-70b-versatile")
        return (
            AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=key.strip()),
            model,
        )
    elif provider_name == "openrouter":
        key = getattr(config, "OPENROUTER_API_KEY", "") or ""
        if not key.strip():
            raise ValueError("OPENROUTER_API_KEY is not set, but LLM_PROVIDER is 'openrouter'. Add it to your .env file.")
        model = getattr(config, "OPENROUTER_MODEL", "openrouter/free")
        return (
            AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key.strip(),
                default_headers={
                    "HTTP-Referer": "https://github.com/mwaura-cmd/wayfinder",
                    "X-Title": "Wayfinder Research Agent",
                },
            ),
            model,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER value '{provider_name}' — expected 'groq' or 'openrouter'.")
