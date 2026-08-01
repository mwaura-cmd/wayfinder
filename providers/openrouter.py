"""
providers/openrouter.py — OpenRouter LLM provider for Wayfinder.

OpenRouter exposes an OpenAI-compatible Chat Completions endpoint, so we can
talk to dozens of models (Claude, Llama, Mistral, Gemini, etc.) through one
API key without touching any model-specific SDK.

Endpoint: https://openrouter.ai/api/v1/chat/completions
Auth:     Authorization: Bearer <OPENROUTER_API_KEY>
Docs:     https://openrouter.ai/docs
"""

import uuid
import json
import httpx
from typing import Any, AsyncIterator, List

from core.provider import BaseLLMProvider, LLMRequest, LLMResponse, LLMChunk, Message, TokenUsage, ToolCall
import config

_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
_TIMEOUT  = 120.0   # seconds — research queries can be slow


class OpenRouterProvider(BaseLLMProvider):
    """
    LLM provider backed by OpenRouter's OpenAI-compatible API.

    Model names follow OpenRouter's convention, e.g.:
      "anthropic/claude-sonnet-4-5"
      "meta-llama/llama-3.1-8b-instruct:free"
      "google/gemini-2.0-flash-exp:free"
      "mistralai/mistral-7b-instruct:free"

    Free-tier models are marked with `:free` suffix and have no cost,
    but may have lower rate limits.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.OPENROUTER_MODEL
        self._api_key   = config.OPENROUTER_API_KEY

    # ── Abstract property implementations ─────────────────────────────────────

    @property
    def context_limit(self) -> int:
        # Conservative default; actual limit depends on the chosen model.
        return 128_000

    @property
    def provider_id(self) -> str:
        return "openrouter"

    # ── Tool formatting ────────────────────────────────────────────────────────

    def format_tools(self, tools: List[Any]) -> List[dict]:
        """
        Convert ToolDefinition list → OpenAI function-calling format.
        OpenRouter supports the same tool spec as OpenAI.
        """
        if not tools:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    # ── Message conversion ─────────────────────────────────────────────────────

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        """
        Convert normalised Message objects → OpenAI chat message dicts.
        Roles: 'system', 'user', 'assistant' are passed through directly.
        'tool', 'observation' → 'user' (wrapped with context prefix).
        """
        result = []
        for m in messages:
            role = m.role
            if role in ("tool", "observation"):
                role = "user"
            elif role in ("assistant", "model"):
                role = "assistant"
            result.append({"role": role, "content": m.content})
        return result

    # ── Non-streaming completion ───────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        messages = self._convert_messages(request.messages)

        body: dict = {
            "model":       self.model_name,
            "messages":    messages,
            "max_tokens":  request.max_tokens,
            "temperature": request.temperature,
        }

        # Attach tools only when the provider/model supports them
        if request.tools:
            body["tools"]       = request.tools  # already formatted by format_tools()
            body["tool_choice"] = "auto"
            # OpenRouter recommends disabling parallel tool calls for reliability
            body["parallel_tool_calls"] = False

        headers = {
            "Authorization":  f"Bearer {self._api_key}",
            "Content-Type":   "application/json",
            # Optional but recommended — lets OpenRouter show your app in analytics
            "HTTP-Referer":   "https://github.com/mwaura-cmd/wayfinder",
            "X-Title":        "Wayfinder Research Agent",
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(_BASE_URL, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        choice   = data["choices"][0]
        message  = choice["message"]
        content  = message.get("content") or ""
        finish   = choice.get("finish_reason", "stop")

        # Map finish_reason → our internal stop_reason vocabulary
        stop_reason_map = {
            "stop":          "end_turn",
            "tool_calls":    "tool_use",
            "length":        "max_tokens",
            "content_filter":"end_turn",
        }
        stop_reason = stop_reason_map.get(finish, "end_turn")

        # Parse tool calls if present
        tool_calls: List[ToolCall] = []
        raw_tool_calls = message.get("tool_calls") or []
        for tc in raw_tool_calls:
            fn   = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                tool_name=name,
                inputs=args,
                call_id=tc.get("id") or f"call_{name}_{uuid.uuid4().hex[:8]}",
            ))

        if tool_calls:
            stop_reason = "tool_use"

        # Token usage
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
            usage=usage,
            raw=data,
        )

    # ── Streaming (not yet wired) ──────────────────────────────────────────────

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        raise NotImplementedError("OpenRouter streaming not yet wired.")

    # ── Action parsing ─────────────────────────────────────────────────────────

    def parse_action(self, response: LLMResponse) -> Any:
        # Handled by OutputParser in the Execution layer (Layer 2).
        pass
