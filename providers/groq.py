"""
providers/groq.py — Groq LLM provider for Wayfinder.

Uses standard OpenAI SDK pointing to Groq's API endpoint.
Supports automatic fallback to OpenRouter if Groq is temporarily unavailable.
"""

import uuid
import json
import logging
import asyncio
from typing import Any, AsyncIterator, List

import openai

from core.provider import BaseLLMProvider, LLMRequest, LLMResponse, LLMChunk, Message, TokenUsage, ToolCall
import config
from llm_provider import get_async_llm_client_and_model

logger = logging.getLogger(__name__)


class GroqProvider(BaseLLMProvider):
    """
    LLM provider backed by Groq via the OpenAI SDK.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.GROQ_MODEL

    @property
    def context_limit(self) -> int:
        return 128_000

    @property
    def provider_id(self) -> str:
        return "groq"

    def format_tools(self, tools: List[Any]) -> List[dict]:
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

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        result = []
        for m in messages:
            role = m.role
            if role == "model":
                role = "assistant"
            elif role == "observation":
                if m.tool_call_id:
                    result.append({
                        "role": "tool",
                        "content": m.content,
                        "tool_call_id": m.tool_call_id
                    })
                    continue
                else:
                    role = "user"
                    m.content = f"Tool result: {m.content}"
            elif role == "tool":
                role = "user"
                m.content = f"Tool usage logged: {m.content}"

            result.append({"role": role, "content": m.content})
        return result

    async def complete(self, request: LLMRequest) -> LLMResponse:
        client, model_name = get_async_llm_client_and_model("groq")

        messages = self._convert_messages(request.messages)
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})

        kwargs = {
            "model": model_name,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = False

        max_retries = getattr(config, "MAX_RETRIES", 3)
        base_delay = getattr(config, "RETRY_BASE_SECONDS", 1.0)

        response = None
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Groq request attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue

        # Optional Automatic Fallback to OpenRouter if Groq retries exhausted
        if response is None and config.OPENROUTER_API_KEY and config.OPENROUTER_API_KEY.strip():
            logger.warning("Groq unavailable, falling back to OpenRouter for this request")
            try:
                fb_client, fb_model = get_async_llm_client_and_model("openrouter")
                kwargs["model"] = fb_model
                response = await fb_client.chat.completions.create(**kwargs)
            except Exception as fb_err:
                logger.error(f"Fallback to OpenRouter also failed: {fb_err}")
                raise last_error or fb_err

        if response is None:
            raise RuntimeError(f"Groq API call failed: {last_error}") from last_error

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        finish = choice.finish_reason

        stop_reason_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "end_turn",
        }
        stop_reason = stop_reason_map.get(finish, "end_turn")

        tool_calls: List[ToolCall] = []
        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                fn = tc.function
                try:
                    args = json.loads(fn.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    tool_name=fn.name,
                    inputs=args,
                    call_id=tc.id or f"call_{fn.name}_{uuid.uuid4().hex[:8]}",
                ))

        if tool_calls:
            stop_reason = "tool_use"

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if getattr(response, "usage", None) else 0,
            completion_tokens=response.usage.completion_tokens if getattr(response, "usage", None) else 0,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
            usage=usage,
            raw=response.model_dump(),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        raise NotImplementedError("Streaming not implemented.")

    def parse_action(self, response: LLMResponse) -> Any:
        pass
