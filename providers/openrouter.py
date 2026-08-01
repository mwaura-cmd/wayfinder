"""
providers/openrouter.py — OpenRouter LLM provider for Wayfinder.

Uses the official `openai` SDK pointing to OpenRouter's API endpoint.
This is the OpenAI-compatible approach to routing between models.
"""

import uuid
import json
from typing import Any, AsyncIterator, List

from openai import AsyncOpenAI
import openai

from core.provider import BaseLLMProvider, LLMRequest, LLMResponse, LLMChunk, Message, TokenUsage, ToolCall
import config


class OpenRouterProvider(BaseLLMProvider):
    """
    LLM provider backed by OpenRouter via the OpenAI SDK.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.OPENROUTER_MODEL
        # Point the OpenAI client at OpenRouter
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "https://github.com/mwaura-cmd/wayfinder",
                "X-Title": "Wayfinder Research Agent",
            }
        )

    # ── Abstract property implementations ─────────────────────────────────────

    @property
    def context_limit(self) -> int:
        return 128_000

    @property
    def provider_id(self) -> str:
        return "openrouter"

    # ── Tool formatting ────────────────────────────────────────────────────────

    def format_tools(self, tools: List[Any]) -> List[dict]:
        """
        Convert ToolDefinition list → OpenAI function-calling format.
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
        """
        result = []
        for m in messages:
            role = m.role
            # Map roles to strictly allowed OpenAI roles
            if role == "model":
                role = "assistant"
            elif role == "observation":
                # Tools need specific formatting in OpenAI (role="tool", tool_call_id=...)
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

    # ── Non-streaming completion ───────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        messages = self._convert_messages(request.messages)
        
        # Inject the system prompt if present
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = False

        try:
            response = await self.client.chat.completions.create(**kwargs)
        except openai.APIError as e:
            # Map specific OpenRouter errors
            raise e

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        finish = choice.finish_reason

        stop_reason_map = {
            "stop":          "end_turn",
            "tool_calls":    "tool_use",
            "length":        "max_tokens",
            "content_filter":"end_turn",
        }
        stop_reason = stop_reason_map.get(finish, "end_turn")

        # Parse tool calls
        tool_calls: List[ToolCall] = []
        if message.tool_calls:
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
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
            usage=usage,
            raw=response.model_dump(),
        )

    # ── Streaming (not yet wired) ──────────────────────────────────────────────

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        raise NotImplementedError("Streaming not yet wired.")

    # ── Action parsing ─────────────────────────────────────────────────────────

    def parse_action(self, response: LLMResponse) -> Any:
        pass
