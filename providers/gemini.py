import uuid
import datetime
from typing import Any, AsyncIterator, List
from google import genai
from google.genai import types as gtypes

from core.provider import BaseLLMProvider, LLMRequest, LLMResponse, LLMChunk, Message, TokenUsage, ToolCall
import config


class GeminiProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gemini-3.6-flash-lite"):
        self.model_name = model_name
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

    @property
    def context_limit(self) -> int:
        return 1_000_000

    @property
    def provider_id(self) -> str:
        return "gemini"

    def format_tools(self, tools: List[Any]) -> List[dict]:
        """
        Converts ToolDefinition instances into Gemini-native tool format.
        Returns an empty list (not None) when no tools are registered.
        """
        if not tools:
            return []

        declarations = []
        for t in tools:
            declarations.append(
                gtypes.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.input_schema,
                )
            )
        return [gtypes.Tool(function_declarations=declarations)]

    def _convert_messages(self, messages: List[Message]) -> List[gtypes.Content]:
        """
        Convert our normalized Messages into Gemini Content objects.
        - 'system' messages are handled via system_instruction in config, skip here.
        - 'user' → role 'user'
        - 'assistant', 'model', 'tool', 'observation' → role 'model'
        Gemini only accepts alternating user/model turns.
        """
        contents = []
        for m in messages:
            if m.role == "system":
                continue  # Passed as system_instruction in GenerateContentConfig
            role = "user" if m.role == "user" else "model"
            contents.append(
                gtypes.Content(role=role, parts=[gtypes.Part.from_text(text=m.content)])
            )
        # Gemini requires the turn sequence to start with 'user' and alternate.
        # Ensure we don't have consecutive same-role messages by merging them.
        merged: List[gtypes.Content] = []
        for c in contents:
            if merged and merged[-1].role == c.role:
                # Merge into the previous turn's parts
                merged[-1] = gtypes.Content(
                    role=c.role,
                    parts=list(merged[-1].parts) + list(c.parts)
                )
            else:
                merged.append(c)
        return merged

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Pull system prompt from request.system (assembled by PromptAssembler)
        # or fall back to extracting the first system message from working memory
        system_instruction = request.system or ""

        gemini_config = gtypes.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            tools=request.tools if request.tools else None,
        )

        contents = self._convert_messages(request.messages)

        # Guard: Gemini requires at least one content turn
        if not contents:
            contents = [gtypes.Content(role="user", parts=[gtypes.Part.from_text(text="Begin.")])]

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=gemini_config,
        )

        # Extract native function calls if present
        tool_calls: List[ToolCall] = []
        stop_reason = "end_turn"

        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        tool_name=fc.name,
                        inputs=dict(fc.args) if fc.args else {},
                        call_id=f"call_{fc.name}_{uuid.uuid4().hex[:8]}",
                    ))
            if tool_calls:
                stop_reason = "tool_use"

        if candidate and candidate.finish_reason:
            fr = candidate.finish_reason
            if hasattr(fr, "value"):
                fr = fr.value
            if fr == 2:
                stop_reason = "max_tokens"

        usage = TokenUsage(
            prompt_tokens=(response.usage_metadata.prompt_token_count
                           if response.usage_metadata else 0),
            completion_tokens=(response.usage_metadata.candidates_token_count
                               if response.usage_metadata else 0),
        )

        return LLMResponse(
            content=response.text or "",
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
            usage=usage,
            raw={"raw": str(response)},
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        raise NotImplementedError("Streaming is not yet wired.")

    def parse_action(self, response: LLMResponse) -> Any:
        # Handled by OutputParser in the Execution layer (Layer 2).
        pass
