import json
from typing import Any, AsyncIterator, List
from google import genai
from google.genai import types as gtypes

from core.provider import BaseLLMProvider, LLMRequest, LLMResponse, LLMChunk, Message, TokenUsage, ToolCall
import config

class GeminiProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

    @property
    def context_limit(self) -> int:
        return 1000000  # 1M tokens for Gemini 1.5/2.5

    @property
    def provider_id(self) -> str:
        return "gemini"

    def format_tools(self, tools: List[Any]) -> List[dict]:
        # Translates our ToolDefinitions into Gemini's format.
        # But wait, the spec says `tools` is a list of ToolDefinition, but the type hint in BaseLLMProvider was Any to avoid circular dependency.
        # We will assume they are ToolDefinition instances.
        gemini_tools = []
        for t in tools:
            # We must map JSON Schema to Gemini FunctionDeclaration
            # Simplified for this stub
            decl = {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema
            }
            gemini_tools.append(decl)
            
        if not gemini_tools:
            return []
            
        return [{"function_declarations": gemini_tools}]

    def _convert_messages(self, messages: List[Message], system_prompt: str) -> List[gtypes.Content]:
        contents = []
        for m in messages:
            role = "user" if m.role == "user" else "model"
            # If it's a tool response, the spec says role="tool"
            # Gemini wants it structured specific ways, but we can do string manipulation for now to match the "ReAct Template" approach
            # Actually, because we are using PromptAssembler with ReAct format (Thought/Action), we are sending raw strings as user/model interactions, not structured function calling.
            # The spec explicitly dictates Block 4 is "REACT FORMAT TEMPLATE". This implies the LLM outputs text that we parse, NOT native tool calls!
            # Let's check the spec: "The provider handles the schema format... Tool result format: functionCall part".
            # Ah, the spec implies native tool calling IS used, but wrapped in a ReAct loop.
            contents.append(gtypes.Content(role=role, parts=[gtypes.Part.from_text(text=m.content)]))
        return contents

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # The prompt assembler puts the system prompt + ReAct instructions in the 'system' field.
        gemini_config = gtypes.GenerateContentConfig(
            system_instruction=request.system,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            tools=request.tools if request.tools else None
        )

        contents = self._convert_messages(request.messages, request.system)
        
        # We use sync client in an async wrapper for now, or just use async client if available
        # google-genai supports async via client.aio
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=gemini_config
        )

        # Parse out tool calls natively if they exist
        tool_calls = []
        stop_reason = "end_turn"
        
        if response.candidates and response.candidates[0].function_calls:
            for fc in response.candidates[0].function_calls:
                tool_calls.append(ToolCall(
                    tool_name=fc.name,
                    inputs=fc.args,
                    call_id="call_" + fc.name # Gemini doesn't always provide call IDs natively like OpenAI
                ))
            stop_reason = "tool_use"
        elif response.candidates and response.candidates[0].finish_reason:
            fr = response.candidates[0].finish_reason
            if fr == 1: # STOP
                stop_reason = "end_turn"
            elif fr == 2: # MAX_TOKENS
                stop_reason = "max_tokens"

        usage = TokenUsage(
            prompt_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            completion_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        )

        return LLMResponse(
            content=response.text or "",
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
            usage=usage,
            raw={"raw": str(response)}
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        raise NotImplementedError("Streaming not yet fully wired for Gemini stub")

    def parse_action(self, response: LLMResponse) -> Any:
        # Layer 1 shouldn't import Layer 2 (AgentAction).
        # The OutputParser in Layer 2 will handle this.
        pass
