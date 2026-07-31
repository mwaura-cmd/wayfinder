from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional
from pydantic import BaseModel
import os

class ToolCall(BaseModel):
    tool_name: str
    inputs: dict
    call_id: str

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

class Message(BaseModel):
    role: str
    content: str
    tool_call_id: Optional[str] = None

class LLMRequest(BaseModel):
    messages: List[Message]
    tools: List[dict]
    system: str
    max_tokens: int
    temperature: float
    stop_sequences: List[str]

class LLMResponse(BaseModel):
    content: str
    tool_calls: Optional[List[ToolCall]]
    stop_reason: str
    usage: TokenUsage
    raw: dict

class LLMChunk(BaseModel):
    content: str

# To avoid circular imports, AgentAction is returned loosely or we type-hint it as Any here.
# Actually, the spec says parse_action returns AgentAction, but AgentAction is in Layer 2 (Execution).
# Layer 1 cannot import Layer 2. So we must type it as Any or define a protocol.
class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        pass

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        pass

    @abstractmethod
    def format_tools(self, tools: List[Any]) -> List[dict]:
        pass

    @abstractmethod
    def parse_action(self, response: LLMResponse) -> Any:
        pass

    @property
    @abstractmethod
    def context_limit(self) -> int:
        pass

    @property
    @abstractmethod
    def provider_id(self) -> str:
        pass

class ProviderRegistry:
    _providers: dict[str, BaseLLMProvider] = {}

    @classmethod
    def register(cls, provider_id: str, provider: BaseLLMProvider) -> None:
        cls._providers[provider_id] = provider

    @classmethod
    def get(cls, provider_id: str) -> BaseLLMProvider:
        if provider_id not in cls._providers:
            raise ValueError(f"Provider {provider_id} not found")
        return cls._providers[provider_id]

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._providers.keys())
