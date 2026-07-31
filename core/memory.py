from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel
from core.provider import Message, BaseLLMProvider
from core.tools import ToolResult
from core.skills import SkillResult

class TrimStrategy(Enum):
    DROP_OLDEST_OBSERVATIONS = "drop_oldest_observations"
    SUMMARIZE_MIDDLE = "summarize_middle"
    TRUNCATE_TOOL_RESULTS = "truncate_tool_results"

class MemorySnapshot(BaseModel):
    messages: List[Message]

class WorkingMemory:
    def __init__(self):
        self._messages: List[Message] = []

    def add_message(self, role: str, content: str, tool_call_id: Optional[str] = None) -> None:
        self._messages.append(Message(role=role, content=content, tool_call_id=tool_call_id))

    def add_tool_result(self, result: ToolResult) -> None:
        self.add_message("tool", result.output, tool_call_id=result.call_id)

    def add_skill_result(self, result: SkillResult) -> None:
        # According to the spec, skill results are normalized into strings and added to context
        self.add_message("observation", f"Skill {result.skill_name} completed: {result.output}")

    def add_observation(self, observation: str) -> None:
        self.add_message("observation", observation)

    def get_context(self) -> List[Message]:
        return list(self._messages)

    def get_token_count(self, provider: BaseLLMProvider) -> int:
        # A simple approximation for now. This will be provider-specific in reality.
        # Ideally we'd call provider.count_tokens() if it existed, but spec doesn't show it.
        # So we estimate 4 chars per token.
        text = "".join(m.content for m in self._messages)
        return len(text) // 4

    def trim_to_fit(self, limit: int, strategy: TrimStrategy) -> None:
        # Implementation of trimming goes here
        pass

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(messages=self.get_context())

class ReActStep(BaseModel):
    step: int
    thought: Optional[str] = None
    action: Optional[Any] = None # AgentAction
    observation: Optional[str] = None

class EpisodicMemory:
    def __init__(self):
        self._steps: dict[int, ReActStep] = {}

    def _get_or_create(self, step: int) -> ReActStep:
        if step not in self._steps:
            self._steps[step] = ReActStep(step=step)
        return self._steps[step]

    def log_thought(self, thought: str, step: int) -> None:
        s = self._get_or_create(step)
        s.thought = thought

    def log_action(self, action: Any, step: int) -> None:
        s = self._get_or_create(step)
        s.action = action

    def log_observation(self, observation: str, step: int) -> None:
        s = self._get_or_create(step)
        s.observation = observation

    def get_trace(self) -> List[ReActStep]:
        return [self._steps[k] for k in sorted(self._steps.keys())]

    def get_summary(self) -> str:
        # Compressed narrative for Orchestrator Manager
        lines = []
        for step in self.get_trace():
            if step.action and getattr(step.action, "narrative", None):
                lines.append(f"Step {step.step}: {step.action.narrative}")
            if step.observation:
                lines.append(f"Result: {step.observation[:100]}...")
        return "\n".join(lines)

class MemoryChunk(BaseModel):
    key: str
    content: str
    metadata: dict

class SemanticMemoryStore:
    async def store(self, key: str, content: str, metadata: dict) -> None:
        raise NotImplementedError

    async def retrieve(self, query: str, top_k: int) -> List[MemoryChunk]:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError
