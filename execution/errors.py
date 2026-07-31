from enum import Enum
from typing import Optional
from pydantic import BaseModel
from execution.parser import AgentAction, ActionType
from execution.governor import IterationGovernor
from core.memory import WorkingMemory, TrimStrategy
from core.provider import BaseLLMProvider, LLMRequest
from core.tools import ToolError

class ResolutionStrategy(str, Enum):
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_WITH_MODIFIED_INPUT = "retry_with_modified_input"
    INJECT_AS_OBSERVATION = "inject_as_observation"
    ABORT = "abort"

class ErrorResolution(BaseModel):
    strategy: ResolutionStrategy
    modified_input: Optional[dict] = None
    observation_message: Optional[str] = None

class ProviderError(Exception):
    pass

class ErrorHandler:
    def handle_tool_error(self, error: ToolError, action: AgentAction, memory: WorkingMemory) -> ErrorResolution:
        if error.code == "timeout":
            return ErrorResolution(
                strategy=ResolutionStrategy.INJECT_AS_OBSERVATION,
                observation_message=f"Tool {action.tool_call.tool_name if action.tool_call else 'unknown'} timed out. Consider an alternative approach."
            )
        elif error.code == "validation_failed":
            return ErrorResolution(
                strategy=ResolutionStrategy.INJECT_AS_OBSERVATION,
                observation_message=f"Tool {action.tool_call.tool_name if action.tool_call else 'unknown'} rejected input: {error.message}. Adjust your parameters."
            )
        else:
            return ErrorResolution(
                strategy=ResolutionStrategy.INJECT_AS_OBSERVATION,
                observation_message=f"Tool execution failed: {error.message}."
            )

    def handle_provider_error(self, error: ProviderError, request: LLMRequest) -> ErrorResolution:
        # Default fallback
        return ErrorResolution(strategy=ResolutionStrategy.ABORT)

    def handle_context_overflow(self, memory: WorkingMemory, provider: BaseLLMProvider) -> ErrorResolution:
        return ErrorResolution(
            strategy=ResolutionStrategy.RETRY_WITH_MODIFIED_INPUT,
            # In a real app we'd tell the loop engine to trim memory
        )

    def handle_parse_failure(self, raw_response: str, governor: IterationGovernor) -> AgentAction:
        # Auto-generate fallback narrative and increment malformed
        governor.record_malformed()
        return AgentAction(
            type=ActionType.THOUGHT_ONLY,
            narrative="Parsing output...",
            thought="The previous output was malformed. Re-evaluating.",
            raw_response=raw_response
        )
