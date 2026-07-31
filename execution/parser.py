from enum import Enum
from typing import Optional
from pydantic import BaseModel
from core.provider import LLMResponse
from core.tools import ToolCall


class ActionType(str, Enum):
    TOOL_CALL = "tool_call"
    SKILL_CALL = "skill_call"
    FINAL_ANSWER = "final_answer"
    THOUGHT_ONLY = "thought_only"

# Since SkillCall isn't in core.skills yet, let's define it here or rely on dicts.
# According to spec:
class SkillCall(BaseModel):
    skill_name: str
    inputs: dict
    call_id: str

class AgentAction(BaseModel):
    type: ActionType
    narrative: str
    thought: str
    tool_call: Optional[ToolCall] = None
    skill_call: Optional[SkillCall] = None
    final_answer: Optional[str] = None
    raw_response: str

class OutputParser:
    @classmethod
    def parse(cls, response: LLMResponse) -> AgentAction:
        # Implementation of parsing LLM response into AgentAction
        # Step 1: Attempt structured extraction from response.tool_calls
        # Step 2: Fallback to text parsing (ReAct regex)
        # Step 3: Handle empty narrative (inject fallback)
        # Step 4: Handle malformed (THOUGHT_ONLY)
        
        # This is a stub that will be fleshed out with the actual regex/json parsing logic.
        
        # Simplified stub implementation:
        action_type = ActionType.THOUGHT_ONLY
        tool_call = None
        final_answer = None
        narrative = "Processing..."
        thought = "Reasoning..."

        if response.tool_calls and len(response.tool_calls) > 0:
            action_type = ActionType.TOOL_CALL
            tc = response.tool_calls[0]
            tool_call = tc
            narrative = f"Analyzing using {tc.tool_name}..." # Fallback narrative
        else:
            # Fallback regex parsing (stubbed)
            if "Final Answer:" in response.content:
                action_type = ActionType.FINAL_ANSWER
                final_answer = response.content.split("Final Answer:")[-1].strip()
                narrative = "Synthesizing final answer..."
            elif "Action:" in response.content:
                action_type = ActionType.TOOL_CALL
                # Fake parsing for stub
                import uuid
                tool_call = ToolCall(tool_name="unknown", inputs={}, call_id=str(uuid.uuid4()))
                narrative = "Executing tool..."

        return AgentAction(
            type=action_type,
            narrative=narrative,
            thought=thought,
            tool_call=tool_call,
            final_answer=final_answer,
            raw_response=response.content
        )
