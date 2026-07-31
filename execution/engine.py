import uuid
import datetime
from typing import List, Optional
from pydantic import BaseModel

from core.provider import BaseLLMProvider, LLMRequest
from core.tools import ToolDefinition, ToolResult
from core.skills import SkillDefinition
from core.memory import WorkingMemory, EpisodicMemory
from observability.telemetry import TelemetryHub, TelemetryEvent, Payload

from execution.parser import OutputParser, ActionType
from execution.governor import IterationGovernor
from execution.conditions import StopCondition
from execution.errors import ErrorHandler

class AgentRunResult(BaseModel):
    track_id: str
    final_output: str
    steps_taken: int
    stop_reason: str
    episodic_summary: str
    success: bool
    error: Optional[str] = None

class LoopEngine:
    @classmethod
    async def run(
        cls,
        prompt: str,
        provider: BaseLLMProvider,
        tools: List[ToolDefinition],
        skills: List[SkillDefinition],
        working_memory: WorkingMemory,
        episodic_memory: EpisodicMemory,
        telemetry: TelemetryHub,
        track_id: str,
        parent_track_id: Optional[str],
        stop_conditions: List[StopCondition],
        governor: IterationGovernor,
        error_handler: ErrorHandler
    ) -> AgentRunResult:
        
        while True:
            # 1. Tick governor
            governor.tick()

            # 2. Build LLMRequest
            # In a full implementation, we pass the current context from working_memory
            # Here we just create a stub request
            req = LLMRequest(
                messages=working_memory.get_context(),
                tools=provider.format_tools(tools),
                system="", # Already in working_memory as a system message
                max_tokens=2048,
                temperature=0.0,
                stop_sequences=[]
            )

            # 3. Call Provider
            response = await provider.complete(req)

            # 4. Parse Action
            action = OutputParser.parse(response)

            # Record thought to episodic memory
            episodic_memory.log_thought(action.thought, governor.steps)
            episodic_memory.log_action(action, governor.steps)

            # 5. Emit Telemetry
            if action.narrative:
                await telemetry.emit(TelemetryEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    track_id=track_id,
                    parent_track_id=parent_track_id,
                    track_type="root" if parent_track_id is None else "subagent",
                    source_type="agent",
                    payload=Payload(
                        type="narrative_start",
                        node_id=str(uuid.uuid4()),
                        narrative=action.narrative
                    )
                ))

            # 6. Execute Tool/Skill if any
            if action.type == ActionType.TOOL_CALL and action.tool_call:
                tool_def = next((t for t in tools if t.name == action.tool_call.tool_name), None)
                if tool_def:
                    result = await tool_def.executor.execute(
                        inputs=action.tool_call.inputs,
                        telemetry=telemetry,
                        track_id=track_id
                    )
                    working_memory.add_tool_result(result)
                    episodic_memory.log_observation(result.output, governor.steps)
            elif action.type == ActionType.FINAL_ANSWER:
                working_memory.add_observation("Final answer provided.")
                episodic_memory.log_observation("Final answer provided.", governor.steps)

            # 7. Check Stop Conditions
            for cond in stop_conditions:
                if cond.check(action, governor, working_memory):
                    payload_dict = cond.get_payload()
                    await telemetry.emit(TelemetryEvent(
                        event_id=str(uuid.uuid4()),
                        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        track_id=track_id,
                        parent_track_id=parent_track_id,
                        track_type="root" if parent_track_id is None else "subagent",
                        source_type="system",
                        payload=Payload(
                            type="run_complete",
                            summary=payload_dict.get("summary", "Complete"),
                            final_output=action.final_answer if action.type == ActionType.FINAL_ANSWER else None
                        )
                    ))
                    
                    return AgentRunResult(
                        track_id=track_id,
                        final_output=action.final_answer or "",
                        steps_taken=governor.steps,
                        stop_reason=cond.name,
                        episodic_summary=episodic_memory.get_summary(),
                        success=(action.type == ActionType.FINAL_ANSWER),
                        error=None
                    )
