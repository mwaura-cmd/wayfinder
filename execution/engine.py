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
            # Extract the system prompt from working_memory (it's the first message with role 'system')
            context = working_memory.get_context()
            system_msg = next((m.content for m in context if m.role == "system"), "")

            req = LLMRequest(
                messages=context,
                tools=provider.format_tools(tools),
                system=system_msg,
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
                    # Add the model's tool call turn first so context is valid
                    working_memory.add_message("assistant", response.content or f"Action: {action.tool_call.tool_name}")
                    # Then add the tool result as a user observation turn
                    working_memory.add_message("user", f"Observation: {result.output}")
                    episodic_memory.log_observation(result.output, governor.steps)
                else:
                    # Tool not found — inject as error observation
                    working_memory.add_message("assistant", response.content or "")
                    working_memory.add_message("user", f"Observation: Tool '{action.tool_call.tool_name}' not found in registry.")
                    episodic_memory.log_observation("Tool not found.", governor.steps)
            elif action.type == ActionType.FINAL_ANSWER:
                episodic_memory.log_observation("Final answer provided.", governor.steps)
            else:
                # THOUGHT_ONLY — add model output back so loop can continue
                if response.content:
                    working_memory.add_message("assistant", response.content)

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
