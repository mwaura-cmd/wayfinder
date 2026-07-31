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
    search_count: int = 0
    elapsed_seconds: float = 0.0
    sources: List[str] = []
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

        run_start = datetime.datetime.now(datetime.timezone.utc)
        search_count = 0
        sources: List[str] = []

        while True:
            # 1. Tick governor
            governor.tick()

            # 2. Build LLMRequest
            context = working_memory.get_context()
            system_msg = next((m.content for m in context if m.role == "system"), "")

            req = LLMRequest(
                messages=context,
                tools=[], # Force text-based ReAct mode by not passing native tools
                system=system_msg,
                max_tokens=4096,
                temperature=0.0,
                stop_sequences=[]
            )

            # 3. Call Provider
            response = await provider.complete(req)

            # 4. Parse Action
            action = OutputParser.parse(response)

            # Record to episodic memory
            episodic_memory.log_thought(action.thought, governor.steps)
            episodic_memory.log_action(action, governor.steps)

            # 5. Emit narrative telemetry (always — even for THOUGHT_ONLY)
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
                    search_count += 1
                    # Collect source URLs from raw Tavily response
                    if isinstance(result.raw, dict):
                        for r in result.raw.get("results", []):
                            url = r.get("url")
                            if url and url not in sources:
                                sources.append(url)

                    # Add the model's tool call turn first so context is valid
                    working_memory.add_message("assistant", response.content or f"Calling {action.tool_call.tool_name}.")
                    # Then add the tool result as a user observation turn
                    working_memory.add_message("user", f"Observation: {result.output}")
                    episodic_memory.log_observation(result.output, governor.steps)

                    # Ask the model to produce a follow-up narrative after each observation
                    working_memory.add_message(
                        "user",
                        "Based on the above search results, continue your research or provide your Final Answer."
                    )
                else:
                    # Tool not found — inject as error observation
                    working_memory.add_message("assistant", response.content or "")
                    working_memory.add_message("user", f"Observation: Tool '{action.tool_call.tool_name}' not found in registry.")
                    episodic_memory.log_observation("Tool not found.", governor.steps)

            elif action.type == ActionType.FINAL_ANSWER:
                episodic_memory.log_observation("Final answer provided.", governor.steps)

            else:
                # THOUGHT_ONLY — feed back to continue the loop
                if response.content:
                    working_memory.add_message("assistant", response.content)
                    working_memory.add_message("user", "Please continue and provide your Final Answer when ready.")

            # 7. Check Stop Conditions
            for cond in stop_conditions:
                if cond.check(action, governor, working_memory):
                    elapsed = round(
                        (datetime.datetime.now(datetime.timezone.utc) - run_start).total_seconds(), 1
                    )
                    summary = cls._build_summary(
                        stop_reason=cond.name,
                        search_count=search_count,
                        source_count=len(sources),
                        elapsed=elapsed,
                        is_final_answer=(action.type == ActionType.FINAL_ANSWER)
                    )

                    await telemetry.emit(TelemetryEvent(
                        event_id=str(uuid.uuid4()),
                        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        track_id=track_id,
                        parent_track_id=parent_track_id,
                        track_type="root" if parent_track_id is None else "subagent",
                        source_type="system",
                        payload=Payload(
                            type="run_complete",
                            summary=summary,
                            final_output=action.final_answer if action.type == ActionType.FINAL_ANSWER else None,
                            # Rich meta packed into label so frontend can parse it
                            label=f"search_count={search_count}|elapsed={elapsed}|source_count={len(sources)}|sources={','.join(sources)}"
                        )
                    ))

                    return AgentRunResult(
                        track_id=track_id,
                        final_output=action.final_answer or "",
                        steps_taken=governor.steps,
                        stop_reason=cond.name,
                        episodic_summary=episodic_memory.get_summary(),
                        success=(action.type == ActionType.FINAL_ANSWER),
                        search_count=search_count,
                        elapsed_seconds=elapsed,
                        sources=sources,
                        error=None
                    )

    @staticmethod
    def _build_summary(
        stop_reason: str,
        search_count: int,
        source_count: int,
        elapsed: float,
        is_final_answer: bool
    ) -> str:
        if not is_final_answer:
            return f"Run stopped: {stop_reason} after {elapsed}s"
        if search_count == 0:
            return f"Answered in {elapsed}s using memory and prior knowledge — 0 searches"
        return (
            f"Reviewed {source_count} source{'s' if source_count != 1 else ''} "
            f"and answered in {elapsed}s — {search_count} search{'es' if search_count != 1 else ''}"
        )
