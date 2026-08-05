import uuid
import datetime
import logging
from typing import List, Optional
from pydantic import BaseModel

from core.provider import BaseLLMProvider, LLMRequest, Message
from core.tools import ToolDefinition, ToolResult
from core.skills import SkillDefinition
from core.memory import WorkingMemory, EpisodicMemory
from observability.telemetry import TelemetryHub, TelemetryEvent, Payload

from execution.parser import OutputParser, ActionType, AgentAction, clean_final_answer
from execution.governor import IterationGovernor
from execution.conditions import StopCondition
from execution.errors import ErrorHandler

log = logging.getLogger(__name__)

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
    model_used: Optional[str] = None
    level: str = "standard"
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
        error_handler: ErrorHandler,
        level: str = "standard"
    ) -> AgentRunResult:

        run_start = datetime.datetime.now(datetime.timezone.utc)
        search_count = 0
        sources: List[str] = []
        stall_turn_count = 0
        last_model_used = getattr(provider, "model_name", None)

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
            if response.raw and isinstance(response.raw, dict):
                actual_model = response.raw.get("model")
                if actual_model:
                    last_model_used = actual_model

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

            # 6. Execute Tool/Skill if any & manage stall tracking
            if action.type == ActionType.TOOL_CALL and action.tool_call:
                tool_def = next((t for t in tools if t.name == action.tool_call.tool_name), None)
                if tool_def:
                    # Legitimate tool execution resets the stall counter
                    stall_turn_count = 0

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
                    stall_turn_count += 1
                    # Tool not found — inject as error observation
                    working_memory.add_message("assistant", response.content or "")
                    working_memory.add_message("user", f"Observation: Tool '{action.tool_call.tool_name}' not found in registry.")
                    episodic_memory.log_observation("Tool not found.", governor.steps)

            elif action.type == ActionType.FINAL_ANSWER:
                stall_turn_count = 0
                episodic_memory.log_observation("Final answer provided.", governor.steps)

            else:
                # Announcement-only / THOUGHT_ONLY turn without tool call or final answer
                stall_turn_count += 1

                if stall_turn_count == 1:
                    # 1st stall turn — standard nudge
                    if response.content:
                        working_memory.add_message("assistant", response.content)
                    working_memory.add_message(
                        "user",
                        "Please continue with your search action or provide your Final Answer directly."
                    )

                elif stall_turn_count == 2:
                    # 2nd consecutive stall turn — inject targeted instruction
                    if response.content:
                        working_memory.add_message("assistant", response.content)
                    working_memory.add_message(
                        "user",
                        "You have stated you are ready to answer multiple times without doing so. Produce the Final Answer: block now, based on the evidence already gathered."
                    )

                else:
                    # 3rd consecutive stall turn — safety fallback: force best-effort synthesis
                    fallback_narrative = "I've reached my search limit and will now synthesize the answer from the gathered evidence."
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
                            narrative=fallback_narrative
                        )
                    ))

                    # Quick best-effort synthesis request
                    synth_messages = working_memory.get_context() + [
                        Message(
                            role="user",
                            content="Provide your Final Answer now summarizing all findings gathered so far into a clear, comprehensive answer."
                        )
                    ]
                    synthesis_req = LLMRequest(
                        messages=synth_messages,
                        tools=[],
                        system=system_msg,
                        max_tokens=2048,
                        temperature=0.0
                    )
                    try:
                        synth_resp = await provider.complete(synthesis_req)
                        synth_action = OutputParser.parse(synth_resp)
                        best_effort_answer = synth_action.final_answer or synth_resp.content or "Synthesized answer based on gathered research."
                    except Exception:
                        best_effort_answer = response.content or "Synthesized answer based on gathered research."

                    action = AgentAction(
                        type=ActionType.FINAL_ANSWER,
                        narrative=fallback_narrative,
                        thought="Forcing final answer synthesis after stall threshold reached.",
                        final_answer=best_effort_answer.strip(),
                        raw_response=response.content or "",
                    )
                    episodic_memory.log_observation("Forced final answer after stall limit.", governor.steps)

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

                    # Safety net: reliably resolve final output text
                    final_answer_text = clean_final_answer(action.final_answer or "")
                    if not final_answer_text:
                        raw_candidate = (
                            getattr(action, "raw_response", None)
                            or getattr(response, "content", None)
                            or ""
                        ).strip()
                        if raw_candidate:
                            log.warning(
                                f"Track {track_id}: action.final_answer was empty/whitespace (stop={cond.name}); "
                                f"falling back to raw model response ({len(raw_candidate)} chars)."
                            )
                            final_answer_text = clean_final_answer(raw_candidate)
                            action.final_answer = final_answer_text

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
                            final_output=final_answer_text if final_answer_text else None,
                            # Rich meta packed into label so frontend can parse it
                            label=f"search_count={search_count}|elapsed={elapsed}|source_count={len(sources)}|sources={','.join(sources)}|model_used={last_model_used or ''}|level={level}"
                        )
                    ))

                    return AgentRunResult(
                        track_id=track_id,
                        final_output=final_answer_text,
                        steps_taken=governor.steps,
                        stop_reason=cond.name,
                        episodic_summary=episodic_memory.get_summary(),
                        success=bool(final_answer_text),
                        search_count=search_count,
                        elapsed_seconds=elapsed,
                        sources=sources,
                        model_used=last_model_used,
                        level=level,
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
