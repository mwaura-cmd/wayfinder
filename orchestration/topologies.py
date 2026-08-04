import uuid
import datetime
from typing import List, Optional, Dict, Any

import config
from core.provider import ProviderRegistry
from core.tools import ToolRegistry
from core.skills import SkillRegistry
from core.memory import WorkingMemory, EpisodicMemory
from core.prompt import PromptAssembler
from observability.telemetry import TelemetryHub, TelemetryEvent, Payload
from execution.engine import LoopEngine, AgentRunResult
from execution.governor import IterationGovernor
from execution.conditions import StopConditionRegistry
from execution.errors import ErrorHandler

async def run_sequential_agent(
    prompt: str,
    provider_id: str,
    tool_categories: List[str],
    skill_domain: str,
    role_prompt: str,
    telemetry: TelemetryHub,
    track_id: str,          # Passed in from main.py so frontend can subscribe by the same ID
    max_steps: Optional[int] = None,
    level: str = "standard",
    prior_messages: Optional[List[Dict[str, Any]]] = None,
) -> AgentRunResult:
    
    provider = ProviderRegistry.get(provider_id)
    tools = ToolRegistry.get_scope(tool_categories)
    
    # Optional skills depending on whether they are registered
    skills = SkillRegistry.get_by_domain(skill_domain) if skill_domain else []

    # Determine turn cap based on research level
    if max_steps is None:
        level_cfg = config.RESEARCH_LEVELS.get(
            level, config.RESEARCH_LEVELS.get(config.DEFAULT_RESEARCH_LEVEL, {"max_turns": 4})
        )
        effective_max_steps = level_cfg["max_turns"]
    else:
        effective_max_steps = max_steps

    working_memory = WorkingMemory()
    episodic_memory = EpisodicMemory()
    governor = IterationGovernor(max_steps=effective_max_steps, timeout_seconds=120)
    stop_conditions = StopConditionRegistry.defaults()
    error_handler = ErrorHandler()

    system_prompt = PromptAssembler.build_system(
        role_prompt=role_prompt,
        tools=tools,
        skills=skills,
        provider=provider
    )

    working_memory.add_message("system", system_prompt)

    # If this is a conversational follow-up, seed working memory with prior thread context
    if prior_messages:
        for pm in prior_messages:
            q = pm.get("question") or ""
            a = pm.get("answer") or ""
            if q:
                working_memory.add_message("user", q)
            if a:
                working_memory.add_message("model", a)

    working_memory.add_message("user", prompt)

    await telemetry.emit(TelemetryEvent(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        track_id=track_id,
        parent_track_id=None,
        track_type="root",
        source_type="system",
        payload=Payload(
            type="narrative_start",
            node_id=str(uuid.uuid4()),
            narrative="Initiating analysis..."
        )
    ))

    return await LoopEngine.run(
        prompt=prompt,
        provider=provider,
        tools=tools,
        skills=skills,
        working_memory=working_memory,
        episodic_memory=episodic_memory,
        telemetry=telemetry,
        track_id=track_id,
        parent_track_id=None,
        stop_conditions=stop_conditions,
        governor=governor,
        error_handler=error_handler,
        level=level
    )

