import uuid
import datetime
from typing import List

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
    max_steps: int = 15
) -> AgentRunResult:
    
    provider = ProviderRegistry.get(provider_id)
    tools = ToolRegistry.get_scope(tool_categories)
    
    # Optional skills depending on whether they are registered
    skills = SkillRegistry.get_by_domain(skill_domain) if skill_domain else []

    # Use the provided track_id so the frontend subscription matches
    working_memory = WorkingMemory()
    episodic_memory = EpisodicMemory()
    governor = IterationGovernor(max_steps=max_steps, timeout_seconds=120)
    stop_conditions = StopConditionRegistry.defaults()
    error_handler = ErrorHandler()

    system_prompt = PromptAssembler.build_system(
        role_prompt=role_prompt,
        tools=tools,
        skills=skills,
        provider=provider
    )

    working_memory.add_message("system", system_prompt)
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
        error_handler=error_handler
    )
