from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class SkillResult(BaseModel):
    skill_name: str
    output: str
    track_id: str
    steps_taken: int
    success: bool
    error: Optional[str] = None

class SkillDefinition(BaseModel):
    name: str
    description: str
    domain: str
    tools: List[str]
    prompt_template: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    max_steps: int

class SkillRegistry:
    _skills: dict[str, SkillDefinition] = {}

    @classmethod
    def register(cls, skill: SkillDefinition) -> None:
        if skill.name in cls._skills:
            raise ValueError(f"Skill {skill.name} already registered")
        cls._skills[skill.name] = skill

    @classmethod
    def get(cls, name: str) -> SkillDefinition:
        if name not in cls._skills:
            raise ValueError(f"Skill {name} not found")
        return cls._skills[name]

    @classmethod
    def get_by_domain(cls, domain: str) -> List[SkillDefinition]:
        return [s for s in cls._skills.values() if s.domain == domain]

    @classmethod
    def get_all(cls) -> List[SkillDefinition]:
        return list(cls._skills.values())

class SkillExecutor:
    async def execute(
        self,
        skill: SkillDefinition,
        inputs: dict,
        parent_track_id: str,
        telemetry: Any, # TelemetryHub
        provider: Any,  # BaseLLMProvider
        memory: Any     # WorkingMemory
    ) -> SkillResult:
        # Implementation of bounded skill execution happens here.
        # This will be fully implemented in the Execution phase when engine is ready.
        raise NotImplementedError
