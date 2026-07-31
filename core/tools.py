from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ToolError(BaseModel):
    code: str
    message: str
    recoverable: bool

class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    output: str
    raw: dict
    success: bool
    error: Optional[ToolError] = None

class BaseToolExecutor:
    async def execute(
        self,
        inputs: dict,
        telemetry: Any, # TelemetryHub
        track_id: str
    ) -> ToolResult:
        raise NotImplementedError

class ToolDefinition(BaseModel):
    name: str
    description: str
    category: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    timeout_seconds: int
    executor: BaseToolExecutor

    model_config = {"arbitrary_types_allowed": True}

class ToolRegistry:
    _tools: dict[str, ToolDefinition] = {}

    @classmethod
    def register(cls, tool: ToolDefinition) -> None:
        if tool.name in cls._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> ToolDefinition:
        if name not in cls._tools:
            raise ValueError(f"Tool {name} not found")
        return cls._tools[name]

    @classmethod
    def get_scope(cls, categories: List[str]) -> List[ToolDefinition]:
        return [t for t in cls._tools.values() if t.category in categories]

    @classmethod
    def get_all(cls) -> List[ToolDefinition]:
        return list(cls._tools.values())

    @classmethod
    def describe_for_prompt(cls, tools: List[ToolDefinition]) -> str:
        lines = []
        for t in tools:
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)
