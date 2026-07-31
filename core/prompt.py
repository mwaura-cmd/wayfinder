from typing import List
from core.provider import BaseLLMProvider
from core.tools import ToolDefinition, ToolRegistry
from core.skills import SkillDefinition

class PromptAssembler:
    @classmethod
    def build_system(
        cls,
        role_prompt: str,
        tools: List[ToolDefinition],
        skills: List[SkillDefinition],
        provider: BaseLLMProvider,
        narrative_injection: bool = True
    ) -> str:
        
        # Block 1: Role
        blocks = [role_prompt]

        # Block 2: Narrative Injection
        if narrative_injection:
            blocks.append(
                "Before taking any action, state your intent in one clear present-tense sentence."
            )

        # Block 3: Tool & Skill Descriptions
        tool_desc = ToolRegistry.describe_for_prompt(tools)
        if tool_desc:
            blocks.append("Available Tools:\n" + tool_desc)
            
        if skills:
            skill_lines = []
            for s in skills:
                skill_lines.append(f"- {s.name}: {s.description}")
            blocks.append("Available Skills:\n" + "\n".join(skill_lines))

        # Block 4: ReAct Format Template
        react_format = """
Respond in the following format:
Thought: [your reasoning]
Action: [tool_name or skill_name]
Action Input: [JSON parameters]

Once you have a final answer, use this format:
Thought: [your reasoning]
Final Answer: [response to the user]
"""
        blocks.append(react_format)

        return "\n\n".join(blocks)
