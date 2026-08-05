from typing import List, Any
from core.provider import BaseLLMProvider
from core.tools import ToolDefinition
from core.skills import SkillDefinition


class PromptAssembler:
    @classmethod
    def build_system(
        cls,
        role_prompt: str,
        tools: List[ToolDefinition],
        skills: List[SkillDefinition],
        provider: BaseLLMProvider,
    ) -> str:
        blocks = []

        # Block 1: Role
        blocks.append(role_prompt)

        # Block 2: Tool descriptions (tell the model what's available)
        if tools:
            tool_lines = []
            for t in tools:
                # Extract input field descriptions from schema
                props = t.input_schema.get("properties", {})
                param_desc = ", ".join(
                    f'"{k}": {v.get("description", k)}' for k, v in props.items()
                )
                tool_lines.append(
                    f"- {t.name}: {t.description}\n"
                    f"  Input JSON: {{{param_desc}}}"
                )
            blocks.append("Available Tools:\n" + "\n".join(tool_lines))

        if skills:
            skill_lines = [f"- {s.name}: {s.description}" for s in skills]
            blocks.append("Available Skills:\n" + "\n".join(skill_lines))

        # Block 3: Strict output format (text-based ReAct)
        # This is the ONLY format the model should use — no deviation.
        tool_name_example = tools[0].name if tools else "tool_name"
        format_block = f"""You MUST respond in one of these two formats ONLY — no other format is acceptable.

FORMAT A — when you need to use a tool:
NARRATIVE: <One first-person sentence describing what you are about to do — this is shown to the user>
Thought: <Your reasoning>
Action: {tool_name_example}
Action Input: {{"query": "your search terms"}}

FORMAT B — when you have enough information to give a complete answer:
NARRATIVE: <One first-person sentence stating research is complete and synthesis is ready>
Thought: <Your reasoning>
Final Answer: <Your synthesized response. CRITICAL: Always begin directly with an executive summary / lead synthesis paragraph framing the macro insight, followed by structured deep-dive sections with markdown headings (###), bolded key points, and specific data.>

Rules:
- NARRATIVE must always be present and must be a natural, specific, first-person sentence.
- Never skip the NARRATIVE line.
- Never mix Format A and Format B in one response.
- Initial Turn: For any research inquiry, you MUST ALWAYS begin with FORMAT A to execute an active search and retrieve grounded sources before concluding. Never provide Format B on step 1 without performing research.
- In Final Answer: ALWAYS open directly with an engaging, high-level executive summary paragraph (1–3 sentences) summarizing the big picture and bottom-line answer before diving into structured numbered points or sections. Do NOT include label headers like 'Executive Summary:' or 'Opening Synthesis:' — begin directly with the text. Never start directly with bullet points or numbered lists.
- Do NOT produce a narrative node that merely announces you are about to answer (e.g. "I can now provide a summary", "I have gathered enough evidence") without immediately including the actual Final Answer: block in that same response. If you have enough evidence to answer, output the Final Answer: block directly — do not narrate your intent to answer as a separate step first.
- If the user's message is a follow-up to a previous answer in this thread, use the prior answer and sources as context — only search the web again if the follow-up genuinely requires new information the prior answer didn't cover. Do not re-research from scratch for a simple clarification or extension of a prior topic.
- After each Observation you receive, output the next Format A or Format B — never stop mid-way.
- When in doubt, do one more search rather than guessing."""

        blocks.append(format_block)

        return "\n\n".join(blocks)
