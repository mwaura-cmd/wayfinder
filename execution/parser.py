import re
import uuid
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from core.provider import LLMResponse, ToolCall


class ActionType(str, Enum):
    TOOL_CALL = "tool_call"
    SKILL_CALL = "skill_call"
    FINAL_ANSWER = "final_answer"
    THOUGHT_ONLY = "thought_only"


class SkillCall(BaseModel):
    skill_name: str
    inputs: dict
    call_id: str


class AgentAction(BaseModel):
    type: ActionType
    narrative: str
    thought: str
    tool_call: Optional[ToolCall] = None
    skill_call: Optional[SkillCall] = None
    final_answer: Optional[str] = None
    raw_response: str


class OutputParser:
    # ── Regex patterns for text-mode ReAct output ──────────────────────────
    _NARRATIVE_RE = re.compile(r"NARRATIVE:\s*(.+?)(?=\n|$)", re.IGNORECASE)
    _THOUGHT_RE   = re.compile(r"Thought:\s*(.+?)(?=\nAction:|$)", re.IGNORECASE | re.DOTALL)
    _FINAL_RE     = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)

    @classmethod
    def parse(cls, response: LLMResponse) -> AgentAction:
        raw = response.content or ""
        tool_calls = response.tool_calls or []

        # ── Step 1: Extract narrative from model text ──────────────────────
        # Use NARRATIVE: tag if present (ReAct text mode).
        narrative = ""
        m = cls._NARRATIVE_RE.search(raw)
        if m:
            narrative = m.group(1).strip()

        # If no explicit NARRATIVE tag, use the first non-empty text line
        # (this is common in native function-calling mode where the model
        # includes a short preamble sentence before the function call).
        if not narrative and raw:
            first_line = raw.strip().split("\n")[0].strip()
            # Accept it as narrative if it looks like a sentence (not a keyword)
            if first_line and not first_line.lower().startswith(("thought:", "action:", "observation:")):
                narrative = first_line

        # ── Step 2: Extract thought ────────────────────────────────────────
        thought = "Reasoning..."
        m = cls._THOUGHT_RE.search(raw)
        if m:
            thought = m.group(1).strip()
        elif raw and not narrative:
            thought = raw[:200]

        # ── Step 3: Native function call (Gemini function-calling mode) ────
        if tool_calls:
            tc = tool_calls[0]
            # Generate a natural narrative from the tool call if we don't have one
            if not narrative:
                query = tc.inputs.get("query", "")
                if query:
                    narrative = f"Let me search for '{query}'."
                else:
                    narrative = f"Let me use {tc.tool_name} to gather information."

            return AgentAction(
                type=ActionType.TOOL_CALL,
                narrative=narrative,
                thought=thought,
                tool_call=tc,
                raw_response=raw,
            )

        # ── Step 4: Final Answer detection (text-mode ReAct) ───────────────
        m = cls._FINAL_RE.search(raw)
        if m:
            final_answer = m.group(1).strip()
            if not narrative:
                narrative = "I have enough information. Let me put together the final answer."
            return AgentAction(
                type=ActionType.FINAL_ANSWER,
                narrative=narrative,
                thought=thought,
                final_answer=final_answer,
                raw_response=raw,
            )

        # ── Step 5: Treat the entire response as the final answer if it
        #    looks substantive (Gemini sometimes returns a direct answer
        #    without the "Final Answer:" prefix when no tools are registered
        #    or all searches are done).
        if raw and len(raw.strip()) > 80:
            if not narrative:
                narrative = "I have gathered enough information. Here is my answer."
            return AgentAction(
                type=ActionType.FINAL_ANSWER,
                narrative=narrative,
                thought=thought,
                final_answer=raw.strip(),
                raw_response=raw,
            )

        # ── Step 6: Fallback — thought-only (model is deliberating) ────────
        if not narrative:
            narrative = "Let me think about this..."
        return AgentAction(
            type=ActionType.THOUGHT_ONLY,
            narrative=narrative,
            thought=thought,
            raw_response=raw,
        )
