import json
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


# ── Compiled patterns (text-based ReAct format) ──────────────────────────────
_NARRATIVE_RE  = re.compile(r"(?:\*\*|###\s*)?(?:NARRATIVE|NARRATIVE_START)(?:\*\*)?[:\s]*(?:\*\*)?[:\s]*\s*(.+?)(?=\n(?:\*\*|###\s*)?(?:Thought|Action|Final\s*Answer|FINAL_ANSWER|Final\s*Synthesis)|\Z)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE    = re.compile(r"(?:\*\*|###\s*)?Thought(?:\*\*)?[:\s]*(?:\*\*)?[:\s]*\s*(.+?)(?=\n(?:\*\*|###\s*)?(?:Action|Final\s*Answer|FINAL_ANSWER|Final\s*Synthesis)|\Z)", re.IGNORECASE | re.DOTALL)
_ACTION_RE     = re.compile(r"(?:\*\*|###\s*)?Action(?:\*\*)?[:\s]*(?:\*\*)?[:\s]*\s*([\w_]+)", re.IGNORECASE)
_INPUT_RE      = re.compile(r"(?:\*\*|###\s*)?Action Input(?:\*\*)?[:\s]*(?:\*\*)?[:\s]*\s*(\{.*)", re.IGNORECASE | re.DOTALL)
_FINAL_RE      = re.compile(r"(?:\*\*|###\s*)?(?:Final\s*Answer|FINAL_ANSWER|Final\s*Synthesis)(?:\*\*)?[:\s]*(?:\*\*)?[:\s]*\s*(.+)", re.IGNORECASE | re.DOTALL)


def _extract_first_json(text: str) -> dict:
    """Extract the first valid JSON object from text."""
    # Find the first { and try progressively longer slices until valid JSON
    start = text.find("{")
    if start == -1:
        return {}
    for end in range(len(text), start, -1):
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    return {}


class OutputParser:
    @classmethod
    def parse(cls, response: LLMResponse) -> AgentAction:
        raw = response.content or ""
        native_calls = response.tool_calls or []

        # ── Priority 1: Native function-call response (Gemini FC mode) ────────
        # This only triggers if we registered native tools AND Gemini used them.
        if native_calls:
            tc = native_calls[0]
            narrative = f"Let me search for '{tc.inputs.get('query', tc.tool_name)}'."
            # Try to extract a NARRATIVE from any accompanying text
            m = _NARRATIVE_RE.search(raw)
            if m:
                narrative = m.group(1).strip()
            return AgentAction(
                type=ActionType.TOOL_CALL,
                narrative=narrative,
                thought=raw[:200] if raw else "Using tool.",
                tool_call=tc,
                raw_response=raw,
            )

        # ── Priority 2: Text-based ReAct parsing ──────────────────────────────
        # Extract NARRATIVE
        narrative = ""
        m = _NARRATIVE_RE.search(raw)
        if m:
            narrative = m.group(1).strip()

        # Extract Thought
        thought = "Reasoning..."
        m = _THOUGHT_RE.search(raw)
        if m:
            thought = m.group(1).strip()

        # ── Check for Action: + Action Input: (tool call in text format) ──────
        action_m = _ACTION_RE.search(raw)
        input_m  = _INPUT_RE.search(raw)

        if action_m and input_m:
            tool_name = action_m.group(1).strip()
            raw_input = input_m.group(1).strip()
            inputs = _extract_first_json(raw_input)

            if not narrative:
                query = inputs.get("query", "")
                if query:
                    narrative = f"Let me search for '{query}'."
                else:
                    narrative = f"Let me use {tool_name} to gather more information."

            return AgentAction(
                type=ActionType.TOOL_CALL,
                narrative=narrative,
                thought=thought,
                tool_call=ToolCall(
                    tool_name=tool_name,
                    inputs=inputs,
                    call_id=str(uuid.uuid4()),
                ),
                raw_response=raw,
            )

        # ── Check for Final Answer: ────────────────────────────────────────────
        m = _FINAL_RE.search(raw)
        if m:
            extracted_ans = m.group(1).strip()
            final_answer = extracted_ans if extracted_ans else raw.strip()
            if not narrative:
                narrative = "I have enough information. Here is my answer."
            return AgentAction(
                type=ActionType.FINAL_ANSWER,
                narrative=narrative,
                thought=thought,
                final_answer=final_answer,
                raw_response=raw,
            )

        # ── Fallback: model is still deliberating (thought-only) ──────────────
        # Do NOT treat this as a final answer — let the loop continue.
        if not narrative:
            narrative = "Let me think about how to approach this..."
        return AgentAction(
            type=ActionType.THOUGHT_ONLY,
            narrative=narrative,
            thought=thought,
            raw_response=raw,
        )
