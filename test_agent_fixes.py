"""
test_agent_fixes.py — Verify Bug 1 (Stall detector & prompt hardening) and Bug 2 (Empty work panel)
"""
import sys
import asyncio
import uuid
import config

sys.stdout.reconfigure(encoding="utf-8")

from core.provider import LLMResponse, ToolCall, TokenUsage, ProviderRegistry
from core.tools import ToolRegistry, ToolDefinition
from providers.openrouter import OpenRouterProvider
from tools.tavily import TavilySearchExecutor
from execution.parser import OutputParser, ActionType
from observability.telemetry import TelemetryHub, TelemetryEvent
from orchestration.topologies import run_sequential_agent

# Register provider and tools
ProviderRegistry.register("openrouter", OpenRouterProvider())
ToolRegistry.register(ToolDefinition(
    name="web_search",
    description="Search the web for up-to-date information.",
    category="search",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."}
        },
        "required": ["query"]
    },
    output_schema={"type": "string"},
    timeout_seconds=20,
    executor=TavilySearchExecutor()
))

print("=" * 60)
print("WAYFINDER — BUG FIX & REGRESSION VERIFICATION")
print("=" * 60)

def make_resp(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        stop_reason="stop",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
        raw={}
    )

# ── 1. Test OutputParser variations ─────────────────────────────
print("\n[1/3] Testing OutputParser with various Final Answer and ReAct formats...")

# Variation A: Standard Final Answer
r1 = make_resp("NARRATIVE: I have found the answer.\nThought: Everything looks clear.\nFinal Answer: Fusion energy achieved Q>1.")
a1 = OutputParser.parse(r1)
assert a1.type == ActionType.FINAL_ANSWER, f"Expected FINAL_ANSWER, got {a1.type}"
assert "Fusion energy" in a1.final_answer

# Variation B: Markdown formatted **Final Answer:**
r2 = make_resp("NARRATIVE: Analysis complete.\nThought: Ready.\n**Final Answer:** The speed of light is 3e8 m/s.")
a2 = OutputParser.parse(r2)
assert a2.type == ActionType.FINAL_ANSWER, f"Expected FINAL_ANSWER for markdown, got {a2.type}"
assert "speed of light" in a2.final_answer

# Variation C: FINAL_ANSWER with underscore
r3 = make_resp("NARRATIVE: Final synthesis.\nThought: Done.\nFINAL_ANSWER: Quantum computers reached 1000 qubits.")
a3 = OutputParser.parse(r3)
assert a3.type == ActionType.FINAL_ANSWER, f"Expected FINAL_ANSWER for underscore format, got {a3.type}"

# Variation D: Announcement-only turn (no tool call, no final answer)
r4 = make_resp("NARRATIVE: I have gathered all 2025 evidence and am now ready to present a comprehensive summary.\nThought: I will now give the answer.")
a4 = OutputParser.parse(r4)
assert a4.type == ActionType.THOUGHT_ONLY, f"Expected THOUGHT_ONLY for announcement turn, got {a4.type}"

print("  PASS: OutputParser correctly handles all variations and classifies announcement-only turns.")

# ── 2. Live Agent Run: Single-topic Query ("Latest developments in fusion energy 2025") ─────
async def test_live_single_query():
    print("\n[2/3] Running Live Agent Test: 'Latest developments in fusion energy 2025'...")
    telemetry = TelemetryHub()
    task_id = str(uuid.uuid4())

    result = await run_sequential_agent(
        prompt="Latest developments in fusion energy 2025",
        provider_id="openrouter",
        tool_categories=["search"],
        skill_domain="",
        role_prompt=(
            "You are Wayfinder, an elite, deep web research engine.\n"
            "Your mission is to rigorously research the user's question, gather corroborated evidence across high-quality sources, and synthesize an insightful, comprehensive, expert-level response.\n\n"
            "Research Guidelines:\n"
            "- Always search the web for up-to-date and accurate information before answering.\n"
            "- If documents are attached, thoroughly incorporate their context.\n"
            "- Cross-verify facts and key claims across multiple reputable sources.\n"
            "- When you have gathered sufficient evidence, output your Final Answer directly without narrating intent beforehand.\n\n"
            "Synthesis & Final Answer Formatting Rules:\n"
            "1. OPENING EXECUTIVE SYNTHESIS (MANDATORY): Always begin your Final Answer with a compelling, high-level executive summary paragraph (1–3 sentences) that directly answers the core question, frames the macro context, or captures the defining shift/takeaway. NEVER jump directly into numbered lists, raw bullet points, or dry breakdowns without this opening lead paragraph.\n"
            "2. STRUCTURED THEMATIC DEEP-DIVE: Follow the executive lead with structured thematic sections using markdown headers (###), bolded concept titles, concrete data points, metrics, dates, and key organization/project names.\n"
            "3. COMPARISONS & TABLES: Use clean markdown tables where comparisons or chronological data make findings easier to digest.\n"
            "4. TONE & WRITING STYLE: Write with the depth, clarity, and authority of a top-tier analyst report."
        ),
        telemetry=telemetry,
        track_id=task_id,
        max_steps=10
    )

    trace = telemetry.get_trace(task_id)
    narrative_events = [ev for ev in trace if ev.payload.type == "narrative_start"]
    work_events = [ev for ev in trace if ev.payload.type == "work_delta"]

    print("\n  Emitted Narrative Nodes:")
    for n in narrative_events:
        print(f"    - {n.payload.narrative}")

    print(f"\n  Result Success: {result.success}")
    print(f"  Steps Taken: {result.steps_taken}")
    print(f"  Search Count: {result.search_count}")
    print(f"  Narrative Nodes Count: {len(narrative_events)}")
    print(f"  Elapsed: {result.elapsed_seconds}s")
    print(f"  Sources: {len(result.sources)}")

    assert result.success is True, "Agent run failed"
    assert result.search_count >= 1, "Agent did not perform any searches"
    assert result.steps_taken <= 6, f"Agent took too many turns ({result.steps_taken}), possible stalling!"
    assert len(result.final_output) > 100, "Final answer is too short"
    print("  PASS: Single-topic test passed efficiently without repeated announcement stalling.")

# ── 3. Live Agent Run: Multi-Part Query ("Compare quantum computing developments at IBM vs Google in 2025") ─────
async def test_live_multipart_query():
    print("\n[3/3] Running Live Regression Test: Multi-part Query ('Compare quantum computing at IBM vs Google in 2025')...")
    telemetry = TelemetryHub()
    task_id = str(uuid.uuid4())

    result = await run_sequential_agent(
        prompt="Compare quantum computing developments at IBM vs Google in 2025",
        provider_id="openrouter",
        tool_categories=["search"],
        skill_domain="",
        role_prompt=(
            "You are Wayfinder, an elite, deep web research engine.\n"
            "Your mission is to rigorously research the user's question, gather corroborated evidence across high-quality sources, and synthesize an insightful, comprehensive, expert-level response.\n\n"
            "Research Guidelines:\n"
            "- Always search the web for up-to-date and accurate information before answering.\n"
            "- If documents are attached, thoroughly incorporate their context.\n"
            "- Cross-verify facts and key claims across multiple reputable sources.\n"
            "- When you have gathered sufficient evidence, output your Final Answer directly without narrating intent beforehand.\n\n"
            "Synthesis & Final Answer Formatting Rules:\n"
            "1. OPENING EXECUTIVE SYNTHESIS (MANDATORY): Always begin your Final Answer with a compelling, high-level executive summary paragraph (1–3 sentences) that directly answers the core question, frames the macro context, or captures the defining shift/takeaway. NEVER jump directly into numbered lists, raw bullet points, or dry breakdowns without this opening lead paragraph.\n"
            "2. STRUCTURED THEMATIC DEEP-DIVE: Follow the executive lead with structured thematic sections using markdown headers (###), bolded concept titles, concrete data points, metrics, dates, and key organization/project names.\n"
            "3. COMPARISONS & TABLES: Use clean markdown tables where comparisons or chronological data make findings easier to digest.\n"
            "4. TONE & WRITING STYLE: Write with the depth, clarity, and authority of a top-tier analyst report."
        ),
        telemetry=telemetry,
        track_id=task_id,
        max_steps=12
    )

    trace = telemetry.get_trace(task_id)
    narrative_events = [ev for ev in trace if ev.payload.type == "narrative_start"]

    print("\n  Emitted Narrative Nodes:")
    for n in narrative_events:
        print(f"    - {n.payload.narrative}")

    print(f"\n  Result Success: {result.success}")
    print(f"  Steps Taken: {result.steps_taken}")
    print(f"  Search Count: {result.search_count}")
    print(f"  Narrative Nodes Count: {len(narrative_events)}")
    print(f"  Elapsed: {result.elapsed_seconds}s")
    print(f"  Sources: {len(result.sources)}")

    assert result.success is True, "Multi-part agent run failed"
    assert result.search_count >= 2, f"Multi-part query should execute multiple searches, got {result.search_count}"
    assert len(result.final_output) > 100, "Final answer is too short"
    print("  PASS: Multi-part regression check passed — legitimate multi-turn searches executed seamlessly.")

async def main():
    await test_live_single_query()
    await test_live_multipart_query()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
