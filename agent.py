"""
agent.py — The Wayfinder ReAct agent loop (§2).

Architecture: single agent, two tools, plain function-calling loop.
No LangGraph, no CrewAI, no heavyweight framework. (§7)

Uses the new `google-genai` SDK (google.genai), not the deprecated
`google-generativeai` package.

Loop mechanics
--------------
  turn 0   : Model receives system prompt + user question
  turn n   : Model either calls a tool OR produces a final answer
  per turn : If tool call → execute → feed result back → repeat
  exit cond: Model produces final_answer text OR max_turns exceeded

Context management (§4)
-----------------------
Only trimmed results (URL + short snippet) are kept. After COMPACTION_THRESHOLD
turns, old raw search tool results are replaced with a compact placeholder so
the context window doesn't blow up across long multi-search sessions.
This is distinct from §9 persistent memory.

Each event yielded from the loop is a dict:
  {"type": "narrative", "text": "..."}          — Layer-1 node (§8)
  {"type": "work",      "text": "..."}          — Layer-2 work-panel update (§8)
  {"type": "answer",    "text": "...", "meta": {sources, turn_count, elapsed}}
  {"type": "error",     "text": "..."}
"""
import json
import logging
import re
import time
from collections.abc import Generator
from typing import Any, Optional

from google import genai
from google.genai import types as gtypes

import config
import memory as mem_module
import tools as tool_module

log = logging.getLogger(__name__)


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Wayfinder, a precise and diligent web research agent.
Your job is to answer the user's question by searching the web methodically and
synthesising what you find — not by recalling training data.

═══════════════════════════════════════
CORE OPERATING RULES
═══════════════════════════════════════

1. ALWAYS call check_memory FIRST — before any web search — using 2-5 keywords
   from the user's question. If memory contains a relevant past session, open
   a narrative node saying "I've researched something similar before — let me
   check what I found," then use that context to inform your current plan.

2. Break multi-part questions into SEPARATE sub-questions. Call web_search
   once per sub-question with a focused query. Never combine multiple
   sub-questions into a single search call.

3. After each search, decide one of:
   (a) Search again with different terms — if results were thin, stale, or
       unclear.
   (b) Search a different sub-question — if this sub-question is answered
       and more remain.
   (c) Answer — only when you have enough reliable, current evidence.

4. If results conflict, do NOT guess or split the difference. Re-search with
   more specific or authoritative terms (e.g. add "official", "announcement",
   or the primary source name) before concluding.

5. Never fabricate a search result. If a search returns nothing useful, say
   so plainly. Never fill gaps from your own training data without stating
   explicitly that you are doing so and why.

6. You have a maximum of {max_turns} search turns. If you reach the cap
   without a confident answer, say so explicitly: "I wasn't able to find
   a confident answer within the search limit. Here is what I found so far:…"
   — NEVER produce a finished-looking answer to avoid admitting you're stuck.

7. Periodically remind yourself (in your reasoning) of the ORIGINAL question.
   Do not drift: each search turn should connect back to answering that
   original question, not a tangentially related tangent that emerged.

═══════════════════════════════════════
NARRATIVE NODE GENERATION (UI §8)
═══════════════════════════════════════

Each time your objective changes — after checking memory, after each search,
after deciding to switch sub-questions — output a narrative node in this format:

  NARRATIVE: <short, first-person statement describing your current intent>

Write it as a competent professional naturally would. Capture:
  - What you just learned (from memory or a search result)
  - What you're doing next
  - Why (briefly)

GOOD: "I found several sources on the policy change — I want to compare
      the dates before drawing a conclusion."
GOOD: "The results are thin here; let me try a more specific query."
GOOD: "Memory shows I've researched this topic before — I'll use that as
      a baseline and check for anything newer."

BAD:  "Searching." / "Step 3." / "Calling web_search." / "Executing tool."
BAD:  Confidence scores, internal calculations, implementation details.

Do NOT expose raw chain-of-thought, tool implementation details, or
confidence percentages in narrative nodes. Keep them conversational.

═══════════════════════════════════════
FINAL ANSWER FORMAT
═══════════════════════════════════════

When you have enough evidence, produce your final answer using this structure:

FINAL_ANSWER:
<Your answer here — clear, well-organised, citing sources inline as [Source N]>

SOURCES:
- [Source 1] Title — URL
- [Source 2] Title — URL
(list every source you actually used)

Never omit the SOURCES section. If you used memory from a past session,
include those sources too, noting they came from a prior research session.
""".format(max_turns=config.MAX_TURNS)


# ── Tool schema for google-genai SDK ─────────────────────────────────────────

def _build_tools() -> list[gtypes.Tool]:
    """Convert our TOOL_DECLARATIONS dicts into google-genai Tool objects."""
    declarations = []
    for t in tool_module.TOOL_DECLARATIONS:
        props: dict[str, gtypes.Schema] = {}
        for pname, pdef in t["parameters"]["properties"].items():
            if pdef["type"] == "array":
                item_type = _map_type(pdef["items"]["type"])
                props[pname] = gtypes.Schema(
                    type=_map_type(pdef["type"]),
                    items=gtypes.Schema(type=item_type),
                    description=pdef.get("description", ""),
                )
            else:
                props[pname] = gtypes.Schema(
                    type=_map_type(pdef["type"]),
                    description=pdef.get("description", ""),
                )

        declarations.append(
            gtypes.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=gtypes.Schema(
                    type="OBJECT",
                    properties=props,
                    required=t["parameters"].get("required", []),
                ),
            )
        )
    return [gtypes.Tool(function_declarations=declarations)]


def _map_type(t: str) -> str:
    return {"string": "STRING", "array": "ARRAY", "object": "OBJECT", "integer": "INTEGER"}.get(t, "STRING")


# ── Retry helper for Gemini calls ──────────────────────────────────────────────

def _call_with_retry(fn, *args, **kwargs):
    """Exponential-backoff retry (§6): 1s → 2s → 4s, max MAX_RETRIES attempts."""
    delay = config.RETRY_BASE_SECONDS
    last_err = ""
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs), None
        except Exception as exc:
            exc_str = str(exc)
            is_rate = "429" in exc_str or "quota" in exc_str.lower() or "rate" in exc_str.lower()
            last_err = (
                "Gemini's free-tier rate limit was hit — please wait ~1 minute."
                if is_rate
                else f"Gemini API error: {exc}"
            )
            log.warning("Gemini attempt %d/%d failed: %s", attempt, config.MAX_RETRIES, exc)
            if attempt < config.MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    return None, last_err


# ── Helpers ────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"URL:\s*(https?://\S+)")
_NARRATIVE_RE = re.compile(r"NARRATIVE:\s*(.+?)(?=\nNARRATIVE:|\nFINAL_ANSWER:|$)", re.DOTALL)


def _extract_sources(text: str) -> list[str]:
    urls = _URL_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _extract_narratives(text: str) -> list[str]:
    return [m.strip() for m in _NARRATIVE_RE.findall(text)]


def _safe_text(response) -> str:
    """Extract text from a Gemini response safely."""
    try:
        return response.text or ""
    except Exception:
        pass
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                return part.text
    except Exception:
        pass
    return ""


def _get_function_calls(response) -> list:
    """Extract function_call parts from a Gemini response."""
    calls = []
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                calls.append(part.function_call)
    except Exception:
        pass
    return calls


# ── Main agent loop ────────────────────────────────────────────────────────────

def run_agent(
    question: str,
    db_conn,
) -> Generator[dict, None, None]:
    """
    Execute the full ReAct loop for `question`.
    Yields event dicts consumed by the SSE endpoint in main.py.
    """
    if not config.GEMINI_API_KEY:
        yield {"type": "error", "text": "GEMINI_API_KEY is not set. Add it to your env file."}
        return

    start_time = time.time()
    all_sources: list[str] = []
    turn = 0

    # ── Build client ──────────────────────────────────────────────────────────
    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
    except Exception as exc:
        yield {"type": "error", "text": f"Failed to initialise Gemini client: {exc}"}
        return

    tools = _build_tools()
    gen_config = gtypes.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        temperature=0.3,
    )

    yield {"type": "work", "text": f"📋 Research task received: {question}"}

    # ── Build conversation history ─────────────────────────────────────────────
    # We maintain the history manually so we can compact old entries (§4).
    contents: list[gtypes.Content] = [
        gtypes.Content(role="user", parts=[gtypes.Part(text=question)])
    ]

    # ── Agent loop ─────────────────────────────────────────────────────────────
    while turn <= config.MAX_TURNS:

        # Call the model
        resp, err = _call_with_retry(
            client.models.generate_content,
            model=config.GEMINI_MODEL,
            contents=contents,
            config=gen_config,
        )
        if err:
            yield {"type": "error", "text": err}
            return

        # Add model response to history
        try:
            contents.append(resp.candidates[0].content)
        except Exception:
            pass

        # ── Check for text output (narrative nodes or final answer) ───────────
        response_text = _safe_text(resp)

        if response_text:
            for narrative in _extract_narratives(response_text):
                yield {"type": "narrative", "text": narrative}

        # ── Check for tool calls ──────────────────────────────────────────────
        function_calls = _get_function_calls(resp)

        if not function_calls:
            # No tool calls → final answer
            final_text = response_text
            if not final_text:
                final_text = "The agent produced an empty response. Please try again."

            # Parse FINAL_ANSWER block if present
            if "FINAL_ANSWER:" in final_text:
                answer_body = final_text.split("FINAL_ANSWER:", 1)[1]
                if "SOURCES:" in answer_body:
                    answer_clean = answer_body.split("SOURCES:")[0].strip()
                    sources_block = answer_body.split("SOURCES:")[1].strip()
                    found_urls = _extract_sources(sources_block)
                    all_sources = list(dict.fromkeys(all_sources + found_urls))
                else:
                    answer_clean = answer_body.strip()
            else:
                answer_clean = final_text
                found_urls = _extract_sources(final_text)
                all_sources = list(dict.fromkeys(all_sources + found_urls))

            elapsed = round(time.time() - start_time, 1)

            # Persist session (§9)
            mem_module.save_session(
                db_conn,
                question=question,
                answer=answer_clean[:2000],
                sources=all_sources[:20],
            )

            yield {
                "type": "answer",
                "text": answer_clean,
                "meta": {
                    "sources": all_sources,
                    "source_count": len(all_sources),
                    "search_count": turn,
                    "elapsed_seconds": elapsed,
                    "turn_count": turn,
                },
            }
            return

        # ── Execute tool calls ────────────────────────────────────────────────
        turn += 1
        tool_response_parts: list[gtypes.Part] = []

        for fc in function_calls:
            tool_name = fc.name
            # fc.args is a MapComposite — convert to plain dict
            tool_args = dict(fc.args) if fc.args else {}

            if tool_name == "web_search":
                query = tool_args.get("query", "")
                yield {"type": "work", "text": f"🔍 Searching: {query}"}
                result_text = tool_module.web_search(query)
                found_urls = _URL_RE.findall(result_text)
                all_sources = list(dict.fromkeys(all_sources + found_urls))
                yield {"type": "work", "text": f"📄 Got {len(found_urls)} source(s)"}

            elif tool_name == "check_memory":
                keywords = list(tool_args.get("keywords", []))
                yield {"type": "work", "text": f"🧠 Checking memory: {', '.join(keywords)}"}
                result_text = tool_module.check_memory(db_conn, keywords)
                if "Found" in result_text and "past session" in result_text:
                    yield {
                        "type": "narrative",
                        "text": "I've researched something similar before — let me check what I found and use it as a starting point.",
                    }
                snippet = result_text[:200] + ("…" if len(result_text) > 200 else "")
                yield {"type": "work", "text": f"🧠 Memory: {snippet}"}

            else:
                result_text = f"Unknown tool '{tool_name}'."

            tool_response_parts.append(
                gtypes.Part.from_function_response(
                    name=tool_name,
                    response={"result": result_text},
                )
            )

        # Add tool results to conversation history
        contents.append(
            gtypes.Content(role="tool", parts=tool_response_parts)
        )

        # ── Context compaction (§4) ───────────────────────────────────────────
        # Replace raw tool-result text in entries older than 3 turns with
        # a compact placeholder. Keeps token usage bounded over long sessions.
        if turn > 3:
            compact_before = len(contents) - 6  # keep last 3 tool-result pairs
            for i, entry in enumerate(contents[:compact_before]):
                if (
                    hasattr(entry, "role") and entry.role == "tool"
                    and not getattr(entry, "_compacted", False)
                ):
                    try:
                        contents[i] = gtypes.Content(
                            role="tool",
                            parts=[gtypes.Part(text="[Search result compacted — content already processed by the model]")],
                        )
                        contents[i]._compacted = True  # type: ignore
                    except Exception:
                        pass

    # ── Max-turns reached ─────────────────────────────────────────────────────
    yield {
        "type": "narrative",
        "text": "I've reached my search limit. Let me share my best answer from what I've found so far.",
    }

    resp, err = _call_with_retry(
        client.models.generate_content,
        model=config.GEMINI_MODEL,
        contents=contents + [
            gtypes.Content(
                role="user",
                parts=[gtypes.Part(text=(
                    f"You have reached the maximum of {config.MAX_TURNS} search turns. "
                    "Produce your best answer now based on what you have found. "
                    "Be explicit about any gaps or uncertainties. "
                    "Do NOT fabricate information."
                ))],
            )
        ],
        config=gen_config,
    )

    if err:
        yield {"type": "error", "text": err}
        return

    final_text = _safe_text(resp) or "Reached search cap — unable to produce a complete answer."
    elapsed = round(time.time() - start_time, 1)

    mem_module.save_session(
        db_conn,
        question=question,
        answer=final_text[:2000],
        sources=all_sources[:20],
    )

    yield {
        "type": "answer",
        "text": final_text,
        "meta": {
            "sources": all_sources,
            "source_count": len(all_sources),
            "search_count": turn,
            "elapsed_seconds": elapsed,
            "turn_count": turn,
            "hit_cap": True,
        },
    }
