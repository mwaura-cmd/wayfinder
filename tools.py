"""
tools.py — The two tools the Wayfinder agent can call (§3, §7).

  1. web_search   — Query the web via Tavily.
  2. check_memory — Look up the agent's own past research sessions.

Tool errors are caught here and returned as plain-text descriptions the
model can reason about. Raw exceptions NEVER propagate to the model or user.

Both tools have exponential-backoff retry logic (§6):
  attempts: 3, delays: 1s → 2s → 4s, cap at MAX_RETRIES.

Guardrail note (§5): Neither tool writes to external services, sends messages,
or deletes anything. web_search reads the web. check_memory reads a local DB.
The only write action this agent performs is saving completed sessions to its
own memory DB (handled in agent.py after task completion).
"""
import time
import logging
from typing import Any

import httpx

import config
import memory as mem_module

log = logging.getLogger(__name__)


# ── Retry helper ───────────────────────────────────────────────────────────────

def _with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential back-off.
    Returns (result, None) on success or (None, error_message) on exhausted retries.
    """
    delay = config.RETRY_BASE_SECONDS
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs), None
        except httpx.TimeoutException:
            err = "Request timed out."
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                err = (
                    f"Rate limit hit (HTTP 429) on attempt {attempt}. "
                    "Tavily's free-tier limit may be reached — try again in ~1 minute."
                )
            elif status >= 500:
                err = f"Tavily server error (HTTP {status}) on attempt {attempt}."
            else:
                err = f"HTTP error {status}: {exc.response.text[:200]}"
                # Non-retriable client errors (4xx except 429) — bail immediately
                return None, err
        except Exception as exc:  # noqa: BLE001
            err = f"Unexpected error on attempt {attempt}: {type(exc).__name__}: {exc}"

        if attempt < config.MAX_RETRIES:
            log.warning("Tool call failed (%s). Retrying in %.0fs…", err, delay)
            time.sleep(delay)
            delay *= 2
        else:
            log.error("Tool call failed after %d attempts: %s", config.MAX_RETRIES, err)
    return None, err


# ── Tool 1: web_search ─────────────────────────────────────────────────────────

# Tavily search endpoint
_TAVILY_URL = "https://api.tavily.com/search"


def _raw_tavily_search(query: str) -> dict:
    """Make one Tavily search request (may raise — caller handles retries)."""
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            _TAVILY_URL,
            json={
                "api_key": config.TAVILY_API_KEY,
                "query": query,
                "max_results": config.TAVILY_MAX_RESULTS,
                "search_depth": "advanced",
                "include_answer": False,
            },
        )
        resp.raise_for_status()
        return resp.json()


def web_search(query: str) -> str:
    """
    Search the live web for `query` using Tavily.

    Returns a plain-text summary: one entry per result containing the
    page title, URL, and a relevant snippet (trimmed to
    SNIPPET_MAX_CHARS characters). This trimmed format keeps context
    consumption low across multi-turn sessions (§4).

    On any failure (timeout, rate-limit, API error) returns a descriptive
    error string the agent can reason about — never raises.
    """
    if not config.TAVILY_API_KEY:
        return (
            "ERROR: TAVILY_API_KEY is not set. "
            "Please add it to your environment variables."
        )

    result, err = _with_retry(_raw_tavily_search, query)

    if err:
        # Surface a specific message for rate-limit errors (§6)
        if "429" in err or "rate limit" in err.lower():
            return (
                "SEARCH FAILED — Tavily's free-tier rate limit was hit. "
                "Please wait ~1 minute before retrying this query."
            )
        return f"SEARCH FAILED — {err}"

    raw_results: list[dict] = result.get("results", [])
    if not raw_results:
        return f"No results found for query: '{query}'. The web returned nothing useful."

    lines = [f"Search results for: {query!r}\n"]
    for i, r in enumerate(raw_results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        # Prefer 'content' field; fall back to 'snippet'
        raw_snippet = r.get("content") or r.get("snippet") or ""
        snippet = raw_snippet[: config.SNIPPET_MAX_CHARS].strip()
        if len(raw_snippet) > config.SNIPPET_MAX_CHARS:
            snippet += "…"
        lines.append(f"[{i}] {title}\n    URL: {url}\n    Snippet: {snippet}\n")

    return "\n".join(lines)


# ── Tool 2: check_memory ───────────────────────────────────────────────────────

def check_memory(conn, keywords: list[str]) -> str:
    """
    Search this agent's own past research sessions for topics related to
    the given keywords.

    Call this BEFORE searching the web for any new query. If a relevant
    past session exists, use it as a starting point — it saves Tavily
    credits and gives you your own prior findings as context.

    `keywords` should be 2–5 meaningful topic words extracted from the
    user's question (not stop-words).

    Returns a plain-text summary of matching past sessions (question,
    answer excerpt, sources, date), or a message saying nothing was found.
    Memory lookup failures degrade gracefully — returns an empty-result
    message rather than raising.
    """
    matches = mem_module.lookup_memory(conn, keywords)

    if not matches:
        kw_str = ", ".join(f'"{k}"' for k in keywords)
        return f"No past research found matching keywords: {kw_str}."

    lines = [f"Found {len(matches)} past session(s) related to your query:\n"]
    for i, m in enumerate(matches, 1):
        answer_excerpt = m["answer"][:400].strip()
        if len(m["answer"]) > 400:
            answer_excerpt += "…"
        sources_str = ", ".join(m["sources"][:3]) or "none recorded"
        lines.append(
            f"[{i}] Date: {m['created_at'][:10]}\n"
            f"     Question: {m['question']}\n"
            f"     Answer excerpt: {answer_excerpt}\n"
            f"     Sources used: {sources_str}\n"
        )

    return "\n".join(lines)


# ── Tool schema (Gemini function-call format) ──────────────────────────────────

# These dicts are passed to the Gemini SDK as function declarations.
# Descriptions are written to be unambiguous (§3) — the model should
# never be confused about what a tool does or what arguments it takes.

TOOL_DECLARATIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the live web for current information on a specific topic or question. "
            "Use ONE focused query per call — do NOT combine multiple sub-questions into "
            "one query. For multi-part questions, call this tool once per sub-question. "
            "Returns a list of web sources with titles, URLs, and text snippets. "
            "If results are thin, contradictory, or seem stale, call again with "
            "different search terms rather than guessing from training data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A focused, specific search query. Phrase it as a web search "
                        "(short noun phrase or clear question). Example: "
                        "'latest Python 3.13 release date' not "
                        "'what is the release date of Python 3.13 and what are its features'."
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_memory",
        "description": (
            "Check this agent's own database of past completed research sessions "
            "for topics related to the current question. "
            "ALWAYS call this BEFORE performing any web search — it may save "
            "Tavily credits and give you your own prior findings as context. "
            "If memory contains a relevant past session, report it as a named "
            "narrative step ('I've researched something similar before…') and "
            "use it to inform your current research plan. "
            "Returns summaries of matching sessions (question, answer, sources, date) "
            "or a message that no related sessions were found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "2 to 5 meaningful topic words from the user's question "
                        "(skip common stop-words like 'what', 'the', 'is'). "
                        "Example for 'What caused the 2008 financial crisis?': "
                        "['2008', 'financial', 'crisis', 'causes']"
                    ),
                }
            },
            "required": ["keywords"],
        },
    },
]
