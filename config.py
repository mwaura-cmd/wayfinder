"""
config.py — Central configuration for Wayfinder Research Agent.
All tunables live here. Secrets are read from environment variables,
with automatic fallback to a local `.env` file (stdlib only, no dotenv).
"""
import os
from pathlib import Path

# ── Auto-load local `.env` file (if present — local dev only) ─────────────────
# On Render, secrets are set as environment variables in the dashboard.
# The .env file is gitignored and only used locally.
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip()
            if _key and _val and _key not in os.environ:
                os.environ[_key] = _val

# ── LLM — OpenRouter ──────────────────────────────────────────────────────────
# Sign up at https://openrouter.ai and grab a free API key.
# Set OPENROUTER_API_KEY in your .env (local) or Render dashboard (prod).
# Browse models at https://openrouter.ai/models — free ones are marked :free
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.environ.get(
    "OPENROUTER_MODEL",
    "anthropic/claude-sonnet-4-5",          # best quality on free tier
)
OPENROUTER_FALLBACK_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"

# ── Search ────────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS: int = 5
SNIPPET_MAX_CHARS: int = 600

# ── Agent loop ────────────────────────────────────────────────────────────────
MAX_TURNS: int = 8
MAX_RETRIES: int = 3
RETRY_BASE_SECONDS: float = 1.0

# ── Persistent memory ─────────────────────────────────────────────────────────
DB_PATH: Path = Path(__file__).parent / "wayfinder_memory.db"
MAX_MEMORY_SESSIONS: int = int(os.environ.get("MAX_MEMORY_SESSIONS", "200"))
MEMORY_LOOKUP_LIMIT: int = 3

# ── Auth ──────────────────────────────────────────────────────────────────────
# WAYFINDER_API_KEY guards the /research, /stream, and /history endpoints.
# Set this env var in Render dashboard (and in your local .env for dev).
# If not set, the server starts but logs a prominent warning — useful for
# first-run / local dev without auth. Never leave unset in production.
WAYFINDER_API_KEY: str = os.environ.get("WAYFINDER_API_KEY", "")

# ── Server ────────────────────────────────────────────────────────────────────
# HOST: 0.0.0.0 accepts external connections (required for Render).
# PORT: Render injects $PORT automatically; fallback to 8000 for local dev.
HOST: str = "0.0.0.0"
PORT: int = int(os.environ.get("PORT", "8000"))
