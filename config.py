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

# ── Firebase ──────────────────────────────────────────────────────────────────
# Place your firebase-adminsdk.json inside the root directory, or
# paste the raw JSON string into the FIREBASE_CREDENTIALS_JSON env variable.
FIREBASE_PROJECT_ID: str = os.environ.get("FIREBASE_PROJECT_ID", "wayfinder-b98c7")
FIREBASE_CREDENTIALS_PATH: str = os.environ.get(
    "FIREBASE_CREDENTIALS_PATH",
    str(Path(__file__).parent / "firebase-adminsdk.json")
)
FIREBASE_CREDENTIALS_JSON: str = os.environ.get("FIREBASE_CREDENTIALS_JSON", "")

# ── LLM Provider Selection & Settings ─────────────────────────────────────────
# LLM_PROVIDER controls which provider is active at runtime ("groq" | "openrouter").
# Both providers' keys and models remain defined and switchable.
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "groq")

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.environ.get(
    "OPENROUTER_MODEL",
    "openrouter/free",          # Auto-routes to available free models
)

# ── Search ────────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS: int = 5
SNIPPET_MAX_CHARS: int = 600

# ── Agent loop ────────────────────────────────────────────────────────────────
MAX_TURNS: int = 8
MAX_RETRIES: int = 3
RETRY_BASE_SECONDS: float = 1.0

# ── Research Levels ───────────────────────────────────────────────────────────
RESEARCH_LEVELS = {
    "standard": {"max_turns": 4, "label": "Standard"},
    "extended": {"max_turns": 12, "label": "Extended"},
}
DEFAULT_RESEARCH_LEVEL: str = "standard"


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
