"""
memory.py — Persistent memory store for Wayfinder (§9).

Uses SQLite (Python stdlib — no install, no account, no service).
The agent APPENDS completed sessions only; it never edits or deletes past
entries autonomously. Manual clearing is a separate CLI operation.

Retention cap: MAX_MEMORY_SESSIONS. When exceeded, the oldest rows are
pruned automatically and the pruning is logged (never silent).
"""
import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

log = logging.getLogger(__name__)


# ── Schema ─────────────────────────────────────────────────────────────────────

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    sources     TEXT    NOT NULL,   -- JSON array of URLs
    created_at  TEXT    NOT NULL    -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions (created_at);
"""


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")  # concurrent read safety
    return conn


def init_db(path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Open (or create) the SQLite DB and return a connection.
    Degrades gracefully — if the file is corrupt or unreadable, returns None
    so the agent can run without memory rather than crashing.
    """
    db_path = path or config.DB_PATH
    try:
        conn = _connect(db_path)
        conn.executescript(_INIT_SQL)
        conn.commit()
        log.info("Memory DB ready at %s", db_path)
        return conn
    except sqlite3.DatabaseError as exc:
        log.error(
            "Memory DB unavailable (%s) — agent will run without memory: %s",
            db_path,
            exc,
        )
        return None


# ── Write ──────────────────────────────────────────────────────────────────────

def save_session(
    conn: sqlite3.Connection,
    question: str,
    answer: str,
    sources: list[str],
) -> None:
    """
    Append a completed research session to the DB.
    Prunes oldest entries if MAX_MEMORY_SESSIONS is exceeded.
    """
    if conn is None:
        return
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO sessions (question, answer, sources, created_at)"
            " VALUES (?, ?, ?, ?)",
            (question, answer, json.dumps(sources), timestamp),
        )
        conn.commit()
        _prune(conn)
    except sqlite3.DatabaseError as exc:
        log.error("Failed to save session: %s", exc)


def _prune(conn: sqlite3.Connection) -> None:
    """Delete oldest rows beyond MAX_MEMORY_SESSIONS cap."""
    try:
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        excess = count - config.MAX_MEMORY_SESSIONS
        if excess > 0:
            conn.execute(
                """
                DELETE FROM sessions WHERE id IN (
                    SELECT id FROM sessions ORDER BY created_at ASC LIMIT ?
                )
                """,
                (excess,),
            )
            conn.commit()
            log.info("Pruned %d oldest memory session(s) to stay under cap.", excess)
    except sqlite3.DatabaseError as exc:
        log.error("Memory pruning failed: %s", exc)


# ── Read ───────────────────────────────────────────────────────────────────────

def lookup_memory(
    conn: sqlite3.Connection,
    keywords: list[str],
    limit: int = config.MEMORY_LOOKUP_LIMIT,
) -> list[dict]:
    """
    Simple keyword match against past questions (no embeddings needed at
    hobby scale). Returns up to `limit` most-recent matching sessions.

    Each returned dict has: question, answer, sources (list), created_at.
    """
    if conn is None or not keywords:
        return []
    try:
        # Build a LIKE clause for each keyword — OR across keywords so a
        # partial overlap still surfaces useful context.
        clauses = " OR ".join(
            ["LOWER(question) LIKE ?"] * len(keywords)
        )
        params = [f"%{kw.lower()}%" for kw in keywords] + [limit]
        rows = conn.execute(
            f"""
            SELECT question, answer, sources, created_at
            FROM   sessions
            WHERE  {clauses}
            ORDER  BY created_at DESC
            LIMIT  ?
            """,
            params,
        ).fetchall()
        return [
            {
                "question": r["question"],
                "answer": r["answer"],
                "sources": json.loads(r["sources"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    except sqlite3.DatabaseError as exc:
        log.error("Memory lookup failed: %s", exc)
        return []
