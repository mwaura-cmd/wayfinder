"""
memory.py — Persistent conversational memory store for Wayfinder.

Uses SQLite (Python stdlib). Manages threads and multi-turn messages,
automatic startup migration from legacy sessions schema, and thread-level
retention management.
"""
import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import config

log = logging.getLogger(__name__)

# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS threads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,   -- first question's text, truncated
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id     INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    question      TEXT    NOT NULL,
    answer        TEXT    NOT NULL,
    sources       TEXT    NOT NULL,   -- JSON array, same as before
    model_used    TEXT,               -- the actual model OpenRouter routed to
    level         TEXT    NOT NULL DEFAULT 'standard',  -- 'standard' | 'extended'
    feedback      TEXT,               -- 'up' | 'down' | NULL
    feedback_note TEXT,               -- optional user comment, NULL if none
    created_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages (thread_id);
"""


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")  # concurrent read safety
    conn.execute("PRAGMA foreign_keys=ON;")   # enable cascade deletes
    return conn


def _run_migration(conn: sqlite3.Connection) -> None:
    """
    One-time startup migration:
    If the legacy 'sessions' table exists (and not yet renamed to 'sessions_deprecated'),
    migrate each row into threads + messages, then rename sessions -> sessions_deprecated.
    """
    try:
        cursor = conn.cursor()
        # Check if legacy sessions table exists
        table_exists = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions';"
        ).fetchone()

        if not table_exists:
            return

        log.info("Legacy 'sessions' table detected. Starting one-time migration to threads/messages schema...")
        legacy_rows = cursor.execute(
            "SELECT id, question, answer, sources, created_at FROM sessions ORDER BY id ASC;"
        ).fetchall()

        migrated_count = 0
        for row in legacy_rows:
            q = row["question"] or "Untitled Research"
            title = q[:60].strip()
            created_at = row["created_at"] or datetime.now(timezone.utc).isoformat()

            # Create thread
            cursor.execute(
                "INSERT INTO threads (title, created_at, updated_at) VALUES (?, ?, ?);",
                (title, created_at, created_at)
            )
            thread_id = cursor.lastrowid

            # Parse / normalize sources
            raw_sources = row["sources"]
            if not raw_sources:
                sources_json = "[]"
            elif isinstance(raw_sources, str):
                sources_json = raw_sources
            else:
                sources_json = json.dumps(raw_sources)

            # Create message row
            cursor.execute(
                """
                INSERT INTO messages (
                    thread_id, question, answer, sources, model_used,
                    level, feedback, feedback_note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    thread_id,
                    q,
                    row["answer"] or "",
                    sources_json,
                    None,           # model_used not tracked in legacy rows
                    "standard",     # default level
                    None,           # feedback
                    None,           # feedback_note
                    created_at
                )
            )
            migrated_count += 1

        # Rename old sessions table to preserve legacy data safely
        cursor.execute("ALTER TABLE sessions RENAME TO sessions_deprecated;")
        conn.commit()
        log.info(
            "Migration successful: %d row(s) migrated into threads/messages. Legacy table renamed to sessions_deprecated.",
            migrated_count
        )

    except sqlite3.DatabaseError as exc:
        log.error("Migration error: %s", exc)
        conn.rollback()


def init_db(path: Optional[Path] = None) -> Optional[sqlite3.Connection]:
    """
    Open (or create) the SQLite DB and return a connection.
    Initializes tables and executes any required migration.
    Degrades gracefully — if unreadable, returns None.
    """
    db_path = path or config.DB_PATH
    try:
        conn = _connect(db_path)
        # Create schema if not exists
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

        # Run migration if legacy sessions table exists
        _run_migration(conn)

        log.info("Memory DB ready at %s", db_path)
        return conn
    except sqlite3.DatabaseError as exc:
        log.error(
            "Memory DB unavailable (%s) — agent will run without memory: %s",
            db_path,
            exc,
        )
        return None


# ── Thread Management ─────────────────────────────────────────────────────────

def create_thread(conn: sqlite3.Connection, title: str) -> int:
    """Create a new research thread and return its ID."""
    now = datetime.now(timezone.utc).isoformat()
    clean_title = (title or "Untitled Research")[:60].strip()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO threads (title, created_at, updated_at) VALUES (?, ?, ?);",
        (clean_title, now, now)
    )
    conn.commit()
    return cursor.lastrowid


def get_or_create_thread(conn: sqlite3.Connection, thread_id: Optional[int], first_question: str) -> int:
    """Get an existing thread ID if valid, otherwise create a new thread."""
    if thread_id is not None:
        try:
            row = conn.execute("SELECT id FROM threads WHERE id = ?;", (int(thread_id),)).fetchone()
            if row:
                return int(row["id"])
        except Exception:
            pass
    return create_thread(conn, first_question)


def save_message(
    conn: sqlite3.Connection,
    thread_id: int,
    question: str,
    answer: str,
    sources: List[Any],
    model_used: Optional[str] = None,
    level: str = "standard",
) -> int:
    """
    Append a completed message to an existing thread.
    Updates the thread's updated_at timestamp and prunes old threads if cap is exceeded.
    """
    if conn is None:
        return 0
    try:
        now = datetime.now(timezone.utc).isoformat()
        formatted_sources = json.dumps(sources) if not isinstance(sources, str) else sources

        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages (
                thread_id, question, answer, sources, model_used,
                level, feedback, feedback_note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                thread_id,
                question,
                answer,
                formatted_sources,
                model_used,
                level,
                None,
                None,
                now
            )
        )
        msg_id = cursor.lastrowid

        # Update thread's updated_at
        cursor.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?;",
            (now, thread_id)
        )
        conn.commit()

        _prune(conn)
        return msg_id
    except sqlite3.DatabaseError as exc:
        log.error("Failed to save message: %s", exc)
        return 0


def save_session(
    conn: sqlite3.Connection,
    question: str,
    answer: str,
    sources: List[Any],
    model_used: Optional[str] = None,
    level: str = "standard",
    thread_id: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Backwards-compatible wrapper.
    Creates or reuses thread_id and saves the message.
    Returns (thread_id, message_id).
    """
    if conn is None:
        return (0, 0)
    tid = get_or_create_thread(conn, thread_id, question)
    mid = save_message(conn, tid, question, answer, sources, model_used, level)
    return (tid, mid)


def update_feedback(
    conn: sqlite3.Connection,
    message_id: int,
    feedback: Optional[str],
    feedback_note: Optional[str] = None,
) -> bool:
    """Update feedback ('up' | 'down' | None) and feedback_note on a message."""
    if conn is None:
        return False
    try:
        conn.execute(
            """
            UPDATE messages
            SET feedback = ?, feedback_note = ?
            WHERE id = ?;
            """,
            (feedback, feedback_note, message_id)
        )
        conn.commit()
        return True
    except sqlite3.DatabaseError as exc:
        log.error("Failed to update feedback for message %s: %s", message_id, exc)
        return False


def _prune(conn: sqlite3.Connection) -> None:
    """
    Retention cap logic:
    Caps total THREADS at config.MAX_MEMORY_SESSIONS.
    Pruning deletes oldest threads (and cascades to messages).
    """
    if conn is None:
        return
    try:
        count = conn.execute("SELECT COUNT(*) FROM threads;").fetchone()[0]
        excess = count - config.MAX_MEMORY_SESSIONS
        if excess > 0:
            cursor = conn.cursor()
            # Find oldest thread IDs
            oldest_threads = cursor.execute(
                "SELECT id FROM threads ORDER BY updated_at ASC LIMIT ?;",
                (excess,)
            ).fetchall()
            thread_ids = [r["id"] for r in oldest_threads]

            if thread_ids:
                placeholders = ",".join("?" * len(thread_ids))
                # Delete messages first (or via cascade if foreign keys enabled)
                cursor.execute(
                    f"DELETE FROM messages WHERE thread_id IN ({placeholders});",
                    thread_ids
                )
                cursor.execute(
                    f"DELETE FROM threads WHERE id IN ({placeholders});",
                    thread_ids
                )
                conn.commit()
                log.info("Pruned %d oldest thread(s) to stay under cap (%d).", excess, config.MAX_MEMORY_SESSIONS)
    except sqlite3.DatabaseError as exc:
        log.error("Thread memory pruning failed: %s", exc)


# ── Read & Retrieval ──────────────────────────────────────────────────────────

def get_threads(conn: sqlite3.Connection, limit: int = 30) -> List[Dict[str, Any]]:
    """Return most recently updated threads with message counts and latest preview."""
    if conn is None:
        return []
    try:
        rows = conn.execute(
            """
            SELECT 
                t.id,
                t.title,
                t.created_at,
                t.updated_at,
                COUNT(m.id) AS message_count,
                MAX(m.created_at) AS last_message_at
            FROM threads t
            LEFT JOIN messages m ON t.id = m.thread_id
            GROUP BY t.id
            ORDER BY t.updated_at DESC
            LIMIT ?;
            """,
            (limit,)
        ).fetchall()

        threads = []
        for r in rows:
            threads.append({
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "message_count": r["message_count"],
                "last_message_at": r["last_message_at"],
            })
        return threads
    except sqlite3.DatabaseError as exc:
        log.error("Failed to fetch threads: %s", exc)
        return []


def get_thread(conn: sqlite3.Connection, thread_id: int) -> Optional[Dict[str, Any]]:
    """Return a thread and all its ordered messages."""
    if conn is None:
        return None
    try:
        t_row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM threads WHERE id = ?;",
            (thread_id,)
        ).fetchone()

        if not t_row:
            return None

        m_rows = conn.execute(
            """
            SELECT 
                id, thread_id, question, answer, sources,
                model_used, level, feedback, feedback_note, created_at
            FROM messages
            WHERE thread_id = ?
            ORDER BY id ASC;
            """,
            (thread_id,)
        ).fetchall()

        messages = []
        for m in m_rows:
            try:
                srcs = json.loads(m["sources"]) if m["sources"] else []
            except Exception:
                srcs = []
            messages.append({
                "id": m["id"],
                "thread_id": m["thread_id"],
                "question": m["question"],
                "answer": m["answer"],
                "sources": srcs,
                "model_used": m["model_used"],
                "level": m["level"],
                "feedback": m["feedback"],
                "feedback_note": m["feedback_note"],
                "created_at": m["created_at"],
            })

        return {
            "id": t_row["id"],
            "title": t_row["title"],
            "created_at": t_row["created_at"],
            "updated_at": t_row["updated_at"],
            "messages": messages
        }
    except sqlite3.DatabaseError as exc:
        log.error("Failed to fetch thread %s: %s", thread_id, exc)
        return None


def get_thread_messages(conn: sqlite3.Connection, thread_id: int) -> List[Dict[str, Any]]:
    """Return chronological list of Q&A exchanges for a thread."""
    thread = get_thread(conn, thread_id)
    return thread.get("messages", []) if thread else []


def lookup_memory(
    conn: sqlite3.Connection,
    keywords: List[str],
    limit: int = config.MEMORY_LOOKUP_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Search past messages across threads matching any of the given keywords.
    Used for cross-thread recall on initial queries.
    """
    if conn is None or not keywords:
        return []
    try:
        clauses = " OR ".join(["LOWER(m.question) LIKE ?" for _ in keywords])
        params = [f"%{kw.lower()}%" for kw in keywords] + [limit]
        rows = conn.execute(
            f"""
            SELECT 
                m.id, m.thread_id, m.question, m.answer,
                m.sources, m.model_used, m.level, m.created_at
            FROM messages m
            WHERE {clauses}
            ORDER BY m.created_at DESC
            LIMIT ?;
            """,
            params,
        ).fetchall()

        results = []
        for r in rows:
            try:
                srcs = json.loads(r["sources"]) if r["sources"] else []
            except Exception:
                srcs = []
            results.append({
                "id": r["id"],
                "thread_id": r["thread_id"],
                "question": r["question"],
                "answer": r["answer"],
                "sources": srcs,
                "model_used": r["model_used"],
                "level": r["level"],
                "created_at": r["created_at"],
            })
        return results
    except sqlite3.DatabaseError as exc:
        log.error("Memory lookup failed: %s", exc)
        return []
