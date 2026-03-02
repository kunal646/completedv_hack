"""
SpecFlow — Persistent store backed by SQLite.
Handles raw intake queue + classified work items.
"""

import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path(__file__).parent / "specflow.db"

# Status lifecycle per lane
LANE_STATUSES = {
    "issue":     ["new", "confirmed", "in_progress", "resolved"],
    "feature":   ["new", "evaluated", "backlog", "shipped"],
    "idea":      ["new", "interesting", "parked", "promoted"],
    "marketing": ["new", "saved", "used"],
    "unclassified": ["new"],
}

LANE_META = {
    "issue":      {"emoji": "🐛", "label": "Issues",    "color": "#f85149"},
    "feature":    {"emoji": "✨", "label": "Features",  "color": "#58a6ff"},
    "idea":       {"emoji": "💡", "label": "Ideas",     "color": "#d29922"},
    "marketing":  {"emoji": "📣", "label": "Marketing", "color": "#3fb950"},
    "unclassified": {"emoji": "📥", "label": "Inbox",   "color": "#8b949e"},
}

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source       TEXT    NOT NULL,
                content      TEXT    NOT NULL,
                channel      TEXT,
                author       TEXT,
                -- intake status
                status       TEXT    NOT NULL DEFAULT 'pending',
                msg_hash     TEXT    UNIQUE,
                -- classification
                lane         TEXT    NOT NULL DEFAULT 'unclassified',
                title        TEXT,
                summary      TEXT,
                priority     TEXT    NOT NULL DEFAULT 'medium',
                -- item lifecycle
                item_status  TEXT    NOT NULL DEFAULT 'new',
                -- issue card + plan (stored as markdown)
                issue_card   TEXT,
                tech_plan    TEXT,
                -- team notes
                notes        TEXT    DEFAULT '',
                created_at   DATETIME DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS config (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS context_files (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                file_type  TEXT    DEFAULT 'text',
                created_at DATETIME DEFAULT (datetime('now'))
            );
        """)
        # Migrate existing DB — add columns if missing
        existing = {row[1] for row in c.execute("PRAGMA table_info(messages)").fetchall()}
        migrations = {
            "lane":        "TEXT NOT NULL DEFAULT 'unclassified'",
            "title":       "TEXT",
            "summary":     "TEXT",
            "priority":    "TEXT NOT NULL DEFAULT 'medium'",
            "item_status": "TEXT NOT NULL DEFAULT 'new'",
            "issue_card":  "TEXT",
            "tech_plan":   "TEXT",
            "lane_output": "TEXT",   # unified output for feature/idea/marketing
            "needs_review":"INTEGER DEFAULT 0",
            "notes":       "TEXT DEFAULT ''",
        }
        for col, typedef in migrations.items():
            if col not in existing:
                c.execute(f"ALTER TABLE messages ADD COLUMN {col} {typedef}")


# ── Intake ─────────────────────────────────────────────────────────────────────

def add_message(source: str, content: str, channel: str = None, author: str = None) -> bool:
    """Add raw message to intake queue. Returns False if duplicate."""
    if not content.strip():
        return False
    msg_hash = hashlib.md5(f"{source}:{content.strip()}".encode()).hexdigest()
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO messages (source, content, channel, author, msg_hash) VALUES (?,?,?,?,?)",
                (source, content.strip(), channel, author, msg_hash),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_messages(status: str = "pending") -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE status=? ORDER BY created_at DESC", (status,)
        ).fetchall()
    return [dict(r) for r in rows]


def set_status(msg_id: int, status: str):
    with _conn() as c:
        c.execute("UPDATE messages SET status=? WHERE id=?", (status, msg_id))


# ── Items (classified) ─────────────────────────────────────────────────────────

def classify_item(msg_id: int, lane: str, title: str, summary: str, priority: str):
    """Set classification on an item and move it out of pending intake."""
    item_status = LANE_STATUSES.get(lane, ["new"])[0]
    with _conn() as c:
        c.execute(
            """UPDATE messages
               SET lane=?, title=?, summary=?, priority=?, item_status=?, status='classified'
               WHERE id=?""",
            (lane, title, summary, priority, item_status, msg_id),
        )


def save_issue_card(msg_id: int, issue_card: str):
    with _conn() as c:
        c.execute("UPDATE messages SET issue_card=? WHERE id=?", (issue_card, msg_id))


def save_tech_plan(msg_id: int, tech_plan: str):
    with _conn() as c:
        c.execute("UPDATE messages SET tech_plan=?, item_status='reviewed' WHERE id=?", (tech_plan, msg_id))


def save_lane_output(msg_id: int, output: str):
    with _conn() as c:
        c.execute("UPDATE messages SET lane_output=?, item_status='reviewed' WHERE id=?", (output, msg_id))


def save_auto_process_result(msg_id: int, lane: str, title: str, summary: str,
                              priority: str, output: str, needs_review: bool):
    """Save the full result from auto_process() in one call."""
    item_status = LANE_STATUSES.get(lane, ["new"])[0]
    if not needs_review and output:
        item_status = "reviewed"
    with _conn() as c:
        c.execute("""
            UPDATE messages
            SET lane=?, title=?, summary=?, priority=?,
                lane_output=?, needs_review=?,
                item_status=?, status='classified'
            WHERE id=?
        """, (lane, title, summary, priority, output,
              1 if needs_review else 0, item_status, msg_id))


def update_item(msg_id: int, **kwargs):
    """Update any item fields by keyword."""
    allowed = {"lane", "title", "summary", "priority", "item_status", "notes", "issue_card", "tech_plan"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE messages SET {set_clause} WHERE id=?", (*fields.values(), msg_id))


def get_items(lane: str = None, item_status: str = None) -> list[dict]:
    """Get classified items, optionally filtered by lane and/or item_status."""
    query = "SELECT * FROM messages WHERE status='classified'"
    params = []
    if lane:
        query += " AND lane=?"
        params.append(lane)
    if item_status:
        query += " AND item_status=?"
        params.append(item_status)
    query += " ORDER BY CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, created_at DESC"
    with _conn() as c:
        rows = c.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_item(msg_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM messages WHERE id=?", (msg_id,)).fetchone()
    return dict(row) if row else None


def delete_item(msg_id: int):
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE id=?", (msg_id,))


def board_counts() -> dict:
    """Returns count per lane for the board header."""
    with _conn() as c:
        rows = c.execute(
            "SELECT lane, COUNT(*) as cnt FROM messages WHERE status='classified' GROUP BY lane"
        ).fetchall()
    result = {lane: 0 for lane in LANE_META}
    for r in rows:
        result[r["lane"]] = r["cnt"]
    # Inbox = pending
    row = c.execute("SELECT COUNT(*) as cnt FROM messages WHERE status='pending'").fetchone()
    result["pending"] = row["cnt"] if row else 0
    return result


# ── Context Knowledge Base ────────────────────────────────────────────────────

def add_context_file(filename: str, content: str, file_type: str = "text") -> int:
    """Store a file in the knowledge base. Caps at 60k chars to stay within token limits."""
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO context_files (filename, content, file_type) VALUES (?,?,?)",
            (filename, content[:60000], file_type),
        )
        return cur.lastrowid


def get_context_files() -> list[dict]:
    """Return metadata for all context files (no content, for listing)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id, filename, file_type, LENGTH(content) as char_count, created_at "
            "FROM context_files ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_context_file(file_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM context_files WHERE id=?", (file_id,)).fetchone()
    return dict(row) if row else None


def delete_context_file(file_id: int):
    with _conn() as c:
        c.execute("DELETE FROM context_files WHERE id=?", (file_id,))


def build_context_str(max_chars: int = 14000) -> str:
    """
    Assembles product brief + uploaded files into a grounded context block.
    Injected as a prefix to every agent system prompt.
    Returns empty string if no context is configured.
    """
    parts = []

    brief = get_cfg("product_brief", "").strip()
    if brief:
        parts.append(f"## Product / Project Brief\n{brief}")

    files = get_context_files()
    total = len(brief)
    for meta in files:
        if total >= max_chars:
            break
        row = get_context_file(meta["id"])
        if not row:
            continue
        budget = max_chars - total
        snippet = row["content"][:budget]
        parts.append(f"## File: {row['filename']}\n```\n{snippet}\n```")
        total += len(snippet)

    if not parts:
        return ""

    header = (
        "=== PRODUCT & CODEBASE CONTEXT ===\n"
        "Use the information below to ground your output in the actual product and codebase. "
        "Reference specific files, functions, or product decisions where relevant.\n\n"
    )
    footer = "\n=== END CONTEXT ==="
    return header + "\n\n".join(parts) + footer


# ── Config ────────────────────────────────────────────────────────────────────

def get_cfg(key: str, default: str = "") -> str:
    with _conn() as c:
        row = c.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_cfg(key: str, value: str):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO config (key,value) VALUES (?,?)", (key, value))


# Init on import
init_db()
