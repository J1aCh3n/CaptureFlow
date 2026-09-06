import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from .models import Capture, Conversation, DEFAULT_WORKSPACE_ID


class CaptureStore:
    """Small SQLite repository for the deterministic Phase 1 vertical slice."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS captures (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    source_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO workspaces (id, created_at) VALUES (?, ?)",
                (DEFAULT_WORKSPACE_ID, "1970-01-01T00:00:00+00:00"),
            )

    def save_conversation(self, conversation: Conversation) -> Conversation:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO workspaces (id, created_at) VALUES (?, ?)",
                (conversation.workspace_id, conversation.created_at.isoformat()),
            )
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO conversations (id, workspace_id, title, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    conversation.id,
                    conversation.workspace_id,
                    conversation.title,
                    conversation.created_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_conversation(conversation.id)
            if existing != conversation:
                raise ValueError("conversation ID already exists with different content")
        return self.get_conversation(conversation.id)

    def get_conversation(self, conversation_id: str) -> Conversation:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, workspace_id, title, created_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"conversation not found: {conversation_id}")
        return Conversation(**dict(row))

    def list_conversations(self) -> list[Conversation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, workspace_id, title, created_at
                FROM conversations
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [Conversation(**dict(row)) for row in rows]

    def save_capture(self, capture: Capture) -> Capture:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO captures (id, conversation_id, source_text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    capture.id,
                    capture.conversation_id,
                    capture.source_text,
                    capture.created_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_capture(capture.id)
            if existing != capture:
                raise ValueError("capture ID already exists with different content")
        return self.get_capture(capture.id)

    def list_captures(self, conversation_id: str) -> list[Capture]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, source_text, created_at
                FROM captures
                WHERE conversation_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (conversation_id,),
            ).fetchall()
        return [Capture(**dict(row)) for row in rows]

    def get_capture(self, capture_id: str) -> Capture:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, conversation_id, source_text, created_at
                FROM captures
                WHERE id = ?
                """,
                (capture_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"capture not found: {capture_id}")
        return Capture(**dict(row))

    def delete_capture(self, capture_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
        return cursor.rowcount == 1

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
