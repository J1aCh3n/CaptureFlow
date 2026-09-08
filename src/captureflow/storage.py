import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
import json
from pathlib import Path

from .memory import MAX_RETRIEVED_MEMORIES, lexical_match_score
from .models import (
    Capture,
    CaptureRun,
    Conversation,
    DEFAULT_WORKSPACE_ID,
    MemoryEvent,
    MemoryItem,
    PendingMemoryReview,
    TaskItem,
    WorkflowCheckpoint,
    utc_now,
)


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

                CREATE TABLE IF NOT EXISTS capture_runs (
                    id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL REFERENCES captures(id)
                        ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_items (
                    id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    run_id TEXT NOT NULL REFERENCES capture_runs(id) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    source_quote TEXT NOT NULL,
                    due_expression TEXT,
                    resolved_due_date TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    source_quote TEXT NOT NULL,
                    write_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_reviews (
                    id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    review_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );

                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT REFERENCES memory_items(id),
                    action TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    run_id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                    phase TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete one conversation and all state derived from its captures."""

        with self._connect() as connection:
            memory_rows = connection.execute(
                """
                SELECT memory_items.id
                FROM memory_items
                JOIN captures ON captures.id = memory_items.source_capture_id
                WHERE captures.conversation_id = ?
                """,
                (conversation_id,),
            ).fetchall()
            memory_ids = [row["id"] for row in memory_rows]
            if memory_ids:
                placeholders = ", ".join("?" for _ in memory_ids)
                connection.execute(
                    f"DELETE FROM memory_events WHERE memory_id IN ({placeholders})",
                    memory_ids,
                )
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        return cursor.rowcount == 1

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

    def save_run(self, run: CaptureRun) -> CaptureRun:
        serialized = run.model_dump(mode="json")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO capture_runs (id, capture_id, status, result_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.capture_id,
                    run.status,
                    json.dumps(serialized, sort_keys=True),
                    run.created_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_run(run.id)
            if existing != run:
                raise ValueError("run ID already exists with different content")
        return self.get_run(run.id)

    def get_run(self, run_id: str) -> CaptureRun:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM capture_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"run not found: {run_id}")
        return _decode_run(row["result_json"])

    def latest_run_for_capture(self, capture_id: str) -> CaptureRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM capture_runs
                WHERE capture_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (capture_id,),
            ).fetchone()
        return None if row is None else _decode_run(row["result_json"])

    def list_runs_for_capture(self, capture_id: str) -> list[CaptureRun]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT result_json FROM capture_runs
                WHERE capture_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (capture_id,),
            ).fetchall()
        return [_decode_run(row["result_json"]) for row in rows]

    def save_task(self, task: TaskItem) -> TaskItem:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO task_items
                (id, capture_id, run_id, text, source_quote, due_expression, resolved_due_date,
                 status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.capture_id,
                    task.run_id,
                    task.text,
                    task.source_quote,
                    task.due_expression,
                    None if task.resolved_due_date is None else task.resolved_due_date.isoformat(),
                    task.status,
                    task.created_at.isoformat(),
                    task.updated_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_task(task.id)
            if existing != task:
                raise ValueError("task ID already exists with different content")
        return self.get_task(task.id)

    def get_task(self, task_id: str) -> TaskItem:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM task_items WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"task not found: {task_id}")
        return TaskItem(**dict(row))

    def list_tasks_for_capture(self, capture_id: str) -> list[TaskItem]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM task_items WHERE capture_id = ? ORDER BY created_at DESC, id DESC",
                (capture_id,),
            ).fetchall()
        return [TaskItem(**dict(row)) for row in rows]

    def save_memory(
        self,
        item: MemoryItem,
        event_action: str,
        event_details: dict | None = None,
    ) -> MemoryItem:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO memory_items
                (id, workspace_id, memory_type, content, source_capture_id, source_quote,
                 write_mode, status, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.workspace_id,
                    item.memory_type,
                    item.content,
                    item.source_capture_id,
                    item.source_quote,
                    item.write_mode,
                    item.status,
                    item.confidence,
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                self._record_event(connection, item.id, event_action, event_details or {})
        if not inserted:
            existing = self.get_memory(item.id)
            if _without_timestamps(existing) != _without_timestamps(item):
                raise ValueError("memory ID already exists with different content")
        return self.get_memory(item.id)

    def get_memory(self, memory_id: str) -> MemoryItem:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memory_items WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(f"memory not found: {memory_id}")
        return MemoryItem(**dict(row))

    def list_memories(
        self,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        statuses: tuple[str, ...] = ("active",),
    ) -> list[MemoryItem]:
        placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM memory_items
                WHERE workspace_id = ? AND status IN ({placeholders})
                ORDER BY updated_at DESC, id DESC
                """,
                (workspace_id, *statuses),
            ).fetchall()
        return [MemoryItem(**dict(row)) for row in rows]

    def search_memories(
        self,
        query: str,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        limit: int = MAX_RETRIEVED_MEMORIES,
    ) -> list[MemoryItem]:
        if not 1 <= limit <= MAX_RETRIEVED_MEMORIES:
            raise ValueError(f"memory search limit must be between 1 and {MAX_RETRIEVED_MEMORIES}")
        ranked = [
            (lexical_match_score(query, item), item)
            for item in self.list_memories(workspace_id=workspace_id, statuses=("active",))
        ]
        return [item for score, item in sorted(ranked, key=lambda pair: pair[0], reverse=True) if score][:limit]

    def set_memory_status(
        self,
        memory_id: str,
        status: str,
        action: str,
        event_details: dict | None = None,
    ) -> MemoryItem:
        if status not in {"active", "superseded", "deleted"}:
            raise ValueError("invalid memory status")
        item = self.get_memory(memory_id)
        updated = item.model_copy(update={"status": status, "updated_at": utc_now()})
        with self._connect() as connection:
            connection.execute(
                "UPDATE memory_items SET status = ?, updated_at = ? WHERE id = ?",
                (updated.status, updated.updated_at.isoformat(), updated.id),
            )
            details = {"previous_status": item.status, **(event_details or {})}
            self._record_event(connection, memory_id, action, details)
        return updated

    def save_pending_review(self, review: PendingMemoryReview) -> PendingMemoryReview:
        payload = review.model_dump(mode="json")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO memory_reviews
                (id, capture_id, review_json, status, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    review.id,
                    review.capture_id,
                    json.dumps(payload, sort_keys=True),
                    review.status,
                    review.created_at.isoformat(),
                    None if review.resolved_at is None else review.resolved_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_review(review.id)
            if _without_timestamps(existing) != _without_timestamps(review):
                raise ValueError("memory review ID already exists with different content")
        return self.get_review(review.id)

    def get_review(self, review_id: str) -> PendingMemoryReview:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT review_json FROM memory_reviews WHERE id = ?", (review_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"memory review not found: {review_id}")
        return PendingMemoryReview.model_validate_json(row["review_json"])

    def list_pending_reviews(self) -> list[PendingMemoryReview]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT review_json FROM memory_reviews WHERE status = 'pending' ORDER BY created_at DESC"
            ).fetchall()
        return [PendingMemoryReview.model_validate_json(row["review_json"]) for row in rows]

    def resolve_review(
        self, review_id: str, status: str
    ) -> PendingMemoryReview:
        if status not in {"approved", "rejected"}:
            raise ValueError("invalid memory review status")
        review = self.get_review(review_id)
        if review.status != "pending":
            if review.status == status:
                return review
            raise ValueError("a resolved memory review cannot change status")
        updated = review.model_copy(update={"status": status, "resolved_at": utc_now()})
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE memory_reviews
                SET review_json = ?, status = ?, resolved_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(updated.model_dump(mode="json"), sort_keys=True),
                    updated.status,
                    updated.resolved_at.isoformat(),
                    updated.id,
                ),
            )
        return updated

    def list_memory_events(self, memory_id: str) -> list[MemoryEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_events WHERE memory_id = ? ORDER BY id DESC", (memory_id,)
            ).fetchall()
        return [
            MemoryEvent(
                id=row["id"],
                memory_id=row["memory_id"],
                action=row["action"],
                details=json.loads(row["details_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> WorkflowCheckpoint:
        payload = checkpoint.model_dump(mode="json")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO workflow_checkpoints
                (run_id, capture_id, phase, state, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.run_id,
                    checkpoint.capture_id,
                    checkpoint.phase,
                    checkpoint.state,
                    json.dumps(payload, sort_keys=True),
                    checkpoint.updated_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_checkpoint(checkpoint.run_id)
            if existing != checkpoint:
                raise ValueError("checkpoint run ID already exists with different content")
        return self.get_checkpoint(checkpoint.run_id)

    def get_checkpoint(self, run_id: str) -> WorkflowCheckpoint:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM workflow_checkpoints WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"checkpoint not found: {run_id}")
        return WorkflowCheckpoint.model_validate_json(row["payload_json"])

    def commit_checkpoint(self, run_id: str, payload: dict) -> WorkflowCheckpoint:
        """Advance one prepared checkpoint to committed without allowing rewrites."""

        existing = self.get_checkpoint(run_id)
        if existing.state == "committed":
            if existing.payload != payload:
                raise ValueError("committed checkpoint payload cannot change")
            return existing
        committed = existing.model_copy(
            update={"state": "committed", "payload": payload, "updated_at": utc_now()}
        )
        serialized = committed.model_dump(mode="json")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE workflow_checkpoints
                SET state = ?, payload_json = ?, updated_at = ?
                WHERE run_id = ? AND state = 'prepared'
                """,
                (
                    committed.state,
                    json.dumps(serialized, sort_keys=True),
                    committed.updated_at.isoformat(),
                    committed.run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("checkpoint was not in the prepared state")
        return self.get_checkpoint(run_id)

    @staticmethod
    def _record_event(
        connection: sqlite3.Connection, memory_id: str | None, action: str, details: dict
    ) -> None:
        connection.execute(
            "INSERT INTO memory_events (memory_id, action, details_json, created_at) VALUES (?, ?, ?, ?)",
            (memory_id, action, json.dumps(details, sort_keys=True), utc_now().isoformat()),
        )

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


def _decode_run(serialized: str) -> CaptureRun:
    """Read pre-repair Phase 2 traces without rewriting their saved JSON evidence."""

    payload = json.loads(serialized)
    plan = payload.get("plan")
    for step in plan.get("steps", []) if isinstance(plan, dict) else []:
        step.pop("depends_on", None)
        step.pop("run_if", None)
    return CaptureRun.model_validate(payload)


def _without_timestamps(model) -> dict:
    return model.model_dump(exclude={"created_at", "updated_at", "resolved_at"})
