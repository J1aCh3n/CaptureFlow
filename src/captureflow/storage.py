import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
import json
from datetime import date, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .credentials import contains_credential_disclosure
from .migrations import apply_migrations
from .models import (
    CalendarEvent,
    CalendarEventUpdateOperation,
    Conversation,
    DEFAULT_WORKSPACE_ID,
    EvidenceReference,
    Message,
    LocalContextSource,
    LocalContextSourceRef,
    Project,
    ProjectChange,
    ProjectChangeOperation,
    ProjectRecord,
    ProjectRecordEvent,
    SavedItem,
    ToolOperation,
    utc_now,
)


class CaptureStore:
    """The SQLite repository behind every CaptureFlow read and write."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            apply_migrations(connection)
            cursor = connection.execute(
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
        """Delete one conversation and everything that cascades from it."""

        with self._connect() as connection:
            # Databases from the retired capture workflow may still hold memories derived
            # from this conversation. Their events have no cascade, so clear them first.
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

    def save_project(self, project: Project) -> Project:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO workspaces (id, created_at) VALUES (?, ?)",
                (project.workspace_id, project.created_at.isoformat()),
            )
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO projects
                (id, workspace_id, name, description, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.id,
                    project.workspace_id,
                    project.name,
                    project.description,
                    project.status,
                    project.created_at.isoformat(),
                    project.updated_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_project(project.id)
            if existing != project:
                raise ValueError("project ID already exists with different content")
        return self.get_project(project.id)

    def get_project(self, project_id: str) -> Project:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"project not found: {project_id}")
        return Project(**dict(row))

    def list_projects(
        self,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        statuses: tuple[str, ...] = ("active",),
    ) -> list[Project]:
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM projects
                WHERE workspace_id = ? AND status IN ({placeholders})
                ORDER BY updated_at DESC, id DESC
                """,
                (workspace_id, *statuses),
            ).fetchall()
        return [Project(**dict(row)) for row in rows]

    def update_project(
        self, project_id: str, *, name: str, description: str | None
    ) -> Project:
        existing = self.get_project(project_id)
        updated = Project(
            **{
                **existing.model_dump(),
                "name": name,
                "description": description,
                "updated_at": utc_now(),
            }
        )
        with self._connect() as connection:
            connection.execute(
                "UPDATE projects SET name = ?, description = ?, updated_at = ? WHERE id = ?",
                (
                    updated.name,
                    updated.description,
                    updated.updated_at.isoformat(),
                    project_id,
                ),
            )
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> bool:
        """Delete project-only state while retaining its conversations and their local records."""

        self.get_project(project_id)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM project_change_operations WHERE project_id = ?",
                (project_id,),
            )
            connection.execute(
                "DELETE FROM project_conversation_state WHERE project_id = ?",
                (project_id,),
            )
            connection.execute(
                "DELETE FROM conversation_projects WHERE project_id = ?",
                (project_id,),
            )
            connection.execute(
                "DELETE FROM project_records WHERE project_id = ?",
                (project_id,),
            )
            cursor = connection.execute(
                "DELETE FROM projects WHERE id = ?",
                (project_id,),
            )
        return cursor.rowcount == 1

    def set_project_status(self, project_id: str, status: str) -> Project:
        if status not in {"active", "archived"}:
            raise ValueError("invalid project status")
        existing = self.get_project(project_id)
        if existing.status == status:
            return existing
        with self._connect() as connection:
            connection.execute(
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now().isoformat(), project_id),
            )
        return self.get_project(project_id)

    def get_conversation_project(self, conversation_id: str) -> Project | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT project_id FROM conversation_projects WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return None if row is None else self.get_project(row["project_id"])

    def assign_conversation_project(self, conversation_id: str, project_id: str) -> Project:
        self.get_conversation(conversation_id)
        project = self.get_project(project_id)
        if project.status != "active":
            raise ValueError("an archived project cannot receive a conversation")
        existing = self.get_conversation_project(conversation_id)
        if existing is not None and existing.id == project_id:
            return existing
        if self.conversation_has_project_turn(conversation_id):
            raise ValueError("conversation project is locked after its first project chat turn")
        if existing is not None and self.list_project_records_for_conversation(conversation_id):
            raise ValueError("conversation project is locked after its first project record")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_projects (conversation_id, project_id, assigned_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    assigned_at = excluded.assigned_at
                """,
                (conversation_id, project_id, utc_now().isoformat()),
            )
        return project

    def conversation_has_project_turn(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM project_conversation_state WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return row is not None

    def begin_project_conversation(self, conversation_id: str) -> Project:
        project = self.get_conversation_project(conversation_id)
        if project is None:
            raise ValueError("assign a project before starting project chat")
        if project.status != "active":
            raise ValueError("an archived project cannot receive project chat")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT project_id FROM project_conversation_state WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is not None and row["project_id"] != project.id:
                raise ValueError("project chat state does not match the conversation project")
            if row is None:
                connection.execute(
                    """
                    INSERT INTO project_conversation_state (conversation_id, project_id, started_at)
                    VALUES (?, ?, ?)
                    """,
                    (conversation_id, project.id, utc_now().isoformat()),
                )
        return project

    def save_project_record(self, record: ProjectRecord) -> ProjectRecord:
        project = self.get_project(record.project_id)
        if project.status != "active":
            raise ValueError("cannot add a record to an archived project")
        source = self._require_source_message(record.source_message_id, record.conversation_id)
        assigned = self.get_conversation_project(record.conversation_id)
        if assigned is None or assigned.id != record.project_id:
            raise ValueError("a project record must use its conversation's assigned project")
        self._require_source_quote(source, record.source_quote)
        if record.supersedes_record_id is not None:
            superseded = self.get_project_record(record.supersedes_record_id)
            if (
                superseded.project_id != record.project_id
                or superseded.kind != record.kind
                or superseded.state != "active"
            ):
                raise ValueError("a replacement must target an active record of the same project and kind")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO project_records
                (id, project_id, conversation_id, kind, content, rationale, due_at, state,
                 source_message_id, source_quote, supersedes_record_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.project_id,
                    record.conversation_id,
                    record.kind,
                    record.content,
                    record.rationale,
                    None if record.due_at is None else record.due_at.isoformat(),
                    record.state,
                    record.source_message_id,
                    record.source_quote,
                    record.supersedes_record_id,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                self._save_project_record_event_in_connection(
                    connection,
                    ProjectRecordEvent(
                        project_record_id=record.id,
                        action="created",
                        source_message_id=record.source_message_id,
                        new_state=record.state,
                    ),
                )
        if not inserted:
            existing = self.get_project_record(record.id)
            if existing != record:
                raise ValueError("project record ID already exists with different content")
        return self.get_project_record(record.id)

    def get_project_record(self, record_id: str) -> ProjectRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM project_records WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(f"project record not found: {record_id}")
        return ProjectRecord(**dict(row))

    def list_project_records(
        self,
        project_id: str,
        states: tuple[str, ...] = ("active", "completed", "resolved", "superseded"),
    ) -> list[ProjectRecord]:
        if not states:
            return []
        placeholders = ", ".join("?" for _ in states)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM project_records
                WHERE project_id = ? AND state IN ({placeholders})
                ORDER BY updated_at DESC, id DESC
                """,
                (project_id, *states),
            ).fetchall()
        return [ProjectRecord(**dict(row)) for row in rows]

    def list_task_deadlines(
        self, workspace_id: str = DEFAULT_WORKSPACE_ID, states: tuple[str, ...] = ("active",)
    ) -> list[ProjectRecord]:
        """Active tracker tasks that carry a deadline, for the read-only Calendar view."""

        if not states:
            return []
        placeholders = ", ".join("?" for _ in states)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT project_records.* FROM project_records
                JOIN projects ON projects.id = project_records.project_id
                WHERE projects.workspace_id = ?
                  AND project_records.kind = 'task'
                  AND project_records.due_at IS NOT NULL
                  AND project_records.state IN ({placeholders})
                ORDER BY project_records.due_at ASC, project_records.id ASC
                """,
                (workspace_id, *states),
            ).fetchall()
        return [ProjectRecord(**dict(row)) for row in rows]

    def list_project_records_for_conversation(self, conversation_id: str) -> list[ProjectRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM project_records WHERE conversation_id = ? ORDER BY created_at ASC, id ASC",
                (conversation_id,),
            ).fetchall()
        return [ProjectRecord(**dict(row)) for row in rows]

    def list_project_record_events(self, record_id: str) -> list[ProjectRecordEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM project_record_events
                WHERE project_record_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (record_id,),
            ).fetchall()
        return [_decode_project_record_event(row) for row in rows]

    def save_project_answer_sources(
        self, assistant_message_id: str, record_ids: list[str]
    ) -> list[ProjectRecord]:
        """Persist the bounded record IDs that support one project answer."""

        if not record_ids or len(record_ids) > 6 or len(set(record_ids)) != len(record_ids):
            raise ValueError("a project answer needs one to six unique source records")
        assistant = self.get_message(assistant_message_id)
        if assistant.role != "assistant" or assistant.reply_to_message_id is None:
            raise ValueError("project answer sources require an assistant reply")
        source = self._require_source_message(assistant.reply_to_message_id, assistant.conversation_id)
        project = self.get_conversation_project(source.conversation_id)
        if project is None:
            raise ValueError("project answer sources require an assigned project")
        records = [self.get_project_record(record_id) for record_id in record_ids]
        if any(record.project_id != project.id for record in records):
            raise ValueError("project answer sources must belong to the reply project")
        with self._connect() as connection:
            for rank, record_id in enumerate(record_ids):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO project_answer_sources
                    (assistant_message_id, project_record_id, source_rank)
                    VALUES (?, ?, ?)
                    """,
                    (assistant_message_id, record_id, rank),
                )
        saved = self.list_project_answer_sources(assistant_message_id)
        if [record.id for record in saved] != record_ids:
            raise ValueError("project answer sources already exist with different records")
        return saved

    def list_project_answer_sources(self, assistant_message_id: str) -> list[ProjectRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT project_records.*
                FROM project_answer_sources
                JOIN project_records ON project_records.id = project_answer_sources.project_record_id
                WHERE project_answer_sources.assistant_message_id = ?
                ORDER BY project_answer_sources.source_rank ASC, project_records.id ASC
                """,
                (assistant_message_id,),
            ).fetchall()
        return [ProjectRecord(**dict(row)) for row in rows]

    def save_local_context_answer_sources(
        self, assistant_message_id: str, references: list[LocalContextSourceRef]
    ) -> list[LocalContextSource]:
        """Persist the bounded Calendar/Note/Idea references supporting one read-only answer."""

        if not references or len(references) > 6 or len(set(references)) != len(references):
            raise ValueError("a local-context answer needs one to six unique sources")
        assistant = self.get_message(assistant_message_id)
        if assistant.role != "assistant" or assistant.reply_to_message_id is None:
            raise ValueError("local-context sources require an assistant reply")
        conversation = self.get_conversation(assistant.conversation_id)
        for reference in references:
            if reference.source_type == "calendar_event":
                source = self.get_calendar_event(reference.source_id)
            else:
                source = self.get_saved_item(reference.source_id)
            if source.workspace_id != conversation.workspace_id:
                raise ValueError("local-context sources must belong to the reply workspace")
        with self._connect() as connection:
            for rank, reference in enumerate(references):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO local_context_answer_sources
                    (assistant_message_id, source_type, source_id, source_rank)
                    VALUES (?, ?, ?, ?)
                    """,
                    (assistant_message_id, reference.source_type, reference.source_id, rank),
                )
        saved = self.list_local_context_answer_sources(assistant_message_id)
        if [source.reference for source in saved] != references:
            raise ValueError("local-context sources already exist with different records")
        return saved

    def list_local_context_answer_sources(self, assistant_message_id: str) -> list[LocalContextSource]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_type, source_id
                FROM local_context_answer_sources
                WHERE assistant_message_id = ?
                ORDER BY source_rank ASC, source_type ASC, source_id ASC
                """,
                (assistant_message_id,),
            ).fetchall()
        sources: list[LocalContextSource] = []
        for row in rows:
            reference = LocalContextSourceRef(source_type=row["source_type"], source_id=row["source_id"])
            try:
                if reference.source_type == "calendar_event":
                    event = self.get_calendar_event(reference.source_id)
                    sources.append(
                        LocalContextSource(
                            reference=reference,
                            label=(
                                f"Calendar · {event.start_at.isoformat()}"
                                if event.end_at is None
                                else (
                                    f"Calendar · {event.start_at.isoformat()} to "
                                    f"{event.end_at.isoformat()}"
                                )
                            ),
                            content=(
                                event.title
                                if event.description is None
                                else f"{event.title}\n\n{event.description}"
                            ),
                            source_quote=event.source_quote,
                        )
                    )
                else:
                    item = self.get_saved_item(reference.source_id)
                    sources.append(
                        LocalContextSource(
                            reference=reference,
                            label=item.kind.title(),
                            content=item.content,
                            source_quote=item.source_quote,
                        )
                    )
            except KeyError:
                continue
        return sources

    def transition_project_record(
        self,
        record_id: str,
        state: str,
        *,
        source_message_id: str | None = None,
        details: dict | None = None,
    ) -> ProjectRecord:
        existing = self.get_project_record(record_id)
        if existing.state == state:
            return existing
        if existing.state == "deleted":
            raise ValueError("restore a deleted project record instead of transitioning it")
        transition_actions = {
            ("task", "active", "completed"): "completed",
            ("open_question", "active", "resolved"): "resolved",
            ("blocker", "active", "resolved"): "resolved",
            ("decision", "active", "superseded"): "superseded",
            ("constraint", "active", "superseded"): "superseded",
        }
        action = transition_actions.get((existing.kind, existing.state, state))
        if action is None:
            raise ValueError("invalid project record lifecycle transition")
        self._validate_project_event_source(existing, source_message_id)
        updated = existing.model_copy(update={"state": state, "updated_at": utc_now()})
        with self._connect() as connection:
            connection.execute(
                "UPDATE project_records SET state = ?, updated_at = ? WHERE id = ?",
                (updated.state, updated.updated_at.isoformat(), updated.id),
            )
            self._save_project_record_event_in_connection(
                connection,
                ProjectRecordEvent(
                    project_record_id=updated.id,
                    action=action,
                    source_message_id=source_message_id,
                    previous_state=existing.state,
                    new_state=updated.state,
                    details=details or {},
                ),
            )
        return self.get_project_record(record_id)

    def update_project_record(
        self, record_id: str, *, content: str, rationale: str | None = None
    ) -> ProjectRecord:
        existing = self.get_project_record(record_id)
        if existing.state == "deleted":
            raise ValueError("restore a deleted project record before editing it")
        updated = existing.model_copy(
            update={"content": content, "rationale": rationale, "updated_at": utc_now()}
        )
        if updated.content == existing.content and updated.rationale == existing.rationale:
            return existing
        with self._connect() as connection:
            connection.execute(
                "UPDATE project_records SET content = ?, rationale = ?, updated_at = ? WHERE id = ?",
                (updated.content, updated.rationale, updated.updated_at.isoformat(), updated.id),
            )
            self._save_project_record_event_in_connection(
                connection,
                ProjectRecordEvent(
                    project_record_id=updated.id,
                    action="edited",
                    previous_state=existing.state,
                    new_state=updated.state,
                    details={"origin": "ui"},
                ),
            )
        return self.get_project_record(record_id)

    def delete_project_record(self, record_id: str) -> ProjectRecord:
        existing = self.get_project_record(record_id)
        if existing.state == "deleted":
            return existing
        updated = existing.model_copy(update={"state": "deleted", "updated_at": utc_now()})
        with self._connect() as connection:
            connection.execute(
                "UPDATE project_records SET state = ?, updated_at = ? WHERE id = ?",
                (updated.state, updated.updated_at.isoformat(), updated.id),
            )
            self._save_project_record_event_in_connection(
                connection,
                ProjectRecordEvent(
                    project_record_id=updated.id,
                    action="deleted",
                    previous_state=existing.state,
                    new_state="deleted",
                    details={"origin": "ui"},
                ),
            )
        return self.get_project_record(record_id)

    def restore_project_record(self, record_id: str) -> ProjectRecord:
        existing = self.get_project_record(record_id)
        if existing.state != "deleted":
            return existing
        events = self.list_project_record_events(record_id)
        deleted_events = [event for event in events if event.action == "deleted"]
        if not deleted_events or deleted_events[-1].previous_state is None:
            raise ValueError("deleted project record has no restorable state")
        restored_state = deleted_events[-1].previous_state
        restored = existing.model_copy(update={"state": restored_state, "updated_at": utc_now()})
        with self._connect() as connection:
            connection.execute(
                "UPDATE project_records SET state = ?, updated_at = ? WHERE id = ?",
                (restored.state, restored.updated_at.isoformat(), restored.id),
            )
            self._save_project_record_event_in_connection(
                connection,
                ProjectRecordEvent(
                    project_record_id=restored.id,
                    action="restored",
                    previous_state="deleted",
                    new_state=restored.state,
                    details={"origin": "ui"},
                ),
            )
        return self.get_project_record(record_id)

    def save_project_change_operation(
        self, operation: ProjectChangeOperation
    ) -> ProjectChangeOperation:
        """Persist a validated proposal before any tracker mutation."""

        source = self._require_source_message(operation.source_message_id)
        project = self.get_conversation_project(source.conversation_id)
        if project is None or project.id != operation.project_id:
            raise ValueError("a project change source must belong to its project")
        change = ProjectChange.model_validate(operation.change)
        self._require_source_quote(self._change_quote_source(change, source), change.source_quote)
        if contains_credential_disclosure(change.source_quote) or (
            change.content is not None and contains_credential_disclosure(change.content)
        ):
            raise ValueError("credentials cannot be persisted in a project change")
        if operation.target_record_id != change.target_record_id:
            raise ValueError("project change target must match its request")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO project_change_operations
                (id, source_message_id, project_id, change_index, change_json, target_record_id,
                 result_json, status, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    operation.id,
                    operation.source_message_id,
                    operation.project_id,
                    operation.change_index,
                    json.dumps(operation.change, sort_keys=True),
                    operation.target_record_id,
                    None if operation.result is None else json.dumps(operation.result, sort_keys=True),
                    operation.status,
                    operation.error,
                    operation.created_at.isoformat(),
                    operation.updated_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_project_change_operation(operation.id)
            if existing != operation:
                raise ValueError("project change operation ID already exists with different content")
        return self.get_project_change_operation(operation.id)

    def get_project_change_operation(self, operation_id: str) -> ProjectChangeOperation:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM project_change_operations WHERE id = ?", (operation_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"project change operation not found: {operation_id}")
        return _decode_project_change_operation(row)

    def list_project_change_operations(
        self, source_message_id: str
    ) -> list[ProjectChangeOperation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM project_change_operations
                WHERE source_message_id = ?
                ORDER BY change_index ASC, id ASC
                """,
                (source_message_id,),
            ).fetchall()
        return [_decode_project_change_operation(row) for row in rows]

    def commit_prepared_project_change(
        self, operation_id: str
    ) -> ProjectChangeOperation:
        """Atomically apply one prepared request using only its persisted contents."""

        with self._connect() as connection:
            operation = self._project_change_operation_in_connection(connection, operation_id)
            if operation.status == "committed":
                return operation
            if operation.status != "prepared":
                raise ValueError("only a prepared project change can be committed")
            change = ProjectChange.model_validate(operation.change)
            source = self._require_source_message(operation.source_message_id)
            project = self.get_conversation_project(source.conversation_id)
            if project is None or project.id != operation.project_id or project.status != "active":
                raise ValueError("project change no longer has an active assigned project")
            quote_source = self._change_quote_source(change, source)
            self._require_source_quote(quote_source, change.source_quote)
            if change.action == "create_record":
                record = ProjectRecord(
                    id=str(uuid5(NAMESPACE_URL, f"captureflow:project-change-record:{operation.id}")),
                    project_id=operation.project_id,
                    conversation_id=source.conversation_id,
                    kind=change.kind,
                    content=change.content,
                    rationale=change.rationale,
                    due_at=change.due_at,
                    state="completed" if change.kind == "milestone" else "active",
                    source_message_id=quote_source.id,
                    source_quote=change.source_quote,
                )
                self._insert_project_record_in_connection(connection, record)
                result = {
                    "record_id": record.id,
                    "previous_state": None,
                    "new_state": record.state,
                }
            elif change.action == "set_deadline":
                record = self._project_record_in_connection(connection, change.target_record_id)
                self._validate_project_change_target(record, operation.project_id, change.target_record_id)
                if record.kind != "task":
                    raise ValueError("only a task can carry a deadline")
                previous_due_at = None if record.due_at is None else record.due_at.isoformat()
                self._write_project_record_due_at(connection, record.id, change.due_at)
                self._save_project_record_event_in_connection(
                    connection,
                    ProjectRecordEvent(
                        project_record_id=record.id,
                        action="edited",
                        source_message_id=source.id,
                        previous_state=record.state,
                        new_state=record.state,
                        details={
                            "origin": "project_change_operation",
                            "operation_id": operation.id,
                            "previous_due_at": previous_due_at,
                            "due_at": None if change.due_at is None else change.due_at.isoformat(),
                        },
                    ),
                )
                result = {
                    "record_id": record.id,
                    "previous_state": record.state,
                    "new_state": record.state,
                    "previous_due_at": previous_due_at,
                }
            elif change.action == "transition_record":
                record = self._project_record_in_connection(connection, change.target_record_id)
                action = self._transition_action(record, change.target_state)
                self._validate_project_change_target(record, operation.project_id, change.target_record_id)
                updated_at = utc_now()
                connection.execute(
                    "UPDATE project_records SET state = ?, updated_at = ? WHERE id = ?",
                    (change.target_state, updated_at.isoformat(), record.id),
                )
                self._save_project_record_event_in_connection(
                    connection,
                    ProjectRecordEvent(
                        project_record_id=record.id,
                        action=action,
                        source_message_id=source.id,
                        previous_state=record.state,
                        new_state=change.target_state,
                        details={"origin": "project_change_operation", "operation_id": operation.id},
                    ),
                )
                result = {
                    "record_id": record.id,
                    "previous_state": record.state,
                    "new_state": change.target_state,
                }
            else:
                target = self._project_record_in_connection(connection, change.target_record_id)
                self._validate_project_change_target(target, operation.project_id, change.target_record_id)
                if target.kind != change.kind or target.kind not in {"decision", "constraint"}:
                    raise ValueError("a supersession must target an active record of the same kind")
                replacement = ProjectRecord(
                    id=str(uuid5(NAMESPACE_URL, f"captureflow:project-change-record:{operation.id}")),
                    project_id=operation.project_id,
                    conversation_id=source.conversation_id,
                    kind=change.kind,
                    content=change.content,
                    rationale=change.rationale,
                    state="active",
                    source_message_id=quote_source.id,
                    source_quote=change.source_quote,
                    supersedes_record_id=target.id,
                )
                self._insert_project_record_in_connection(connection, replacement)
                updated_at = utc_now()
                connection.execute(
                    "UPDATE project_records SET state = ?, updated_at = ? WHERE id = ?",
                    ("superseded", updated_at.isoformat(), target.id),
                )
                self._save_project_record_event_in_connection(
                    connection,
                    ProjectRecordEvent(
                        project_record_id=target.id,
                        action="superseded",
                        source_message_id=source.id,
                        previous_state="active",
                        new_state="superseded",
                        details={"origin": "project_change_operation", "operation_id": operation.id},
                    ),
                )
                result = {
                    "record_id": replacement.id,
                    "target_record_id": target.id,
                    "previous_state": "active",
                    "new_state": "active",
                }
            self._commit_project_change_operation(connection, operation, result)
        return self.get_project_change_operation(operation_id)

    def approve_project_change_operation(self, operation_id: str) -> ProjectChangeOperation:
        with self._connect() as connection:
            operation = self._project_change_operation_in_connection(connection, operation_id)
            if operation.status == "committed":
                return operation
            if operation.status != "review":
                raise ValueError("only a review project change can be approved")
            connection.execute(
                "UPDATE project_change_operations SET status = ?, updated_at = ? WHERE id = ?",
                ("prepared", utc_now().isoformat(), operation.id),
            )
        return self.commit_prepared_project_change(operation_id)

    def reject_project_change_operation(self, operation_id: str) -> ProjectChangeOperation:
        with self._connect() as connection:
            operation = self._project_change_operation_in_connection(connection, operation_id)
            if operation.status == "failed":
                return operation
            if operation.status != "review":
                raise ValueError("only a review project change can be rejected")
            connection.execute(
                """
                UPDATE project_change_operations
                SET status = ?, error = ?, updated_at = ? WHERE id = ?
                """,
                ("failed", "rejected by user", utc_now().isoformat(), operation.id),
            )
        return self.get_project_change_operation(operation_id)

    def fail_prepared_project_change(self, operation_id: str, error: str) -> ProjectChangeOperation:
        with self._connect() as connection:
            operation = self._project_change_operation_in_connection(connection, operation_id)
            if operation.status == "failed":
                return operation
            if operation.status != "prepared":
                raise ValueError("only a prepared project change can fail")
            connection.execute(
                """
                UPDATE project_change_operations
                SET status = ?, error = ?, updated_at = ? WHERE id = ?
                """,
                ("failed", error, utc_now().isoformat(), operation.id),
            )
        return self.get_project_change_operation(operation_id)

    def undo_project_change_operation(self, operation_id: str) -> ProjectChangeOperation:
        """Undo only an unchanged committed change and retain the audit history."""

        with self._connect() as connection:
            operation = self._project_change_operation_in_connection(connection, operation_id)
            if operation.status == "undone":
                return operation
            if operation.status != "committed" or operation.result is None:
                raise ValueError("only a committed project change can be undone")
            change = ProjectChange.model_validate(operation.change)
            source = self._require_source_message(operation.source_message_id)
            if change.action == "create_record":
                record = self._project_record_in_connection(connection, operation.result["record_id"])
                if record.state != operation.result["new_state"]:
                    raise ValueError("project record changed after this operation and cannot be undone")
                self._set_project_record_state_in_connection(
                    connection, record, "deleted", "deleted", source.id, operation.id
                )
            elif change.action == "set_deadline":
                record = self._project_record_in_connection(connection, operation.result["record_id"])
                if record.due_at != change.due_at:
                    raise ValueError("project record changed after this operation and cannot be undone")
                previous = operation.result.get("previous_due_at")
                self._write_project_record_due_at(
                    connection, record.id, None if previous is None else _parse_due_at(previous)
                )
                self._save_project_record_event_in_connection(
                    connection,
                    ProjectRecordEvent(
                        project_record_id=record.id,
                        action="edited",
                        source_message_id=source.id,
                        previous_state=record.state,
                        new_state=record.state,
                        details={"reverted_action": "set_deadline", "operation_id": operation.id},
                    ),
                )
            elif change.action == "transition_record":
                record = self._project_record_in_connection(connection, operation.result["record_id"])
                if record.state != operation.result["new_state"]:
                    raise ValueError("project record changed after this operation and cannot be undone")
                self._set_project_record_state_in_connection(
                    connection,
                    record,
                    operation.result["previous_state"],
                    "edited",
                    source.id,
                    operation.id,
                    {"reverted_action": "transition"},
                )
            else:
                replacement = self._project_record_in_connection(connection, operation.result["record_id"])
                target = self._project_record_in_connection(connection, operation.result["target_record_id"])
                if replacement.state != "active" or target.state != "superseded":
                    raise ValueError("supersession changed after this operation and cannot be undone")
                self._set_project_record_state_in_connection(
                    connection, replacement, "deleted", "deleted", source.id, operation.id
                )
                self._set_project_record_state_in_connection(
                    connection,
                    target,
                    "active",
                    "edited",
                    source.id,
                    operation.id,
                    {"reverted_action": "supersession"},
                )
            connection.execute(
                "UPDATE project_change_operations SET status = ?, updated_at = ? WHERE id = ?",
                ("undone", utc_now().isoformat(), operation.id),
            )
        return self.get_project_change_operation(operation_id)

    def save_message(self, message: Message) -> Message:
        if message.role == "user" and message.reply_to_message_id is not None:
            raise ValueError("a user message cannot reply to another message")
        if message.role == "assistant":
            if message.reply_to_message_id is None:
                raise ValueError("an assistant message must reference its user message")
            parent = self.get_message(message.reply_to_message_id)
            if parent.role != "user" or parent.conversation_id != message.conversation_id:
                raise ValueError("an assistant message must reply to a user in its conversation")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO messages
                (id, conversation_id, role, content, reply_to_message_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.reply_to_message_id,
                    message.created_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_message(message.id)
            if existing != message:
                raise ValueError("message ID already exists with different content")
        return self.get_message(message.id)

    def get_message(self, message_id: str) -> Message:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"message not found: {message_id}")
        return Message(**dict(row))

    def list_messages(self, conversation_id: str) -> list[Message]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                -- UUIDs are not chronological. Two adjacent messages can have the
                -- same timestamp, so use SQLite's insertion order as the stable
                -- tie-breaker and keep a user turn ahead of its reply.
                ORDER BY created_at ASC, rowid ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [Message(**dict(row)) for row in rows]

    def get_assistant_reply(self, user_message_id: str) -> Message | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM messages
                WHERE reply_to_message_id = ? AND role = 'assistant'
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """,
                (user_message_id,),
            ).fetchone()
        return None if row is None else Message(**dict(row))

    def save_calendar_event(self, event: CalendarEvent) -> CalendarEvent:
        self._require_source_message(event.source_message_id, event.conversation_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO calendar_events
                (id, workspace_id, conversation_id, title, description, source_message_id,
                 source_quote, start_at, end_at, all_day, timezone, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.workspace_id,
                    event.conversation_id,
                    event.title,
                    event.description,
                    event.source_message_id,
                    event.source_quote,
                    event.start_at.isoformat(),
                    None if event.end_at is None else event.end_at.isoformat(),
                    int(event.all_day),
                    event.timezone,
                    event.status,
                    event.created_at.isoformat(),
                    event.updated_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                self._save_calendar_event_sources(
                    connection,
                    event,
                    [
                        EvidenceReference(
                            message_id=event.source_message_id,
                            source_quote=event.source_quote,
                        )
                    ],
                )
        if not inserted:
            existing = self.get_calendar_event(event.id)
            if existing != event:
                raise ValueError("calendar event ID already exists with different content")
        return self.get_calendar_event(event.id)

    def get_calendar_event(self, event_id: str) -> CalendarEvent:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM calendar_events WHERE id = ?", (event_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"calendar event not found: {event_id}")
        return CalendarEvent(**dict(row))

    def list_calendar_events(
        self,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        statuses: tuple[str, ...] = ("active",),
    ) -> list[CalendarEvent]:
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM calendar_events
                WHERE workspace_id = ? AND status IN ({placeholders})
                ORDER BY start_at ASC, id ASC
                """,
                (workspace_id, *statuses),
            ).fetchall()
        return [CalendarEvent(**dict(row)) for row in rows]

    def list_calendar_event_sources(self, event_id: str) -> list[EvidenceReference]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_message_id, source_quote
                FROM calendar_event_sources
                WHERE event_id = ?
                ORDER BY source_rank ASC
                """,
                (event_id,),
            ).fetchall()
        return [
            EvidenceReference(message_id=row["source_message_id"], source_quote=row["source_quote"])
            for row in rows
        ]

    def update_calendar_event(
        self, event_id: str, *, title: str, start_at: date | datetime
    ) -> CalendarEvent:
        existing = self.get_calendar_event(event_id)
        updated = CalendarEvent(
            **{
                **existing.model_dump(),
                "title": title,
                "start_at": start_at,
                "updated_at": utc_now(),
            }
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE calendar_events
                SET title = ?, start_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (updated.title, updated.start_at.isoformat(), updated.updated_at.isoformat(), event_id),
            )
        return self.get_calendar_event(event_id)

    def prepare_calendar_event_update(
        self, operation: CalendarEventUpdateOperation, evidence: list[EvidenceReference]
    ) -> CalendarEventUpdateOperation:
        """Persist a validated update intent and its bounded message provenance."""

        if operation.status not in {"prepared", "review"}:
            raise ValueError("only a prepared or review calendar update can be saved")
        if not evidence:
            raise ValueError("a calendar update requires evidence")
        if len(evidence) > 3:
            raise ValueError("a calendar update supports at most three evidence references")
        if len({reference.message_id for reference in evidence}) != len(evidence):
            raise ValueError("calendar update evidence messages must be unique")

        with self._connect() as connection:
            existing = self._calendar_event_in_connection(connection, operation.event_id)
            if existing.workspace_id != operation.workspace_id:
                raise ValueError("calendar update target is outside the intended workspace")
            if existing.status != "active":
                raise ValueError("only active calendar events can be updated")
            if existing != operation.before:
                raise ValueError("calendar update target changed before preparation")
            source = self._source_message_in_workspace(
                connection, operation.source_message_id, operation.workspace_id
            )
            if source.role != "user":
                raise ValueError("a calendar update source message must have the user role")
            for reference in evidence:
                message = self._source_message_in_workspace(
                    connection, reference.message_id, operation.workspace_id
                )
                if message.role != "user":
                    raise ValueError("calendar update evidence must reference user messages")
                self._require_source_quote(message, reference.source_quote)
            cursor = connection.execute(
                """
                INSERT INTO calendar_event_update_operations
                (id, source_message_id, action_index, event_id, workspace_id, before_json,
                 after_json, status, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    operation.id,
                    operation.source_message_id,
                    operation.action_index,
                    operation.event_id,
                    operation.workspace_id,
                    json.dumps(operation.before.model_dump(mode="json"), sort_keys=True),
                    json.dumps(operation.after.model_dump(mode="json"), sort_keys=True),
                    operation.status,
                    operation.error,
                    operation.created_at.isoformat(),
                    operation.updated_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                connection.executemany(
                    """
                    INSERT INTO calendar_event_update_sources
                    (update_operation_id, source_message_id, source_quote, source_rank)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (operation.id, reference.message_id, reference.source_quote, rank)
                        for rank, reference in enumerate(evidence)
                    ],
                )
        if not inserted:
            existing_operation = self.get_calendar_event_update_operation(operation.id)
            if existing_operation != operation or self.list_calendar_event_update_sources(operation.id) != evidence:
                raise ValueError("calendar update operation ID already exists with different content")
        return self.get_calendar_event_update_operation(operation.id)

    def get_calendar_event_update_operation(self, operation_id: str) -> CalendarEventUpdateOperation:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM calendar_event_update_operations WHERE id = ?", (operation_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"calendar update operation not found: {operation_id}")
        return _decode_calendar_event_update_operation(row)

    def list_calendar_event_update_operations(
        self, source_message_id: str
    ) -> list[CalendarEventUpdateOperation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM calendar_event_update_operations
                WHERE source_message_id = ?
                ORDER BY action_index ASC, id ASC
                """,
                (source_message_id,),
            ).fetchall()
        return [_decode_calendar_event_update_operation(row) for row in rows]

    def list_calendar_event_update_sources(self, operation_id: str) -> list[EvidenceReference]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_message_id, source_quote
                FROM calendar_event_update_sources
                WHERE update_operation_id = ?
                ORDER BY source_rank ASC
                """,
                (operation_id,),
            ).fetchall()
        return [
            EvidenceReference(message_id=row["source_message_id"], source_quote=row["source_quote"])
            for row in rows
        ]

    def commit_prepared_calendar_event_update(
        self, operation_id: str
    ) -> tuple[CalendarEventUpdateOperation, CalendarEvent]:
        with self._connect() as connection:
            operation = self._calendar_event_update_operation_in_connection(connection, operation_id)
            if operation.status == "committed":
                return operation, self._calendar_event_in_connection(connection, operation.event_id)
            if operation.status != "prepared":
                raise ValueError("only a prepared calendar update can be committed")
            existing = self._calendar_event_in_connection(connection, operation.event_id)
            if existing != operation.before:
                raise ValueError("calendar update target changed before commit")
            self._write_calendar_event(connection, operation.after)
            self._set_calendar_event_update_status(connection, operation, "committed")
        return self.get_calendar_event_update_operation(operation_id), self.get_calendar_event(operation.event_id)

    def approve_calendar_event_update(
        self, operation_id: str
    ) -> tuple[CalendarEventUpdateOperation, CalendarEvent]:
        """Apply a change that waited for the user, such as a deletion."""

        with self._connect() as connection:
            operation = self._calendar_event_update_operation_in_connection(connection, operation_id)
            if operation.status == "committed":
                return operation, self._calendar_event_in_connection(connection, operation.event_id)
            if operation.status != "review":
                raise ValueError("only a calendar change in review can be approved")
            self._set_calendar_event_update_status(connection, operation, "prepared")
        return self.commit_prepared_calendar_event_update(operation_id)

    def reject_calendar_event_update(self, operation_id: str) -> CalendarEventUpdateOperation:
        with self._connect() as connection:
            operation = self._calendar_event_update_operation_in_connection(connection, operation_id)
            if operation.status == "failed":
                return operation
            if operation.status != "review":
                raise ValueError("only a calendar change in review can be rejected")
            connection.execute(
                """
                UPDATE calendar_event_update_operations
                SET status = ?, error = ?, updated_at = ? WHERE id = ?
                """,
                ("failed", "rejected by user", utc_now().isoformat(), operation.id),
            )
        return self.get_calendar_event_update_operation(operation_id)

    def undo_calendar_event_update(self, operation_id: str) -> CalendarEventUpdateOperation:
        with self._connect() as connection:
            operation = self._calendar_event_update_operation_in_connection(connection, operation_id)
            if operation.status == "undone":
                return operation
            if operation.status != "committed":
                raise ValueError("only a committed calendar update can be undone")
            existing = self._calendar_event_in_connection(connection, operation.event_id)
            if existing != operation.after:
                raise ValueError("calendar update target changed after commit")
            self._write_calendar_event(connection, operation.before)
            self._set_calendar_event_update_status(connection, operation, "undone")
        return self.get_calendar_event_update_operation(operation_id)

    def set_calendar_event_status(self, event_id: str, status: str) -> CalendarEvent:
        if status not in {"active", "deleted"}:
            raise ValueError("invalid calendar event status")
        existing = self.get_calendar_event(event_id)
        if existing.status == status:
            return existing
        updated_at = utc_now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE calendar_events SET status = ?, updated_at = ? WHERE id = ?",
                (status, updated_at.isoformat(), event_id),
            )
        return self.get_calendar_event(event_id)

    def save_saved_item(self, item: SavedItem) -> SavedItem:
        self._require_source_message(item.source_message_id, item.conversation_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO saved_items
                (id, workspace_id, conversation_id, kind, content, source_message_id,
                 source_quote, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.workspace_id,
                    item.conversation_id,
                    item.kind,
                    item.content,
                    item.source_message_id,
                    item.source_quote,
                    item.status,
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_saved_item(item.id)
            if existing != item:
                raise ValueError("saved item ID already exists with different content")
        return self.get_saved_item(item.id)

    def get_saved_item(self, item_id: str) -> SavedItem:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM saved_items WHERE id = ?", (item_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"saved item not found: {item_id}")
        return SavedItem(**dict(row))

    def list_saved_items(
        self,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        kinds: tuple[str, ...] = ("idea", "note"),
        statuses: tuple[str, ...] = ("active",),
    ) -> list[SavedItem]:
        if not kinds or not statuses:
            return []
        kind_placeholders = ", ".join("?" for _ in kinds)
        status_placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM saved_items
                WHERE workspace_id = ?
                  AND kind IN ({kind_placeholders})
                  AND status IN ({status_placeholders})
                ORDER BY updated_at DESC, id DESC
                """,
                (workspace_id, *kinds, *statuses),
            ).fetchall()
        return [SavedItem(**dict(row)) for row in rows]

    def list_saved_items_for_project(
        self,
        project_id: str,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        kinds: tuple[str, ...] = ("idea", "note"),
        statuses: tuple[str, ...] = ("active",),
    ) -> list[SavedItem]:
        if not kinds or not statuses:
            return []
        kind_placeholders = ", ".join("?" for _ in kinds)
        status_placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT saved_items.*
                FROM saved_items
                JOIN conversation_projects
                  ON conversation_projects.conversation_id = saved_items.conversation_id
                WHERE conversation_projects.project_id = ?
                  AND saved_items.workspace_id = ?
                  AND saved_items.kind IN ({kind_placeholders})
                  AND saved_items.status IN ({status_placeholders})
                ORDER BY saved_items.updated_at DESC, saved_items.id DESC
                """,
                (project_id, workspace_id, *kinds, *statuses),
            ).fetchall()
        return [SavedItem(**dict(row)) for row in rows]

    def update_saved_item(self, item_id: str, *, content: str) -> SavedItem:
        existing = self.get_saved_item(item_id)
        updated = SavedItem(
            **{**existing.model_dump(), "content": content, "updated_at": utc_now()}
        )
        with self._connect() as connection:
            connection.execute(
                "UPDATE saved_items SET content = ?, updated_at = ? WHERE id = ?",
                (updated.content, updated.updated_at.isoformat(), item_id),
            )
        return self.get_saved_item(item_id)

    def set_saved_item_status(self, item_id: str, status: str) -> SavedItem:
        if status not in {"active", "deleted"}:
            raise ValueError("invalid saved item status")
        existing = self.get_saved_item(item_id)
        if existing.status == status:
            return existing
        updated_at = utc_now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE saved_items SET status = ?, updated_at = ? WHERE id = ?",
                (status, updated_at.isoformat(), item_id),
            )
        return self.get_saved_item(item_id)

    def save_tool_operation(self, operation: ToolOperation) -> ToolOperation:
        self._require_source_message(operation.source_message_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tool_operations
                (id, source_message_id, action_index, tool_name, request_json, result_json,
                 status, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    operation.id,
                    operation.source_message_id,
                    operation.action_index,
                    operation.tool_name,
                    json.dumps(operation.request, sort_keys=True),
                    None if operation.result is None else json.dumps(operation.result, sort_keys=True),
                    operation.status,
                    operation.error,
                    operation.created_at.isoformat(),
                    operation.updated_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
        if not inserted:
            existing = self.get_tool_operation(operation.id)
            if existing != operation:
                raise ValueError("tool operation ID already exists with different content")
        return self.get_tool_operation(operation.id)

    def get_tool_operation(self, operation_id: str) -> ToolOperation:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tool_operations WHERE id = ?", (operation_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"tool operation not found: {operation_id}")
        return _decode_tool_operation(row)

    def list_tool_operations(self, source_message_id: str) -> list[ToolOperation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM tool_operations
                WHERE source_message_id = ?
                ORDER BY action_index ASC, id ASC
                """,
                (source_message_id,),
            ).fetchall()
        return [_decode_tool_operation(row) for row in rows]

    def update_tool_operation(self, operation: ToolOperation) -> ToolOperation:
        existing = self.get_tool_operation(operation.id)
        if existing.status == "undone":
            if operation == existing:
                return existing
            raise ValueError("an undone operation cannot change")
        if existing.status != "committed" or operation.status != "undone":
            raise ValueError("only a committed operation can transition to undone")
        if (
            operation.source_message_id != existing.source_message_id
            or operation.action_index != existing.action_index
            or operation.tool_name != existing.tool_name
            or operation.request != existing.request
            or operation.result != existing.result
            or operation.error != existing.error
        ):
            raise ValueError("undo cannot change operation content")
        with self._connect() as connection:
            connection.execute(
                "UPDATE tool_operations SET status = ?, updated_at = ? WHERE id = ?",
                (operation.status, operation.updated_at.isoformat(), operation.id),
            )
        return self.get_tool_operation(operation.id)

    def commit_prepared_calendar_event(
        self,
        operation_id: str,
        event: CalendarEvent,
        evidence: list[EvidenceReference] | None = None,
    ) -> tuple[ToolOperation, CalendarEvent]:
        event_evidence = evidence or [
            EvidenceReference(message_id=event.source_message_id, source_quote=event.source_quote)
        ]
        with self._connect() as connection:
            operation = self._prepared_operation(connection, operation_id, "create_local_event")
            result = {"event_id": event.id}
            if operation.status == "committed":
                return operation, self.get_calendar_event(event.id)
            if operation.status != "prepared":
                raise ValueError("only a prepared operation can be committed")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO calendar_events
                (id, workspace_id, conversation_id, title, description, source_message_id,
                 source_quote, start_at, end_at, all_day, timezone, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.workspace_id,
                    event.conversation_id,
                    event.title,
                    event.description,
                    event.source_message_id,
                    event.source_quote,
                    event.start_at.isoformat(),
                    None if event.end_at is None else event.end_at.isoformat(),
                    int(event.all_day),
                    event.timezone,
                    event.status,
                    event.created_at.isoformat(),
                    event.updated_at.isoformat(),
                ),
            )
            if cursor.rowcount == 0:
                existing = CalendarEvent(
                    **dict(
                        connection.execute(
                            "SELECT * FROM calendar_events WHERE id = ?", (event.id,)
                        ).fetchone()
                    )
                )
                if existing != event:
                    raise ValueError("calendar event ID already exists with different content")
            self._save_calendar_event_sources(connection, event, event_evidence)
            self._commit_operation(connection, operation, result)
        return self.get_tool_operation(operation_id), self.get_calendar_event(event.id)

    def commit_prepared_saved_item(
        self, operation_id: str, item: SavedItem
    ) -> tuple[ToolOperation, SavedItem]:
        with self._connect() as connection:
            operation = self._prepared_operation(connection, operation_id, f"save_{item.kind}")
            result = {"item_id": item.id}
            if operation.status == "committed":
                return operation, self.get_saved_item(item.id)
            if operation.status != "prepared":
                raise ValueError("only a prepared operation can be committed")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO saved_items
                (id, workspace_id, conversation_id, kind, content, source_message_id,
                 source_quote, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.workspace_id,
                    item.conversation_id,
                    item.kind,
                    item.content,
                    item.source_message_id,
                    item.source_quote,
                    item.status,
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            )
            if cursor.rowcount == 0:
                existing = SavedItem(
                    **dict(
                        connection.execute(
                            "SELECT * FROM saved_items WHERE id = ?", (item.id,)
                        ).fetchone()
                    )
                )
                if existing != item:
                    raise ValueError("saved item ID already exists with different content")
            self._commit_operation(connection, operation, result)
        return self.get_tool_operation(operation_id), self.get_saved_item(item.id)

    def _validate_project_event_source(
        self, record: ProjectRecord, source_message_id: str | None
    ) -> None:
        if source_message_id is None:
            return
        source = self._require_source_message(source_message_id)
        project = self.get_conversation_project(source.conversation_id)
        if project is None or project.id != record.project_id:
            raise ValueError("a project record event source must belong to the same project")

    @staticmethod
    def _project_change_operation_in_connection(
        connection: sqlite3.Connection, operation_id: str
    ) -> ProjectChangeOperation:
        row = connection.execute(
            "SELECT * FROM project_change_operations WHERE id = ?", (operation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"project change operation not found: {operation_id}")
        return _decode_project_change_operation(row)

    @staticmethod
    def _project_record_in_connection(
        connection: sqlite3.Connection, record_id: str | None
    ) -> ProjectRecord:
        if record_id is None:
            raise ValueError("project change requires a target record")
        row = connection.execute("SELECT * FROM project_records WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(f"project record not found: {record_id}")
        return ProjectRecord(**dict(row))

    @staticmethod
    def _transition_action(record: ProjectRecord, target_state: str | None) -> str:
        transition_actions = {
            ("task", "active", "completed"): "completed",
            ("open_question", "active", "resolved"): "resolved",
            ("blocker", "active", "resolved"): "resolved",
        }
        action = transition_actions.get((record.kind, record.state, target_state))
        if action is None:
            raise ValueError("invalid project record lifecycle transition")
        return action

    @staticmethod
    def _validate_project_change_target(
        record: ProjectRecord, project_id: str, target_record_id: str | None
    ) -> None:
        if record.id != target_record_id or record.project_id != project_id or record.state != "active":
            raise ValueError("project change target must be an active supplied record in the same project")

    @staticmethod
    def _write_project_record_due_at(
        connection: sqlite3.Connection, record_id: str, due_at: date | datetime | None
    ) -> None:
        connection.execute(
            "UPDATE project_records SET due_at = ?, updated_at = ? WHERE id = ?",
            (
                None if due_at is None else due_at.isoformat(),
                utc_now().isoformat(),
                record_id,
            ),
        )

    def _insert_project_record_in_connection(
        self, connection: sqlite3.Connection, record: ProjectRecord
    ) -> None:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO project_records
            (id, project_id, conversation_id, kind, content, rationale, due_at, state,
             source_message_id, source_quote, supersedes_record_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.project_id,
                record.conversation_id,
                record.kind,
                record.content,
                record.rationale,
                None if record.due_at is None else record.due_at.isoformat(),
                record.state,
                record.source_message_id,
                record.source_quote,
                record.supersedes_record_id,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
            ),
        )
        if cursor.rowcount == 0:
            existing = self._project_record_in_connection(connection, record.id)
            if existing != record:
                raise ValueError("project record ID already exists with different content")
            return
        self._save_project_record_event_in_connection(
            connection,
            ProjectRecordEvent(
                project_record_id=record.id,
                action="created",
                source_message_id=record.source_message_id,
                new_state=record.state,
                details={"origin": "project_change_operation"},
            ),
        )

    def _set_project_record_state_in_connection(
        self,
        connection: sqlite3.Connection,
        record: ProjectRecord,
        state: str,
        action: str,
        source_message_id: str,
        operation_id: str,
        details: dict | None = None,
    ) -> None:
        updated_at = utc_now()
        connection.execute(
            "UPDATE project_records SET state = ?, updated_at = ? WHERE id = ?",
            (state, updated_at.isoformat(), record.id),
        )
        self._save_project_record_event_in_connection(
            connection,
            ProjectRecordEvent(
                project_record_id=record.id,
                action=action,
                source_message_id=source_message_id,
                previous_state=record.state,
                new_state=state,
                details={"origin": "project_change_undo", "operation_id": operation_id, **(details or {})},
            ),
        )

    @staticmethod
    def _commit_project_change_operation(
        connection: sqlite3.Connection, operation: ProjectChangeOperation, result: dict
    ) -> None:
        connection.execute(
            """
            UPDATE project_change_operations
            SET result_json = ?, status = ?, updated_at = ?
            WHERE id = ? AND status = 'prepared'
            """,
            (json.dumps(result, sort_keys=True), "committed", utc_now().isoformat(), operation.id),
        )

    @staticmethod
    def _save_project_record_event_in_connection(
        connection: sqlite3.Connection, event: ProjectRecordEvent
    ) -> None:
        connection.execute(
            """
            INSERT INTO project_record_events
            (id, project_record_id, action, source_message_id, previous_state, new_state,
             details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.project_record_id,
                event.action,
                event.source_message_id,
                event.previous_state,
                event.new_state,
                json.dumps(event.details, sort_keys=True),
                event.created_at.isoformat(),
            ),
        )

    def _change_quote_source(self, change: ProjectChange, source: Message) -> Message:
        """The user message a tracker change quotes: the one it names, else its operation's."""

        if change.source_message_id is None or change.source_message_id == source.id:
            return source
        return self._require_source_message(change.source_message_id, source.conversation_id)

    @staticmethod
    def _require_source_quote(message: Message, source_quote: str | None) -> None:
        if not source_quote:
            raise ValueError("a project record needs a source quote")
        normalized_message = " ".join(message.content.split())
        normalized_quote = " ".join(source_quote.split())
        if normalized_quote not in normalized_message:
            raise ValueError("a project record source quote must occur in its user message")

    def _require_source_message(
        self, message_id: str, conversation_id: str | None = None
    ) -> Message:
        message = self.get_message(message_id)
        if message.role != "user":
            raise ValueError("a source message must have the user role")
        if conversation_id is not None and message.conversation_id != conversation_id:
            raise ValueError("a source message must belong to the target conversation")
        return message

    @staticmethod
    def _save_calendar_event_sources(
        connection: sqlite3.Connection,
        event: CalendarEvent,
        evidence: list[EvidenceReference],
    ) -> None:
        if not evidence:
            raise ValueError("a calendar event requires evidence")
        if len(evidence) > 3:
            raise ValueError("a calendar event supports at most three evidence references")
        if len({reference.message_id for reference in evidence}) != len(evidence):
            raise ValueError("calendar event evidence messages must be unique")
        for reference in evidence:
            message = CaptureStore._source_message_in_workspace(
                connection, reference.message_id, event.workspace_id
            )
            if message.role != "user":
                raise ValueError("calendar event evidence must reference user messages")
            CaptureStore._require_source_quote(message, reference.source_quote)
        connection.executemany(
            """
            INSERT INTO calendar_event_sources
            (event_id, source_message_id, source_quote, source_rank)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(event_id, source_message_id) DO NOTHING
            """,
            [
                (event.id, reference.message_id, reference.source_quote, rank)
                for rank, reference in enumerate(evidence)
            ],
        )

    @staticmethod
    def _calendar_event_in_connection(
        connection: sqlite3.Connection, event_id: str
    ) -> CalendarEvent:
        row = connection.execute("SELECT * FROM calendar_events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(f"calendar event not found: {event_id}")
        return CalendarEvent(**dict(row))

    @staticmethod
    def _calendar_event_update_operation_in_connection(
        connection: sqlite3.Connection, operation_id: str
    ) -> CalendarEventUpdateOperation:
        row = connection.execute(
            "SELECT * FROM calendar_event_update_operations WHERE id = ?", (operation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"calendar update operation not found: {operation_id}")
        return _decode_calendar_event_update_operation(row)

    @staticmethod
    def _source_message_in_workspace(
        connection: sqlite3.Connection, message_id: str, workspace_id: str
    ) -> Message:
        row = connection.execute(
            """
            SELECT messages.*
            FROM messages JOIN conversations ON conversations.id = messages.conversation_id
            WHERE messages.id = ? AND conversations.workspace_id = ?
            """,
            (message_id, workspace_id),
        ).fetchone()
        if row is None:
            raise ValueError("calendar update source is outside the intended workspace")
        return Message(**dict(row))

    @staticmethod
    def _write_calendar_event(connection: sqlite3.Connection, event: CalendarEvent) -> None:
        connection.execute(
            """
            UPDATE calendar_events
            SET title = ?, description = ?, start_at = ?, end_at = ?, all_day = ?, timezone = ?,
                status = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                event.title,
                event.description,
                event.start_at.isoformat(),
                None if event.end_at is None else event.end_at.isoformat(),
                int(event.all_day),
                event.timezone,
                event.status,
                event.updated_at.isoformat(),
                event.id,
            ),
        )

    @staticmethod
    def _set_calendar_event_update_status(
        connection: sqlite3.Connection,
        operation: CalendarEventUpdateOperation,
        status: str,
    ) -> None:
        connection.execute(
            """
            UPDATE calendar_event_update_operations
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, utc_now().isoformat(), operation.id),
        )

    @staticmethod
    def _prepared_operation(
        connection: sqlite3.Connection, operation_id: str, tool_name: str | None = None
    ) -> ToolOperation:
        row = connection.execute(
            "SELECT * FROM tool_operations WHERE id = ?", (operation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"tool operation not found: {operation_id}")
        operation = _decode_tool_operation(row)
        if tool_name is not None and operation.tool_name != tool_name:
            raise ValueError("operation tool does not match its target")
        return operation

    @staticmethod
    def _commit_operation(
        connection: sqlite3.Connection, operation: ToolOperation, result: dict
    ) -> None:
        committed = operation.model_copy(
            update={"status": "committed", "result": result, "updated_at": utc_now()}
        )
        connection.execute(
            """
            UPDATE tool_operations
            SET result_json = ?, status = ?, updated_at = ?
            WHERE id = ? AND status = 'prepared'
            """,
            (
                json.dumps(committed.result, sort_keys=True),
                committed.status,
                committed.updated_at.isoformat(),
                committed.id,
            ),
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


def _parse_due_at(value: str) -> date | datetime:
    # datetime.fromisoformat accepts a bare date, so keep an all-day deadline a date.
    return datetime.fromisoformat(value) if "T" in value else date.fromisoformat(value)


def _decode_tool_operation(row: sqlite3.Row) -> ToolOperation:
    payload = dict(row)
    payload["request"] = json.loads(payload.pop("request_json"))
    result_json = payload.pop("result_json")
    payload["result"] = None if result_json is None else json.loads(result_json)
    return ToolOperation(**payload)


def _decode_calendar_event_update_operation(row: sqlite3.Row) -> CalendarEventUpdateOperation:
    payload = dict(row)
    payload["before"] = CalendarEvent.model_validate_json(payload.pop("before_json"))
    payload["after"] = CalendarEvent.model_validate_json(payload.pop("after_json"))
    return CalendarEventUpdateOperation(**payload)


def _decode_project_record_event(row: sqlite3.Row) -> ProjectRecordEvent:
    payload = dict(row)
    payload["details"] = json.loads(payload.pop("details_json"))
    return ProjectRecordEvent(**payload)


def _decode_project_change_operation(row: sqlite3.Row) -> ProjectChangeOperation:
    payload = dict(row)
    payload["change"] = json.loads(payload.pop("change_json"))
    result_json = payload.pop("result_json")
    payload["result"] = None if result_json is None else json.loads(result_json)
    return ProjectChangeOperation(**payload)
