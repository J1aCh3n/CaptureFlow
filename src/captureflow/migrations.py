import sqlite3
from dataclasses import dataclass

from .models import utc_now


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


MIGRATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
)
"""


MIGRATIONS = (
    Migration(
        version=1,
        name="existing_captureflow_schema",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id),
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS captures (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id)
                    ON DELETE CASCADE,
                source_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS capture_runs (
                id TEXT PRIMARY KEY,
                capture_id TEXT NOT NULL REFERENCES captures(id)
                    ON DELETE CASCADE,
                status TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
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
            )
            """,
            """
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
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_reviews (
                id TEXT PRIMARY KEY,
                capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                review_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT REFERENCES memory_items(id),
                action TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                run_id TEXT PRIMARY KEY,
                capture_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                phase TEXT NOT NULL,
                state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
        ),
    ),
    Migration(
        version=2,
        name="v2_chat_calendar_and_saved_items",
        statements=(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                reply_to_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE calendar_events (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id),
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                source_quote TEXT NOT NULL,
                start_at TEXT NOT NULL,
                end_at TEXT,
                all_day INTEGER NOT NULL CHECK (all_day IN (0, 1)),
                timezone TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('active', 'deleted')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE saved_items (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id),
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                kind TEXT NOT NULL CHECK (kind IN ('idea', 'note')),
                content TEXT NOT NULL,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                source_quote TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('active', 'deleted')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE tool_operations (
                id TEXT PRIMARY KEY,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                action_index INTEGER NOT NULL CHECK (action_index BETWEEN 0 AND 2),
                tool_name TEXT NOT NULL CHECK (
                    tool_name IN ('create_local_event', 'save_idea', 'save_note')
                ),
                request_json TEXT NOT NULL,
                result_json TEXT,
                status TEXT NOT NULL CHECK (
                    status IN ('prepared', 'committed', 'failed', 'undone')
                ),
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (source_message_id, action_index, tool_name)
            )
            """,
            """
            CREATE TABLE pending_clarifications (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                original_action_json TEXT NOT NULL,
                missing_field TEXT NOT NULL CHECK (missing_field = 'date'),
                status TEXT NOT NULL CHECK (status IN ('pending', 'resolved', 'cancelled')),
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """,
            "CREATE INDEX messages_conversation_order ON messages(conversation_id, created_at, id)",
            "CREATE INDEX active_events_order ON calendar_events(status, start_at, id)",
            "CREATE INDEX saved_items_filter ON saved_items(kind, status, updated_at, id)",
            "CREATE INDEX operations_message_status ON tool_operations(source_message_id, status)",
            "CREATE INDEX clarifications_conversation_status ON pending_clarifications(conversation_id, status)",
            """
            CREATE UNIQUE INDEX one_pending_clarification_per_conversation
            ON pending_clarifications(conversation_id)
            WHERE status = 'pending'
            """,
        ),
    ),
    Migration(
        version=3,
        name="v2_project_tracker_foundation",
        statements=(
            """
            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id),
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE conversation_projects (
                conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                project_id TEXT NOT NULL REFERENCES projects(id),
                assigned_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE project_records (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id),
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                kind TEXT NOT NULL CHECK (
                    kind IN ('task', 'milestone', 'decision', 'constraint',
                             'open_question', 'blocker')
                ),
                content TEXT NOT NULL,
                rationale TEXT,
                state TEXT NOT NULL CHECK (
                    state IN ('active', 'completed', 'resolved', 'superseded', 'deleted')
                ),
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                source_quote TEXT NOT NULL,
                supersedes_record_id TEXT REFERENCES project_records(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE project_record_events (
                id TEXT PRIMARY KEY,
                project_record_id TEXT NOT NULL REFERENCES project_records(id) ON DELETE CASCADE,
                action TEXT NOT NULL CHECK (
                    action IN ('created', 'completed', 'resolved', 'superseded',
                               'edited', 'deleted', 'restored')
                ),
                source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                previous_state TEXT,
                new_state TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX projects_workspace_status ON projects(workspace_id, status, updated_at, id)",
            "CREATE INDEX conversation_projects_project ON conversation_projects(project_id, conversation_id)",
            "CREATE INDEX project_records_filter ON project_records(project_id, state, kind, updated_at, id)",
            "CREATE INDEX project_record_events_order ON project_record_events(project_record_id, created_at, id)",
        ),
    ),
    Migration(
        version=4,
        name="v2_project_conversation_state",
        statements=(
            """
            CREATE TABLE project_conversation_state (
                conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                project_id TEXT NOT NULL REFERENCES projects(id),
                started_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX project_conversation_state_project ON project_conversation_state(project_id, conversation_id)",
        ),
    ),
    Migration(
        version=5,
        name="v2_project_change_operations",
        statements=(
            """
            CREATE TABLE project_change_operations (
                id TEXT PRIMARY KEY,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                project_id TEXT NOT NULL REFERENCES projects(id),
                change_index INTEGER NOT NULL CHECK (change_index BETWEEN 0 AND 2),
                change_json TEXT NOT NULL,
                target_record_id TEXT REFERENCES project_records(id) ON DELETE SET NULL,
                result_json TEXT,
                status TEXT NOT NULL CHECK (
                    status IN ('prepared', 'committed', 'review', 'failed', 'undone')
                ),
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (source_message_id, change_index)
            )
            """,
            "CREATE INDEX project_change_operations_message ON project_change_operations(source_message_id, change_index)",
            "CREATE INDEX project_change_operations_project_status ON project_change_operations(project_id, status, updated_at, id)",
        ),
    ),
    Migration(
        version=6,
        name="v2_project_answer_sources",
        statements=(
            """
            CREATE TABLE project_answer_sources (
                assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                project_record_id TEXT NOT NULL REFERENCES project_records(id) ON DELETE CASCADE,
                source_rank INTEGER NOT NULL CHECK (source_rank BETWEEN 0 AND 5),
                PRIMARY KEY (assistant_message_id, project_record_id),
                UNIQUE (assistant_message_id, source_rank)
            )
            """,
            "CREATE INDEX project_answer_sources_record ON project_answer_sources(project_record_id, assistant_message_id)",
        ),
    ),
    Migration(
        version=7,
        name="v2_local_context_answer_sources",
        statements=(
            """
            CREATE TABLE local_context_answer_sources (
                assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                source_type TEXT NOT NULL CHECK (source_type IN ('calendar_event', 'saved_item')),
                source_id TEXT NOT NULL,
                source_rank INTEGER NOT NULL CHECK (source_rank BETWEEN 0 AND 5),
                PRIMARY KEY (assistant_message_id, source_type, source_id),
                UNIQUE (assistant_message_id, source_rank)
            )
            """,
            "CREATE INDEX local_context_answer_sources_lookup ON local_context_answer_sources(source_type, source_id, assistant_message_id)",
        ),
    ),
    Migration(
        version=8,
        name="v2_calendar_event_descriptions",
        statements=("ALTER TABLE calendar_events ADD COLUMN description TEXT",),
    ),
    Migration(
        version=9,
        name="v2_calendar_event_update_operations",
        statements=(
            """
            CREATE TABLE calendar_event_update_operations (
                id TEXT PRIMARY KEY,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                action_index INTEGER NOT NULL CHECK (action_index BETWEEN 0 AND 2),
                event_id TEXT NOT NULL REFERENCES calendar_events(id) ON DELETE CASCADE,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id),
                before_json TEXT NOT NULL,
                after_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('prepared', 'committed', 'failed', 'undone')
                ),
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (source_message_id, action_index, event_id)
            )
            """,
            """
            CREATE TABLE calendar_event_update_sources (
                update_operation_id TEXT NOT NULL
                    REFERENCES calendar_event_update_operations(id) ON DELETE CASCADE,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                source_quote TEXT NOT NULL,
                source_rank INTEGER NOT NULL CHECK (source_rank BETWEEN 0 AND 2),
                PRIMARY KEY (update_operation_id, source_message_id),
                UNIQUE (update_operation_id, source_rank)
            )
            """,
            "CREATE INDEX calendar_event_update_operations_event_status ON calendar_event_update_operations(event_id, status, updated_at, id)",
            "CREATE INDEX calendar_event_update_operations_message ON calendar_event_update_operations(source_message_id, action_index)",
        ),
    ),
    Migration(
        version=10,
        name="v2_calendar_event_sources",
        statements=(
            """
            CREATE TABLE calendar_event_sources (
                event_id TEXT NOT NULL REFERENCES calendar_events(id) ON DELETE CASCADE,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                source_quote TEXT NOT NULL,
                source_rank INTEGER NOT NULL CHECK (source_rank BETWEEN 0 AND 2),
                PRIMARY KEY (event_id, source_message_id),
                UNIQUE (event_id, source_rank)
            )
            """,
            """
            INSERT INTO calendar_event_sources
            (event_id, source_message_id, source_quote, source_rank)
            SELECT id, source_message_id, source_quote, 0
            FROM calendar_events
            """,
            "CREATE INDEX calendar_event_sources_message ON calendar_event_sources(source_message_id, event_id)",
        ),
    ),
    Migration(
        version=11,
        name="v2_calendar_clarification_fields",
        statements=(
            "DROP INDEX clarifications_conversation_status",
            "DROP INDEX one_pending_clarification_per_conversation",
            "ALTER TABLE pending_clarifications RENAME TO pending_clarifications_legacy",
            """
            CREATE TABLE pending_clarifications (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                original_action_json TEXT NOT NULL,
                missing_field TEXT NOT NULL CHECK (missing_field IN ('date', 'time', 'duration')),
                status TEXT NOT NULL CHECK (status IN ('pending', 'resolved', 'cancelled')),
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """,
            """
            INSERT INTO pending_clarifications
            (id, conversation_id, source_message_id, assistant_message_id,
             original_action_json, missing_field, status, created_at, resolved_at)
            SELECT id, conversation_id, source_message_id, assistant_message_id,
                   original_action_json, missing_field, status, created_at, resolved_at
            FROM pending_clarifications_legacy
            """,
            "DROP TABLE pending_clarifications_legacy",
            "CREATE INDEX clarifications_conversation_status ON pending_clarifications(conversation_id, status)",
            """
            CREATE UNIQUE INDEX one_pending_clarification_per_conversation
            ON pending_clarifications(conversation_id)
            WHERE status = 'pending'
            """,
        ),
    ),
    Migration(
        version=12,
        name="v2_follow_up_clarification_field",
        statements=(
            "DROP INDEX clarifications_conversation_status",
            "DROP INDEX one_pending_clarification_per_conversation",
            "ALTER TABLE pending_clarifications RENAME TO pending_clarifications_legacy",
            """
            CREATE TABLE pending_clarifications (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                original_action_json TEXT NOT NULL,
                missing_field TEXT NOT NULL
                    CHECK (missing_field IN ('date', 'time', 'duration', 'follow_up')),
                status TEXT NOT NULL CHECK (status IN ('pending', 'resolved', 'cancelled')),
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """,
            """
            INSERT INTO pending_clarifications
            (id, conversation_id, source_message_id, assistant_message_id,
             original_action_json, missing_field, status, created_at, resolved_at)
            SELECT id, conversation_id, source_message_id, assistant_message_id,
                   original_action_json, missing_field, status, created_at, resolved_at
            FROM pending_clarifications_legacy
            """,
            "DROP TABLE pending_clarifications_legacy",
            "CREATE INDEX clarifications_conversation_status ON pending_clarifications(conversation_id, status)",
            """
            CREATE UNIQUE INDEX one_pending_clarification_per_conversation
            ON pending_clarifications(conversation_id)
            WHERE status = 'pending'
            """,
        ),
    ),
    Migration(
        version=13,
        name="v2_project_record_due_at",
        statements=(
            "ALTER TABLE project_records ADD COLUMN due_at TEXT",
            "CREATE INDEX project_records_due ON project_records(due_at, state, project_id)",
        ),
    ),
    Migration(
        version=14,
        name="v2_timezone_clarification_field",
        statements=(
            "DROP INDEX clarifications_conversation_status",
            "DROP INDEX one_pending_clarification_per_conversation",
            "ALTER TABLE pending_clarifications RENAME TO pending_clarifications_legacy",
            """
            CREATE TABLE pending_clarifications (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                original_action_json TEXT NOT NULL,
                missing_field TEXT NOT NULL CHECK (
                    missing_field IN ('date', 'time', 'duration', 'timezone', 'follow_up')
                ),
                status TEXT NOT NULL CHECK (status IN ('pending', 'resolved', 'cancelled')),
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """,
            """
            INSERT INTO pending_clarifications
            (id, conversation_id, source_message_id, assistant_message_id,
             original_action_json, missing_field, status, created_at, resolved_at)
            SELECT id, conversation_id, source_message_id, assistant_message_id,
                   original_action_json, missing_field, status, created_at, resolved_at
            FROM pending_clarifications_legacy
            """,
            "DROP TABLE pending_clarifications_legacy",
            "CREATE INDEX clarifications_conversation_status ON pending_clarifications(conversation_id, status)",
            """
            CREATE UNIQUE INDEX one_pending_clarification_per_conversation
            ON pending_clarifications(conversation_id)
            WHERE status = 'pending'
            """,
        ),
    ),
    Migration(
        version=15,
        name="v2_recurrence_clarification_field",
        statements=(
            "DROP INDEX clarifications_conversation_status",
            "DROP INDEX one_pending_clarification_per_conversation",
            "ALTER TABLE pending_clarifications RENAME TO pending_clarifications_legacy",
            """
            CREATE TABLE pending_clarifications (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                original_action_json TEXT NOT NULL,
                missing_field TEXT NOT NULL CHECK (
                    missing_field IN
                        ('date', 'time', 'duration', 'timezone', 'recurrence', 'follow_up')
                ),
                status TEXT NOT NULL CHECK (status IN ('pending', 'resolved', 'cancelled')),
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """,
            """
            INSERT INTO pending_clarifications
            (id, conversation_id, source_message_id, assistant_message_id,
             original_action_json, missing_field, status, created_at, resolved_at)
            SELECT id, conversation_id, source_message_id, assistant_message_id,
                   original_action_json, missing_field, status, created_at, resolved_at
            FROM pending_clarifications_legacy
            """,
            "DROP TABLE pending_clarifications_legacy",
            "CREATE INDEX clarifications_conversation_status ON pending_clarifications(conversation_id, status)",
            """
            CREATE UNIQUE INDEX one_pending_clarification_per_conversation
            ON pending_clarifications(conversation_id)
            WHERE status = 'pending'
            """,
        ),
    ),
)

LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version


def apply_migrations(
    connection: sqlite3.Connection,
    target_version: int = LATEST_SCHEMA_VERSION,
) -> None:
    if not 1 <= target_version <= LATEST_SCHEMA_VERSION:
        raise ValueError(f"target schema version must be between 1 and {LATEST_SCHEMA_VERSION}")

    connection.execute(MIGRATION_TABLE_SQL)
    connection.commit()
    applied_rows = connection.execute("SELECT version FROM schema_migrations").fetchall()
    applied = {row[0] for row in applied_rows}
    unsupported = applied - {migration.version for migration in MIGRATIONS}
    if unsupported:
        raise RuntimeError(f"database contains unsupported schema versions: {sorted(unsupported)}")

    for migration in MIGRATIONS:
        if migration.version > target_version:
            break
        if migration.version not in applied:
            _apply_migration(connection, migration)


def _apply_migration(connection: sqlite3.Connection, migration: Migration) -> None:
    try:
        connection.execute("BEGIN IMMEDIATE")
        for statement in migration.statements:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (migration.version, migration.name, utc_now().isoformat()),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
