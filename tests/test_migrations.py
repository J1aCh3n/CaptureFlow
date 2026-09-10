import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from captureflow.migrations import (
    LATEST_SCHEMA_VERSION,
    MIGRATION_TABLE_SQL,
    Migration,
    _apply_migration,
    apply_migrations,
)
from captureflow.storage import CaptureStore


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "captureflow.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_clean_database_initializes_to_latest_version(self) -> None:
        store = CaptureStore(self.database_path)
        store.initialize()

        with closing(sqlite3.connect(self.database_path)) as connection:
            versions = connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            calendar_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(calendar_events)")
            }

        self.assertEqual(
            versions,
            [(version,) for version in range(1, LATEST_SCHEMA_VERSION + 1)],
        )
        self.assertTrue(
            {
                "messages",
                "calendar_events",
                "saved_items",
                "tool_operations",
                "pending_clarifications",
                "projects",
                "conversation_projects",
                "project_records",
                "project_record_events",
                "project_conversation_state",
                "project_change_operations",
                "project_answer_sources",
                "local_context_answer_sources",
                "calendar_event_update_operations",
                "calendar_event_update_sources",
                "calendar_event_sources",
            }.issubset(tables)
        )
        self.assertIn("description", calendar_columns)

    def test_existing_v1_database_upgrades_without_losing_capture(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            apply_migrations(connection, target_version=1)
            connection.execute(
                "INSERT INTO workspaces (id, created_at) VALUES ('default', '2026-09-08')"
            )
            connection.execute(
                """
                INSERT INTO conversations (id, workspace_id, title, created_at)
                VALUES ('conversation-1', 'default', 'Legacy', '2026-09-08')
                """
            )
            connection.execute(
                """
                INSERT INTO captures (id, conversation_id, source_text, created_at)
                VALUES ('capture-1', 'conversation-1', 'Legacy capture', '2026-09-08')
                """
            )
            connection.commit()

        store = CaptureStore(self.database_path)
        store.initialize()

        self.assertEqual(store.get_capture("capture-1").source_text, "Legacy capture")
        with closing(sqlite3.connect(self.database_path)) as connection:
            versions = connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        self.assertEqual(versions, [(version,) for version in range(1, LATEST_SCHEMA_VERSION + 1)])

    def test_existing_v7_event_upgrades_with_an_empty_description(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            apply_migrations(connection, target_version=7)
            connection.execute(
                "INSERT INTO workspaces (id, created_at) VALUES ('default', '2026-09-08')"
            )
            connection.execute(
                """
                INSERT INTO conversations (id, workspace_id, title, created_at)
                VALUES ('conversation-1', 'default', 'Calendar', '2026-09-08')
                """
            )
            connection.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at)
                VALUES ('message-1', 'conversation-1', 'user', 'Meeting', '2026-09-08')
                """
            )
            connection.execute(
                """
                INSERT INTO calendar_events
                (id, workspace_id, conversation_id, title, source_message_id, source_quote,
                 start_at, end_at, all_day, timezone, status, created_at, updated_at)
                VALUES ('event-1', 'default', 'conversation-1', 'Meeting', 'message-1',
                        'Meeting', '2026-09-11', NULL, 1, 'America/Vancouver', 'active',
                        '2026-09-08', '2026-09-08')
                """
            )
            connection.commit()
            apply_migrations(connection)
            event = connection.execute(
                "SELECT title, description FROM calendar_events WHERE id = 'event-1'"
            ).fetchone()
            sources = connection.execute(
                "SELECT source_message_id, source_quote, source_rank FROM calendar_event_sources WHERE event_id = 'event-1'"
            ).fetchall()

        self.assertEqual(event, ("Meeting", None))
        self.assertEqual(sources, [("message-1", "Meeting", 0)])

    def test_existing_date_clarification_is_preserved_when_fields_expand(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            apply_migrations(connection, target_version=10)
            connection.execute(
                "INSERT INTO workspaces (id, created_at) VALUES ('default', '2026-09-08')"
            )
            connection.execute(
                """
                INSERT INTO conversations (id, workspace_id, title, created_at)
                VALUES ('conversation-1', 'default', 'Calendar', '2026-09-08')
                """
            )
            connection.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at)
                VALUES ('user-1', 'conversation-1', 'user', 'Meeting', '2026-09-08'),
                       ('assistant-1', 'conversation-1', 'assistant', 'Which date?', '2026-09-08')
                """
            )
            connection.execute(
                """
                INSERT INTO pending_clarifications
                (id, conversation_id, source_message_id, assistant_message_id,
                 original_action_json, missing_field, status, created_at, resolved_at)
                VALUES ('clarification-1', 'conversation-1', 'user-1', 'assistant-1',
                        '{}', 'date', 'pending', '2026-09-08', NULL)
                """
            )
            connection.commit()
            apply_migrations(connection)
            clarification = connection.execute(
                "SELECT missing_field, status FROM pending_clarifications WHERE id = 'clarification-1'"
            ).fetchone()

        self.assertEqual(clarification, ("date", "pending"))

    def test_repeated_initialize_does_not_repeat_migrations(self) -> None:
        store = CaptureStore(self.database_path)
        store.initialize()
        store.initialize()

        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT version, COUNT(*) FROM schema_migrations GROUP BY version"
            ).fetchall()
        self.assertEqual(rows, [(version, 1) for version in range(1, LATEST_SCHEMA_VERSION + 1)])

    def test_failed_migration_rolls_back_schema_and_version(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(MIGRATION_TABLE_SQL)
            connection.commit()
            broken = Migration(
                version=99,
                name="broken",
                statements=(
                    "CREATE TABLE partial_table (id TEXT PRIMARY KEY)",
                    "THIS IS NOT SQL",
                ),
            )

            with self.assertRaises(sqlite3.OperationalError):
                _apply_migration(connection, broken)

            partial = connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'partial_table'"
            ).fetchone()
            version = connection.execute(
                "SELECT version FROM schema_migrations WHERE version = 99"
            ).fetchone()

        self.assertIsNone(partial)
        self.assertIsNone(version)
