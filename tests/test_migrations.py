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

        self.assertEqual(versions, [(1,), (2,), (3,), (4,), (5,), (6,), (LATEST_SCHEMA_VERSION,)])
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
            }.issubset(tables)
        )

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
        self.assertEqual(versions, [(1,), (2,), (3,), (4,), (5,), (6,), (7,)])

    def test_repeated_initialize_does_not_repeat_migrations(self) -> None:
        store = CaptureStore(self.database_path)
        store.initialize()
        store.initialize()

        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT version, COUNT(*) FROM schema_migrations GROUP BY version"
            ).fetchall()
        self.assertEqual(rows, [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)])

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
