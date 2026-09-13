import tempfile
import unittest
from pathlib import Path

from captureflow.models import Conversation
from captureflow.storage import CaptureStore


class ConversationStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_repeated_save_with_same_conversation_id_does_not_duplicate(self) -> None:
        conversation = Conversation(id="conversation-1", title="Inbox")

        self.store.save_conversation(conversation)
        self.store.save_conversation(conversation)

        self.assertEqual(self.store.list_conversations(), [conversation])

    def test_delete_conversation_clears_rows_left_by_the_retired_capture_workflow(self) -> None:
        conversation = self.store.save_conversation(
            Conversation(id="conversation-delete", title="Delete me")
        )
        with self.store._connect() as connection:
            connection.execute(
                "INSERT INTO captures (id, conversation_id, source_text, created_at) "
                "VALUES ('capture-1', ?, 'I prefer concise notes.', '2026-09-06')",
                (conversation.id,),
            )
            connection.execute(
                "INSERT INTO memory_items (id, workspace_id, memory_type, content, source_capture_id, "
                "source_quote, write_mode, status, confidence, created_at, updated_at) VALUES "
                "('memory-1', 'default', 'preference', 'Concise notes.', 'capture-1', "
                "'I prefer concise notes.', 'auto', 'active', 1, '2026-09-06', '2026-09-06')"
            )
            connection.execute(
                "INSERT INTO memory_events (memory_id, action, details_json, created_at) "
                "VALUES ('memory-1', 'created_auto', '{}', '2026-09-06')"
            )

        self.assertTrue(self.store.delete_conversation(conversation.id))
        self.assertFalse(self.store.delete_conversation(conversation.id))

        with self.store._connect() as connection:
            for table in ("captures", "memory_items", "memory_events"):
                with self.subTest(table=table):
                    count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    self.assertEqual(count, 0)
        with self.assertRaises(KeyError):
            self.store.get_conversation(conversation.id)
