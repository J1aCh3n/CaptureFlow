import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from captureflow.models import Capture, CaptureRun, Conversation
from captureflow.storage import CaptureStore


class CaptureStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "captureflow.sqlite3"
        self.store = CaptureStore(self.database_path)
        self.store.initialize()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_saved_capture_is_available_after_store_restart(self) -> None:
        created_at = datetime(2026, 9, 6, tzinfo=timezone.utc)
        conversation = Conversation(id="conversation-1", title="Inbox", created_at=created_at)
        capture = Capture(
            id="capture-1",
            conversation_id=conversation.id,
            source_text="Read the project brief.",
            created_at=created_at,
        )
        self.store.save_conversation(conversation)
        self.store.save_capture(capture)

        restarted_store = CaptureStore(self.database_path)
        restarted_store.initialize()

        self.assertEqual(restarted_store.get_capture(capture.id), capture)
        self.assertEqual(restarted_store.list_captures(conversation.id), [capture])

    def test_repeated_save_with_same_capture_id_does_not_duplicate(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="conversation-1", title="Inbox"))
        capture = Capture(
            id="capture-1",
            conversation_id=conversation.id,
            source_text="Read the project brief.",
        )

        self.store.save_capture(capture)
        self.store.save_capture(capture)

        self.assertEqual(self.store.list_captures(conversation.id), [capture])

    def test_repeated_save_with_same_conversation_id_does_not_duplicate(self) -> None:
        conversation = Conversation(id="conversation-1", title="Inbox")

        self.store.save_conversation(conversation)
        self.store.save_conversation(conversation)

        self.assertEqual(self.store.list_conversations(), [conversation])

    def test_repeated_delete_is_safe(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="conversation-1", title="Inbox"))
        capture = self.store.save_capture(
            Capture(
                id="capture-1",
                conversation_id=conversation.id,
                source_text="Read the project brief.",
            )
        )

        self.assertTrue(self.store.delete_capture(capture.id))
        self.assertFalse(self.store.delete_capture(capture.id))
        self.assertEqual(self.store.list_captures(conversation.id), [])

    def test_run_trace_survives_store_restart(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="conversation-1", title="Inbox"))
        capture = self.store.save_capture(
            Capture(id="capture-1", conversation_id=conversation.id, source_text="Read the brief.")
        )
        run = CaptureRun(id="run-1", capture_id=capture.id, status="completed", model_calls=0)
        self.store.save_run(run)

        restarted_store = CaptureStore(self.database_path)
        restarted_store.initialize()

        self.assertEqual(restarted_store.latest_run_for_capture(capture.id), run)
