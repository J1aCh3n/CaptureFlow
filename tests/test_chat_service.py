import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from captureflow.chat_service import DeterministicChatService
from captureflow.models import CalendarEventUpdateOperation, Conversation, EvidenceReference
from captureflow.storage import CaptureStore


class DeterministicChatServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.conversation = self.store.save_conversation(
            Conversation(id="chat-conversation", title="Chat")
        )
        self.service = DeterministicChatService(self.store)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_event_command_is_idempotent_and_persists_a_timeline(self) -> None:
        first = self.service.submit(
            self.conversation.id,
            "/event 2026-09-15 | Submit report",
            message_id="event-message",
        )
        repeated = self.service.submit(
            self.conversation.id,
            "/event 2026-09-15 | Submit report",
            message_id="event-message",
        )

        self.assertEqual(first, repeated)
        self.assertEqual(len(self.store.list_messages(self.conversation.id)), 2)
        self.assertEqual(len(self.store.list_calendar_events()), 1)
        self.assertEqual(first.operation.status, "committed")
        self.assertIn("Created local calendar event", first.assistant_message.content)

    def test_idea_note_and_plain_message_have_explicit_outcomes(self) -> None:
        idea = self.service.submit(
            self.conversation.id, "/idea Test the import path.", message_id="idea-message"
        )
        note = self.service.submit(
            self.conversation.id, "/note Keep SQLite.", message_id="note-message"
        )
        plain = self.service.submit(
            self.conversation.id, "hello", message_id="plain-message"
        )

        self.assertEqual([item.kind for item in self.store.list_saved_items()], ["note", "idea"])
        self.assertEqual(idea.operation.tool_name, "save_idea")
        self.assertEqual(note.operation.tool_name, "save_note")
        self.assertIsNone(plain.operation)
        self.assertIn("No local action", plain.assistant_message.content)

    def test_invalid_event_creates_no_target_and_records_a_reply(self) -> None:
        turn = self.service.submit(
            self.conversation.id, "/event next Tuesday | Submit report", message_id="invalid-event"
        )

        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(self.store.list_tool_operations(turn.user_message.id), [])
        self.assertEqual(turn.assistant_message.reply_to_message_id, turn.user_message.id)
        self.assertIn("YYYY-MM-DD", turn.assistant_message.content)

    def test_undo_is_idempotent_and_keeps_the_original_event(self) -> None:
        turn = self.service.submit(
            self.conversation.id, "/event 2026-09-15 | Submit report", message_id="undo-message"
        )

        undone = self.service.undo(turn.operation.id)
        repeated = self.service.undo(turn.operation.id)
        event_id = undone.result["event_id"]

        self.assertEqual(undone, repeated)
        self.assertEqual(undone.status, "undone")
        self.assertEqual(self.store.get_calendar_event(event_id).status, "deleted")

    def test_undo_restores_a_committed_calendar_update(self) -> None:
        created = self.service.submit(
            self.conversation.id,
            "/event 2026-09-15 | Submit report",
            message_id="update-create-message",
        )
        correction = self.service.submit(
            self.conversation.id,
            "Move the report to September 16.",
            message_id="update-correction-message",
        )
        event = self.store.get_calendar_event(created.operation.result["event_id"])
        update = CalendarEventUpdateOperation(
            id="calendar-update-undo",
            source_message_id=correction.user_message.id,
            action_index=0,
            event_id=event.id,
            before=event,
            after=event.model_copy(
                update={
                    "start_at": date(2026, 9, 16),
                    "updated_at": datetime(2026, 9, 9, tzinfo=timezone.utc),
                }
            ),
            status="prepared",
        )
        self.store.prepare_calendar_event_update(
            update,
            [
                EvidenceReference(
                    message_id=correction.user_message.id,
                    source_quote="Move the report",
                )
            ],
        )
        self.store.commit_prepared_calendar_event_update(update.id)

        undone = self.service.undo(update.id)

        self.assertEqual(undone.status, "undone")
        self.assertEqual(self.store.get_calendar_event(event.id), event)
