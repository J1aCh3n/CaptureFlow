import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from captureflow.local_context_service import LocalContextService, is_local_context_query
from captureflow.models import CalendarEvent, Conversation, Message, SavedItem
from captureflow.storage import CaptureStore


class LocalContextServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.first = self.store.save_conversation(Conversation(id="meeting-source", title="Meeting"))
        self.later = self.store.save_conversation(Conversation(id="meeting-later", title="Later"))
        fixture_path = Path(__file__).parent / "fixtures" / "phase9_local_context_cases.json"
        self.fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        meeting = self.fixture["meeting"]
        source = self.store.save_message(
            Message(
                id="meeting-source-message",
                conversation_id=self.first.id,
                role="user",
                content="Tomorrow at 15:00 is the workflow meeting. The meeting uses voting and idea merging.",
            )
        )
        self.event = self.store.save_calendar_event(
            CalendarEvent(
                id=meeting["event_id"],
                conversation_id=self.first.id,
                title=meeting["title"],
                source_message_id=source.id,
                source_quote=source.content,
                start_at=date.fromisoformat(meeting["date"]),
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        self.note = self.store.save_saved_item(
            SavedItem(
                id=meeting["note_id"],
                conversation_id=self.first.id,
                kind="note",
                content=meeting["note"],
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        self.service = LocalContextService(
            self.store, reference_date=date(2026, 9, 8), timezone="America/Vancouver"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_cross_conversation_meeting_query_returns_event_and_note_sources(self) -> None:
        turn = self.service.submit(self.later.id, self.fixture["meeting"]["query"])

        self.assertEqual(turn.outcome, "grounded_local_context")
        self.assertEqual({source.id for source in turn.sources}, {self.event.id, self.note.id})
        self.assertIn(self.fixture["required_reply_fragments"][0], turn.assistant_message.content)
        self.assertIn(self.note.content, turn.assistant_message.content)
        self.assertIn(self.fixture["required_reply_fragments"][1], turn.assistant_message.content)
        self.assertEqual(
            [source.reference.source_id for source in self.store.list_local_context_answer_sources(turn.assistant_message.id)],
            [self.event.id, self.note.id],
        )

    def test_deleted_event_and_unrelated_notes_are_not_returned(self) -> None:
        unrelated = self.store.save_saved_item(
            SavedItem(
                id="unrelated-note",
                conversation_id=self.first.id,
                kind="note",
                content="Buy groceries after work.",
                source_message_id="meeting-source-message",
                source_quote="Tomorrow at 15:00 is the workflow meeting. The meeting uses voting and idea merging.",
            )
        )
        self.store.set_calendar_event_status(self.event.id, "deleted")

        turn = self.service.submit(self.later.id, "明天会议的内容是什么？")

        self.assertEqual(turn.outcome, "grounded_local_context")
        self.assertEqual([source.id for source in turn.sources], [self.note.id])
        self.assertNotIn(unrelated.content, turn.assistant_message.content)

    def test_missing_local_context_returns_unknown_without_a_model(self) -> None:
        turn = self.service.submit(self.later.id, "What is the calendar agenda for Friday?")

        self.assertEqual(turn.outcome, "insufficient_context")
        self.assertIn("do not have matching local", turn.assistant_message.content)
        self.assertEqual(self.store.list_local_context_answer_sources(turn.assistant_message.id), [])

    def test_query_classifier_is_narrow_and_deterministic(self) -> None:
        self.assertTrue(is_local_context_query("我明天的会议要做什么？"))
        self.assertTrue(is_local_context_query("When is the calendar meeting?"))
        self.assertFalse(is_local_context_query("Track a meeting task."))
