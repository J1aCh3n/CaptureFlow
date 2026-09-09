import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from captureflow.llm import ModelError
from captureflow.local_context_service import LocalContextService, is_local_context_query
from captureflow.models import CalendarEvent, Conversation, Message, SavedItem
from captureflow.storage import CaptureStore


class ScriptedLocalContextModel:
    def __init__(self, decisions, repairs=()):
        self.decisions = list(decisions)
        self.repairs = list(repairs)
        self.calls = []

    def answer_local_context(self, user_message, runtime_context, events, saved_items):
        self.calls.append(("answer", runtime_context, events, saved_items))
        candidate = self.decisions.pop(0)
        if isinstance(candidate, Exception):
            raise candidate
        return candidate

    def repair_local_context_answer(
        self, user_message, runtime_context, events, saved_items, invalid_decision, validation_error
    ):
        self.calls.append(("repair", runtime_context, events, saved_items, invalid_decision))
        candidate = self.repairs.pop(0)
        if isinstance(candidate, Exception):
            raise candidate
        return candidate


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

    def test_model_synthesizes_only_selected_local_sources_and_python_renders_time(self) -> None:
        model = ScriptedLocalContextModel(
            [
                {
                    "outcome": "answer",
                    "selected_event_ids": [self.event.id],
                    "selected_item_ids": [self.note.id],
                    "summary_points": [
                        {
                            "text": "The meeting uses voting and idea merging.",
                            "source_ids": [self.note.id],
                        }
                    ],
                    "preparation_suggestions": [
                        {
                            "text": "Bring the existing ideas ready for voting.",
                            "based_on_source_ids": [self.note.id],
                        }
                    ],
                    "missing_information": ["preparation"],
                    "clarification_question": None,
                }
            ]
        )
        service = LocalContextService(
            self.store,
            reference_date=date(2026, 9, 8),
            timezone="America/Vancouver",
            model=model,
        )

        turn = service.submit(self.later.id, "What time is my meeting tomorrow and what should I prepare?")

        self.assertEqual(turn.outcome, "grounded_local_context_model")
        self.assertEqual(turn.model_calls, 1)
        self.assertFalse(turn.fallback_used)
        self.assertIn("2026-09-09", turn.assistant_message.content)
        self.assertIn("The meeting uses voting and idea merging.", turn.assistant_message.content)
        self.assertIn("Possible preparation (inferred", turn.assistant_message.content)
        self.assertEqual(model.calls[0][1]["target_date"], "2026-09-09")
        self.assertEqual(
            {source.reference.source_id for source in self.store.list_local_context_answer_sources(turn.assistant_message.id)},
            {self.event.id, self.note.id},
        )

    def test_foreign_source_gets_one_repair_before_a_grounded_answer(self) -> None:
        invalid = {
            "outcome": "answer",
            "selected_event_ids": [self.event.id],
            "selected_item_ids": [self.note.id],
            "summary_points": [{"text": "Unsupported.", "source_ids": ["foreign-source"]}],
            "preparation_suggestions": [],
            "missing_information": [],
            "clarification_question": None,
        }
        repaired = {
            **invalid,
            "summary_points": [
                {"text": "The meeting uses voting and idea merging.", "source_ids": [self.note.id]}
            ],
        }
        model = ScriptedLocalContextModel([invalid], [repaired])
        service = LocalContextService(
            self.store,
            reference_date=date(2026, 9, 8),
            timezone="America/Vancouver",
            model=model,
        )

        turn = service.submit(self.later.id, "What time is my meeting tomorrow?")

        self.assertEqual(turn.model_calls, 2)
        self.assertFalse(turn.fallback_used)
        self.assertEqual([call[0] for call in model.calls], ["answer", "repair"])
        self.assertNotIn("foreign-source", turn.assistant_message.content)

    def test_model_failure_preserves_deterministic_local_recall(self) -> None:
        model = ScriptedLocalContextModel([ModelError("unavailable")])
        service = LocalContextService(
            self.store,
            reference_date=date(2026, 9, 8),
            timezone="America/Vancouver",
            model=model,
        )

        turn = service.submit(self.later.id, "What time is my meeting tomorrow?")

        self.assertEqual(turn.outcome, "grounded_local_context")
        self.assertEqual(turn.model_calls, 1)
        self.assertTrue(turn.fallback_used)
        self.assertIn(self.note.content, turn.assistant_message.content)
        self.assertEqual([call[0] for call in model.calls], ["answer"])

    def test_model_time_text_gets_repaired_so_python_is_the_time_authority(self) -> None:
        invalid = {
            "outcome": "answer",
            "selected_event_ids": [self.event.id],
            "selected_item_ids": [self.note.id],
            "summary_points": [
                {"text": "The meeting starts tomorrow at 15:00.", "source_ids": [self.event.id]}
            ],
            "preparation_suggestions": [],
            "missing_information": [],
            "clarification_question": None,
        }
        repaired = {
            **invalid,
            "summary_points": [
                {"text": "The meeting uses voting and idea merging.", "source_ids": [self.note.id]}
            ],
        }
        model = ScriptedLocalContextModel([invalid], [repaired])
        service = LocalContextService(
            self.store,
            reference_date=date(2026, 9, 8),
            timezone="America/Vancouver",
            model=model,
        )

        turn = service.submit(self.later.id, "What time is my meeting tomorrow?")

        self.assertEqual(turn.model_calls, 2)
        self.assertIn("2026-09-09", turn.assistant_message.content)
        self.assertNotIn("starts tomorrow at 15:00", turn.assistant_message.content)

    def test_query_classifier_is_narrow_and_deterministic(self) -> None:
        self.assertTrue(is_local_context_query("我明天的会议要做什么？"))
        self.assertTrue(is_local_context_query("When is the calendar meeting?"))
        self.assertFalse(is_local_context_query("Track a meeting task."))
