import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from captureflow.local_context_service import LocalContextService
from captureflow.models import (
    CalendarEvent,
    Conversation,
    Message,
    Project,
    ProjectRecord,
    SavedItem,
)
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

    def test_event_description_range_and_day_after_tomorrow_are_retrievable(self) -> None:
        source = self.store.save_message(
            Message(
                id="range-source-message",
                conversation_id=self.first.id,
                role="user",
                content="后天下午四点到五点有组会，所有人带好工牌。",
            )
        )
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="range-event",
                conversation_id=self.first.id,
                title="组会",
                description="所有人带好工牌。",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=datetime.fromisoformat("2026-09-10T16:00:00-07:00"),
                end_at=datetime.fromisoformat("2026-09-10T17:00:00-07:00"),
                all_day=False,
                timezone="America/Vancouver",
            )
        )

        turn = self.service.submit(self.later.id, "我后天有什么安排？")

        self.assertEqual([source.id for source in turn.sources], [event.id])
        self.assertIn(event.description, turn.assistant_message.content)
        self.assertIn(event.end_at.isoformat(), turn.assistant_message.content)

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

    def test_missing_local_context_returns_unknown(self) -> None:
        turn = self.service.submit(self.later.id, "What is the calendar agenda for Friday?")

        self.assertEqual(turn.outcome, "insufficient_context")
        self.assertIn("do not have matching local", turn.assistant_message.content)
        self.assertEqual(self.store.list_local_context_answer_sources(turn.assistant_message.id), [])

    def test_same_project_record_answers_agenda_without_claiming_a_calendar_event(self) -> None:
        project = self.store.save_project(Project(id="workflow-project", name="Workflow"))
        self.store.assign_conversation_project(self.first.id, project.id)
        self.store.assign_conversation_project(self.later.id, project.id)
        self.store.set_calendar_event_status(self.event.id, "deleted")
        self.store.set_saved_item_status(self.note.id, "deleted")
        record = self.store.save_project_record(
            ProjectRecord(
                id="workflow-purpose",
                project_id=project.id,
                conversation_id=self.first.id,
                kind="decision",
                content="工作流会议用于对已有点子投票、合并和深度优化。",
                state="active",
                source_message_id="meeting-source-message",
                source_quote="Tomorrow at 15:00 is the workflow meeting. The meeting uses voting and idea merging.",
            )
        )
        turn = self.service.submit(
            self.later.id,
            "我明天的工作流会议几点开？会议目的是什么？我应该准备什么？",
        )

        self.assertEqual(turn.outcome, "grounded_local_context")
        self.assertEqual([source.id for source in turn.sources], [record.id])
        self.assertIn("No matching Calendar event is stored", turn.assistant_message.content)
        self.assertIn(record.content, turn.assistant_message.content)
        self.assertEqual(
            [source.id for source in self.store.list_project_answer_sources(turn.assistant_message.id)],
            [record.id],
        )
        self.assertEqual(
            self.store.list_local_context_answer_sources(turn.assistant_message.id), []
        )

    def test_assigned_project_excludes_saved_items_from_other_projects(self) -> None:
        project = self.store.save_project(Project(id="selected-project", name="Selected"))
        other_project = self.store.save_project(Project(id="other-project", name="Other"))
        self.store.assign_conversation_project(self.first.id, project.id)
        self.store.assign_conversation_project(self.later.id, project.id)
        other_conversation = self.store.save_conversation(
            Conversation(id="other-conversation", title="Other")
        )
        self.store.assign_conversation_project(other_conversation.id, other_project.id)
        other_source = self.store.save_message(
            Message(
                id="other-source",
                conversation_id=other_conversation.id,
                role="user",
                content="另一个项目的工作流会议使用每日无干扰时段。",
            )
        )
        unrelated = self.store.save_saved_item(
            SavedItem(
                id="other-project-idea",
                conversation_id=other_conversation.id,
                kind="idea",
                content="另一个项目的工作流会议使用每日无干扰时段。",
                source_message_id=other_source.id,
                source_quote=other_source.content,
            )
        )

        turn = LocalContextService(
            self.store,
            reference_date=date(2026, 9, 8),
            timezone="America/Vancouver",
        ).submit(self.later.id, "我明天的工作流会议有什么内容？")

        self.assertNotIn(unrelated.id, {source.id for source in turn.sources})
        self.assertNotIn(unrelated.content, turn.assistant_message.content)

    def test_calendar_creation_task_does_not_count_as_a_scheduled_activity(self) -> None:
        project = self.store.save_project(Project(id="task-only-project", name="Task only"))
        self.store.assign_conversation_project(self.first.id, project.id)
        self.store.assign_conversation_project(self.later.id, project.id)
        self.store.set_calendar_event_status(self.event.id, "deleted")
        self.store.set_saved_item_status(self.note.id, "deleted")
        self.store.save_project_record(
            ProjectRecord(
                id="create-calendar-task",
                project_id=project.id,
                conversation_id=self.first.id,
                kind="task",
                content="在明天下午两点创建日历：工作流会议。",
                state="active",
                source_message_id="meeting-source-message",
                source_quote="Tomorrow at 15:00 is the workflow meeting. The meeting uses voting and idea merging.",
            )
        )

        turn = LocalContextService(
            self.store,
            reference_date=date(2026, 9, 8),
            timezone="America/Vancouver",
        ).submit(self.later.id, "我明天有什么活动")

        self.assertEqual(turn.outcome, "insufficient_context")
        self.assertEqual(turn.sources, ())
        self.assertIn("do not have matching local Calendar", turn.assistant_message.content)
        self.assertEqual(
            self.store.list_project_answer_sources(turn.assistant_message.id), []
        )
