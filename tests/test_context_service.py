import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from captureflow.context_service import select_context_window, select_router_candidates
from captureflow.models import (
    CalendarEvent,
    Conversation,
    Message,
    Project,
    ProjectRecord,
    SavedItem,
)
from captureflow.storage import CaptureStore


class RouterContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "captureflow.sqlite3"
        self.store = CaptureStore(self.database_path)
        self.store.initialize()
        self.conversation = self.store.save_conversation(
            Conversation(id="conversation-project-chat", title="Project chat")
        )
        self.project = self.store.save_project(Project(id="project-context", name="CaptureFlow"))
        self.store.assign_conversation_project(self.conversation.id, self.project.id)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_context_window_is_recent_contiguous_and_reports_omission(self) -> None:
        messages = [
            Message(id=f"message-{index}", conversation_id="conversation", role="user", content="x" * 10)
            for index in range(4)
        ]

        window = select_context_window(messages, max_messages=2, max_characters=30)

        self.assertEqual([message.id for message in window.messages], ["message-2", "message-3"])
        self.assertEqual(window.omitted_message_count, 2)

    def test_context_window_uses_reply_link_when_timestamp_order_is_ambiguous(self) -> None:
        user = Message(
            id="z-user",
            conversation_id="conversation",
            role="user",
            content="Original project statement.",
        )
        assistant = Message(
            id="a-assistant",
            conversation_id="conversation",
            role="assistant",
            content="Original project reply.",
            reply_to_message_id=user.id,
            created_at=user.created_at,
        )

        window = select_context_window([assistant, user])

        self.assertEqual([message.id for message in window.messages], [user.id, assistant.id])

    def test_router_context_carries_message_ids(self) -> None:
        messages = [
            Message(id="user-1", conversation_id="conversation", role="user", content="Schedule it."),
            Message(
                id="assistant-1",
                conversation_id="conversation",
                role="assistant",
                content="Which time?",
                reply_to_message_id="user-1",
            ),
        ]

        window = select_context_window(messages)

        self.assertEqual(
            window.for_router(),
            [
                {"message_id": "user-1", "role": "user", "content": "Schedule it."},
                {
                    "message_id": "assistant-1",
                    "role": "assistant",
                    "content": "Which time?",
                },
            ],
        )

    def test_router_candidates_are_bounded_and_project_scoped(self) -> None:
        source = self.store.save_message(
            Message(
                id="candidate-source",
                conversation_id=self.conversation.id,
                role="user",
                content="Candidate source.",
            )
        )
        other_conversation = self.store.save_conversation(
            Conversation(id="candidate-other", title="Other project")
        )
        other_project = self.store.save_project(Project(id="project-other", name="Other"))
        self.store.assign_conversation_project(other_conversation.id, other_project.id)
        other_source = self.store.save_message(
            Message(
                id="candidate-other-source",
                conversation_id=other_conversation.id,
                role="user",
                content="Other candidate source.",
            )
        )
        events = [
            self.store.save_calendar_event(
                CalendarEvent(
                    id=f"candidate-event-{index}",
                    conversation_id=self.conversation.id,
                    title=f"Event {index}",
                    source_message_id=source.id,
                    source_quote=source.content,
                    start_at=date(2026, 9, 10 + index),
                    all_day=True,
                    timezone="America/Vancouver",
                )
            )
            for index in range(4)
        ]
        same_project_item = self.store.save_saved_item(
            SavedItem(
                id="same-project-item",
                conversation_id=self.conversation.id,
                kind="idea",
                content="Same project idea.",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        other_project_item = self.store.save_saved_item(
            SavedItem(
                id="other-project-item",
                conversation_id=other_conversation.id,
                kind="note",
                content="Other project note.",
                source_message_id=other_source.id,
                source_quote=other_source.content,
            )
        )
        active_record = self.store.save_project_record(
            ProjectRecord(
                id="active-project-record",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content="Active project task.",
                state="active",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        self.store.save_project_record(
            ProjectRecord(
                id="completed-project-record",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content="Completed project task.",
                state="completed",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )

        candidates = select_router_candidates(self.store, self.conversation.id)

        self.assertEqual([event.id for event in candidates.events], [event.id for event in events[:3]])
        self.assertEqual([item.id for item in candidates.saved_items], [same_project_item.id])
        self.assertNotIn(other_project_item.id, [item.id for item in candidates.saved_items])
        self.assertEqual([record.id for record in candidates.project_records], [active_record.id])

    def test_router_candidates_follow_the_reference_date_not_the_oldest_events(self) -> None:
        source = self.store.save_message(
            Message(
                id="nearest-source",
                conversation_id=self.conversation.id,
                role="user",
                content="Nearest candidate source.",
            )
        )
        starts = {
            "stale-1": "2026-08-03T10:00:00-07:00",
            "stale-2": "2026-08-10T10:00:00-07:00",
            "stale-3": "2026-08-17T10:00:00-07:00",
            "yesterday": "2026-09-10T16:00:00-07:00",
            "tomorrow": "2026-09-12T09:00:00-07:00",
            "next-week": "2026-09-16T20:00:00-07:00",
            "far": "2026-12-01T09:00:00-07:00",
        }
        for suffix, start_at in starts.items():
            self.store.save_calendar_event(
                CalendarEvent(
                    id=f"nearest-{suffix}",
                    conversation_id=self.conversation.id,
                    title=suffix,
                    source_message_id=source.id,
                    source_quote=source.content,
                    start_at=datetime.fromisoformat(start_at),
                    all_day=False,
                    timezone="America/Vancouver",
                )
            )

        candidates = select_router_candidates(
            self.store, self.conversation.id, date(2026, 9, 11)
        )

        self.assertEqual(
            [event.id for event in candidates.events],
            ["nearest-yesterday", "nearest-tomorrow", "nearest-next-week"],
        )

    def _active_tasks(self, contents: list[str]) -> list[ProjectRecord]:
        records = []
        for index, content in enumerate(contents):
            source = self.store.save_message(
                Message(
                    id=f"relevance-source-{index}",
                    conversation_id=self.conversation.id,
                    role="user",
                    content=content,
                )
            )
            records.append(
                self.store.save_project_record(
                    ProjectRecord(
                        id=f"relevance-task-{index}",
                        project_id=self.project.id,
                        conversation_id=self.conversation.id,
                        kind="task",
                        content=content,
                        state="active",
                        source_message_id=source.id,
                        source_quote=content,
                        updated_at=datetime(2026, 9, 1 + index, tzinfo=timezone.utc),
                    )
                )
            )
        return records

    def test_router_record_candidates_follow_the_message_not_only_recency(self) -> None:
        self._active_tasks(
            ["完成驱动调试", "完成代码审查", "整理会议纪要", "更新部署文档", "修复登录页样式"]
        )

        about_oldest = select_router_candidates(
            self.store, self.conversation.id, message="驱动调试的任务已经做完了"
        )
        unrelated = select_router_candidates(
            self.store, self.conversation.id, message="hello"
        )

        self.assertEqual(about_oldest.project_records[0].id, "relevance-task-0")
        self.assertEqual(len(about_oldest.project_records), 3)
        self.assertEqual(
            [record.id for record in unrelated.project_records],
            ["relevance-task-4", "relevance-task-3", "relevance-task-2"],
        )

    def test_router_candidates_keep_the_stored_order_without_a_reference_date(self) -> None:
        source = self.store.save_message(
            Message(
                id="ordered-source",
                conversation_id=self.conversation.id,
                role="user",
                content="Ordered candidate source.",
            )
        )
        for suffix, start_at in {
            "stale": "2026-08-03T10:00:00-07:00",
            "tomorrow": "2026-09-12T09:00:00-07:00",
        }.items():
            self.store.save_calendar_event(
                CalendarEvent(
                    id=f"ordered-{suffix}",
                    conversation_id=self.conversation.id,
                    title=suffix,
                    source_message_id=source.id,
                    source_quote=source.content,
                    start_at=datetime.fromisoformat(start_at),
                    all_day=False,
                    timezone="America/Vancouver",
                )
            )

        candidates = select_router_candidates(self.store, self.conversation.id, max_per_kind=1)

        self.assertEqual([event.id for event in candidates.events], ["ordered-stale"])

    def test_unassigned_router_candidates_exclude_project_records(self) -> None:
        unassigned = self.store.save_conversation(
            Conversation(id="unassigned-router", title="Unassigned")
        )
        source = self.store.save_message(
            Message(
                id="unassigned-source",
                conversation_id=unassigned.id,
                role="user",
                content="An unassigned idea.",
            )
        )
        item = self.store.save_saved_item(
            SavedItem(
                id="unassigned-item",
                conversation_id=unassigned.id,
                kind="note",
                content="An unassigned note.",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )

        candidates = select_router_candidates(self.store, unassigned.id)

        self.assertIn(item.id, [candidate.id for candidate in candidates.saved_items])
        self.assertEqual(candidates.project_records, ())

    def test_router_candidates_exclude_deleted_and_foreign_workspace_records(self) -> None:
        source = self.store.save_message(
            Message(
                id="deleted-candidate-source",
                conversation_id=self.conversation.id,
                role="user",
                content="Deleted candidate source.",
            )
        )
        deleted_event = self.store.save_calendar_event(
            CalendarEvent(
                id="deleted-candidate-event",
                conversation_id=self.conversation.id,
                title="Deleted event",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=date(2026, 9, 10),
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        deleted_item = self.store.save_saved_item(
            SavedItem(
                id="deleted-candidate-item",
                conversation_id=self.conversation.id,
                kind="note",
                content="Deleted note.",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        self.store.save_project_record(
            ProjectRecord(
                id="deleted-candidate-record",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content="Deleted task.",
                state="deleted",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        self.store.set_calendar_event_status(deleted_event.id, "deleted")
        self.store.set_saved_item_status(deleted_item.id, "deleted")
        other_workspace_conversation = self.store.save_conversation(
            Conversation(
                id="other-workspace-conversation",
                workspace_id="other-workspace",
                title="Other workspace",
            )
        )
        other_workspace_source = self.store.save_message(
            Message(
                id="other-workspace-source",
                conversation_id=other_workspace_conversation.id,
                role="user",
                content="Other workspace source.",
            )
        )
        foreign_event = self.store.save_calendar_event(
            CalendarEvent(
                id="other-workspace-event",
                workspace_id="other-workspace",
                conversation_id=other_workspace_conversation.id,
                title="Foreign event",
                source_message_id=other_workspace_source.id,
                source_quote=other_workspace_source.content,
                start_at=date(2026, 9, 11),
                all_day=True,
                timezone="America/Vancouver",
            )
        )

        candidates = select_router_candidates(self.store, self.conversation.id)

        self.assertNotIn(deleted_event.id, [candidate.id for candidate in candidates.events])
        self.assertNotIn(deleted_item.id, [candidate.id for candidate in candidates.saved_items])
        self.assertNotIn(foreign_event.id, [candidate.id for candidate in candidates.events])
        self.assertEqual(candidates.project_records, ())

    def test_first_project_chat_turn_locks_the_assignment(self) -> None:
        self.store.begin_project_conversation(self.conversation.id)
        other_project = self.store.save_project(Project(id="project-other-context", name="Other"))

        with self.assertRaisesRegex(ValueError, "first project chat turn"):
            self.store.assign_conversation_project(self.conversation.id, other_project.id)
