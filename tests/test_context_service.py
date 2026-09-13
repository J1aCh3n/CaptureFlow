import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
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
        self.temporary_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
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

    def test_the_last_two_turns_stay_whole_and_older_messages_fill_the_budget(self) -> None:
        messages = [
            Message(id=f"message-{index}", conversation_id="conversation", role="user", content="x" * 10)
            for index in range(8)
        ]

        window = select_context_window(messages, max_characters=60)

        self.assertEqual([message.id for message in window.messages], [f"message-{index}" for index in range(2, 8)])
        self.assertEqual(window.omitted_message_count, 2)
        self.assertTrue(window.truncated)

    def test_the_last_two_turns_are_kept_even_over_the_budget(self) -> None:
        messages = [
            Message(id=f"long-{index}", conversation_id="conversation", role="user", content="y" * 700)
            for index in range(5)
        ]

        window = select_context_window(messages, max_characters=1_000)

        self.assertEqual([message.id for message in window.messages], ["long-1", "long-2", "long-3", "long-4"])

    def test_an_older_long_message_becomes_a_one_line_reference(self) -> None:
        pasted = "会议纪要：" + "内容" * 500
        messages = [
            Message(id="pasted", conversation_id="conversation", role="user", content=pasted),
            *[
                Message(id=f"later-{index}", conversation_id="conversation", role="user", content="继续")
                for index in range(4)
            ],
        ]

        older = select_context_window(messages).for_router()
        recent = select_context_window(messages[:4]).for_router()

        self.assertEqual(older[0]["message_id"], "pasted")
        self.assertNotIn("content", older[0])
        self.assertIn(f"{len(pasted)} characters", older[0]["reference"])
        self.assertIn(pasted[:80], older[0]["reference"])
        self.assertEqual(recent[0]["content"], pasted)

    def test_a_blocked_credential_message_is_left_out_of_the_context(self) -> None:
        messages = [
            Message(
                id="secret", conversation_id="conversation", role="user", content="我的密钥是 sk-test-1234567890abcdef"
            ),
            Message(id="after", conversation_id="conversation", role="user", content="你好"),
        ]

        window = select_context_window(messages)

        self.assertEqual([message.id for message in window.messages], ["after"])

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

    def _source(self, message_id: str = "candidate-source", conversation_id: str | None = None) -> Message:
        return self.store.save_message(
            Message(
                id=message_id,
                conversation_id=conversation_id or self.conversation.id,
                role="user",
                content="Candidate source.",
            )
        )

    def _event(self, event_id: str, title: str, start_at: date | datetime, source: Message, **fields) -> CalendarEvent:
        return self.store.save_calendar_event(
            CalendarEvent(
                id=event_id,
                conversation_id=source.conversation_id,
                title=title,
                source_message_id=source.id,
                source_quote=source.content,
                start_at=start_at,
                all_day=not isinstance(start_at, datetime),
                timezone="America/Vancouver",
                **fields,
            )
        )

    def test_records_and_ideas_stay_in_the_project_but_events_are_shared(self) -> None:
        source = self._source()
        other_conversation = self.store.save_conversation(Conversation(id="candidate-other", title="Other project"))
        other_project = self.store.save_project(Project(id="project-other", name="Other"))
        self.store.assign_conversation_project(other_conversation.id, other_project.id)
        other_source = self._source("candidate-other-source", other_conversation.id)
        shared_event = self._event("other-project-event", "牙医预约", date(2026, 9, 12), other_source)
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
        self.store.save_saved_item(
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

        candidates = select_router_candidates(self.store, self.conversation.id, date(2026, 9, 11))

        self.assertEqual([event.id for event in candidates.events], [shared_event.id])
        self.assertEqual([item.id for item in candidates.saved_items], [same_project_item.id])
        self.assertEqual([record.id for record in candidates.project_records], [active_record.id])
        self.assertEqual(candidates.truncated_kinds, ())

    def test_events_come_from_the_calendar_window_nearest_the_reference_date(self) -> None:
        source = self._source()
        for suffix, start_at in {
            "stale": "2026-08-03T10:00:00-07:00",
            "yesterday": "2026-09-10T16:00:00-07:00",
            "tomorrow": "2026-09-12T09:00:00-07:00",
            "next-week": "2026-09-16T20:00:00-07:00",
            "far": "2026-12-01T09:00:00-08:00",
        }.items():
            self._event(f"nearest-{suffix}", suffix, datetime.fromisoformat(start_at), source)

        candidates = select_router_candidates(self.store, self.conversation.id, date(2026, 9, 11))

        self.assertEqual(
            [event.id for event in candidates.events],
            ["nearest-yesterday", "nearest-tomorrow", "nearest-next-week"],
        )

    def test_a_cut_event_list_is_marked_and_keeps_the_nearest(self) -> None:
        source = self._source()
        for offset in range(-7, 20):
            self._event(f"day-{offset + 7:02d}", f"Sync {offset}", date(2026, 9, 11) + timedelta(days=offset), source)

        candidates = select_router_candidates(self.store, self.conversation.id, date(2026, 9, 11))

        self.assertEqual(len(candidates.events), 15)
        self.assertIn("events", candidates.truncated_kinds)
        self.assertEqual(candidates.events[0].start_date, date(2026, 9, 4))
        self.assertEqual(candidates.events[-1].start_date, date(2026, 9, 18))

    def test_an_event_the_message_names_is_added_beyond_the_window(self) -> None:
        source = self._source()
        roadmap = self._event("roadmap", "季度路线图评审", date(2026, 12, 1), source)
        self._event("other-far", "年终总结", date(2026, 12, 20), source)

        named = select_router_candidates(self.store, self.conversation.id, date(2026, 9, 11), "季度路线图评审改到下午")
        vague = select_router_candidates(self.store, self.conversation.id, date(2026, 9, 11), "这个评审改到下午")

        self.assertEqual([event.id for event in named.events], [roadmap.id])
        self.assertEqual(vague.events, ())

    def test_records_are_bounded_most_recently_updated_first(self) -> None:
        source = self._source("tasks-source")
        for index in range(31):
            self.store.save_project_record(
                ProjectRecord(
                    id=f"task-{index:02d}",
                    project_id=self.project.id,
                    conversation_id=self.conversation.id,
                    kind="task",
                    content=f"Task {index}",
                    state="active",
                    source_message_id=source.id,
                    source_quote=source.content,
                    updated_at=datetime(2026, 9, 1, 0, index, tzinfo=timezone.utc),
                )
            )

        candidates = select_router_candidates(self.store, self.conversation.id, date(2026, 9, 11))

        self.assertEqual(len(candidates.project_records), 30)
        self.assertEqual(candidates.project_records[0].id, "task-30")
        self.assertNotIn("task-00", [record.id for record in candidates.project_records])
        self.assertEqual(candidates.truncated_kinds, ("project_records",))

    def test_candidates_for_the_router_are_one_line_each(self) -> None:
        source = self._source()
        self._event(
            "review",
            "算法优化与代码审查会议",
            datetime.fromisoformat("2026-09-17T14:00:00-07:00"),
            source,
            end_at=datetime.fromisoformat("2026-09-17T15:00:00-07:00"),
        )
        self._event("offsite", "团建", date(2026, 9, 18), source)
        self.store.save_project_record(
            ProjectRecord(
                id="r-12",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content="排查延迟",
                state="active",
                due_at=datetime.fromisoformat("2026-09-20T18:00:00-07:00"),
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        self.store.save_saved_item(
            SavedItem(
                id="i-3",
                conversation_id=self.conversation.id,
                kind="idea",
                content="用 " + "很长的想法 " * 100,
                source_message_id=source.id,
                source_quote=source.content,
            )
        )

        lines = select_router_candidates(self.store, self.conversation.id, date(2026, 9, 11)).for_router()

        self.assertEqual(
            lines["candidate_events"],
            ["review | 算法优化与代码审查会议 | 2026-09-17 14:00–15:00", "offsite | 团建 | 2026-09-18 all day"],
        )
        self.assertEqual(lines["candidate_project_records"], ["r-12 | task | 排查延迟 | active | due 2026-09-20 18:00"])
        self.assertTrue(lines["candidate_saved_items"][0].startswith("i-3 | idea | 用 "))
        self.assertLess(len(lines["candidate_saved_items"][0]), 230)

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

        candidates = select_router_candidates(self.store, unassigned.id, date(2026, 9, 11))

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

        candidates = select_router_candidates(self.store, self.conversation.id, date(2026, 9, 11))

        self.assertNotIn(deleted_event.id, [candidate.id for candidate in candidates.events])
        self.assertNotIn(deleted_item.id, [candidate.id for candidate in candidates.saved_items])
        self.assertNotIn(foreign_event.id, [candidate.id for candidate in candidates.events])
        self.assertEqual(candidates.project_records, ())

    def test_first_project_chat_turn_locks_the_assignment(self) -> None:
        self.store.begin_project_conversation(self.conversation.id)
        other_project = self.store.save_project(Project(id="project-other-context", name="Other"))

        with self.assertRaisesRegex(ValueError, "first project chat turn"):
            self.store.assign_conversation_project(self.conversation.id, other_project.id)
