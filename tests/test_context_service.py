import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from captureflow.context_service import active_project_records, select_context_window
from captureflow.models import (
    Conversation,
    Message,
    Project,
    ProjectRecord,
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

    def test_only_the_projects_own_active_records_are_returned(self) -> None:
        source = self._source()
        other_conversation = self.store.save_conversation(Conversation(id="candidate-other", title="Other project"))
        other_project = self.store.save_project(Project(id="project-other", name="Other"))
        self.store.assign_conversation_project(other_conversation.id, other_project.id)
        other_source = self._source("candidate-other-source", other_conversation.id)
        active = self._record("active-project-record", "Active project task.", "active", source)
        self._record("completed-project-record", "Completed project task.", "completed", source)
        self._record("deleted-project-record", "Deleted project task.", "deleted", source)
        self.store.save_project_record(
            ProjectRecord(
                id="other-project-record",
                project_id=other_project.id,
                conversation_id=other_conversation.id,
                kind="task",
                content="Other project task.",
                state="active",
                source_message_id=other_source.id,
                source_quote=other_source.content,
            )
        )

        records = active_project_records(self.store, self.conversation.id)

        self.assertEqual([record.id for record in records], [active.id])

    def test_an_unassigned_conversation_has_no_records_to_compare_against(self) -> None:
        unassigned = self.store.save_conversation(
            Conversation(id="unassigned-router", title="Unassigned")
        )

        self.assertEqual(active_project_records(self.store, unassigned.id), ())

    def test_the_list_is_not_bounded_because_a_rule_compares_against_all_of_it(self) -> None:
        # It used to be cut at 30, most recently updated first, to keep the prompt short.
        # The prompt is gone, and the one caller refuses a duplicate by comparing against
        # this list: a record past the bound would simply not have been compared against.
        source = self._source("tasks-source")
        for index in range(31):
            self._record(f"task-{index:02d}", f"Task {index}", "active", source)

        records = active_project_records(self.store, self.conversation.id)

        self.assertEqual(len(records), 31)
        self.assertIn("task-00", [record.id for record in records])

    def _record(self, record_id: str, content: str, state: str, source: Message) -> ProjectRecord:
        return self.store.save_project_record(
            ProjectRecord(
                id=record_id,
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content=content,
                state=state,
                source_message_id=source.id,
                source_quote=source.content,
            )
        )

    def test_first_project_chat_turn_locks_the_assignment(self) -> None:
        self.store.begin_project_conversation(self.conversation.id)
        other_project = self.store.save_project(Project(id="project-other-context", name="Other"))

        with self.assertRaisesRegex(ValueError, "first project chat turn"):
            self.store.assign_conversation_project(self.conversation.id, other_project.id)
