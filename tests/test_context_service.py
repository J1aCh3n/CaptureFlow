import tempfile
import unittest
from pathlib import Path

from captureflow.context_service import ProjectConversationService, select_project_context
from captureflow.llm import ModelError
from captureflow.models import Conversation, Message, Project, ProjectConversationDecision
from captureflow.storage import CaptureStore


class ScriptedProjectModel:
    def __init__(self, decisions=(), fail=False) -> None:
        self.decisions = list(decisions)
        self.fail = fail
        self.calls = []

    def respond_to_project_turn(self, project, recent_messages, context_truncated):
        self.calls.append(
            {
                "project_id": project.id,
                "messages": recent_messages,
                "context_truncated": context_truncated,
            }
        )
        if self.fail:
            raise ModelError("project responder unavailable")
        return self.decisions.pop(0)


def decision(reply: str, turn_kind: str) -> ProjectConversationDecision:
    return ProjectConversationDecision(reply=reply, turn_kind=turn_kind)


class ProjectConversationServiceTests(unittest.TestCase):
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

    def test_follow_up_receives_recent_project_context_without_writing_records(self) -> None:
        model = ScriptedProjectModel(
            [
                decision("SQLite remains the local store.", "new_subject"),
                decision("Because it keeps the MVP local and additive.", "follow_up"),
            ]
        )
        service = ProjectConversationService(self.store, model)

        service.submit(self.conversation.id, "We should keep SQLite for this project.", "turn-1")
        second = service.submit(self.conversation.id, "Why?", "turn-2")

        self.assertEqual(second.decision.turn_kind, "follow_up")
        self.assertEqual(
            [message["content"] for message in model.calls[1]["messages"]],
            [
                "We should keep SQLite for this project.",
                "SQLite remains the local store.",
                "Why?",
            ],
        )
        self.assertTrue(self.store.conversation_has_project_turn(self.conversation.id))
        self.assertEqual(self.store.list_project_records(self.project.id), [])

    def test_context_window_is_recent_contiguous_and_reports_omission(self) -> None:
        messages = [
            Message(id=f"message-{index}", conversation_id="conversation", role="user", content="x" * 10)
            for index in range(4)
        ]

        window = select_project_context(messages, max_messages=2, max_characters=30)

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

        window = select_project_context([assistant, user])

        self.assertEqual([message.id for message in window.messages], [user.id, assistant.id])

    def test_service_discloses_when_older_messages_fall_outside_the_window(self) -> None:
        for index in range(4):
            user = self.store.save_message(
                Message(
                    id=f"prior-user-{index}",
                    conversation_id=self.conversation.id,
                    role="user",
                    content=f"Prior project detail {index}.",
                )
            )
            self.store.save_message(
                Message(
                    id=f"prior-assistant-{index}",
                    conversation_id=self.conversation.id,
                    role="assistant",
                    content=f"Acknowledged {index}.",
                    reply_to_message_id=user.id,
                )
            )
        model = ScriptedProjectModel([decision("I can continue from the recent context.", "follow_up")])

        turn = ProjectConversationService(self.store, model).submit(
            self.conversation.id, "Continue.", "truncated-turn"
        )

        self.assertTrue(turn.context_window.truncated)
        self.assertIn("outside this turn's context window", turn.assistant_message.content)
        self.assertTrue(model.calls[0]["context_truncated"])
        self.assertEqual(len(model.calls[0]["messages"]), 8)

    def test_classification_covers_correction_cancellation_and_clarification_answer(self) -> None:
        model = ScriptedProjectModel(
            [
                decision("I understand this is a correction.", "correction"),
                decision("I understand the cancellation.", "cancellation"),
                decision("I will treat that as the answer to the earlier question.", "clarification_answer"),
            ]
        )
        service = ProjectConversationService(self.store, model)

        observed = [
            service.submit(self.conversation.id, text, f"classification-{index}").decision.turn_kind
            for index, text in enumerate(("Actually, use the local file.", "Cancel that.", "September 11."))
        ]

        self.assertEqual(observed, ["correction", "cancellation", "clarification_answer"])
        self.assertEqual(self.store.list_project_records(self.project.id), [])

    def test_other_conversation_never_enters_the_context_window(self) -> None:
        other_conversation = self.store.save_conversation(
            Conversation(id="conversation-other-project-chat", title="Other chat")
        )
        self.store.assign_conversation_project(other_conversation.id, self.project.id)
        self.store.save_message(
            Message(
                id="other-message",
                conversation_id=other_conversation.id,
                role="user",
                content="This must stay out of the first conversation.",
            )
        )
        model = ScriptedProjectModel([decision("Starting the first conversation.", "new_subject")])

        ProjectConversationService(self.store, model).submit(
            self.conversation.id, "Start this project discussion.", "isolated-turn"
        )

        self.assertNotIn(
            "This must stay out of the first conversation.",
            [message["content"] for message in model.calls[0]["messages"]],
        )

    def test_restart_reuses_persisted_history_and_same_message_does_not_recall_the_model(self) -> None:
        first_model = ScriptedProjectModel([decision("First contextual reply.", "new_subject")])
        first_service = ProjectConversationService(self.store, first_model)
        first_service.submit(self.conversation.id, "Keep the tracker deterministic.", "restart-turn-1")

        restarted_store = CaptureStore(self.database_path)
        restarted_store.initialize()
        restarted_model = ScriptedProjectModel([decision("That follows the earlier discussion.", "follow_up")])
        restarted_service = ProjectConversationService(restarted_store, restarted_model)
        second = restarted_service.submit(self.conversation.id, "Why does that matter?", "restart-turn-2")
        replay = restarted_service.submit(self.conversation.id, "Why does that matter?", "restart-turn-2")

        self.assertEqual(second.model_calls, 1)
        self.assertEqual(replay.model_calls, 0)
        self.assertIn("Keep the tracker deterministic.", [m["content"] for m in restarted_model.calls[0]["messages"]])

    def test_model_failure_persists_the_user_message_without_a_tracker_write(self) -> None:
        turn = ProjectConversationService(self.store, ScriptedProjectModel(fail=True)).submit(
            self.conversation.id, "This reply will fail.", "failed-turn"
        )

        self.assertEqual(turn.outcome, "model_failure")
        self.assertEqual(turn.user_message.content, "This reply will fail.")
        self.assertIn("no tracker record was changed", turn.assistant_message.content)
        self.assertEqual(self.store.list_project_records(self.project.id), [])

    def test_first_project_chat_turn_locks_the_assignment(self) -> None:
        model = ScriptedProjectModel([decision("Acknowledged.", "new_subject")])
        ProjectConversationService(self.store, model).submit(
            self.conversation.id, "Start project chat.", "locking-turn"
        )
        other_project = self.store.save_project(Project(id="project-other-context", name="Other"))

        with self.assertRaisesRegex(ValueError, "first project chat turn"):
            self.store.assign_conversation_project(self.conversation.id, other_project.id)
