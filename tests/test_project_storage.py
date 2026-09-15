import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from captureflow.models import Conversation, Message, Project, ProjectRecord
from captureflow.storage import CaptureStore


class ProjectStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "captureflow.sqlite3"
        self.store = CaptureStore(self.database_path)
        self.store.initialize()
        self.conversation = self.store.save_conversation(
            Conversation(id="conversation-project", title="CaptureFlow V2")
        )
        self.message = self.store.save_message(
            Message(
                id="message-project",
                conversation_id=self.conversation.id,
                role="user",
                content="Phase 3 is complete; multi-turn context remains an open question.",
            )
        )
        self.project = self.store.save_project(Project(id="project-captureflow", name="CaptureFlow"))
        self.store.assign_conversation_project(self.conversation.id, self.project.id)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_project_record_round_trip_creates_a_source_audit_event(self) -> None:
        record = self.store.save_project_record(
            ProjectRecord(
                id="record-open-question",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="open_question",
                content="Add bounded multi-turn context.",
                state="active",
                source_message_id=self.message.id,
                source_quote="multi-turn context remains an open question",
            )
        )

        self.assertEqual(self.store.get_project_record(record.id), record)
        self.assertEqual(self.store.list_project_records(self.project.id), [record])
        events = self.store.list_project_record_events(record.id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, "created")
        self.assertEqual(events[0].source_message_id, self.message.id)

    def test_an_answer_and_its_sources_are_saved_together_or_not_at_all(self) -> None:
        record = self.store.save_project_record(
            ProjectRecord(
                id="record-answer-source",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content="Implement the tracker foundation.",
                state="active",
                source_message_id=self.message.id,
                source_quote="Phase 3 is complete",
            )
        )
        answer = Message(
            id="answer-with-sources",
            conversation_id=self.conversation.id,
            role="assistant",
            content="The tracker foundation is open.",
            reply_to_message_id=self.message.id,
        )

        def fail_inside_the_transaction(connection, assistant_message_id, references):
            raise ValueError("simulated source write failure")

        self.store._insert_local_context_answer_sources = fail_inside_the_transaction
        with self.assertRaisesRegex(ValueError, "simulated"):
            self.store.save_answer_with_sources(answer, [record.id], [])
        with self.assertRaises(KeyError):
            self.store.get_message(answer.id)
        self.assertEqual(self.store.list_project_answer_sources(answer.id), [])

        del self.store._insert_local_context_answer_sources
        saved = self.store.save_answer_with_sources(answer, [record.id], [])
        self.assertEqual(saved, answer)
        self.assertEqual([source.id for source in self.store.list_project_answer_sources(answer.id)], [record.id])

    def test_assignment_locks_after_the_first_project_record(self) -> None:
        self.store.save_project_record(
            ProjectRecord(
                id="record-task",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content="Implement the tracker foundation.",
                state="active",
                source_message_id=self.message.id,
                source_quote="Phase 3 is complete",
            )
        )
        other = self.store.save_project(Project(id="project-other", name="Other"))

        with self.assertRaisesRegex(ValueError, "locked"):
            self.store.assign_conversation_project(self.conversation.id, other.id)

    def test_lifecycle_delete_restore_and_audit_are_transactional(self) -> None:
        record = self.store.save_project_record(
            ProjectRecord(
                id="record-task",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content="Implement the tracker foundation.",
                state="active",
                source_message_id=self.message.id,
                source_quote="Phase 3 is complete",
            )
        )

        completed = self.store.transition_project_record(record.id, "completed")
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            self.store.transition_project_record(record.id, "resolved")
        deleted = self.store.delete_project_record(record.id)
        restored = self.store.restore_project_record(record.id)

        self.assertEqual(completed.state, "completed")
        self.assertEqual(deleted.state, "deleted")
        self.assertEqual(restored.state, "completed")
        self.assertEqual(
            [event.action for event in self.store.list_project_record_events(record.id)],
            ["created", "completed", "deleted", "restored"],
        )
        self.assertEqual(self.store.delete_project_record(record.id), self.store.delete_project_record(record.id))

    def test_project_and_conversation_isolation_rejects_cross_project_records(self) -> None:
        other_conversation = self.store.save_conversation(
            Conversation(id="conversation-other", title="Other project")
        )
        other_message = self.store.save_message(
            Message(
                id="message-other",
                conversation_id=other_conversation.id,
                role="user",
                content="This belongs to another project.",
            )
        )
        other_project = self.store.save_project(Project(id="project-other", name="Other"))
        self.store.assign_conversation_project(other_conversation.id, other_project.id)

        with self.assertRaisesRegex(ValueError, "assigned project"):
            self.store.save_project_record(
                ProjectRecord(
                    id="record-cross-project",
                    project_id=self.project.id,
                    conversation_id=other_conversation.id,
                    kind="blocker",
                    content="This must not cross projects.",
                    state="active",
                    source_message_id=other_message.id,
                    source_quote=other_message.content,
                )
            )

    def test_invalid_initial_state_and_ungrounded_quote_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ProjectRecord(
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="milestone",
                content="Phase 4 complete.",
                state="active",
                source_message_id=self.message.id,
                source_quote="Phase 3 is complete",
            )

        with self.assertRaisesRegex(ValueError, "source quote"):
            self.store.save_project_record(
                ProjectRecord(
                    id="record-ungrounded",
                    project_id=self.project.id,
                    conversation_id=self.conversation.id,
                    kind="decision",
                    content="Use PostgreSQL.",
                    state="active",
                    source_message_id=self.message.id,
                    source_quote="Use PostgreSQL.",
                )
            )

    def test_conversation_delete_cascades_records_but_keeps_the_project(self) -> None:
        record = self.store.save_project_record(
            ProjectRecord(
                id="record-cascade",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="constraint",
                content="Keep the implementation local and single-user.",
                state="active",
                source_message_id=self.message.id,
                source_quote="Phase 3 is complete",
            )
        )

        self.assertTrue(self.store.delete_conversation(self.conversation.id))
        with self.assertRaises(KeyError):
            self.store.get_project_record(record.id)
        self.assertEqual(self.store.get_project(self.project.id), self.project)
        self.assertIsNone(self.store.get_conversation_project(self.conversation.id))

    def test_project_can_be_renamed_and_deleted_without_deleting_its_conversation(self) -> None:
        renamed = self.store.update_project(
            self.project.id,
            name="CaptureFlow workspace",
            description="A local project assistant.",
        )
        record = self.store.save_project_record(
            ProjectRecord(
                id="record-before-project-delete",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content="Keep the conversation after deleting the project.",
                state="active",
                source_message_id=self.message.id,
                source_quote="multi-turn context remains an open question",
            )
        )
        self.store.begin_project_conversation(self.conversation.id)

        self.assertEqual(renamed.name, "CaptureFlow workspace")
        self.assertEqual(renamed.description, "A local project assistant.")
        self.assertTrue(self.store.delete_project(self.project.id))

        with self.assertRaises(KeyError):
            self.store.get_project(self.project.id)
        with self.assertRaises(KeyError):
            self.store.get_project_record(record.id)
        self.assertEqual(self.store.get_conversation(self.conversation.id), self.conversation)
        self.assertEqual(self.store.get_message(self.message.id), self.message)
        self.assertIsNone(self.store.get_conversation_project(self.conversation.id))
        replacement = self.store.save_project(Project(id="replacement-project", name="Replacement"))
        self.assertEqual(
            self.store.assign_conversation_project(self.conversation.id, replacement.id),
            replacement,
        )
