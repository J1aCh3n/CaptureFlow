import tempfile
import unittest
from pathlib import Path

from captureflow.models import (
    Conversation,
    Message,
    Project,
    ProjectChange,
    ProjectChangeOperation,
    ProjectRecord,
)
from captureflow.project_service import apply_validated_changes
from captureflow.storage import CaptureStore


class ApplyValidatedChangesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.conversation = self.store.save_conversation(
            Conversation(id="conversation-tracker", title="Tracker")
        )
        self.project = self.store.save_project(Project(id="project-tracker", name="CaptureFlow"))
        self.store.assign_conversation_project(self.conversation.id, self.project.id)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_each_record_kind_can_be_created_from_an_exact_current_turn_quote(self) -> None:
        cases = (
            ("task", "Implement the grounded tracker update."),
            ("milestone", "Phase 6 is complete."),
            ("decision", "We will keep SQLite as the local store."),
            ("constraint", "This remains a single-user local project."),
            ("open_question", "How should Phase 7 retrieve prior records?"),
            ("blocker", "The provider key is not configured."),
        )
        for kind, quote in cases:
            committed, reviewed = self._apply(
                quote, {"action": "create_record", "kind": kind, "content": quote, "source_quote": quote}
            )
            self.assertEqual([operation.status for operation in committed], ["committed"])
            self.assertEqual(reviewed, [])

        records = self.store.list_project_records(self.project.id)
        self.assertEqual({record.kind for record in records}, {case[0] for case in cases})
        milestone = next(record for record in records if record.kind == "milestone")
        self.assertEqual(milestone.state, "completed")

    def test_completion_and_resolution_commit_only_valid_lifecycle_transitions(self) -> None:
        task = self._record("task", "active", "Implement Phase 6.")
        question = self._record("open_question", "active", "Which retrieval strategy?")
        blocker = self._record("blocker", "active", "No provider key.")
        for record, state, message in (
            (task, "completed", "The implementation task is complete."),
            (question, "resolved", "The retrieval question is resolved."),
            (blocker, "resolved", "The provider blocker is resolved."),
        ):
            self._apply(
                message,
                {
                    "action": "transition_record",
                    "target_record_id": record.id,
                    "target_state": state,
                    "source_quote": message,
                },
                candidates=[record],
            )

        self.assertEqual(self.store.get_project_record(task.id).state, "completed")
        self.assertEqual(self.store.get_project_record(question.id).state, "resolved")
        self.assertEqual(self.store.get_project_record(blocker.id).state, "resolved")

    def test_invalid_transition_and_unsupplied_target_write_nothing(self) -> None:
        milestone = self._record("milestone", "completed", "Phase 6 complete.")
        other_project = self.store.save_project(Project(id="project-other", name="Other"))
        foreign_task = ProjectRecord(
            id="foreign-task",
            project_id=other_project.id,
            conversation_id=self.conversation.id,
            kind="task",
            content="Finish the other project task.",
            state="active",
            source_message_id="unused",
            source_quote="Finish the other project task.",
        )

        with self.assertRaisesRegex(ValueError, "lifecycle transition"):
            self._apply(
                "Mark the milestone complete.",
                {
                    "action": "transition_record",
                    "target_record_id": milestone.id,
                    "target_state": "completed",
                    "source_quote": "Mark the milestone complete.",
                },
                candidates=[milestone],
            )
        with self.assertRaisesRegex(ValueError, "not supplied as an active candidate"):
            self._apply(
                "Finish the task.",
                {
                    "action": "transition_record",
                    "target_record_id": foreign_task.id,
                    "target_state": "completed",
                    "source_quote": "Finish the task.",
                },
            )

        self.assertEqual(self.store.get_project_record(milestone.id).state, "completed")
        for message_id in ("message-mark-the-milestone-complete", "message-finish-the-task"):
            self.assertEqual(self.store.list_project_change_operations(message_id), [])

    def test_a_quote_not_in_its_message_falls_back_to_the_whole_message(self) -> None:
        committed, _ = self._apply(
            "Thanks for the summary.",
            {
                "action": "create_record",
                "kind": "task",
                "content": "Implement Phase 6.",
                "source_quote": "Implement Phase 6.",
            },
        )

        record = self.store.get_project_record(committed[0].result["record_id"])
        self.assertEqual(record.source_quote, "Thanks for the summary.")

    def test_a_quote_may_come_from_an_earlier_user_message_in_the_context(self) -> None:
        pasted = self.store.save_message(
            Message(
                id="pasted-minutes",
                conversation_id=self.conversation.id,
                role="user",
                content="Minutes: finish the driver check. Update the deployment guide.",
            )
        )
        current = self.store.save_message(
            Message(id="continue", conversation_id=self.conversation.id, role="user", content="Keep going.")
        )
        change = ProjectChange.model_validate(
            {
                "action": "create_record",
                "kind": "task",
                "content": "Update the deployment guide.",
                "source_quote": "Update the deployment guide.",
                "source_message_id": pasted.id,
            }
        )

        committed, _ = apply_validated_changes(
            self.store, self.project, current, [change], [], (pasted, current)
        )
        undone = self.store.undo_project_change_operation(committed[0].id)

        record = self.store.get_project_record(committed[0].result["record_id"])
        self.assertEqual((record.source_message_id, record.source_quote), (pasted.id, "Update the deployment guide."))
        self.assertEqual(committed[0].source_message_id, current.id)
        self.assertEqual(undone.status, "undone")

    def test_a_change_stored_before_phase_13_still_commits_and_undoes(self) -> None:
        message = self.store.save_message(
            Message(
                id="stored-before",
                conversation_id=self.conversation.id,
                role="user",
                content="Implement the stored change.",
            )
        )
        operation = self.store.save_project_change_operation(
            ProjectChangeOperation(
                id="stored-before-operation",
                source_message_id=message.id,
                project_id=self.project.id,
                change_index=0,
                change={
                    "action": "create_record",
                    "kind": "task",
                    "content": "Implement the stored change.",
                    "source_quote": "Implement the stored change.",
                },
                status="prepared",
            )
        )

        committed = self.store.commit_prepared_project_change(operation.id)
        record = self.store.get_project_record(committed.result["record_id"])

        self.assertEqual(record.source_message_id, message.id)
        self.assertEqual(self.store.undo_project_change_operation(operation.id).status, "undone")

    def test_an_identical_active_record_is_not_created_twice(self) -> None:
        existing = self._record("task", "active", "Implement Phase 6.")

        with self.assertRaisesRegex(ValueError, "identical active project record"):
            self._apply(
                "Implement Phase 6.",
                {
                    "action": "create_record",
                    "kind": "task",
                    "content": "implement  phase 6.",
                    "source_quote": "Implement Phase 6.",
                },
                candidates=[existing],
            )

    def test_supersession_is_reviewed_then_can_be_approved_and_undone(self) -> None:
        previous = self._record("decision", "active", "Use SQLite.")
        message = "We replace the old database decision with SQLite and FTS5."

        committed, reviewed = self._apply(
            message,
            {
                "action": "supersede_record",
                "target_record_id": previous.id,
                "kind": "decision",
                "content": "Use SQLite with FTS5 before adding a vector store.",
                "source_quote": message,
            },
            candidates=[previous],
        )

        self.assertEqual(committed, [])
        self.assertEqual(reviewed[0].status, "review")
        self.assertEqual(self.store.get_project_record(previous.id).state, "active")
        approved = self.store.approve_project_change_operation(reviewed[0].id)
        replacement = self.store.get_project_record(approved.result["record_id"])
        self.assertEqual(approved.status, "committed")
        self.assertEqual(self.store.get_project_record(previous.id).state, "superseded")
        self.assertEqual(replacement.supersedes_record_id, previous.id)

        undone = self.store.undo_project_change_operation(approved.id)
        self.assertEqual(undone.status, "undone")
        self.assertEqual(self.store.get_project_record(previous.id).state, "active")
        self.assertEqual(self.store.get_project_record(replacement.id).state, "deleted")

    def _apply(self, content: str, change: dict, candidates: list[ProjectRecord] = ()):
        message = self._message(content)
        return apply_validated_changes(
            self.store,
            self.project,
            message,
            [ProjectChange.model_validate(change)],
            list(candidates),
        )

    def _message(self, content: str) -> Message:
        message_id = "message-" + "-".join(content.lower().strip(".").split())
        try:
            return self.store.get_message(message_id)
        except KeyError:
            return self.store.save_message(
                Message(
                    id=message_id,
                    conversation_id=self.conversation.id,
                    role="user",
                    content=content,
                )
            )

    def _record(self, kind: str, state: str, content: str) -> ProjectRecord:
        message = self._message(content)
        return self.store.save_project_record(
            ProjectRecord(
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind=kind,
                content=content,
                state=state,
                source_message_id=message.id,
                source_quote=content,
            )
        )
