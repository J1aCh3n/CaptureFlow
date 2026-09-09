import tempfile
import unittest
from pathlib import Path

from captureflow.llm import ModelError
from captureflow.models import (
    Conversation,
    Message,
    Project,
    ProjectChangeOperation,
    ProjectRecord,
    ProjectTrackerDecision,
)
from captureflow.project_service import ProjectTrackerService
from captureflow.storage import CaptureStore


class ScriptedTrackerModel:
    def __init__(self, decisions=(), fail=False) -> None:
        self.decisions = list(decisions)
        self.fail = fail
        self.calls = []

    def propose_project_changes(self, project, recent_messages, candidate_records, context_truncated):
        self.calls.append(
            {
                "project_id": project.id,
                "messages": recent_messages,
                "candidate_records": candidate_records,
                "context_truncated": context_truncated,
            }
        )
        if self.fail:
            raise ModelError("tracker proposer unavailable")
        return self.decisions.pop(0)


def decision(reply="Acknowledged.", turn_kind="new_subject", changes=()):
    return ProjectTrackerDecision(
        reply=reply,
        turn_kind=turn_kind,
        changes=list(changes),
    )


class ProjectTrackerServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "captureflow.sqlite3"
        self.store = CaptureStore(self.database_path)
        self.store.initialize()
        self.conversation = self.store.save_conversation(
            Conversation(id="conversation-phase-six", title="Phase six")
        )
        self.project = self.store.save_project(Project(id="project-phase-six", name="CaptureFlow"))
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
        for index, (kind, quote) in enumerate(cases):
            model = ScriptedTrackerModel(
                [
                    decision(
                        changes=(
                            {
                                "action": "create_record",
                                "kind": kind,
                                "content": quote,
                                "source_quote": quote,
                            },
                        )
                    )
                ]
            )
            turn = ProjectTrackerService(self.store, model).submit(
                self.conversation.id, quote, f"create-{index}"
            )
            self.assertEqual(turn.operations[0].status, "committed")

        records = self.store.list_project_records(self.project.id)
        self.assertEqual({record.kind for record in records}, {case[0] for case in cases})
        milestones = [record for record in records if record.kind == "milestone"]
        self.assertEqual(milestones[0].state, "completed")

    def test_completion_and_resolution_commit_only_valid_lifecycle_transitions(self) -> None:
        task = self._record("task", "active", "Implement Phase 6.")
        question = self._record("open_question", "active", "Which retrieval strategy?")
        blocker = self._record("blocker", "active", "No provider key.")
        model = ScriptedTrackerModel(
            [
                decision(
                    changes=(
                        {
                            "action": "transition_record",
                            "target_record_id": task.id,
                            "target_state": "completed",
                            "source_quote": "The implementation task is complete.",
                        },
                    )
                ),
                decision(
                    changes=(
                        {
                            "action": "transition_record",
                            "target_record_id": question.id,
                            "target_state": "resolved",
                            "source_quote": "The retrieval question is resolved.",
                        },
                    )
                ),
                decision(
                    changes=(
                        {
                            "action": "transition_record",
                            "target_record_id": blocker.id,
                            "target_state": "resolved",
                            "source_quote": "The provider blocker is resolved.",
                        },
                    )
                ),
            ]
        )
        service = ProjectTrackerService(self.store, model)
        service.submit(self.conversation.id, "The implementation task is complete.", "complete-task")
        service.submit(self.conversation.id, "The retrieval question is resolved.", "resolve-question")
        service.submit(self.conversation.id, "The provider blocker is resolved.", "resolve-blocker")

        self.assertEqual(self.store.get_project_record(task.id).state, "completed")
        self.assertEqual(self.store.get_project_record(question.id).state, "resolved")
        self.assertEqual(self.store.get_project_record(blocker.id).state, "resolved")

    def test_invalid_transition_and_invented_target_leave_tracker_unchanged(self) -> None:
        milestone = self._record("milestone", "completed", "Phase 6 complete.")
        model = ScriptedTrackerModel(
            [
                decision(
                    changes=(
                        {
                            "action": "transition_record",
                            "target_record_id": milestone.id,
                            "target_state": "completed",
                            "source_quote": "Mark the milestone complete.",
                        },
                    )
                ),
                decision(
                    changes=(
                        {
                            "action": "transition_record",
                            "target_record_id": "invented-record",
                            "target_state": "completed",
                            "source_quote": "Finish the task.",
                        },
                    )
                ),
            ]
        )
        service = ProjectTrackerService(self.store, model)
        first = service.submit(self.conversation.id, "Mark the milestone complete.", "invalid-transition")
        second = service.submit(self.conversation.id, "Finish the task.", "invented-target")

        self.assertEqual(first.outcome, "model_or_validation_failure")
        self.assertEqual(second.outcome, "model_or_validation_failure")
        self.assertEqual(self.store.get_project_record(milestone.id).state, "completed")
        self.assertEqual(self.store.list_project_change_operations(first.user_message.id), [])
        self.assertEqual(self.store.list_project_change_operations(second.user_message.id), [])

    def test_cross_project_target_is_not_available_to_the_model_or_write_path(self) -> None:
        other_conversation = self.store.save_conversation(
            Conversation(id="conversation-other-phase-six", title="Other")
        )
        other_project = self.store.save_project(Project(id="project-other-phase-six", name="Other"))
        self.store.assign_conversation_project(other_conversation.id, other_project.id)
        other_message = self.store.save_message(
            Message(
                id="other-project-source",
                conversation_id=other_conversation.id,
                role="user",
                content="Finish the other project task.",
            )
        )
        other_record = self.store.save_project_record(
            ProjectRecord(
                id="other-project-task",
                project_id=other_project.id,
                conversation_id=other_conversation.id,
                kind="task",
                content="Finish the other project task.",
                state="active",
                source_message_id=other_message.id,
                source_quote=other_message.content,
            )
        )
        model = ScriptedTrackerModel(
            [
                decision(
                    changes=(
                        {
                            "action": "transition_record",
                            "target_record_id": other_record.id,
                            "target_state": "completed",
                            "source_quote": "Finish the task here.",
                        },
                    )
                )
            ]
        )

        turn = ProjectTrackerService(self.store, model).submit(
            self.conversation.id, "Finish the task here.", "cross-project-target"
        )

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertEqual(self.store.get_project_record(other_record.id).state, "active")
        self.assertEqual(model.calls[0]["candidate_records"], [])

    def test_supersession_is_reviewed_then_can_be_approved_and_undone(self) -> None:
        previous = self._record("decision", "active", "Use SQLite.")
        model = ScriptedTrackerModel(
            [
                decision(
                    changes=(
                        {
                            "action": "supersede_record",
                            "target_record_id": previous.id,
                            "kind": "decision",
                            "content": "Use SQLite with FTS5 before adding a vector store.",
                            "source_quote": "We replace the old database decision with SQLite and FTS5.",
                        },
                    )
                )
            ]
        )
        turn = ProjectTrackerService(self.store, model).submit(
            self.conversation.id,
            "We replace the old database decision with SQLite and FTS5.",
            "supersede-decision",
        )

        review = turn.operations[0]
        self.assertEqual(review.status, "review")
        self.assertEqual(self.store.get_project_record(previous.id).state, "active")
        committed = self.store.approve_project_change_operation(review.id)
        replacement = self.store.get_project_record(committed.result["record_id"])
        self.assertEqual(committed.status, "committed")
        self.assertEqual(self.store.get_project_record(previous.id).state, "superseded")
        self.assertEqual(replacement.supersedes_record_id, previous.id)

        undone = self.store.undo_project_change_operation(committed.id)
        self.assertEqual(undone.status, "undone")
        self.assertEqual(self.store.get_project_record(previous.id).state, "active")
        self.assertEqual(self.store.get_project_record(replacement.id).state, "deleted")

    def test_small_talk_and_replayed_message_do_not_duplicate_writes(self) -> None:
        model = ScriptedTrackerModel(
            [
                decision(reply="Sounds good."),
                decision(
                    changes=(
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "Implement Phase 6.",
                            "source_quote": "Implement Phase 6.",
                        },
                    )
                ),
            ]
        )
        service = ProjectTrackerService(self.store, model)
        small_talk = service.submit(self.conversation.id, "Thanks for the summary.", "small-talk")
        first = service.submit(self.conversation.id, "Implement Phase 6.", "replayed-change")
        replay = ProjectTrackerService(self.store, model).submit(
            self.conversation.id, "Implement Phase 6.", "replayed-change"
        )

        self.assertEqual(small_talk.outcome, "no_change")
        self.assertEqual(first.model_calls, 1)
        self.assertEqual(replay.model_calls, 0)
        self.assertEqual(len(self.store.list_project_records(self.project.id)), 1)
        self.assertEqual(len(self.store.list_project_change_operations(first.user_message.id)), 1)

    def test_prepared_change_resumes_after_restart_without_a_second_model_call(self) -> None:
        source = self.store.save_message(
            Message(
                id="prepared-restart-source",
                conversation_id=self.conversation.id,
                role="user",
                content="Implement the prepared change.",
            )
        )
        operation = self.store.save_project_change_operation(
            ProjectChangeOperation(
                id="prepared-restart-operation",
                source_message_id=source.id,
                project_id=self.project.id,
                change_index=0,
                change={
                    "action": "create_record",
                    "kind": "task",
                    "content": "Implement the prepared change.",
                    "source_quote": "Implement the prepared change.",
                },
                status="prepared",
            )
        )
        restarted_store = CaptureStore(self.database_path)
        restarted_store.initialize()
        model = ScriptedTrackerModel()

        turn = ProjectTrackerService(restarted_store, model).submit(
            self.conversation.id, source.content, source.id
        )

        self.assertEqual(turn.outcome, "resumed_prepared_change")
        self.assertEqual(turn.model_calls, 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(restarted_store.get_project_change_operation(operation.id).status, "committed")
        self.assertEqual(len(restarted_store.list_project_records(self.project.id)), 1)

    def _record(self, kind: str, state: str, content: str) -> ProjectRecord:
        message = self.store.save_message(
            Message(
                id=f"source-{kind}-{state}-{len(self.store.list_messages(self.conversation.id))}",
                conversation_id=self.conversation.id,
                role="user",
                content=content,
            )
        )
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
