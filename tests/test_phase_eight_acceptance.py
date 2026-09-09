import json
import tempfile
import unittest
from pathlib import Path

from captureflow.models import (
    Conversation,
    Message,
    Project,
    ProjectAnswerDecision,
    ProjectChangeOperation,
    ProjectRecord,
    ProjectTrackerDecision,
)
from captureflow.project_service import ProjectTrackerService
from captureflow.retrieval_service import ProjectRetrievalService
from captureflow.storage import CaptureStore


class AcceptanceTrackerModel:
    """A fixed scripted model for the frozen Phase 8 local acceptance sequence."""

    def propose_project_changes(self, project, recent_messages, candidate_records, context_truncated):
        user_message = recent_messages[-1]["content"]
        if user_message == "Track a task: finish Phase 8 acceptance.":
            return _tracker_decision(
                user_message,
                {
                    "action": "create_record",
                    "kind": "task",
                    "content": "Finish Phase 8 acceptance.",
                },
            )
        if user_message == "That task is complete.":
            return _tracker_decision(
                user_message,
                {
                    "action": "transition_record",
                    "target_record_id": _candidate_id(candidate_records, "Finish Phase 8 acceptance."),
                    "target_state": "completed",
                },
            )
        if user_message == "Decision: keep SQLite. Rationale: this is a local single-user MVP.":
            return _tracker_decision(
                user_message,
                {
                    "action": "create_record",
                    "kind": "decision",
                    "content": "Keep SQLite.",
                    "rationale": "This is a local single-user MVP.",
                },
            )
        if user_message == "Replace that decision: use SQLite FTS5 before any vector store.":
            return _tracker_decision(
                user_message,
                {
                    "action": "supersede_record",
                    "target_record_id": _candidate_id(candidate_records, "Keep SQLite."),
                    "kind": "decision",
                    "content": "Use SQLite FTS5 before any vector store.",
                },
            )
        if user_message == "Blocker: do not run paid live acceptance without approval.":
            return _tracker_decision(
                user_message,
                {
                    "action": "create_record",
                    "kind": "blocker",
                    "content": "Do not run paid live acceptance without approval.",
                },
            )
        if user_message == "The approval blocker is resolved.":
            return _tracker_decision(
                user_message,
                {
                    "action": "transition_record",
                    "target_record_id": _candidate_id(
                        candidate_records, "Do not run paid live acceptance without approval."
                    ),
                    "target_state": "resolved",
                },
            )
        if user_message == "Thanks for the update.":
            return ProjectTrackerDecision(reply="You are welcome.", turn_kind="follow_up")
        raise AssertionError(f"unexpected frozen tracker input: {user_message}")


class AcceptanceRetrievalModel:
    def answer_project_query(
        self, project, recent_messages, retrieved_records, answer_kind, context_truncated
    ):
        if answer_kind == "current_status":
            record = retrieved_records[0]
            return ProjectAnswerDecision(
                reply=f"Current tracker state includes: {record['content']}",
                answer_kind="current_status",
                supporting_record_ids=[record["id"]],
            )
        record = next(record for record in retrieved_records if record["content"] == "Keep SQLite.")
        return ProjectAnswerDecision(
            reply="SQLite was retained as the local single-user MVP decision.",
            answer_kind="history",
            supporting_record_ids=[record["id"]],
        )


class PhaseEightAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "captureflow.sqlite3"
        self.store = CaptureStore(self.database_path)
        self.store.initialize()
        self.project = self.store.save_project(Project(id="phase-eight-project", name="CaptureFlow"))
        self.conversation = self._conversation("phase-eight-main", self.project.id)
        self.fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "phase8_tracker_acceptance.json").read_text(
                encoding="utf-8"
            )
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_frozen_sequence_tracks_captureflow_across_conversations(self) -> None:
        self.assertEqual(
            [case["name"] for case in self.fixture["cases"]],
            [
                "task_create_and_follow_up_completion",
                "decision_rationale_and_supersession",
                "blocker_create_and_resolution",
                "cross_conversation_current_status",
                "cross_conversation_history",
                "deterministic_inaccurate_record_correction",
                "casual_turn_no_record",
                "two_project_isolation",
                "restart_prepared_transition",
            ],
        )
        tracker = ProjectTrackerService(self.store, AcceptanceTrackerModel())

        task = tracker.submit(
            self.conversation.id, "Track a task: finish Phase 8 acceptance.", "phase8-task"
        )
        completed = tracker.submit(
            self.conversation.id, "That task is complete.", "phase8-task-complete"
        )
        self.assertEqual(self.store.get_project_record(task.operations[0].result["record_id"]).state, "completed")
        self.assertEqual(completed.operations[0].status, "committed")

        decision = tracker.submit(
            self.conversation.id,
            "Decision: keep SQLite. Rationale: this is a local single-user MVP.",
            "phase8-decision",
        )
        original_decision = self.store.get_project_record(decision.operations[0].result["record_id"])
        replacement = tracker.submit(
            self.conversation.id,
            "Replace that decision: use SQLite FTS5 before any vector store.",
            "phase8-supersede",
        )
        self.assertEqual(replacement.operations[0].status, "review")
        approved = self.store.approve_project_change_operation(replacement.operations[0].id)
        self.assertEqual(self.store.get_project_record(original_decision.id).state, "superseded")
        self.assertEqual(self.store.get_project_record(approved.result["record_id"]).state, "active")

        blocker = tracker.submit(
            self.conversation.id,
            "Blocker: do not run paid live acceptance without approval.",
            "phase8-blocker",
        )
        resolved = tracker.submit(
            self.conversation.id,
            "The approval blocker is resolved.",
            "phase8-blocker-resolved",
        )
        self.assertEqual(self.store.get_project_record(blocker.operations[0].result["record_id"]).state, "resolved")
        self.assertEqual(resolved.operations[0].status, "committed")

        inaccurate = self._record("phase8-inaccurate", "task", "active", "Incorrect deployment claim.")
        self.assertEqual(self.store.delete_project_record(inaccurate.id).state, "deleted")
        before_casual = len(self.store.list_project_records(self.project.id, states=("active", "completed", "resolved", "superseded")))
        casual = tracker.submit(
            self.conversation.id, "Thanks for the update.", "phase8-casual"
        )
        self.assertEqual(casual.outcome, "no_change")
        self.assertEqual(
            len(self.store.list_project_records(self.project.id, states=("active", "completed", "resolved", "superseded"))),
            before_casual,
        )

        later_conversation = self._conversation("phase-eight-later", self.project.id)
        retrieval = ProjectRetrievalService(self.store, AcceptanceRetrievalModel())
        status = retrieval.submit(
            later_conversation.id, "现在做到哪里了？", "current", "phase8-status"
        )
        history = retrieval.submit(
            later_conversation.id, "Why did we keep SQLite?", "history", "phase8-history"
        )
        self.assertEqual(status.outcome, "grounded_answer")
        self.assertEqual(history.outcome, "grounded_answer")
        self.assertIn(
            status.records[0].id,
            {record.id for record in self.store.list_project_records(self.project.id)},
        )
        self.assertEqual(self.store.list_project_answer_sources(history.assistant_message.id)[0].id, original_decision.id)

        other_project = self.store.save_project(Project(id="phase-eight-other", name="Other"))
        other_conversation = self._conversation("phase-eight-other-chat", other_project.id)
        other_record = self._record(
            "phase8-other-task", "task", "active", "Only the other project task.", other_conversation
        )
        other_status = ProjectRetrievalService(self.store, AcceptanceRetrievalModel()).submit(
            other_conversation.id,
            "What is the status of the other project?",
            "current",
            "phase8-other-status",
        )
        self.assertEqual(other_status.records[0].id, other_record.id)
        self.assertNotIn(other_record.id, {record.id for record in self.store.list_project_records(self.project.id)})

        prepared_source = self.store.save_message(
            Message(
                id="phase8-prepared-source",
                conversation_id=self.conversation.id,
                role="user",
                content="Resume the prepared local tracker change.",
            )
        )
        prepared = self.store.save_project_change_operation(
            ProjectChangeOperation(
                id="phase8-prepared-operation",
                source_message_id=prepared_source.id,
                project_id=self.project.id,
                change_index=0,
                change={
                    "action": "create_record",
                    "kind": "open_question",
                    "content": "What is the next acceptance step?",
                    "source_quote": prepared_source.content,
                },
                status="prepared",
            )
        )
        restarted_store = CaptureStore(self.database_path)
        restarted_store.initialize()
        resumed = ProjectTrackerService(restarted_store, AcceptanceTrackerModel()).submit(
            self.conversation.id, prepared_source.content, prepared_source.id
        )
        self.assertEqual(resumed.outcome, "resumed_prepared_change")
        self.assertEqual(restarted_store.get_project_change_operation(prepared.id).status, "committed")

    def _conversation(self, conversation_id: str, project_id: str) -> Conversation:
        conversation = self.store.save_conversation(Conversation(id=conversation_id, title=conversation_id))
        self.store.assign_conversation_project(conversation.id, project_id)
        return conversation

    def _record(
        self,
        record_id: str,
        kind: str,
        state: str,
        content: str,
        conversation: Conversation | None = None,
    ) -> ProjectRecord:
        conversation = conversation or self.conversation
        source = self.store.save_message(
            Message(
                id=f"source-{record_id}",
                conversation_id=conversation.id,
                role="user",
                content=content,
            )
        )
        project = self.store.get_conversation_project(conversation.id)
        return self.store.save_project_record(
            ProjectRecord(
                id=record_id,
                project_id=project.id,
                conversation_id=conversation.id,
                kind=kind,
                content=content,
                state=state,
                source_message_id=source.id,
                source_quote=content,
            )
        )


def _tracker_decision(source_quote: str, change: dict) -> ProjectTrackerDecision:
    return ProjectTrackerDecision(
        reply="Scripted tracker response.",
        turn_kind="follow_up",
        changes=[{**change, "source_quote": source_quote}],
    )


def _candidate_id(candidate_records: list[dict], content: str) -> str:
    return next(record["id"] for record in candidate_records if record["content"] == content)
