import json
import tempfile
import unittest
from pathlib import Path

from captureflow.llm import ModelError
from captureflow.models import Conversation, Message, Project, ProjectAnswerDecision, ProjectRecord
from captureflow.retrieval import (
    FrozenRetrievalCase,
    classify_project_query,
    evaluate_retrieval,
    retrieve_project_records,
)
from captureflow.retrieval_service import ProjectRetrievalService
from captureflow.storage import CaptureStore


class ScriptedRetrievalModel:
    def __init__(self, decisions=(), fail=False) -> None:
        self.decisions = list(decisions)
        self.fail = fail
        self.calls = []

    def answer_project_query(
        self, project, recent_messages, retrieved_records, answer_kind, context_truncated
    ):
        self.calls.append(
            {
                "project_id": project.id,
                "messages": recent_messages,
                "records": retrieved_records,
                "answer_kind": answer_kind,
                "context_truncated": context_truncated,
            }
        )
        if self.fail:
            raise ModelError("retrieval responder unavailable")
        return self.decisions.pop(0)


class ProjectRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "captureflow.sqlite3"
        self.store = CaptureStore(self.database_path)
        self.store.initialize()
        self.project = self.store.save_project(Project(id="project-retrieval", name="CaptureFlow"))
        self.other_project = self.store.save_project(Project(id="project-other-retrieval", name="Other"))
        self.conversation = self._conversation("conversation-retrieval", self.project.id)
        self.later_conversation = self._conversation("conversation-retrieval-later", self.project.id)
        self.other_conversation = self._conversation("conversation-other-retrieval", self.other_project.id)
        self._seed_frozen_records()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_frozen_cases_have_full_recall_at_k_and_report_no_failures(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "project_retrieval_cases.json"
        cases = [
            FrozenRetrievalCase(
                name=payload["name"],
                query=payload["query"],
                mode=payload["mode"],
                expected_record_ids=tuple(payload["expected_record_ids"]),
            )
            for payload in json.loads(fixture_path.read_text(encoding="utf-8"))
        ]

        evaluation = evaluate_retrieval(
            cases,
            lambda query, mode: [
                record.id
                for record in retrieve_project_records(self.store, self.project.id, query, mode)
            ],
        )

        self.assertEqual(evaluation.recall_at_k, 1.0)
        self.assertEqual(evaluation.failures, ())

    def test_retrieval_is_project_isolated_and_excludes_superseded_current_state(self) -> None:
        other_record = self._record(
            "other-sqlite", "decision", "active", "Use SQLite in the other project.", self.other_conversation
        )
        current = retrieve_project_records(self.store, self.project.id, "SQLite status?", "current")
        history = retrieve_project_records(self.store, self.project.id, "old database decision", "history")

        self.assertIn("decision-sqlite", [record.id for record in current])
        self.assertNotIn("decision-old-database", [record.id for record in current])
        self.assertNotIn(other_record.id, [record.id for record in current])
        self.assertIn("decision-old-database", [record.id for record in history])

    def test_query_classification_is_read_only_and_deterministic(self) -> None:
        self.assertEqual(classify_project_query("现在做到哪里了？"), "current")
        self.assertEqual(classify_project_query("Why did we choose SQLite?"), "history")
        self.assertIsNone(classify_project_query("Implement Phase 7."))

    def test_cross_conversation_answer_persists_only_valid_source_cards(self) -> None:
        model = ScriptedRetrievalModel(
            [
                ProjectAnswerDecision(
                    reply="Phase 7 is the active task, supported by the tracker.",
                    answer_kind="current_status",
                    supporting_record_ids=["task-phase-seven", "milestone-phase-six"],
                )
            ]
        )
        turn = ProjectRetrievalService(self.store, model).submit(
            self.later_conversation.id, "What is the project status?", "current", "retrieval-turn"
        )

        self.assertEqual(turn.outcome, "grounded_answer")
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(
            [record.id for record in self.store.list_project_answer_sources(turn.assistant_message.id)],
            ["task-phase-seven", "milestone-phase-six"],
        )
        self.assertIn("task-phase-seven", [record["id"] for record in model.calls[0]["records"]])
        self.assertEqual(model.calls[0]["answer_kind"], "current_status")

        replay = ProjectRetrievalService(self.store, model).submit(
            self.later_conversation.id, "What is the project status?", "current", "retrieval-turn"
        )
        self.assertEqual(replay.model_calls, 0)

    def test_history_answer_includes_events_but_cannot_cite_an_unretrieved_record(self) -> None:
        model = ScriptedRetrievalModel(
            [
                ProjectAnswerDecision(
                    reply="SQLite was selected for the local project.",
                    answer_kind="history",
                    supporting_record_ids=["decision-sqlite"],
                ),
                ProjectAnswerDecision(
                    reply="Invalid source.",
                    answer_kind="history",
                    supporting_record_ids=["other-sqlite"],
                ),
            ]
        )
        first = ProjectRetrievalService(self.store, model).submit(
            self.later_conversation.id, "Why did we choose SQLite?", "history", "history-valid"
        )
        second = ProjectRetrievalService(self.store, model).submit(
            self.later_conversation.id, "Why did we choose SQLite again?", "history", "history-invalid"
        )

        self.assertEqual(first.outcome, "grounded_answer")
        self.assertIn("events", model.calls[0]["records"][0])
        self.assertEqual(second.outcome, "model_or_validation_failure")
        self.assertEqual(self.store.list_project_answer_sources(second.assistant_message.id), [])

    def test_missing_evidence_returns_unknown_without_a_model_call(self) -> None:
        empty_project = self.store.save_project(Project(id="project-empty", name="Empty"))
        empty_conversation = self._conversation("conversation-empty", empty_project.id)
        model = ScriptedRetrievalModel()

        turn = ProjectRetrievalService(self.store, model).submit(
            empty_conversation.id, "What is the status?", "current", "missing-evidence"
        )

        self.assertEqual(turn.outcome, "insufficient_context")
        self.assertEqual(turn.model_calls, 0)
        self.assertIn("do not have enough stored", turn.assistant_message.content)

    def _conversation(self, conversation_id: str, project_id: str) -> Conversation:
        conversation = self.store.save_conversation(Conversation(id=conversation_id, title=conversation_id))
        self.store.assign_conversation_project(conversation.id, project_id)
        return conversation

    def _seed_frozen_records(self) -> None:
        self._record("task-phase-seven", "task", "active", "Implement Phase 7 retrieval.", self.later_conversation)
        self._record("decision-sqlite", "decision", "active", "Use SQLite for the local project database.", self.conversation)
        self._record("constraint-local", "constraint", "active", "Keep the project local and single-user.", self.conversation)
        self._record("question-evaluation", "open_question", "active", "How should we evaluate retrieval?", self.conversation)
        self._record("blocker-live-run", "blocker", "active", "Live model acceptance is still blocked.", self.conversation)
        self._record("milestone-phase-six", "milestone", "completed", "Phase 6 implementation completed.", self.conversation)
        self._record(
            "decision-old-database",
            "decision",
            "superseded",
            "Use PostgreSQL as the old database decision.",
            self.conversation,
        )

    def _record(
        self, record_id: str, kind: str, state: str, content: str, conversation: Conversation
    ) -> ProjectRecord:
        message = self.store.save_message(
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
                source_message_id=message.id,
                source_quote=content,
            )
        )
