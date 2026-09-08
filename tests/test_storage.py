import tempfile
import unittest
from datetime import datetime, timezone
import json
from pathlib import Path

from captureflow.models import (
    Capture,
    CaptureRun,
    Conversation,
    MemoryItem,
    MemoryProposal,
    PendingMemoryReview,
    TaskItem,
    WorkflowCheckpoint,
)
from captureflow.storage import CaptureStore


class CaptureStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "captureflow.sqlite3"
        self.store = CaptureStore(self.database_path)
        self.store.initialize()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_saved_capture_is_available_after_store_restart(self) -> None:
        created_at = datetime(2026, 9, 6, tzinfo=timezone.utc)
        conversation = Conversation(id="conversation-1", title="Inbox", created_at=created_at)
        capture = Capture(
            id="capture-1",
            conversation_id=conversation.id,
            source_text="Read the project brief.",
            created_at=created_at,
        )
        self.store.save_conversation(conversation)
        self.store.save_capture(capture)

        restarted_store = CaptureStore(self.database_path)
        restarted_store.initialize()

        self.assertEqual(restarted_store.get_capture(capture.id), capture)
        self.assertEqual(restarted_store.list_captures(conversation.id), [capture])

    def test_repeated_save_with_same_capture_id_does_not_duplicate(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="conversation-1", title="Inbox"))
        capture = Capture(
            id="capture-1",
            conversation_id=conversation.id,
            source_text="Read the project brief.",
        )

        self.store.save_capture(capture)
        self.store.save_capture(capture)

        self.assertEqual(self.store.list_captures(conversation.id), [capture])

    def test_repeated_save_with_same_conversation_id_does_not_duplicate(self) -> None:
        conversation = Conversation(id="conversation-1", title="Inbox")

        self.store.save_conversation(conversation)
        self.store.save_conversation(conversation)

        self.assertEqual(self.store.list_conversations(), [conversation])

    def test_repeated_delete_is_safe(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="conversation-1", title="Inbox"))
        capture = self.store.save_capture(
            Capture(
                id="capture-1",
                conversation_id=conversation.id,
                source_text="Read the project brief.",
            )
        )

        self.assertTrue(self.store.delete_capture(capture.id))
        self.assertFalse(self.store.delete_capture(capture.id))
        self.assertEqual(self.store.list_captures(conversation.id), [])

    def test_delete_conversation_cascades_all_capture_derived_state(self) -> None:
        conversation = self.store.save_conversation(
            Conversation(id="conversation-delete", title="Delete me")
        )
        capture = self.store.save_capture(
            Capture(
                id="capture-delete",
                conversation_id=conversation.id,
                source_text="I prefer concise notes.",
            )
        )
        run = self.store.save_run(
            CaptureRun(id="run-delete", capture_id=capture.id, status="completed", model_calls=0)
        )
        task = self.store.save_task(
            TaskItem(
                id="task-delete",
                capture_id=capture.id,
                run_id=run.id,
                text="Send notes.",
                source_quote="I prefer concise notes.",
            )
        )
        memory = self.store.save_memory(
            MemoryItem(
                id="memory-delete",
                memory_type="preference",
                content="Prefers concise notes.",
                source_capture_id=capture.id,
                source_quote=capture.source_text,
                write_mode="auto",
                confidence=1,
            ),
            event_action="created_auto",
        )
        review = self.store.save_pending_review(
            PendingMemoryReview(
                id="review-delete",
                capture_id=capture.id,
                proposal=MemoryProposal(
                    memory_type="idea",
                    content="Create a note assistant.",
                    source_quote=capture.source_text,
                    confidence=1,
                    is_explicit=True,
                ),
                decision_reason="Ideas require review.",
            )
        )
        checkpoint = self.store.save_checkpoint(
            WorkflowCheckpoint(
                run_id=run.id,
                capture_id=capture.id,
                state="prepared",
                payload={"result": "prepared"},
            )
        )

        self.assertTrue(self.store.delete_conversation(conversation.id))
        self.assertFalse(self.store.delete_conversation(conversation.id))

        for getter, item_id in (
            (self.store.get_conversation, conversation.id),
            (self.store.get_capture, capture.id),
            (self.store.get_run, run.id),
            (self.store.get_task, task.id),
            (self.store.get_memory, memory.id),
            (self.store.get_review, review.id),
            (self.store.get_checkpoint, checkpoint.run_id),
        ):
            with self.assertRaises(KeyError):
                getter(item_id)
        self.assertEqual(self.store.list_memory_events(memory.id), [])

    def test_run_trace_survives_store_restart(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="conversation-1", title="Inbox"))
        capture = self.store.save_capture(
            Capture(id="capture-1", conversation_id=conversation.id, source_text="Read the brief.")
        )
        run = CaptureRun(id="run-1", capture_id=capture.id, status="completed", model_calls=0)
        self.store.save_run(run)

        restarted_store = CaptureStore(self.database_path)
        restarted_store.initialize()

        self.assertEqual(restarted_store.latest_run_for_capture(capture.id), run)
        self.assertEqual(restarted_store.list_runs_for_capture(capture.id), [run])

    def test_legacy_plan_trace_with_retired_fields_remains_readable(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="conversation-1", title="Inbox"))
        capture = self.store.save_capture(
            Capture(id="capture-1", conversation_id=conversation.id, source_text="Read the brief.")
        )
        legacy_payload = {
            "id": "run-1",
            "capture_id": capture.id,
            "status": "completed",
            "plan": {
                "steps": [
                    {
                        "id": "task",
                        "action": "extract_task",
                        "source_span": "Read the brief.",
                        "depends_on": [],
                        "run_if": None,
                        "reason": "Legacy plan.",
                    }
                ],
                "user_clarification_required": False,
                "clarification_question": None,
            },
            "output": {"summary_claims": [], "tasks": []},
            "grounding_report": None,
            "warnings": [],
            "errors": [],
            "model_calls": 3,
            "created_at": "2026-09-06T00:00:00+00:00",
        }
        with self.store._connect() as connection:
            connection.execute(
                """
                INSERT INTO capture_runs (id, capture_id, status, result_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("run-1", capture.id, "completed", json.dumps(legacy_payload), legacy_payload["created_at"]),
            )

        run = self.store.get_run("run-1")

        self.assertEqual(run.plan.steps[0].action, "extract_task")
