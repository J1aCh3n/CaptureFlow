import tempfile
import unittest
from pathlib import Path

from captureflow.memory_service import MemoryService
from captureflow.models import (
    Capture,
    CaptureRun,
    Conversation,
    MemoryAgentDraft,
    MemoryItem,
    MemoryProposal,
    TaskItem,
    WorkflowCheckpoint,
)
from captureflow.storage import CaptureStore


class MemoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        conversation = self.store.save_conversation(Conversation(id="conversation-1", title="Inbox"))
        self.capture = self.store.save_capture(
            Capture(
                id="capture-1",
                conversation_id=conversation.id,
                source_text="I prefer concise meeting notes.",
            )
        )
        self.run = self.store.save_run(
            CaptureRun(id="run-1", capture_id=self.capture.id, status="completed", model_calls=0)
        )
        self.service = MemoryService(self.store)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def proposal(self, **changes) -> MemoryProposal:
        values = {
            "memory_type": "preference",
            "content": "Prefers concise meeting notes.",
            "source_quote": "I prefer concise meeting notes.",
            "confidence": 0.95,
            "is_explicit": True,
        }
        values.update(changes)
        return MemoryProposal(**values)

    def test_explicit_low_risk_preference_is_auto_written_idempotently(self) -> None:
        draft = MemoryAgentDraft(proposals=[self.proposal()])

        first = self.service.apply_draft(
            self.run.id, self.capture.id, self.capture.source_text, draft, []
        )
        second = self.service.apply_draft(
            self.run.id, self.capture.id, self.capture.source_text, draft, []
        )

        self.assertEqual(first[0].decision, "auto")
        self.assertEqual(second[0].decision, "auto")
        memories = self.store.list_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].write_mode, "auto")
        self.assertEqual(self.store.list_memory_events(memories[0].id)[0].action, "created_auto")

    def test_secret_is_blocked_without_a_write(self) -> None:
        proposal = self.proposal(
            memory_type="fact",
            content="My API key is sk-test.",
            source_quote="My API key is sk-test.",
        )
        decisions = self.service.apply_draft(
            self.run.id, self.capture.id, "My API key is sk-test.", MemoryAgentDraft(proposals=[proposal]), []
        )

        self.assertEqual(decisions[0].decision, "blocked")
        self.assertEqual(self.store.list_memories(), [])

    def test_conflict_requires_review_then_can_supersede(self) -> None:
        existing = self.store.save_memory(
            MemoryItem(
                id="memory-old",
                memory_type="preference",
                content="Prefers detailed meeting notes.",
                source_capture_id=self.capture.id,
                source_quote="I prefer concise meeting notes.",
                write_mode="reviewed",
                confidence=1,
            ),
            event_action="created_reviewed",
        )
        proposal = self.proposal(related_memory_ids=[existing.id])

        decisions = self.service.apply_draft(
            self.run.id,
            self.capture.id,
            self.capture.source_text,
            MemoryAgentDraft(proposals=[proposal]),
            [existing],
        )
        review = self.store.list_pending_reviews()[0]
        replacement = self.service.approve_review(review.id, supersede_conflicts=True)

        self.assertEqual(decisions[0].decision, "review")
        self.assertEqual(self.store.get_review(review.id).status, "approved")
        self.assertEqual(replacement.write_mode, "reviewed")
        self.assertEqual(self.store.get_memory(existing.id).status, "superseded")

    def test_auto_memory_can_be_undone_deleted_and_restored(self) -> None:
        self.service.apply_draft(
            self.run.id,
            self.capture.id,
            self.capture.source_text,
            MemoryAgentDraft(proposals=[self.proposal()]),
            [],
        )
        memory_id = self.store.list_memories()[0].id

        undone = self.service.undo_auto_memory(memory_id)
        restored = self.service.restore_memory(memory_id)
        deleted = self.service.delete_memory(memory_id)

        self.assertEqual(undone.status, "deleted")
        self.assertEqual(restored.status, "active")
        self.assertEqual(deleted.status, "deleted")

    def test_unchanged_edit_does_not_create_a_replacement(self) -> None:
        original = self.store.save_memory(
            MemoryItem(
                id="memory-unchanged",
                memory_type="preference",
                content="Prefers concise meeting notes.",
                source_capture_id=self.capture.id,
                source_quote=self.capture.source_text,
                write_mode="reviewed",
                confidence=1,
            ),
            event_action="created_reviewed",
        )

        result = self.service.edit_memory(original.id, original.content)

        self.assertEqual(result.id, original.id)
        self.assertEqual(self.store.get_memory(original.id).status, "active")
        self.assertEqual(len(self.store.list_memories(statuses=("active", "superseded"))), 1)
        self.assertEqual(len(self.store.list_memory_events(original.id)), 1)

    def test_conflict_approval_supersedes_memory_edited_while_review_was_pending(self) -> None:
        original = self.store.save_memory(
            MemoryItem(
                id="memory-before-review",
                memory_type="preference",
                content="Prefers concise meeting notes.",
                source_capture_id=self.capture.id,
                source_quote=self.capture.source_text,
                write_mode="reviewed",
                confidence=1,
            ),
            event_action="created_reviewed",
        )
        proposal = self.proposal(
            content="Prefers detailed meeting notes.", related_memory_ids=[original.id]
        )
        self.service.apply_draft(
            self.run.id,
            self.capture.id,
            self.capture.source_text,
            MemoryAgentDraft(proposals=[proposal]),
            [original],
        )
        review = self.store.list_pending_reviews()[0]
        edited = self.service.edit_memory(original.id, "Prefers very concise meeting notes.")

        approved = self.service.approve_review(review.id, supersede_conflicts=True)

        self.assertEqual(self.store.get_memory(original.id).status, "superseded")
        self.assertEqual(self.store.get_memory(edited.id).status, "superseded")
        self.assertEqual(approved.status, "active")
        old_event = self.store.list_memory_events(original.id)[0]
        new_event = self.store.list_memory_events(edited.id)[-1]
        self.assertEqual(old_event.details["replacement_memory_id"], edited.id)
        self.assertEqual(new_event.details["replaces_memory_id"], original.id)

    def test_conflict_approval_finds_legacy_edit_by_source_provenance(self) -> None:
        original = self.store.save_memory(
            MemoryItem(
                id="legacy-original",
                memory_type="preference",
                content="Prefers concise meeting notes.",
                source_capture_id=self.capture.id,
                source_quote=self.capture.source_text,
                write_mode="reviewed",
                confidence=1,
            ),
            event_action="created_reviewed",
        )
        proposal = self.proposal(
            content="Prefers detailed meeting notes.", related_memory_ids=[original.id]
        )
        self.service.apply_draft(
            self.run.id,
            self.capture.id,
            self.capture.source_text,
            MemoryAgentDraft(proposals=[proposal]),
            [original],
        )
        review = self.store.list_pending_reviews()[0]
        legacy_successor = self.store.save_memory(
            original.model_copy(
                update={"id": "legacy-successor", "content": "Prefers brief meeting notes."}
            ),
            event_action="edited_replacement",
        )
        self.store.set_memory_status(original.id, "superseded", "superseded_by_edit")

        approved = self.service.approve_review(review.id, supersede_conflicts=True)

        self.assertEqual(self.store.get_memory(legacy_successor.id).status, "superseded")
        self.assertEqual(approved.status, "active")

    def test_lexical_retrieval_is_bounded_and_excludes_deleted_memories(self) -> None:
        for index in range(6):
            self.store.save_memory(
                MemoryItem(
                    id=f"memory-{index}",
                    memory_type="fact",
                    content=f"Project Atlas preference {index}",
                    source_capture_id=self.capture.id,
                    source_quote="I prefer concise meeting notes.",
                    write_mode="reviewed",
                    confidence=1,
                ),
                event_action="created_reviewed",
            )
        self.store.set_memory_status("memory-0", "deleted", "deleted")

        retrieved = self.service.retrieve("Atlas preference")

        self.assertEqual(len(retrieved), 5)
        self.assertNotIn("memory-0", [item.id for item in retrieved])

    def test_checkpointed_task_commit_does_not_duplicate_on_resume(self) -> None:
        checkpoint = WorkflowCheckpoint(
            run_id=self.run.id,
            capture_id=self.capture.id,
            state="prepared",
            payload={"effect": "commit"},
        )
        self.store.save_checkpoint(checkpoint)
        task = TaskItem(
            capture_id=self.capture.id,
            run_id=self.run.id,
            text="Send the meeting notes.",
            source_quote="I prefer concise meeting notes.",
        )

        self.service.commit_tasks(self.run.id, self.capture.id, [task])
        self.service.commit_tasks(self.run.id, self.capture.id, [task])

        self.assertEqual(len(self.store.list_tasks_for_capture(self.capture.id)), 1)
        self.assertEqual(self.store.get_checkpoint(self.run.id), checkpoint)
