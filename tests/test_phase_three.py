from datetime import date
import tempfile
import unittest
from pathlib import Path

from captureflow.llm import ScriptedModel
from captureflow.memory_service import MemoryService
from captureflow.models import (
    Capture,
    CapturePlan,
    Conversation,
    MemoryAgentDraft,
    MemoryItem,
    MemoryProposal,
    PlanStep,
)
from captureflow.phase_three import PhaseThreeWorkflow
from captureflow.storage import CaptureStore


class PhaseThreeWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        conversation = self.store.save_conversation(Conversation(id="conversation-1", title="Inbox"))
        self.capture = self.store.save_capture(
            Capture(
                id="capture-1",
                conversation_id=conversation.id,
                source_text="I prefer concise meeting notes and weekly status updates.",
            )
        )
        self.service = MemoryService(self.store)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def memory_plan(self) -> CapturePlan:
        return CapturePlan(
            steps=[
                PlanStep(
                    id="memory-1",
                    action="inspect_memory",
                    source_span="I prefer concise meeting notes",
                    reason="The capture states a durable preference.",
                )
            ]
        )

    def test_memory_subgraph_retrieves_then_auto_writes_a_policy_approved_proposal(self) -> None:
        prior = self.store.save_memory(
            MemoryItem(
                id="memory-prior",
                memory_type="fact",
                content="Uses weekly status updates.",
                source_capture_id=self.capture.id,
                source_quote="weekly status updates",
                write_mode="reviewed",
                confidence=1,
            ),
            event_action="created_reviewed",
        )
        model = ScriptedModel(
            plans=[self.memory_plan()],
            drafts=[],
            reports=[],
            memory_drafts=[
                MemoryAgentDraft(
                    proposals=[
                        MemoryProposal(
                            memory_type="preference",
                            content="Prefers concise meeting notes.",
                            source_quote="I prefer concise meeting notes",
                            confidence=0.95,
                            is_explicit=True,
                        )
                    ]
                )
            ],
        )

        result = PhaseThreeWorkflow(model, self.service).run(
            capture_id=self.capture.id,
            source_text=self.capture.source_text,
            run_id="run-1",
            reference_date=date(2026, 9, 7),
            timezone="America/Vancouver",
        )

        self.assertEqual(model.calls, ["plan", "memory"])
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(result.retrieved_memory_ids, [prior.id])
        self.assertEqual(result.memory_decisions[0]["decision"], "auto")
        self.assertEqual(
            [memory.content for memory in self.store.list_memories()],
            ["Prefers concise meeting notes.", "Uses weekly status updates."],
        )

    def test_memory_model_failure_keeps_phase_two_result_and_writes_nothing(self) -> None:
        model = ScriptedModel(plans=[self.memory_plan()], drafts=[], reports=[])

        result = PhaseThreeWorkflow(model, self.service).run(
            capture_id=self.capture.id,
            source_text=self.capture.source_text,
            run_id="run-2",
            reference_date=date(2026, 9, 7),
            timezone="America/Vancouver",
        )

        self.assertEqual(result.status, "completed_with_omissions")
        self.assertEqual(result.errors[0].stage, "memory")
        self.assertEqual(self.store.list_memories(), [])

