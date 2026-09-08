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
    GroundingReport,
    MemoryAgentDraft,
    MemoryItem,
    MemoryProposal,
    OrganizerDraft,
    PlanStep,
    TaskCandidate,
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
        self.assertEqual(self.store.get_checkpoint(result.id).state, "committed")

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

    def test_credential_disclosure_is_inspected_without_a_task_or_memory_model_call(self) -> None:
        source_text = "Subject: do not retain this\n\nMy production API key is cf_live_EXAMPLE_NOT_A_REAL_SECRET_001."
        capture = self.store.save_capture(
            Capture(id="capture-sensitive", conversation_id="conversation-1", source_text=source_text)
        )
        model = ScriptedModel(plans=[], drafts=[], reports=[], memory_drafts=[])

        result = PhaseThreeWorkflow(model, self.service).run(
            capture_id=capture.id,
            source_text=source_text,
            run_id="run-sensitive",
            reference_date=date(2026, 9, 8),
            timezone="America/Vancouver",
        )

        self.assertEqual(model.calls, [])
        self.assertEqual(result.status, "completed")
        self.assertEqual([step.action for step in result.plan.steps], ["inspect_memory"])
        self.assertIsNone(result.output)
        self.assertEqual(result.memory_decisions, [])
        self.assertEqual(self.store.list_memories(), [])

    def test_memory_proposal_citing_an_extracted_task_is_omitted(self) -> None:
        source_text = (
            "I want CaptureFlow to turn customer-interview transcripts into a concise decision "
            "log. Please create a task to sketch the first screen by 2026-10-20."
        )
        capture = self.store.save_capture(
            Capture(id="capture-task-and-idea", conversation_id="conversation-1", source_text=source_text)
        )
        task_span = "Please create a task to sketch the first screen by 2026-10-20."
        plan = CapturePlan(
            steps=[
                PlanStep(
                    id="idea-1",
                    action="inspect_memory",
                    source_span="I want CaptureFlow to turn customer-interview transcripts into a concise decision log.",
                    reason="The capture contains a durable product idea.",
                ),
                PlanStep(
                    id="task-1",
                    action="extract_task",
                    source_span=task_span,
                    reason="The capture requests an actionable task.",
                ),
            ]
        )
        model = ScriptedModel(
            plans=[plan],
            drafts=[OrganizerDraft(tasks=[TaskCandidate(text="Sketch the first screen", source_quote=task_span)])],
            reports=[GroundingReport(verdict="pass")],
            memory_drafts=[
                MemoryAgentDraft(
                    proposals=[
                        MemoryProposal(
                            memory_type="idea",
                            content="Wants CaptureFlow to turn customer-interview transcripts into a concise decision log.",
                            source_quote="I want CaptureFlow to turn customer-interview transcripts into a concise decision log.",
                            confidence=1,
                            is_explicit=True,
                        ),
                        MemoryProposal(
                            memory_type="decision",
                            content="Create a task to sketch the first screen by 2026-10-20.",
                            source_quote=task_span,
                            confidence=1,
                            is_explicit=True,
                        ),
                    ]
                )
            ],
        )

        result = PhaseThreeWorkflow(model, self.service).run(
            capture_id=capture.id,
            source_text=source_text,
            run_id="run-task-and-idea",
            reference_date=date(2026, 9, 8),
            timezone="America/Vancouver",
        )

        self.assertEqual([task.text for task in result.output.tasks], ["Sketch the first screen"])
        self.assertEqual(
            [(decision["decision"], decision["proposal"]["memory_type"])
            for decision in result.memory_decisions],
            [("review", "idea")],
        )
        self.assertEqual(len(self.store.list_pending_reviews()), 1)

    def test_prepared_checkpoint_resumes_after_restart_without_duplicate_effects(self) -> None:
        class InterruptAfterTaskCommit(MemoryService):
            def commit_tasks(self, run_id, capture_id, tasks):
                committed = super().commit_tasks(run_id, capture_id, tasks)
                raise RuntimeError("simulated process interruption")

        source_text = "I prefer concise meeting notes. Please send the minutes."
        capture = self.store.save_capture(
            Capture(
                id="capture-resume",
                conversation_id="conversation-1",
                source_text=source_text,
            )
        )
        plan = CapturePlan(
            steps=[
                PlanStep(
                    id="task-1",
                    action="extract_task",
                    source_span="Please send the minutes.",
                    reason="The capture contains an explicit request.",
                ),
                PlanStep(
                    id="memory-1",
                    action="inspect_memory",
                    source_span="I prefer concise meeting notes.",
                    reason="The capture contains a durable preference.",
                ),
            ]
        )
        model = ScriptedModel(
            plans=[plan],
            drafts=[
                OrganizerDraft(
                    tasks=[
                        TaskCandidate(
                            text="Send the minutes.",
                            source_quote="Please send the minutes.",
                        )
                    ]
                )
            ],
            reports=[GroundingReport(verdict="pass")],
            memory_drafts=[
                MemoryAgentDraft(
                    proposals=[
                        MemoryProposal(
                            memory_type="preference",
                            content="Prefers concise meeting notes.",
                            source_quote="I prefer concise meeting notes.",
                            confidence=0.95,
                            is_explicit=True,
                        )
                    ]
                )
            ],
        )
        interrupted = PhaseThreeWorkflow(model, InterruptAfterTaskCommit(self.store))

        with self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
            interrupted.run(
                capture_id=capture.id,
                source_text=source_text,
                run_id="run-resume",
                reference_date=date(2026, 9, 7),
                timezone="America/Vancouver",
            )

        self.assertEqual(self.store.get_checkpoint("run-resume").state, "prepared")
        self.assertEqual(len(self.store.list_tasks_for_capture(capture.id)), 1)
        self.assertEqual(len(self.store.list_memories()), 1)

        restarted_store = CaptureStore(self.store.database_path)
        restarted_store.initialize()
        resume_model = ScriptedModel(plans=[], drafts=[], reports=[])
        restarted = PhaseThreeWorkflow(resume_model, MemoryService(restarted_store))

        result = restarted.resume("run-resume")
        restarted.resume("run-resume")

        self.assertEqual(result.status, "completed")
        self.assertEqual(resume_model.calls, [])
        self.assertEqual(restarted_store.get_checkpoint("run-resume").state, "committed")
        self.assertEqual(len(restarted_store.list_tasks_for_capture(capture.id)), 1)
        self.assertEqual(len(restarted_store.list_memories()), 1)
