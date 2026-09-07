import unittest
from datetime import date

from captureflow.llm import ScriptedModel
from captureflow.models import CapturePlan, GroundingReport, OrganizerDraft, PlanStep, StepCondition
from captureflow.workflow import PhaseTwoWorkflow, validate_plan


SOURCE = "Ship the release next Wednesday. The release checklist is ready."
PLAN = CapturePlan(
    steps=[
        PlanStep(
            id="summary",
            action="summarize",
            source_span="The release checklist is ready.",
            reason="The capture states release readiness.",
        ),
        PlanStep(
            id="task",
            action="extract_task",
            source_span="Ship the release next Wednesday.",
            reason="The capture includes an explicit commitment.",
        ),
    ]
)
DRAFT = OrganizerDraft.model_validate(
    {
        "summary_claims": [
            {
                "text": "The release checklist is ready.",
                "source_quote": "The release checklist is ready.",
            }
        ],
        "tasks": [
            {
                "text": "Ship the release.",
                "source_quote": "Ship the release next Wednesday.",
                "due_expression": "next Wednesday",
            }
        ],
    }
)
PASS = GroundingReport(verdict="pass")
REVISE = GroundingReport.model_validate(
    {
        "verdict": "revise",
        "feedback": [
            {"item_kind": "summary", "item_index": 0, "reason": "Use the cited wording."}
        ],
    }
)


class PhaseTwoWorkflowTests(unittest.TestCase):
    def run_workflow(self, model: ScriptedModel, **kwargs):
        return PhaseTwoWorkflow(model).run(
            capture_id="capture-1",
            source_text=SOURCE,
            run_id="run-1",
            reference_date=date(2026, 9, 6),
            timezone="America/Vancouver",
            **kwargs,
        )

    def test_valid_plan_runs_organizer_date_tool_and_critic(self) -> None:
        model = ScriptedModel(plans=[PLAN], drafts=[DRAFT], reports=[PASS])

        result = self.run_workflow(model)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.model_calls, 3)
        self.assertEqual(model.calls, ["plan", "organize", "critique"])
        self.assertEqual(result.output.tasks[0].resolved_due_date, date(2026, 9, 9))

    def test_invalid_plan_fails_without_a_fabricated_fallback(self) -> None:
        invalid_plan = CapturePlan(
            steps=[
                PlanStep(
                    id="summary",
                    action="summarize",
                    source_span="Not in the source.",
                    reason="Invalid test plan.",
                )
            ]
        )
        model = ScriptedModel(plans=[invalid_plan], drafts=[], reports=[])

        result = self.run_workflow(model)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.errors[0].stage, "plan")
        self.assertEqual(model.calls, ["plan"])

    def test_critic_revision_runs_once_then_can_pass(self) -> None:
        model = ScriptedModel(plans=[PLAN], drafts=[DRAFT, DRAFT], reports=[REVISE, PASS])

        result = self.run_workflow(model)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.model_calls, 5)
        self.assertEqual(model.calls, ["plan", "organize", "critique", "organize", "critique"])

    def test_grounding_exhaustion_omits_generated_items(self) -> None:
        model = ScriptedModel(plans=[PLAN], drafts=[DRAFT, DRAFT], reports=[REVISE, REVISE])

        result = self.run_workflow(model)

        self.assertEqual(result.status, "grounding_failed")
        self.assertEqual(result.output.summary_claims, [])
        self.assertEqual(result.output.tasks, [])
        self.assertIn("generated items were omitted", result.warnings[-1])

    def test_date_tool_failure_is_visible_without_failing_open(self) -> None:
        model = ScriptedModel(plans=[PLAN], drafts=[DRAFT], reports=[PASS])
        workflow = PhaseTwoWorkflow(
            model,
            date_resolver=lambda expression, reference_date, timezone: (_ for _ in ()).throw(
                RuntimeError("tool timeout")
            ),
        )

        result = workflow.run(
            capture_id="capture-1",
            source_text=SOURCE,
            run_id="run-1",
            reference_date=date(2026, 9, 6),
            timezone="America/Vancouver",
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output.tasks[0].date_status, "failed")
        self.assertIn("Date tool failed", result.warnings[0])

    def test_dependency_cycle_is_rejected(self) -> None:
        cyclic_plan = CapturePlan(
            steps=[
                PlanStep(
                    id="one",
                    action="summarize",
                    source_span="Ship the release next Wednesday.",
                    depends_on=["two"],
                    reason="Cycle test.",
                ),
                PlanStep(
                    id="two",
                    action="extract_task",
                    source_span="The release checklist is ready.",
                    depends_on=["one"],
                    reason="Cycle test.",
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_plan(cyclic_plan, SOURCE)

    def test_conditional_step_is_rejected_until_its_router_exists(self) -> None:
        conditional_plan = CapturePlan(
            steps=[
                PlanStep(
                    id="task",
                    action="extract_task",
                    source_span="Ship the release next Wednesday.",
                    reason="Extract the stated task.",
                ),
                PlanStep(
                    id="summary",
                    action="summarize",
                    source_span="The release checklist is ready.",
                    depends_on=["task"],
                    run_if=StepCondition(source_step_id="task", predicate="has_tasks"),
                    reason="This must not be silently ignored.",
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "not implemented"):
            validate_plan(conditional_plan, SOURCE)
