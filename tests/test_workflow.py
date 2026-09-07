import unittest
from datetime import date

from captureflow.llm import ScriptedModel
from captureflow.models import CapturePlan, GroundingReport, OrganizerDraft, PlanStep
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

    def test_meeting_task_revision_cites_problem_action_and_deadline_together(self) -> None:
        source = "背景：上线临近；问题：接口延迟阻塞联调；待办：P0，9/8修复交付。"
        plan = CapturePlan(
            steps=[
                PlanStep(
                    id="task",
                    action="extract_task",
                    source_span="问题：接口延迟阻塞联调；待办：P0，9/8修复交付。",
                    reason="The capture identifies a blocking issue and a P0 delivery task.",
                )
            ]
        )
        initial_draft = OrganizerDraft.model_validate(
            {
                "tasks": [
                    {
                        "text": "完成接口延迟修复交付",
                        "source_quote": "待办：P0，9/8修复交付。",
                        "due_expression": "9/8",
                    }
                ]
            }
        )
        revised_draft = OrganizerDraft.model_validate(
            {
                "tasks": [
                    {
                        "text": "完成接口延迟修复交付",
                        "source_quote": "问题：接口延迟阻塞联调；待办：P0，9/8修复交付。",
                        "due_expression": "9/8",
                    }
                ]
            }
        )
        revise = GroundingReport.model_validate(
            {
                "verdict": "revise",
                "feedback": [
                    {
                        "item_kind": "task",
                        "item_index": 0,
                        "reason": "Include the blocking issue in the cited source quote.",
                    }
                ],
            }
        )
        model = ScriptedModel(
            plans=[plan], drafts=[initial_draft, revised_draft], reports=[revise, PASS]
        )

        result = PhaseTwoWorkflow(model).run(
            capture_id="capture-1",
            source_text=source,
            run_id="run-1",
            reference_date=date(2026, 9, 6),
            timezone="America/Vancouver",
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output.tasks[0].resolved_due_date, date(2026, 9, 8))
        self.assertIn("接口延迟阻塞联调", result.output.tasks[0].source_quote)

    def test_critic_receives_due_expression_but_not_typed_date_resolution(self) -> None:
        model = ScriptedModel(plans=[PLAN], drafts=[DRAFT], reports=[PASS])

        result = self.run_workflow(model)

        critic_task = model.critique_drafts[0].tasks[0]
        self.assertEqual(result.output.tasks[0].resolved_due_date, date(2026, 9, 9))
        self.assertEqual(critic_task.due_expression, "next Wednesday")
        self.assertIsNone(critic_task.resolved_due_date)
        self.assertEqual(critic_task.date_status, "not_requested")

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

    def test_external_search_request_becomes_a_capability_clarification(self) -> None:
        model = ScriptedModel(plans=[], drafts=[], reports=[])

        result = PhaseTwoWorkflow(model).run(
            capture_id="capture-1",
            source_text="帮我搜搜邮编 91006 最近的 PetSmart 是哪家店",
            run_id="run-1",
            reference_date=date(2026, 9, 6),
            timezone="America/Vancouver",
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(model.calls, [])
        self.assertTrue(result.plan.user_clarification_required)
        self.assertIn("cannot perform web searches", result.plan.clarification_question)

    def test_task_only_plan_discards_redundant_summary_before_grounding(self) -> None:
        task_only_plan = CapturePlan(
            steps=[
                PlanStep(
                    id="task",
                    action="extract_task",
                    source_span="Ship the release next Wednesday.",
                    reason="The capture contains an explicit task.",
                ),
            ]
        )
        model = ScriptedModel(plans=[task_only_plan], drafts=[DRAFT], reports=[PASS])

        result = self.run_workflow(model)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output.summary_claims, [])
        self.assertEqual(result.output.tasks[0].text, "Ship the release.")
        self.assertIn("summaries not requested", result.warnings[0])

    def test_task_is_not_silently_accepted_when_plan_omits_extract_task(self) -> None:
        summary_only_plan = CapturePlan(
            steps=[
                PlanStep(
                    id="summary",
                    action="summarize",
                    source_span="The release checklist is ready.",
                    reason="The capture states release readiness.",
                ),
            ]
        )
        model = ScriptedModel(plans=[summary_only_plan], drafts=[DRAFT], reports=[])

        result = self.run_workflow(model)

        self.assertEqual(result.status, "failed")
        self.assertIn("tasks not requested", result.errors[0].message)

    def test_review_note_preserves_an_ambiguous_source_term(self) -> None:
        plan = CapturePlan(
            steps=[
                PlanStep(
                    id="task",
                    action="extract_task",
                    source_span="买毛条",
                    reason="The capture contains a purchase task.",
                )
            ]
        )
        draft = OrganizerDraft.model_validate(
            {
                "tasks": [{"text": "购买毛条", "source_quote": "买毛条"}],
                "review_notes": [
                    {
                        "question": "“毛条”是否需要确认？",
                        "source_quote": "毛条",
                    }
                ],
            }
        )
        model = ScriptedModel(plans=[plan], drafts=[draft], reports=[PASS])

        result = PhaseTwoWorkflow(model).run(
            capture_id="capture-1",
            source_text="周二去 PetSmart 买毛条。",
            run_id="run-1",
            reference_date=date(2026, 9, 6),
            timezone="America/Vancouver",
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output.tasks[0].text, "购买毛条")
        self.assertEqual(result.output.review_notes[0].source_quote, "毛条")

    def test_phase_two_plan_schema_does_not_expose_conditions_or_dependencies(self) -> None:
        fields = PlanStep.model_json_schema()["properties"]

        self.assertNotIn("run_if", fields)
        self.assertNotIn("depends_on", fields)
