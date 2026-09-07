from collections.abc import Callable
from datetime import date, datetime
from typing import Literal, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langgraph.graph import END, START, StateGraph

from .date_tool import DateResolution, resolve_relative_date
from .llm import ModelError, PhaseTwoModel
from .models import (
    CapturePlan,
    CaptureRun,
    GroundingIssue,
    GroundingReport,
    OrganizerDraft,
    RunError,
    TaskCandidate,
)


class PhaseTwoState(TypedDict, total=False):
    run_id: str
    capture_id: str
    source_text: str
    reference_date: date
    timezone: str
    plan: CapturePlan
    draft: OrganizerDraft
    grounding_report: GroundingReport
    revision_feedback: GroundingReport
    retries: int
    model_calls: int
    warnings: list[str]
    errors: list[RunError]
    status: Literal["running", "completed", "grounding_failed", "failed"]


def validate_plan(plan: CapturePlan, source_text: str) -> None:
    """Enforce the bounded plan contract before the graph executes it."""

    if plan.user_clarification_required:
        if plan.steps:
            raise ValueError("a clarification plan must not contain executable steps")
        return

    source = _normalize(source_text)
    step_ids = {step.id for step in plan.steps}
    if len(step_ids) != len(plan.steps):
        raise ValueError("plan step IDs must be unique")
    for step in plan.steps:
        if _normalize(step.source_span) not in source:
            raise ValueError(f"plan step {step.id} cites a source span not present in the capture")
        unknown_dependencies = set(step.depends_on) - step_ids
        if unknown_dependencies:
            raise ValueError(f"plan step {step.id} has unknown dependencies")
        if step.id in step.depends_on:
            raise ValueError(f"plan step {step.id} cannot depend on itself")
        if step.run_if:
            raise ValueError("conditional plan steps are not implemented in Phase 2")
    _assert_acyclic(plan)


def _assert_acyclic(plan: CapturePlan) -> None:
    dependencies = {step.id: set(step.depends_on) for step in plan.steps}
    resolved: set[str] = set()
    while dependencies:
        ready = {step_id for step_id, needs in dependencies.items() if needs <= resolved}
        if not ready:
            raise ValueError("plan dependencies contain a cycle")
        resolved.update(ready)
        dependencies = {step_id: needs for step_id, needs in dependencies.items() if step_id not in ready}


class PhaseTwoWorkflow:
    def __init__(
        self,
        model: PhaseTwoModel,
        date_resolver: Callable[[str, date, str], DateResolution] = resolve_relative_date,
    ) -> None:
        self.model = model
        self.date_resolver = date_resolver
        self.graph = self._build_graph()

    def run(
        self,
        capture_id: str,
        source_text: str,
        run_id: str,
        reference_date: date,
        timezone: str,
    ) -> CaptureRun:
        state = self.graph.invoke(
            {
                "run_id": run_id,
                "capture_id": capture_id,
                "source_text": source_text,
                "reference_date": reference_date,
                "timezone": timezone,
                "retries": 0,
                "model_calls": 0,
                "warnings": [],
                "errors": [],
                "status": "running",
            }
        )
        status = state["status"]
        return CaptureRun(
            id=run_id,
            capture_id=capture_id,
            status=status,
            plan=state.get("plan"),
            output=state.get("draft") if status == "completed" else OrganizerDraft(),
            grounding_report=state.get("grounding_report"),
            warnings=state["warnings"],
            errors=state["errors"],
            model_calls=state["model_calls"],
        )

    def _build_graph(self):
        graph = StateGraph(PhaseTwoState)
        graph.add_node("plan", self._plan)
        graph.add_node("organize", self._organize)
        graph.add_node("resolve_dates", self._resolve_dates)
        graph.add_node("ground", self._ground)
        graph.add_node("prepare_revision", self._prepare_revision)
        graph.add_node("complete", self._complete)
        graph.add_node("grounding_failed", self._grounding_failed)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            self._after_plan,
            {"organize": "organize", "complete": "complete", "end": END},
        )
        graph.add_edge("organize", "resolve_dates")
        graph.add_edge("resolve_dates", "ground")
        graph.add_conditional_edges(
            "ground",
            self._after_grounding,
            {
                "complete": "complete",
                "revise": "prepare_revision",
                "grounding_failed": "grounding_failed",
                "failed": END,
            },
        )
        graph.add_edge("prepare_revision", "organize")
        graph.add_edge("complete", END)
        graph.add_edge("grounding_failed", END)
        return graph.compile()

    def _plan(self, state: PhaseTwoState) -> PhaseTwoState:
        calls = state["model_calls"] + 1
        try:
            plan = self.model.plan(state["source_text"])
            validate_plan(plan, state["source_text"])
            return {"plan": plan, "model_calls": calls}
        except (ModelError, ValueError) as error:
            return self._failure(state, "plan", str(error), calls)

    def _after_plan(self, state: PhaseTwoState) -> str:
        if state["status"] == "failed":
            return "end"
        if state["plan"].user_clarification_required:
            return "complete"
        return "organize"

    def _organize(self, state: PhaseTwoState) -> PhaseTwoState:
        calls = state["model_calls"] + 1
        try:
            draft = self.model.organize(
                state["source_text"], state["plan"], state.get("revision_feedback")
            )
            self._validate_draft_actions(draft, state["plan"])
            return {"draft": draft, "model_calls": calls}
        except (ModelError, ValueError) as error:
            return self._failure(state, "organize", str(error), calls)

    def _resolve_dates(self, state: PhaseTwoState) -> PhaseTwoState:
        draft = state["draft"]
        warnings = list(state["warnings"])
        resolved_tasks: list[TaskCandidate] = []
        for task in draft.tasks:
            if not task.due_expression:
                resolved_tasks.append(
                    task.model_copy(update={"resolved_due_date": None, "date_status": "not_requested"})
                )
                continue
            try:
                resolution = self.date_resolver(
                    task.due_expression, state["reference_date"], state["timezone"]
                )
                resolved_tasks.append(
                    task.model_copy(
                        update={
                            "resolved_due_date": resolution.resolved_date,
                            "date_status": resolution.status,
                        }
                    )
                )
                if resolution.status != "resolved":
                    warnings.append(
                        f"Date '{task.due_expression}' was {resolution.status}: {resolution.explanation}"
                    )
            except Exception as error:
                resolved_tasks.append(
                    task.model_copy(update={"resolved_due_date": None, "date_status": "failed"})
                )
                warnings.append(f"Date tool failed for '{task.due_expression}': {error}")
        return {"draft": draft.model_copy(update={"tasks": resolved_tasks}), "warnings": warnings}

    def _ground(self, state: PhaseTwoState) -> PhaseTwoState:
        deterministic_feedback = self._deterministic_grounding(
            state["source_text"], state["draft"]
        )
        if deterministic_feedback:
            return {
                "grounding_report": GroundingReport(
                    verdict="revise", feedback=deterministic_feedback
                )
            }
        calls = state["model_calls"] + 1
        try:
            report = self.model.critique(state["source_text"], state["draft"])
            return {"grounding_report": report, "model_calls": calls}
        except ModelError as error:
            return self._failure(state, "grounding", str(error), calls)

    def _after_grounding(self, state: PhaseTwoState) -> str:
        if state["status"] == "failed":
            return "failed"
        if state["grounding_report"].verdict == "pass":
            return "complete"
        if state["retries"] < 1:
            return "revise"
        return "grounding_failed"

    def _prepare_revision(self, state: PhaseTwoState) -> PhaseTwoState:
        return {
            "retries": state["retries"] + 1,
            "revision_feedback": state["grounding_report"],
        }

    @staticmethod
    def _complete(state: PhaseTwoState) -> PhaseTwoState:
        return {"status": "completed"}

    @staticmethod
    def _grounding_failed(state: PhaseTwoState) -> PhaseTwoState:
        return {
            "status": "grounding_failed",
            "warnings": [
                *state["warnings"],
                "Grounding did not pass after one revision; generated items were omitted.",
            ],
        }

    @staticmethod
    def _failure(
        state: PhaseTwoState,
        stage: Literal["plan", "organize", "date_tool", "grounding", "configuration"],
        message: str,
        calls: int,
    ) -> PhaseTwoState:
        return {
            "status": "failed",
            "model_calls": calls,
            "errors": [*state["errors"], RunError(stage=stage, message=message)],
        }

    @staticmethod
    def _validate_draft_actions(draft: OrganizerDraft, plan: CapturePlan) -> None:
        actions = {step.action for step in plan.steps}
        if draft.summary_claims and "summarize" not in actions:
            raise ValueError("Organizer returned summaries not requested by the plan")
        if draft.tasks and "extract_task" not in actions:
            raise ValueError("Organizer returned tasks not requested by the plan")

    @staticmethod
    def _deterministic_grounding(source_text: str, draft: OrganizerDraft) -> list[GroundingIssue]:
        source = _normalize(source_text)
        feedback: list[GroundingIssue] = []
        for index, claim in enumerate(draft.summary_claims):
            if _normalize(claim.source_quote) not in source:
                feedback.append(
                    GroundingIssue(
                        item_kind="summary",
                        item_index=index,
                        reason="The cited quote does not occur in the source text.",
                    )
                )
        for index, task in enumerate(draft.tasks):
            if _normalize(task.source_quote) not in source:
                feedback.append(
                    GroundingIssue(
                        item_kind="task",
                        item_index=index,
                        reason="The cited quote does not occur in the source text.",
                    )
                )
        return feedback


def configured_reference_date(timezone: str) -> date:
    try:
        return datetime.now(ZoneInfo(timezone)).date()
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown timezone: {timezone}") from error


def _normalize(value: str) -> str:
    return " ".join(value.split())
