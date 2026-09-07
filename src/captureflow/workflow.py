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
    pruned_after_revision: bool
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
    if len({step.id for step in plan.steps}) != len(plan.steps):
        raise ValueError("plan step IDs must be unique")
    for step in plan.steps:
        if _normalize(step.source_span) not in source:
            raise ValueError(f"plan step {step.id} cites a source span not present in the capture")


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
                "pruned_after_revision": False,
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
        graph.add_node("prune_rejected", self._prune_rejected)
        graph.add_node("complete", self._complete)
        graph.add_node("grounding_failed", self._grounding_failed)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            self._after_plan,
            {"organize": "organize", "complete": "complete", "end": END},
        )
        graph.add_conditional_edges(
            "organize",
            self._after_organize,
            {"resolve_dates": "resolve_dates", "end": END},
        )
        graph.add_edge("resolve_dates", "ground")
        graph.add_conditional_edges(
            "ground",
            self._after_grounding,
            {
                "complete": "complete",
                "revise": "prepare_revision",
                "prune": "prune_rejected",
                "grounding_failed": "grounding_failed",
                "failed": END,
            },
        )
        graph.add_edge("prepare_revision", "organize")
        graph.add_conditional_edges(
            "prune_rejected",
            self._after_prune,
            {"ground": "ground", "grounding_failed": "grounding_failed"},
        )
        graph.add_edge("complete", END)
        graph.add_edge("grounding_failed", END)
        return graph.compile()

    def _plan(self, state: PhaseTwoState) -> PhaseTwoState:
        if _requires_web_search(state["source_text"]):
            return {
                "plan": CapturePlan(
                    user_clarification_required=True,
                    clarification_question=(
                        "This local version cannot perform web searches. Would you like to "
                        "save this as a manual research task instead?"
                    ),
                )
            }
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
            draft, warnings = self._validate_draft_actions(draft, state["plan"])
            return {
                "draft": draft,
                "model_calls": calls,
                "warnings": [*state["warnings"], *warnings],
            }
        except (ModelError, ValueError) as error:
            return self._failure(state, "organize", str(error), calls)

    @staticmethod
    def _after_organize(state: PhaseTwoState) -> str:
        return "end" if state["status"] == "failed" else "resolve_dates"

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
            if _is_non_calendar_urgency(task.due_expression):
                resolved_tasks.append(
                    task.model_copy(
                        update={
                            "due_expression": None,
                            "resolved_due_date": None,
                            "date_status": "not_requested",
                        }
                    )
                )
                warnings.append(
                    f"Urgency phrase '{task.due_expression}' was not treated as a calendar date."
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
            report = self.model.critique(
                state["source_text"], self._source_grounding_view(state["draft"])
            )
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
        if not state["pruned_after_revision"]:
            return "prune"
        return "grounding_failed"

    def _prepare_revision(self, state: PhaseTwoState) -> PhaseTwoState:
        return {
            "retries": state["retries"] + 1,
            "revision_feedback": state["grounding_report"],
        }

    def _prune_rejected(self, state: PhaseTwoState) -> PhaseTwoState:
        report = state["grounding_report"]
        rejected = {
            kind: {issue.item_index for issue in report.feedback if issue.item_kind == kind}
            for kind in ("summary", "task", "review")
        }
        draft = state["draft"]
        pruned = draft.model_copy(
            update={
                "summary_claims": [
                    item for index, item in enumerate(draft.summary_claims) if index not in rejected["summary"]
                ],
                "tasks": [item for index, item in enumerate(draft.tasks) if index not in rejected["task"]],
                "review_notes": [
                    item for index, item in enumerate(draft.review_notes) if index not in rejected["review"]
                ],
            }
        )
        omitted = [
            f"{kind} {index + 1}"
            for kind, indexes in rejected.items()
            for index in sorted(index for index in indexes if index is not None)
        ]
        warnings = list(state["warnings"])
        if omitted:
            warnings.append(
                "Critic rejected " + ", ".join(omitted) + "; those items were omitted."
            )
        if not (pruned.summary_claims or pruned.tasks or pruned.review_notes):
            return {
                "draft": pruned,
                "pruned_after_revision": True,
                "status": "grounding_failed",
                "warnings": warnings,
            }
        return {"draft": pruned, "pruned_after_revision": True, "warnings": warnings}

    @staticmethod
    def _after_prune(state: PhaseTwoState) -> str:
        return "grounding_failed" if state["status"] == "grounding_failed" else "ground"

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
    def _validate_draft_actions(
        draft: OrganizerDraft, plan: CapturePlan
    ) -> tuple[OrganizerDraft, list[str]]:
        actions = {step.action for step in plan.steps}
        if draft.tasks and "extract_task" not in actions:
            raise ValueError("Organizer returned tasks not requested by the plan")
        warnings: list[str] = []
        if draft.summary_claims and "summarize" not in actions:
            draft = draft.model_copy(update={"summary_claims": []})
            warnings.append("Organizer returned summaries not requested by the plan; they were omitted.")
        elif len(draft.summary_claims) > 3:
            draft = draft.model_copy(update={"summary_claims": draft.summary_claims[:3]})
            warnings.append("Organizer returned more than three summaries; extra items were omitted.")
        return draft, warnings

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
        for index, note in enumerate(draft.review_notes):
            if _normalize(note.source_quote) not in source:
                feedback.append(
                    GroundingIssue(
                        item_kind="review",
                        item_index=index,
                        reason="The review note's quoted source does not occur in the capture.",
                    )
                )
        return feedback

    @staticmethod
    def _source_grounding_view(draft: OrganizerDraft) -> OrganizerDraft:
        """Hide deterministic date-tool outputs from the source-only Critic."""

        return draft.model_copy(
            update={
                "tasks": [
                    task.model_copy(
                        update={"resolved_due_date": None, "date_status": "not_requested"}
                    )
                    for task in draft.tasks
                ]
            }
        )


def configured_reference_date(timezone: str) -> date:
    try:
        return datetime.now(ZoneInfo(timezone)).date()
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown timezone: {timezone}") from error


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _requires_web_search(source_text: str) -> bool:
    normalized = source_text.lower()
    return any(
        phrase in normalized
        for phrase in (
            "帮我搜",
            "帮我搜索",
            "附近有什么",
            "search nearby",
            "find nearby",
            "nearest ",
        )
    )


def _is_non_calendar_urgency(expression: str) -> bool:
    return _normalize(expression).casefold().strip(".!") in {
        "as soon as possible",
        "asap",
        "at your earliest convenience",
        "when you can",
        "尽快",
        "有空时",
        "方便时",
    }
