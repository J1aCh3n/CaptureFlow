from datetime import date, datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_WORKSPACE_ID = "default"
MAX_CAPTURE_CHARACTERS = 20_000
MAX_MESSAGE_CHARACTERS = 8_000


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(BaseModel):
    """A locally persisted conversation container; chat is not implemented yet."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = DEFAULT_WORKSPACE_ID
    title: str = Field(min_length=1, max_length=120)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("title")
    @classmethod
    def title_is_not_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value


class Capture(BaseModel):
    """Source text entered manually in Phase 1; it has no generated result."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    source_text: str = Field(min_length=1, max_length=MAX_CAPTURE_CHARACTERS)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("source_text")
    @classmethod
    def source_text_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_text must not be blank")
        return value


class PlanStep(BaseModel):
    """Phase 2 exposes only the actions implemented by its Organizer flow."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    action: Literal["summarize", "extract_task", "inspect_memory"]
    source_span: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class CapturePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[PlanStep] = Field(default_factory=list)
    user_clarification_required: bool = False
    clarification_question: str | None = None

    @model_validator(mode="after")
    def clarification_has_question(self) -> "CapturePlan":
        if self.user_clarification_required and not self.clarification_question:
            raise ValueError("clarification_question is required when clarification is needed")
        if not self.user_clarification_required and not self.steps:
            raise ValueError("a plan needs at least one step or a clarification question")
        return self


class SummaryClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)


class TaskCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    due_expression: str | None = None
    resolved_due_date: date | None = None
    date_status: Literal["not_requested", "resolved", "ambiguous", "unsupported", "failed"] = (
        "not_requested"
    )


class ReviewNote(BaseModel):
    """A user confirmation request that preserves, rather than corrects, source text."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)


class OrganizerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_claims: list[SummaryClaim] = Field(default_factory=list)
    tasks: list[TaskCandidate] = Field(default_factory=list)
    review_notes: list[ReviewNote] = Field(default_factory=list)


class GroundingIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_kind: Literal["summary", "task", "review", "plan"]
    item_index: int | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1)


class GroundingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["pass", "revise"]
    feedback: list[GroundingIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def verdict_matches_feedback(self) -> "GroundingReport":
        if self.verdict == "pass" and self.feedback:
            raise ValueError("a passing grounding report cannot contain revision feedback")
        if self.verdict == "revise" and not self.feedback:
            raise ValueError("a revision report needs at least one grounding issue")
        return self


class RunError(BaseModel):
    stage: Literal["plan", "organize", "date_tool", "grounding", "memory", "configuration"]
    message: str


class CaptureRun(BaseModel):
    """Persisted, user-visible Phase 2 trace; private reasoning is excluded."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    capture_id: str
    status: Literal[
        "completed",
        "completed_with_omissions",
        "needs_review",
        "grounding_failed",
        "failed",
    ]
    plan: CapturePlan | None = None
    output: OrganizerDraft | None = None
    grounding_report: GroundingReport | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[RunError] = Field(default_factory=list)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    memory_decisions: list[dict] = Field(default_factory=list)
    model_calls: int = Field(ge=0, le=6)
    created_at: datetime = Field(default_factory=utc_now)


MemoryType = Literal["preference", "decision", "idea", "fact"]
MemoryStatus = Literal["active", "superseded", "deleted"]
MemoryWriteMode = Literal["auto", "reviewed"]
MemoryDecisionKind = Literal["auto", "review", "blocked", "duplicate"]


class MemoryProposal(BaseModel):
    """Untrusted Memory Agent suggestion. It cannot write storage directly."""

    model_config = ConfigDict(extra="forbid")

    memory_type: MemoryType
    content: str = Field(min_length=1, max_length=500)
    source_quote: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    is_explicit: bool
    related_memory_ids: list[str] = Field(default_factory=list)


class MemoryAgentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[MemoryProposal] = Field(default_factory=list, max_length=5)


class MemoryItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = DEFAULT_WORKSPACE_ID
    memory_type: MemoryType
    content: str = Field(min_length=1, max_length=500)
    source_capture_id: str
    source_quote: str = Field(min_length=1)
    write_mode: MemoryWriteMode
    status: MemoryStatus = "active"
    confidence: float = Field(ge=0, le=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal: MemoryProposal
    decision: MemoryDecisionKind
    reason: str
    conflicting_memory_ids: list[str] = Field(default_factory=list)


class PendingMemoryReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    capture_id: str
    proposal: MemoryProposal
    decision_reason: str
    conflicting_memory_ids: list[str] = Field(default_factory=list)
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class MemoryEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    memory_id: str | None = None
    action: str
    details: dict = Field(default_factory=dict)
    created_at: datetime


class TaskItem(BaseModel):
    """A committed task candidate with a stable ID for checkpoint-safe retries."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    capture_id: str
    run_id: str
    text: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    due_expression: str | None = None
    resolved_due_date: date | None = None
    status: Literal["open", "completed", "deleted"] = "open"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class WorkflowCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    capture_id: str
    phase: Literal["phase_3"] = "phase_3"
    state: Literal["prepared", "committed"]
    payload: dict
    updated_at: datetime = Field(default_factory=utc_now)


class Message(BaseModel):
    """One immutable user or assistant turn in a V2 conversation."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARACTERS)
    reply_to_message_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("content")
    @classmethod
    def content_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class CalendarEvent(BaseModel):
    """A local SQLite-backed calendar event; no external calendar is implied."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = DEFAULT_WORKSPACE_ID
    conversation_id: str
    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4_000)
    source_message_id: str
    source_quote: str = Field(min_length=1)
    start_at: date | datetime
    end_at: datetime | None = None
    all_day: bool
    timezone: str = Field(min_length=1, max_length=100)
    status: Literal["active", "deleted"] = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("title", "timezone")
    @classmethod
    def text_fields_are_not_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("description")
    @classmethod
    def description_is_not_whitespace_when_present(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def date_shape_matches_event_kind(self) -> "CalendarEvent":
        starts_with_time = isinstance(self.start_at, datetime)
        if self.all_day and starts_with_time:
            raise ValueError("an all-day event must start with a date, not a datetime")
        if not self.all_day and not starts_with_time:
            raise ValueError("a timed event must start with a datetime")
        if self.end_at is not None:
            if self.all_day:
                raise ValueError("an all-day event cannot have a timed end")
            if self.end_at <= self.start_at:
                raise ValueError("event end must be after its start")
        return self


class SavedItem(BaseModel):
    """An explicit user-authored idea or note, separate from agent memory."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = DEFAULT_WORKSPACE_ID
    conversation_id: str
    kind: Literal["idea", "note"]
    content: str = Field(min_length=1, max_length=4_000)
    source_message_id: str
    source_quote: str = Field(min_length=1)
    status: Literal["active", "deleted"] = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("content")
    @classmethod
    def saved_content_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("saved item content must not be blank")
        return value


class ToolOperation(BaseModel):
    """A restart-safe intent for one deterministic V2 local write."""

    model_config = ConfigDict(frozen=True)

    id: str
    source_message_id: str
    action_index: int = Field(ge=0, le=2)
    tool_name: Literal["create_local_event", "save_idea", "save_note"]
    request: dict
    result: dict | None = None
    status: Literal["prepared", "committed", "failed", "undone"]
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def result_and_error_match_status(self) -> "ToolOperation":
        if self.status == "failed" and not self.error:
            raise ValueError("a failed operation requires an error")
        if self.status != "failed" and self.error is not None:
            raise ValueError("only a failed operation may contain an error")
        if self.status in {"committed", "undone"} and self.result is None:
            raise ValueError("a committed or undone operation requires a result")
        return self


class PendingClarification(BaseModel):
    """One persisted bounded Calendar clarification for a V2 conversation."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    source_message_id: str
    assistant_message_id: str
    original_action: dict
    missing_field: Literal[
        "date", "time", "duration", "timezone", "recurrence", "target", "follow_up"
    ] = "date"
    status: Literal["pending", "resolved", "cancelled"] = "pending"
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None

    @model_validator(mode="after")
    def resolution_time_matches_status(self) -> "PendingClarification":
        if self.status == "pending" and self.resolved_at is not None:
            raise ValueError("a pending clarification cannot have a resolution time")
        if self.status != "pending" and self.resolved_at is None:
            raise ValueError("a resolved or cancelled clarification needs a resolution time")
        return self


ProjectRecordKind = Literal[
    "task", "milestone", "decision", "constraint", "open_question", "blocker"
]
ProjectRecordState = Literal["active", "completed", "resolved", "superseded", "deleted"]


class Project(BaseModel):
    """One local project whose records may span several conversations."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = DEFAULT_WORKSPACE_ID
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1_000)
    status: Literal["active", "archived"] = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def project_name_is_not_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("project name must not be blank")
        return value

    @field_validator("description")
    @classmethod
    def description_is_not_whitespace_when_present(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ProjectRecord(BaseModel):
    """A source-grounded, user-visible item in one local project tracker."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str
    conversation_id: str
    kind: ProjectRecordKind
    content: str = Field(min_length=1, max_length=4_000)
    rationale: str | None = Field(default=None, max_length=1_000)
    due_at: date | datetime | None = None
    state: ProjectRecordState
    source_message_id: str
    source_quote: str = Field(min_length=1, max_length=8_000)
    supersedes_record_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("content", "source_quote")
    @classmethod
    def record_text_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("project record text must not be blank")
        return value

    @field_validator("rationale")
    @classmethod
    def rationale_is_not_whitespace_when_present(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def state_matches_record_kind(self) -> "ProjectRecord":
        allowed_states = {
            "task": {"active", "completed", "deleted"},
            "milestone": {"completed", "deleted"},
            "decision": {"active", "superseded", "deleted"},
            "constraint": {"active", "superseded", "deleted"},
            "open_question": {"active", "resolved", "deleted"},
            "blocker": {"active", "resolved", "deleted"},
        }
        if self.state not in allowed_states[self.kind]:
            raise ValueError(f"{self.kind} cannot use state {self.state}")
        if self.supersedes_record_id and self.kind not in {"decision", "constraint"}:
            raise ValueError("only a decision or constraint may supersede a record")
        if self.due_at is not None and self.kind != "task":
            raise ValueError("only a task may carry a deadline")
        return self


class ProjectRecordEvent(BaseModel):
    """Append-only audit event for a project-record state or content change."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    project_record_id: str
    action: Literal[
        "created", "completed", "resolved", "superseded", "edited", "deleted", "restored"
    ]
    source_message_id: str | None = None
    previous_state: ProjectRecordState | None = None
    new_state: ProjectRecordState
    details: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def event_matches_state_transition(self) -> "ProjectRecordEvent":
        expected_state = {
            "completed": "completed",
            "resolved": "resolved",
            "superseded": "superseded",
            "deleted": "deleted",
        }.get(self.action)
        if expected_state and self.new_state != expected_state:
            raise ValueError(f"{self.action} event requires state {expected_state}")
        if self.action == "created" and self.previous_state is not None:
            raise ValueError("a created event cannot have a previous state")
        if self.action in {"completed", "resolved", "superseded", "deleted", "restored"}:
            if self.previous_state is None:
                raise ValueError(f"a {self.action} event requires a previous state")
        return self


class ProjectConversationDecision(BaseModel):
    """A Phase 5 reply classification; it cannot propose a tracker write."""

    model_config = ConfigDict(extra="forbid")

    reply: str = Field(min_length=1, max_length=1_000)
    turn_kind: Literal[
        "new_subject",
        "follow_up",
        "correction",
        "cancellation",
        "clarification_answer",
        "clarify",
    ]
    clarification_question: str | None = Field(default=None, max_length=300)

    @field_validator("reply")
    @classmethod
    def reply_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("project conversation reply must not be blank")
        return value

    @model_validator(mode="after")
    def clarification_matches_turn_kind(self) -> "ProjectConversationDecision":
        if self.turn_kind == "clarify" and not self.clarification_question:
            raise ValueError("a clarify turn requires a clarification question")
        if self.turn_kind != "clarify" and self.clarification_question is not None:
            raise ValueError("only a clarify turn may include a clarification question")
        return self


class ProjectChange(BaseModel):
    """One untrusted Phase 6 proposal; storage decides whether it may run."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["create_record", "transition_record", "supersede_record", "set_deadline"]
    kind: ProjectRecordKind | None = None
    content: str | None = Field(default=None, max_length=4_000)
    rationale: str | None = Field(default=None, max_length=1_000)
    due_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    due_time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    recurring: bool = False
    due_at: date | datetime | None = None
    source_quote: str = Field(min_length=1, max_length=8_000)
    target_record_id: str | None = None
    target_state: Literal["completed", "resolved"] | None = None

    @model_validator(mode="before")
    @classmethod
    def drop_retired_deadline_expressions(cls, value: object) -> object:
        # Operations stored before Phase 11 still carry the copied expressions, and they
        # are re-validated on commit and undo. Their resolved value already lives in due_at.
        if not isinstance(value, dict) or not (
            "due_date_expression" in value or "due_time_expression" in value
        ):
            return value
        current = {
            key: item
            for key, item in value.items()
            if key not in {"due_date_expression", "due_time_expression"}
        }
        stored_due_at = current.get("due_at")
        if isinstance(stored_due_at, str) and "due_date" not in current:
            current["due_date"] = stored_due_at[:10]
            if "T" in stored_due_at:
                current["due_time"] = stored_due_at[11:16]
        return current

    @field_validator("content", "rationale")
    @classmethod
    def optional_text_is_not_whitespace(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("due_date")
    @classmethod
    def due_date_is_a_real_calendar_day(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def fields_match_change_action(self) -> "ProjectChange":
        self._deadline_fields_are_bounded()
        if self.action == "create_record":
            if self.kind is None or self.content is None:
                raise ValueError("a record creation requires kind and content")
            if self.target_record_id is not None or self.target_state is not None:
                raise ValueError("a record creation cannot target an existing record")
        elif self.action == "transition_record":
            if self.target_record_id is None or self.target_state is None:
                raise ValueError("a transition requires a target record and target state")
            if any(value is not None for value in (self.kind, self.content, self.rationale)):
                raise ValueError("a transition cannot create replacement content")
        elif self.action == "set_deadline":
            if self.target_record_id is None or (self.due_date is None and not self.recurring):
                raise ValueError("a deadline change requires a target task and a deadline date")
            if any(
                value is not None
                for value in (self.kind, self.content, self.rationale, self.target_state)
            ):
                raise ValueError("a deadline change cannot alter record content or state")
        else:
            if self.target_record_id is None or self.kind not in {"decision", "constraint"}:
                raise ValueError("a supersession requires a decision or constraint target")
            if self.content is None:
                raise ValueError("a supersession requires replacement content")
            if self.target_state is not None:
                raise ValueError("a supersession cannot supply a transition state")
        return self

    def _deadline_fields_are_bounded(self) -> None:
        """A deadline belongs to a task; Python, not the model, resolves it."""

        if self.action == "set_deadline":
            if self.due_time is not None and self.due_date is None:
                raise ValueError("a deadline time needs a deadline date")
            return
        if self.action != "create_record" or self.kind != "task":
            if any(value is not None for value in (self.due_date, self.due_time, self.due_at)) or self.recurring:
                raise ValueError("only a new task may carry a deadline")
            return
        if self.due_time is not None and self.due_date is None:
            raise ValueError("a deadline time needs a deadline date")
        if self.due_at is not None and self.due_date is None:
            raise ValueError("a resolved deadline needs its normalized date")


class ProjectTrackerDecision(ProjectConversationDecision):
    """A bounded project reply plus zero to three proposed tracker changes."""

    changes: list[ProjectChange] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def clarification_cannot_change_the_tracker(self) -> "ProjectTrackerDecision":
        if self.turn_kind == "clarify" and self.changes:
            raise ValueError("a clarification turn cannot propose tracker changes")
        return self


class ProjectAnswerDecision(BaseModel):
    """A Phase 7 answer grounded in a bounded set of retrieved project records."""

    model_config = ConfigDict(extra="forbid")

    reply: str = Field(min_length=1, max_length=1_000)
    answer_kind: Literal["current_status", "history"]
    supporting_record_ids: list[str] = Field(min_length=1, max_length=6)

    @field_validator("reply")
    @classmethod
    def answer_reply_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("project answer reply must not be blank")
        return value

    @field_validator("supporting_record_ids")
    @classmethod
    def supporting_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("supporting record IDs must be unique")
        return value


class LocalContextSourceRef(BaseModel):
    """One bounded Calendar or SavedItem source selected for a local answer."""

    model_config = ConfigDict(frozen=True)

    source_type: Literal["calendar_event", "saved_item"]
    source_id: str = Field(min_length=1)


class LocalContextSource(BaseModel):
    """A display-safe local source, reconstructed from its persisted reference."""

    model_config = ConfigDict(frozen=True)

    reference: LocalContextSourceRef
    label: str
    content: str
    source_quote: str


class SupportedPoint(BaseModel):
    """One answer statement that must identify its local supporting records."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=600)
    source_ids: list[str] = Field(min_length=1, max_length=6)

    @field_validator("text")
    @classmethod
    def text_is_not_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("supported point text must not be blank")
        return value

    @field_validator("source_ids")
    @classmethod
    def source_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("supported point source IDs must be unique")
        return value


class PreparationSuggestion(BaseModel):
    """A clearly labeled inference, never a persisted task or requirement."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=400)
    based_on_source_ids: list[str] = Field(min_length=1, max_length=6)

    @field_validator("text")
    @classmethod
    def text_is_not_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("preparation suggestion text must not be blank")
        return value

    @field_validator("based_on_source_ids")
    @classmethod
    def source_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("preparation suggestion source IDs must be unique")
        return value


class LocalContextAnswerDecision(BaseModel):
    """A bounded model synthesis over selected Phase 9 local records."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["answer", "insufficient_context", "clarify"]
    selected_event_ids: list[str] = Field(default_factory=list, max_length=3)
    selected_item_ids: list[str] = Field(default_factory=list, max_length=3)
    selected_project_record_ids: list[str] = Field(default_factory=list, max_length=3)
    summary_points: list[SupportedPoint] = Field(default_factory=list, max_length=3)
    preparation_suggestions: list[PreparationSuggestion] = Field(default_factory=list, max_length=3)
    missing_information: list[Literal["meeting", "time", "agenda", "preparation"]] = Field(
        default_factory=list, max_length=4
    )
    clarification_question: str | None = Field(default=None, max_length=300)

    @field_validator(
        "selected_event_ids",
        "selected_item_ids",
        "selected_project_record_ids",
        "missing_information",
    )
    @classmethod
    def list_values_are_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("local-context decision list values must be unique")
        return value

    @field_validator("clarification_question")
    @classmethod
    def clarification_is_not_whitespace(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def fields_match_outcome(self) -> "LocalContextAnswerDecision":
        selected_ids = [
            *self.selected_event_ids,
            *self.selected_item_ids,
            *self.selected_project_record_ids,
        ]
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("a local source ID cannot be selected under multiple source types")
        if self.outcome == "answer":
            if not selected_ids:
                raise ValueError("an answer needs at least one selected local source")
            if not self.summary_points:
                raise ValueError("an answer needs at least one supported point")
            if self.clarification_question is not None:
                raise ValueError("an answer cannot include a clarification question")
        elif self.outcome == "clarify":
            if not self.clarification_question:
                raise ValueError("a clarification needs a question")
            if selected_ids or self.summary_points or self.preparation_suggestions:
                raise ValueError("a clarification cannot select sources or include answer content")
        else:
            if self.clarification_question is not None:
                raise ValueError("only a clarification may include a clarification question")
            if selected_ids or self.summary_points or self.preparation_suggestions:
                raise ValueError("an insufficient-context result cannot cite answer content")
        return self


class ProjectChangeOperation(BaseModel):
    """Restart-safe audit record for one validated Phase 6 tracker proposal."""

    model_config = ConfigDict(frozen=True)

    id: str
    source_message_id: str
    project_id: str
    change_index: int = Field(ge=0, le=2)
    change: dict
    target_record_id: str | None = None
    result: dict | None = None
    status: Literal["prepared", "committed", "review", "failed", "undone"]
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def result_and_error_match_status(self) -> "ProjectChangeOperation":
        if self.status == "failed" and not self.error:
            raise ValueError("a failed project change operation requires an error")
        if self.status != "failed" and self.error is not None:
            raise ValueError("only a failed project change operation may contain an error")
        if self.status in {"committed", "undone"} and self.result is None:
            raise ValueError("a committed or undone project change operation requires a result")
        return self


ConversationIntent = Literal[
    "calendar_create",
    "calendar_update",
    "calendar_delete",
    "save_idea",
    "save_note",
    "tracker_change",
    "local_answer",
    "project_answer",
    "clarify",
    "reply_only",
]


class EvidenceReference(BaseModel):
    """One exact quote from a persisted conversation message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str = Field(min_length=1)
    source_quote: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARACTERS)

    @field_validator("message_id", "source_quote")
    @classmethod
    def evidence_text_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence fields must not be blank")
        return value


class ContextualAction(BaseModel):
    """One untrusted local action with one or more persisted-message citations.

    Dates and times arrive already normalized by the model. Python checks their
    structure and attaches the configured timezone; it does not interpret wording.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["create_event", "update_event", "delete_event", "save_idea", "save_note"]
    content: str | None = Field(default=None, max_length=4_000)
    description: str | None = Field(default=None, max_length=4_000)
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    start_time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end_time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    duration_minutes: int | None = Field(default=None, ge=1, le=720)
    timezone: str | None = Field(default=None, min_length=1, max_length=100)
    recurring: bool = False
    target_event_id: str | None = Field(default=None, min_length=1)
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)

    @field_validator("content", "description", "timezone")
    @classmethod
    def optional_text_is_not_whitespace(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("date")
    @classmethod
    def date_is_a_real_calendar_day(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value

    @field_validator("evidence")
    @classmethod
    def evidence_messages_are_unique(cls, value: list[EvidenceReference]) -> list[EvidenceReference]:
        message_ids = [reference.message_id for reference in value]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("an action cannot cite the same message more than once")
        return value

    @model_validator(mode="after")
    def action_fields_match_kind(self) -> "ContextualAction":
        schedule = (self.date, self.start_time, self.end_time, self.duration_minutes)
        if self.end_time is not None and self.duration_minutes is not None:
            raise ValueError("an event cannot have both an end time and a duration")
        if self.action == "create_event":
            # A creation may lack its date; Python then asks for it instead of guessing.
            if self.content is None:
                raise ValueError("an event creation requires title content")
            if self.target_event_id is not None:
                raise ValueError("an event creation cannot target an existing event")
            # A repeating schedule has a time but no single date; Python refuses it by policy.
            if self.start_time is not None and self.date is None and not self.recurring:
                raise ValueError("an event start time needs a date")
            if (self.end_time is not None or self.duration_minutes is not None) and self.start_time is None:
                raise ValueError("an event end or duration needs a start time")
        elif self.action == "update_event":
            # A committed update needs a replacement field; that is checked on the decision,
            # because a pending update may be waiting for exactly the field it lacks.
            pass
        elif self.action == "delete_event":
            # A committed deletion needs one candidate target; an unresolved one may
            # still be carried as the pending action of a clarification.
            if any(value is not None for value in (self.content, self.description, *schedule, self.timezone)):
                raise ValueError("an event deletion cannot change event fields")
            if self.recurring:
                raise ValueError("an event deletion cannot be recurring")
        else:
            if self.content is None:
                raise ValueError("a saved item requires content")
            if any(
                value is not None
                for value in (self.description, *schedule, self.timezone, self.target_event_id)
            ) or self.recurring:
                raise ValueError("a saved item cannot include calendar fields or an event target")
        return self

    def has_replacement(self) -> bool:
        return any(
            value is not None
            for value in (
                self.content,
                self.description,
                self.date,
                self.start_time,
                self.end_time,
                self.duration_minutes,
            )
        )


class ConversationDecision(BaseModel):
    """The future one-call contextual routing contract; it does not write storage itself."""

    model_config = ConfigDict(extra="forbid")

    intent: ConversationIntent
    action: ContextualAction | None = None
    project_changes: list[ProjectChange] = Field(default_factory=list, max_length=3)
    clarification_question: str | None = Field(default=None, max_length=300)
    pending_action: ContextualAction | None = None
    deferred_action: ContextualAction | None = None
    follow_up_question: str | None = Field(default=None, max_length=300)
    reply: str | None = Field(default=None, max_length=1_000)
    source_ids: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("clarification_question", "follow_up_question", "reply")
    @classmethod
    def optional_reply_text_is_not_whitespace(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("source_ids")
    @classmethod
    def source_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("answer source IDs must be unique")
        return value

    @model_validator(mode="after")
    def intent_matches_payload(self) -> "ConversationDecision":
        self._deferred_second_item_is_bounded()
        action_for_intent = {
            "calendar_create": "create_event",
            "calendar_update": "update_event",
            "calendar_delete": "delete_event",
            "save_idea": "save_idea",
            "save_note": "save_note",
        }
        expected_action = action_for_intent.get(self.intent)
        if expected_action is not None:
            if self.action is None or self.action.action != expected_action:
                raise ValueError(f"{self.intent} requires a {expected_action} action")
            if (
                self.intent in {"calendar_update", "calendar_delete"}
                and self.action.target_event_id is None
            ):
                raise ValueError(f"a {self.intent.replace('_', ' ')} requires a candidate event target")
            if self.intent == "calendar_update" and not self.action.has_replacement():
                raise ValueError("an event update requires at least one replacement field")
            if self.project_changes or self.clarification_question or self.pending_action or self.source_ids:
                raise ValueError("a local write cannot include tracker or clarification payload")
        elif self.intent == "tracker_change":
            if (
                self.action
                or not self.project_changes
                or self.clarification_question
                or self.pending_action
                or self.source_ids
            ):
                raise ValueError("a tracker change requires project changes only")
        elif self.intent == "clarify":
            if self.action or self.project_changes or not self.clarification_question or self.source_ids:
                raise ValueError("a clarification cannot include a committed action or tracker change")
            if self.pending_action is None or self.pending_action.action not in {
                "create_event",
                "update_event",
                "delete_event",
            }:
                raise ValueError("a clarification requires one pending calendar action")
        else:
            if self.action or self.project_changes or self.clarification_question or self.pending_action:
                raise ValueError("a read-only decision cannot include a write or clarification payload")
            if self.intent == "reply_only" and self.source_ids:
                raise ValueError("a reply-only decision cannot cite local sources")
            if not self.reply:
                raise ValueError("a read-only decision requires a reply")
        return self

    def _deferred_second_item_is_bounded(self) -> None:
        """One turn still commits one action; a second item may only be asked about."""

        if (self.deferred_action is None) != (self.follow_up_question is None):
            raise ValueError("a deferred second item needs both an action and a follow-up question")
        if self.deferred_action is None:
            return
        if self.intent not in {
            "calendar_create",
            "calendar_update",
            "calendar_delete",
            "save_idea",
            "save_note",
            "tracker_change",
        }:
            raise ValueError("only a committed local action may defer a second item")
        if self.deferred_action.action not in {"create_event", "save_idea", "save_note"}:
            raise ValueError("a deferred second item may only create a new local record")


class CalendarEventUpdateOperation(BaseModel):
    """An auditable, reversible update to one existing local calendar event."""

    model_config = ConfigDict(frozen=True)

    id: str
    source_message_id: str
    action_index: int = Field(ge=0, le=2)
    event_id: str
    workspace_id: str = DEFAULT_WORKSPACE_ID
    before: CalendarEvent
    after: CalendarEvent
    status: Literal["prepared", "committed", "failed", "undone"]
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def snapshots_describe_one_active_event(self) -> "CalendarEventUpdateOperation":
        immutable_fields = (
            "id",
            "workspace_id",
            "conversation_id",
            "source_message_id",
            "source_quote",
            "created_at",
        )
        editable_fields = ("title", "description", "start_at", "end_at", "all_day", "timezone")
        if self.before.id != self.event_id or self.after.id != self.event_id:
            raise ValueError("update snapshots must match the target event")
        if self.before.workspace_id != self.workspace_id or self.after.workspace_id != self.workspace_id:
            raise ValueError("update snapshots must match the target workspace")
        if self.before.status != "active" or self.after.status not in {"active", "deleted"}:
            raise ValueError("only active calendar events can be updated")
        if self.after.status == "deleted" and any(
            getattr(self.before, field) != getattr(self.after, field) for field in editable_fields
        ):
            raise ValueError("a calendar deletion cannot also change event fields")
        if any(getattr(self.before, field) != getattr(self.after, field) for field in immutable_fields):
            raise ValueError("calendar updates cannot change event identity or original provenance")
        if self.status == "failed" and not self.error:
            raise ValueError("a failed calendar update requires an error")
        if self.status != "failed" and self.error is not None:
            raise ValueError("only a failed calendar update may contain an error")
        return self
