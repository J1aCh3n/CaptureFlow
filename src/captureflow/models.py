from datetime import date, datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_WORKSPACE_ID = "default"
MAX_CAPTURE_CHARACTERS = 20_000


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
    action: Literal["summarize", "extract_task"]
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
    stage: Literal["plan", "organize", "date_tool", "grounding", "configuration"]
    message: str


class CaptureRun(BaseModel):
    """Persisted, user-visible Phase 2 trace; private reasoning is excluded."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    capture_id: str
    status: Literal["completed", "grounding_failed", "failed"]
    plan: CapturePlan | None = None
    output: OrganizerDraft | None = None
    grounding_report: GroundingReport | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[RunError] = Field(default_factory=list)
    model_calls: int = Field(ge=0, le=6)
    created_at: datetime = Field(default_factory=utc_now)
