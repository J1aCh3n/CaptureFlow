from datetime import date, datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_WORKSPACE_ID = "default"
MAX_MESSAGE_CHARACTERS = 8_000


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(BaseModel):
    """A locally persisted chat conversation."""

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


class Message(BaseModel):
    """One immutable user or assistant turn in a conversation."""

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

    @property
    def start_date(self) -> date:
        return self.start_at.date() if isinstance(self.start_at, datetime) else self.start_at


class SavedItem(BaseModel):
    """An explicit user-authored idea or note."""

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
    """A restart-safe intent for one local Calendar event or idea/note write."""

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
    """The one question a conversation is waiting on, with the action it would complete."""

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


class ProjectChange(BaseModel):
    """One untrusted tracker change proposal; Python decides whether it may run."""

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

    @field_validator("recurring", mode="before")
    @classmethod
    def unstated_recurrence_is_not_recurring(cls, value: object) -> object:
        # The prompt asks for null on anything the message does not state; for this
        # flag, unstated means not repeating.
        return False if value is None else value

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


class ProjectChangeOperation(BaseModel):
    """Restart-safe audit record for one validated tracker change."""

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

    @field_validator("recurring", mode="before")
    @classmethod
    def unstated_recurrence_is_not_recurring(cls, value: object) -> object:
        # The prompt asks for null on anything the message does not state; for this
        # flag, unstated means not repeating.
        return False if value is None else value

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
        # An end time and a duration may both arrive; the router accepts them when they
        # agree and asks about the length when they do not.
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
            # An update may lack every replacement field: the router asks for it rather
            # than rejecting a model that could not tell what the new value is.
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
    """The one decision a model makes per chat turn; it does not write storage itself."""

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

    @model_validator(mode="before")
    @classmethod
    def ignore_sources_outside_answers(cls, value: object) -> object:
        # source_ids only feed an answer's source cards. A write or a question that cites
        # the event or record it concerns is not wrong, so the list is dropped, not rejected.
        if isinstance(value, dict) and value.get("intent") not in {"local_answer", "project_answer", "reply_only"}:
            return {key: item for key, item in value.items() if key != "source_ids"}
        return value

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
            if extra := self._present("project_changes", "clarification_question", "pending_action"):
                raise ValueError(f"a local write cannot include tracker or clarification payload: {extra}")
        elif self.intent == "tracker_change":
            if not self.project_changes:
                raise ValueError("a tracker change requires project changes")
            if extra := self._present("action", "clarification_question", "pending_action"):
                raise ValueError(f"a tracker change requires project changes only: {extra}")
        elif self.intent == "clarify":
            if not self.clarification_question:
                raise ValueError("a clarification requires a clarification question")
            if extra := self._present("action", "project_changes"):
                raise ValueError(f"a clarification cannot include a committed action or tracker change: {extra}")
            if self.pending_action is None or self.pending_action.action not in {
                "create_event",
                "update_event",
                "delete_event",
            }:
                raise ValueError("a clarification requires one pending calendar action")
        else:
            if extra := self._present("action", "project_changes", "clarification_question", "pending_action"):
                raise ValueError(f"a read-only decision cannot include a write or clarification payload: {extra}")
            if self.intent == "reply_only" and self.source_ids:
                raise ValueError("a reply-only decision cannot cite local sources")
            if not self.reply:
                raise ValueError("a read-only decision requires a reply")
        return self

    def _present(self, *fields: str) -> str:
        """The named fields that carry a value, by name only, for a precise error."""

        return ", ".join(field for field in fields if getattr(self, field))

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
