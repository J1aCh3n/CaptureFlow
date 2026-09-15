"""Bounded, read-only tools for the Phase A agent loop.

The model sees query parameters, never workspace or project identifiers.  Scope is
bound from the current conversation when this toolset is constructed, and every
returned row is recorded in a server-side provenance ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .credentials import contains_credential_disclosure
from .local_context_service import match_terms
from .models import CalendarEvent, Conversation, Project, ProjectRecord, SavedItem
from .storage import CaptureStore


MAX_TOOL_ROWS = 20
MAX_CALENDAR_DAYS = 62

ReadAgentRoute = Literal["local_answer", "project_answer"]
ResourceType = Literal["calendar_event", "project_record", "saved_item"]
ScopeKind = Literal["workspace", "project"]
ToolErrorCategory = Literal[
    "retryable", "clarification_required", "policy_denied", "conflict", "stale_target"
]


class _ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _KeywordArguments(_ToolArguments):
    keywords: list[str] = Field(default_factory=list, max_length=12)
    limit: int = Field(default=10, ge=1, le=MAX_TOOL_ROWS)

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("keywords cannot contain blank values")
        if any(len(value) > 100 for value in normalized):
            raise ValueError("each keyword must contain at most 100 characters")
        return normalized


class SearchCalendarArgs(_KeywordArguments):
    start_date: date
    end_date: date
    this_project_only: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> "SearchCalendarArgs":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        inclusive_days = (self.end_date - self.start_date).days + 1
        if inclusive_days > MAX_CALENDAR_DAYS:
            raise ValueError(f"calendar range cannot exceed {MAX_CALENDAR_DAYS} days")
        return self


class SearchProjectCalendarArgs(_KeywordArguments):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_range(self) -> "SearchProjectCalendarArgs":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        inclusive_days = (self.end_date - self.start_date).days + 1
        if inclusive_days > MAX_CALENDAR_DAYS:
            raise ValueError(f"calendar range cannot exceed {MAX_CALENDAR_DAYS} days")
        return self


class SearchProjectRecordsArgs(_KeywordArguments):
    kinds: list[
        Literal["task", "milestone", "decision", "constraint", "open_question", "blocker"]
    ] = Field(default_factory=list)
    states: list[Literal["active", "completed", "resolved", "superseded"]] = Field(
        default_factory=lambda: ["active"]
    )


class SearchSavedItemsArgs(_KeywordArguments):
    kinds: list[Literal["idea", "note"]] = Field(default_factory=lambda: ["idea", "note"])


class ReadToolError(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: ToolErrorCategory
    message: str


class ReadToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    scope: ScopeKind | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    error: ReadToolError | None = None

    @model_validator(mode="after")
    def result_shape_matches_status(self) -> "ReadToolResult":
        if self.ok == (self.error is not None):
            raise ValueError("a successful result cannot have an error and a failed result must have one")
        if not self.ok and (self.items or self.truncated):
            raise ValueError("a failed result cannot contain rows")
        return self


@dataclass(frozen=True)
class ObservedResource:
    resource_type: ResourceType
    scope_kind: ScopeKind
    scope_id: str


@dataclass(frozen=True)
class ReadToolDefinition:
    name: str
    description: str
    arguments_model: type[_ToolArguments]

    def for_model(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_model.model_json_schema(),
            },
        }


@dataclass
class BoundReadTools:
    """Read tool registry with immutable caller scope and mutable provenance."""

    store: CaptureStore
    conversation: Conversation
    route: ReadAgentRoute
    project: Project | None
    observed_resources: dict[str, ObservedResource] = field(default_factory=dict)
    _definitions: dict[str, ReadToolDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.route == "project_answer" and self.project is None:
            raise ValueError("project_answer tools require an assigned project")

        calendar_arguments = (
            SearchProjectCalendarArgs if self.route == "project_answer" else SearchCalendarArgs
        )
        definitions = {
            "search_calendar": ReadToolDefinition(
                name="search_calendar",
                description=(
                    "Search active calendar events in the current project."
                    if self.route == "project_answer"
                    else "Search active calendar events in the current workspace; optionally narrow to the current project."
                ),
                arguments_model=calendar_arguments,
            ),
            "search_saved_items": ReadToolDefinition(
                name="search_saved_items",
                description=(
                    "Search active notes and ideas in the current project."
                    if self.route == "project_answer"
                    else "Search active notes and ideas in the current workspace."
                ),
                arguments_model=SearchSavedItemsArgs,
            ),
        }
        if self.project is not None:
            definitions["search_project_records"] = ReadToolDefinition(
                name="search_project_records",
                description="Search tracker records in the current project.",
                arguments_model=SearchProjectRecordsArgs,
            )
        self._definitions = definitions

    @classmethod
    def for_conversation(
        cls, store: CaptureStore, conversation_id: str, route: ReadAgentRoute
    ) -> "BoundReadTools":
        conversation = store.get_conversation(conversation_id)
        return cls(
            store=store,
            conversation=conversation,
            route=route,
            project=store.get_conversation_project(conversation_id),
        )

    def definitions_for_model(self) -> list[dict[str, Any]]:
        return [definition.for_model() for definition in self._definitions.values()]

    def resolve_observed(self, source_ids: list[str]) -> list[CalendarEvent | ProjectRecord | SavedItem]:
        """Resolve final citations only after the loop has checked the provenance ledger."""

        resolved: list[CalendarEvent | ProjectRecord | SavedItem] = []
        for source_id in source_ids:
            observed = self.observed_resources.get(source_id)
            if observed is None:
                raise ValueError("source ID was not returned by a tool in this run")
            if observed.resource_type == "calendar_event":
                resolved.append(self.store.get_calendar_event(source_id))
            elif observed.resource_type == "project_record":
                resolved.append(self.store.get_project_record(source_id))
            else:
                resolved.append(self.store.get_saved_item(source_id))
        return resolved

    def invoke(self, name: str, arguments: dict[str, Any]) -> ReadToolResult:
        definition = self._definitions.get(name)
        if definition is None:
            return _error("policy_denied", "That tool is not available in the current scope.")
        if _contains_credential(arguments):
            return _error(
                "policy_denied",
                "The request contains credential-like content and cannot be used for a local search.",
            )
        try:
            parsed = definition.arguments_model.model_validate(arguments)
        except ValidationError:
            return _error("retryable", "The tool arguments are invalid; correct them and try again.")

        if name == "search_calendar":
            return self._search_calendar(parsed)
        if name == "search_project_records":
            return self._search_project_records(parsed)
        if name == "search_saved_items":
            return self._search_saved_items(parsed)
        raise AssertionError(f"unhandled read tool: {name}")

    def _search_calendar(
        self, arguments: SearchCalendarArgs | SearchProjectCalendarArgs
    ) -> ReadToolResult:
        project_only = self.route == "project_answer" or (
            isinstance(arguments, SearchCalendarArgs) and arguments.this_project_only
        )
        if project_only and self.project is None:
            return _error(
                "clarification_required",
                "The current conversation is not assigned to a project, so project-only calendar search is unavailable.",
            )

        events = [
            event
            for event in self.store.list_calendar_events(self.conversation.workspace_id)
            if arguments.start_date <= event.start_date <= arguments.end_date
        ]
        if project_only:
            events = [
                event
                for event in events
                if (project := self.store.get_conversation_project(event.conversation_id)) is not None
                and project.id == self.project.id
            ]
        events = _rank_if_requested(
            events,
            arguments.keywords,
            lambda event: f"{event.title} {event.description or ''} {event.source_quote}",
        )
        selected, truncated = _bounded(events, arguments.limit)
        scope_kind: ScopeKind = "project" if project_only else "workspace"
        scope_id = self.project.id if project_only else self.conversation.workspace_id
        self._observe(selected, "calendar_event", scope_kind, scope_id)
        return ReadToolResult(
            ok=True,
            scope=scope_kind,
            items=[_calendar_item(event) for event in selected],
            truncated=truncated,
        )

    def _search_project_records(self, arguments: _ToolArguments) -> ReadToolResult:
        if not isinstance(arguments, SearchProjectRecordsArgs):
            raise TypeError("search_project_records received unexpected arguments")
        if self.project is None:
            return _error("policy_denied", "No project is available in the current scope.")
        states = tuple(arguments.states)
        records = self.store.list_project_records(self.project.id, states=states)
        if arguments.kinds:
            requested_kinds = set(arguments.kinds)
            records = [record for record in records if record.kind in requested_kinds]
        records = _rank_if_requested(
            records,
            arguments.keywords,
            lambda record: f"{record.content} {record.rationale or ''} {record.source_quote}",
        )
        selected, truncated = _bounded(records, arguments.limit)
        self._observe(selected, "project_record", "project", self.project.id)
        return ReadToolResult(
            ok=True,
            scope="project",
            items=[_project_record_item(record) for record in selected],
            truncated=truncated,
        )

    def _search_saved_items(self, arguments: _ToolArguments) -> ReadToolResult:
        if not isinstance(arguments, SearchSavedItemsArgs):
            raise TypeError("search_saved_items received unexpected arguments")
        kinds = tuple(arguments.kinds)
        if self.route == "project_answer":
            assert self.project is not None
            scope_kind: ScopeKind = "project"
            scope_id = self.project.id
            items = self.store.list_saved_items_for_project(
                self.project.id,
                workspace_id=self.conversation.workspace_id,
                kinds=kinds,
            )
        else:
            scope_kind = "workspace"
            scope_id = self.conversation.workspace_id
            items = self.store.list_saved_items(
                workspace_id=self.conversation.workspace_id,
                kinds=kinds,
            )
        items = _rank_if_requested(
            items,
            arguments.keywords,
            lambda item: f"{item.content} {item.source_quote}",
        )
        selected, truncated = _bounded(items, arguments.limit)
        self._observe(selected, "saved_item", scope_kind, scope_id)
        return ReadToolResult(
            ok=True,
            scope=scope_kind,
            items=[_saved_item(item) for item in selected],
            truncated=truncated,
        )

    def _observe(
        self,
        rows: list[CalendarEvent] | list[ProjectRecord] | list[SavedItem],
        resource_type: ResourceType,
        scope_kind: ScopeKind,
        scope_id: str,
    ) -> None:
        for row in rows:
            self.observed_resources[row.id] = ObservedResource(
                resource_type=resource_type,
                scope_kind=scope_kind,
                scope_id=scope_id,
            )


def _error(category: ToolErrorCategory, message: str) -> ReadToolResult:
    return ReadToolResult(ok=False, error=ReadToolError(category=category, message=message))


def _contains_credential(value: Any) -> bool:
    if isinstance(value, str):
        return contains_credential_disclosure(value)
    if isinstance(value, dict):
        return any(_contains_credential(key) or _contains_credential(item) for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_credential(item) for item in value)
    return False


def _rank_if_requested(rows: list, keywords: list[str], text_for_row) -> list:
    if not keywords:
        return rows
    wanted = match_terms(" ".join(keywords))
    ranked = []
    for position, row in enumerate(rows):
        score = len(wanted & match_terms(text_for_row(row)))
        if score:
            ranked.append((-score, position, row))
    return [row for _, _, row in sorted(ranked, key=lambda entry: (entry[0], entry[1]))]


def _bounded(rows: list, limit: int) -> tuple[list, bool]:
    return rows[:limit], len(rows) > limit


def _calendar_item(event: CalendarEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "type": "calendar_event",
        "title": event.title,
        "description": event.description,
        "start_at": event.start_at.isoformat(),
        "end_at": event.end_at.isoformat() if event.end_at is not None else None,
        "all_day": event.all_day,
        "timezone": event.timezone,
    }


def _project_record_item(record: ProjectRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "type": "project_record",
        "kind": record.kind,
        "content": record.content,
        "rationale": record.rationale,
        "due_at": record.due_at.isoformat() if record.due_at is not None else None,
        "state": record.state,
    }


def _saved_item(item: SavedItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "type": "saved_item",
        "kind": item.kind,
        "content": item.content,
    }
