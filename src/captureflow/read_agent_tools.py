"""Bounded, read-only tools for the Phase A agent loop.

The model sees query parameters, never workspace or project identifiers.  Scope is
bound from the current conversation when this toolset is constructed, and every
returned row is recorded in a server-side provenance ledger.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .credentials import contains_credential_disclosure
from .models import CalendarEvent, Conversation, Project, ProjectRecord, SavedItem
from .storage import CaptureStore


MAX_TOOL_ROWS = 20
# The record states search_project_records may return, and so a citation may still be in.
SEARCHABLE_RECORD_STATES = ("active", "completed", "resolved", "superseded")

ResourceType = Literal["calendar_event", "project_record", "saved_item"]
ScopeKind = Literal["workspace", "project"]
ToolErrorCategory = Literal[
    "retryable", "clarification_required", "policy_denied", "conflict", "stale_target"
]


class _ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _KeywordArguments(_ToolArguments):
    keywords: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Literal words or phrases in the item's text, matched case-insensitively. Keywords are alternatives (OR) and rank the items that contain more of them first. When no item contains any of them, the other filters' items are returned with keyword_matches 0, so judge relevance from the items themselves. Leave empty to list items using the other filters.",
    )
    limit: int = Field(default=10, ge=1, le=MAX_TOOL_ROWS)
    offset: int = Field(
        default=0,
        ge=0,
        description="Skip this many matching items. When truncated is true, repeat the same filters and increase offset by the number of returned items.",
    )

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("keywords cannot contain blank values")
        if any(len(value) > 100 for value in normalized):
            raise ValueError("each keyword must contain at most 100 characters")
        return normalized


class _CalendarArguments(_KeywordArguments):
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "_CalendarArguments":
        if self.start_date is not None and self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        return self


class SearchCalendarArgs(_CalendarArguments):
    """Offered when the conversation has a project, which is also the default scope."""

    scope: Literal["project", "workspace"] = Field(
        default="project",
        description="Where to search. Defaults to the current project; use workspace only "
        "when the question is not about this project.",
    )


class SearchWorkspaceCalendarArgs(_CalendarArguments):
    """Offered when the conversation has no project, so the workspace is the only scope."""


class SearchProjectRecordsArgs(_KeywordArguments):
    kinds: list[
        Literal["task", "milestone", "decision", "constraint", "open_question", "blocker"]
    ] = Field(default_factory=list, description="Record types to include; empty includes all types.")
    # Keep in step with SEARCHABLE_RECORD_STATES.
    states: list[Literal["active", "completed", "resolved", "superseded"]] = Field(
        default_factory=lambda: ["active"]
    )


class SearchSavedItemsArgs(_KeywordArguments):
    kinds: list[Literal["idea", "note"]] = Field(
        default_factory=lambda: ["idea", "note"], description="Item types to include."
    )


class SaveItemArgs(_ToolArguments):
    content: str = Field(
        min_length=1,
        max_length=4_000,
        description="The text to store, in the user's own words. Keep what the message says; "
        "do not add anything it does not.",
    )

    @field_validator("content")
    @classmethod
    def saved_content_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("stored text must not be blank")
        return value


class CreateRecordArgs(_ToolArguments):
    """The rules a tracker record follows, as its parameters rather than as prose."""

    kind: Literal["task", "milestone", "decision", "constraint", "open_question", "blocker"] = Field(
        description="What the message states. A deadline belongs to a task; a decision or "
        "constraint that replaces an earlier one is not created here."
    )
    content: str = Field(min_length=1, max_length=4_000, description="The record's text.")
    rationale: str | None = Field(
        default=None, max_length=1_000, description="Why, when the message gives a reason."
    )
    due_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="A deadline the message states, as YYYY-MM-DD. Resolve wording such as "
        "下周一 from reference_date and the supplied calendar; omit it when the message states none.",
    )
    due_time: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="The deadline's clock time, 24-hour HH:MM, only when the message states one.",
    )

    @field_validator("content")
    @classmethod
    def record_content_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a record must not be blank")
        return value


class TurnWriter(Protocol):
    """The router's own write path, which a write tool calls instead of writing itself."""

    @property
    def wrote(self) -> bool: ...

    def save_item(self, kind: str, content: str) -> SavedItem: ...

    def create_record(self, **fields: Any) -> tuple[ProjectRecord | None, str]: ...


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
    # How many rows the filters selected, which is what offset pages through, and how many of
    # them matched a keyword. Without them an empty page cannot be told apart from an empty scope.
    total: int | None = None
    keyword_matches: int | None = None
    # What a write did beyond returning its row, such as a deadline that was not kept or a
    # change held for review.
    detail: str | None = None
    error: ReadToolError | None = None

    @model_validator(mode="after")
    def result_shape_matches_status(self) -> "ReadToolResult":
        if self.ok == (self.error is not None):
            raise ValueError("a successful result cannot have an error and a failed result must have one")
        if not self.ok and (
            self.items or self.truncated or self.total is not None or self.keyword_matches is not None
        ):
            raise ValueError("a failed result cannot contain rows")
        return self


@dataclass(frozen=True)
class ObservedResource:
    resource_type: ResourceType
    scope_kind: ScopeKind
    scope_id: str
    # The row exactly as the model saw it, so a citation to a row that changed is refused.
    snapshot: str


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
    project: Project | None
    # Without a writer the registry is read-only, which is what a read-only caller gets.
    writer: TurnWriter | None = None
    observed_resources: dict[str, ObservedResource] = field(default_factory=dict)
    _definitions: dict[str, ReadToolDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # The scope comes from the conversation, not from a route: the agent runs before any
        # intent exists. The workspace is the hard bound either way.
        in_project = self.project is not None
        definitions = {
            "search_calendar": ReadToolDefinition(
                name="search_calendar",
                description=(
                    "Search active calendar events in the current project, or in the whole workspace."
                    if in_project
                    else "Search active calendar events in the current workspace."
                )
                + " Filter by start_date and/or end_date, or omit both to search all dates. Keywords filter titles and descriptions.",
                arguments_model=SearchCalendarArgs if in_project else SearchWorkspaceCalendarArgs,
            ),
            "search_saved_items": ReadToolDefinition(
                name="search_saved_items",
                description=(
                    "Search active notes and ideas in the current project."
                    if self.project is not None
                    else "Search active notes and ideas in the current workspace."
                ),
                arguments_model=SearchSavedItemsArgs,
            ),
        }
        if self.writer is not None:
            # One stored change per turn, across every write tool: the turn is what a
            # prepared operation's stable ID is derived from.
            for name, what in (("save_note", "note"), ("save_idea", "idea")):
                definitions[name] = ReadToolDefinition(
                    name=name,
                    description=(
                        f"Store {'a' if what == 'note' else 'an'} {what} the message asks you to keep. "
                        "CaptureFlow writes it, checks it and returns the stored row, which you may "
                        "cite. One stored change per turn: a second write is refused."
                    ),
                    arguments_model=SaveItemArgs,
                )
            if self.project is not None:
                definitions["create_record"] = ReadToolDefinition(
                    name="create_record",
                    description=(
                        "Create a tracker record in the current project for work, a milestone, a "
                        "decision, a constraint, an open question or a blocker the message states. "
                        "CaptureFlow checks it, stores it and returns the row, which you may cite. "
                        "Finishing or replacing a record that already exists is not done here. One "
                        "stored change per turn: a second write is refused."
                    ),
                    arguments_model=CreateRecordArgs,
                )
        if self.project is not None:
            definitions["search_project_records"] = ReadToolDefinition(
                name="search_project_records",
                description="Search tracker records in the current project. Use kinds and states to list categories such as active tasks; keywords filter content and rationale.",
                arguments_model=SearchProjectRecordsArgs,
            )
        self._definitions = definitions

    @classmethod
    def for_conversation(
        cls, store: CaptureStore, conversation_id: str, writer: TurnWriter | None = None
    ) -> "BoundReadTools":
        conversation = store.get_conversation(conversation_id)
        return cls(
            store=store,
            conversation=conversation,
            project=store.get_conversation_project(conversation_id),
            writer=writer,
        )

    @property
    def wrote_this_turn(self) -> bool:
        return self.writer is not None and self.writer.wrote

    def definitions_for_model(self) -> list[dict[str, Any]]:
        return [definition.for_model() for definition in self._definitions.values()]

    def resolve_observed(self, source_ids: list[str]) -> list[CalendarEvent | ProjectRecord | SavedItem]:
        """Read each citation again from SQLite and require it to be the row its tool returned.

        Raises ValueError for a citation outside the ledger, a row that no longer exists, one
        that has since left the state or scope it was returned under, or one whose fields the
        model saw have changed.
        """

        resolved: list[CalendarEvent | ProjectRecord | SavedItem] = []
        for source_id in source_ids:
            observed = self.observed_resources.get(source_id)
            if observed is None:
                raise ValueError("source ID was not returned by a tool in this run")
            try:
                row = self._current_row(source_id, observed)
            except KeyError as error:
                raise ValueError("a cited source no longer exists") from error
            if row is None:
                raise ValueError("a cited source is no longer in the state or scope it was returned under")
            if _snapshot(_ITEM_FOR_MODEL[observed.resource_type](row)) != observed.snapshot:
                # The model chose and described this row from what it saw; a moved date or an
                # edited text could make that choice wrong, so it is not rendered.
                raise ValueError("a cited source changed after a tool returned it")
            resolved.append(row)
        return resolved

    def _current_row(
        self, source_id: str, observed: ObservedResource
    ) -> CalendarEvent | ProjectRecord | SavedItem | None:
        if observed.resource_type == "calendar_event":
            event = self.store.get_calendar_event(source_id)
            usable = event.status == "active" and self._in_scope(event.conversation_id, event.workspace_id, observed)
            return event if usable else None
        if observed.resource_type == "project_record":
            record = self.store.get_project_record(source_id)
            usable = (
                record.state in SEARCHABLE_RECORD_STATES
                and observed.scope_kind == "project"
                and record.project_id == observed.scope_id
            )
            return record if usable else None
        item = self.store.get_saved_item(source_id)
        usable = item.status == "active" and self._in_scope(item.conversation_id, item.workspace_id, observed)
        return item if usable else None

    def _in_scope(self, conversation_id: str, workspace_id: str, observed: ObservedResource) -> bool:
        if observed.scope_kind == "workspace":
            return workspace_id == observed.scope_id
        project = self.store.get_conversation_project(conversation_id)
        return project is not None and project.id == observed.scope_id

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
        except ValidationError as error:
            detail = "; ".join(
                f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
                for item in error.errors(include_input=False, include_url=False)
            )
            return _error("retryable", f"The tool arguments are invalid: {detail}")

        if name == "search_calendar":
            return self._search_calendar(parsed)
        if name == "search_project_records":
            return self._search_project_records(parsed)
        if name in {"save_note", "save_idea"}:
            return self._save_item(name.removeprefix("save_"), parsed)
        if name == "create_record":
            return self._create_record(parsed)
        if name == "search_saved_items":
            return self._search_saved_items(parsed)
        raise AssertionError(f"unhandled read tool: {name}")

    def _search_calendar(
        self, arguments: SearchCalendarArgs | SearchWorkspaceCalendarArgs
    ) -> ReadToolResult:
        project_only = isinstance(arguments, SearchCalendarArgs) and arguments.scope == "project"
        if project_only and self.project is None:
            return _error(
                "clarification_required",
                "The current conversation is not assigned to a project, so project-only calendar search is unavailable.",
            )

        events = self.store.list_calendar_events(self.conversation.workspace_id)
        events = [
            event
            for event in events
            if (arguments.start_date is None or event.start_date >= arguments.start_date)
            and (arguments.end_date is None or event.start_date <= arguments.end_date)
        ]
        if project_only:
            events = [
                event
                for event in events
                if (project := self.store.get_conversation_project(event.conversation_id)) is not None
                and project.id == self.project.id
            ]
        events, keyword_matches = _rank_by_keywords(
            events,
            arguments.keywords,
            lambda event: f"{event.title} {event.description or ''}",
        )
        selected, truncated = _bounded(events, arguments.limit, arguments.offset)
        scope_kind: ScopeKind = "project" if project_only else "workspace"
        scope_id = self.project.id if project_only else self.conversation.workspace_id
        self._observe(selected, "calendar_event", scope_kind, scope_id)
        return ReadToolResult(
            ok=True,
            scope=scope_kind,
            items=[_calendar_item(event) for event in selected],
            truncated=truncated,
            total=len(events),
            keyword_matches=keyword_matches,
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
        records, keyword_matches = _rank_by_keywords(
            records,
            arguments.keywords,
            lambda record: f"{record.content} {record.rationale or ''}",
        )
        selected, truncated = _bounded(records, arguments.limit, arguments.offset)
        self._observe(selected, "project_record", "project", self.project.id)
        return ReadToolResult(
            ok=True,
            scope="project",
            items=[_project_record_item(record) for record in selected],
            truncated=truncated,
            total=len(records),
            keyword_matches=keyword_matches,
        )

    def _save_item(self, kind: str, arguments: _ToolArguments) -> ReadToolResult:
        if not isinstance(arguments, SaveItemArgs):
            raise TypeError("a save tool received unexpected arguments")
        already = self._already_wrote()
        if already is not None:
            return already
        assert self.writer is not None
        item = self.writer.save_item(kind, arguments.content)
        scope_kind, scope_id = self._write_scope()
        self._observe([item], "saved_item", scope_kind, scope_id)
        return ReadToolResult(ok=True, scope=scope_kind, items=[_saved_item(item)], total=1)

    def _create_record(self, arguments: _ToolArguments) -> ReadToolResult:
        if not isinstance(arguments, CreateRecordArgs):
            raise TypeError("create_record received unexpected arguments")
        if self.project is None:
            return _error("policy_denied", "No project is available in the current scope.")
        already = self._already_wrote()
        if already is not None:
            return already
        assert self.writer is not None
        try:
            record, note = self.writer.create_record(**arguments.model_dump())
        except ValueError as error:
            # A rule refusing the change is the model's to read and react to, not a failed turn.
            return _error("policy_denied", f"The change was refused: {error}")
        if record is None:
            # Stored, but awaiting review rather than applied, so there is no row to cite.
            return ReadToolResult(ok=True, scope="project", items=[], total=0, detail=note)
        self._observe([record], "project_record", "project", self.project.id)
        return ReadToolResult(
            ok=True,
            scope="project",
            items=[_project_record_item(record, recorded_at=False)],
            total=1,
            detail=note,
        )

    def _already_wrote(self) -> ReadToolResult | None:
        """One stored change per turn: it is what keeps a prepared operation's stable ID
        unique, and so what makes a crash in the middle of a turn recoverable."""

        if self.writer is not None and self.writer.wrote:
            return _error("conflict", "This turn has already stored a change; nothing was written.")
        return None

    def _write_scope(self) -> tuple[ScopeKind, str]:
        if self.project is not None:
            return "project", self.project.id
        return "workspace", self.conversation.workspace_id

    def _search_saved_items(self, arguments: _ToolArguments) -> ReadToolResult:
        if not isinstance(arguments, SearchSavedItemsArgs):
            raise TypeError("search_saved_items received unexpected arguments")
        kinds = tuple(arguments.kinds)
        # Notes keep the router's candidate rule: the conversation's project when it has
        # one, otherwise the workspace.
        if self.project is not None:
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
        items, keyword_matches = _rank_by_keywords(
            items,
            arguments.keywords,
            lambda item: item.content,
        )
        selected, truncated = _bounded(items, arguments.limit, arguments.offset)
        self._observe(selected, "saved_item", scope_kind, scope_id)
        return ReadToolResult(
            ok=True,
            scope=scope_kind,
            items=[_saved_item(item) for item in selected],
            truncated=truncated,
            total=len(items),
            keyword_matches=keyword_matches,
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
                snapshot=_snapshot(_ITEM_FOR_MODEL[resource_type](row)),
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


def _rank_by_keywords(rows: list, keywords: list[str], text_for_row) -> tuple[list, int | None]:
    """Match literal phrases; rank by matched keywords, keeping storage order for ties.

    Keywords that match nothing fall back to the rows the other filters selected, because
    an empty result would say the scope is empty, and that is a different fact. Whether an
    unmatched row answers the question is the model's judgement, not this function's.
    """

    if not keywords:
        return rows, None
    patterns = [
        re.compile(
            rf"(?<![a-z0-9_]){re.escape(keyword)}(?![a-z0-9_])" if keyword.isascii() else re.escape(keyword),
            re.IGNORECASE,
        )
        for keyword in keywords
    ]
    ranked = []
    for position, row in enumerate(rows):
        text = text_for_row(row)
        score = sum(pattern.search(text) is not None for pattern in patterns)
        if score:
            ranked.append((-score, position, row))
    if not ranked:
        return rows, 0
    ranked.sort(key=lambda entry: (entry[0], entry[1]))
    return [row for _, _, row in ranked], len(ranked)


def _bounded(rows: list, limit: int, offset: int) -> tuple[list, bool]:
    end = offset + limit
    return rows[offset:end], len(rows) > end


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


def _project_record_item(record: ProjectRecord, *, recorded_at: bool = True) -> dict[str, Any]:
    """A tracker row as the model sees it.

    A search includes when the row last changed, which is how a decision's history reads.
    A write's own result does not: the model just made this row, so when the database
    recorded it tells it nothing, and a timestamp from inside the turn would make the turn
    impossible to replay offline.
    """

    item = {
        "id": record.id,
        "type": "project_record",
        "kind": record.kind,
        "content": record.content,
        "rationale": record.rationale,
        "due_at": record.due_at.isoformat() if record.due_at is not None else None,
        "state": record.state,
        "supersedes_record_id": record.supersedes_record_id,
    }
    if recorded_at:
        item["updated_at"] = record.updated_at.isoformat()
    return item


def _saved_item(item: SavedItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "type": "saved_item",
        "kind": item.kind,
        "content": item.content,
    }


# How each resource type appears in a tool result; the ledger's snapshot uses the same form.
_ITEM_FOR_MODEL = {
    "calendar_event": _calendar_item,
    "project_record": _project_record_item,
    "saved_item": _saved_item,
}


def _snapshot(item: dict[str, Any]) -> str:
    return json.dumps(item, ensure_ascii=False, sort_keys=True)
