"""What the contextual router sees: the recent conversation and the stored state, in layers.

Each layer has a budget. A single user's data fits inside every budget, so the model
in practice sees everything; the bounds only decide what gives way as data grows.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .credentials import contains_credential_disclosure
from .models import CalendarEvent, Message, ProjectRecord, SavedItem
from .storage import CaptureStore


# The conversation: recent turns in full, older long messages as one-line references.
MAX_CONTEXT_CHARACTERS = 12_000
RECENT_FULL_TEXT_MESSAGES = 4
LONG_MESSAGE_CHARACTERS = 800
REFERENCE_PREVIEW_CHARACTERS = 80

# The stored state: one compact line per item.
EVENT_DAYS_BEFORE = 7
EVENT_DAYS_AFTER = 27
MAX_CANDIDATE_EVENTS = 15
MAX_NAMED_EVENTS = 5
MAX_CANDIDATE_RECORDS = 30
MAX_CANDIDATE_SAVED_ITEMS = 20
MAX_LINE_TEXT_CHARACTERS = 200


@dataclass(frozen=True)
class ContextWindow:
    messages: tuple[Message, ...]
    omitted_message_count: int

    @property
    def truncated(self) -> bool:
        return self.omitted_message_count > 0

    def for_router(self) -> list[dict]:
        """Recent messages in full; an older long message as a reference to its ID."""

        recent = set(message.id for message in self.messages[-RECENT_FULL_TEXT_MESSAGES:])
        return [
            _message_for_router(message, full_text=message.id in recent)
            for message in self.messages
        ]

    def user_message(self, message_id: str) -> Message | None:
        return next(
            (message for message in self.messages if message.id == message_id and message.role == "user"),
            None,
        )


@dataclass(frozen=True)
class RouterCandidates:
    """The workspace's stored state a contextual router may reference."""

    events: tuple[CalendarEvent, ...]
    saved_items: tuple[SavedItem, ...]
    project_records: tuple[ProjectRecord, ...]
    # The kinds whose list was cut at its bound, so a missing item may exist.
    truncated_kinds: tuple[str, ...] = ()

    def for_router(self) -> dict[str, list[str]]:
        return {
            "candidate_events": [_event_line(event) for event in self.events],
            "candidate_saved_items": [_saved_item_line(item) for item in self.saved_items],
            "candidate_project_records": [_record_line(record) for record in self.project_records],
        }


def select_context_window(
    messages: list[Message], *, max_characters: int = MAX_CONTEXT_CHARACTERS
) -> ContextWindow:
    """The recent conversation, newest first, until the character budget is spent.

    The last two turns always keep their full text. An older long message costs only
    its one-line reference. A message the credential guard blocked is never included.
    """

    if max_characters < 1:
        raise ValueError("the context budget must be positive")
    if not messages:
        raise ValueError("a context window needs at least one message")

    ordered = [
        message
        for message in _logical_turn_order(messages)
        if not (message.role == "user" and contains_credential_disclosure(message.content))
    ]
    selected_reversed: list[Message] = []
    character_count = 0
    for index in range(len(ordered) - 1, -1, -1):
        message = ordered[index]
        recent = len(selected_reversed) < RECENT_FULL_TEXT_MESSAGES
        cost = len(message.content) if recent else _older_message_cost(message)
        if not recent and character_count + cost > max_characters:
            return ContextWindow(tuple(reversed(selected_reversed)), omitted_message_count=index + 1)
        selected_reversed.append(message)
        character_count += cost
    return ContextWindow(tuple(reversed(selected_reversed)), omitted_message_count=0)


def select_router_candidates(
    store: CaptureStore,
    conversation_id: str,
    reference_date: date,
    message: str | None = None,
) -> RouterCandidates:
    """Every stored item a turn may be about, within the bounds, without widening scope.

    Events come from the whole workspace, because the calendar is shared: an event made
    in another conversation must still be found. Records and ideas stay in the project.
    """

    conversation = store.get_conversation(conversation_id)
    all_events = store.list_calendar_events(conversation.workspace_id)
    events, events_truncated = _events_in_window(all_events, reference_date)
    events += _named_events(all_events, events, message)

    project = store.get_conversation_project(conversation_id)
    if project is None:
        saved_items = store.list_saved_items(conversation.workspace_id)
        records: list[ProjectRecord] = []
    else:
        saved_items = store.list_saved_items_for_project(project.id, conversation.workspace_id)
        records = store.list_project_records(project.id, states=("active",))

    truncated_kinds = [
        kind
        for kind, truncated in (
            ("events", events_truncated),
            ("saved_items", len(saved_items) > MAX_CANDIDATE_SAVED_ITEMS),
            ("project_records", len(records) > MAX_CANDIDATE_RECORDS),
        )
        if truncated
    ]
    # Both lists are stored most recently updated first.
    return RouterCandidates(
        events=tuple(sorted(events, key=lambda event: (event.start_date, event.id))),
        saved_items=tuple(saved_items[:MAX_CANDIDATE_SAVED_ITEMS]),
        project_records=tuple(records[:MAX_CANDIDATE_RECORDS]),
        truncated_kinds=tuple(truncated_kinds),
    )


def _events_in_window(
    events: list[CalendarEvent], reference_date: date
) -> tuple[list[CalendarEvent], bool]:
    """Events in the calendar table's window, nearest the reference date first when cut."""

    earliest = reference_date - timedelta(days=EVENT_DAYS_BEFORE)
    latest = reference_date + timedelta(days=EVENT_DAYS_AFTER)
    in_window = [event for event in events if earliest <= event.start_date <= latest]
    nearest = sorted(
        in_window,
        key=lambda event: (
            abs((event.start_date - reference_date).days),
            event.start_date < reference_date,
            event.start_date,
            event.id,
        ),
    )
    return nearest[:MAX_CANDIDATE_EVENTS], len(nearest) > MAX_CANDIDATE_EVENTS


def _named_events(
    events: list[CalendarEvent], chosen: list[CalendarEvent], message: str | None
) -> list[CalendarEvent]:
    """Events outside the bound whose titles the message names.

    Without these, a user who answers "which meeting?" with its name still could not
    reach an event beyond the bound. This only retrieves; the model decides the target.
    """

    if not message:
        return []
    terms = _title_terms(message)
    chosen_ids = {event.id for event in chosen}
    scored = [
        (len(terms & _title_terms(event.title)), event)
        for event in events
        if event.id not in chosen_ids
    ]
    matches = [(score, event) for score, event in scored if score >= 2]
    matches.sort(key=lambda item: (-item[0], item[1].start_date, item[1].id))
    return [event for _, event in matches[:MAX_NAMED_EVENTS]]


def _title_terms(value: str) -> set[str]:
    """Latin words and Chinese character pairs, enough to tell one title from another."""

    words = set(re.findall(r"[a-z0-9]{2,}", value.casefold()))
    pairs = {
        run[index : index + 2]
        for run in re.findall(r"[一-鿿]{2,}", value)
        for index in range(len(run) - 1)
    }
    return words | pairs


def _older_message_cost(message: Message) -> int:
    if len(message.content) <= LONG_MESSAGE_CHARACTERS:
        return len(message.content)
    return REFERENCE_PREVIEW_CHARACTERS + 40


def _message_for_router(message: Message, *, full_text: bool) -> dict:
    if full_text or len(message.content) <= LONG_MESSAGE_CHARACTERS:
        return {"message_id": message.id, "role": message.role, "content": message.content}
    return {
        "message_id": message.id,
        "role": message.role,
        "reference": (
            f"long message, {len(message.content)} characters, starting: "
            f"{message.content[:REFERENCE_PREVIEW_CHARACTERS]}…"
        ),
    }


def _event_line(event: CalendarEvent) -> str:
    return f"{event.id} | {_short(event.title)} | {_event_time(event)}"


def _event_time(event: CalendarEvent) -> str:
    if not isinstance(event.start_at, datetime):
        return f"{event.start_at.isoformat()} all day"
    timing = event.start_at.strftime("%Y-%m-%d %H:%M")
    if event.end_at is not None:
        end_format = "%H:%M" if event.end_at.date() == event.start_at.date() else "%Y-%m-%d %H:%M"
        timing += f"–{event.end_at.strftime(end_format)}"
    return timing


def _record_line(record: ProjectRecord) -> str:
    parts = [record.id, record.kind, _short(record.content), record.state]
    if record.due_at is not None:
        due = record.due_at
        parts.append(f"due {due.strftime('%Y-%m-%d %H:%M') if isinstance(due, datetime) else due.isoformat()}")
    return " | ".join(parts)


def _saved_item_line(item: SavedItem) -> str:
    return f"{item.id} | {item.kind} | {_short(item.content)}"


def _short(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_LINE_TEXT_CHARACTERS:
        return text
    return f"{text[:MAX_LINE_TEXT_CHARACTERS]}…"


def _logical_turn_order(messages: list[Message]) -> list[Message]:
    """Keep an assistant reply immediately after its user message for model context."""

    replies_by_user: dict[str, list[Message]] = {}
    for message in messages:
        if message.role == "assistant" and message.reply_to_message_id is not None:
            replies_by_user.setdefault(message.reply_to_message_id, []).append(message)

    ordered: list[Message] = []
    attached_reply_ids = set()
    for message in messages:
        if message.role == "assistant" and message.reply_to_message_id is not None:
            continue
        ordered.append(message)
        for reply in replies_by_user.get(message.id, []):
            ordered.append(reply)
            attached_reply_ids.add(reply.id)
    unattached = [
        message
        for message in messages
        if message.id not in attached_reply_ids and message not in ordered
    ]
    return [*ordered, *unattached]
