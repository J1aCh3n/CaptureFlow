"""What the contextual router sees: a recent message window and a few candidate records."""

from dataclasses import dataclass
from datetime import date

from .local_context_service import match_terms
from .models import CalendarEvent, Message, ProjectRecord, SavedItem
from .storage import CaptureStore


MAX_CONTEXT_MESSAGES = 8
MAX_CONTEXT_CHARACTERS = 12_000
MAX_ROUTER_CANDIDATES_PER_KIND = 3


@dataclass(frozen=True)
class ContextWindow:
    messages: tuple[Message, ...]
    omitted_message_count: int

    @property
    def truncated(self) -> bool:
        return self.omitted_message_count > 0

    def for_router(self) -> list[dict[str, str]]:
        """Expose stable message IDs for cross-message router evidence."""

        return [
            {"message_id": message.id, "role": message.role, "content": message.content}
            for message in self.messages
        ]


@dataclass(frozen=True)
class RouterCandidates:
    """Bounded, workspace-safe records that a contextual router may reference."""

    events: tuple[CalendarEvent, ...]
    saved_items: tuple[SavedItem, ...]
    project_records: tuple[ProjectRecord, ...]


def select_context_window(
    messages: list[Message],
    *,
    max_messages: int = MAX_CONTEXT_MESSAGES,
    max_characters: int = MAX_CONTEXT_CHARACTERS,
) -> ContextWindow:
    """Choose one contiguous recent window without truncating a persisted message."""

    if max_messages < 1 or max_characters < 1:
        raise ValueError("context window limits must be positive")
    if not messages:
        raise ValueError("a context window needs at least one message")

    ordered_messages = _logical_turn_order(messages)
    selected_reversed: list[Message] = []
    character_count = 0
    for index in range(len(ordered_messages) - 1, -1, -1):
        message = ordered_messages[index]
        would_exceed_message_limit = len(selected_reversed) >= max_messages
        would_exceed_character_limit = character_count + len(message.content) > max_characters
        if selected_reversed and (would_exceed_message_limit or would_exceed_character_limit):
            return ContextWindow(
                messages=tuple(reversed(selected_reversed)), omitted_message_count=index + 1
            )
        selected_reversed.append(message)
        character_count += len(message.content)

    return ContextWindow(messages=tuple(reversed(selected_reversed)), omitted_message_count=0)


def select_router_candidates(
    store: CaptureStore,
    conversation_id: str,
    reference_date: date | None = None,
    message: str | None = None,
    *,
    max_per_kind: int = MAX_ROUTER_CANDIDATES_PER_KIND,
) -> RouterCandidates:
    """Select stable local candidates without widening the current workspace or Project scope."""

    if max_per_kind < 1:
        raise ValueError("router candidate limit must be positive")
    conversation = store.get_conversation(conversation_id)
    events = tuple(
        _events_nearest(
            store.list_calendar_events(conversation.workspace_id), reference_date, max_per_kind
        )
    )
    project = store.get_conversation_project(conversation_id)
    if project is None:
        return RouterCandidates(
            events=events,
            saved_items=tuple(
                _most_relevant(
                    store.list_saved_items(conversation.workspace_id), message, max_per_kind
                )
            ),
            project_records=(),
        )

    return RouterCandidates(
        events=events,
        saved_items=tuple(
            _most_relevant(
                store.list_saved_items_for_project(project.id, conversation.workspace_id),
                message,
                max_per_kind,
            )
        ),
        project_records=tuple(
            _most_relevant(
                store.list_project_records(project.id, states=("active",)), message, max_per_kind
            )
        ),
    )


def _most_relevant(
    records: list[ProjectRecord] | list[SavedItem], message: str | None, limit: int
) -> list:
    """Keep the records a message is most likely about, most recently updated on a tie.

    The stored order is recency, so a fixed prefix of it hides an older task the
    moment a project holds more active records than the budget. A message that
    shares no terms with any record keeps exactly the old recency order.
    """

    if not message:
        return records[:limit]
    terms = match_terms(message)
    scored = [
        (len(terms & match_terms(f"{record.content} {record.source_quote}")), position, record)
        for position, record in enumerate(records)
    ]
    return [record for _, _, record in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def _events_nearest(
    events: list[CalendarEvent], reference_date: date | None, limit: int
) -> list[CalendarEvent]:
    """Keep the few events closest to the reference date, preferring an upcoming one.

    The stored order is the whole workspace from its earliest event, so a fixed
    prefix of it hides everything a conversation is actually about as soon as a
    few old events accumulate.
    """

    if reference_date is None:
        return events[:limit]
    nearest = sorted(
        events,
        key=lambda event: (
            abs((event.start_date - reference_date).days),
            event.start_date < reference_date,
            event.start_date,
            event.id,
        ),
    )[:limit]
    return sorted(nearest, key=lambda event: (event.start_date, event.id))


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
