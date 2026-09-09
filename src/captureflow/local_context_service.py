"""Bounded, deterministic recall over CaptureFlow's own local Calendar and Notes."""

from dataclasses import dataclass
from datetime import date, datetime
import re
from uuid import NAMESPACE_URL, uuid4, uuid5

from .date_tool import resolve_relative_date
from .models import CalendarEvent, LocalContextSourceRef, Message, SavedItem, utc_now
from .storage import CaptureStore

MAX_LOCAL_CONTEXT_SOURCES = 6


@dataclass(frozen=True)
class LocalContextTurn:
    user_message: Message
    assistant_message: Message
    sources: tuple[CalendarEvent | SavedItem, ...]
    outcome: str


def is_local_context_query(query: str) -> bool:
    """Recognize a small read-only Calendar/Note query vocabulary before model routing."""

    normalized = query.casefold()
    topic_markers = ("calendar", "event", "meeting", "日历", "日程", "会议", "议程")
    question_markers = (
        "?", "？", "when", "what", "where", "agenda", "prepare", "time", "tomorrow",
        "几点", "什么时候", "什么时间", "内容", "准备", "要做什么", "安排", "明天",
    )
    return any(marker in normalized for marker in topic_markers) and any(
        marker in normalized for marker in question_markers
    )


class LocalContextService:
    """Return only selected local facts; it neither calls a model nor changes local state."""

    def __init__(self, store: CaptureStore, reference_date: date, timezone: str) -> None:
        self.store = store
        self.reference_date = reference_date
        self.timezone = timezone

    def submit(
        self, conversation_id: str, content: str, message_id: str | None = None
    ) -> LocalContextTurn:
        user_message = self._user_message(conversation_id, content, message_id)
        existing_reply = self.store.get_assistant_reply(user_message.id)
        if existing_reply is not None:
            return LocalContextTurn(
                user_message, existing_reply, (), "reloaded"
            )
        sources = self._select_sources(content)
        if not sources:
            return LocalContextTurn(
                user_message,
                self._reply(user_message, "I do not have matching local Calendar or Note context for that yet."),
                (),
                "insufficient_context",
            )
        assistant = self._reply(user_message, _format_reply(content, sources))
        references = [
            LocalContextSourceRef(
                source_type="calendar_event" if isinstance(source, CalendarEvent) else "saved_item",
                source_id=source.id,
            )
            for source in sources
        ]
        self.store.save_local_context_answer_sources(assistant.id, references)
        return LocalContextTurn(user_message, assistant, tuple(sources), "grounded_local_context")

    def _select_sources(self, query: str) -> list[CalendarEvent | SavedItem]:
        target_date = _query_date(query, self.reference_date, self.timezone)
        events = self.store.list_calendar_events()
        if target_date is not None:
            events = [event for event in events if _event_date(event) == target_date]
        else:
            events = _rank(query, events, lambda event: f"{event.title} {event.source_quote}")
        notes = _rank(
            query,
            self.store.list_saved_items(kinds=("note", "idea")),
            lambda item: f"{item.content} {item.source_quote}",
        )
        return [*events[:3], *notes[:3]][:MAX_LOCAL_CONTEXT_SOURCES]

    def _user_message(self, conversation_id: str, content: str, message_id: str | None) -> Message:
        if message_id is not None:
            try:
                existing = self.store.get_message(message_id)
            except KeyError:
                pass
            else:
                if existing.role != "user" or existing.conversation_id != conversation_id or existing.content != content:
                    raise ValueError("message ID already exists with different content")
                return existing
        return self.store.save_message(
            Message(id=message_id or str(uuid4()), conversation_id=conversation_id, role="user", content=content)
        )

    def _reply(self, user_message: Message, content: str) -> Message:
        return self.store.save_message(
            Message(
                id=str(uuid5(NAMESPACE_URL, f"captureflow:local-context:{user_message.id}")),
                conversation_id=user_message.conversation_id,
                role="assistant",
                content=content,
                reply_to_message_id=user_message.id,
                created_at=utc_now(),
            )
        )


def _query_date(query: str, reference_date: date, timezone: str) -> date | None:
    normalized = query.casefold()
    for expression in ("明天", "tomorrow", "今天", "today"):
        if expression in normalized:
            result = resolve_relative_date(expression, reference_date, timezone)
            return result.resolved_date
    match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", query)
    if match:
        result = resolve_relative_date(match.group(), reference_date, timezone)
        return result.resolved_date
    return None


def _event_date(event: CalendarEvent) -> date:
    return event.start_at.date() if isinstance(event.start_at, datetime) else event.start_at


def _rank(query: str, candidates: list, text_for_candidate) -> list:
    terms = _terms(query)
    ranked = [
        (len(terms & _terms(text_for_candidate(candidate))), candidate)
        for candidate in candidates
    ]
    return [candidate for score, candidate in sorted(ranked, key=lambda item: (item[0], item[1].id), reverse=True) if score]


def _terms(value: str) -> set[str]:
    ignored_words = {"a", "an", "the", "and", "for", "is", "of", "to", "what", "when", "where", "my"}
    words = set(re.findall(r"[a-z0-9]+", value.casefold())) - ignored_words
    ignored = set("我你他她它的是了在有和与或为把将要做什么明天今天请帮问")
    chinese = {character for character in value if "\u4e00" <= character <= "\u9fff" and character not in ignored}
    return words | chinese


def _format_reply(query: str, sources: list[CalendarEvent | SavedItem]) -> str:
    events = [source for source in sources if isinstance(source, CalendarEvent)]
    notes = [source for source in sources if isinstance(source, SavedItem)]
    lines: list[str] = []
    if events:
        heading = "I found these local calendar events:" if len(events) > 1 else "I found this local calendar event:"
        lines.append(heading)
        lines.extend(f"- {event.start_at.isoformat()} — {event.title}" for event in events)
    if notes:
        lines.append("Saved local note/idea context:")
        lines.extend(f"- {item.kind.title()}: {item.content}" for item in notes)
    if any(marker in query.casefold() for marker in ("prepare", "准备", "要做什么")):
        lines.append("No explicit personal preparation task is stored in the selected local records.")
    if len(events) > 1 or len(notes) > 1:
        lines.append("If you mean one specific item, name it and I will narrow the local lookup.")
    return "\n".join(lines)
