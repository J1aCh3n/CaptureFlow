"""Bounded recall and source-grounded synthesis over local Calendar and Notes."""

from dataclasses import dataclass
from datetime import date, datetime
import re
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import ValidationError

from .date_tool import resolve_relative_date
from .llm import InvalidStructuredOutput, LocalContextModel, ModelError
from .models import (
    CalendarEvent,
    LocalContextAnswerDecision,
    LocalContextSourceRef,
    Message,
    ProjectRecord,
    SavedItem,
    utc_now,
)
from .storage import CaptureStore

MAX_LOCAL_CONTEXT_SOURCES = 6


@dataclass(frozen=True)
class LocalContextTurn:
    user_message: Message
    assistant_message: Message
    sources: tuple[CalendarEvent | SavedItem | ProjectRecord, ...]
    outcome: str
    decision: LocalContextAnswerDecision | None = None
    model_calls: int = 0
    model_seconds: float = 0
    fallback_used: bool = False


LOCAL_WRITE_MARKERS = (
    "create",
    "add to",
    "save",
    "schedule a",
    "schedule an",
    "schedule the",
    "schedule my",
    "set up",
    "remind me",
    "创建",
    "新建",
    "新增",
    "添加",
    "加入日历",
    "记到日历",
    "提醒我",
    "帮我安排",
    "请安排",
    "保存",
)
LOCAL_WRITE_TARGETS = (
    "calendar",
    "event",
    "meeting",
    "idea",
    "note",
    "日历",
    "日程",
    "会议",
    "想法",
    "笔记",
    "备忘",
)


def is_local_write_request(query: str) -> bool:
    """Recognize an explicit request for one of V2's bounded local writes."""

    normalized = query.casefold()
    return any(marker in normalized for marker in LOCAL_WRITE_MARKERS) and any(
        marker in normalized for marker in LOCAL_WRITE_TARGETS
    )


def is_local_context_query(query: str) -> bool:
    """Recognize a small read-only Calendar/Note query vocabulary before model routing."""

    normalized = query.casefold()
    if is_local_write_request(query):
        return False
    topic_markers = (
        "calendar",
        "event",
        "meeting",
        "activity",
        "activities",
        "appointment",
        "schedule",
        "日历",
        "日程",
        "会议",
        "议程",
        "活动",
        "安排",
        "行程",
    )
    question_markers = (
        "?",
        "？",
        "when",
        "what",
        "where",
        "which",
        "show me",
        "find",
        "look up",
        "list",
        "tell me",
        "几点",
        "什么时候",
        "什么时间",
        "什么内容",
        "内容是什么",
        "需要准备什么",
        "该准备什么",
        "要做什么",
        "有哪些",
        "有什么",
        "如何准备",
        "怎么准备",
        "查询",
        "查看",
        "告诉我",
    )
    return any(marker in normalized for marker in topic_markers) and any(
        marker in normalized for marker in question_markers
    )


class LocalContextService:
    """Keep local selection and provenance deterministic around an optional one-call synthesis."""

    def __init__(
        self,
        store: CaptureStore,
        reference_date: date,
        timezone: str,
        model: LocalContextModel | None = None,
    ) -> None:
        self.store = store
        self.reference_date = reference_date
        self.timezone = timezone
        self.model = model

    def submit(
        self, conversation_id: str, content: str, message_id: str | None = None
    ) -> LocalContextTurn:
        user_message = self._user_message(conversation_id, content, message_id)
        existing_reply = self.store.get_assistant_reply(user_message.id)
        if existing_reply is not None:
            return LocalContextTurn(
                user_message, existing_reply, (), "reloaded"
            )
        sources = self._select_sources(conversation_id, content)
        if not sources:
            return LocalContextTurn(
                user_message,
                self._reply(
                    user_message,
                    "I do not have matching local Calendar, Note/Idea, or same-project tracker "
                    "context for that yet.",
                ),
                (),
                "insufficient_context",
            )
        if self.model is None:
            return self._fallback_turn(user_message, content, sources)

        runtime_context = self._runtime_context(content)
        model_events = [
            self._event_for_model(source) for source in sources if isinstance(source, CalendarEvent)
        ]
        model_items = [
            self._item_for_model(source) for source in sources if isinstance(source, SavedItem)
        ]
        model_project_records = [
            self._project_record_for_model(source)
            for source in sources
            if isinstance(source, ProjectRecord)
        ]
        started = perf_counter()
        model_calls = 1
        candidate: LocalContextAnswerDecision | object | None = None
        try:
            candidate = self.model.answer_local_context(
                content,
                runtime_context,
                model_events,
                model_items,
                model_project_records,
            )
            decision = self._validate_decision(candidate, content, sources)
        except InvalidStructuredOutput as error:
            invalid_decision = self._invalid_decision_for_repair(error, candidate)
            try:
                repaired = self.model.repair_local_context_answer(
                    content,
                    runtime_context,
                    model_events,
                    model_items,
                    model_project_records,
                    invalid_decision,
                    str(error),
                )
                model_calls = 2
                decision = self._validate_decision(repaired, content, sources)
            except (ModelError, ValidationError, ValueError):
                return self._fallback_turn(
                    user_message,
                    content,
                    sources,
                    model_calls=2,
                    model_seconds=perf_counter() - started,
                )
        except (ValidationError, ValueError) as error:
            invalid_decision = self._invalid_decision_for_repair(error, candidate)
            try:
                repaired = self.model.repair_local_context_answer(
                    content,
                    runtime_context,
                    model_events,
                    model_items,
                    model_project_records,
                    invalid_decision,
                    str(error),
                )
                model_calls = 2
                decision = self._validate_decision(repaired, content, sources)
            except (ModelError, ValidationError, ValueError):
                return self._fallback_turn(
                    user_message,
                    content,
                    sources,
                    model_calls=2,
                    model_seconds=perf_counter() - started,
                )
        except ModelError:
            return self._fallback_turn(
                user_message,
                content,
                sources,
                model_calls=1,
                model_seconds=perf_counter() - started,
            )

        model_seconds = perf_counter() - started
        selected_sources = self._sources_for_decision(decision, sources)
        assistant = self._reply(
            user_message, _format_model_reply(content, decision, selected_sources)
        )
        if selected_sources:
            self._save_sources(assistant, selected_sources)
        return LocalContextTurn(
            user_message,
            assistant,
            tuple(selected_sources),
            "grounded_local_context_model" if decision.outcome == "answer" else decision.outcome,
            decision=decision,
            model_calls=model_calls,
            model_seconds=model_seconds,
        )

    def _fallback_turn(
        self,
        user_message: Message,
        query: str,
        sources: list[CalendarEvent | SavedItem | ProjectRecord],
        *,
        model_calls: int = 0,
        model_seconds: float = 0,
    ) -> LocalContextTurn:
        assistant = self._reply(user_message, _format_reply(query, sources))
        self._save_sources(assistant, sources)
        return LocalContextTurn(
            user_message,
            assistant,
            tuple(sources),
            "grounded_local_context",
            model_calls=model_calls,
            model_seconds=model_seconds,
            fallback_used=True,
        )

    def _save_sources(
        self, assistant: Message, sources: list[CalendarEvent | SavedItem | ProjectRecord]
    ) -> None:
        local_references = [
            LocalContextSourceRef(
                source_type="calendar_event" if isinstance(source, CalendarEvent) else "saved_item",
                source_id=source.id,
            )
            for source in sources
            if isinstance(source, (CalendarEvent, SavedItem))
        ]
        if local_references:
            self.store.save_local_context_answer_sources(assistant.id, local_references)
        project_record_ids = [source.id for source in sources if isinstance(source, ProjectRecord)]
        if project_record_ids:
            self.store.save_project_answer_sources(assistant.id, project_record_ids)

    def _select_sources(
        self, conversation_id: str, query: str
    ) -> list[CalendarEvent | SavedItem | ProjectRecord]:
        target_date = _query_date(query, self.reference_date, self.timezone)
        events = self.store.list_calendar_events()
        if target_date is not None:
            events = [event for event in events if _event_date(event) == target_date]
        else:
            events = _rank(query, events, lambda event: f"{event.title} {event.source_quote}")

        project = self.store.get_conversation_project(conversation_id)
        project_records: list[ProjectRecord] = []
        saved_items = self.store.list_saved_items(kinds=("note", "idea"))
        if project is not None:
            project_records = _rank(
                query,
                self.store.list_project_records(project.id, states=("active",)),
                lambda record: f"{record.content} {record.rationale or ''} {record.source_quote}",
            )
            saved_items = [
                item
                for item in saved_items
                if (assigned := self.store.get_conversation_project(item.conversation_id)) is not None
                and assigned.id == project.id
            ]
        notes = _rank(
            query,
            saved_items,
            lambda item: f"{item.content} {item.source_quote}",
        )
        return [*events[:3], *project_records[:3], *notes[:3]][:MAX_LOCAL_CONTEXT_SOURCES]

    def _runtime_context(self, query: str) -> dict[str, str | None]:
        target_date = _query_date(query, self.reference_date, self.timezone)
        return {
            "reference_date": self.reference_date.isoformat(),
            "timezone": self.timezone,
            "target_date": None if target_date is None else target_date.isoformat(),
        }

    @staticmethod
    def _event_for_model(event: CalendarEvent) -> dict:
        return {
            "source_id": event.id,
            "title": event.title,
            "start_at": event.start_at.isoformat(),
            "end_at": None if event.end_at is None else event.end_at.isoformat(),
            "all_day": event.all_day,
            "source_quote": event.source_quote,
        }

    @staticmethod
    def _item_for_model(item: SavedItem) -> dict:
        return {
            "source_id": item.id,
            "kind": item.kind,
            "content": item.content,
            "source_quote": item.source_quote,
        }

    @staticmethod
    def _project_record_for_model(record: ProjectRecord) -> dict:
        return {
            "source_id": record.id,
            "kind": record.kind,
            "content": record.content,
            "state": record.state,
            "source_quote": record.source_quote,
        }

    def _validate_decision(
        self,
        candidate: LocalContextAnswerDecision | object,
        query: str,
        sources: list[CalendarEvent | SavedItem | ProjectRecord],
    ) -> LocalContextAnswerDecision:
        decision = LocalContextAnswerDecision.model_validate(candidate)
        events = {source.id: source for source in sources if isinstance(source, CalendarEvent)}
        items = {source.id: source for source in sources if isinstance(source, SavedItem)}
        project_records = {
            source.id: source for source in sources if isinstance(source, ProjectRecord)
        }
        selected_ids = {
            *decision.selected_event_ids,
            *decision.selected_item_ids,
            *decision.selected_project_record_ids,
        }
        if any(source_id not in events for source_id in decision.selected_event_ids):
            raise ValueError("local-context answer selected an event outside the supplied candidates")
        if any(source_id not in items for source_id in decision.selected_item_ids):
            raise ValueError("local-context answer selected an item outside the supplied candidates")
        if any(
            source_id not in project_records
            for source_id in decision.selected_project_record_ids
        ):
            raise ValueError(
                "local-context answer selected a project record outside the supplied candidates"
            )
        if any(
            source_id not in selected_ids
            for point in decision.summary_points
            for source_id in point.source_ids
        ):
            raise ValueError("local-context summary cited an unselected source")
        if any(
            source_id not in selected_ids
            for suggestion in decision.preparation_suggestions
            for source_id in suggestion.based_on_source_ids
        ):
            raise ValueError("local-context suggestion cited an unselected source")
        if any(_contains_calendar_time_or_date(point.text) for point in decision.summary_points):
            raise ValueError("local-context summary must leave Calendar date and time to local rendering")

        if decision.outcome == "insufficient_context":
            raise ValueError(
                "selected local evidence requires a partial answer; list unsupported details in "
                "missing_information instead"
            )

        target_date = _query_date(query, self.reference_date, self.timezone)
        if target_date is not None and any(
            _event_date(events[source_id]) != target_date
            for source_id in decision.selected_event_ids
        ):
            raise ValueError("local-context answer selected an event outside the requested date")
        if len(events) > 1 and decision.outcome == "answer" and set(decision.selected_event_ids) != set(events):
            raise ValueError("multiple local calendar events require every candidate or clarification")
        if _asks_for_event_time(query) and not decision.selected_event_ids:
            if "time" not in decision.missing_information:
                raise ValueError(
                    "a local time answer without a selected Calendar event must mark time missing"
                )
        return decision

    @staticmethod
    def _invalid_decision_for_repair(
        error: Exception, candidate: object | None
    ) -> LocalContextAnswerDecision | str:
        if isinstance(error, InvalidStructuredOutput):
            return error.raw_output
        if isinstance(candidate, LocalContextAnswerDecision):
            return candidate
        return str(candidate) if candidate is not None else str(error)

    @staticmethod
    def _sources_for_decision(
        decision: LocalContextAnswerDecision,
        sources: list[CalendarEvent | SavedItem | ProjectRecord],
    ) -> list[CalendarEvent | SavedItem | ProjectRecord]:
        if decision.outcome == "clarify":
            return sources
        selected_ids = {
            *decision.selected_event_ids,
            *decision.selected_item_ids,
            *decision.selected_project_record_ids,
        }
        return [source for source in sources if source.id in selected_ids]

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


def _asks_for_event_time(query: str) -> bool:
    normalized = query.casefold()
    return any(
        marker in normalized
        for marker in ("when", "time", "what time", "几点", "什么时候", "什么时间", "时间")
    )


def _asks_for_calendar_fact(query: str) -> bool:
    normalized = query.casefold()
    return _asks_for_event_time(query) or any(
        marker in normalized
        for marker in (
            "calendar",
            "event",
            "activity",
            "activities",
            "appointment",
            "日历",
            "日程",
            "活动",
            "安排",
            "行程",
        )
    )


def _contains_calendar_time_or_date(value: str) -> bool:
    return bool(
        re.search(
            r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:a\.?m\.?|p\.?m\.?)\b",
            value,
            re.IGNORECASE,
        )
        or re.search(
            r"(?:上午|下午|晚上)\s*(?:[一二三四五六七八九十两]+|\d{1,2})点(?:半|\d{1,2}分?)?",
            value,
        )
    )


def _format_model_reply(
    query: str,
    decision: LocalContextAnswerDecision,
    sources: list[CalendarEvent | SavedItem | ProjectRecord],
) -> str:
    events = [source for source in sources if isinstance(source, CalendarEvent)]
    if decision.outcome == "clarify":
        candidates = [f"- {event.start_at.isoformat()} — {event.title}" for event in events]
        prefix = "I found multiple possible local calendar events:"
        return "\n".join([prefix, *candidates, decision.clarification_question or "Which event do you mean?"])
    if decision.outcome == "insufficient_context":
        missing = ", ".join(decision.missing_information) or "the requested detail"
        return f"The selected local records do not establish {missing}."

    lines: list[str] = []
    if _asks_for_calendar_fact(query) and not events:
        lines.append("No matching Calendar event is stored for the requested date or description.")
    if events:
        lines.append("Saved local calendar:")
        lines.extend(f"- {event.start_at.isoformat()} — {event.title}" for event in events)
    lines.append("Grounded local context:")
    lines.extend(f"- {point.text}" for point in decision.summary_points)
    if decision.preparation_suggestions:
        lines.append("Possible preparation (inferred from saved local context):")
        lines.extend(f"- {suggestion.text}" for suggestion in decision.preparation_suggestions)
    if decision.missing_information:
        lines.append("Not stored locally: " + ", ".join(decision.missing_information) + ".")
    return "\n".join(lines)


def _format_reply(
    query: str, sources: list[CalendarEvent | SavedItem | ProjectRecord]
) -> str:
    events = [source for source in sources if isinstance(source, CalendarEvent)]
    notes = [source for source in sources if isinstance(source, SavedItem)]
    project_records = [source for source in sources if isinstance(source, ProjectRecord)]
    lines: list[str] = []
    if _asks_for_calendar_fact(query) and not events:
        lines.append("No matching Calendar event is stored for the requested date or description.")
    if events:
        heading = "I found these local calendar events:" if len(events) > 1 else "I found this local calendar event:"
        lines.append(heading)
        lines.extend(f"- {event.start_at.isoformat()} — {event.title}" for event in events)
    if notes:
        lines.append("Saved local note/idea context:")
        lines.extend(f"- {item.kind.title()}: {item.content}" for item in notes)
    if project_records:
        lines.append("Same-project tracker context (not Calendar events):")
        lines.extend(
            f"- {record.kind.title()}: {record.content}" for record in project_records
        )
    if any(marker in query.casefold() for marker in ("prepare", "准备", "要做什么")):
        lines.append("No explicit personal preparation task is stored in the selected local records.")
    if len(events) > 1 or len(notes) > 1:
        lines.append("If you mean one specific item, name it and I will narrow the local lookup.")
    return "\n".join(lines)
