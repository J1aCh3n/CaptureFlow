"""Bounded V2 model routing; models propose, deterministic code commits."""

import re
from dataclasses import dataclass
from datetime import date, datetime
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from .date_tool import resolve_relative_date
from .llm import InvalidStructuredOutput, ModelError, TurnRouterModel
from .models import (
    CalendarEvent,
    Message,
    PendingClarification,
    ProposedAction,
    SavedItem,
    ToolOperation,
    TurnDecision,
)
from .storage import CaptureStore
from .time_tool import parse_time as _parse_time


_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(?:api[_ -]?key|password|secret|token)\b\s*[:=]"
)


@dataclass(frozen=True)
class TurnDiagnostics:
    model_seconds: float = 0
    local_seconds: float = 0
    repair_used: bool = False
    outcome: str = "unknown"


@dataclass(frozen=True)
class RouterTurn:
    user_message: Message
    assistant_message: Message
    operations: tuple[ToolOperation, ...] = ()
    model_calls: int = 0
    diagnostics: TurnDiagnostics = TurnDiagnostics(outcome="cached")


class BoundedChatService:
    """Execute at most three validated local actions from one router decision."""

    def __init__(
        self,
        store: CaptureStore,
        router: TurnRouterModel,
        reference_date: date,
        timezone: str = "America/Vancouver",
    ) -> None:
        self.store = store
        self.router = router
        self.reference_date = reference_date
        self.timezone = timezone

    def submit(self, conversation_id: str, content: str, message_id: str | None = None) -> RouterTurn:
        user_message = self._user_message(conversation_id, content, message_id)
        if reply := self.store.get_assistant_reply(user_message.id):
            return RouterTurn(
                user_message,
                reply,
                tuple(self.store.list_tool_operations(user_message.id)),
            )
        if prepared := self.store.list_prepared_tool_operations(user_message.id):
            return self._resume_prepared(user_message, prepared)
        if _CREDENTIAL_PATTERN.search(user_message.content):
            return self._reply_only(
                user_message,
                "I did not save that message because it appears to contain a credential.",
                outcome="blocked_credential",
            )

        pending = self.store.get_pending_clarification(conversation_id)
        context = None if pending is None else self._clarification_context(pending)
        candidate, model_calls, model_seconds = self._route(user_message.content, context)
        if candidate is None:
            return self._reply_only(
                user_message,
                "I could not validate a local action, so nothing was saved.",
                model_calls,
                model_seconds=model_seconds,
                outcome="model_failure",
            )
        try:
            decision = self._validate_decision(candidate, user_message.content, context)
        except ValueError as error:
            repaired, repair_calls, repair_seconds = self._repair(
                user_message.content, candidate, str(error), context
            )
            model_calls += repair_calls
            model_seconds += repair_seconds
            if repaired is None:
                return self._reply_only(
                    user_message,
                    "I could not validate a local action after one repair, so nothing was saved.",
                    model_calls,
                    model_seconds=model_seconds,
                    repair_used=True,
                    outcome="repair_failed",
                )
            try:
                decision = self._validate_decision(repaired, user_message.content, context)
            except ValueError:
                return self._reply_only(
                    user_message,
                    "I could not validate a local action after one repair, so nothing was saved.",
                    model_calls,
                    model_seconds=model_seconds,
                    repair_used=True,
                    outcome="repair_invalid",
                )

        if decision.route == "reply_only":
            return self._reply_only(
                user_message,
                decision.reply or "I can save a local event, idea, or note.",
                model_calls,
                model_seconds=model_seconds,
                repair_used=model_calls == 2,
                outcome="reply_only",
            )
        if decision.route == "clarify":
            if pending is not None:
                return self._reply_only(
                    user_message,
                    decision.clarification_question
                    or "I still need an exact date for the pending local event.",
                    model_calls,
                    model_seconds=model_seconds,
                    repair_used=model_calls == 2,
                    outcome="clarification_repeated",
                )
            return self._save_clarification(
                user_message, decision, model_calls, model_seconds, model_calls == 2
            )
        return self._execute(
            user_message, decision, pending, model_calls, model_seconds, model_calls == 2
        )

    def cancel_pending_clarification(self, conversation_id: str) -> PendingClarification | None:
        pending = self.store.get_pending_clarification(conversation_id)
        if pending is None:
            return None
        return self.store.resolve_pending_clarification(pending.id, "cancelled")

    def _route(
        self, content: str, context: dict | None
    ) -> tuple[TurnDecision | object | None, int, float]:
        started = perf_counter()
        try:
            return (
                self.router.route_turn(
                    content, self.reference_date.isoformat(), self.timezone, context
                ),
                1,
                perf_counter() - started,
            )
        except InvalidStructuredOutput as error:
            return error.raw_output, 1, perf_counter() - started
        except (ModelError, ValidationError):
            return None, 1, perf_counter() - started

    def _repair(
        self, content: str, invalid: TurnDecision | object, error: str, context: dict | None
    ) -> tuple[TurnDecision | None, int, float]:
        started = perf_counter()
        try:
            return (
                self.router.repair_turn(
                    content,
                    self.reference_date.isoformat(),
                    self.timezone,
                    invalid if isinstance(invalid, TurnDecision) else str(invalid),
                    error,
                    context,
                ),
                1,
                perf_counter() - started,
            )
        except (ModelError, ValidationError):
            return None, 1, perf_counter() - started

    def _execute(
        self,
        user_message: Message,
        decision: TurnDecision,
        pending: PendingClarification | None,
        model_calls: int,
        model_seconds: float,
        repair_used: bool,
    ) -> RouterTurn:
        local_started = perf_counter()
        prepared: list[tuple[ToolOperation, CalendarEvent | SavedItem]] = []
        for index, action in enumerate(decision.actions):
            target_or_question = self._target_for_action(user_message, action, index)
            if isinstance(target_or_question, str):
                return self._save_date_clarification(
                    user_message,
                    action,
                    target_or_question,
                    model_calls,
                    model_seconds,
                    repair_used,
                )
            operation, target = target_or_question
            prepared.append((operation, target))

        for operation, _ in prepared:
            self.store.save_tool_operation(operation)

        committed: list[ToolOperation] = []
        try:
            for operation, target in prepared:
                if isinstance(target, CalendarEvent):
                    operation, _ = self.store.commit_prepared_calendar_event(operation.id, target)
                else:
                    operation, _ = self.store.commit_prepared_saved_item(operation.id, target)
                committed.append(operation)
        except (KeyError, ValueError) as error:
            for operation, _ in prepared:
                if operation.status == "prepared":
                    try:
                        self.store.fail_prepared_tool_operation(operation.id, str(error))
                    except ValueError:
                        pass
            return self._reply_only(
                user_message,
                f"Local write failed: {error}",
                model_calls,
                model_seconds=model_seconds,
                local_seconds=perf_counter() - local_started,
                repair_used=repair_used,
                outcome="write_failed",
            )

        if pending is not None:
            self.store.resolve_pending_clarification(pending.id, "resolved")
        summaries = []
        for operation in committed:
            target_id = (operation.result or {}).get("event_id") or (operation.result or {}).get("item_id")
            summaries.append(f"{operation.tool_name} committed ({target_id}).")
        return self._reply_only(
            user_message,
            " ".join(summaries),
            model_calls,
            tuple(committed),
            model_seconds=model_seconds,
            local_seconds=perf_counter() - local_started,
            repair_used=repair_used,
            outcome="committed",
        )

    def _resume_prepared(
        self, user_message: Message, prepared: list[ToolOperation]
    ) -> RouterTurn:
        local_started = perf_counter()
        committed: list[ToolOperation] = []
        try:
            for operation in prepared:
                request = operation.request
                if operation.tool_name == "create_local_event":
                    start_at = _parse_start_at(request["start_at"])
                    target = CalendarEvent(
                        id=_stable_id(user_message.id, f"event:{operation.action_index}"),
                        conversation_id=user_message.conversation_id,
                        title=request["title"],
                        description=request.get("description"),
                        source_message_id=user_message.id,
                        source_quote=request["source_quote"],
                        start_at=start_at,
                        end_at=(
                            None
                            if request.get("end_at") is None
                            else datetime.fromisoformat(request["end_at"])
                        ),
                        all_day=request["all_day"],
                        timezone=request["timezone"],
                        created_at=user_message.created_at,
                        updated_at=user_message.created_at,
                    )
                    operation, _ = self.store.commit_prepared_calendar_event(operation.id, target)
                else:
                    target = SavedItem(
                        id=_stable_id(user_message.id, f"item:{operation.action_index}"),
                        conversation_id=user_message.conversation_id,
                        kind="idea" if operation.tool_name == "save_idea" else "note",
                        content=request["content"],
                        source_message_id=user_message.id,
                        source_quote=request["source_quote"],
                        created_at=user_message.created_at,
                        updated_at=user_message.created_at,
                    )
                    operation, _ = self.store.commit_prepared_saved_item(operation.id, target)
                committed.append(operation)
        except (KeyError, ValueError) as error:
            return self._reply_only(
                user_message,
                f"Could not resume local write: {error}",
                local_seconds=perf_counter() - local_started,
                outcome="resume_failed",
            )
        return self._reply_only(
            user_message,
            "Resumed local write without another model call.",
            operations=tuple(committed),
            local_seconds=perf_counter() - local_started,
            outcome="resumed",
        )

    def _target_for_action(
        self, user_message: Message, action: ProposedAction, index: int
    ) -> tuple[ToolOperation, CalendarEvent | SavedItem] | str:
        tool_name = {
            "create_event": "create_local_event",
            "save_idea": "save_idea",
            "save_note": "save_note",
        }[action.action]
        operation_id = _stable_id(user_message.id, f"operation:{index}:{tool_name}")
        if action.action == "create_event":
            resolution = resolve_relative_date(
                action.date_expression or "", self.reference_date, self.timezone
            )
            if resolution.status != "resolved" or resolution.resolved_date is None:
                return "Which exact date should I use for this local event?"
            start_at: date | datetime = resolution.resolved_date
            end_at: datetime | None = None
            all_day = True
            if action.time_expression:
                parsed_time = _parse_time(action.time_expression)
                if parsed_time is None:
                    return "Which exact time should I use (for example, 15:30)?"
                start_at = datetime.combine(
                    resolution.resolved_date, parsed_time, tzinfo=ZoneInfo(self.timezone)
                )
                all_day = False
                if action.end_time_expression:
                    parsed_end_time = _parse_time(action.end_time_expression)
                    if parsed_end_time is None:
                        return "Which exact end time should I use (for example, 16:30)?"
                    end_at = datetime.combine(
                        resolution.resolved_date,
                        parsed_end_time,
                        tzinfo=ZoneInfo(self.timezone),
                    )
                    if end_at <= start_at:
                        return "The event end time must be after its start time."
            event = CalendarEvent(
                id=_stable_id(user_message.id, f"event:{index}"),
                conversation_id=user_message.conversation_id,
                title=action.content,
                description=action.description,
                source_message_id=user_message.id,
                source_quote=action.source_quote,
                start_at=start_at,
                end_at=end_at,
                all_day=all_day,
                timezone=self.timezone,
                created_at=user_message.created_at,
                updated_at=user_message.created_at,
            )
            request = {
                "title": event.title,
                "description": event.description,
                "source_quote": event.source_quote,
                "date_expression": action.date_expression,
                "time_expression": action.time_expression,
                "end_time_expression": action.end_time_expression,
                "start_at": event.start_at.isoformat(),
                "end_at": None if event.end_at is None else event.end_at.isoformat(),
                "all_day": event.all_day,
                "timezone": event.timezone,
            }
            return (
                ToolOperation(
                    id=operation_id,
                    source_message_id=user_message.id,
                    action_index=index,
                    tool_name=tool_name,
                    request=request,
                    status="prepared",
                    created_at=user_message.created_at,
                    updated_at=user_message.created_at,
                ),
                event,
            )
        item = SavedItem(
            id=_stable_id(user_message.id, f"item:{index}"),
            conversation_id=user_message.conversation_id,
            kind="idea" if action.action == "save_idea" else "note",
            content=action.content,
            source_message_id=user_message.id,
            source_quote=action.source_quote,
            created_at=user_message.created_at,
            updated_at=user_message.created_at,
        )
        return (
            ToolOperation(
                id=operation_id,
                source_message_id=user_message.id,
                action_index=index,
                tool_name=tool_name,
                request={"content": item.content, "source_quote": item.source_quote},
                status="prepared",
                created_at=user_message.created_at,
                updated_at=user_message.created_at,
            ),
            item,
        )

    def _save_clarification(
        self,
        user_message: Message,
        decision: TurnDecision,
        model_calls: int,
        model_seconds: float,
        repair_used: bool,
    ) -> RouterTurn:
        local_started = perf_counter()
        action = decision.pending_action
        assert action is not None
        assistant = self._reply_message(user_message, decision.clarification_question or "Which date?")
        clarification = PendingClarification(
            id=_stable_id(user_message.id, "clarification"),
            conversation_id=user_message.conversation_id,
            source_message_id=user_message.id,
            assistant_message_id=assistant.id,
            original_action=action.model_dump(mode="json"),
            created_at=user_message.created_at,
        )
        self.store.save_pending_clarification(clarification)
        return RouterTurn(
            user_message,
            assistant,
            model_calls=model_calls,
            diagnostics=TurnDiagnostics(
                model_seconds=model_seconds,
                local_seconds=perf_counter() - local_started,
                repair_used=repair_used,
                outcome="clarification_created",
            ),
        )

    def _save_date_clarification(
        self,
        user_message: Message,
        action: ProposedAction,
        question: str,
        model_calls: int,
        model_seconds: float,
        repair_used: bool,
    ) -> RouterTurn:
        if self.store.get_pending_clarification(user_message.conversation_id) is not None:
            return self._reply_only(
                user_message,
                question,
                model_calls,
                model_seconds=model_seconds,
                repair_used=repair_used,
                outcome="clarification_repeated",
            )
        decision = TurnDecision(
            route="clarify",
            clarification_question=question,
            pending_action=action.model_copy(
                update={
                    "date_expression": None,
                    "time_expression": None,
                    "end_time_expression": None,
                }
            ),
        )
        return self._save_clarification(
            user_message, decision, model_calls, model_seconds, repair_used
        )

    def _reply_only(
        self,
        user_message: Message,
        content: str,
        model_calls: int = 0,
        operations: tuple[ToolOperation, ...] = (),
        *,
        model_seconds: float = 0,
        local_seconds: float = 0,
        repair_used: bool = False,
        outcome: str = "reply_only",
    ) -> RouterTurn:
        local_started = perf_counter()
        assistant = self._reply_message(user_message, content)
        return RouterTurn(
            user_message,
            assistant,
            operations,
            model_calls,
            TurnDiagnostics(
                model_seconds=model_seconds,
                local_seconds=local_seconds + perf_counter() - local_started,
                repair_used=repair_used,
                outcome=outcome,
            ),
        )

    def _reply_message(self, user_message: Message, content: str) -> Message:
        return self.store.save_message(
            Message(
                id=_stable_id(user_message.id, "router-assistant"),
                conversation_id=user_message.conversation_id,
                role="assistant",
                content=content,
                reply_to_message_id=user_message.id,
                created_at=user_message.created_at,
            )
        )

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

    @staticmethod
    def _clarification_context(pending: PendingClarification) -> dict:
        return {
            "original_action": pending.original_action,
            "clarification_question": "A date is required for the pending local event.",
        }

    @staticmethod
    def _validate_decision(
        candidate: TurnDecision | object, message: str, context: dict | None
    ) -> TurnDecision:
        decision = TurnDecision.model_validate(candidate)
        normalized_message = _normalize(message)
        context_quote = _normalize(str((context or {}).get("original_action", {}).get("source_quote", "")))
        seen = set()
        for action in [*decision.actions, *([decision.pending_action] if decision.pending_action else [])]:
            quote = _normalize(action.source_quote)
            if not quote or (quote not in normalized_message and quote != context_quote):
                raise ValueError("every source quote must occur in the current message or pending source")
            if action.description:
                description = _normalize(action.description)
                if description not in normalized_message and description not in context_quote:
                    raise ValueError("an event description must occur in its current or pending source")
            if action.action == "create_event" and decision.route == "execute" and not action.date_expression:
                raise ValueError("an event needs a date or clarification")
            if action.action == "create_event" and action.end_time_expression:
                if not action.time_expression:
                    raise ValueError("an event end time requires a start time")
                start_time = _parse_time(action.time_expression)
                end_time = _parse_time(action.end_time_expression)
                if start_time is None or end_time is None or end_time <= start_time:
                    raise ValueError("an event needs a valid end time after its start time")
            if action.action != "create_event" and (
                action.description
                or action.date_expression
                or action.time_expression
                or action.end_time_expression
            ):
                raise ValueError("ideas and notes cannot use event fields")
            identity = (action.action, _normalize(action.content))
            if identity in seen:
                raise ValueError("duplicate actions are not allowed in one turn")
            seen.add(identity)
        if context is not None and decision.route == "execute":
            if len(decision.actions) != 1 or decision.actions[0].action != "create_event":
                raise ValueError("a pending date clarification can resolve only its event")
        return decision


def _parse_start_at(value: str) -> date | datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return date.fromisoformat(value)


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _stable_id(message_id: str, suffix: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"captureflow:{message_id}:{suffix}"))
