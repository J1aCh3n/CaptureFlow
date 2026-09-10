"""One bounded, contextual entry point for ordinary CaptureFlow chat."""

from dataclasses import dataclass
from datetime import date, datetime, time
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from .context_service import ProjectContextWindow, RouterCandidates, select_project_context, select_router_candidates
from .date_tool import resolve_relative_date
from .llm import InvalidStructuredOutput, ModelError, ContextualConversationModel
from .local_context_service import LocalContextService
from .memory import contains_credential_disclosure
from .models import (
    CalendarEvent,
    CalendarEventUpdateOperation,
    ContextualAction,
    ConversationDecision,
    EvidenceReference,
    LocalContextSourceRef,
    Message,
    PendingClarification,
    ProjectChangeOperation,
    ProjectRecord,
    SavedItem,
    ToolOperation,
    utc_now,
)
from .project_service import ProjectTrackerService
from .storage import CaptureStore


RouterOperation = ToolOperation | CalendarEventUpdateOperation | ProjectChangeOperation


@dataclass(frozen=True)
class ConversationRouterTurn:
    user_message: Message
    assistant_message: Message
    decision: ConversationDecision | None
    context_window: ProjectContextWindow
    candidates: RouterCandidates
    operations: tuple[RouterOperation, ...]
    model_calls: int
    model_seconds: float
    local_seconds: float
    outcome: str


class ConversationRouterService:
    """Persist one normal-language turn, then execute exactly one validated decision."""

    def __init__(
        self,
        store: CaptureStore,
        model: ContextualConversationModel,
        reference_date: date,
        timezone: str = "America/Vancouver",
    ) -> None:
        self.store = store
        self.model = model
        self.reference_date = reference_date
        self.timezone = timezone

    def submit(
        self, conversation_id: str, content: str, message_id: str | None = None) -> ConversationRouterTurn:
        user_message = self._user_message(conversation_id, content, message_id)
        context_window = select_project_context(self.store.list_messages(conversation_id))
        candidates = select_router_candidates(self.store, conversation_id)
        if existing_reply := self.store.get_assistant_reply(user_message.id):
            return self._turn(
                user_message,
                existing_reply,
                None,
                context_window,
                candidates,
                self._persisted_operations(user_message.id),
                outcome="reloaded",
            )
        if resumed := self._resume_persisted_operations(user_message):
            return self._reply(
                user_message,
                "Resumed the saved local operation without another model call.",
                None,
                context_window,
                candidates,
                resumed,
                outcome="resumed",
            )
        if contains_credential_disclosure(user_message.content):
            return self._reply(
                user_message,
                "I did not process that message because it appears to contain a credential.",
                None,
                context_window,
                candidates,
                (),
                outcome="blocked_credential",
            )

        pending = self.store.get_pending_clarification(conversation_id)
        started = perf_counter()
        try:
            decision = ConversationDecision.model_validate(
                self.model.propose_conversation_decision(
                    current_message=self._message_for_router(user_message),
                    recent_messages=context_window.for_router(),
                    candidate_events=[self._event_for_router(event) for event in candidates.events],
                    candidate_saved_items=[self._item_for_router(item) for item in candidates.saved_items],
                    candidate_project_records=[self._record_for_router(record) for record in candidates.project_records],
                    reference_date=self.reference_date.isoformat(),
                    timezone=self.timezone,
                    pending_clarification=None if pending is None else pending.original_action,
                    context_truncated=context_window.truncated,
                )
            )
            self._validate_decision(decision, user_message, context_window, candidates)
        except (InvalidStructuredOutput, ModelError, ValidationError, ValueError):
            return self._reply(
                user_message,
                "I could not safely validate this conversation decision, so no local record was changed.",
                None,
                context_window,
                candidates,
                (),
                model_calls=1,
                model_seconds=perf_counter() - started,
                outcome="model_or_validation_failure",
            )

        model_seconds = perf_counter() - started
        local_started = perf_counter()
        try:
            if decision.intent == "calendar_create":
                if self._date_needs_clarification(decision.action):
                    assistant = self._save_calendar_date_clarification(user_message, decision.action)
                    return self._turn(
                        user_message,
                        assistant,
                        decision,
                        context_window,
                        candidates,
                        (),
                        model_calls=1,
                        model_seconds=model_seconds,
                        local_seconds=perf_counter() - local_started,
                        outcome="clarification_created",
                    )
                operations, reply = self._create_event(user_message, decision.action)
                if pending is not None:
                    self.store.resolve_pending_clarification(pending.id, "resolved")
            elif decision.intent == "calendar_update":
                if self._date_needs_clarification(decision.action):
                    assistant = self._save_calendar_date_clarification(user_message, decision.action)
                    return self._turn(
                        user_message,
                        assistant,
                        decision,
                        context_window,
                        candidates,
                        (),
                        model_calls=1,
                        model_seconds=model_seconds,
                        local_seconds=perf_counter() - local_started,
                        outcome="clarification_created",
                    )
                operations, reply = self._update_event(user_message, decision.action)
                if pending is not None:
                    self.store.resolve_pending_clarification(pending.id, "resolved")
            elif decision.intent in {"save_idea", "save_note"}:
                operations, reply = self._save_item(user_message, decision.action)
            elif decision.intent == "tracker_change":
                operations, reply = self._apply_tracker_change(user_message, decision, candidates)
            elif decision.intent == "local_answer":
                local_turn = LocalContextService(
                    self.store, self.reference_date, self.timezone, model=None
                ).submit(conversation_id, content, user_message.id)
                return self._turn(
                    user_message,
                    local_turn.assistant_message,
                    decision,
                    context_window,
                    candidates,
                    (),
                    model_calls=1,
                    model_seconds=model_seconds,
                    local_seconds=perf_counter() - local_started,
                    outcome=local_turn.outcome,
                )
            elif decision.intent == "project_answer":
                operations, reply = (), decision.reply or "I do not have enough stored project context to answer that."
            elif decision.intent == "clarify":
                if pending is not None:
                    operations, reply = (), decision.clarification_question or "Which local event do you mean?"
                else:
                    assistant = self._save_clarification(user_message, decision)
                    return self._turn(
                        user_message,
                        assistant,
                        decision,
                        context_window,
                        candidates,
                        (),
                        model_calls=1,
                        model_seconds=model_seconds,
                        local_seconds=perf_counter() - local_started,
                        outcome="clarification_created",
                    )
            else:
                operations, reply = (), decision.reply or "I cannot take a local action for that."
        except (KeyError, ValidationError, ValueError) as error:
            return self._reply(
                user_message,
                f"I could not safely apply that local change: {error}",
                decision,
                context_window,
                candidates,
                (),
                model_calls=1,
                model_seconds=model_seconds,
                local_seconds=perf_counter() - local_started,
                outcome="write_or_dispatch_failure",
            )

        outcome = decision.intent if decision.intent != "reply_only" or operations else "reply_only"
        return self._reply(
            user_message,
            reply,
            decision,
            context_window,
            candidates,
            operations,
            model_calls=1,
            model_seconds=model_seconds,
            local_seconds=perf_counter() - local_started,
            outcome=outcome,
        )

    def _create_event(
        self, user_message: Message, action: ContextualAction | None
    ) -> tuple[tuple[ToolOperation, ...], str]:
        assert action is not None
        source = self._current_evidence(action, user_message)
        resolved_date = self._resolved_date(action.date_expression)
        start_at: date | datetime = resolved_date
        end_at: datetime | None = None
        all_day = True
        if action.time_expression is not None:
            start_time = _parse_time(action.time_expression)
            if start_time is None:
                raise ValueError("event start time is not valid")
            start_at = datetime.combine(resolved_date, start_time, tzinfo=ZoneInfo(self.timezone))
            all_day = False
            if action.end_time_expression is not None:
                end_time = _parse_time(action.end_time_expression)
                if end_time is None:
                    raise ValueError("event end time is not valid")
                end_at = datetime.combine(resolved_date, end_time, tzinfo=ZoneInfo(self.timezone))
                if end_at <= start_at:
                    raise ValueError("event end time must be after its start time")
        event = CalendarEvent(
            id=_stable_id(user_message.id, "contextual-event"),
            conversation_id=user_message.conversation_id,
            title=action.content or "",
            description=action.description,
            source_message_id=source.message_id,
            source_quote=source.source_quote,
            start_at=start_at,
            end_at=end_at,
            all_day=all_day,
            timezone=self.timezone,
            created_at=user_message.created_at,
            updated_at=user_message.created_at,
        )
        operation = ToolOperation(
            id=_stable_id(user_message.id, "contextual-create-event-operation"),
            source_message_id=user_message.id,
            action_index=0,
            tool_name="create_local_event",
            request={
                "title": event.title,
                "description": event.description,
                "source_quote": event.source_quote,
                "evidence": [reference.model_dump(mode="json") for reference in action.evidence],
                "start_at": event.start_at.isoformat(),
                "end_at": None if event.end_at is None else event.end_at.isoformat(),
                "all_day": event.all_day,
                "timezone": event.timezone,
            },
            status="prepared",
            created_at=user_message.created_at,
            updated_at=user_message.created_at,
        )
        saved = self.store.save_tool_operation(operation)
        committed, saved_event = self.store.commit_prepared_calendar_event(
            saved.id, event, action.evidence
        )
        return (committed,), f"Created local calendar event: {saved_event.title}."

    def _update_event(
        self, user_message: Message, action: ContextualAction | None
    ) -> tuple[tuple[CalendarEventUpdateOperation, ...], str]:
        assert action is not None and action.target_event_id is not None
        self._current_evidence(action, user_message)
        existing = self.store.get_calendar_event(action.target_event_id)
        target_date = self._event_date(existing)
        if action.date_expression is not None:
            target_date = self._resolved_date(action.date_expression)
        after = self._updated_event(existing, action, target_date)
        operation = CalendarEventUpdateOperation(
            id=_stable_id(user_message.id, f"contextual-update-event:{existing.id}"),
            source_message_id=user_message.id,
            action_index=0,
            event_id=existing.id,
            workspace_id=existing.workspace_id,
            before=existing,
            after=after,
            status="prepared",
            created_at=user_message.created_at,
            updated_at=user_message.created_at,
        )
        saved = self.store.prepare_calendar_event_update(operation, action.evidence)
        committed, event = self.store.commit_prepared_calendar_event_update(saved.id)
        return (committed,), f"Updated local calendar event: {event.title}."

    def _date_needs_clarification(self, action: ContextualAction | None) -> bool:
        if action is None or action.date_expression is None:
            return False
        resolution = resolve_relative_date(action.date_expression, self.reference_date, self.timezone)
        return resolution.status != "resolved"

    def _save_item(
        self, user_message: Message, action: ContextualAction | None
    ) -> tuple[tuple[ToolOperation, ...], str]:
        assert action is not None
        source = self._current_evidence(action, user_message)
        kind = "idea" if action.action == "save_idea" else "note"
        item = SavedItem(
            id=_stable_id(user_message.id, f"contextual-{kind}"),
            conversation_id=user_message.conversation_id,
            kind=kind,
            content=action.content or "",
            source_message_id=source.message_id,
            source_quote=source.source_quote,
            created_at=user_message.created_at,
            updated_at=user_message.created_at,
        )
        operation = ToolOperation(
            id=_stable_id(user_message.id, f"contextual-{kind}-operation"),
            source_message_id=user_message.id,
            action_index=0,
            tool_name="save_idea" if kind == "idea" else "save_note",
            request={"content": item.content, "source_quote": item.source_quote},
            status="prepared",
            created_at=user_message.created_at,
            updated_at=user_message.created_at,
        )
        saved = self.store.save_tool_operation(operation)
        committed, saved_item = self.store.commit_prepared_saved_item(saved.id, item)
        return (committed,), f"Saved {kind}: {saved_item.content}"

    def _apply_tracker_change(
        self,
        user_message: Message,
        decision: ConversationDecision,
        candidates: RouterCandidates,
    ) -> tuple[tuple[ProjectChangeOperation, ...], str]:
        project = self.store.get_conversation_project(user_message.conversation_id)
        if project is None:
            raise ValueError("assign a project before changing its tracker")
        committed, reviewed = ProjectTrackerService.apply_validated_changes(
            self.store,
            project,
            user_message,
            decision.project_changes,
            list(candidates.project_records),
        )
        operations = tuple([*committed, *reviewed])
        summaries = [
            "Tracker update applied."
            if operation.status == "committed"
            else "Tracker update needs review before it can be applied."
            if operation.status == "review"
            else "Tracker update was not applied."
            for operation in operations
        ]
        return operations, "\n".join(summaries) or "No tracker change was applied."

    def _save_clarification(self, user_message: Message, decision: ConversationDecision) -> Message:
        assert decision.pending_action is not None
        assistant = self._assistant_message(
            user_message, decision.clarification_question or "Which local event do you mean?"
        )
        clarification = PendingClarification(
            id=_stable_id(user_message.id, "contextual-clarification"),
            conversation_id=user_message.conversation_id,
            source_message_id=user_message.id,
            assistant_message_id=assistant.id,
            original_action=decision.pending_action.model_dump(mode="json"),
            created_at=user_message.created_at,
        )
        self.store.save_pending_clarification(clarification)
        return assistant

    def _save_calendar_date_clarification(
        self, user_message: Message, action: ContextualAction | None
    ) -> Message:
        assert action is not None
        assistant = self._assistant_message(
            user_message, "Which exact date should I use for this local event?"
        )
        self.store.save_pending_clarification(
            PendingClarification(
                id=_stable_id(user_message.id, "contextual-calendar-date-clarification"),
                conversation_id=user_message.conversation_id,
                source_message_id=user_message.id,
                assistant_message_id=assistant.id,
                original_action=action.model_dump(mode="json"),
                created_at=user_message.created_at,
            )
        )
        return assistant

    def _save_answer_sources(
        self, assistant_message_id: str, decision: ConversationDecision, candidates: RouterCandidates
    ) -> None:
        if not decision.source_ids:
            return
        event_ids = {event.id for event in candidates.events}
        item_ids = {item.id for item in candidates.saved_items}
        local_sources = [
            LocalContextSourceRef(
                source_type="calendar_event" if source_id in event_ids else "saved_item",
                source_id=source_id,
            )
            for source_id in decision.source_ids
            if source_id in event_ids | item_ids
        ]
        if local_sources:
            self.store.save_local_context_answer_sources(assistant_message_id, local_sources)
        record_ids = {record.id for record in candidates.project_records}
        selected_records = [source_id for source_id in decision.source_ids if source_id in record_ids]
        if selected_records:
            self.store.save_project_answer_sources(assistant_message_id, selected_records)

    def _resume_persisted_operations(self, user_message: Message) -> tuple[RouterOperation, ...]:
        operations: list[RouterOperation] = []
        for operation in self.store.list_tool_operations(user_message.id):
            if operation.status == "prepared":
                operations.append(self._resume_tool_operation(user_message, operation))
                continue
            operations.append(operation)
        for operation in self.store.list_calendar_event_update_operations(user_message.id):
            if operation.status == "prepared":
                committed, _ = self.store.commit_prepared_calendar_event_update(operation.id)
                operations.append(committed)
            else:
                operations.append(operation)
        for operation in self.store.list_project_change_operations(user_message.id):
            if operation.status == "prepared":
                operations.append(self.store.commit_prepared_project_change(operation.id))
            else:
                operations.append(operation)
        return tuple(operations)

    def _resume_tool_operation(self, user_message: Message, operation: ToolOperation) -> ToolOperation:
        request = operation.request
        if operation.tool_name == "create_local_event":
            start_at = _parse_start_at(request["start_at"])
            event = CalendarEvent(
                id=_stable_id(user_message.id, "contextual-event"),
                conversation_id=user_message.conversation_id,
                title=request["title"],
                description=request.get("description"),
                source_message_id=user_message.id,
                source_quote=request["source_quote"],
                start_at=start_at,
                end_at=(None if request.get("end_at") is None else datetime.fromisoformat(request["end_at"])),
                all_day=request["all_day"],
                timezone=request["timezone"],
                created_at=user_message.created_at,
                updated_at=user_message.created_at,
            )
            evidence = [
                EvidenceReference.model_validate(reference)
                for reference in request.get(
                    "evidence",
                    [{"message_id": user_message.id, "source_quote": request["source_quote"]}],
                )
            ]
            committed, _ = self.store.commit_prepared_calendar_event(operation.id, event, evidence)
            return committed
        kind = "idea" if operation.tool_name == "save_idea" else "note"
        item = SavedItem(
            id=_stable_id(user_message.id, f"contextual-{kind}"),
            conversation_id=user_message.conversation_id,
            kind=kind,
            content=request["content"],
            source_message_id=user_message.id,
            source_quote=request["source_quote"],
            created_at=user_message.created_at,
            updated_at=user_message.created_at,
        )
        committed, _ = self.store.commit_prepared_saved_item(operation.id, item)
        return committed

    def _persisted_operations(self, message_id: str) -> tuple[RouterOperation, ...]:
        return tuple(
            [
                *self.store.list_tool_operations(message_id),
                *self.store.list_calendar_event_update_operations(message_id),
                *self.store.list_project_change_operations(message_id),
            ]
        )

    def _updated_event(
        self, existing: CalendarEvent, action: ContextualAction, target_date: date
    ) -> CalendarEvent:
        start_at = existing.start_at
        end_at = existing.end_at
        all_day = existing.all_day
        if action.time_expression is not None:
            start_time = _parse_time(action.time_expression)
            if start_time is None:
                raise ValueError("event start time is not valid")
            start_at = datetime.combine(target_date, start_time, tzinfo=ZoneInfo(existing.timezone))
            all_day = False
            end_at = None
            if action.end_time_expression is not None:
                end_time = _parse_time(action.end_time_expression)
                if end_time is None:
                    raise ValueError("event end time is not valid")
                end_at = datetime.combine(target_date, end_time, tzinfo=ZoneInfo(existing.timezone))
                if end_at <= start_at:
                    raise ValueError("event end time must be after its start time")
        elif action.date_expression is not None:
            if isinstance(existing.start_at, datetime):
                start_at = datetime.combine(
                    target_date, existing.start_at.timetz(), tzinfo=existing.start_at.tzinfo
                )
                if existing.end_at is not None:
                    end_at = start_at + (existing.end_at - existing.start_at)
            else:
                start_at = target_date
        return existing.model_copy(
            update={
                "title": action.content if action.content is not None else existing.title,
                "description": action.description if action.description is not None else existing.description,
                "start_at": start_at,
                "end_at": end_at,
                "all_day": all_day,
                "updated_at": utc_now(),
            }
        )

    def _resolved_date(self, expression: str | None) -> date:
        resolution = resolve_relative_date(expression or "", self.reference_date, self.timezone)
        if resolution.status != "resolved" or resolution.resolved_date is None:
            raise ValueError("calendar date needs clarification")
        return resolution.resolved_date

    @staticmethod
    def _event_date(event: CalendarEvent) -> date:
        return event.start_at.date() if isinstance(event.start_at, datetime) else event.start_at

    @staticmethod
    def _current_evidence(action: ContextualAction, user_message: Message) -> EvidenceReference:
        for evidence in action.evidence:
            if evidence.message_id == user_message.id:
                return evidence
        raise ValueError("a local write must cite the current user message")

    @staticmethod
    def _validate_decision(
        decision: ConversationDecision,
        user_message: Message,
        context_window: ProjectContextWindow,
        candidates: RouterCandidates,
    ) -> None:
        messages = {message.id: message for message in context_window.messages}
        event_ids = {event.id for event in candidates.events}
        item_ids = {item.id for item in candidates.saved_items}
        record_ids = {record.id for record in candidates.project_records}
        for action in (decision.action, decision.pending_action):
            if action is None:
                continue
            if action.target_event_id is not None and action.target_event_id not in event_ids:
                raise ValueError("calendar update target is outside supplied candidates")
            for evidence in action.evidence:
                message = messages.get(evidence.message_id)
                if (
                    message is None
                    or message.role != "user"
                    or _normalized(evidence.source_quote) not in _normalized(message.content)
                ):
                    raise ValueError("action evidence is outside bounded message context")
        if decision.intent in {"calendar_create", "save_idea", "save_note", "calendar_update"}:
            assert decision.action is not None
            ConversationRouterService._current_evidence(decision.action, user_message)
        if any(
            change.target_record_id is not None and change.target_record_id not in record_ids
            for change in decision.project_changes
        ):
            raise ValueError("project change target is outside supplied candidates")
        if any(source_id not in event_ids | item_ids | record_ids for source_id in decision.source_ids):
            raise ValueError("answer source is outside supplied candidates")

    def _reply(
        self,
        user_message: Message,
        content: str,
        decision: ConversationDecision | None,
        context_window: ProjectContextWindow,
        candidates: RouterCandidates,
        operations: tuple[RouterOperation, ...],
        *,
        model_calls: int = 0,
        model_seconds: float = 0,
        local_seconds: float = 0,
        outcome: str,
    ) -> ConversationRouterTurn:
        assistant = self._assistant_message(user_message, content)
        if decision is not None and decision.intent == "project_answer":
            self._save_answer_sources(assistant.id, decision, candidates)
        return self._turn(
            user_message,
            assistant,
            decision,
            context_window,
            candidates,
            operations,
            model_calls=model_calls,
            model_seconds=model_seconds,
            local_seconds=local_seconds,
            outcome=outcome,
        )

    @staticmethod
    def _turn(
        user_message: Message,
        assistant_message: Message,
        decision: ConversationDecision | None,
        context_window: ProjectContextWindow,
        candidates: RouterCandidates,
        operations: tuple[RouterOperation, ...],
        *,
        model_calls: int = 0,
        model_seconds: float = 0,
        local_seconds: float = 0,
        outcome: str,
    ) -> ConversationRouterTurn:
        return ConversationRouterTurn(
            user_message,
            assistant_message,
            decision,
            context_window,
            candidates,
            operations,
            model_calls,
            model_seconds,
            local_seconds,
            outcome,
        )

    def _assistant_message(self, user_message: Message, content: str) -> Message:
        return self.store.save_message(
            Message(
                id=_stable_id(user_message.id, "contextual-assistant"),
                conversation_id=user_message.conversation_id,
                role="assistant",
                content=content,
                reply_to_message_id=user_message.id,
                created_at=utc_now(),
            )
        )

    def _user_message(self, conversation_id: str, content: str, message_id: str | None) -> Message:
        if message_id is not None:
            try:
                existing = self.store.get_message(message_id)
            except KeyError:
                pass
            else:
                if (
                    existing.role != "user"
                    or existing.conversation_id != conversation_id
                    or existing.content != content
                ):
                    raise ValueError("message ID already exists with different content")
                return existing
        return self.store.save_message(
            Message(id=message_id or str(uuid4()), conversation_id=conversation_id, role="user", content=content)
        )

    @staticmethod
    def _message_for_router(message: Message) -> dict[str, str]:
        return {"message_id": message.id, "role": message.role, "content": message.content}

    @staticmethod
    def _event_for_router(event: CalendarEvent) -> dict:
        return {
            "id": event.id,
            "title": event.title,
            "description": event.description,
            "start_at": event.start_at.isoformat(),
            "end_at": None if event.end_at is None else event.end_at.isoformat(),
            "all_day": event.all_day,
            "timezone": event.timezone,
        }

    @staticmethod
    def _item_for_router(item: SavedItem) -> dict:
        return {"id": item.id, "kind": item.kind, "content": item.content}

    @staticmethod
    def _record_for_router(record: ProjectRecord) -> dict:
        return {"id": record.id, "kind": record.kind, "content": record.content, "state": record.state}


def _stable_id(message_id: str, suffix: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"captureflow:{message_id}:{suffix}"))


def _normalized(value: str) -> str:
    return " ".join(value.split())


def _parse_time(value: str) -> time | None:
    normalized = value.strip().casefold().replace("：", ":")
    if ":" in normalized:
        hour_text, minute_text = normalized.rsplit(":", 1)
        hour_text = hour_text.replace("下午", "").replace("上午", "").replace("am", "").replace("pm", "").strip()
        try:
            hour, minute = int(hour_text), int(minute_text)
        except ValueError:
            return None
        if "下午" in normalized or "pm" in normalized:
            hour = hour + 12 if hour < 12 else hour
        return time(hour, minute) if 0 <= hour <= 23 and 0 <= minute <= 59 else None
    words = {"noon": 12, "midnight": 0, "下午两点": 14, "下午三点": 15, "下午四点": 16, "下午五点": 17}
    hour = words.get(normalized)
    return time(hour) if hour is not None else None


def _parse_start_at(value: str) -> date | datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return date.fromisoformat(value)
