"""One bounded, contextual entry point for ordinary CaptureFlow chat."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from .context_service import ContextWindow, RouterCandidates, select_context_window, select_router_candidates
from .credentials import contains_credential_disclosure
from .date_tool import calendar_reference
from .llm import ContextualConversationModel, InvalidStructuredOutput, ModelError, describe_validation_error
from .local_context_service import LocalContextService
from .models import (
    CalendarEvent,
    CalendarEventUpdateOperation,
    ContextualAction,
    ConversationDecision,
    EvidenceReference,
    LocalContextSourceRef,
    Message,
    ProjectChange,
    ProjectChangeOperation,
    ProjectRecord,
    SavedItem,
    ToolOperation,
    utc_now,
)
from .project_answer_service import answer_project_question, build_project_snapshot
from .project_service import apply_validated_changes
from .storage import CaptureStore


RouterOperation = ToolOperation | CalendarEventUpdateOperation | ProjectChangeOperation

_SAVED_ITEM_WRITES = {"save_idea", "save_note"}
_CALENDAR_WRITES = {"calendar_create", "calendar_update", "calendar_delete"}
# A stored event keeps at most this many source messages.
MAX_EVENT_SOURCES = 3


@dataclass(frozen=True)
class ConversationRouterTurn:
    user_message: Message
    assistant_message: Message
    decision: ConversationDecision | None
    context_window: ContextWindow
    candidates: RouterCandidates
    operations: tuple[RouterOperation, ...]
    outcome: str
    model_calls: int = 0
    model_seconds: float = 0
    local_seconds: float = 0
    # Why a turn failed, as error type and rule; never the model's raw output.
    failure_reason: str | None = None


class ConversationRouterService:
    """Persist one normal-language turn, then execute exactly one validated decision."""

    def __init__(
        self,
        store: CaptureStore,
        model: ContextualConversationModel,
        reference_date: date,
        timezone: str = "America/Vancouver",
        reference_time: time | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.reference_date = reference_date
        self.timezone = timezone
        self.reference_time = reference_time

    def submit(
        self, conversation_id: str, content: str, message_id: str | None = None
    ) -> ConversationRouterTurn:
        user_message = self._user_message(conversation_id, content, message_id)
        context_window = select_context_window(self.store.list_messages(conversation_id))
        candidates = select_router_candidates(
            self.store, conversation_id, self.reference_date, user_message.content
        )
        if existing_reply := self.store.get_assistant_reply(user_message.id):
            return ConversationRouterTurn(
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

        started = perf_counter()
        try:
            decision = ConversationDecision.model_validate(
                self.model.propose_conversation_decision(
                    current_message=self._message_for_router(user_message),
                    recent_messages=context_window.for_router(),
                    **candidates.for_router(),
                    truncated_candidate_lists=list(candidates.truncated_kinds),
                    reference_date=self.reference_date.isoformat(),
                    reference_time=(
                        None if self.reference_time is None else self.reference_time.strftime("%H:%M")
                    ),
                    timezone=self.timezone,
                    context_truncated=context_window.truncated,
                    calendar=calendar_reference(self.reference_date),
                )
            )
            self._validate_decision(decision, candidates)
        except (InvalidStructuredOutput, ModelError, ValidationError, ValueError) as error:
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
                failure_reason=_failure_reason(error),
            )

        model_seconds = perf_counter() - started
        local_started = perf_counter()

        def turn_without_writes(assistant: Message, outcome: str) -> ConversationRouterTurn:
            return ConversationRouterTurn(
                user_message,
                assistant,
                decision,
                context_window,
                candidates,
                (),
                outcome=outcome,
                model_calls=1,
                model_seconds=model_seconds,
                local_seconds=perf_counter() - local_started,
            )

        try:
            if decision.intent in _CALENDAR_WRITES and (
                missing_field := self._calendar_clarification_field(decision.action)
            ):
                # Python's policy question is a plain reply; nothing waits for the answer.
                assistant = self._assistant_message(user_message, self._question(missing_field))
                return turn_without_writes(assistant, "clarification_created")
            if decision.intent == "clarify":
                assistant = self._assistant_message(user_message, decision.reply or "")
                return turn_without_writes(assistant, "clarification_created")
            if decision.intent == "local_answer":
                local_turn = LocalContextService(self.store, self.reference_date).submit(
                    conversation_id,
                    content,
                    user_message.id,
                    selected=self._named_candidates(decision, candidates),
                )
                return turn_without_writes(local_turn.assistant_message, local_turn.outcome)
            if decision.intent == "project_answer":
                return self._answer_project(user_message, decision, context_window, candidates, model_seconds)
            operations, reply = self._execute(decision, user_message, context_window, candidates)
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
                failure_reason=_failure_reason(error),
            )

        outcome = decision.intent if decision.intent != "reply_only" or operations else "reply_only"
        if decision.intent == "calendar_update" and not operations:
            outcome = "calendar_unchanged"
        if decision.intent == "calendar_delete":
            outcome = "calendar_delete_review"
        # A second item is asked about only once this turn has saved its own.
        if operations and decision.follow_up_question:
            reply = f"{reply}\n\n{decision.follow_up_question}"
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

    @staticmethod
    def _named_candidates(
        decision: ConversationDecision, candidates: RouterCandidates
    ) -> list[CalendarEvent | SavedItem | ProjectRecord]:
        """The stored items a local question is about, in the order the model named them."""

        by_id = {
            item.id: item
            for item in (*candidates.events, *candidates.saved_items, *candidates.project_records)
        }
        return [by_id[source_id] for source_id in decision.source_ids if source_id in by_id]

    def _answer_project(
        self,
        user_message: Message,
        decision: ConversationDecision,
        context_window: ContextWindow,
        candidates: RouterCandidates,
        routing_seconds: float,
    ) -> ConversationRouterTurn:
        """Answer a question about the whole project with a second call over its snapshot."""

        project = self.store.get_conversation_project(user_message.conversation_id)
        if project is None:
            return self._reply(
                user_message,
                "This conversation is not assigned to a project, so I cannot answer a question about project progress.",
                decision,
                context_window,
                candidates,
                (),
                model_calls=1,
                model_seconds=routing_seconds,
                outcome="project_answer_unavailable",
            )
        snapshot_started = perf_counter()
        snapshot = build_project_snapshot(self.store, project, self.reference_date)
        snapshot_seconds = perf_counter() - snapshot_started
        answer_started = perf_counter()
        try:
            result = answer_project_question(
                self.model, snapshot, user_message, context_window.for_router(), self.reference_date
            )
        except (InvalidStructuredOutput, ModelError, ValidationError, ValueError) as error:
            return self._reply(
                user_message,
                "I could not answer from the stored project records, so nothing was changed.",
                decision,
                context_window,
                candidates,
                (),
                model_calls=2,
                model_seconds=routing_seconds + perf_counter() - answer_started,
                local_seconds=snapshot_seconds,
                outcome="model_or_validation_failure",
                failure_reason=_failure_reason(error),
            )
        answer_seconds = perf_counter() - answer_started
        assistant = self._assistant_message(user_message, result.reply)
        if result.records:
            self.store.save_project_answer_sources(assistant.id, [record.id for record in result.records])
        if result.events:
            self.store.save_local_context_answer_sources(
                assistant.id,
                [LocalContextSourceRef(source_type="calendar_event", source_id=event.id) for event in result.events],
            )
        return ConversationRouterTurn(
            user_message,
            assistant,
            decision,
            context_window,
            candidates,
            (),
            outcome="project_answer",
            model_calls=2,
            model_seconds=routing_seconds + answer_seconds,
            local_seconds=snapshot_seconds,
        )

    def _execute(
        self,
        decision: ConversationDecision,
        user_message: Message,
        context_window: ContextWindow,
        candidates: RouterCandidates,
    ) -> tuple[tuple[RouterOperation, ...], str]:
        """Commit the decision's one local write, or return the reply for a turn with none."""

        if decision.intent == "calendar_create":
            return self._create_event(user_message, decision.action, context_window)
        if decision.intent == "calendar_update":
            return self._update_event(user_message, decision.action, context_window)
        if decision.intent == "calendar_delete":
            return self._delete_event(user_message, decision.action, context_window)
        if decision.intent in _SAVED_ITEM_WRITES:
            return self._save_item(user_message, decision.action)
        if decision.intent == "tracker_change":
            return self._apply_tracker_change(user_message, decision, context_window, candidates)
        return (), decision.reply or "I cannot take a local action for that."

    def _create_event(
        self, user_message: Message, action: ContextualAction | None, context_window: ContextWindow
    ) -> tuple[tuple[ToolOperation, ...], str]:
        assert action is not None
        sources = self._event_sources(action, user_message, context_window)
        if action.date is None:
            raise ValueError("calendar date needs clarification")
        event_date = date.fromisoformat(action.date)
        start_at: date | datetime = event_date
        end_at: datetime | None = None
        all_day = action.start_time is None
        if action.start_time is not None:
            start_at = self._local_datetime(event_date, action.start_time)
            end_at = self._end_at(start_at, action)
        event = CalendarEvent(
            id=_stable_id(user_message.id, "contextual-event"),
            conversation_id=user_message.conversation_id,
            title=action.content or "",
            description=action.description,
            source_message_id=user_message.id,
            source_quote=user_message.content,
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
                "evidence": [reference.model_dump(mode="json") for reference in sources],
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
        committed, saved_event = self.store.commit_prepared_calendar_event(saved.id, event, sources)
        return (
            (committed,),
            f"Created local calendar event: {saved_event.title} — {_event_schedule(saved_event)}.",
        )

    def _update_event(
        self, user_message: Message, action: ContextualAction | None, context_window: ContextWindow
    ) -> tuple[tuple[CalendarEventUpdateOperation, ...], str]:
        assert action is not None and action.target_event_id is not None
        existing = self.store.get_calendar_event(action.target_event_id)
        target_date = existing.start_date
        if action.date is not None:
            target_date = date.fromisoformat(action.date)
        after = self._updated_event(existing, action, target_date)
        if after.model_copy(update={"updated_at": existing.updated_at}) == existing:
            # An update that changes nothing is not a change; do not audit it as one.
            return (), (
                f"Local calendar event {existing.title} is already {_event_schedule(existing)}; "
                "nothing was changed."
            )
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
        saved = self.store.prepare_calendar_event_update(
            operation, self._event_sources(action, user_message, context_window)
        )
        committed, event = self.store.commit_prepared_calendar_event_update(saved.id)
        return (committed,), f"Updated local calendar event: {event.title} — {_event_schedule(event)}."

    def _event_sources(
        self, action: ContextualAction, user_message: Message, context_window: ContextWindow
    ) -> list[EvidenceReference]:
        """The user messages an event comes from, each quoted whole, in conversation order.

        The model names them; Python keeps the ones that are user messages in the context
        and always includes the current message, which is what authorizes the write. An
        assistant or unknown ID is dropped rather than failing the turn.
        """

        named = {
            message_id
            for message_id in action.source_message_ids
            if message_id != user_message.id and context_window.user_message(message_id) is not None
        }
        earlier = [message for message in context_window.messages if message.id in named]
        kept = [*earlier[-(MAX_EVENT_SOURCES - 1) :], user_message]
        return [EvidenceReference(message_id=message.id, source_quote=message.content) for message in kept]

    def _delete_event(
        self, user_message: Message, action: ContextualAction | None, context_window: ContextWindow
    ) -> tuple[tuple[CalendarEventUpdateOperation, ...], str]:
        """Prepare a deletion that waits for the user to confirm it."""

        assert action is not None and action.target_event_id is not None
        existing = self.store.get_calendar_event(action.target_event_id)
        operation = CalendarEventUpdateOperation(
            id=_stable_id(user_message.id, f"contextual-delete-event:{existing.id}"),
            source_message_id=user_message.id,
            action_index=0,
            event_id=existing.id,
            workspace_id=existing.workspace_id,
            before=existing,
            after=existing.model_copy(update={"status": "deleted", "updated_at": utc_now()}),
            status="review",
            created_at=user_message.created_at,
            updated_at=user_message.created_at,
        )
        saved = self.store.prepare_calendar_event_update(
            operation, self._event_sources(action, user_message, context_window)
        )
        return (saved,), (
            f"Delete local calendar event {existing.title} — {_event_schedule(existing)}? "
            "Confirm or keep it below."
        )

    def _calendar_clarification_field(self, action: ContextualAction | None) -> str | None:
        """The field a calendar action must ask about before it may be written, if any."""

        if action is None:
            return None
        if action.timezone is not None and action.timezone != self.timezone:
            return "timezone"
        if action.recurring:
            return "recurrence"
        if action.action == "update_event" and not action.has_replacement():
            # The model found the event but not its new value, as in 改到周四 when two
            # Thursdays are plausible; ask instead of rejecting the whole decision.
            return "date"
        if action.action == "create_event" and action.date is None:
            return "date"
        if action.date is not None and not self._plausible_date(date.fromisoformat(action.date)):
            return "date"
        if self._end_and_duration_disagree(action):
            return "duration"
        if (
            action.action == "update_event"
            and action.start_time is None
            and (action.end_time is not None or action.duration_minutes is not None)
            and action.target_event_id is not None
            and not isinstance(self.store.get_calendar_event(action.target_event_id).start_at, datetime)
        ):
            # A new end or length for an all-day event still needs a start time.
            return "time"
        return None

    def _end_and_duration_disagree(self, action: ContextualAction) -> bool:
        """Whether a stated end time and a stated duration name different ends.

        Both may be stated for one meeting, as in 四点到五点，开一个小时; when they
        agree the end time is used. A start is needed to compare them: the action's
        own, or the start of the timed event an update targets.
        """

        if action.end_time is None or action.duration_minutes is None:
            return False
        start_clock = None if action.start_time is None else time.fromisoformat(action.start_time)
        if start_clock is None and action.action == "update_event" and action.target_event_id is not None:
            existing = self.store.get_calendar_event(action.target_event_id).start_at
            if isinstance(existing, datetime):
                start_clock = existing.time()
        if start_clock is None:
            return False
        start = datetime.combine(self.reference_date, start_clock)
        by_end = datetime.combine(self.reference_date, time.fromisoformat(action.end_time))
        if by_end <= start:
            by_end += timedelta(days=1)
        return by_end != start + timedelta(minutes=action.duration_minutes)

    def _plausible_date(self, value: date) -> bool:
        """A model-normalized date far outside the planning horizon is more likely a misread."""

        earliest = self.reference_date - timedelta(days=366)
        latest = self.reference_date + timedelta(days=731)
        return earliest <= value <= latest

    def _local_datetime(self, day: date, clock: str) -> datetime:
        """Attach the configured timezone; zoneinfo picks the offset that date actually has."""

        return datetime.combine(day, time.fromisoformat(clock), tzinfo=ZoneInfo(self.timezone))

    def _end_at(self, start_at: datetime, action: ContextualAction) -> datetime | None:
        if action.end_time is not None:
            end_at = datetime.combine(
                start_at.date(), time.fromisoformat(action.end_time), tzinfo=start_at.tzinfo
            )
            return end_at + timedelta(days=1) if end_at <= start_at else end_at
        if action.duration_minutes is not None:
            return start_at + timedelta(minutes=action.duration_minutes)
        return None

    def _save_item(
        self, user_message: Message, action: ContextualAction | None
    ) -> tuple[tuple[ToolOperation, ...], str]:
        assert action is not None
        kind = "idea" if action.action == "save_idea" else "note"
        item = SavedItem(
            id=_stable_id(user_message.id, f"contextual-{kind}"),
            conversation_id=user_message.conversation_id,
            kind=kind,
            content=action.content or "",
            source_message_id=user_message.id,
            source_quote=user_message.content,
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
        context_window: ContextWindow,
        candidates: RouterCandidates,
    ) -> tuple[tuple[ProjectChangeOperation, ...], str]:
        project = self.store.get_conversation_project(user_message.conversation_id)
        if project is None:
            raise ValueError("assign a project before changing its tracker")
        resolved = [self._resolved_deadline(change) for change in decision.project_changes]
        deadline_notes = [note for _, note in resolved if note is not None]
        changes = [change for change, _ in resolved if change is not None]
        if not changes:
            return (), "\n".join(deadline_notes) or "No tracker change was applied."
        committed, reviewed = apply_validated_changes(
            self.store,
            project,
            user_message,
            changes,
            list(candidates.project_records),
            context_window.messages,
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
        summaries.extend(deadline_notes)
        summaries.extend(self._deadline_summaries(committed))
        return operations, "\n".join(summaries) or "No tracker change was applied."

    def _resolved_deadline(self, change: ProjectChange) -> tuple[ProjectChange | None, str | None]:
        """Attach the configured timezone to a model-normalized deadline.

        Returns the change to apply and the sentence explaining a deadline that was not
        kept. A task is still created without its deadline, and the user can set it
        afterwards with an ordinary follow-up message.
        """

        if change.recurring:
            return self._deadline_not_set(
                change, "I cannot store a repeating deadline, only a single dated one."
            )
        if change.due_date is None:
            return change, None
        due_date = date.fromisoformat(change.due_date)
        if not self._plausible_date(due_date):
            return self._deadline_not_set(
                change, f"The deadline date {change.due_date} is outside the planning horizon."
            )
        due: date | datetime = due_date
        if change.due_time is not None:
            due = self._local_datetime(due_date, change.due_time)
        return change.model_copy(update={"due_at": due}), None

    @staticmethod
    def _deadline_not_set(change: ProjectChange, reason: str) -> tuple[ProjectChange | None, str]:
        """Keep a new task without its deadline; a deadline-only change has nothing left to do."""

        if change.action != "create_record":
            return None, f"{reason} No deadline was set."
        cleared = {"due_date": None, "due_time": None, "due_at": None, "recurring": False}
        return change.model_copy(update=cleared), (
            f"{reason} The task has no deadline yet; tell me an exact date or time and I will add it."
        )

    def _deadline_summaries(self, committed: list[ProjectChangeOperation]) -> list[str]:
        summaries = []
        for operation in committed:
            record_id = (operation.result or {}).get("record_id")
            if operation.status != "committed" or record_id is None:
                continue
            record = self.store.get_project_record(record_id)
            if record.due_at is not None:
                summaries.append(f"Deadline on the calendar: {record.due_at.isoformat()}.")
        return summaries

    def _question(self, missing_field: str) -> str:
        return {
            "date": "Which exact date should I use for this local event?",
            "time": "What exact start time should I use for this local event?",
            "duration": "How long should this local event last (for example, 30 minutes)?",
            "timezone": (
                f"I only store {self.timezone} times and cannot convert another timezone. "
                f"What exact {self.timezone} time should I use for this local event?"
            ),
            "recurrence": (
                "I cannot store a repeating schedule, only single dated records. "
                "Which one exact date should I use?"
            ),
        }[missing_field]

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
        self,
        existing: CalendarEvent,
        action: ContextualAction,
        target_date: date,
    ) -> CalendarEvent:
        start_at = existing.start_at
        end_at = existing.end_at
        all_day = existing.all_day
        if action.start_time is not None:
            start_at = datetime.combine(
                target_date, time.fromisoformat(action.start_time), tzinfo=ZoneInfo(existing.timezone)
            )
            all_day = False
            end_at = None
        elif action.date is not None:
            if isinstance(existing.start_at, datetime):
                start_at = datetime.combine(
                    target_date, existing.start_at.timetz(), tzinfo=existing.start_at.tzinfo
                )
                if existing.end_at is not None:
                    end_at = start_at + (existing.end_at - existing.start_at)
            else:
                start_at = target_date
        if action.end_time is not None or action.duration_minutes is not None:
            if not isinstance(start_at, datetime):
                raise ValueError("event end time requires a start time")
            end_at = self._end_at(start_at, action)
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

    @staticmethod
    def _validate_decision(decision: ConversationDecision, candidates: RouterCandidates) -> None:
        """Every ID a decision writes to or cites must be one the router supplied."""

        event_ids = {event.id for event in candidates.events}
        item_ids = {item.id for item in candidates.saved_items}
        record_ids = {record.id for record in candidates.project_records}
        if decision.action is not None and decision.action.target_event_id is not None:
            if decision.action.target_event_id not in event_ids:
                raise ValueError("calendar update target is outside supplied candidates")
        if any(change.due_at is not None for change in decision.project_changes):
            raise ValueError("a tracker deadline must be resolved locally, not supplied by the model")
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
        context_window: ContextWindow,
        candidates: RouterCandidates,
        operations: tuple[RouterOperation, ...],
        *,
        model_calls: int = 0,
        model_seconds: float = 0,
        local_seconds: float = 0,
        outcome: str,
        failure_reason: str | None = None,
    ) -> ConversationRouterTurn:
        assistant = self._assistant_message(user_message, content)
        return ConversationRouterTurn(
            user_message,
            assistant,
            decision,
            context_window,
            candidates,
            operations,
            outcome=outcome,
            model_calls=model_calls,
            model_seconds=model_seconds,
            local_seconds=local_seconds,
            failure_reason=failure_reason,
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


def _failure_reason(error: Exception) -> str:
    detail = describe_validation_error(error) if isinstance(error, ValidationError) else str(error)
    return f"{type(error).__name__}: {detail}"[:1_000]


def _stable_id(message_id: str, suffix: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"captureflow:{message_id}:{suffix}"))


def _parse_start_at(value: str) -> date | datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return date.fromisoformat(value)


def _event_schedule(event: CalendarEvent) -> str:
    timing = event.start_at.isoformat()
    if event.end_at is not None:
        timing = f"{timing} to {event.end_at.isoformat()}"
    return f"{timing} ({event.timezone})"
