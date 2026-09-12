"""One bounded, contextual entry point for ordinary CaptureFlow chat."""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from .context_service import ProjectContextWindow, RouterCandidates, select_project_context, select_router_candidates
from .date_tool import calendar_reference, resolve_relative_date, states_recurrence
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
    ProjectChange,
    ProjectChangeOperation,
    ProjectRecord,
    SavedItem,
    ToolOperation,
    utc_now,
)
from .project_service import ProjectTrackerService
from .storage import CaptureStore
from .time_tool import parse_duration, parse_end_time, parse_time, stated_timezone, with_stated_period


RouterOperation = ToolOperation | CalendarEventUpdateOperation | ProjectChangeOperation

# The committed intents that answer a pending question about each kind of action.
_CALENDAR_WRITES = {"calendar_create", "calendar_update", "calendar_delete"}
_SAVED_ITEM_WRITES = {"save_idea", "save_note"}
_ANSWERING_INTENTS = {
    "create_event": _CALENDAR_WRITES,
    "update_event": _CALENDAR_WRITES,
    "delete_event": _CALENDAR_WRITES,
    "save_idea": _SAVED_ITEM_WRITES,
    "save_note": _SAVED_ITEM_WRITES,
}

# Words that point at "a meeting" without saying which one.
_GENERIC_EVENT_TERMS = {
    "会议", "例会", "开会", "这个", "那个", "这场", "那场", "这次", "那次", "日程", "安排", "活动",
    "meeting", "meetings", "event", "call", "sync", "this", "that", "the", "it",
}


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
        candidates = select_router_candidates(
            self.store, conversation_id, self.reference_date, user_message.content
        )
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
                    calendar=calendar_reference(self.reference_date),
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
        if decision.action is not None and decision.intent in {"calendar_create", "calendar_update"}:
            decision = decision.model_copy(
                update={"action": self._with_stated_periods(decision.action, user_message)}
            )
        try:
            if decision.intent == "calendar_create":
                if field := self._calendar_clarification_field(decision.action, user_message):
                    assistant = self._save_calendar_clarification(user_message, decision.action, field)
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
            elif decision.intent == "calendar_update":
                if field := self._calendar_clarification_field(
                    decision.action, user_message, context_window
                ):
                    assistant = self._save_calendar_clarification(user_message, decision.action, field)
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
            elif decision.intent == "calendar_delete":
                if field := self._calendar_clarification_field(
                    decision.action, user_message, context_window
                ):
                    assistant = self._save_calendar_clarification(user_message, decision.action, field)
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
                operations, reply = self._delete_event(user_message, decision.action)
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
        if decision.intent == "calendar_update" and not operations:
            outcome = "calendar_unchanged"
        if pending is not None and operations and decision.intent in _ANSWERING_INTENTS.get(
            pending.original_action.get("action"), set()
        ):
            # Only a write of the pending kind answers it; an unrelated task must not
            # silently discard a calendar event that is still waiting for its time.
            self.store.resolve_pending_clarification(pending.id, "resolved")
        deferred = self._deferred_second_item(decision, operations)
        if deferred is not None:
            reply = f"{reply}\n\n{decision.follow_up_question}"
        turn = self._reply(
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
        if deferred is not None:
            self._save_follow_up_clarification(user_message, turn.assistant_message, deferred)
        return turn

    @staticmethod
    def _deferred_second_item(
        decision: ConversationDecision, operations: tuple[RouterOperation, ...]
    ) -> ContextualAction | None:
        """Return the second item to ask about, once this turn has committed its one action."""

        if decision.deferred_action is None or not operations:
            return None
        return decision.deferred_action

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
            start_time = parse_time(action.time_expression)
            if start_time is None:
                raise ValueError("event start time is not valid")
            start_at = datetime.combine(resolved_date, start_time, tzinfo=ZoneInfo(self.timezone))
            all_day = False
            if action.end_time_expression is not None:
                end_time = parse_end_time(action.end_time_expression, action.time_expression)
                if end_time is None:
                    raise ValueError("event end time is not valid")
                end_at = datetime.combine(resolved_date, end_time, tzinfo=ZoneInfo(self.timezone))
                if end_at <= start_at:
                    end_at += timedelta(days=1)
            elif action.duration_expression is not None:
                duration = parse_duration(action.duration_expression)
                if duration is None:
                    raise ValueError("event duration is not valid")
                end_at = start_at + duration
        event = CalendarEvent(
            id=_stable_id(user_message.id, "contextual-event"),
            conversation_id=user_message.conversation_id,
            title=action.content or "",
            description=self._quoted_description(action, user_message),
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
        return (
            (committed,),
            f"Created local calendar event: {saved_event.title} — {_event_schedule(saved_event)}.",
        )

    def _update_event(
        self, user_message: Message, action: ContextualAction | None
    ) -> tuple[tuple[CalendarEventUpdateOperation, ...], str]:
        assert action is not None and action.target_event_id is not None
        self._current_evidence(action, user_message)
        existing = self.store.get_calendar_event(action.target_event_id)
        target_date = self._event_date(existing)
        if action.date_expression is not None:
            target_date = self._resolved_date(action.date_expression)
        after = self._updated_event(
            existing, action, target_date, self._quoted_description(action, user_message)
        )
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
        saved = self.store.prepare_calendar_event_update(operation, action.evidence)
        committed, event = self.store.commit_prepared_calendar_event_update(saved.id)
        return (committed,), f"Updated local calendar event: {event.title}."

    def _quoted_description(
        self, action: ContextualAction, user_message: Message
    ) -> str | None:
        """Keep a description only while it is the user's own words.

        `source_quote` is validated everywhere else; `description` was the one stored
        Calendar field a model could write freely, so an unquoted one is dropped rather
        than kept as if the user had said it. A clarification answer may still describe
        the event from the earlier message it cites, so every cited message counts.
        """

        if action.description is None:
            return None
        cited = [user_message.content] + [
            self.store.get_message(evidence.message_id).content
            for evidence in action.evidence
            if evidence.message_id != user_message.id
        ]
        normalized = _normalized(action.description)
        if any(normalized in _normalized(source) for source in cited):
            return action.description
        return None

    def _delete_event(
        self, user_message: Message, action: ContextualAction | None
    ) -> tuple[tuple[CalendarEventUpdateOperation, ...], str]:
        assert action is not None and action.target_event_id is not None
        self._current_evidence(action, user_message)
        existing = self.store.get_calendar_event(action.target_event_id)
        operation = CalendarEventUpdateOperation(
            id=_stable_id(user_message.id, f"contextual-delete-event:{existing.id}"),
            source_message_id=user_message.id,
            action_index=0,
            event_id=existing.id,
            workspace_id=existing.workspace_id,
            before=existing,
            after=existing.model_copy(update={"status": "deleted", "updated_at": utc_now()}),
            status="prepared",
            created_at=user_message.created_at,
            updated_at=user_message.created_at,
        )
        saved = self.store.prepare_calendar_event_update(operation, action.evidence)
        committed, event = self.store.commit_prepared_calendar_event_update(saved.id)
        return (committed,), f"Deleted local calendar event: {event.title}. It can be undone."

    def _calendar_clarification_field(
        self,
        action: ContextualAction | None,
        user_message: Message | None = None,
        context_window: ProjectContextWindow | None = None,
    ) -> str | None:
        if action is None:
            return None
        if (
            user_message is not None
            and context_window is not None
            and self._target_is_ungrounded(action, user_message, context_window)
        ):
            return "target"
        if user_message is not None and self._states_other_timezone(user_message.content):
            return "timezone"
        if action.date_expression is not None and states_recurrence(action.date_expression):
            return "recurrence"
        if action.date_expression is not None:
            resolution = resolve_relative_date(
                action.date_expression, self.reference_date, self.timezone
            )
            if resolution.status != "resolved":
                return "date"
        if action.time_expression is not None and parse_time(action.time_expression) is None:
            return "time"
        if action.end_time_expression is not None and parse_end_time(action.end_time_expression, action.time_expression) is None:
            return "time"
        if action.action == "create_event" and action.end_time_expression is not None and action.time_expression is None:
            return "time"
        if action.duration_expression is not None:
            if parse_duration(action.duration_expression) is None:
                return "duration"
            if action.time_expression is None:
                if action.action == "create_event":
                    return "time"
                assert action.target_event_id is not None
                if not isinstance(self.store.get_calendar_event(action.target_event_id).start_at, datetime):
                    return "time"
        return None

    @staticmethod
    def _with_stated_periods(action: ContextualAction, user_message: Message) -> ContextualAction:
        return action.model_copy(
            update={
                "time_expression": with_stated_period(action.time_expression, user_message.content),
                "end_time_expression": with_stated_period(
                    action.end_time_expression, user_message.content
                ),
            }
        )

    def _target_is_ungrounded(
        self, action: ContextualAction, user_message: Message, context_window: ProjectContextWindow
    ) -> bool:
        """Whether an update or delete binds to an event this conversation never identified.

        The model still chooses the target. This only refuses to write when nothing in the
        current conversation supports that choice: the event was not created or changed by
        a message in the bounded window, and the current message does not name it. A
        demonstrative such as `这个会议` in a new conversation must not silently bind to
        whichever event happens to exist.
        """

        if action.action not in {"update_event", "delete_event"} or action.target_event_id is None:
            return False
        window_ids = {message.id for message in context_window.messages} - {user_message.id}
        if window_ids & self.store.list_calendar_event_message_ids(action.target_event_id):
            return False
        title = self.store.get_calendar_event(action.target_event_id).title
        return not (_distinctive_terms(user_message.content) & _distinctive_terms(title))

    def _states_other_timezone(self, content: str) -> str | None:
        """Notice a message about a timezone this workspace does not store."""

        stated = stated_timezone(content)
        if stated is None or stated[1] == self.timezone:
            return None
        return stated[0]

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
        resolved = [self._resolved_deadline(change) for change in decision.project_changes]
        deadline_notes = [note for _, note in resolved if note is not None]
        changes = [change for change, _ in resolved if change is not None]
        if not changes:
            return (), "\n".join(deadline_notes) or "No tracker change was applied."
        committed, reviewed = ProjectTrackerService.apply_validated_changes(
            self.store,
            project,
            user_message,
            changes,
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
        summaries.extend(deadline_notes)
        summaries.extend(self._deadline_summaries(committed))
        return operations, "\n".join(summaries) or "No tracker change was applied."

    def _resolved_deadline(self, change: ProjectChange) -> tuple[ProjectChange | None, str | None]:
        """Resolve a copied deadline locally; never guess one, and never drop it silently.

        Returns the change to apply and the sentence explaining an unresolved deadline.
        A task whose deadline could not be resolved is still created without one, and the
        user can set it afterwards with an ordinary follow-up message.
        """

        if change.due_date_expression is None:
            return change, None
        if repeat := states_recurrence(change.due_date_expression):
            return self._deadline_not_set(
                change,
                f'I cannot store a repeating deadline such as "{repeat}", only a single dated one.',
            )
        resolution = resolve_relative_date(
            change.due_date_expression, self.reference_date, self.timezone
        )
        if resolution.status != "resolved" or resolution.resolved_date is None:
            return self._deadline_not_set(
                change, f'I could not resolve the deadline date "{change.due_date_expression}".'
            )
        due: date | datetime = resolution.resolved_date
        if change.due_time_expression is not None:
            due_time = parse_time(change.due_time_expression)
            if due_time is None:
                return self._deadline_not_set(
                    change,
                    f'I could not resolve the deadline time "{change.due_time_expression}"'
                    " (for example 上午十点 or 晚上十点).",
                )
            due = datetime.combine(resolution.resolved_date, due_time, tzinfo=ZoneInfo(self.timezone))
        return change.model_copy(update={"due_at": due}), None

    @staticmethod
    def _deadline_not_set(change: ProjectChange, reason: str) -> tuple[ProjectChange | None, str]:
        """Keep a new task without its deadline; a deadline-only change has nothing left to do."""

        if change.action != "create_record":
            return None, f"{reason} No deadline was set."
        cleared = {"due_date_expression": None, "due_time_expression": None, "due_at": None}
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
            missing_field=self._unresolved_field(decision.pending_action),
            created_at=user_message.created_at,
        )
        self.store.save_pending_clarification(clarification)
        return assistant

    @staticmethod
    def _unresolved_field(pending_action: ContextualAction) -> str:
        """Record why the turn stopped, rather than defaulting every reason to a date."""

        if pending_action.action in {"update_event", "delete_event"}:
            if pending_action.target_event_id is None:
                return "target"
        return "date"

    def _save_calendar_clarification(
        self, user_message: Message, action: ContextualAction | None, missing_field: str
    ) -> Message:
        assert action is not None
        return self._save_pending_question(
            user_message, missing_field, action.model_dump(mode="json")
        )

    def _save_pending_question(
        self, user_message: Message, missing_field: str, action: dict
    ) -> Message:
        """Ask one bounded question and keep the unexecuted action, writing nothing."""

        self._replace_unanswered_question(user_message)
        assistant = self._assistant_message(user_message, self._question(missing_field))
        self.store.save_pending_clarification(
            PendingClarification(
                id=_stable_id(user_message.id, f"contextual-calendar-{missing_field}-clarification"),
                conversation_id=user_message.conversation_id,
                source_message_id=user_message.id,
                assistant_message_id=assistant.id,
                original_action=action,
                missing_field=missing_field,
                created_at=user_message.created_at,
            )
        )
        return assistant

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
            "target": (
                "This conversation has not mentioned that event yet. "
                "Which local calendar event do you mean? Please name it."
            ),
        }[missing_field]

    def _replace_unanswered_question(self, user_message: Message) -> None:
        """Only one question may be pending; a newer one replaces a question left unanswered."""

        stale = self.store.get_pending_clarification(user_message.conversation_id)
        if stale is not None and stale.source_message_id != user_message.id:
            self.store.resolve_pending_clarification(stale.id, "cancelled")

    def _save_follow_up_clarification(
        self, user_message: Message, assistant: Message, deferred_action: ContextualAction
    ) -> PendingClarification:
        """Keep the unexecuted second item of a compound message as one pending question."""

        self._replace_unanswered_question(user_message)
        return self.store.save_pending_clarification(
            PendingClarification(
                id=_stable_id(user_message.id, "contextual-follow-up-clarification"),
                conversation_id=user_message.conversation_id,
                source_message_id=user_message.id,
                assistant_message_id=assistant.id,
                original_action=deferred_action.model_dump(mode="json"),
                missing_field="follow_up",
                created_at=user_message.created_at,
            )
        )

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
        self,
        existing: CalendarEvent,
        action: ContextualAction,
        target_date: date,
        description: str | None,
    ) -> CalendarEvent:
        start_at = existing.start_at
        end_at = existing.end_at
        all_day = existing.all_day
        if action.time_expression is not None:
            start_time = parse_time(action.time_expression)
            if start_time is None:
                raise ValueError("event start time is not valid")
            start_at = datetime.combine(target_date, start_time, tzinfo=ZoneInfo(existing.timezone))
            all_day = False
            end_at = None
        elif action.date_expression is not None:
            if isinstance(existing.start_at, datetime):
                start_at = datetime.combine(
                    target_date, existing.start_at.timetz(), tzinfo=existing.start_at.tzinfo
                )
                if existing.end_at is not None:
                    end_at = start_at + (existing.end_at - existing.start_at)
            else:
                start_at = target_date
        if action.end_time_expression is not None:
            if not isinstance(start_at, datetime):
                raise ValueError("event end time requires a start time")
            end_time = parse_end_time(action.end_time_expression, action.time_expression)
            if end_time is None:
                raise ValueError("event end time is not valid")
            end_at = datetime.combine(start_at.date(), end_time, tzinfo=ZoneInfo(existing.timezone))
            if end_at <= start_at:
                end_at += timedelta(days=1)
        elif action.duration_expression is not None:
            if not isinstance(start_at, datetime):
                raise ValueError("event duration requires a start time")
            duration = parse_duration(action.duration_expression)
            if duration is None:
                raise ValueError("event duration is not valid")
            end_at = start_at + duration
        return existing.model_copy(
            update={
                "title": action.content if action.content is not None else existing.title,
                "description": description if description is not None else existing.description,
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
        for action in (decision.action, decision.pending_action, decision.deferred_action):
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


def _distinctive_terms(value: str) -> set[str]:
    """Chinese character pairs and Latin words that can name one event, minus generic ones.

    Pairs rather than single characters, so `周四` in a date does not match `周例会`.
    """

    lowered = value.casefold()
    words = {word for word in re.findall(r"[a-z0-9]{2,}", lowered)}
    pairs = {
        run[index : index + 2]
        for run in re.findall(r"[一-鿿]{2,}", value)
        for index in range(len(run) - 1)
    }
    return (words | pairs) - _GENERIC_EVENT_TERMS


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
