"""Phase 6: model-proposed, locally validated project tracker changes."""

from dataclasses import dataclass
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import ValidationError

from .context_service import ProjectContextWindow, select_project_context
from .llm import ModelError, ProjectTrackerModel
from .memory import contains_credential_disclosure
from .models import (
    Message,
    Project,
    ProjectChange,
    ProjectChangeOperation,
    ProjectRecord,
    ProjectTrackerDecision,
    utc_now,
)
from .storage import CaptureStore


MAX_CANDIDATE_RECORDS = 20


@dataclass(frozen=True)
class ProjectTrackerTurn:
    user_message: Message
    assistant_message: Message
    decision: ProjectTrackerDecision | None
    context_window: ProjectContextWindow
    operations: tuple[ProjectChangeOperation, ...]
    model_calls: int
    model_seconds: float
    outcome: str


class ProjectTrackerService:
    """Keep the model outside the write boundary for a project conversation turn."""

    def __init__(self, store: CaptureStore, model: ProjectTrackerModel) -> None:
        self.store = store
        self.model = model

    def submit(
        self, conversation_id: str, content: str, message_id: str | None = None
    ) -> ProjectTrackerTurn:
        project = self.store.begin_project_conversation(conversation_id)
        user_message = self._user_message(conversation_id, content, message_id)
        context_window = select_project_context(self.store.list_messages(conversation_id))
        existing_reply = self.store.get_assistant_reply(user_message.id)
        if existing_reply is not None:
            return ProjectTrackerTurn(
                user_message=user_message,
                assistant_message=existing_reply,
                decision=None,
                context_window=context_window,
                operations=tuple(self.store.list_project_change_operations(user_message.id)),
                model_calls=0,
                model_seconds=0,
                outcome="reloaded",
            )

        persisted_operations = self.store.list_project_change_operations(user_message.id)
        if persisted_operations:
            resumed: list[ProjectChangeOperation] = []
            for operation in persisted_operations:
                if operation.status == "prepared":
                    try:
                        operation = self.store.commit_prepared_project_change(operation.id)
                    except (KeyError, ValidationError, ValueError) as error:
                        operation = self.store.fail_prepared_project_change(operation.id, str(error))
                resumed.append(operation)
            assistant = self._reply_message(
                user_message, self._resumed_reply(resumed, context_window)
            )
            return ProjectTrackerTurn(
                user_message=user_message,
                assistant_message=assistant,
                decision=None,
                context_window=context_window,
                operations=tuple(resumed),
                model_calls=0,
                model_seconds=0,
                outcome="resumed_prepared_change",
            )

        candidates = self._candidate_records(project)
        started = perf_counter()
        try:
            decision = ProjectTrackerDecision.model_validate(
                self.model.propose_project_changes(
                    project,
                    context_window.for_model(),
                    [self._candidate_for_model(record) for record in candidates],
                    context_window.truncated,
                )
            )
            committed, reviewed = self.apply_validated_changes(
                self.store, project, user_message, decision.changes, candidates
            )
        except (ModelError, ValidationError, ValueError) as error:
            return self._failure_reply(user_message, context_window, perf_counter() - started, str(error))

        reply = self._reply_text(decision, committed, reviewed, context_window)
        assistant = self._reply_message(user_message, reply)
        operations = tuple([*committed, *reviewed])
        outcome = "no_change" if not operations else "tracker_change"
        if any(operation.status == "failed" for operation in operations):
            outcome = "tracker_change_failed"
        return ProjectTrackerTurn(
            user_message=user_message,
            assistant_message=assistant,
            decision=decision,
            context_window=context_window,
            operations=operations,
            model_calls=1,
            model_seconds=perf_counter() - started,
            outcome=outcome,
        )

    @staticmethod
    def apply_validated_changes(
        store: CaptureStore,
        project: Project,
        user_message: Message,
        changes: list[ProjectChange],
        candidates: list[ProjectRecord],
    ) -> tuple[list[ProjectChangeOperation], list[ProjectChangeOperation]]:
        """Apply an already-model-validated tracker proposal without another model call."""

        proposed_operations = ProjectTrackerService._prepare_operations(
            project, user_message, changes, candidates
        )
        committed: list[ProjectChangeOperation] = []
        reviewed: list[ProjectChangeOperation] = []
        for operation in proposed_operations:
            saved = store.save_project_change_operation(operation)
            if saved.status == "review":
                reviewed.append(saved)
                continue
            try:
                committed.append(store.commit_prepared_project_change(saved.id))
            except (KeyError, ValidationError, ValueError) as error:
                committed.append(store.fail_prepared_project_change(saved.id, str(error)))
        return committed, reviewed

    def _candidate_records(self, project: Project) -> list[ProjectRecord]:
        # Phase 6 uses a deterministic recency-bounded set; Phase 7 adds retrieval.
        return self.store.list_project_records(project.id, states=("active",))[:MAX_CANDIDATE_RECORDS]

    @staticmethod
    def _candidate_for_model(record: ProjectRecord) -> dict[str, str]:
        return {
            "id": record.id,
            "kind": record.kind,
            "content": record.content,
            "state": record.state,
        }

    @staticmethod
    def _prepare_operations(
        project: Project,
        user_message: Message,
        changes: list[ProjectChange],
        candidates: list[ProjectRecord],
    ) -> list[ProjectChangeOperation]:
        candidate_by_id = {record.id: record for record in candidates}
        active_by_kind_content = {
            (record.kind, ProjectTrackerService._normalized(record.content)): record
            for record in candidates
        }
        signatures: set[tuple[str, str, str]] = set()
        review_indexes: set[int] = set()
        for index, change in enumerate(changes):
            ProjectTrackerService._validate_change_source(change, user_message)
            signature = (
                change.action,
                change.target_record_id or "",
                ProjectTrackerService._normalized(change.content or ""),
            )
            if signature in signatures:
                raise ValueError("duplicate project changes in the same turn are not allowed")
            signatures.add(signature)
            if change.action == "create_record":
                if (
                    change.kind,
                    ProjectTrackerService._normalized(change.content or ""),
                ) in active_by_kind_content:
                    raise ValueError("an identical active project record already exists")
                if change.kind in {"decision", "constraint"} and any(
                    record.kind == change.kind for record in candidates
                ):
                    review_indexes.add(index)
            elif change.action == "transition_record":
                record = candidate_by_id.get(change.target_record_id or "")
                if record is None:
                    raise ValueError("project change target was not supplied as an active candidate")
                allowed = {
                    ("task", "completed"),
                    ("open_question", "resolved"),
                    ("blocker", "resolved"),
                }
                if (record.kind, change.target_state) not in allowed:
                    raise ValueError("invalid project record lifecycle transition")
            else:
                record = candidate_by_id.get(change.target_record_id or "")
                if record is None or record.kind != change.kind or record.kind not in {
                    "decision",
                    "constraint",
                }:
                    raise ValueError("a supersession must target a supplied active record of the same kind")
                review_indexes.add(index)

        operations: list[ProjectChangeOperation] = []
        for index, change in enumerate(changes):
            operation_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"captureflow:project-change:{user_message.id}:{index}:{change.action}",
                )
            )
            operations.append(
                ProjectChangeOperation(
                    id=operation_id,
                    source_message_id=user_message.id,
                    project_id=project.id,
                    change_index=index,
                    change=change.model_dump(exclude_none=True),
                    target_record_id=change.target_record_id,
                    status="review" if index in review_indexes else "prepared",
                )
            )
        return operations

    @staticmethod
    def _validate_change_source(change: ProjectChange, user_message: Message) -> None:
        normalized_source = ProjectTrackerService._normalized(user_message.content)
        normalized_quote = ProjectTrackerService._normalized(change.source_quote)
        if normalized_quote not in normalized_source:
            raise ValueError("a project change source quote must occur in the current user message")
        if contains_credential_disclosure(change.source_quote) or (
            change.content is not None and contains_credential_disclosure(change.content)
        ):
            raise ValueError("credentials cannot be persisted in a project change")

    @staticmethod
    def _normalized(value: str) -> str:
        return " ".join(value.split()).casefold()

    def _reply_text(
        self,
        decision: ProjectTrackerDecision,
        committed: list[ProjectChangeOperation],
        reviewed: list[ProjectChangeOperation],
        context_window: ProjectContextWindow,
    ) -> str:
        reply = decision.reply
        if decision.turn_kind == "clarify":
            reply = f"{reply}\n\n{decision.clarification_question}"
        summaries: list[str] = []
        for operation in committed:
            action = operation.change["action"]
            if operation.status == "committed":
                summaries.append(f"Tracker update applied: {action.replace('_', ' ')}.")
            else:
                summaries.append("Tracker update was not applied; see the change card for the reason.")
        for operation in reviewed:
            summaries.append("Tracker update needs review before it can be applied.")
        if summaries:
            reply = f"{reply}\n\n" + "\n".join(summaries)
        if context_window.truncated:
            reply = (
                f"{reply}\n\n_Context note: I used only the most recent "
                f"{len(context_window.messages)} messages; "
                f"{context_window.omitted_message_count} older message(s) were outside this turn's context window._"
            )
        return reply

    @staticmethod
    def _resumed_reply(
        operations: list[ProjectChangeOperation], context_window: ProjectContextWindow
    ) -> str:
        if any(operation.status == "review" for operation in operations):
            reply = "I restored the pending tracker change. It still needs review before it can be applied."
        elif any(operation.status == "committed" for operation in operations):
            reply = "I resumed the prepared tracker change from its saved local operation."
        else:
            reply = "I restored the prior tracker request, but it could not be applied safely."
        if context_window.truncated:
            return (
                f"{reply}\n\n_Context note: I used only the most recent "
                f"{len(context_window.messages)} messages; "
                f"{context_window.omitted_message_count} older message(s) were outside this turn's context window._"
            )
        return reply

    def _failure_reply(
        self,
        user_message: Message,
        context_window: ProjectContextWindow,
        model_seconds: float,
        _error: str,
    ) -> ProjectTrackerTurn:
        assistant = self._reply_message(
            user_message,
            "I could not safely process this project conversation turn, so no tracker record was changed.",
        )
        return ProjectTrackerTurn(
            user_message=user_message,
            assistant_message=assistant,
            decision=None,
            context_window=context_window,
            operations=(),
            model_calls=1,
            model_seconds=model_seconds,
            outcome="model_or_validation_failure",
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
            Message(
                id=message_id or str(uuid4()),
                conversation_id=conversation_id,
                role="user",
                content=content,
            )
        )

    def _reply_message(self, user_message: Message, content: str) -> Message:
        return self.store.save_message(
            Message(
                id=str(uuid5(NAMESPACE_URL, f"captureflow:project-tracker:{user_message.id}")),
                conversation_id=user_message.conversation_id,
                role="assistant",
                content=content,
                reply_to_message_id=user_message.id,
                created_at=utc_now(),
            )
        )
