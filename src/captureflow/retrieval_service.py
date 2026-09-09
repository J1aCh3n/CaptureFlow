"""Phase 7 project answers grounded in bounded local retrieval results."""

from dataclasses import dataclass
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import ValidationError

from .context_service import ProjectContextWindow, select_project_context
from .llm import ModelError, ProjectRetrievalModel
from .models import Message, Project, ProjectAnswerDecision, ProjectRecord, utc_now
from .retrieval import RetrievalMode, retrieve_project_records
from .storage import CaptureStore


@dataclass(frozen=True)
class ProjectRetrievalTurn:
    user_message: Message
    assistant_message: Message
    decision: ProjectAnswerDecision | None
    context_window: ProjectContextWindow
    records: tuple[ProjectRecord, ...]
    model_calls: int
    model_seconds: float
    outcome: str


class ProjectRetrievalService:
    """Persist a read-only answer whose stored source cards survive reruns."""

    def __init__(self, store: CaptureStore, model: ProjectRetrievalModel) -> None:
        self.store = store
        self.model = model

    def submit(
        self,
        conversation_id: str,
        content: str,
        mode: RetrievalMode,
        message_id: str | None = None,
    ) -> ProjectRetrievalTurn:
        project = self.store.begin_project_conversation(conversation_id)
        user_message = self._user_message(conversation_id, content, message_id)
        context_window = select_project_context(self.store.list_messages(conversation_id))
        existing_reply = self.store.get_assistant_reply(user_message.id)
        if existing_reply is not None:
            return ProjectRetrievalTurn(
                user_message=user_message,
                assistant_message=existing_reply,
                decision=None,
                context_window=context_window,
                records=tuple(self.store.list_project_answer_sources(existing_reply.id)),
                model_calls=0,
                model_seconds=0,
                outcome="reloaded",
            )

        records = retrieve_project_records(self.store, project.id, content, mode)
        if not records:
            assistant = self._reply_message(
                user_message,
                "I do not have enough stored project context to answer that yet.",
            )
            return ProjectRetrievalTurn(
                user_message=user_message,
                assistant_message=assistant,
                decision=None,
                context_window=context_window,
                records=(),
                model_calls=0,
                model_seconds=0,
                outcome="insufficient_context",
            )

        started = perf_counter()
        answer_kind = "history" if mode == "history" else "current_status"
        try:
            decision = ProjectAnswerDecision.model_validate(
                self.model.answer_project_query(
                    project,
                    context_window.for_model(),
                    [self._record_for_model(record, mode) for record in records],
                    answer_kind,
                    context_window.truncated,
                )
            )
            if decision.answer_kind != answer_kind:
                raise ValueError("project answer kind does not match retrieval mode")
            record_by_id = {record.id: record for record in records}
            if any(record_id not in record_by_id for record_id in decision.supporting_record_ids):
                raise ValueError("project answer cited a record outside the retrieved set")
        except (ModelError, ValidationError, ValueError):
            return self._failure_reply(user_message, context_window, records, perf_counter() - started)

        reply = decision.reply
        if context_window.truncated:
            reply = (
                f"{reply}\n\n_Context note: I used only the most recent "
                f"{len(context_window.messages)} messages; "
                f"{context_window.omitted_message_count} older message(s) were outside this turn's context window._"
            )
        assistant = self._reply_message(user_message, reply)
        supporting_records = [record_by_id[record_id] for record_id in decision.supporting_record_ids]
        self.store.save_project_answer_sources(assistant.id, decision.supporting_record_ids)
        return ProjectRetrievalTurn(
            user_message=user_message,
            assistant_message=assistant,
            decision=decision,
            context_window=context_window,
            records=tuple(supporting_records),
            model_calls=1,
            model_seconds=perf_counter() - started,
            outcome="grounded_answer",
        )

    def _failure_reply(
        self,
        user_message: Message,
        context_window: ProjectContextWindow,
        records: list[ProjectRecord],
        model_seconds: float,
    ) -> ProjectRetrievalTurn:
        assistant = self._reply_message(
            user_message,
            "I could not safely form a source-grounded project answer from the retrieved records.",
        )
        return ProjectRetrievalTurn(
            user_message=user_message,
            assistant_message=assistant,
            decision=None,
            context_window=context_window,
            records=tuple(records),
            model_calls=1,
            model_seconds=model_seconds,
            outcome="model_or_validation_failure",
        )

    def _record_for_model(self, record: ProjectRecord, mode: RetrievalMode) -> dict:
        payload = {
            "id": record.id,
            "kind": record.kind,
            "content": record.content,
            "rationale": record.rationale,
            "state": record.state,
            "source_quote": record.source_quote,
        }
        if mode == "history":
            payload["events"] = [
                {
                    "action": event.action,
                    "previous_state": event.previous_state,
                    "new_state": event.new_state,
                    "details": event.details,
                }
                for event in self.store.list_project_record_events(record.id)
            ]
        return payload

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
                id=str(uuid5(NAMESPACE_URL, f"captureflow:project-retrieval:{user_message.id}")),
                conversation_id=user_message.conversation_id,
                role="assistant",
                content=content,
                reply_to_message_id=user_message.id,
                created_at=utc_now(),
            )
        )
