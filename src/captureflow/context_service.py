"""Bounded, project-scoped conversation replies for V2 Phase 5."""

from dataclasses import dataclass
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import ValidationError

from .llm import ModelError, ProjectConversationModel
from .models import (
    CalendarEvent,
    Message,
    Project,
    ProjectConversationDecision,
    ProjectRecord,
    SavedItem,
    utc_now,
)
from .storage import CaptureStore


MAX_CONTEXT_MESSAGES = 8
MAX_CONTEXT_CHARACTERS = 12_000
MAX_ROUTER_CANDIDATES_PER_KIND = 3


@dataclass(frozen=True)
class ProjectContextWindow:
    messages: tuple[Message, ...]
    omitted_message_count: int

    @property
    def truncated(self) -> bool:
        return self.omitted_message_count > 0

    def for_model(self) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in self.messages]

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


@dataclass(frozen=True)
class ProjectConversationTurn:
    user_message: Message
    assistant_message: Message
    decision: ProjectConversationDecision | None
    context_window: ProjectContextWindow
    model_calls: int
    model_seconds: float
    outcome: str


def select_project_context(
    messages: list[Message],
    *,
    max_messages: int = MAX_CONTEXT_MESSAGES,
    max_characters: int = MAX_CONTEXT_CHARACTERS,
) -> ProjectContextWindow:
    """Choose one contiguous recent window without truncating a persisted message."""

    if max_messages < 1 or max_characters < 1:
        raise ValueError("project context limits must be positive")
    if not messages:
        raise ValueError("project context requires at least one message")

    ordered_messages = _logical_turn_order(messages)
    selected_reversed: list[Message] = []
    character_count = 0
    for index in range(len(ordered_messages) - 1, -1, -1):
        message = ordered_messages[index]
        would_exceed_message_limit = len(selected_reversed) >= max_messages
        would_exceed_character_limit = character_count + len(message.content) > max_characters
        if selected_reversed and (would_exceed_message_limit or would_exceed_character_limit):
            return ProjectContextWindow(
                messages=tuple(reversed(selected_reversed)), omitted_message_count=index + 1
            )
        selected_reversed.append(message)
        character_count += len(message.content)

    return ProjectContextWindow(messages=tuple(reversed(selected_reversed)), omitted_message_count=0)


def select_router_candidates(
    store: CaptureStore,
    conversation_id: str,
    *,
    max_per_kind: int = MAX_ROUTER_CANDIDATES_PER_KIND,
) -> RouterCandidates:
    """Select stable local candidates without widening the current workspace or Project scope."""

    if max_per_kind < 1:
        raise ValueError("router candidate limit must be positive")
    conversation = store.get_conversation(conversation_id)
    events = tuple(store.list_calendar_events(conversation.workspace_id)[:max_per_kind])
    project = store.get_conversation_project(conversation_id)
    if project is None:
        return RouterCandidates(
            events=events,
            saved_items=tuple(store.list_saved_items(conversation.workspace_id)[:max_per_kind]),
            project_records=(),
        )

    return RouterCandidates(
        events=events,
        saved_items=tuple(
            store.list_saved_items_for_project(project.id, conversation.workspace_id)[:max_per_kind]
        ),
        project_records=tuple(
            store.list_project_records(project.id, states=("active",))[:max_per_kind]
        ),
    )


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


class ProjectConversationService:
    """Persist one bounded contextual reply; Phase 5 deliberately writes no ProjectRecord."""

    def __init__(self, store: CaptureStore, model: ProjectConversationModel) -> None:
        self.store = store
        self.model = model

    def submit(
        self, conversation_id: str, content: str, message_id: str | None = None
    ) -> ProjectConversationTurn:
        project = self.store.begin_project_conversation(conversation_id)
        user_message = self._user_message(conversation_id, content, message_id)
        context_window = select_project_context(self.store.list_messages(conversation_id))
        existing_reply = self.store.get_assistant_reply(user_message.id)
        if existing_reply is not None:
            return ProjectConversationTurn(
                user_message=user_message,
                assistant_message=existing_reply,
                decision=None,
                context_window=context_window,
                model_calls=0,
                model_seconds=0,
                outcome="reloaded",
            )

        started = perf_counter()
        try:
            decision = ProjectConversationDecision.model_validate(
                self.model.respond_to_project_turn(
                    project, context_window.for_model(), context_window.truncated
                )
            )
        except (ModelError, ValidationError, ValueError):
            return self._failure_reply(user_message, context_window, perf_counter() - started)

        reply = decision.reply
        if decision.turn_kind == "clarify":
            reply = f"{reply}\n\n{decision.clarification_question}"
        if context_window.truncated:
            reply = (
                f"{reply}\n\n_Context note: I used only the most recent "
                f"{len(context_window.messages)} messages; "
                f"{context_window.omitted_message_count} older message(s) were outside this turn's context window._"
            )
        assistant = self._reply_message(user_message, reply)
        return ProjectConversationTurn(
            user_message=user_message,
            assistant_message=assistant,
            decision=decision,
            context_window=context_window,
            model_calls=1,
            model_seconds=perf_counter() - started,
            outcome="contextual_reply",
        )

    def _failure_reply(
        self, user_message: Message, context_window: ProjectContextWindow, model_seconds: float
    ) -> ProjectConversationTurn:
        assistant = self._reply_message(
            user_message,
            "I could not process this project conversation turn, so no tracker record was changed.",
        )
        return ProjectConversationTurn(
            user_message=user_message,
            assistant_message=assistant,
            decision=None,
            context_window=context_window,
            model_calls=1,
            model_seconds=model_seconds,
            outcome="model_failure",
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
                id=str(uuid5(NAMESPACE_URL, f"captureflow:project-context:{user_message.id}")),
                conversation_id=user_message.conversation_id,
                role="assistant",
                content=content,
                reply_to_message_id=user_message.id,
                created_at=utc_now(),
            )
        )
