"""Deterministic V2 chat commands with no model dependency."""

from dataclasses import dataclass
from datetime import date
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import ValidationError

from .models import (
    CalendarEvent,
    CalendarEventUpdateOperation,
    Message,
    SavedItem,
    ToolOperation,
    utc_now,
)
from .storage import CaptureStore


@dataclass(frozen=True)
class ChatTurn:
    user_message: Message
    assistant_message: Message
    operation: ToolOperation | None = None


class CommandError(ValueError):
    """A user-correctable error in the explicit Phase 2 command language."""


class DeterministicChatService:
    """Persist a small, explicit command language for the Phase 2 shell."""

    def __init__(self, store: CaptureStore, timezone: str = "America/Vancouver") -> None:
        self.store = store
        self.timezone = timezone

    def submit(self, conversation_id: str, content: str, message_id: str | None = None) -> ChatTurn:
        user_message = self._user_message(conversation_id, content, message_id)
        try:
            command = _parse_command(user_message.content)
        except CommandError as error:
            return self._reply_only(user_message, str(error))
        if command is None:
            return self._reply_only(
                user_message,
                "No local action was created. Use `/event YYYY-MM-DD | title`, "
                "`/idea text`, or `/note text`.",
            )

        command_kind, payload = command
        try:
            if command_kind == "event":
                target = CalendarEvent(
                    id=_stable_id(user_message.id, "event"),
                    conversation_id=conversation_id,
                    title=payload["title"],
                    source_message_id=user_message.id,
                    source_quote=user_message.content,
                    start_at=payload["start_at"],
                    all_day=True,
                    timezone=self.timezone,
                    created_at=user_message.created_at,
                    updated_at=user_message.created_at,
                )
                saved = self.store.save_calendar_event(target)
                operation = ToolOperation(
                    id=_stable_id(user_message.id, "operation"),
                    source_message_id=user_message.id,
                    action_index=0,
                    tool_name="create_local_event",
                    request={"title": saved.title, "start_at": saved.start_at.isoformat()},
                    result={"event_id": saved.id},
                    status="committed",
                    created_at=user_message.created_at,
                    updated_at=user_message.created_at,
                )
                reply = f"Created local calendar event: {saved.title} on {saved.start_at.isoformat()}."
            else:
                saved = self.store.save_saved_item(
                    SavedItem(
                        id=_stable_id(user_message.id, command_kind),
                        conversation_id=conversation_id,
                        kind=command_kind,
                        content=payload["content"],
                        source_message_id=user_message.id,
                        source_quote=user_message.content,
                        created_at=user_message.created_at,
                        updated_at=user_message.created_at,
                    )
                )
                operation = ToolOperation(
                    id=_stable_id(user_message.id, "operation"),
                    source_message_id=user_message.id,
                    action_index=0,
                    tool_name="save_idea" if command_kind == "idea" else "save_note",
                    request={"content": saved.content},
                    result={"item_id": saved.id},
                    status="committed",
                    created_at=user_message.created_at,
                    updated_at=user_message.created_at,
                )
                reply = f"Saved {command_kind}: {saved.content}"
        except ValidationError as error:
            return self._reply_only(user_message, error.errors()[0]["msg"])

        operation = self.store.save_tool_operation(operation)
        assistant_message = self.store.save_message(
            Message(
                id=_stable_id(user_message.id, "assistant"),
                conversation_id=conversation_id,
                role="assistant",
                content=reply,
                reply_to_message_id=user_message.id,
                created_at=user_message.created_at,
            )
        )
        return ChatTurn(user_message, assistant_message, operation)

    def undo(self, operation_id: str) -> ToolOperation | CalendarEventUpdateOperation:
        try:
            operation = self.store.get_tool_operation(operation_id)
        except KeyError:
            return self.store.undo_calendar_event_update(operation_id)
        if operation.status == "undone":
            return operation
        if operation.status != "committed":
            raise ValueError("only a committed operation can be undone")
        result = operation.result or {}
        if event_id := result.get("event_id"):
            self.store.set_calendar_event_status(event_id, "deleted")
        elif item_id := result.get("item_id"):
            self.store.set_saved_item_status(item_id, "deleted")
        else:
            raise ValueError("committed operation does not identify a local target")
        return self.store.update_tool_operation(
            operation.model_copy(update={"status": "undone", "updated_at": utc_now()})
        )

    def submit_reply_only(
        self, conversation_id: str, content: str, reply: str, message_id: str | None = None
    ) -> ChatTurn:
        """Persist a deterministic visible failure without attempting a model call."""

        return self._reply_only(self._user_message(conversation_id, content, message_id), reply)

    def _reply_only(self, user_message: Message, content: str) -> ChatTurn:
        assistant_message = self.store.save_message(
            Message(
                id=_stable_id(user_message.id, "assistant"),
                conversation_id=user_message.conversation_id,
                role="assistant",
                content=content,
                reply_to_message_id=user_message.id,
                created_at=user_message.created_at,
            )
        )
        return ChatTurn(user_message, assistant_message)

    def _user_message(
        self, conversation_id: str, content: str, message_id: str | None
    ) -> Message:
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


def _parse_command(content: str) -> tuple[str, dict] | None:
    text = content.strip()
    if text.startswith("/event "):
        date_text, separator, title = text.removeprefix("/event ").partition("|")
        if not separator or not title.strip():
            raise CommandError("Use `/event YYYY-MM-DD | title`.")
        try:
            start_at = date.fromisoformat(date_text.strip())
        except ValueError as error:
            raise CommandError("Event date must use YYYY-MM-DD.") from error
        return "event", {"title": title.strip(), "start_at": start_at}
    for kind in ("idea", "note"):
        prefix = f"/{kind} "
        if text.startswith(prefix):
            content = text.removeprefix(prefix).strip()
            if not content:
                raise CommandError(f"{kind.capitalize()} content must not be blank.")
            return kind, {"content": content}
    return None


def _stable_id(message_id: str, suffix: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"captureflow:{message_id}:{suffix}"))
