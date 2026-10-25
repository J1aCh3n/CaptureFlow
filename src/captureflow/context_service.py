"""What the agent sees of the conversation, and the one stored list Python still needs.

The recent conversation goes to the model in layers, each with a budget. A single user's
data fits inside every budget, so the model in practice sees everything; the bounds only
decide what gives way as data grows.

Stored state is no longer pre-selected into the prompt. The router used to hand the model
three lists -- events from the whole workspace, the project's notes and ideas, its active
records -- for it to choose a write's target from. The agent searches for what it needs and
the provenance ledger vouches for the target it found, so those lists went unread: two of
the three were built and discarded on every turn, read-only turns included.
"""

from dataclasses import dataclass
from datetime import datetime

from .credentials import contains_credential_disclosure
from .models import Message, ProjectRecord
from .storage import CaptureStore


# The conversation: recent turns in full, older long messages as one-line references.
MAX_CONTEXT_CHARACTERS = 12_000
RECENT_FULL_TEXT_MESSAGES = 4
LONG_MESSAGE_CHARACTERS = 800
REFERENCE_PREVIEW_CHARACTERS = 80


@dataclass(frozen=True)
class ContextWindow:
    messages: tuple[Message, ...]
    omitted_message_count: int

    @property
    def truncated(self) -> bool:
        return self.omitted_message_count > 0

    def for_router(self) -> list[dict]:
        """Recent messages in full; an older long message as a reference to its ID."""

        recent = set(message.id for message in self.messages[-RECENT_FULL_TEXT_MESSAGES:])
        return [
            _message_for_router(message, full_text=message.id in recent)
            for message in self.messages
        ]

    def user_message(self, message_id: str) -> Message | None:
        return next(
            (message for message in self.messages if message.id == message_id and message.role == "user"),
            None,
        )


def select_context_window(
    messages: list[Message], *, max_characters: int = MAX_CONTEXT_CHARACTERS
) -> ContextWindow:
    """The recent conversation, newest first, until the character budget is spent.

    The last two turns always keep their full text. An older long message costs only
    its one-line reference. A message the credential guard blocked is never included.
    """

    if max_characters < 1:
        raise ValueError("the context budget must be positive")
    if not messages:
        raise ValueError("a context window needs at least one message")

    ordered = [
        message
        for message in _logical_turn_order(messages)
        if not (message.role == "user" and contains_credential_disclosure(message.content))
    ]
    selected_reversed: list[Message] = []
    character_count = 0
    for index in range(len(ordered) - 1, -1, -1):
        message = ordered[index]
        recent = len(selected_reversed) < RECENT_FULL_TEXT_MESSAGES
        cost = len(message.content) if recent else _older_message_cost(message)
        if not recent and character_count + cost > max_characters:
            return ContextWindow(tuple(reversed(selected_reversed)), omitted_message_count=index + 1)
        selected_reversed.append(message)
        character_count += cost
    return ContextWindow(tuple(reversed(selected_reversed)), omitted_message_count=0)


def active_project_records(
    store: CaptureStore, conversation_id: str
) -> tuple[ProjectRecord, ...]:
    """Every active record of the conversation's project, for the rules that compare against it.

    This is all that is left of candidate preselection, and creating a record is all that
    still needs it: an identical active record is refused, and a second decision or
    constraint is held for review. Both are decided by comparing against everything the
    project holds, so the list is not bounded any more. The old bound of 30 was there to
    keep the prompt short; with the prompt gone, a record past it would simply not have been
    compared against, which is a duplicate let through rather than a shorter prompt.

    A write that names something already stored does not come through here at all: the
    ledger hands the row it vouched for straight to the rule.
    """

    project = store.get_conversation_project(conversation_id)
    if project is None:
        return ()
    return tuple(store.list_project_records(project.id, states=("active",)))


def _older_message_cost(message: Message) -> int:
    if len(message.content) <= LONG_MESSAGE_CHARACTERS:
        return len(message.content)
    return REFERENCE_PREVIEW_CHARACTERS + 40


def _message_for_router(message: Message, *, full_text: bool) -> dict:
    if full_text or len(message.content) <= LONG_MESSAGE_CHARACTERS:
        return {"message_id": message.id, "role": message.role, "content": message.content}
    return {
        "message_id": message.id,
        "role": message.role,
        "reference": (
            f"long message, {len(message.content)} characters, starting: "
            f"{message.content[:REFERENCE_PREVIEW_CHARACTERS]}…"
        ),
    }


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
