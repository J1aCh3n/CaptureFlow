"""Render the stored dates of an answer's cited rows underneath it.

This is what remains of the project-answer service. The router used to make a second model
call over a snapshot of the whole project; the agent now searches the project itself. The
time policy survives unchanged: a date or time shown to the user is read from SQLite and
never written by the model (Phases 11 and 12).
"""

from .models import CalendarEvent, ProjectRecord


_STORED_DATES_HEADER = "Stored dates:"


def with_stored_dates(
    reply: str, records: tuple[ProjectRecord, ...], events: tuple[CalendarEvent, ...]
) -> str:
    reply = _without_a_copied_block(reply)
    dated = [
        f"- Task deadline {record.due_at.isoformat()}: {record.content}"
        for record in records
        if record.due_at is not None
    ] + [
        f"- Calendar {event.start_at.isoformat()}"
        + ("" if event.end_at is None else f" to {event.end_at.isoformat()}")
        + f": {event.title}"
        for event in events
    ]
    if not dated:
        return reply
    return "\n".join([reply, "", _STORED_DATES_HEADER, *dated])


def _without_a_copied_block(reply: str) -> str:
    """Drop a block the model wrote in this one's shape, header and date lines together.

    The model sees earlier replies in the conversation, and has copied this block out of
    one. The dates a user reads come from SQLite, so a copy does not get through, whatever
    it says. Only the exact shape rendered below is removed; the words in ordinary prose
    are left alone.
    """

    lines = reply.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != _STORED_DATES_HEADER:
            kept.append(lines[index])
            index += 1
            continue
        index += 1
        while index < len(lines) and lines[index].startswith("- "):
            index += 1
        while kept and not kept[-1].strip():
            kept.pop()
    return "\n".join(kept).strip()
