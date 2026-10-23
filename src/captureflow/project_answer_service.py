"""Render the stored dates of an answer's cited rows underneath it.

This is what remains of the project-answer service. The router used to make a second model
call over a snapshot of the whole project; the agent now searches the project itself. The
time policy survives unchanged: a date or time shown to the user is read from SQLite and
never written by the model (Phases 11 and 12).
"""

from .models import CalendarEvent, ProjectRecord


def with_stored_dates(
    reply: str, records: tuple[ProjectRecord, ...], events: tuple[CalendarEvent, ...]
) -> str:
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
    return "\n".join([reply, "", "Stored dates:", *dated])
