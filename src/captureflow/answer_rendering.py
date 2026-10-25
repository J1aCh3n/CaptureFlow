"""How an answer presents the rows it cited: their stored dates, and their source cards.

This is what is left of the two answer services. The router used to answer a question
about specific stored items from candidates the model named, and to make a second model
call over a snapshot of the whole project; the agent searches for what it needs and cites
what it found, so neither remains. What survives is the part that was never the model's
job: a date or time a user reads is read from SQLite and never written by the model
(Phases 11 and 12), and a cited row is turned into the kind of source card SQLite keeps.
"""

from .models import CalendarEvent, LocalContextSourceRef, ProjectRecord, SavedItem


_STORED_DATES_HEADER = "Stored dates:"


def answer_source_references(
    sources: list[CalendarEvent | SavedItem | ProjectRecord],
) -> tuple[list[str], list[LocalContextSourceRef]]:
    """Split an answer's sources into project record IDs and Calendar/Note/Idea references."""

    project_record_ids = [source.id for source in sources if isinstance(source, ProjectRecord)]
    local_references = [
        LocalContextSourceRef(
            source_type="calendar_event" if isinstance(source, CalendarEvent) else "saved_item",
            source_id=source.id,
        )
        for source in sources
        if isinstance(source, (CalendarEvent, SavedItem))
    ]
    return project_record_ids, local_references


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
