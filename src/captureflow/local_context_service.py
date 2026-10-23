"""Split an answer's cited rows into the two kinds of source card SQLite keeps.

This is what remains of the local-answer service. The router used to answer a question
about specific stored items itself, from candidates the model named; the agent now
searches for them and cites what it found, so only the source-card split is still needed.
"""

from .models import CalendarEvent, LocalContextSourceRef, ProjectRecord, SavedItem


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
