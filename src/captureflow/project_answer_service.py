"""Answer a question about a whole project from a bounded snapshot of its stored records."""

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from .models import CalendarEvent, Message, Project, ProjectAnswer, ProjectRecord
from .storage import CaptureStore

MAX_OPEN_RECORDS = 40
MAX_CLOSED_RECORDS = 15
MAX_SUPERSEDED_RECORDS = 15
MAX_UPCOMING_EVENTS = 5
MAX_SOURCE_CARDS = 6

_CLOSED_STATES = ("completed", "resolved")


class ProjectAnswerModel(Protocol):
    def answer_project_question(
        self,
        question: dict[str, str],
        recent_messages: list[dict[str, str]],
        project_snapshot: dict,
        reference_date: str,
    ) -> ProjectAnswer: ...


@dataclass(frozen=True)
class ProjectSnapshot:
    project: Project
    open_records: tuple[ProjectRecord, ...]
    closed_records: tuple[ProjectRecord, ...]
    superseded_records: tuple[ProjectRecord, ...]
    upcoming_events: tuple[CalendarEvent, ...]

    @property
    def records(self) -> tuple[ProjectRecord, ...]:
        return (*self.open_records, *self.closed_records, *self.superseded_records)

    def for_model(self) -> dict:
        replaced_by = {
            record.supersedes_record_id: record.id
            for record in self.records
            if record.supersedes_record_id is not None
        }
        return {
            "project": {"name": self.project.name, "description": self.project.description},
            "open_records": [_record_for_model(record, replaced_by) for record in self.open_records],
            "closed_records": [_record_for_model(record, replaced_by) for record in self.closed_records],
            "superseded_records": [
                _record_for_model(record, replaced_by) for record in self.superseded_records
            ],
            "upcoming_events": [
                {
                    "id": event.id,
                    "title": event.title,
                    "description": event.description,
                    "start_at": event.start_at.isoformat(),
                    "end_at": None if event.end_at is None else event.end_at.isoformat(),
                }
                for event in self.upcoming_events
            ],
        }


def build_project_snapshot(store: CaptureStore, project: Project, reference_date: date) -> ProjectSnapshot:
    """Collect what a project-level answer may use, newest first within each bounded part."""

    open_records = store.list_project_records(project.id, states=("active",))
    closed_records = store.list_project_records(project.id, states=_CLOSED_STATES)
    superseded = store.list_project_records(project.id, states=("superseded",))
    events = [
        event
        for event in store.list_calendar_events(project.workspace_id)
        if event.start_date >= reference_date
        and (assigned := store.get_conversation_project(event.conversation_id)) is not None
        and assigned.id == project.id
    ]
    return ProjectSnapshot(
        project=project,
        open_records=tuple(open_records[:MAX_OPEN_RECORDS]),
        closed_records=tuple(closed_records[:MAX_CLOSED_RECORDS]),
        superseded_records=tuple(superseded[:MAX_SUPERSEDED_RECORDS]),
        upcoming_events=tuple(events[:MAX_UPCOMING_EVENTS]),
    )


@dataclass(frozen=True)
class ProjectAnswerResult:
    reply: str
    records: tuple[ProjectRecord, ...]
    events: tuple[CalendarEvent, ...]


def answer_project_question(
    model: ProjectAnswerModel,
    snapshot: ProjectSnapshot,
    question: Message,
    recent_messages: list[dict[str, str]],
    reference_date: date,
) -> ProjectAnswerResult:
    """Ask the model for a grounded answer, then render stored dates under it.

    Raises ValueError when the answer cites something outside the snapshot, since
    its text may rest on a fact that is not stored.
    """

    answer = ProjectAnswer.model_validate(
        model.answer_project_question(
            question={"message_id": question.id, "content": question.content},
            recent_messages=recent_messages,
            project_snapshot=snapshot.for_model(),
            reference_date=reference_date.isoformat(),
        )
    )
    records_by_id = {record.id: record for record in snapshot.records}
    events_by_id = {event.id: event for event in snapshot.upcoming_events}
    unknown = [
        source_id
        for source_id in answer.source_ids
        if source_id not in records_by_id and source_id not in events_by_id
    ]
    if unknown:
        raise ValueError(f"a project answer cited {len(unknown)} ID(s) outside the project snapshot")
    records = tuple(records_by_id[i] for i in answer.source_ids if i in records_by_id)[:MAX_SOURCE_CARDS]
    events = tuple(events_by_id[i] for i in answer.source_ids if i in events_by_id)[:MAX_SOURCE_CARDS]
    return ProjectAnswerResult(with_stored_dates(answer.reply, records, events), records, events)


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


def _record_for_model(record: ProjectRecord, replaced_by: dict[str, str]) -> dict:
    return {
        "id": record.id,
        "kind": record.kind,
        "state": record.state,
        "content": record.content,
        "rationale": record.rationale,
        "due_at": None if record.due_at is None else record.due_at.isoformat(),
        "supersedes_record_id": record.supersedes_record_id,
        "superseded_by_record_id": replaced_by.get(record.id),
        "updated_on": record.updated_at.date().isoformat(),
    }
