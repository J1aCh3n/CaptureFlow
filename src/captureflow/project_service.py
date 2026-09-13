"""Validate model-proposed tracker changes and commit the ones that need no review."""

from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from .credentials import contains_credential_disclosure
from .models import Message, Project, ProjectChange, ProjectChangeOperation, ProjectRecord
from .storage import CaptureStore

# The only lifecycle transitions a chat turn may apply directly.
_ALLOWED_TRANSITIONS = {
    ("task", "completed"),
    ("open_question", "resolved"),
    ("blocker", "resolved"),
}


def apply_validated_changes(
    store: CaptureStore,
    project: Project,
    user_message: Message,
    changes: list[ProjectChange],
    candidates: list[ProjectRecord],
) -> tuple[list[ProjectChangeOperation], list[ProjectChangeOperation]]:
    """Return (committed or failed, awaiting review) operations for one turn's changes.

    Every change is checked before any is saved, so one invalid change leaves the
    tracker untouched. Replacing a decision or constraint is saved for review.
    """

    committed: list[ProjectChangeOperation] = []
    reviewed: list[ProjectChangeOperation] = []
    for operation in _prepare_operations(project, user_message, changes, candidates):
        saved = store.save_project_change_operation(operation)
        if saved.status == "review":
            reviewed.append(saved)
            continue
        try:
            committed.append(store.commit_prepared_project_change(saved.id))
        except (KeyError, ValidationError, ValueError) as error:
            committed.append(store.fail_prepared_project_change(saved.id, str(error)))
    return committed, reviewed


def _prepare_operations(
    project: Project,
    user_message: Message,
    changes: list[ProjectChange],
    candidates: list[ProjectRecord],
) -> list[ProjectChangeOperation]:
    candidate_by_id = {record.id: record for record in candidates}
    active_by_kind_content = {(record.kind, _normalized(record.content)) for record in candidates}
    signatures: set[tuple[str, str, str]] = set()
    review_indexes: set[int] = set()
    for index, change in enumerate(changes):
        _validate_change_source(change, user_message)
        signature = (change.action, change.target_record_id or "", _normalized(change.content or ""))
        if signature in signatures:
            raise ValueError("duplicate project changes in the same turn are not allowed")
        signatures.add(signature)

        target = candidate_by_id.get(change.target_record_id or "")
        if change.action == "create_record":
            if (change.kind, _normalized(change.content or "")) in active_by_kind_content:
                raise ValueError("an identical active project record already exists")
            if change.kind in {"decision", "constraint"} and any(
                record.kind == change.kind for record in candidates
            ):
                review_indexes.add(index)
        elif change.action == "transition_record":
            if target is None:
                raise ValueError("project change target was not supplied as an active candidate")
            if (target.kind, change.target_state) not in _ALLOWED_TRANSITIONS:
                raise ValueError("invalid project record lifecycle transition")
        elif change.action == "set_deadline":
            if target is None or target.kind != "task":
                raise ValueError("a deadline must target a supplied active task")
        else:
            if target is None or target.kind != change.kind or target.kind not in {"decision", "constraint"}:
                raise ValueError("a supersession must target a supplied active record of the same kind")
            review_indexes.add(index)

    return [
        ProjectChangeOperation(
            id=str(
                uuid5(NAMESPACE_URL, f"captureflow:project-change:{user_message.id}:{index}:{change.action}")
            ),
            source_message_id=user_message.id,
            project_id=project.id,
            change_index=index,
            change=change.model_dump(mode="json", exclude_none=True),
            target_record_id=change.target_record_id,
            status="review" if index in review_indexes else "prepared",
        )
        for index, change in enumerate(changes)
    ]


def _validate_change_source(change: ProjectChange, user_message: Message) -> None:
    if _normalized(change.source_quote) not in _normalized(user_message.content):
        raise ValueError("a project change source quote must occur in the current user message")
    if contains_credential_disclosure(change.source_quote) or (
        change.content is not None and contains_credential_disclosure(change.content)
    ):
        raise ValueError("credentials cannot be persisted in a project change")


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()
