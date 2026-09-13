"""Run the frozen Phase 10 contextual-router sequence with one approved live model."""

import argparse
import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter

from captureflow.conversation_router_service import ConversationRouterService
from captureflow.llm import ModelSettings, OpenAICompatibleModel
from captureflow.models import CalendarEventUpdateOperation, Conversation, Project, ProjectChangeOperation
from captureflow.storage import CaptureStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-model-override",
        action="store_true",
        help="run with CAPTUREFLOW_MODEL even when it differs from the frozen protocol model",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")
    args.output_dir.mkdir(parents=True)

    root = Path(__file__).parents[1]
    fixture = json.loads(
        (root / "tests" / "fixtures" / "phase10_live_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    settings = ModelSettings.from_environment()
    protocol = fixture["protocol"]
    model_overridden = _validate_model_selection(
        settings.model, protocol["model"], args.allow_model_override
    )

    store = CaptureStore(args.output_dir / "captureflow.sqlite3")
    store.initialize()
    trace = _trace_header(root, fixture, settings, model_overridden)
    project = store.save_project(Project(id="phase-ten-live-project", name="CaptureFlow"))
    main_conversation = _conversation(store, "phase-ten-live-main", project.id)
    model = OpenAICompatibleModel(settings)
    tokens = TokenCounter(model)
    router = ConversationRouterService(
        store, model, date.fromisoformat(protocol["reference_date"]), protocol["timezone"]
    )
    inputs = {case["name"]: case["input"] for case in fixture["cases"]}

    created = _submit(
        trace, tokens, "calendar_create", router, main_conversation.id, inputs["calendar_create"], "phase10-create"
    )
    _check(
        trace,
        "calendar_create_effect",
        created is not None
        and created.outcome == "calendar_create"
        and _has_tool_operation(created, "create_local_event")
        and len(store.list_calendar_events()) == 1,
    )

    updated = _submit(
        trace, tokens, "calendar_update", router, main_conversation.id, inputs["calendar_update"], "phase10-update"
    )
    event = store.list_calendar_events()[0] if store.list_calendar_events() else None
    _check(
        trace,
        "calendar_update_effect",
        updated is not None
        and updated.outcome == "calendar_update"
        and any(isinstance(operation, CalendarEventUpdateOperation) for operation in updated.operations)
        and event is not None
        and isinstance(event.start_at, datetime)
        and event.start_at.hour == 14
        and event.end_at is not None
        and event.end_at.hour == 15,
    )

    tracked = _submit(
        trace, tokens, "tracker_change", router, main_conversation.id, inputs["tracker_change"], "phase10-tracker"
    )
    _check(
        trace,
        "tracker_change_effect",
        tracked is not None
        and tracked.outcome == "tracker_change"
        and any(
            isinstance(operation, ProjectChangeOperation)
            and operation.change["action"] == "create_record"
            and operation.status == "committed"
            for operation in tracked.operations
        )
        and len(store.list_project_records(project.id)) == 1,
    )
    tracked_records = store.list_project_records(project.id)
    tracked_record_id = tracked_records[0].id if len(tracked_records) == 1 else None

    meeting_answer = _submit(
        trace, tokens, "meeting_question", router, main_conversation.id, inputs["meeting_question"], "phase10-question"
    )
    _check(
        trace,
        "meeting_question_effect",
        answered_from_stored_event(store, meeting_answer, None if event is None else event.id),
    )

    replay = _submit(
        trace,
        tokens,
        "replay_idempotency",
        router,
        main_conversation.id,
        inputs["calendar_create"],
        "phase10-create",
    )
    _check(
        trace,
        "replay_idempotency_effect",
        replay is not None
        and replay.outcome == "reloaded"
        and replay.model_calls == 0
        and len(store.list_calendar_events()) == 1,
    )

    later_conversation = _conversation(store, "phase-ten-live-later", project.id)
    cross_local = _submit(
        trace,
        tokens,
        "cross_conversation_local_retrieval",
        router,
        later_conversation.id,
        inputs["cross_conversation_local_retrieval"],
        "phase10-cross-local",
    )
    _check(
        trace,
        "cross_conversation_local_retrieval_effect",
        answered_from_stored_event(store, cross_local, None if event is None else event.id),
    )

    cross_project = _submit(
        trace,
        tokens,
        "cross_conversation_project_retrieval",
        router,
        later_conversation.id,
        inputs["cross_conversation_project_retrieval"],
        "phase10-cross-project",
    )
    _check(
        trace,
        "cross_conversation_project_retrieval_effect",
        answered_from_project_record(store, cross_project, tracked_record_id),
    )

    # Found, so act: this conversation just asked about the meeting.
    found = _submit(
        trace,
        tokens,
        "target_found_acts",
        router,
        later_conversation.id,
        inputs["target_found_acts"],
        "phase10-found",
    )
    _check(
        trace,
        "target_found_acts_effect",
        event is not None and moved_event(store, found, event.id, hour=15),
    )

    # Not found, so ask, then complete with the event the user names.
    unmentioned = _conversation(store, "phase-ten-live-unmentioned", project.id)
    before_question = None if event is None else store.get_calendar_event(event.id)
    asked = _submit(
        trace,
        tokens,
        "target_not_found_asks",
        router,
        unmentioned.id,
        inputs["target_not_found_asks"],
        "phase10-not-found",
    )
    _check(
        trace,
        "target_not_found_asks_effect",
        before_question is not None and asked_without_writing(store, asked, before_question),
    )
    named = _submit(
        trace,
        tokens,
        "target_named_completes",
        router,
        unmentioned.id,
        inputs["target_named_completes"],
        "phase10-named",
    )
    _check(
        trace,
        "target_named_completes_effect",
        event is not None and moved_event(store, named, event.id, day=date(2026, 9, 17)),
    )

    trace["finished_at"] = datetime.now(timezone.utc).isoformat()
    trace["passed"] = all(case["status"] == "passed" for case in trace["cases"])
    (args.output_dir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if trace["passed"] else 1


def answered_from_stored_event(store: CaptureStore, turn, event_id: str | None) -> bool:
    """A meeting question passes when the local lookup answered with that event's card.

    The check is the effect a user sees, not the intent label: nothing was written,
    the deterministic lookup wrote the reply, and the created event is cited.
    """

    return (
        turn is not None
        and event_id is not None
        and turn.outcome == "grounded_local_context"
        and not turn.operations
        and event_id
        in {source.reference.source_id for source in store.list_local_context_answer_sources(turn.assistant_message.id)}
    )


def answered_from_project_record(store: CaptureStore, turn, record_id: str | None) -> bool:
    """A project question passes when the project answer wrote nothing and cites the tracked task."""

    return (
        turn is not None
        and record_id is not None
        and turn.outcome == "project_answer"
        and not turn.operations
        and record_id in {record.id for record in store.list_project_answer_sources(turn.assistant_message.id)}
    )


def moved_event(
    store: CaptureStore, turn, event_id: str, *, day: date | None = None, hour: int | None = None
) -> bool:
    """An update passes when it committed and the stored event now starts at the stated day or hour."""

    if turn is None or turn.outcome != "calendar_update":
        return False
    if not any(
        isinstance(operation, CalendarEventUpdateOperation) and operation.status == "committed"
        for operation in turn.operations
    ):
        return False
    event = store.get_calendar_event(event_id)
    return (day is None or event.start_date == day) and (
        hour is None or (isinstance(event.start_at, datetime) and event.start_at.hour == hour)
    )


def asked_without_writing(store: CaptureStore, turn, event_before) -> bool:
    """A question passes when the turn asked, wrote nothing, and left the event as it was."""

    return (
        turn is not None
        and turn.outcome == "clarification_created"
        and not turn.operations
        and store.get_calendar_event(event_before.id) == event_before
    )


class TokenCounter:
    """Record each model request's token counts; the prompt and output are not kept."""

    def __init__(self, model: OpenAICompatibleModel) -> None:
        self.requests: list[tuple[int | None, int | None]] = []
        completions = model.client.chat.completions
        create = completions.create

        def counted_create(**kwargs):
            response = create(**kwargs)
            usage = getattr(response, "usage", None)
            self.requests.append(
                (getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))
            )
            return response

        completions.create = counted_create

    def since(self, start: int) -> dict[str, int | None]:
        requests = self.requests[start:]
        return {
            "input_tokens": _total(count for count, _ in requests),
            "output_tokens": _total(count for _, count in requests),
        }


def _total(counts) -> int | None:
    counts = list(counts)
    return None if not counts or None in counts else sum(counts)


def _conversation(store: CaptureStore, conversation_id: str, project_id: str) -> Conversation:
    conversation = store.save_conversation(Conversation(id=conversation_id, title=conversation_id))
    store.assign_conversation_project(conversation.id, project_id)
    return conversation


def _submit(
    trace: dict,
    tokens: TokenCounter,
    name: str,
    service: ConversationRouterService,
    conversation_id: str,
    content: str,
    message_id: str,
):
    started = perf_counter()
    first_request = len(tokens.requests)
    try:
        turn = service.submit(conversation_id, content, message_id)
    except Exception as error:
        trace["cases"].append(
            {"name": name, "status": "failed", "error_type": type(error).__name__, "elapsed_seconds": perf_counter() - started}
        )
        return None
    trace["cases"].append(
        {
            "name": name,
            "status": "passed",
            "elapsed_seconds": perf_counter() - started,
            "outcome": turn.outcome,
            "intent": None if turn.decision is None else turn.decision.intent,
            "model_calls": turn.model_calls,
            "model_seconds": turn.model_seconds,
            **tokens.since(first_request),
            "local_seconds": turn.local_seconds,
            "candidate_counts": {
                "events": len(turn.candidates.events),
                "saved_items": len(turn.candidates.saved_items),
                "project_records": len(turn.candidates.project_records),
            },
            "operation_ids": [operation.id for operation in turn.operations],
            "operation_kinds": [_operation_kind(operation) for operation in turn.operations],
            "assistant_message_id": turn.assistant_message.id,
            "failure_reason": turn.failure_reason,
        }
    )
    return turn


def _check(trace: dict, name: str, condition: bool) -> None:
    trace["cases"].append({"name": name, "status": "passed" if condition else "failed"})


def _has_tool_operation(turn, tool_name: str) -> bool:
    return any(getattr(operation, "tool_name", None) == tool_name for operation in turn.operations)


def _operation_kind(operation) -> str:
    if isinstance(operation, CalendarEventUpdateOperation):
        return "calendar_update"
    if isinstance(operation, ProjectChangeOperation):
        return f"project:{operation.change['action']}"
    return getattr(operation, "tool_name", type(operation).__name__)


def _validate_model_selection(
    configured_model: str, protocol_model: str, allow_override: bool
) -> bool:
    model_overridden = configured_model != protocol_model
    if model_overridden and not allow_override:
        raise SystemExit(
            f"expected CAPTUREFLOW_MODEL={protocol_model}; "
            "pass --allow-model-override for a cross-model run"
        )
    return model_overridden


def _trace_header(
    root: Path, fixture: dict, settings: ModelSettings, model_overridden: bool
) -> dict:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "protocol_model": fixture["protocol"]["model"],
        "protocol_version": fixture["protocol"]["version"],
        "model_overridden": model_overridden,
        "endpoint_family": "configured_openai_compatible" if settings.base_url else "openai_default",
        "timeout_seconds": settings.timeout_seconds,
        "timezone": fixture["protocol"]["timezone"],
        "database_seed": fixture["protocol"]["database_seed"],
        "commit": _git_revision(),
        "working_tree_dirty": _working_tree_dirty(),
        "fixture_sha256": hashlib.sha256(
            (root / "tests" / "fixtures" / "phase10_live_acceptance.json").read_bytes()
        ).hexdigest(),
        "source_fingerprints": {
            relative_path: hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
            for relative_path in (
                "src/captureflow/conversation_router_service.py",
                "src/captureflow/llm.py",
                "src/captureflow/ui.py",
                "scripts/run_phase10_live_acceptance.py",
            )
        },
        "cases": [],
    }


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _working_tree_dirty() -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
