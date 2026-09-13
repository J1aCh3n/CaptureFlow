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
    router = ConversationRouterService(
        store,
        OpenAICompatibleModel(settings),
        date.fromisoformat(protocol["reference_date"]),
        protocol["timezone"],
    )
    inputs = {case["name"]: case["input"] for case in fixture["cases"]}

    created = _submit(
        trace, "calendar_create", router, main_conversation.id, inputs["calendar_create"], "phase10-create"
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
        trace, "calendar_update", router, main_conversation.id, inputs["calendar_update"], "phase10-update"
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
        trace, "tracker_change", router, main_conversation.id, inputs["tracker_change"], "phase10-tracker"
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

    meeting_answer = _submit(
        trace, "meeting_question", router, main_conversation.id, inputs["meeting_question"], "phase10-question"
    )
    _check(
        trace,
        "meeting_question_effect",
        meeting_answer is not None
        and meeting_answer.decision is not None
        and meeting_answer.decision.intent == "local_answer"
        and not meeting_answer.operations
        and bool(store.list_local_context_answer_sources(meeting_answer.assistant_message.id)),
    )

    replay = _submit(
        trace,
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
    later_router = ConversationRouterService(
        store,
        OpenAICompatibleModel(settings),
        date.fromisoformat(protocol["reference_date"]),
        protocol["timezone"],
    )
    cross_local = _submit(
        trace,
        "cross_conversation_local_retrieval",
        later_router,
        later_conversation.id,
        inputs["cross_conversation_local_retrieval"],
        "phase10-cross-local",
    )
    _check(
        trace,
        "cross_conversation_local_retrieval_effect",
        cross_local is not None
        and cross_local.decision is not None
        and cross_local.decision.intent == "local_answer"
        and not cross_local.operations
        and bool(store.list_local_context_answer_sources(cross_local.assistant_message.id)),
    )

    cross_project = _submit(
        trace,
        "cross_conversation_project_retrieval",
        later_router,
        later_conversation.id,
        inputs["cross_conversation_project_retrieval"],
        "phase10-cross-project",
    )
    _check(
        trace,
        "cross_conversation_project_retrieval_effect",
        cross_project is not None
        and cross_project.outcome == "project_answer"
        and not cross_project.operations
        and bool(store.list_project_answer_sources(cross_project.assistant_message.id)),
    )

    ambiguous = _submit(
        trace,
        "ambiguous_correction",
        later_router,
        later_conversation.id,
        inputs["ambiguous_correction"],
        "phase10-ambiguous",
    )
    _check(
        trace,
        "ambiguous_correction_effect",
        ambiguous is not None
        and ambiguous.outcome == "clarification_created"
        and not ambiguous.operations
        and store.get_pending_clarification(later_conversation.id) is not None,
    )

    trace["finished_at"] = datetime.now(timezone.utc).isoformat()
    trace["passed"] = all(case["status"] == "passed" for case in trace["cases"])
    (args.output_dir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if trace["passed"] else 1


def _conversation(store: CaptureStore, conversation_id: str, project_id: str) -> Conversation:
    conversation = store.save_conversation(Conversation(id=conversation_id, title=conversation_id))
    store.assign_conversation_project(conversation.id, project_id)
    return conversation


def _submit(
    trace: dict, name: str, service: ConversationRouterService, conversation_id: str, content: str, message_id: str
):
    started = perf_counter()
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
            "local_seconds": turn.local_seconds,
            "candidate_counts": {
                "events": len(turn.candidates.events),
                "saved_items": len(turn.candidates.saved_items),
                "project_records": len(turn.candidates.project_records),
            },
            "operation_ids": [operation.id for operation in turn.operations],
            "operation_kinds": [_operation_kind(operation) for operation in turn.operations],
            "assistant_message_id": turn.assistant_message.id,
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
