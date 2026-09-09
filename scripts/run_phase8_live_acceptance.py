"""Run the frozen Phase 8 sequence with one configured live model after approval."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from captureflow.llm import ModelSettings, OpenAICompatibleModel
from captureflow.models import Conversation, Message, Project, ProjectChangeOperation, ProjectRecord
from captureflow.project_service import ProjectTrackerService
from captureflow.retrieval_service import ProjectRetrievalService
from captureflow.storage import CaptureStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("output directory must not already exist")
    args.output_dir.mkdir(parents=True)

    fixture = json.loads(
        (Path(__file__).parents[1] / "tests" / "fixtures" / "phase8_tracker_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    settings = ModelSettings.from_environment()
    store = CaptureStore(args.output_dir / "captureflow.sqlite3")
    store.initialize()
    model = OpenAICompatibleModel(settings)
    trace = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "base_url": settings.base_url,
        "timeout_seconds": settings.timeout_seconds,
        "timezone": fixture["protocol"]["timezone"],
        "database_seed": fixture["protocol"]["database_seed"],
        "commit": _git_revision(),
        "working_tree_dirty": _working_tree_dirty(),
        "source_fingerprints": _source_fingerprints(),
        "fixture": fixture,
        "cases": [],
    }
    project = store.save_project(Project(id="phase-eight-live", name="CaptureFlow"))
    main_conversation = _conversation(store, "phase-eight-live-main", project.id)
    tracker = ProjectTrackerService(store, model)
    inputs = {case["name"]: case for case in fixture["cases"]}

    task = _run_tracker(trace, "task_create", tracker, main_conversation.id, inputs["task_create_and_follow_up_completion"]["input"])
    completed = _run_tracker(trace, "task_complete", tracker, main_conversation.id, inputs["task_create_and_follow_up_completion"]["follow_up"])
    _run(trace, "assert_task_completion", lambda: _assert_operation(task, "create_record", "committed"))
    _run(trace, "assert_task_follow_up", lambda: _assert_operation(completed, "transition_record", "committed"))
    decision = _run_tracker(trace, "decision_create", tracker, main_conversation.id, inputs["decision_rationale_and_supersession"]["input"])
    _run(trace, "assert_decision_rationale", lambda: _assert_rationale(decision))
    supersede = _run_tracker(trace, "decision_supersede", tracker, main_conversation.id, inputs["decision_rationale_and_supersession"]["follow_up"])
    _run(trace, "assert_decision_review", lambda: _assert(
        any(operation.status == "review" for operation in getattr(supersede, "operations", ())),
        "decision replacement did not enter review",
    ))
    if supersede is not None:
        for operation in supersede.operations:
            if operation.status == "review":
                approved = _run(trace, "approve_decision_supersession", lambda: store.approve_project_change_operation(operation.id))
                _run(trace, "assert_decision_supersession", lambda: _assert(getattr(approved, "status", None) == "committed", "decision supersession was not committed"))
    _run_tracker(trace, "blocker_create", tracker, main_conversation.id, inputs["blocker_create_and_resolution"]["input"])
    resolved = _run_tracker(trace, "blocker_resolve", tracker, main_conversation.id, inputs["blocker_create_and_resolution"]["follow_up"])
    _run(trace, "assert_blocker_resolution", lambda: _assert_operation(resolved, "transition_record", "committed"))

    def correct_inaccurate_record():
        source = store.save_message(
            Message(
                id="phase-eight-live-inaccurate-source",
                conversation_id=main_conversation.id,
                role="user",
                content=inputs["deterministic_inaccurate_record_correction"]["input"],
            )
        )
        record = store.save_project_record(
            ProjectRecord(
                id="phase-eight-live-inaccurate",
                project_id=project.id,
                conversation_id=main_conversation.id,
                kind="task",
                content=source.content,
                state="active",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        return store.delete_project_record(record.id)

    deleted = _run(trace, "deterministic_inaccurate_record_correction", correct_inaccurate_record)
    _run(trace, "assert_inaccurate_record_deleted", lambda: _assert(getattr(deleted, "state", None) == "deleted", "inaccurate record was not deleted"))
    casual = _run_tracker(trace, "casual_turn", tracker, main_conversation.id, inputs["casual_turn_no_record"]["input"])
    _run(trace, "assert_casual_no_record", lambda: _assert(getattr(casual, "outcome", None) == "no_change", "casual turn changed the tracker"))

    later_conversation = _conversation(store, "phase-eight-live-later", project.id)
    retrieval = ProjectRetrievalService(store, model)
    status = _run(
        trace,
        "cross_conversation_current_status",
        lambda: retrieval.submit(later_conversation.id, inputs["cross_conversation_current_status"]["input"], "current"),
    )
    history = _run(
        trace,
        "cross_conversation_history",
        lambda: retrieval.submit(later_conversation.id, inputs["cross_conversation_history"]["input"], "history"),
    )
    _run(trace, "assert_status_sources", lambda: _assert(bool(getattr(status, "records", ())), "status answer had no source records"))
    _run(trace, "assert_history_sources", lambda: _assert(bool(getattr(history, "records", ())), "history answer had no source records"))

    other_project = store.save_project(Project(id="phase-eight-live-other", name="Other"))
    other_conversation = _conversation(store, "phase-eight-live-other-chat", other_project.id)
    other_source = store.save_message(
        Message(id="phase-eight-live-other-source", conversation_id=other_conversation.id, role="user", content="Other project task.")
    )
    store.save_project_record(
        ProjectRecord(
            id="phase-eight-live-other-task",
            project_id=other_project.id,
            conversation_id=other_conversation.id,
            kind="task",
            content=other_source.content,
            state="active",
            source_message_id=other_source.id,
            source_quote=other_source.content,
        )
    )
    other_status = _run(
        trace,
        "two_project_isolation",
        lambda: ProjectRetrievalService(store, model).submit(
            other_conversation.id, inputs["two_project_isolation"]["input"], "current"
        ),
    )
    _run(trace, "assert_project_isolation", lambda: _assert(
        all(record.project_id == other_project.id for record in getattr(other_status, "records", ())),
        "other-project answer included a record from another project",
    ))

    def restart_prepared_transition():
        source = store.save_message(
            Message(
                id="phase-eight-live-prepared-source",
                conversation_id=main_conversation.id,
                role="user",
                content=inputs["restart_prepared_transition"]["input"],
            )
        )
        operation = store.save_project_change_operation(
            ProjectChangeOperation(
                id=str(uuid5(NAMESPACE_URL, f"phase-eight-live:{source.id}")),
                source_message_id=source.id,
                project_id=project.id,
                change_index=0,
                change={
                    "action": "create_record",
                    "kind": "open_question",
                    "content": "What is the next acceptance step?",
                    "source_quote": source.content,
                },
                status="prepared",
            )
        )
        restarted = CaptureStore(store.database_path)
        restarted.initialize()
        return ProjectTrackerService(restarted, model).submit(
            main_conversation.id, source.content, source.id
        ), restarted.get_project_change_operation(operation.id)

    resumed = _run(trace, "restart_prepared_transition", restart_prepared_transition)
    _run(trace, "assert_prepared_resume", lambda: _assert(
        resumed is not None and resumed[0].outcome == "resumed_prepared_change" and resumed[1].status == "committed",
        "prepared project change did not resume safely",
    ))
    trace["finished_at"] = datetime.now(timezone.utc).isoformat()
    trace["project_records"] = [record.model_dump(mode="json") for record in store.list_project_records(project.id)]
    (args.output_dir / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if any("error" in case for case in trace["cases"]) else 0


def _conversation(store: CaptureStore, conversation_id: str, project_id: str) -> Conversation:
    conversation = store.save_conversation(Conversation(id=conversation_id, title=conversation_id))
    store.assign_conversation_project(conversation.id, project_id)
    return conversation


def _run_tracker(trace: dict, name: str, service: ProjectTrackerService, conversation_id: str, content: str):
    return _run(trace, name, lambda: service.submit(conversation_id, content))


def _run(trace: dict, name: str, callback):
    try:
        result = callback()
    except Exception as error:
        trace["cases"].append({"name": name, "error": str(error)})
        return None
    trace["cases"].append({"name": name, "result": _serialize(result)})
    return result


def _serialize(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return {key: _serialize(item) for key, item in value.__dict__.items()}
    if isinstance(value, tuple | list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _assert(condition: bool, message: str) -> str:
    if not condition:
        raise AssertionError(message)
    return "passed"


def _assert_operation(turn, action: str, status: str) -> str:
    operations = getattr(turn, "operations", ())
    return _assert(
        any(operation.change["action"] == action and operation.status == status for operation in operations),
        f"expected {action}:{status} operation",
    )


def _assert_rationale(turn) -> str:
    operations = getattr(turn, "operations", ())
    return _assert(
        any(operation.change.get("rationale") for operation in operations),
        "decision rationale was not persisted in the proposed change",
    )


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


def _source_fingerprints() -> dict[str, str]:
    root = Path(__file__).parents[1]
    paths = (
        "src/captureflow/llm.py",
        "src/captureflow/project_service.py",
        "src/captureflow/retrieval_service.py",
        "tests/fixtures/phase8_tracker_acceptance.json",
    )
    return {
        relative_path: hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        for relative_path in paths
    }


if __name__ == "__main__":
    raise SystemExit(main())
