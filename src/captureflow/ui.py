import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from .chat_service import DeterministicChatService
from .local_context_service import LocalContextService, is_local_context_query
from .llm import ModelError, ModelSettings, OpenAICompatibleModel
from .memory_service import MemoryService
from .models import (
    Capture,
    CaptureRun,
    Conversation,
    MAX_CAPTURE_CHARACTERS,
    Project,
    ProjectRecord,
    RunError,
)
from .phase_three import PhaseThreeWorkflow
from .project_service import ProjectTrackerService
from .retrieval import classify_project_query
from .retrieval_service import ProjectRetrievalService
from .router_service import BoundedChatService
from .storage import CaptureStore
from .workflow import configured_reference_date


DIAGNOSTICS_SESSION_KEY = "captureflow_turn_diagnostics"


def database_path() -> Path:
    return Path(os.getenv("CAPTUREFLOW_DATABASE_PATH", "data/captureflow.sqlite3"))


@st.cache_resource
def get_store(path: str) -> CaptureStore:
    store = CaptureStore(path)
    store.initialize()
    return store


def main() -> None:
    st.set_page_config(page_title="CaptureFlow", page_icon="📥")
    st.title("CaptureFlow")
    st.caption("V2 Phase 9.1: bounded project conversation and grounded local recall.")

    store = get_store(str(database_path()))
    diagnostics_enabled = st.sidebar.toggle(
        "Diagnostics",
        value=False,
        help="Show timing for turns submitted while this switch is enabled.",
    )
    page = st.sidebar.radio(
        "Page", ["Chat", "Tracker", "Calendar", "Ideas & Notes", "Memory", "Legacy capture"]
    )
    if page == "Calendar":
        render_calendar_page(store)
        return
    if page == "Ideas & Notes":
        render_saved_items_page(store)
        return
    if page == "Memory":
        render_memory_page(store)
        return
    selected_id = select_conversation(store, show_project_assignment=page != "Tracker")
    if page == "Tracker":
        render_tracker_page(store, selected_id)
        return
    if page == "Chat":
        render_chat_page(store, selected_id, diagnostics_enabled)
        return
    render_legacy_capture_page(store, selected_id)


def select_conversation(store: CaptureStore, *, show_project_assignment: bool = True) -> str:
    conversations = store.list_conversations()
    if not conversations:
        conversation = store.save_conversation(Conversation(title="First capture"))
        conversations = [conversation]

    with st.expander("New conversation"):
        title = st.text_input("Title", key="new_conversation_title")
        projects = store.list_projects(statuses=("active",))
        new_project_id = st.selectbox(
            "Project (optional)",
            options=[None, *[project.id for project in projects]],
            format_func=lambda project_id: (
                "Unassigned" if project_id is None else next(project.name for project in projects if project.id == project_id)
            ),
            key="new-conversation-project",
        )
        if st.button("Create conversation"):
            try:
                created = store.save_conversation(Conversation(title=title))
            except ValidationError as error:
                st.error(error.errors()[0]["msg"])
            else:
                if new_project_id is not None:
                    store.assign_conversation_project(created.id, new_project_id)
                st.session_state["selected_conversation_id"] = created.id
                st.success(f"Created {created.title}.")
                st.rerun()

    conversation_by_id = {conversation.id: conversation for conversation in conversations}
    title_counts = {
        conversation.title: sum(item.title == conversation.title for item in conversations)
        for conversation in conversations
    }
    conversation_ids = list(conversation_by_id)
    if st.session_state.get("selected_conversation_id") not in conversation_by_id:
        st.session_state["selected_conversation_id"] = conversation_ids[0]

    def conversation_label(conversation_id: str) -> str:
        conversation = conversation_by_id[conversation_id]
        if title_counts[conversation.title] == 1:
            return conversation.title
        return f"{conversation.title} · {conversation.id[:8]}"

    selected_id = st.selectbox(
        "Conversation",
        options=conversation_ids,
        format_func=conversation_label,
        key="selected_conversation_id",
    )

    if show_project_assignment:
        with st.expander("Conversation project"):
            current_project = store.get_conversation_project(selected_id)
            if current_project is not None:
                st.caption(f"Assigned project: {current_project.name}")
            else:
                st.caption("Unassigned: local Calendar and Note recall still work; project tracker features do not.")
            projects = store.list_projects(statuses=("active",))
            if not projects:
                st.info("Create a project from Tracker to assign this conversation.")
            elif not store.conversation_has_project_turn(selected_id) and not store.list_project_records_for_conversation(selected_id):
                selected_project_id = st.selectbox(
                    "Project for this conversation",
                    options=[None, *[project.id for project in projects]],
                    index=([None, *[project.id for project in projects]]).index(
                        current_project.id if current_project else None
                    ),
                    format_func=lambda project_id: (
                        "Unassigned" if project_id is None else next(project.name for project in projects if project.id == project_id)
                    ),
                    key=f"conversation-project-{selected_id}",
                )
                if selected_project_id is not None and selected_project_id != (current_project.id if current_project else None):
                    if st.button("Assign project", key=f"assign-conversation-project-{selected_id}"):
                        store.assign_conversation_project(selected_id, selected_project_id)
                        st.rerun()
            else:
                st.caption("Project assignment is locked after the first project turn or tracker record.")

    with st.expander("Conversation settings"):
        st.warning(
            "Deleting this conversation also deletes its captures, runs, tasks, reviews, "
            "checkpoints, and memories derived from those captures."
        )
        confirmed = st.checkbox(
            "I understand this deletes the conversation and its derived data.",
            key=f"confirm-delete-conversation-{selected_id}",
        )
        if st.button(
            "Delete conversation",
            key=f"delete-conversation-{selected_id}",
            disabled=not confirmed,
        ):
            store.delete_conversation(selected_id)
            st.rerun()
    return selected_id


def render_chat_page(
    store: CaptureStore, conversation_id: str, diagnostics_enabled: bool = False
) -> None:
    st.subheader("Chat")
    st.caption(
        "Assigned project conversations use bounded recent context. The model may only propose "
        "source-grounded tracker changes; local validation applies or routes them to review. "
        "Project status and history questions retrieve bounded local project records with visible sources. "
        "`/event`, `/idea`, and `/note` remain deterministic shortcuts."
    )
    local_service = DeterministicChatService(
        store, os.getenv("CAPTUREFLOW_TIMEZONE", "America/Vancouver")
    )
    current_project = store.get_conversation_project(conversation_id)
    st.caption(
        f"Project: {current_project.name}" if current_project is not None else "Project: Unassigned"
    )
    for message in store.list_messages(conversation_id):
        with st.chat_message(message.role):
            st.write(message.content)
            if message.role == "assistant" and message.reply_to_message_id:
                for operation in store.list_tool_operations(message.reply_to_message_id):
                    _render_operation_card(local_service, operation.id)
                for operation in store.list_project_change_operations(message.reply_to_message_id):
                    _render_project_change_card(store, operation.id)
                _render_project_answer_sources(store, message.id)
                _render_local_context_answer_sources(store, message.id)
                if diagnostics_enabled:
                    _render_turn_diagnostics(message.reply_to_message_id)

    if pending := store.get_pending_clarification(conversation_id):
        question = store.get_message(pending.assistant_message_id).content
        st.warning(question)
        if st.button("Cancel pending event", key=f"cancel-clarification-{pending.id}"):
            store.resolve_pending_clarification(pending.id, "cancelled")
            st.rerun()

    if prompt := st.chat_input("Message CaptureFlow"):
        started = perf_counter()
        if prompt.strip().startswith("/"):
            turn = local_service.submit(conversation_id, prompt)
            if diagnostics_enabled:
                _remember_turn_diagnostics(
                    turn.user_message.id,
                    mode="local_shortcut",
                    outcome="committed" if turn.operation else "reply_only",
                    model_calls=0,
                    model_seconds=0,
                    local_seconds=perf_counter() - started,
                    total_seconds=perf_counter() - started,
                    repair_used=False,
                )
        else:
            try:
                timezone = os.getenv("CAPTUREFLOW_TIMEZONE", "America/Vancouver")
                if is_local_context_query(prompt):
                    try:
                        local_context_model = OpenAICompatibleModel(ModelSettings.from_environment())
                    except ModelError:
                        local_context_model = None
                    turn = LocalContextService(
                        store,
                        configured_reference_date(timezone),
                        timezone,
                        model=local_context_model,
                    ).submit(conversation_id, prompt)
                    if diagnostics_enabled:
                        _remember_turn_diagnostics(
                            turn.user_message.id,
                            mode=(
                                "local_context_model"
                                if turn.model_calls and not turn.fallback_used
                                else "local_context_fallback"
                            ),
                            outcome=turn.outcome,
                            model_calls=turn.model_calls,
                            model_seconds=turn.model_seconds,
                            local_seconds=max(0, perf_counter() - started - turn.model_seconds),
                            total_seconds=perf_counter() - started,
                            repair_used=turn.model_calls > 1,
                        )
                else:
                    settings = ModelSettings.from_environment()
                    pending = store.get_pending_clarification(conversation_id)
                    project = store.get_conversation_project(conversation_id)
                    if project is not None and pending is None:
                        retrieval_mode = classify_project_query(prompt)
                        if retrieval_mode is not None:
                            turn = ProjectRetrievalService(
                                store, OpenAICompatibleModel(settings)
                            ).submit(conversation_id, prompt, retrieval_mode)
                            diagnostics_mode = "project_retrieval"
                        else:
                            turn = ProjectTrackerService(
                                store, OpenAICompatibleModel(settings)
                            ).submit(conversation_id, prompt)
                            diagnostics_mode = "project_tracker"
                        if diagnostics_enabled:
                            _remember_turn_diagnostics(
                                turn.user_message.id,
                                mode=diagnostics_mode,
                                outcome=turn.outcome,
                                model_calls=turn.model_calls,
                                model_seconds=turn.model_seconds,
                                local_seconds=max(0, perf_counter() - started - turn.model_seconds),
                                total_seconds=perf_counter() - started,
                                repair_used=False,
                            )
                    else:
                        turn = BoundedChatService(
                            store,
                            OpenAICompatibleModel(settings),
                            configured_reference_date(timezone), timezone
                        ).submit(conversation_id, prompt)
                        if diagnostics_enabled:
                            _remember_turn_diagnostics(
                                turn.user_message.id,
                                mode="model_router",
                                outcome=turn.diagnostics.outcome,
                                model_calls=turn.model_calls,
                                model_seconds=turn.diagnostics.model_seconds,
                                local_seconds=turn.diagnostics.local_seconds,
                                total_seconds=perf_counter() - started,
                                repair_used=turn.diagnostics.repair_used,
                            )
            except (ModelError, ValueError) as error:
                project = store.get_conversation_project(conversation_id)
                if project is not None and store.get_pending_clarification(conversation_id) is None:
                    try:
                        store.begin_project_conversation(conversation_id)
                    except ValueError as project_error:
                        error = ModelError(str(project_error))
                turn = local_service.submit_reply_only(conversation_id, prompt, str(error))
                if diagnostics_enabled:
                    elapsed = perf_counter() - started
                    _remember_turn_diagnostics(
                        turn.user_message.id,
                        mode="configuration_error",
                        outcome="no_model_call",
                        model_calls=0,
                        model_seconds=0,
                        local_seconds=elapsed,
                        total_seconds=elapsed,
                        repair_used=False,
                    )
        st.rerun()


def _remember_turn_diagnostics(message_id: str, **values: object) -> None:
    diagnostics = st.session_state.setdefault(DIAGNOSTICS_SESSION_KEY, {})
    diagnostics[message_id] = values


def _render_turn_diagnostics(message_id: str) -> None:
    diagnostics = st.session_state.get(DIAGNOSTICS_SESSION_KEY, {}).get(message_id)
    if diagnostics is None:
        return
    with st.expander(f"Diagnostics · {message_id[:8]}"):
        st.caption(
            f"Mode: {diagnostics['mode']} · Outcome: {diagnostics['outcome']} · "
            f"Model calls: {diagnostics['model_calls']} · "
            f"Repair: {'yes' if diagnostics['repair_used'] else 'no'}"
        )
        st.caption(
            f"Model: {diagnostics['model_seconds']:.3f}s · "
            f"Local validation/write: {diagnostics['local_seconds']:.3f}s · "
            f"Total submit: {diagnostics['total_seconds']:.3f}s"
        )
        st.caption("Session-only timing; prompts, raw model output, and credentials are not stored.")


def _render_operation_card(service: DeterministicChatService, operation_id: str) -> None:
    operation = service.store.get_tool_operation(operation_id)
    with st.container(border=True):
        st.caption(f"Local action: {operation.tool_name}; status: {operation.status}")
        if operation.status == "committed" and st.button(
            "Undo", key=f"undo-operation-{operation.id}"
        ):
            service.undo(operation.id)
            st.rerun()


def _render_project_change_card(store: CaptureStore, operation_id: str) -> None:
    operation = store.get_project_change_operation(operation_id)
    with st.container(border=True):
        st.caption(
            f"Project change: {operation.change['action'].replace('_', ' ')}; "
            f"status: {operation.status}"
        )
        if operation.status == "committed" and st.button(
            "Undo tracker change", key=f"undo-project-change-{operation.id}"
        ):
            try:
                store.undo_project_change_operation(operation.id)
            except ValueError as error:
                st.error(str(error))
            else:
                st.rerun()
        if operation.status == "review":
            approve, reject = st.columns(2)
            if approve.button("Approve tracker change", key=f"approve-project-change-{operation.id}"):
                try:
                    store.approve_project_change_operation(operation.id)
                except (KeyError, ValueError) as error:
                    st.error(str(error))
                else:
                    st.rerun()
            if reject.button("Reject tracker change", key=f"reject-project-change-{operation.id}"):
                store.reject_project_change_operation(operation.id)
                st.rerun()
        if operation.status == "failed" and operation.error:
            st.caption(f"Not applied: {operation.error}")


def _render_project_answer_sources(store: CaptureStore, assistant_message_id: str) -> None:
    records = store.list_project_answer_sources(assistant_message_id)
    if not records:
        return
    with st.container(border=True):
        st.caption("Project answer sources")
        for record in records:
            st.markdown(f"**{record.kind} · {record.state}:** {record.content}")
            st.caption(f"Source: “{record.source_quote}”")


def _render_local_context_answer_sources(store: CaptureStore, assistant_message_id: str) -> None:
    sources = store.list_local_context_answer_sources(assistant_message_id)
    if not sources:
        return
    with st.container(border=True):
        st.caption("Local context sources")
        for source in sources:
            st.markdown(f"**{source.label}:** {source.content}")
            st.caption(f"Source: “{source.source_quote}”")


def render_tracker_page(store: CaptureStore, conversation_id: str) -> None:
    """Manage Phase 4 project state without a model-driven tracker write."""

    st.subheader("Project Tracker")
    st.caption(
        "Phase 4 is deterministic: records are created from an explicit chat source and "
        "every lifecycle change is retained in local SQLite history."
    )
    current_project = store.get_conversation_project(conversation_id)
    projects = store.list_projects(statuses=("active", "archived"))

    with st.expander("Create project"):
        name = st.text_input("Project name", key="new-project-name")
        description = st.text_area("Description (optional)", key="new-project-description")
        if st.button("Create project", key="create-project"):
            try:
                project = store.save_project(Project(name=name, description=description or None))
            except ValidationError as error:
                st.error(error.errors()[0]["msg"])
            else:
                st.session_state["tracker-project-choice"] = project.id
                st.success(f"Created {project.name}.")
                st.rerun()

    project_ids = [None, *[project.id for project in projects]]
    selected_project_id = st.selectbox(
        "Project for this conversation",
        options=project_ids,
        index=project_ids.index(current_project.id) if current_project else 0,
        format_func=lambda project_id: (
            "Unassigned"
            if project_id is None
            else next(project.name for project in projects if project.id == project_id)
        ),
        key="tracker-project-choice",
    )
    if selected_project_id and (
        current_project is None or selected_project_id != current_project.id
    ):
        if st.button("Assign project", key="assign-conversation-project"):
            try:
                current_project = store.assign_conversation_project(conversation_id, selected_project_id)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"Conversation assigned to {current_project.name}.")
                st.rerun()

    if current_project is None:
        st.info("Create or assign a project before adding tracker records.")
        return

    st.markdown(f"### {current_project.name}")
    if current_project.description:
        st.caption(current_project.description)
    st.caption(f"Project status: {current_project.status}")
    if current_project.status == "active":
        if st.button("Archive project", key=f"archive-project-{current_project.id}"):
            store.set_project_status(current_project.id, "archived")
            st.rerun()
    elif st.button("Restore project", key=f"restore-project-{current_project.id}"):
        store.set_project_status(current_project.id, "active")
        st.rerun()

    user_messages = [
        message for message in store.list_messages(conversation_id) if message.role == "user"
    ]
    if current_project.status == "active":
        with st.expander("Add source-grounded record"):
            if not user_messages:
                st.info("Send a chat message first; Phase 4 records require an explicit user source.")
            else:
                source_id = st.selectbox(
                    "Source user message",
                    options=[message.id for message in user_messages],
                    format_func=lambda message_id: store.get_message(message_id).content,
                    key="tracker-source-message",
                )
                kind = st.selectbox(
                    "Record type",
                    ["task", "milestone", "decision", "constraint", "open_question", "blocker"],
                    key="tracker-record-kind",
                )
                content = st.text_area("Tracker text", key="tracker-record-content")
                source_message = store.get_message(source_id)
                source_quote = st.text_area(
                    "Exact source quote",
                    value=source_message.content,
                    key=f"tracker-source-quote-{source_id}",
                )
                if st.button("Add tracker record", key="add-tracker-record"):
                    initial_state = "completed" if kind == "milestone" else "active"
                    try:
                        store.save_project_record(
                            ProjectRecord(
                                project_id=current_project.id,
                                conversation_id=conversation_id,
                                kind=kind,
                                content=content,
                                state=initial_state,
                                source_message_id=source_id,
                                source_quote=source_quote,
                            )
                        )
                    except (ValidationError, ValueError) as error:
                        st.error(_validation_message(error))
                    else:
                        st.rerun()

    records = store.list_project_records(
        current_project.id,
        states=("active", "completed", "resolved", "superseded", "deleted"),
    )
    active_records = [record for record in records if record.state != "deleted"]
    if not active_records:
        st.info("No active project records yet.")
    for record in active_records:
        _render_project_record(store, record)

    deleted_records = [record for record in records if record.state == "deleted"]
    if deleted_records:
        with st.expander(f"Deleted tracker records ({len(deleted_records)})"):
            for record in deleted_records:
                with st.container(border=True):
                    st.markdown(f"**{record.kind}:** {record.content}")
                    if st.button("Restore record", key=f"restore-project-record-{record.id}"):
                        store.restore_project_record(record.id)
                        st.rerun()


def _render_project_record(store: CaptureStore, record: ProjectRecord) -> None:
    with st.container(border=True):
        st.markdown(f"**{record.kind} · {record.state}**")
        content = st.text_area(
            "Tracker text", value=record.content, key=f"project-record-content-{record.id}", height=80
        )
        rationale = st.text_input(
            "Rationale", value=record.rationale or "", key=f"project-record-rationale-{record.id}"
        )
        if st.button("Save record", key=f"save-project-record-{record.id}"):
            try:
                store.update_project_record(record.id, content=content, rationale=rationale or None)
            except (ValidationError, ValueError) as error:
                st.error(_validation_message(error))
            else:
                st.rerun()
        st.caption(f"Source: {record.source_quote}")
        st.caption(f"Updated {record.updated_at.isoformat()}")
        if record.state == "active":
            transition = {
                "task": ("completed", "Complete task"),
                "open_question": ("resolved", "Resolve question"),
                "blocker": ("resolved", "Resolve blocker"),
                "decision": ("superseded", "Supersede decision"),
                "constraint": ("superseded", "Supersede constraint"),
            }.get(record.kind)
            if transition and st.button(transition[1], key=f"transition-project-record-{record.id}"):
                store.transition_project_record(record.id, transition[0])
                st.rerun()
        if st.button("Delete record", key=f"delete-project-record-{record.id}"):
            store.delete_project_record(record.id)
            st.rerun()
        events = store.list_project_record_events(record.id)
        with st.expander(f"History ({len(events)})"):
            for event in events:
                st.caption(
                    f"{event.created_at.isoformat()} · {event.action} · "
                    f"{event.previous_state or 'none'} → {event.new_state}"
                )


def _validation_message(error: ValidationError | ValueError) -> str:
    if isinstance(error, ValidationError):
        return error.errors()[0]["msg"]
    return str(error)


def render_calendar_page(store: CaptureStore) -> None:
    st.subheader("Calendar")
    st.caption("Local SQLite events only. Editing preserves the original chat source.")
    events = store.list_calendar_events()
    if not events:
        st.info("No active local events.")
    for event in events:
        with st.container(border=True):
            edited_title = st.text_input("Title", value=event.title, key=f"event-title-{event.id}")
            if event.all_day:
                edited_date = st.date_input(
                    "Date", value=event.start_at, key=f"event-date-{event.id}"
                )
                if st.button("Save event", key=f"save-event-{event.id}"):
                    try:
                        store.update_calendar_event(
                            event.id, title=edited_title, start_at=edited_date
                        )
                    except ValidationError as error:
                        st.error(error.errors()[0]["msg"])
                    else:
                        st.rerun()
            else:
                st.caption(f"Timed event: {event.start_at.isoformat()}")
            st.caption(f"Original chat source: {event.source_quote}")
            st.caption(f"Updated {event.updated_at.isoformat()}")
            if st.button("Delete event", key=f"delete-event-{event.id}"):
                store.set_calendar_event_status(event.id, "deleted")
                st.rerun()

    history = store.list_calendar_events(statuses=("deleted",))
    if history:
        with st.expander(f"Calendar history ({len(history)})"):
            for event in history:
                with st.container(border=True):
                    st.write(f"{event.start_at.isoformat()} — {event.title}")
                    st.caption(f"Original chat source: {event.source_quote}")
                    if st.button("Restore event", key=f"restore-event-{event.id}"):
                        store.set_calendar_event_status(event.id, "active")
                        st.rerun()


def render_saved_items_page(store: CaptureStore) -> None:
    st.subheader("Ideas & Notes")
    kind_label = st.selectbox("Type", ["All", "Idea", "Note"])
    query = st.text_input("Filter ideas and notes")
    kinds = {"All": ("idea", "note"), "Idea": ("idea",), "Note": ("note",)}[kind_label]
    items = store.list_saved_items(kinds=kinds)
    if query.strip():
        items = [item for item in items if query.strip().lower() in item.content.lower()]
    if not items:
        st.info("No active ideas or notes match this view.")
    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item.kind.title()}**")
            edited_content = st.text_area(
                "Content", value=item.content, key=f"item-content-{item.id}", height=100
            )
            if st.button("Save item", key=f"save-item-{item.id}"):
                try:
                    store.update_saved_item(item.id, content=edited_content)
                except ValidationError as error:
                    st.error(error.errors()[0]["msg"])
                else:
                    st.rerun()
            st.caption(f"Original chat source: {item.source_quote}")
            st.caption(f"Updated {item.updated_at.isoformat()}")
            if st.button("Delete item", key=f"delete-item-{item.id}"):
                store.set_saved_item_status(item.id, "deleted")
                st.rerun()

    history = store.list_saved_items(kinds=("idea", "note"), statuses=("deleted",))
    if history:
        with st.expander(f"Ideas & Notes history ({len(history)})"):
            for item in history:
                with st.container(border=True):
                    st.write(f"{item.kind.title()}: {item.content}")
                    st.caption(f"Original chat source: {item.source_quote}")
                    if st.button("Restore item", key=f"restore-item-{item.id}"):
                        store.set_saved_item_status(item.id, "active")
                        st.rerun()


def render_legacy_capture_page(store: CaptureStore, selected_id: str) -> None:
    """Keep the completed Phase 1-3 capture workflow available for regression checks."""

    st.subheader("Capture")
    source_text = st.text_area(
        "Paste or write plain text",
        max_chars=MAX_CAPTURE_CHARACTERS,
        height=180,
        placeholder="Analysis uses a configured model only after you select Analyze.",
    )
    if st.button("Save capture", type="primary"):
        try:
            capture = Capture(conversation_id=selected_id, source_text=source_text)
            store.save_capture(capture)
        except ValidationError as error:
            st.error(error.errors()[0]["msg"])
        else:
            st.success("Capture saved locally.")
            st.rerun()

    st.subheader("Saved captures")
    captures = store.list_captures(selected_id)
    if not captures:
        st.info("No captures in this conversation yet.")
        return

    for capture in captures:
        with st.container(border=True):
            st.caption(f"Saved {capture.created_at.isoformat()}")
            st.text(capture.source_text)
            if st.button("Analyze", key=f"analyze-{capture.id}"):
                run = analyze_capture(capture, store)
                store.save_run(run)
                st.rerun()
            render_run(store.latest_run_for_capture(capture.id), store)
            render_run_history(store.list_runs_for_capture(capture.id))
            if st.button("Delete", key=f"delete-{capture.id}"):
                store.delete_capture(capture.id)
                st.rerun()


def analyze_capture(capture: Capture, store: CaptureStore) -> CaptureRun:
    run_id = str(uuid4())
    try:
        settings = ModelSettings.from_environment()
        timezone = os.getenv("CAPTUREFLOW_TIMEZONE", "America/Vancouver")
        reference_date = configured_reference_date(timezone)
        workflow = PhaseThreeWorkflow(OpenAICompatibleModel(settings), MemoryService(store))
        return workflow.run(
            capture_id=capture.id,
            source_text=capture.source_text,
            run_id=run_id,
            reference_date=reference_date,
            timezone=timezone,
        )
    except (ModelError, ValueError) as error:
        return CaptureRun(
            id=run_id,
            capture_id=capture.id,
            status="failed",
            errors=[RunError(stage="configuration", message=str(error))],
            model_calls=0,
        )


def render_run(run: CaptureRun | None, store: CaptureStore) -> None:
    if run is None:
        st.caption("Not analyzed yet.")
        return
    st.caption(f"Latest run: {run.status}; {run.model_calls} model calls")
    if run.plan and run.plan.user_clarification_required:
        st.warning(run.plan.clarification_question)
    elif run.plan:
        with st.expander("Validated plan"):
            for step in run.plan.steps:
                st.write(f"{step.action}: {step.reason}")
    if run.retrieved_memory_ids or run.memory_decisions:
        with st.expander("Memory trace"):
            for memory_id in run.retrieved_memory_ids:
                try:
                    memory = store.get_memory(memory_id)
                except KeyError:
                    st.caption(f"Retrieved {memory_id}: memory is no longer available.")
                else:
                    st.markdown(
                        f"**Retrieved {memory.id[:8]}:** {memory.memory_type} — {memory.content}"
                    )
                    st.caption(f"Status: {memory.status}; full ID: {memory.id}")
            if not run.retrieved_memory_ids:
                st.caption("Retrieved: none")
            for decision in run.memory_decisions:
                proposal = decision.get("proposal", {})
                content = proposal.get("content", "No proposal content")
                st.markdown(f"**Decision — {decision.get('decision', 'unknown')}:** {content}")
                if reason := decision.get("reason"):
                    st.caption(f"Reason: {reason}")
    if run.status in {"completed", "completed_with_omissions", "needs_review"} and run.output:
        if run.status == "needs_review":
            st.info("This run has grounded output that needs user confirmation.")
        for claim in run.output.summary_claims:
            st.markdown(f"**Summary:** {claim.text}")
            st.caption(f"Evidence: {claim.source_quote}")
        for task in run.output.tasks:
            st.markdown(f"**Task candidate:** {task.text}")
            st.caption(f"Evidence: {task.source_quote}")
            if task.due_expression:
                if task.date_status == "ambiguous":
                    st.info(f"Date needs confirmation: {task.due_expression}")
                else:
                    st.caption(
                        f"Date: {task.due_expression} → {task.resolved_due_date or task.date_status}"
                    )
        for note in run.output.review_notes:
            st.info(f"Needs confirmation: {note.question}\n\nOriginal text: {note.source_quote}")
    if run.grounding_report and run.grounding_report.verdict == "revise":
        with st.expander("Grounding feedback"):
            for issue in run.grounding_report.feedback:
                item = issue.item_kind
                if issue.item_index is not None:
                    item = f"{item} {issue.item_index + 1}"
                st.write(f"{item}: {issue.reason}")
    for warning in run.warnings:
        st.warning(warning)
    for error in run.errors:
        st.error(f"{error.stage}: {error.message}")


def render_memory_page(store: CaptureStore) -> None:
    """Render only explicit, reversible memory-management actions."""

    st.subheader("Memory")
    st.caption("Only policy-approved proposals appear here. Automatic writes can be undone.")
    service = MemoryService(store)
    query = st.text_input("Search active memory")
    memories = (
        store.search_memories(query)
        if query.strip()
        else store.list_memories(statuses=("active",))
    )
    if not memories:
        st.info("No active memories match this view.")
    for memory in memories:
        with st.container(border=True):
            st.markdown(f"**{memory.memory_type}:** {memory.content}")
            st.caption(
                f"{memory.status}; {memory.write_mode}; confidence {memory.confidence:.0%}; "
                f"source: {memory.source_quote}"
            )
            lineage = _memory_lineage(store, memory.id)
            if lineage:
                st.caption(lineage)
            edited = st.text_input("Edit memory", value=memory.content, key=f"edit-{memory.id}")
            if st.button("Save edit", key=f"save-edit-{memory.id}"):
                try:
                    service.edit_memory(memory.id, edited)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.rerun()
            if memory.write_mode == "auto" and st.button(
                "Undo automatic write", key=f"undo-{memory.id}"
            ):
                service.undo_auto_memory(memory.id)
                st.rerun()
            if st.button("Delete memory", key=f"delete-memory-{memory.id}"):
                service.delete_memory(memory.id)
                st.rerun()

    history = store.list_memories(statuses=("superseded", "deleted"))
    if history:
        with st.expander(f"Memory history ({len(history)})"):
            for memory in history:
                with st.container(border=True):
                    st.markdown(f"**{memory.memory_type}:** {memory.content}")
                    st.caption(
                        f"{memory.status}; {memory.write_mode}; source: {memory.source_quote}"
                    )
                    lineage = _memory_lineage(store, memory.id)
                    if lineage:
                        st.caption(lineage)
                    if memory.status == "deleted" and st.button(
                        "Restore memory", key=f"restore-{memory.id}"
                    ):
                        service.restore_memory(memory.id)
                        st.rerun()

    reviews = store.list_pending_reviews()
    if reviews:
        st.subheader("Memory review")
    for review in reviews:
        with st.container(border=True):
            proposal = review.proposal
            st.markdown(f"**{proposal.memory_type}:** {proposal.content}")
            st.caption(f"Reason: {review.decision_reason}. Source: {proposal.source_quote}")
            revised = st.text_input("Review text", value=proposal.content, key=f"review-text-{review.id}")
            supersede = False
            if review.conflicting_memory_ids:
                supersede = st.checkbox(
                    "Supersede conflicting active memories", key=f"supersede-{review.id}"
                )
            if st.button("Approve memory", key=f"approve-{review.id}"):
                service.approve_review(review.id, revised, supersede)
                st.rerun()
            if st.button("Reject memory", key=f"reject-{review.id}"):
                service.reject_review(review.id)
                st.rerun()


def _memory_lineage(store: CaptureStore, memory_id: str) -> str | None:
    for event in store.list_memory_events(memory_id):
        if replaced := event.details.get("replacement_memory_id"):
            return f"Replaced by memory {replaced}."
        if previous := event.details.get("replaces_memory_id"):
            return f"Replaces memory {previous}."
    return None


def render_run_history(runs: list[CaptureRun]) -> None:
    if not runs:
        return
    with st.expander(f"Run history ({len(runs)})"):
        for run in runs:
            st.write(f"{run.created_at.isoformat()} — {run.status}; {run.model_calls} model calls")
            if run.grounding_report and run.grounding_report.verdict == "revise":
                for issue in run.grounding_report.feedback:
                    st.caption(f"Grounding: {issue.item_kind}: {issue.reason}")
            for error in run.errors:
                st.caption(f"{error.stage}: {error.message}")
            for warning in run.warnings:
                st.caption(f"Warning: {warning}")
