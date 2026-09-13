import os
from datetime import date, datetime
from html import escape
from pathlib import Path
from time import perf_counter

import streamlit as st
from pydantic import ValidationError

from .chat_service import DeterministicChatService
from .conversation_router_service import ConversationRouterService
from .llm import ModelError, ModelSettings, OpenAICompatibleModel
from .models import CalendarEvent, Conversation, Project, ProjectRecord
from .storage import CaptureStore
from .time_config import configured_reference_date, configured_reference_now, configured_timezone


DIAGNOSTICS_SESSION_KEY = "captureflow_turn_diagnostics"


def _inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cf-primary: #2563eb;
            --cf-primary-soft: #eaf2ff;
            --cf-surface: #ffffff;
            --cf-surface-muted: #f7f9fc;
            --cf-border: #e3e9f2;
            --cf-text-muted: #64748b;
        }
        .stApp {
            background: var(--cf-surface-muted);
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1540px;
            padding-top: 1.75rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: #fbfcfe;
            border-right: 1px solid var(--cf-border);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--cf-text-muted);
        }
        [data-testid="stSidebar"] [role="radiogroup"][aria-label="Conversations"] {
            border-left: 1px solid var(--cf-border);
            margin-left: 0.55rem;
            padding-left: 0.8rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--cf-surface);
            border-color: var(--cf-border);
            border-radius: 14px;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.035);
        }
        div[data-testid="stChatMessage"] {
            background: var(--cf-surface);
            border: 1px solid var(--cf-border);
            border-radius: 14px;
            margin-bottom: 0.7rem;
        }
        [data-testid="stChatInput"] {
            border-color: #cbd5e1;
            border-radius: 12px;
            background: var(--cf-surface);
            box-shadow: 0 8px 26px rgba(15, 23, 42, 0.12);
        }
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        [data-baseweb="select"] > div {
            background: #ffffff;
            border: 1px solid #cbd5e1;
        }
        [data-testid="stTextInputRootElement"],
        [data-testid="stTextAreaRootElement"] {
            background: #ffffff;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px;
        }
        [data-baseweb="input"] > div:focus-within,
        [data-baseweb="textarea"] > div:focus-within,
        [data-baseweb="select"] > div:focus-within,
        [data-testid="stTextInputRootElement"]:focus-within,
        [data-testid="stTextAreaRootElement"]:focus-within {
            border-color: var(--cf-primary);
            box-shadow: 0 0 0 1px var(--cf-primary);
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid var(--cf-border);
            border-radius: 12px;
            padding: 0.75rem;
        }
        .stButton > button {
            border-radius: 10px;
        }
        .cf-sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin: 0.2rem 0 1.25rem;
        }
        .cf-sidebar-mark {
            display: grid;
            place-items: center;
            width: 2.4rem;
            height: 2.4rem;
            border-radius: 12px;
            color: white;
            background: linear-gradient(145deg, #2563eb, #38bdf8);
            font-weight: 750;
        }
        .cf-sidebar-name {
            color: #0f172a;
            font-size: 1.08rem;
            font-weight: 720;
            line-height: 1.15;
        }
        .cf-sidebar-tagline {
            color: var(--cf-text-muted);
            font-size: 0.76rem;
            margin-top: 0.18rem;
        }
        .cf-workspace-header {
            background: linear-gradient(135deg, #ffffff 0%, #f4f8ff 100%);
            border: 1px solid var(--cf-border);
            border-radius: 14px;
            padding: 1rem 1.15rem;
            margin: 0.25rem 0 1rem;
        }
        .cf-workspace-kicker {
            color: var(--cf-primary);
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .cf-workspace-title {
            color: #0f172a;
            font-size: 1.25rem;
            font-weight: 730;
            margin-top: 0.18rem;
        }
        .cf-workspace-copy {
            color: var(--cf-text-muted);
            font-size: 0.85rem;
            margin-top: 0.2rem;
        }
        .cf-planned-views {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.45rem;
            margin: 0.4rem 0 1rem;
        }
        .cf-view-chip {
            background: #f8fafc;
            border: 1px solid var(--cf-border);
            border-radius: 9px;
            color: #94a3b8;
            font-size: 0.72rem;
            padding: 0.55rem 0.35rem;
            text-align: center;
        }
        .cf-view-chip strong {
            color: #475569;
            display: block;
            font-size: 0.8rem;
            margin-bottom: 0.08rem;
        }
        .cf-view-chip-active {
            background: var(--cf-primary-soft);
            border-color: #bfdbfe;
        }
        .cf-view-chip-active strong {
            color: var(--cf-primary);
        }
        .cf-progress-slot {
            align-items: center;
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            margin: 0.2rem 0 0.65rem;
            padding: 0.65rem 0.75rem;
        }
        .cf-progress-slot strong {
            color: #475569;
            font-size: 0.82rem;
        }
        .cf-progress-slot span {
            color: #94a3b8;
            font-size: 0.72rem;
        }
        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def database_path() -> Path:
    return Path(os.getenv("CAPTUREFLOW_DATABASE_PATH", "data/captureflow.sqlite3"))


@st.cache_resource
def get_store(path: str) -> CaptureStore:
    store = CaptureStore(path)
    store.initialize()
    return store


def main() -> None:
    st.set_page_config(
        page_title="CaptureFlow",
        page_icon="🗂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_app_styles()
    st.title("CaptureFlow")
    st.caption("A local project continuity assistant for conversations, tasks, calendar, and notes.")

    store = get_store(str(database_path()))
    st.sidebar.markdown(
        """
        <div class="cf-sidebar-brand">
            <div class="cf-sidebar-mark">C</div>
            <div>
                <div class="cf-sidebar-name">CaptureFlow</div>
                <div class="cf-sidebar-tagline">Your local project workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Workspace", ["Chat", "Tracker", "Calendar", "Ideas & Notes"]
    )
    st.sidebar.divider()
    diagnostics_enabled = st.sidebar.toggle(
        "Diagnostics",
        value=False,
        help="Show timing for turns submitted while this switch is enabled.",
    )
    if page == "Calendar":
        render_calendar_page(store)
        return
    if page == "Ideas & Notes":
        render_saved_items_page(store)
        return
    with st.sidebar:
        selected_id = select_conversation(store, show_project_assignment=page != "Tracker")
    if page == "Tracker":
        render_tracker_page(store, selected_id)
        return
    if page == "Chat":
        render_chat_workspace(store, selected_id, diagnostics_enabled)


def select_conversation(store: CaptureStore, *, show_project_assignment: bool = True) -> str:
    if "next_sidebar_project_id" in st.session_state:
        st.session_state["sidebar_project_id"] = st.session_state.pop(
            "next_sidebar_project_id"
        )
    conversations = store.list_conversations()
    if not conversations:
        conversation = store.save_conversation(Conversation(title="First capture"))
        conversations = [conversation]

    projects = store.list_projects(statuses=("active", "archived"))
    project_by_id = {project.id: project for project in projects}
    conversation_project_ids = {
        conversation.id: (
            project.id
            if (project := store.get_conversation_project(conversation.id)) is not None
            else None
        )
        for conversation in conversations
    }
    conversation_by_id = {conversation.id: conversation for conversation in conversations}
    current_conversation_id = st.session_state.get("selected_conversation_id")
    if current_conversation_id not in conversation_by_id:
        current_conversation_id = conversations[0].id
    project_options = [None, *project_by_id]
    if (
        "sidebar_project_id" not in st.session_state
        or st.session_state["sidebar_project_id"] not in project_options
    ):
        st.session_state["sidebar_project_id"] = conversation_project_ids[current_conversation_id]
    selected_project_id = st.selectbox(
        "Project",
        options=project_options,
        format_func=lambda project_id: (
            "Unassigned"
            if project_id is None
            else (
                f"📁 {project_by_id[project_id].name} · archived"
                if project_by_id[project_id].status == "archived"
                else f"📁 {project_by_id[project_id].name}"
            )
        ),
        key="sidebar_project_id",
    )

    with st.expander("New conversation"):
        title = st.text_input("Title", key="new_conversation_title")
        selected_project = (
            project_by_id.get(selected_project_id) if selected_project_id is not None else None
        )
        can_create = selected_project is None or selected_project.status == "active"
        st.caption(
            "The new conversation will be unassigned."
            if selected_project is None
            else f"The new conversation will belong to {selected_project.name}."
        )
        if not can_create:
            st.caption("Restore this project before adding conversations.")
        if st.button("Create conversation", disabled=not can_create):
            try:
                created = store.save_conversation(Conversation(title=title))
            except ValidationError as error:
                st.error(error.errors()[0]["msg"])
            else:
                if selected_project_id is not None:
                    store.assign_conversation_project(created.id, selected_project_id)
                st.session_state["selected_conversation_id"] = created.id
                st.success(f"Created {created.title}.")
                st.rerun()

    visible_conversations = [
        conversation
        for conversation in conversations
        if conversation_project_ids[conversation.id] == selected_project_id
    ]
    if not visible_conversations:
        st.info("No conversations in this project yet.")
        st.stop()

    title_counts = {
        conversation.title: sum(item.title == conversation.title for item in visible_conversations)
        for conversation in visible_conversations
    }
    visible_ids = [conversation.id for conversation in visible_conversations]
    if st.session_state.get("selected_conversation_id") not in visible_ids:
        st.session_state["selected_conversation_id"] = visible_ids[0]

    def conversation_label(conversation_id: str) -> str:
        conversation = conversation_by_id[conversation_id]
        if title_counts[conversation.title] == 1:
            return conversation.title
        return f"{conversation.title} · {conversation.id[:8]}"

    selected_id = st.radio(
        "Conversations",
        options=visible_ids,
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
                        st.session_state["next_sidebar_project_id"] = selected_project_id
                        st.rerun()
            else:
                st.caption("Project assignment is locked after the first project turn or tracker record.")

        current_project = store.get_conversation_project(selected_id)
        if current_project is not None:
            _render_project_settings(store, current_project, key_prefix="sidebar")

    with st.expander("Conversation settings"):
        st.warning(
            "Deleting this conversation also deletes its messages, local events, saved items, "
            "tracker records, operations, and answer sources."
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


def _render_project_settings(
    store: CaptureStore, project: Project, *, key_prefix: str
) -> None:
    with st.expander("Project settings"):
        name = st.text_input(
            "Project name",
            value=project.name,
            key=f"{key_prefix}-project-name-{project.id}",
        )
        description = st.text_area(
            "Project description",
            value=project.description or "",
            key=f"{key_prefix}-project-description-{project.id}",
        )
        if st.button("Save project", key=f"{key_prefix}-save-project-{project.id}"):
            try:
                store.update_project(project.id, name=name, description=description or None)
            except ValidationError as error:
                st.error(_validation_message(error))
            else:
                st.rerun()

        st.warning(
            "Deleting this project removes its tracker records and project audit history. "
            "Conversations, messages, calendar events, ideas, and notes are retained as unassigned."
        )
        confirmed = st.checkbox(
            "I understand what project deletion removes.",
            key=f"{key_prefix}-confirm-delete-project-{project.id}",
        )
        if st.button(
            "Delete project",
            key=f"{key_prefix}-delete-project-{project.id}",
            disabled=not confirmed,
        ):
            store.delete_project(project.id)
            st.session_state["next_sidebar_project_id"] = None
            st.rerun()


def render_chat_workspace(
    store: CaptureStore, conversation_id: str, diagnostics_enabled: bool = False
) -> None:
    current_project = store.get_conversation_project(conversation_id)
    project_name = current_project.name if current_project is not None else "Unassigned workspace"
    project_copy = (
        current_project.description
        if current_project is not None and current_project.description
        else "Conversation, project state, calendar, and notes in one local workspace."
    )
    st.markdown(
        f"""
        <div class="cf-workspace-header">
            <div class="cf-workspace-kicker">Project workspace</div>
            <div class="cf-workspace-title">{escape(project_name)}</div>
            <div class="cf-workspace-copy">{escape(project_copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chat_column, snapshot_column = st.columns([1.75, 1], gap="large")
    with chat_column:
        render_chat_page(store, conversation_id, diagnostics_enabled)
    with snapshot_column:
        render_project_snapshot(store, conversation_id)
    render_chat_input(store, conversation_id, diagnostics_enabled)


def render_project_snapshot(store: CaptureStore, conversation_id: str) -> None:
    current_project = store.get_conversation_project(conversation_id)
    with st.container(border=True):
        st.subheader("Project snapshot")
        st.caption("A read-only view of locally grounded project context.")
        _render_planned_project_views()
        if current_project is None:
            st.info("Assign this conversation to a project to activate the snapshot.")
            return

        conversation_ids = {
            conversation.id
            for conversation in store.list_conversations()
            if (
                (project := store.get_conversation_project(conversation.id)) is not None
                and project.id == current_project.id
            )
        }
        records = store.list_project_records(
            current_project.id,
            states=("active", "completed", "resolved", "superseded"),
        )
        active_tasks = [
            record for record in records if record.kind == "task" and record.state == "active"
        ]
        completed_tasks = [
            record for record in records if record.kind == "task" and record.state == "completed"
        ]
        active_blockers = [
            record for record in records if record.kind == "blocker" and record.state == "active"
        ]
        active_metric, completed_metric, blocker_metric = st.columns(3)
        active_metric.metric("Active", len(active_tasks))
        completed_metric.metric("Done", len(completed_tasks))
        blocker_metric.metric("Blockers", len(active_blockers))

        reference_date = configured_reference_date(configured_timezone())
        upcoming_events = [
            event
            for event in store.list_calendar_events()
            if event.conversation_id in conversation_ids
            and event.start_date >= reference_date
        ]
        upcoming_deadlines = [
            deadline
            for deadline in store.list_task_deadlines()
            if deadline.project_id == current_project.id
            and _day_of(deadline.due_at) >= reference_date
        ]
        upcoming = sorted(
            [(event.start_date, event) for event in upcoming_events]
            + [(_day_of(deadline.due_at), deadline) for deadline in upcoming_deadlines],
            key=lambda entry: entry[0],
        )[:3]
        st.markdown("#### Upcoming")
        if not upcoming:
            st.caption("No upcoming project events or deadlines.")
        for _, entry in upcoming:
            with st.container(border=True):
                if isinstance(entry, ProjectRecord):
                    st.caption(f"Due {_format_due_at(entry.due_at)} · deadline")
                    st.markdown(f"**{entry.content}**")
                    continue
                st.caption(_format_calendar_timing(entry.start_at, entry.end_at))
                st.markdown(f"**{entry.title}**")
                if entry.description:
                    st.caption(entry.description)

        active_context = [
            record
            for record in records
            if record.state == "active"
            and record.kind in {"blocker", "task", "open_question", "decision", "constraint"}
        ][:4]
        st.markdown("#### Active work")
        if not active_context:
            st.caption("No active project records.")
        for record in active_context:
            with st.container(border=True):
                st.caption(record.kind.replace("_", " ").title())
                st.markdown(record.content)

        recent_items = [
            item
            for item in store.list_saved_items()
            if item.conversation_id in conversation_ids
        ][:3]
        st.markdown("#### Recent ideas & notes")
        if not recent_items:
            st.caption("No project ideas or notes yet.")
        for item in recent_items:
            with st.container(border=True):
                st.caption(item.kind.title())
                st.markdown(item.content)


def _render_planned_project_views() -> None:
    st.markdown(
        """
        <div class="cf-progress-slot">
            <strong>Project progress</strong>
            <span>Planned</span>
        </div>
        <div class="cf-planned-views">
            <div class="cf-view-chip cf-view-chip-active"><strong>All</strong>Current view</div>
            <div class="cf-view-chip"><strong>Today</strong>Planned</div>
            <div class="cf-view-chip"><strong>Priority</strong>Planned</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _day_of(moment: date | datetime) -> date:
    return moment.date() if isinstance(moment, datetime) else moment


def _format_due_at(due_at: date | datetime) -> str:
    """A deadline is one instant or one day, never a span."""

    if isinstance(due_at, datetime):
        return due_at.strftime("%b %d · %H:%M")
    return due_at.strftime("%b %d")


def _format_calendar_timing(start_at: date | datetime, end_at: datetime | None) -> str:
    if not isinstance(start_at, datetime):
        return start_at.strftime("%b %d · All day")
    timing = start_at.strftime("%b %d · %H:%M")
    if end_at is not None:
        if end_at.date() == start_at.date():
            timing += f"–{end_at.strftime('%H:%M')}"
        else:
            timing += f"–{end_at.strftime('%b %d · %H:%M')}"
    return timing


def render_chat_page(
    store: CaptureStore, conversation_id: str, diagnostics_enabled: bool = False
) -> None:
    st.subheader("Chat")
    st.caption(
        "Grounded in recent conversation context and locally stored project records."
    )
    timezone = configured_timezone()
    local_service = DeterministicChatService(store, timezone)
    for message in store.list_messages(conversation_id):
        with st.chat_message(message.role):
            st.write(message.content)
            if message.role == "assistant" and message.reply_to_message_id:
                for operation in store.list_tool_operations(message.reply_to_message_id):
                    _render_operation_card(local_service, operation.id)
                for operation in store.list_calendar_event_update_operations(
                    message.reply_to_message_id
                ):
                    _render_calendar_update_card(local_service, operation.id)
                for operation in store.list_project_change_operations(message.reply_to_message_id):
                    _render_project_change_card(store, operation.id)
                _render_project_answer_sources(store, message.id)
                _render_local_context_answer_sources(store, message.id)
                if diagnostics_enabled:
                    _render_turn_diagnostics(message.reply_to_message_id)

    if pending := store.get_pending_clarification(conversation_id):
        question = store.get_message(pending.assistant_message_id).content
        follow_up = pending.missing_field == "follow_up"
        if follow_up:
            question = question.rsplit("\n\n", 1)[-1]
        st.warning(question)
        if st.button(
            "Dismiss suggestion" if follow_up else "Cancel pending event",
            key=f"cancel-clarification-{pending.id}",
        ):
            store.resolve_pending_clarification(pending.id, "cancelled")
            st.rerun()


def render_chat_input(
    store: CaptureStore, conversation_id: str, diagnostics_enabled: bool = False
) -> None:
    timezone = configured_timezone()
    local_service = DeterministicChatService(store, timezone)
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
                )
        else:
            try:
                settings = ModelSettings.from_environment()
                now = configured_reference_now(timezone)
                turn = ConversationRouterService(
                    store,
                    OpenAICompatibleModel(settings),
                    now.date(),
                    timezone,
                    reference_time=now.time(),
                ).submit(conversation_id, prompt)
                if diagnostics_enabled:
                    _remember_turn_diagnostics(
                        turn.user_message.id,
                        mode="contextual_router",
                        intent=None if turn.decision is None else turn.decision.intent,
                        bounded_context_messages=len(turn.context_window.messages),
                        context_truncated=turn.context_window.truncated,
                        candidate_events=len(turn.candidates.events),
                        candidate_saved_items=len(turn.candidates.saved_items),
                        candidate_project_records=len(turn.candidates.project_records),
                        outcome=turn.outcome,
                        model_calls=turn.model_calls,
                        model_seconds=turn.model_seconds,
                        local_seconds=turn.local_seconds,
                        total_seconds=perf_counter() - started,
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
                        intent=None,
                        bounded_context_messages=0,
                        context_truncated=False,
                        candidate_events=0,
                        candidate_saved_items=0,
                        candidate_project_records=0,
                        outcome="no_model_call",
                        model_calls=0,
                        model_seconds=0,
                        local_seconds=elapsed,
                        total_seconds=elapsed,
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
            f"Model calls: {diagnostics['model_calls']}"
        )
        st.caption(
            f"Model: {diagnostics['model_seconds']:.3f}s · "
            f"Local validation/write: {diagnostics['local_seconds']:.3f}s · "
            f"Total submit: {diagnostics['total_seconds']:.3f}s"
        )
        if "intent" in diagnostics:
            intent = diagnostics["intent"] or "not available"
            st.caption(
                f"Intent: {intent} · Bounded context: "
                f"{diagnostics['bounded_context_messages']} message(s) · "
                f"Truncated: {'yes' if diagnostics['context_truncated'] else 'no'}"
            )
            st.caption(
                f"Candidates: events {diagnostics['candidate_events']} · "
                f"ideas/notes {diagnostics['candidate_saved_items']} · "
                f"project records {diagnostics['candidate_project_records']}"
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


def _render_calendar_update_card(service: DeterministicChatService, operation_id: str) -> None:
    operation = service.store.get_calendar_event_update_operation(operation_id)
    deleted = operation.after.status == "deleted"
    label = "Local calendar deletion" if deleted else "Local calendar update"
    with st.container(border=True):
        st.caption(f"{label}: {operation.after.title}; status: {operation.status}")
        if operation.status == "committed" and st.button(
            "Undo calendar deletion" if deleted else "Undo calendar update",
            key=f"undo-calendar-update-{operation.id}",
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
    """Manage persisted project state and source-grounded records."""

    st.subheader("Project Tracker")
    st.caption(
        "Records are grounded in chat messages, and every lifecycle change is retained locally."
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
                st.session_state["next_sidebar_project_id"] = current_project.id
                st.success(f"Conversation assigned to {current_project.name}.")
                st.rerun()

    if current_project is None:
        st.info("Create or assign a project before adding tracker records.")
        return

    st.markdown(f"### {current_project.name}")
    if current_project.description:
        st.caption(current_project.description)
    st.caption(f"Project status: {current_project.status}")
    _render_project_settings(store, current_project, key_prefix="tracker")
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
                st.info("Send a chat message first; tracker records require an explicit user source.")
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
        st.info("No project records yet.")
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
    _render_task_deadlines(store)
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
                timing = event.start_at.isoformat()
                if event.end_at is not None:
                    timing = f"{timing} to {event.end_at.isoformat()}"
                st.caption(f"Timed event: {timing}")
            if event.description:
                st.markdown(event.description)
            _render_calendar_sources(store, event)
            st.caption(f"Updated {event.updated_at.isoformat()}")
            if st.button("Delete event", key=f"delete-event-{event.id}"):
                store.set_calendar_event_status(event.id, "deleted")
                st.rerun()

    history = store.list_calendar_events(statuses=("deleted",))
    if history:
        with st.expander(f"Calendar history ({len(history)})"):
            for event in history:
                with st.container(border=True):
                    timing = event.start_at.isoformat()
                    if event.end_at is not None:
                        timing = f"{timing} to {event.end_at.isoformat()}"
                    st.write(f"{timing} — {event.title}")
                    if event.description:
                        st.markdown(event.description)
                    _render_calendar_sources(store, event)
                    if st.button("Restore event", key=f"restore-event-{event.id}"):
                        store.set_calendar_event_status(event.id, "active")
                        st.rerun()


def _render_task_deadlines(store: CaptureStore) -> None:
    """Show tracker deadlines read-only; they are not events and hold no time."""

    deadlines = store.list_task_deadlines()
    if not deadlines:
        return
    st.markdown(f"#### Deadlines ({len(deadlines)})")
    st.caption(
        "Due times from the project tracker. They are not events and reserve no time; "
        "complete or delete the task in Tracker to clear one."
    )
    for record in deadlines:
        with st.container(border=True):
            st.caption(
                f"Due {_format_due_at(record.due_at)} · {store.get_project(record.project_id).name}"
            )
            st.markdown(f"**{record.content}**")
    st.markdown("#### Events")


def _render_calendar_sources(store: CaptureStore, event: CalendarEvent) -> None:
    sources = store.list_calendar_event_sources(event.id)
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.caption(source.source_quote)


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
