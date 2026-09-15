import os
from calendar import Calendar
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import streamlit as st
from pydantic import ValidationError

from .chat_service import DeterministicChatService
from .conversation_router_service import ConversationRouterService
from .llm import ModelError, ModelSettings, OpenAICompatibleModel
from .models import CalendarEvent, Conversation, Project, ProjectRecord
from .storage import CaptureStore
from .time_config import configured_reference_date, configured_reference_now, configured_timezone


DIAGNOSTICS_SESSION_KEY = "captureflow_turn_diagnostics"
PAGE_SESSION_KEY = "workspace_page"
CALENDAR_DAY_KEY = "calendar_selected_day"
CALENDAR_MONTH_KEY = "calendar_month"
WORKSPACE_PANEL_HEIGHT = 600

PAGES = ["Chat", "Tracker", "Calendar", "Ideas & Notes"]
PAGE_ICONS = {
    "Chat": ":material/forum:",
    "Tracker": ":material/view_kanban:",
    "Calendar": ":material/calendar_month:",
    "Ideas & Notes": ":material/lightbulb:",
}
KIND_STYLES = {
    "task": ("Task", "blue", ":material/task_alt:"),
    "milestone": ("Milestone", "violet", ":material/flag:"),
    "decision": ("Decision", "green", ":material/gavel:"),
    "constraint": ("Constraint", "orange", ":material/rule:"),
    "open_question": ("Question", "violet", ":material/help:"),
    "blocker": ("Blocker", "red", ":material/block:"),
}
CLOSING_TRANSITIONS = {
    "task": ("completed", "Complete"),
    "open_question": ("resolved", "Resolve"),
    "blocker": ("resolved", "Resolve"),
    "decision": ("superseded", "Supersede"),
    "constraint": ("superseded", "Supersede"),
}
TRACKER_GROUPS = [
    ("Tasks", ":material/task_alt:", ("task",)),
    ("Blockers", ":material/block:", ("blocker",)),
    ("Questions", ":material/help:", ("open_question",)),
    ("Decisions", ":material/gavel:", ("decision", "constraint")),
    ("Milestones", ":material/flag:", ("milestone",)),
]
TOOL_LABELS = {
    "create_local_event": ("Calendar event", ":material/event:"),
    "save_idea": ("Idea", ":material/lightbulb:"),
    "save_note": ("Note", ":material/sticky_note_2:"),
}
STATUS_BADGES = {
    "committed": ":green-badge[:material/check: Applied]",
    "undone": ":gray-badge[:material/undo: Undone]",
    "review": ":orange-badge[:material/pending: Needs review]",
    "rejected": ":gray-badge[:material/close: Rejected]",
    "failed": ":red-badge[:material/error: Not applied]",
}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _inject_app_styles() -> None:
    st.html(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            max-width: 1480px;
            padding-top: 4rem;
            padding-bottom: 5rem;
        }
        [data-testid="stSidebarUserContent"] {
            padding-top: 1.25rem;
        }
        .cf-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin-bottom: 0.5rem;
        }
        .cf-brand-mark {
            display: grid;
            place-items: center;
            width: 2.25rem;
            height: 2.25rem;
            border-radius: 10px;
            color: white;
            background: linear-gradient(145deg, #2563eb, #38bdf8);
            font-weight: 700;
        }
        .cf-brand-name {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .cf-brand-tagline {
            color: #64748b;
            font-size: 0.75rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: #eef4ff;
        }
        [data-testid="stLayoutWrapper"]:has(> .st-key-chat-timeline),
        [data-testid="stLayoutWrapper"]:has(> .st-key-project-snapshot) {
            flex-basis: max(24rem, calc(100vh - 23.5rem)) !important;
            height: max(24rem, calc(100vh - 23.5rem)) !important;
        }
        .st-key-chat-timeline [data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.75rem 1rem;
        }
        [class*="st-key-calcell-"] {
            min-height: 7.25rem;
            gap: 0.2rem;
            padding: 0.35rem 0.4rem;
        }
        [class*="st-key-calcell-muted-"] {
            background: #f8fafc;
            opacity: 0.55;
        }
        [class*="st-key-calcell-today-"] {
            border-color: #93c5fd !important;
            background: #f5f9ff;
        }
        [class*="st-key-calcell-selected-"] {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 1px #2563eb;
        }
        [class*="st-key-calcell-"] button {
            min-height: 1.8rem;
            padding: 0 0.55rem;
        }
        [class*="st-key-calcell-"] [data-testid="stMarkdownContainer"] p {
            font-size: 0.72rem;
            line-height: 1.3;
        }
        [class*="st-key-calcell-"] [data-testid="stMarkdownContainer"] p > span {
            white-space: normal !important;
            display: -webkit-box !important;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            overflow-wrap: anywhere;
        }
        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """
    )


def database_path() -> Path:
    return Path(os.getenv("CAPTUREFLOW_DATABASE_PATH", "data/captureflow.sqlite3"))


def configured_answer_mode() -> str:
    mode = os.getenv("CAPTUREFLOW_ANSWER_MODE", "workflow").strip().casefold()
    if mode not in {"workflow", "agent"}:
        raise ValueError("CAPTUREFLOW_ANSWER_MODE must be workflow or agent.")
    return mode


@st.cache_resource
def get_store(path: str) -> CaptureStore:
    store = CaptureStore(path)
    store.initialize()
    return store


def main() -> None:
    st.set_page_config(
        page_title="CaptureFlow",
        page_icon=":material/stacks:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_app_styles()
    store = get_store(str(database_path()))

    with st.sidebar:
        st.html(
            """
            <div class="cf-brand">
                <div class="cf-brand-mark">C</div>
                <div>
                    <div class="cf-brand-name">CaptureFlow</div>
                    <div class="cf-brand-tagline">Local project continuity assistant</div>
                </div>
            </div>
            """
        )
        page = st.radio(
            "Workspace",
            PAGES,
            key=PAGE_SESSION_KEY,
            format_func=lambda name: f"{PAGE_ICONS[name]}  {name}",
            label_visibility="collapsed",
        )
        selected_id = select_conversation(store) if page in {"Chat", "Tracker"} else None
        st.space("medium")
        diagnostics_enabled = st.toggle(
            "Diagnostics",
            value=False,
            key="diagnostics_enabled",
            help="Show timing for turns submitted while this switch is enabled.",
        )

    if page == "Calendar":
        render_calendar_page(store)
    elif page == "Ideas & Notes":
        render_saved_items_page(store)
    elif selected_id is None:
        st.title(page)
        st.info(
            "This project has no conversations yet. Use **New conversation** in the sidebar to start one.",
            icon=":material/forum:",
        )
    elif page == "Tracker":
        render_tracker_page(store, selected_id)
    else:
        render_chat_workspace(store, selected_id, diagnostics_enabled)


def _go_to_page(page: str) -> None:
    st.session_state[PAGE_SESSION_KEY] = page


def _open_calendar_day(day: date) -> None:
    st.session_state[CALENDAR_DAY_KEY] = day
    st.session_state[CALENDAR_MONTH_KEY] = day.replace(day=1)
    st.session_state[PAGE_SESSION_KEY] = "Calendar"


def select_conversation(store: CaptureStore) -> str | None:
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
            "Unassigned conversations"
            if project_id is None
            else (
                f"{project_by_id[project_id].name} · archived"
                if project_by_id[project_id].status == "archived"
                else project_by_id[project_id].name
            )
        ),
        key="sidebar_project_id",
    )

    selected_project = (
        project_by_id.get(selected_project_id) if selected_project_id is not None else None
    )
    with st.popover("New conversation", icon=":material/add:", width="stretch"):
        title = st.text_input(
            "Title", key="new_conversation_title", placeholder="e.g. Sprint planning"
        )
        can_create = selected_project is None or selected_project.status == "active"
        st.caption(
            "The new conversation will be unassigned."
            if selected_project is None
            else f"The new conversation will belong to {selected_project.name}."
        )
        if not can_create:
            st.caption("Restore this project before adding conversations.")
        if st.button("Create conversation", type="primary", disabled=not can_create):
            try:
                created = store.save_conversation(Conversation(title=title))
            except ValidationError as error:
                st.error(error.errors()[0]["msg"])
            else:
                if selected_project_id is not None:
                    store.assign_conversation_project(created.id, selected_project_id)
                st.session_state["selected_conversation_id"] = created.id
                st.rerun()

    visible_conversations = [
        conversation
        for conversation in conversations
        if conversation_project_ids[conversation.id] == selected_project_id
    ]
    if not visible_conversations:
        st.caption("No conversations in this project yet.")
        return None

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

    return st.radio(
        "Conversations",
        options=visible_ids,
        format_func=conversation_label,
        key="selected_conversation_id",
    )


def _render_project_assignment(
    store: CaptureStore, conversation_id: str, *, key_prefix: str
) -> None:
    current_project = store.get_conversation_project(conversation_id)
    if store.conversation_has_project_turn(conversation_id) or store.list_project_records_for_conversation(
        conversation_id
    ):
        st.caption("Project assignment is locked after the first project turn or tracker record.")
        return
    projects = store.list_projects(statuses=("active",))
    if not projects:
        st.caption("No active projects yet. Create one to link this conversation.")
        return
    options = [None, *[project.id for project in projects]]
    names = {project.id: project.name for project in projects}
    current_id = current_project.id if current_project else None
    chosen_id = st.selectbox(
        "Project for this conversation",
        options=options,
        index=options.index(current_id) if current_id in options else 0,
        format_func=lambda project_id: "Unassigned" if project_id is None else names[project_id],
        key=f"{key_prefix}-conversation-project-{conversation_id}",
    )
    if chosen_id is not None and chosen_id != current_id:
        if st.button(
            "Link project",
            type="primary",
            icon=":material/link:",
            key=f"{key_prefix}-assign-conversation-project-{conversation_id}",
        ):
            try:
                store.assign_conversation_project(conversation_id, chosen_id)
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state["next_sidebar_project_id"] = chosen_id
                st.rerun()


def _render_create_project(
    store: CaptureStore, *, key_prefix: str, conversation_id: str | None = None
) -> None:
    with st.popover("New project", icon=":material/create_new_folder:"):
        name = st.text_input("Project name", key=f"{key_prefix}-new-project-name")
        description = st.text_area(
            "Description (optional)", key=f"{key_prefix}-new-project-description"
        )
        if st.button("Create project", type="primary", key=f"{key_prefix}-create-project"):
            try:
                project = store.save_project(Project(name=name, description=description or None))
            except ValidationError as error:
                st.error(_validation_message(error))
                return
            if conversation_id is not None and store.get_conversation_project(conversation_id) is None:
                try:
                    store.assign_conversation_project(conversation_id, project.id)
                except ValueError:
                    pass
                else:
                    st.session_state["next_sidebar_project_id"] = project.id
            st.rerun()


def _render_project_settings(
    store: CaptureStore, project: Project, *, key_prefix: str
) -> None:
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
    with st.container(horizontal=True):
        if st.button("Save project", type="primary", key=f"{key_prefix}-save-project-{project.id}"):
            try:
                store.update_project(project.id, name=name, description=description or None)
            except ValidationError as error:
                st.error(_validation_message(error))
            else:
                st.rerun()
        if project.status == "active":
            if st.button(
                "Archive",
                icon=":material/archive:",
                key=f"{key_prefix}-archive-project-{project.id}",
            ):
                store.set_project_status(project.id, "archived")
                st.rerun()
        elif st.button(
            "Restore",
            icon=":material/unarchive:",
            key=f"{key_prefix}-restore-project-{project.id}",
        ):
            store.set_project_status(project.id, "active")
            st.rerun()

    st.space("small")
    st.caption(
        "Deleting removes tracker records and project audit history. Conversations, messages, "
        "calendar events, ideas, and notes are kept as unassigned."
    )
    confirmed = st.checkbox(
        "I understand what project deletion removes.",
        key=f"{key_prefix}-confirm-delete-project-{project.id}",
    )
    if st.button(
        "Delete project",
        icon=":material/delete:",
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
    conversation = store.get_conversation(conversation_id)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        with st.container(gap=None):
            st.caption(
                f":material/folder: {current_project.name}"
                if current_project is not None
                else ":material/folder_off: Unassigned conversation"
            )
            st.title(conversation.title)
        if current_project is not None:
            with st.popover("Project", icon=":material/folder:"):
                _render_project_settings(store, current_project, key_prefix="chat")
        with st.popover("Conversation", icon=":material/tune:"):
            _render_conversation_settings(store, conversation_id, current_project)

    chat_column, snapshot_column = st.columns([1.5, 1], gap="medium")
    with chat_column:
        render_chat_page(store, conversation_id, diagnostics_enabled)
    with snapshot_column:
        render_project_snapshot(store, conversation_id)
    render_chat_input(store, conversation_id, diagnostics_enabled)


def _render_conversation_settings(
    store: CaptureStore, conversation_id: str, current_project: Project | None
) -> None:
    if current_project is not None:
        st.markdown("**Project link**")
        _render_project_assignment(store, conversation_id, key_prefix="manage")
        st.space("small")
    st.markdown("**Delete conversation**")
    st.caption(
        "Also deletes its messages, local events, saved items, tracker records, operations, "
        "and answer sources."
    )
    confirmed = st.checkbox(
        "I understand this deletes the conversation and its derived data.",
        key=f"confirm-delete-conversation-{conversation_id}",
    )
    if st.button(
        "Delete conversation",
        icon=":material/delete:",
        key=f"delete-conversation-{conversation_id}",
        disabled=not confirmed,
    ):
        store.delete_conversation(conversation_id)
        st.rerun()


def _close_record(store: CaptureStore, record_id: str, state: str) -> None:
    store.transition_project_record(record_id, state)


def render_project_snapshot(store: CaptureStore, conversation_id: str) -> None:
    current_project = store.get_conversation_project(conversation_id)
    with st.container(border=True, height=WORKSPACE_PANEL_HEIGHT, key="project-snapshot"):
        st.subheader("Project snapshot")
        if current_project is None:
            st.caption(
                "Link this conversation to a project to track its tasks, blockers, and deadlines here."
            )
            _render_project_assignment(store, conversation_id, key_prefix="snapshot")
            _render_create_project(store, key_prefix="snapshot", conversation_id=conversation_id)
            return
        if current_project.description:
            st.caption(current_project.description)

        project_conversation_ids = {
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
        tasks = [record for record in records if record.kind == "task"]
        completed_tasks = [record for record in tasks if record.state == "completed"]
        active_tasks = [record for record in tasks if record.state == "active"]
        active_blockers = [
            record for record in records if record.kind == "blocker" and record.state == "active"
        ]
        st.progress(
            len(completed_tasks) / len(tasks) if tasks else 0.0,
            text=(
                f"{len(completed_tasks)} of {len(tasks)} tasks done"
                if tasks
                else "No tasks tracked yet"
            ),
        )
        active_metric, completed_metric, blocker_metric = st.columns(3)
        active_metric.metric("Active", len(active_tasks))
        completed_metric.metric("Done", len(completed_tasks))
        blocker_metric.metric("Blockers", len(active_blockers))

        reference_date = configured_reference_date(configured_timezone())
        upcoming = _upcoming_entries(
            store,
            reference_date,
            conversation_ids=project_conversation_ids,
            project_id=current_project.id,
        )[:5]
        open_work = sorted(
            (record for record in records if record.state == "active" and record.kind != "milestone"),
            key=lambda record: (record.kind != "blocker", record.kind != "task"),
        )
        notes = store.list_saved_items_for_project(current_project.id)[:5]

        work_tab, upcoming_tab, notes_tab = st.tabs(
            [f"Open work ({len(open_work)})", f"Upcoming ({len(upcoming)})", f"Notes ({len(notes)})"]
        )

        with work_tab:
            if not open_work:
                st.caption("Nothing open. New tasks and blockers from chat appear here.")
            for record in open_work[:6]:
                _render_snapshot_record(store, record)
            if len(open_work) > 6:
                st.caption(f"+{len(open_work) - 6} more in Tracker")
            st.button(
                "Open tracker",
                icon=":material/arrow_forward:",
                type="tertiary",
                key="snapshot-open-tracker",
                on_click=_go_to_page,
                args=("Tracker",),
            )
        with upcoming_tab:
            if not upcoming:
                st.caption("No upcoming project events or deadlines.")
            for day, entry in upcoming:
                if isinstance(entry, ProjectRecord):
                    st.caption(f":orange[:material/flag: Due {_format_due_at(entry.due_at)}]")
                    label, icon = entry.content, ":material/task_alt:"
                else:
                    st.caption(
                        f":blue[:material/event: {_format_calendar_timing(entry.start_at, entry.end_at)}]"
                    )
                    label, icon = entry.title, ":material/calendar_month:"
                st.button(
                    label,
                    icon=icon,
                    type="tertiary",
                    help="Open this day in Calendar",
                    key=f"snapshot-open-day-{entry.id}",
                    on_click=_open_calendar_day,
                    args=(day,),
                )
        with notes_tab:
            if not notes:
                st.caption("No project ideas or notes yet.")
            for item in notes:
                color = "violet" if item.kind == "idea" else "gray"
                st.markdown(f":{color}-badge[{item.kind.title()}] {item.content}")
            st.button(
                "Open ideas & notes",
                icon=":material/arrow_forward:",
                type="tertiary",
                key="snapshot-open-notes",
                on_click=_go_to_page,
                args=("Ideas & Notes",),
            )


def _render_snapshot_record(store: CaptureStore, record: ProjectRecord) -> None:
    name, color, icon = KIND_STYLES[record.kind]
    label = record.content if record.kind == "task" else f":{color}-badge[{icon} {name}] {record.content}"
    if record.due_at is not None:
        label += f"  :orange-badge[Due {_format_due_at(record.due_at)}]"
    closing_state = {"task": "completed", "blocker": "resolved", "open_question": "resolved"}.get(
        record.kind
    )
    if closing_state is None:
        st.markdown(label)
        return
    st.checkbox(
        label,
        key=f"snapshot-close-{record.id}",
        on_change=_close_record,
        args=(store, record.id, closing_state),
    )


def _upcoming_entries(
    store: CaptureStore,
    from_day: date,
    *,
    conversation_ids: set[str] | None = None,
    project_id: str | None = None,
) -> list[tuple[date, CalendarEvent | ProjectRecord]]:
    events = [
        (event.start_date, event)
        for event in store.list_calendar_events()
        if event.start_date >= from_day
        and (conversation_ids is None or event.conversation_id in conversation_ids)
    ]
    deadlines = [
        (_day_of(record.due_at), record)
        for record in store.list_task_deadlines()
        if _day_of(record.due_at) >= from_day
        and (project_id is None or record.project_id == project_id)
    ]
    return sorted(events + deadlines, key=lambda entry: entry[0])


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


def _format_time_of_day(event: CalendarEvent) -> str:
    if not isinstance(event.start_at, datetime):
        return "All day"
    timing = event.start_at.strftime("%H:%M")
    if event.end_at is not None:
        timing += f"–{event.end_at.strftime('%H:%M')}"
    return timing


def _format_timestamp(moment: datetime) -> str:
    try:
        moment = moment.astimezone(ZoneInfo(configured_timezone()))
    except (ValueError, KeyError):
        pass
    return moment.strftime("%b %d, %H:%M")


def _badge_text(text: str, limit: int) -> str:
    text = " ".join(text.split()).replace("[", "(").replace("]", ")")
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def render_chat_page(
    store: CaptureStore, conversation_id: str, diagnostics_enabled: bool = False
) -> None:
    timezone = configured_timezone()
    local_service = DeterministicChatService(store, timezone)
    messages = store.list_messages(conversation_id)
    with st.container(height=WORKSPACE_PANEL_HEIGHT, autoscroll=True, key="chat-timeline"):
        if not messages:
            with st.container(horizontal_alignment="center"):
                st.space("large")
                st.markdown("### :material/forum: Start the conversation")
                st.caption(
                    "Describe what you're working on. Meetings become calendar events, "
                    "and project updates become tracker records you can review."
                )
                st.caption(
                    "Shortcuts: `/event 2026-09-20 | Title` · `/idea …` · `/note …`"
                )
        for message in messages:
            with st.chat_message(message.role):
                st.write(message.content)
                if message.role == "assistant" and message.reply_to_message_id:
                    for operation in store.list_tool_operations(message.reply_to_message_id):
                        _render_operation_card(local_service, operation.id)
                    for operation in store.list_calendar_event_update_operations(
                        message.reply_to_message_id
                    ):
                        _render_calendar_update_card(local_service, operation.id)
                    for operation in store.list_project_change_operations(
                        message.reply_to_message_id
                    ):
                        _render_project_change_card(store, operation.id)
                    _render_project_answer_sources(store, message.id)
                    _render_local_context_answer_sources(store, message.id)
                    if diagnostics_enabled:
                        _render_turn_diagnostics(message.reply_to_message_id)


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
                answer_mode = configured_answer_mode()
                turn = ConversationRouterService(
                    store,
                    OpenAICompatibleModel(settings),
                    now.date(),
                    timezone,
                    reference_time=now.time(),
                    answer_mode=answer_mode,
                ).submit(conversation_id, prompt)
                if diagnostics_enabled:
                    _remember_turn_diagnostics(
                        turn.user_message.id,
                        mode=f"contextual_router_{answer_mode}",
                        intent=None if turn.decision is None else turn.decision.intent,
                        bounded_context_messages=len(turn.context_window.messages),
                        context_truncated=turn.context_window.truncated,
                        candidate_events=len(turn.candidates.events),
                        candidate_saved_items=len(turn.candidates.saved_items),
                        candidate_project_records=len(turn.candidates.project_records),
                        outcome=turn.outcome,
                        failure_reason=turn.failure_reason,
                        model_calls=turn.model_calls,
                        model_seconds=turn.model_seconds,
                        local_seconds=turn.local_seconds,
                        tool_calls=turn.tool_calls,
                        agent_stop_reason=turn.agent_stop_reason,
                        total_seconds=perf_counter() - started,
                    )
            except (ModelError, ValueError) as error:
                project = store.get_conversation_project(conversation_id)
                if project is not None:
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
    with st.expander(f"Diagnostics · {message_id[:8]}", icon=":material/monitoring:"):
        st.caption(
            f"Mode: {diagnostics['mode']} · Outcome: {diagnostics['outcome']} · "
            f"Model calls: {diagnostics['model_calls']}"
        )
        st.caption(
            f"Model: {diagnostics['model_seconds']:.3f}s · "
            f"Local validation/write: {diagnostics['local_seconds']:.3f}s · "
            f"Total submit: {diagnostics['total_seconds']:.3f}s"
        )
        if diagnostics.get("failure_reason"):
            st.caption(f"Failure: {diagnostics['failure_reason']}")
        if diagnostics.get("tool_calls") or diagnostics.get("agent_stop_reason"):
            st.caption(
                f"Agent tools: {diagnostics.get('tool_calls', 0)} · "
                f"Stop: {diagnostics.get('agent_stop_reason') or 'not available'}"
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


def _status_badge(status: str) -> str:
    return STATUS_BADGES.get(status, f":blue-badge[{status}]")


def _render_operation_card(service: DeterministicChatService, operation_id: str) -> None:
    operation = service.store.get_tool_operation(operation_id)
    label, icon = TOOL_LABELS.get(
        operation.tool_name, (operation.tool_name.replace("_", " ").capitalize(), ":material/bolt:")
    )
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        st.markdown(f"{icon} **{label}** saved locally  {_status_badge(operation.status)}")
        if operation.status == "committed" and st.button(
            "Undo", icon=":material/undo:", type="tertiary", key=f"undo-operation-{operation.id}"
        ):
            service.undo(operation.id)
            st.rerun()


def _render_calendar_update_card(service: DeterministicChatService, operation_id: str) -> None:
    operation = service.store.get_calendar_event_update_operation(operation_id)
    deleted = operation.after.status == "deleted"
    label = "Delete event" if deleted else "Update event"
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(
                f":material/event: **{label}** · {operation.after.title}  {_status_badge(operation.status)}"
            )
            if operation.status == "committed" and st.button(
                "Undo",
                icon=":material/undo:",
                type="tertiary",
                key=f"undo-calendar-update-{operation.id}",
            ):
                service.undo(operation.id)
                st.rerun()
        if operation.status == "review":
            with st.container(horizontal=True):
                if st.button(
                    "Delete",
                    type="primary",
                    icon=":material/delete:",
                    key=f"confirm-calendar-update-{operation.id}",
                ):
                    try:
                        service.store.approve_calendar_event_update(operation.id)
                    except (KeyError, ValueError) as error:
                        st.error(str(error))
                    else:
                        st.rerun()
                if st.button("Keep", key=f"keep-calendar-update-{operation.id}"):
                    service.store.reject_calendar_event_update(operation.id)
                    st.rerun()
        if operation.status == "failed" and operation.error:
            st.caption(f"Not applied: {operation.error}")


def _render_project_change_card(store: CaptureStore, operation_id: str) -> None:
    operation = store.get_project_change_operation(operation_id)
    action = operation.change["action"].replace("_", " ").capitalize()
    kind = operation.change.get("kind")
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            badge = ""
            if kind in KIND_STYLES:
                name, color, icon = KIND_STYLES[kind]
                badge = f":{color}-badge[{icon} {name}] "
            st.markdown(
                f":material/view_kanban: **Tracker · {action}**  {badge}{_status_badge(operation.status)}"
            )
            if operation.status == "committed" and st.button(
                "Undo",
                icon=":material/undo:",
                type="tertiary",
                key=f"undo-project-change-{operation.id}",
            ):
                try:
                    store.undo_project_change_operation(operation.id)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.rerun()
        if content := operation.change.get("content"):
            st.caption(content)
        if operation.status == "review":
            with st.container(horizontal=True):
                if st.button(
                    "Approve",
                    type="primary",
                    icon=":material/check:",
                    key=f"approve-project-change-{operation.id}",
                ):
                    try:
                        store.approve_project_change_operation(operation.id)
                    except (KeyError, ValueError) as error:
                        st.error(str(error))
                    else:
                        st.rerun()
                if st.button(
                    "Reject", icon=":material/close:", key=f"reject-project-change-{operation.id}"
                ):
                    store.reject_project_change_operation(operation.id)
                    st.rerun()
        if operation.status == "failed" and operation.error:
            st.caption(f"Not applied: {operation.error}")


def _render_project_answer_sources(store: CaptureStore, assistant_message_id: str) -> None:
    records = store.list_project_answer_sources(assistant_message_id)
    if not records:
        return
    with st.expander(f"Sources ({len(records)})", icon=":material/link:"):
        st.caption("Project answer sources")
        for record in records:
            name, color, icon = KIND_STYLES[record.kind]
            st.markdown(f":{color}-badge[{icon} {name} · {record.state}] {record.content}")
            st.caption(f"Source: “{record.source_quote}”")


def _render_local_context_answer_sources(store: CaptureStore, assistant_message_id: str) -> None:
    sources = store.list_local_context_answer_sources(assistant_message_id)
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/link:"):
        st.caption("Local context sources")
        for source in sources:
            st.markdown(f"**{source.label}:** {source.content}")
            st.caption(f"Source: “{source.source_quote}”")


def render_tracker_page(store: CaptureStore, conversation_id: str) -> None:
    """Manage persisted project state and source-grounded records."""

    current_project = store.get_conversation_project(conversation_id)
    if current_project is None:
        st.title("Tracker")
        with st.container(border=True):
            st.subheader("No project linked")
            st.caption(
                "Tasks, blockers, questions, and decisions are tracked per project. "
                "Link this conversation to a project or create a new one."
            )
            _render_project_assignment(store, conversation_id, key_prefix="tracker")
            _render_create_project(store, key_prefix="tracker", conversation_id=conversation_id)
        return

    with st.container(horizontal=True, vertical_alignment="bottom"):
        with st.container(gap=None):
            st.caption(":material/view_kanban: Project tracker")
            st.title(current_project.name)
        if current_project.status == "active":
            with st.popover("Add record", icon=":material/add:"):
                _render_add_record_form(store, current_project, conversation_id)
        with st.popover("Settings", icon=":material/settings:"):
            _render_project_settings(store, current_project, key_prefix="tracker")
        _render_create_project(store, key_prefix="tracker")
    with st.container(horizontal=True, vertical_alignment="center"):
        if current_project.status == "archived":
            st.badge("Archived", icon=":material/archive:", color="gray")
        else:
            st.badge("Active", icon=":material/radio_button_checked:", color="green")
        if current_project.description:
            st.caption(current_project.description)

    records = store.list_project_records(
        current_project.id,
        states=("active", "completed", "resolved", "superseded", "deleted"),
    )
    live_records = [record for record in records if record.state != "deleted"]
    tasks = [record for record in live_records if record.kind == "task"]
    done_tasks = [record for record in tasks if record.state == "completed"]

    def open_count(*kinds: str) -> int:
        return sum(record.kind in kinds and record.state == "active" for record in live_records)

    open_metric, done_metric, blocker_metric, question_metric = st.columns(4)
    open_metric.metric("Open tasks", open_count("task"), border=True)
    done_metric.metric("Completed", len(done_tasks), border=True)
    blocker_metric.metric("Blockers", open_count("blocker"), border=True)
    question_metric.metric("Open questions", open_count("open_question"), border=True)
    if tasks:
        st.progress(
            len(done_tasks) / len(tasks), text=f"{len(done_tasks)} of {len(tasks)} tasks completed"
        )

    tab_labels = [
        f"{icon} {name} ({open_count(*kinds)})" for name, icon, kinds in TRACKER_GROUPS
    ]
    for tab, (name, _, kinds) in zip(st.tabs(tab_labels), TRACKER_GROUPS):
        with tab:
            _render_record_group(
                store, [record for record in live_records if record.kind in kinds], name
            )

    deleted_records = [record for record in records if record.state == "deleted"]
    if deleted_records:
        with st.expander(f"Deleted records ({len(deleted_records)})", icon=":material/delete:"):
            for record in deleted_records:
                name, color, icon = KIND_STYLES[record.kind]
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f":{color}-badge[{icon} {name}] {record.content}")
                    if st.button(
                        "Restore",
                        icon=":material/restore:",
                        type="tertiary",
                        key=f"restore-project-record-{record.id}",
                    ):
                        store.restore_project_record(record.id)
                        st.rerun()


def _render_add_record_form(
    store: CaptureStore, project: Project, conversation_id: str
) -> None:
    user_messages = [
        message for message in store.list_messages(conversation_id) if message.role == "user"
    ]
    if not user_messages:
        st.caption("Send a chat message first; tracker records require an explicit user source.")
        return
    source_id = st.selectbox(
        "Source user message",
        options=[message.id for message in user_messages],
        format_func=lambda message_id: _badge_text(store.get_message(message_id).content, 80),
        key="tracker-source-message",
    )
    kind = st.selectbox(
        "Record type",
        list(KIND_STYLES),
        format_func=lambda value: KIND_STYLES[value][0],
        key="tracker-record-kind",
    )
    content = st.text_area("Tracker text", key="tracker-record-content")
    source_message = store.get_message(source_id)
    source_quote = st.text_area(
        "Exact source quote",
        value=source_message.content,
        key=f"tracker-source-quote-{source_id}",
    )
    if st.button("Add tracker record", type="primary", key="add-tracker-record"):
        try:
            store.save_project_record(
                ProjectRecord(
                    project_id=project.id,
                    conversation_id=conversation_id,
                    kind=kind,
                    content=content,
                    state="completed" if kind == "milestone" else "active",
                    source_message_id=source_id,
                    source_quote=source_quote,
                )
            )
        except (ValidationError, ValueError) as error:
            st.error(_validation_message(error))
        else:
            st.rerun()


def _render_record_group(store: CaptureStore, records: list[ProjectRecord], name: str) -> None:
    open_records = sorted(
        (record for record in records if record.state == "active"),
        key=lambda record: (record.due_at is None, str(record.due_at or "")),
    )
    closed_records = [record for record in records if record.state != "active"]
    if not records:
        st.caption(f"No {name.lower()} yet. They appear here when chat turns create them.")
        return
    if not open_records:
        st.caption(f"No open {name.lower()}.")
    for record in open_records:
        _render_project_record(store, record)
    if closed_records:
        st.space("small")
        st.caption(f"Closed · {len(closed_records)}")
        for record in closed_records:
            _render_project_record(store, record)


def _render_project_record(store: CaptureStore, record: ProjectRecord) -> None:
    name, color, icon = KIND_STYLES[record.kind]
    is_open = record.state == "active"
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        with st.container(gap="small"):
            st.markdown(record.content if is_open else f":gray[~~{record.content}~~]")
            badges = [f":{color}-badge[{icon} {name}]"]
            if not is_open:
                badges.append(f":gray-badge[{record.state.capitalize()}]")
            if record.due_at is not None:
                badges.append(f":orange-badge[:material/schedule: Due {_format_due_at(record.due_at)}]")
            st.markdown(" ".join(badges) + f"  :gray[Updated {_format_timestamp(record.updated_at)}]")
        transition = CLOSING_TRANSITIONS.get(record.kind)
        if is_open and transition and st.button(
            transition[1],
            icon=":material/check:",
            key=f"transition-project-record-{record.id}",
        ):
            store.transition_project_record(record.id, transition[0])
            st.rerun()
        with st.popover("Details", icon=":material/more_horiz:"):
            st.caption("Source quote")
            st.markdown(f"> {record.source_quote}")
            content = st.text_area(
                "Tracker text",
                value=record.content,
                key=f"project-record-content-{record.id}",
                height=100,
            )
            rationale = st.text_input(
                "Rationale",
                value=record.rationale or "",
                key=f"project-record-rationale-{record.id}",
            )
            with st.container(horizontal=True):
                if st.button("Save", type="primary", key=f"save-project-record-{record.id}"):
                    try:
                        store.update_project_record(
                            record.id, content=content, rationale=rationale or None
                        )
                    except (ValidationError, ValueError) as error:
                        st.error(_validation_message(error))
                    else:
                        st.rerun()
                if st.button(
                    "Delete", icon=":material/delete:", key=f"delete-project-record-{record.id}"
                ):
                    store.delete_project_record(record.id)
                    st.rerun()
            events = store.list_project_record_events(record.id)
            st.caption(f"History ({len(events)})")
            for event in events:
                st.caption(
                    f"{_format_timestamp(event.created_at)} · {event.action} · "
                    f"{event.previous_state or 'none'} → {event.new_state}"
                )


def _validation_message(error: ValidationError | ValueError) -> str:
    if isinstance(error, ValidationError):
        return error.errors()[0]["msg"]
    return str(error)


def _select_calendar_day(day: date) -> None:
    st.session_state[CALENDAR_DAY_KEY] = day
    st.session_state[CALENDAR_MONTH_KEY] = day.replace(day=1)


def _shift_calendar_month(offset: int) -> None:
    month = st.session_state[CALENDAR_MONTH_KEY]
    index = month.year * 12 + month.month - 1 + offset
    st.session_state[CALENDAR_MONTH_KEY] = date(index // 12, index % 12 + 1, 1)


def render_calendar_page(store: CaptureStore) -> None:
    today = configured_reference_date(configured_timezone())
    selected_day = st.session_state.setdefault(CALENDAR_DAY_KEY, today)
    month = st.session_state.setdefault(CALENDAR_MONTH_KEY, selected_day.replace(day=1))

    events_by_day: dict[date, list[CalendarEvent]] = {}
    for event in store.list_calendar_events():
        events_by_day.setdefault(event.start_date, []).append(event)
    deadlines_by_day: dict[date, list[ProjectRecord]] = {}
    for record in store.list_task_deadlines():
        deadlines_by_day.setdefault(_day_of(record.due_at), []).append(record)

    with st.container(horizontal=True, vertical_alignment="center"):
        st.title("Calendar", width="content")
        st.space("stretch")
        st.button(
            "",
            icon=":material/chevron_left:",
            help="Previous month",
            key="calendar-previous-month",
            on_click=_shift_calendar_month,
            args=(-1,),
        )
        st.markdown(f"**{month.strftime('%B %Y')}**", width="content")
        st.button(
            "",
            icon=":material/chevron_right:",
            help="Next month",
            key="calendar-next-month",
            on_click=_shift_calendar_month,
            args=(1,),
        )
        st.button(
            "Today",
            icon=":material/today:",
            key="calendar-today",
            on_click=_select_calendar_day,
            args=(today,),
        )

    grid_column, day_column = st.columns([2.4, 1], gap="large")
    with grid_column:
        header = st.columns(7, gap="small")
        for column, weekday in zip(header, WEEKDAYS):
            column.caption(weekday, text_alignment="center")
        for week in Calendar().monthdatescalendar(month.year, month.month):
            for column, day in zip(st.columns(7, gap="small"), week):
                with column:
                    _render_calendar_cell(
                        day,
                        month=month,
                        today=today,
                        selected_day=selected_day,
                        events=events_by_day.get(day, []),
                        deadlines=deadlines_by_day.get(day, []),
                    )
    with day_column:
        _render_calendar_day(
            store,
            selected_day,
            today=today,
            events=events_by_day.get(selected_day, []),
            deadlines=deadlines_by_day.get(selected_day, []),
        )

    history = store.list_calendar_events(statuses=("deleted",))
    if history:
        with st.expander(f"Deleted events ({len(history)})", icon=":material/history:"):
            for event in history:
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(
                        f":gray-badge[{_format_calendar_timing(event.start_at, event.end_at)}] "
                        f"{event.title}"
                    )
                    if st.button(
                        "Restore",
                        icon=":material/restore:",
                        type="tertiary",
                        key=f"restore-event-{event.id}",
                    ):
                        store.set_calendar_event_status(event.id, "active")
                        st.rerun()


def _render_calendar_cell(
    day: date,
    *,
    month: date,
    today: date,
    selected_day: date,
    events: list[CalendarEvent],
    deadlines: list[ProjectRecord],
) -> None:
    if day == selected_day:
        state = "selected"
    elif day.month != month.month:
        state = "muted"
    elif day == today:
        state = "today"
    else:
        state = "day"
    with st.container(border=True, key=f"calcell-{state}-{day.isoformat()}"):
        st.button(
            str(day.day),
            type="primary" if day == selected_day else "tertiary",
            key=f"calendar-day-{day.isoformat()}",
            on_click=_select_calendar_day,
            args=(day,),
        )
        chips = [
            f":blue-badge[{_badge_text(event.start_at.strftime('%H:%M') + ' ' + event.title, 24)}]"
            if isinstance(event.start_at, datetime)
            else f":blue-badge[{_badge_text(event.title, 24)}]"
            for event in events
        ] + [f":orange-badge[{_badge_text('Due ' + record.content, 24)}]" for record in deadlines]
        for chip in chips[:2]:
            st.markdown(chip)
        if len(chips) > 2:
            st.caption(f"+{len(chips) - 2} more")


def _render_calendar_day(
    store: CaptureStore,
    day: date,
    *,
    today: date,
    events: list[CalendarEvent],
    deadlines: list[ProjectRecord],
) -> None:
    with st.container(border=True):
        st.subheader(day.strftime("%A, %b %d"))
        offset = (day - today).days
        st.caption(
            "Today"
            if offset == 0
            else "Tomorrow"
            if offset == 1
            else "Yesterday"
            if offset == -1
            else f"In {offset} days"
            if offset > 0
            else f"{-offset} days ago"
        )
        if not events and not deadlines:
            st.caption("Nothing scheduled for this day.")
        for record in deadlines:
            with st.container(border=True):
                st.markdown(f":orange-badge[:material/flag: Deadline] **{record.content}**")
                st.caption(
                    f"Due {_format_due_at(record.due_at)} · {store.get_project(record.project_id).name}"
                )
        for event in events:
            _render_calendar_event(store, event)

    coming_up = [
        entry for entry in _upcoming_entries(store, max(day, today)) if entry[0] > day
    ][:5]
    if coming_up:
        st.space("small")
        st.caption("Coming up")
        for entry_day, entry in coming_up:
            with st.container(horizontal=True, vertical_alignment="center"):
                title = entry.content if isinstance(entry, ProjectRecord) else entry.title
                color = "orange" if isinstance(entry, ProjectRecord) else "blue"
                st.markdown(f":{color}-badge[{entry_day.strftime('%b %d')}] {title}", width="stretch")
                st.button(
                    "",
                    icon=":material/arrow_forward:",
                    type="tertiary",
                    help="Show this day",
                    key=f"calendar-coming-up-{entry.id}",
                    on_click=_select_calendar_day,
                    args=(entry_day,),
                )


def _render_calendar_event(store: CaptureStore, event: CalendarEvent) -> None:
    with st.container(border=True):
        with st.container(
            horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", gap="xsmall"
        ):
            st.markdown(f":blue-badge[:material/schedule: {_format_time_of_day(event)}]", width="content")
            with st.container(horizontal=True, width="content", gap="xxsmall"):
                with st.popover("", icon=":material/edit:", help="Edit event"):
                    edited_title = st.text_input(
                        "Title", value=event.title, key=f"event-title-{event.id}"
                    )
                    if event.all_day:
                        edited_start = st.date_input(
                            "Date", value=event.start_at, key=f"event-date-{event.id}"
                        )
                    else:
                        edited_start = event.start_at
                        st.caption("Ask in chat to move a timed event, so its end time stays consistent.")
                    if st.button("Save event", type="primary", key=f"save-event-{event.id}"):
                        try:
                            store.update_calendar_event(
                                event.id, title=edited_title, start_at=edited_start
                            )
                        except ValidationError as error:
                            st.error(_validation_message(error))
                        else:
                            st.rerun()
                if st.button(
                    "",
                    icon=":material/delete:",
                    type="tertiary",
                    help="Delete event",
                    key=f"delete-event-{event.id}",
                ):
                    store.set_calendar_event_status(event.id, "deleted")
                    st.rerun()
        st.markdown(f"**{event.title}**")
        if event.description:
            st.caption(event.description)
        _render_calendar_sources(store, event)


def _render_calendar_sources(store: CaptureStore, event: CalendarEvent) -> None:
    sources = store.list_calendar_event_sources(event.id)
    with st.expander(f"Sources ({len(sources)})", icon=":material/link:"):
        for source in sources:
            st.caption(source.source_quote)
        st.caption(f"Updated {_format_timestamp(event.updated_at)}")


def render_saved_items_page(store: CaptureStore) -> None:
    st.title("Ideas & Notes")
    with st.container(horizontal=True, vertical_alignment="center"):
        kind_label = st.segmented_control(
            "Type",
            ["All", "Ideas", "Notes"],
            default="All",
            required=True,
            key="saved_items_kind",
            label_visibility="collapsed",
        )
        query = st.text_input(
            "Filter ideas and notes",
            placeholder="Search ideas and notes",
            icon=":material/search:",
            label_visibility="collapsed",
        )
    kinds = {"All": ("idea", "note"), "Ideas": ("idea",), "Notes": ("note",)}[kind_label]
    items = store.list_saved_items(kinds=kinds)
    if query.strip():
        items = [item for item in items if query.strip().lower() in item.content.lower()]
    if not items:
        st.caption("No ideas or notes match this view. Use `/idea` or `/note` in chat to add one.")

    columns = st.columns(2, gap="medium")
    for index, item in enumerate(items):
        with columns[index % 2], st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                if item.kind == "idea":
                    st.badge("Idea", icon=":material/lightbulb:", color="violet")
                else:
                    st.badge("Note", icon=":material/sticky_note_2:", color="gray")
                st.caption(_format_timestamp(item.updated_at), width="content")
                st.space("stretch")
                with st.popover("", icon=":material/edit:", help="Edit"):
                    edited_content = st.text_area(
                        "Content", value=item.content, key=f"item-content-{item.id}", height=140
                    )
                    if st.button("Save item", type="primary", key=f"save-item-{item.id}"):
                        try:
                            store.update_saved_item(item.id, content=edited_content)
                        except ValidationError as error:
                            st.error(_validation_message(error))
                        else:
                            st.rerun()
                if st.button(
                    "",
                    icon=":material/delete:",
                    type="tertiary",
                    help="Delete",
                    key=f"delete-item-{item.id}",
                ):
                    store.set_saved_item_status(item.id, "deleted")
                    st.rerun()
            st.markdown(item.content)
            if item.source_quote.strip() != item.content.strip():
                st.caption(f":material/format_quote: {item.source_quote}")

    history = store.list_saved_items(kinds=("idea", "note"), statuses=("deleted",))
    if history:
        with st.expander(f"Deleted ideas & notes ({len(history)})", icon=":material/history:"):
            for item in history:
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f":gray-badge[{item.kind.title()}] {item.content}")
                    if st.button(
                        "Restore",
                        icon=":material/restore:",
                        type="tertiary",
                        key=f"restore-item-{item.id}",
                    ):
                        store.set_saved_item_status(item.id, "active")
                        st.rerun()
