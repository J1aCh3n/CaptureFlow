import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from .llm import ModelError, ModelSettings, OpenAICompatibleModel
from .memory_service import MemoryService
from .models import Capture, CaptureRun, Conversation, MAX_CAPTURE_CHARACTERS, RunError
from .phase_three import PhaseThreeWorkflow
from .storage import CaptureStore
from .workflow import configured_reference_date


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
    st.caption(
        "Phase 3 in progress: bounded planning plus policy-controlled memory proposals."
    )

    store = get_store(str(database_path()))
    if st.sidebar.radio("Page", ["Inbox", "Memory"]) == "Memory":
        render_memory_page(store)
        return
    conversations = store.list_conversations()
    if not conversations:
        conversation = store.save_conversation(Conversation(title="First capture"))
        conversations = [conversation]

    with st.expander("New conversation"):
        title = st.text_input("Title", key="new_conversation_title")
        if st.button("Create conversation"):
            try:
                created = store.save_conversation(Conversation(title=title))
            except ValidationError as error:
                st.error(error.errors()[0]["msg"])
            else:
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
