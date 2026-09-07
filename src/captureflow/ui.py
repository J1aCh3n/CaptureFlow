import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from .llm import ModelError, ModelSettings, OpenAICompatibleModel
from .models import Capture, CaptureRun, Conversation, MAX_CAPTURE_CHARACTERS, RunError
from .storage import CaptureStore
from .workflow import PhaseTwoWorkflow, configured_reference_date


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
        "Phase 2: bounded planning, grounded summaries and tasks, and one revision at most."
    )

    store = get_store(str(database_path()))
    conversations = store.list_conversations()
    if not conversations:
        conversation = store.save_conversation(Conversation(title="First capture"))
        conversations = [conversation]

    selected_id = st.selectbox(
        "Conversation",
        options=[conversation.id for conversation in conversations],
        format_func=lambda conversation_id: next(
            conversation.title
            for conversation in conversations
            if conversation.id == conversation_id
        ),
    )

    with st.expander("New conversation"):
        title = st.text_input("Title", key="new_conversation_title")
        if st.button("Create conversation"):
            try:
                created = store.save_conversation(Conversation(title=title))
            except ValidationError as error:
                st.error(error.errors()[0]["msg"])
            else:
                st.success(f"Created {created.title}.")
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
                run = analyze_capture(capture)
                store.save_run(run)
                st.rerun()
            render_run(store.latest_run_for_capture(capture.id))
            render_run_history(store.list_runs_for_capture(capture.id))
            if st.button("Delete", key=f"delete-{capture.id}"):
                store.delete_capture(capture.id)
                st.rerun()


def analyze_capture(capture: Capture) -> CaptureRun:
    run_id = str(uuid4())
    try:
        settings = ModelSettings.from_environment()
        timezone = os.getenv("CAPTUREFLOW_TIMEZONE", "America/Vancouver")
        reference_date = configured_reference_date(timezone)
        workflow = PhaseTwoWorkflow(OpenAICompatibleModel(settings))
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


def render_run(run: CaptureRun | None) -> None:
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
    if run.status == "completed" and run.output:
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
