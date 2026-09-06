from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from .models import Capture, Conversation, MAX_CAPTURE_CHARACTERS
from .storage import CaptureStore


DATABASE_PATH = Path("data") / "captureflow.sqlite3"


@st.cache_resource
def get_store() -> CaptureStore:
    store = CaptureStore(DATABASE_PATH)
    store.initialize()
    return store


def main() -> None:
    st.set_page_config(page_title="CaptureFlow", page_icon="📥")
    st.title("CaptureFlow")
    st.caption("Phase 1: deterministic local capture storage. No model is called.")

    store = get_store()
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

    st.subheader("Manual capture")
    source_text = st.text_area(
        "Paste or write plain text",
        max_chars=MAX_CAPTURE_CHARACTERS,
        height=180,
        placeholder="This is stored exactly as entered; Phase 1 does not summarize or analyze it.",
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
            if st.button("Delete", key=f"delete-{capture.id}"):
                store.delete_capture(capture.id)
                st.rerun()
