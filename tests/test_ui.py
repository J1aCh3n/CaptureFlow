import os
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from captureflow.models import (
    CalendarEvent,
    Conversation,
    Message,
    PendingClarification,
    Project,
    ProjectChangeOperation,
    ProjectRecord,
    SavedItem,
)
from captureflow.storage import CaptureStore
from captureflow.workflow import configured_reference_date


class _CalendarRoutingUiModel:
    """A deterministic Phase 6 seam: AppTest never reaches an external model."""

    def propose_conversation_decision(
        self,
        *,
        current_message: dict[str, str],
        candidate_events: list[dict],
        reference_date: str,
        **_: object,
    ) -> dict:
        evidence = [
            {
                "message_id": current_message["message_id"],
                "source_quote": current_message["content"],
            }
        ]
        if candidate_events:
            return {
                "intent": "calendar_update",
                "action": {
                    "action": "update_event",
                    "target_event_id": candidate_events[0]["id"],
                    "start_time": "15:00",
                    "evidence": evidence,
                },
            }
        return {
            "intent": "calendar_create",
            "action": {
                "action": "create_event",
                "content": "Algorithm review",
                "description": "Discuss the algorithm bottleneck.",
                "date": (date.fromisoformat(reference_date) + timedelta(days=1)).isoformat(),
                "start_time": "14:00",
                "end_time": "15:00",
                "evidence": evidence,
            },
        }


class _LocalAnswerUiModel:
    def propose_conversation_decision(
        self,
        *,
        candidate_events: list[dict],
        candidate_saved_items: list[dict],
        **_: object,
    ) -> dict:
        return {
            "intent": "local_answer",
            "reply": "Use the selected local records.",
            "source_ids": [
                *[event["id"] for event in candidate_events],
                *[item["id"] for item in candidate_saved_items],
            ],
        }


class UiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.previous_database_path = os.environ.get("CAPTUREFLOW_DATABASE_PATH")
        self.previous_model_settings = {
            key: os.environ.get(key)
            for key in (
                "CAPTUREFLOW_API_KEY",
                "CAPTUREFLOW_MODEL",
                "CAPTUREFLOW_BASE_URL",
                "CAPTUREFLOW_TIMEZONE",
            )
        }
        os.environ["CAPTUREFLOW_DATABASE_PATH"] = str(
            Path(self.temporary_directory.name) / "captureflow.sqlite3"
        )
        for key in self.previous_model_settings:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        if self.previous_database_path is None:
            os.environ.pop("CAPTUREFLOW_DATABASE_PATH", None)
        else:
            os.environ["CAPTUREFLOW_DATABASE_PATH"] = self.previous_database_path
        for key, value in self.previous_model_settings.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temporary_directory.cleanup()

    @staticmethod
    def open_page(app: AppTest, page: str) -> AppTest:
        app.sidebar.radio[0].set_value(page).run()
        return app

    @staticmethod
    def open_project(app: AppTest, project_id: str) -> AppTest:
        next(
            field for field in app.selectbox if field.key == "sidebar_project_id"
        ).set_value(project_id).run()
        return app

    def test_current_chat_shell_renders_without_a_configured_model(self) -> None:
        app = AppTest.from_file("app.py").run()

        self.assertEqual(app.exception, [])
        self.assertEqual(app.title[0].value, "CaptureFlow")
        self.assertIn("project continuity assistant", app.caption[0].value)
        self.assertEqual(len(app.chat_input), 1)
        self.assertEqual(app.sidebar.toggle[0].label, "Diagnostics")
        self.assertFalse(app.sidebar.toggle[0].value)

    def test_navigation_exposes_only_current_product_pages(self) -> None:
        app = AppTest.from_file("app.py").run()

        self.assertEqual(
            list(app.sidebar.radio[0].options),
            ["Chat", "Tracker", "Calendar", "Ideas & Notes"],
        )
        self.assertEqual(
            next(field for field in app.selectbox if field.key == "sidebar_project_id").label,
            "Project",
        )
        self.assertEqual(
            next(
                field for field in app.sidebar.radio if field.key == "selected_conversation_id"
            ).label,
            "Conversations",
        )

    def test_project_can_be_renamed_and_deleted_from_tracker(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        conversation = store.list_conversations()[0]
        project = store.save_project(Project(id="ui-manage-project", name="Before rename"))
        store.assign_conversation_project(conversation.id, project.id)

        self.open_project(app.run(), project.id)
        self.open_page(app, "Tracker")
        next(
            field
            for field in app.text_input
            if field.key == f"tracker-project-name-{project.id}"
        ).set_value("After rename").run()
        next(
            button
            for button in app.button
            if button.key == f"tracker-save-project-{project.id}"
        ).click().run()

        self.assertEqual(store.get_project(project.id).name, "After rename")
        next(
            checkbox
            for checkbox in app.checkbox
            if checkbox.key == f"tracker-confirm-delete-project-{project.id}"
        ).check().run()
        next(
            button
            for button in app.button
            if button.key == f"tracker-delete-project-{project.id}"
        ).click().run()

        with self.assertRaises(KeyError):
            store.get_project(project.id)
        self.assertEqual(store.get_conversation(conversation.id), conversation)
        self.assertIsNone(store.get_conversation_project(conversation.id))

    def test_chat_workspace_shows_cross_conversation_project_snapshot(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        selected_conversation = store.list_conversations()[0]
        project = store.save_project(Project(id="snapshot-project", name="CaptureFlow"))
        store.assign_conversation_project(selected_conversation.id, project.id)
        source_conversation = store.save_conversation(
            Conversation(id="snapshot-source-conversation", title="Project planning")
        )
        store.assign_conversation_project(source_conversation.id, project.id)
        task_source = store.save_message(
            Message(
                id="snapshot-task-source",
                conversation_id=source_conversation.id,
                role="user",
                content="Prepare the UI workspace.",
            )
        )
        store.save_project_record(
            ProjectRecord(
                id="snapshot-task",
                project_id=project.id,
                conversation_id=source_conversation.id,
                kind="task",
                content="Prepare the UI workspace.",
                state="active",
                source_message_id=task_source.id,
                source_quote=task_source.content,
            )
        )
        event_source = store.save_message(
            Message(
                id="snapshot-event-source",
                conversation_id=source_conversation.id,
                role="user",
                content="Review the new workspace tomorrow.",
            )
        )
        event_date = configured_reference_date("America/Vancouver") + timedelta(days=1)
        store.save_calendar_event(
            CalendarEvent(
                id="snapshot-event",
                conversation_id=source_conversation.id,
                title="Workspace review",
                source_message_id=event_source.id,
                source_quote=event_source.content,
                start_at=event_date,
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        note_source = store.save_message(
            Message(
                id="snapshot-note-source",
                conversation_id=source_conversation.id,
                role="user",
                content="Keep the project snapshot read-only.",
            )
        )
        store.save_saved_item(
            SavedItem(
                id="snapshot-note",
                conversation_id=source_conversation.id,
                kind="note",
                content="Keep the project snapshot read-only.",
                source_message_id=note_source.id,
                source_quote=note_source.content,
            )
        )

        self.open_project(app.run(), project.id)

        self.assertEqual(app.exception, [])
        self.assertTrue(any(item.value == "Project snapshot" for item in app.subheader))
        metrics = {metric.label: str(metric.value) for metric in app.metric}
        self.assertEqual(metrics, {"Active": "1", "Done": "0", "Blockers": "0"})
        rendered_markdown = [item.value for item in app.markdown]
        self.assertTrue(any("Project progress" in value for value in rendered_markdown))
        self.assertTrue(any("Today" in value and "Priority" in value for value in rendered_markdown))
        self.assertTrue(any("Prepare the UI workspace." in value for value in rendered_markdown))
        self.assertTrue(any("Workspace review" in value for value in rendered_markdown))
        self.assertTrue(
            any("Keep the project snapshot read-only." in value for value in rendered_markdown)
        )

    def test_diagnostics_switch_shows_session_only_turn_timing(self) -> None:
        app = AppTest.from_file("app.py").run()
        self.assertFalse(any(expander.label.startswith("Diagnostics ·") for expander in app.expander))

        app.sidebar.toggle[0].set_value(True).run()
        app.chat_input[0].set_value("/idea Measure model latency.").run()

        self.assertTrue(any(expander.label.startswith("Diagnostics ·") for expander in app.expander))
        self.assertTrue(any("Model calls: 0" in caption.value for caption in app.caption))
        self.assertTrue(any("Session-only timing" in caption.value for caption in app.caption))

    def test_unconfigured_router_persists_a_visible_no_write_reply(self) -> None:
        app = AppTest.from_file("app.py").run()
        app.chat_input[0].set_value("My report is due tomorrow.").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()

        self.assertEqual(len(store.list_calendar_events()), 0)
        self.assertTrue(any("CAPTUREFLOW_API_KEY" in element.value for element in app.markdown))

    def test_ordinary_chat_routes_calendar_create_and_update_through_contextual_router(self) -> None:
        os.environ["CAPTUREFLOW_API_KEY"] = "ui-test-key"
        os.environ["CAPTUREFLOW_MODEL"] = "ui-test-model"
        with patch(
            "captureflow.ui.OpenAICompatibleModel",
            return_value=_CalendarRoutingUiModel(),
        ):
            app = AppTest.from_file("app.py").run()
            app.sidebar.toggle[0].set_value(True).run()
            app.chat_input[0].set_value(
                "Tomorrow at 14:00 I have an algorithm review meeting."
            ).run()
            app.chat_input[0].set_value("Move that review to 15:00.").run()

        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        event = store.list_calendar_events()[0]
        self.assertEqual(event.title, "Algorithm review")
        self.assertEqual(event.start_at.hour, 15)
        update_operations = [
            operation
            for message in store.list_messages(store.list_conversations()[0].id)
            if message.role == "user"
            for operation in store.list_calendar_event_update_operations(message.id)
        ]
        self.assertEqual(len(update_operations), 1)
        self.assertTrue(
            any(button.key.startswith("undo-calendar-update-") for button in app.button)
        )
        self.assertTrue(any("Intent: calendar_update" in caption.value for caption in app.caption))

    def test_pending_clarification_has_a_visible_cancel_control(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        conversation = store.list_conversations()[0]
        user_message = store.save_message(
            Message(
                id="clarification-user",
                conversation_id=conversation.id,
                role="user",
                content="My report is due Tuesday.",
            )
        )
        assistant_message = store.save_message(
            Message(
                id="clarification-assistant",
                conversation_id=conversation.id,
                role="assistant",
                content="Which exact date should I use?",
                reply_to_message_id=user_message.id,
            )
        )
        clarification = store.save_pending_clarification(
            PendingClarification(
                id="pending-date",
                conversation_id=conversation.id,
                source_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
                original_action={"action": "create_event", "content": "Submit report"},
            )
        )

        app.run()
        next(
            button
            for button in app.button
            if button.key == f"cancel-clarification-{clarification.id}"
        ).click().run()
        self.assertIsNone(store.get_pending_clarification(conversation.id))

    def test_calendar_page_shows_tracker_deadlines_without_making_them_events(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        conversation = store.list_conversations()[0]
        message = store.save_message(
            Message(
                id="deadline-user-message",
                conversation_id=conversation.id,
                role="user",
                content="修复需要下周三晚上八点之前提交",
            )
        )
        project = store.save_project(Project(id="deadline-project", name="测试2"))
        store.assign_conversation_project(conversation.id, project.id)
        record = store.save_project_record(
            ProjectRecord(
                id="deadline-record",
                project_id=project.id,
                conversation_id=conversation.id,
                kind="task",
                content="提交修复",
                due_at=datetime.fromisoformat("2026-09-16T20:00:00-07:00"),
                state="active",
                source_message_id=message.id,
                source_quote="修复需要下周三晚上八点之前提交",
            )
        )

        self.open_page(app.run(), "Calendar")

        self.assertEqual(app.exception, [])
        self.assertTrue(any(record.content in element.value for element in app.markdown))
        self.assertTrue(any("Sep 16 · 20:00" in element.value for element in app.caption))
        self.assertEqual(store.list_calendar_events(), [])
        self.assertFalse(any(button.key.startswith("delete-event-") for button in app.button))

        store.transition_project_record(record.id, "completed")
        app.run()

        self.assertFalse(any(record.content in element.value for element in app.markdown))

    def test_tracker_page_renders_a_source_grounded_project_record(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        conversation = store.list_conversations()[0]
        message = store.save_message(
            Message(
                id="tracker-user-message",
                conversation_id=conversation.id,
                role="user",
                content="Phase 4 needs a deterministic tracker foundation.",
            )
        )
        project = store.save_project(Project(id="tracker-project", name="CaptureFlow"))
        store.assign_conversation_project(conversation.id, project.id)
        record = store.save_project_record(
            ProjectRecord(
                id="tracker-record",
                project_id=project.id,
                conversation_id=conversation.id,
                kind="task",
                content="Implement the tracker foundation.",
                state="active",
                source_message_id=message.id,
                source_quote="Phase 4 needs a deterministic tracker foundation.",
            )
        )

        self.open_project(app.run(), project.id)
        self.open_page(app, "Tracker")

        self.assertEqual(app.exception, [])
        self.assertTrue(any(project.name in element.value for element in app.markdown))
        self.assertTrue(any(field.value == record.content for field in app.text_area))
        self.assertTrue(
            any(button.key == f"transition-project-record-{record.id}" for button in app.button)
        )

    def test_unconfigured_project_chat_persists_a_no_write_contextual_turn(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        conversation = store.list_conversations()[0]
        project = store.save_project(Project(id="ui-project-context", name="CaptureFlow"))
        store.assign_conversation_project(conversation.id, project.id)

        self.open_project(app.run(), project.id)
        app.chat_input[0].set_value("Continue the project discussion.").run()

        messages = store.list_messages(conversation.id)
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertTrue(store.conversation_has_project_turn(conversation.id))
        self.assertEqual(store.list_project_records(project.id), [])
        self.assertIn("CAPTUREFLOW_API_KEY", messages[-1].content)

    def test_project_change_review_card_is_visible_in_the_chat_timeline(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        conversation = store.list_conversations()[0]
        project = store.save_project(Project(id="ui-project-change", name="CaptureFlow"))
        store.assign_conversation_project(conversation.id, project.id)
        source = store.save_message(
            Message(
                id="ui-project-change-source",
                conversation_id=conversation.id,
                role="user",
                content="Replace the tracker decision with SQLite and FTS5.",
            )
        )
        store.save_message(
            Message(
                id="ui-project-change-reply",
                conversation_id=conversation.id,
                role="assistant",
                content="This needs review.",
                reply_to_message_id=source.id,
            )
        )
        operation = store.save_project_change_operation(
            ProjectChangeOperation(
                id="ui-project-change-operation",
                source_message_id=source.id,
                project_id=project.id,
                change_index=0,
                change={
                    "action": "create_record",
                    "kind": "decision",
                    "content": "Use SQLite and FTS5.",
                    "source_quote": "Replace the tracker decision with SQLite and FTS5.",
                },
                status="review",
            )
        )

        self.open_project(app.run(), project.id)

        self.assertTrue(
            any(button.key == f"approve-project-change-{operation.id}" for button in app.button)
        )
        self.assertTrue(
            any(button.key == f"reject-project-change-{operation.id}" for button in app.button)
        )

    def test_project_answer_sources_are_visible_in_the_chat_timeline(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        conversation = store.list_conversations()[0]
        project = store.save_project(Project(id="ui-project-answer", name="CaptureFlow"))
        store.assign_conversation_project(conversation.id, project.id)
        source = store.save_message(
            Message(
                id="ui-project-answer-source",
                conversation_id=conversation.id,
                role="user",
                content="Use SQLite for the local project.",
            )
        )
        record = store.save_project_record(
            ProjectRecord(
                id="ui-project-answer-record",
                project_id=project.id,
                conversation_id=conversation.id,
                kind="decision",
                content="Use SQLite for the local project.",
                state="active",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        answer = store.save_message(
            Message(
                id="ui-project-answer-reply",
                conversation_id=conversation.id,
                role="assistant",
                content="SQLite is the local decision.",
                reply_to_message_id=source.id,
            )
        )
        store.save_project_answer_sources(answer.id, [record.id])

        self.open_project(app.run(), project.id)

        self.assertTrue(any("Project answer sources" in caption.value for caption in app.caption))
        self.assertTrue(any(record.content in markdown.value for markdown in app.markdown))

    def test_duplicate_titles_select_the_new_conversation_and_keep_histories_separate(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()

        app.text_input[0].input("Repeated title").run()
        next(button for button in app.button if button.label == "Create conversation").click().run()
        first = next(
            conversation
            for conversation in store.list_conversations()
            if conversation.title == "Repeated title"
        )
        app.chat_input[0].set_value("/note Belongs only to the first conversation.").run()

        app.text_input[0].input("Repeated title").run()
        next(button for button in app.button if button.label == "Create conversation").click().run()
        matches = [
            conversation
            for conversation in store.list_conversations()
            if conversation.title == "Repeated title"
        ]
        newest = matches[0]

        self.assertEqual(len(matches), 2)
        self.assertNotEqual(first.id, newest.id)
        conversation_selector = next(
            field for field in app.sidebar.radio if field.key == "selected_conversation_id"
        )
        self.assertEqual(conversation_selector.value, newest.id)
        self.assertEqual(
            [item.conversation_id for item in store.list_saved_items()], [first.id]
        )
        self.assertEqual(len(conversation_selector.options), len(set(conversation_selector.options)))

    def test_confirmed_conversation_delete_removes_only_the_selected_conversation(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        retained_id = store.list_conversations()[0].id

        app.text_input[0].input("Delete this conversation").run()
        next(button for button in app.button if button.label == "Create conversation").click().run()
        conversation_selector = next(
            field for field in app.sidebar.radio if field.key == "selected_conversation_id"
        )
        deleted_id = conversation_selector.value
        app.chat_input[0].set_value("/note This note should be deleted with its conversation.").run()

        next(
            checkbox
            for checkbox in app.checkbox
            if checkbox.label == "I understand this deletes the conversation and its derived data."
        ).check().run()
        next(button for button in app.button if button.label == "Delete conversation").click().run()

        with self.assertRaises(KeyError):
            store.get_conversation(deleted_id)
        conversation_selector = next(
            field for field in app.sidebar.radio if field.key == "selected_conversation_id"
        )
        self.assertEqual(conversation_selector.value, retained_id)
        self.assertEqual(store.get_conversation(retained_id).id, retained_id)

    def test_chat_command_writes_once_and_is_visible_on_calendar_page(self) -> None:
        app = AppTest.from_file("app.py").run()

        app.chat_input[0].set_value("/event 2026-09-15 | Submit report").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()

        self.assertEqual(app.exception, [])
        self.assertEqual(len(store.list_messages(store.list_conversations()[0].id)), 2)
        self.assertEqual(len(store.list_calendar_events()), 1)
        self.open_page(app, "Calendar")
        self.assertTrue(any("Submit report" in element.value for element in app.text_input))

    def test_unassigned_chat_reads_local_meeting_context_through_contextual_router(self) -> None:
        os.environ["CAPTUREFLOW_API_KEY"] = "ui-test-key"
        os.environ["CAPTUREFLOW_MODEL"] = "ui-test-model"
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        source_conversation = store.list_conversations()[0]
        later_conversation = store.save_conversation(Conversation(id="ui-local-context-later", title="Later"))
        source = store.save_message(
            Message(
                id="ui-local-context-source",
                conversation_id=source_conversation.id,
                role="user",
                content="The workflow meeting is tomorrow. It uses voting and idea merging.",
            )
        )
        reference_date = configured_reference_date("America/Vancouver")
        event = store.save_calendar_event(
            CalendarEvent(
                id="ui-local-context-event",
                conversation_id=source_conversation.id,
                title="Workflow meeting",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=reference_date + timedelta(days=1),
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        note = store.save_saved_item(
            SavedItem(
                id="ui-local-context-note",
                conversation_id=source_conversation.id,
                kind="note",
                content="会议用于投票、合并和深度优化点子。",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )

        with patch(
            "captureflow.ui.OpenAICompatibleModel",
            return_value=_LocalAnswerUiModel(),
        ):
            app.run()
            next(
                field for field in app.sidebar.radio if field.key == "selected_conversation_id"
            ).set_value(later_conversation.id).run()
            app.chat_input[0].set_value("我明天的会议要做什么？").run()

        answer = store.get_assistant_reply(store.list_messages(later_conversation.id)[0].id)
        self.assertIsNotNone(answer)
        self.assertIn(note.content, answer.content)
        self.assertEqual(
            {source.reference.source_id for source in store.list_local_context_answer_sources(answer.id)},
            {event.id, note.id},
        )
        self.assertTrue(any("Local context sources" in caption.value for caption in app.caption))

    def test_new_conversation_can_be_assigned_to_a_project_from_the_normal_flow(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        project = store.save_project(Project(id="ui-new-conversation-project", name="CaptureFlow"))

        app.run()
        next(field for field in app.selectbox if field.key == "sidebar_project_id").set_value(
            project.id
        ).run()
        next(field for field in app.text_input if field.key == "new_conversation_title").set_value(
            "Project follow-up"
        ).run()
        next(button for button in app.button if button.label == "Create conversation").click().run()

        created = next(
            conversation for conversation in store.list_conversations() if conversation.title == "Project follow-up"
        )
        self.assertEqual(store.get_conversation_project(created.id), project)

    def test_saved_item_can_be_deleted_and_restored_from_management_page(self) -> None:
        app = AppTest.from_file("app.py").run()
        app.chat_input[0].set_value("/note Keep SQLite.").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        item = store.list_saved_items()[0]

        self.open_page(app, "Ideas & Notes")
        next(button for button in app.button if button.key == f"delete-item-{item.id}").click().run()
        self.assertEqual(store.get_saved_item(item.id).status, "deleted")
        next(button for button in app.button if button.key == f"restore-item-{item.id}").click().run()
        self.assertEqual(store.get_saved_item(item.id).status, "active")

    def test_calendar_event_can_be_edited_deleted_and_restored(self) -> None:
        app = AppTest.from_file("app.py").run()
        app.chat_input[0].set_value("/event 2026-09-15 | Submit report").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        event = store.list_calendar_events()[0]

        self.open_page(app, "Calendar")
        self.assertTrue(any(element.label == "Sources (1)" for element in app.expander))
        next(field for field in app.text_input if field.key == f"event-title-{event.id}").set_value(
            "Submit final report"
        ).run()
        next(button for button in app.button if button.key == f"save-event-{event.id}").click().run()
        self.assertEqual(store.get_calendar_event(event.id).title, "Submit final report")
        next(button for button in app.button if button.key == f"delete-event-{event.id}").click().run()
        self.assertEqual(store.get_calendar_event(event.id).status, "deleted")
        next(button for button in app.button if button.key == f"restore-event-{event.id}").click().run()
        self.assertEqual(store.get_calendar_event(event.id).status, "active")

    def test_chat_operation_can_be_undone_inline(self) -> None:
        app = AppTest.from_file("app.py").run()
        app.chat_input[0].set_value("/idea Test the import path.").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        message = next(message for message in store.list_messages(store.list_conversations()[0].id) if message.role == "user")
        operation = store.list_tool_operations(message.id)[0]

        next(button for button in app.button if button.key == f"undo-operation-{operation.id}").click().run()
        self.assertEqual(store.get_tool_operation(operation.id).status, "undone")
        self.assertEqual(store.get_saved_item(operation.result["item_id"]).status, "deleted")
