import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from captureflow.models import (
    CalendarEvent,
    Conversation,
    Message,
    PendingClarification,
    SavedItem,
    ToolOperation,
)
from captureflow.storage import CaptureStore


class V2StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "captureflow.sqlite3"
        self.store = CaptureStore(self.database_path)
        self.store.initialize()
        self.conversation = self.store.save_conversation(
            Conversation(id="conversation-1", title="V2")
        )
        self.user_message = self.store.save_message(
            Message(
                id="message-1",
                conversation_id=self.conversation.id,
                role="user",
                content="Submit the report on September 15.",
            )
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_messages_are_idempotent_ordered_and_restart_safe(self) -> None:
        assistant = Message(
            id="assistant-message",
            conversation_id=self.conversation.id,
            role="assistant",
            content="Created the event.",
            reply_to_message_id=self.user_message.id,
            created_at=self.user_message.created_at,
        )
        self.store.save_message(self.user_message)
        self.store.save_message(assistant)

        restarted = CaptureStore(self.database_path)
        restarted.initialize()

        self.assertEqual(
            restarted.list_messages(self.conversation.id),
            [self.user_message, assistant],
        )

    def test_v2_records_round_trip_through_sqlite(self) -> None:
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="event-1",
                conversation_id=self.conversation.id,
                title="Submit report",
                source_message_id=self.user_message.id,
                source_quote=self.user_message.content,
                start_at=date(2026, 9, 15),
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        saved_item = self.store.save_saved_item(
            SavedItem(
                id="item-1",
                conversation_id=self.conversation.id,
                kind="note",
                content="Keep the migration additive.",
                source_message_id=self.user_message.id,
                source_quote=self.user_message.content,
            )
        )
        operation = self.store.save_tool_operation(
            ToolOperation(
                id="operation-1",
                source_message_id=self.user_message.id,
                action_index=0,
                tool_name="create_local_event",
                request={"title": event.title},
                result={"event_id": event.id},
                status="committed",
            )
        )

        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_events(), [event])
        self.assertEqual(self.store.get_saved_item(saved_item.id), saved_item)
        self.assertEqual(self.store.list_saved_items(), [saved_item])
        self.assertEqual(self.store.get_tool_operation(operation.id), operation)
        self.assertEqual(self.store.list_tool_operations(self.user_message.id), [operation])

    def test_timed_event_round_trip_preserves_datetime(self) -> None:
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="event-timed",
                conversation_id=self.conversation.id,
                title="Review results",
                description="Compare simulation benchmark results.",
                source_message_id=self.user_message.id,
                source_quote=self.user_message.content,
                start_at=datetime(2026, 9, 15, 15, tzinfo=timezone.utc),
                end_at=datetime(2026, 9, 15, 16, tzinfo=timezone.utc),
                all_day=False,
                timezone="America/Vancouver",
            )
        )

        saved = self.store.get_calendar_event(event.id)
        self.assertEqual(saved, event)
        self.assertEqual(saved.description, "Compare simulation benchmark results.")

    def test_only_one_pending_clarification_is_allowed_per_conversation(self) -> None:
        assistant = self.store.save_message(
            Message(
                id="message-2",
                conversation_id=self.conversation.id,
                role="assistant",
                content="Which day next week?",
                reply_to_message_id=self.user_message.id,
            )
        )
        clarification = self.store.save_pending_clarification(
            PendingClarification(
                id="clarification-1",
                conversation_id=self.conversation.id,
                source_message_id=self.user_message.id,
                assistant_message_id=assistant.id,
                original_action={"title": "Submit report"},
            )
        )

        self.assertEqual(
            self.store.get_pending_clarification(self.conversation.id), clarification
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save_pending_clarification(
                clarification.model_copy(update={"id": "clarification-2"})
            )

    def test_v2_records_cannot_reference_a_different_conversation(self) -> None:
        other = self.store.save_conversation(
            Conversation(id="conversation-2", title="Other")
        )

        with self.assertRaises(ValueError):
            self.store.save_calendar_event(
                CalendarEvent(
                    id="event-cross-conversation",
                    conversation_id=other.id,
                    title="Submit report",
                    source_message_id=self.user_message.id,
                    source_quote=self.user_message.content,
                    start_at=date(2026, 9, 15),
                    all_day=True,
                    timezone="America/Vancouver",
                )
            )

    def test_event_and_item_management_updates_preserve_original_source(self) -> None:
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="event-managed",
                conversation_id=self.conversation.id,
                title="Submit report",
                source_message_id=self.user_message.id,
                source_quote=self.user_message.content,
                start_at=date(2026, 9, 15),
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        item = self.store.save_saved_item(
            SavedItem(
                id="item-managed",
                conversation_id=self.conversation.id,
                kind="note",
                content="Keep the migration additive.",
                source_message_id=self.user_message.id,
                source_quote=self.user_message.content,
            )
        )

        updated_event = self.store.update_calendar_event(
            event.id, title="Submit final report", start_at=date(2026, 9, 16)
        )
        updated_item = self.store.update_saved_item(item.id, content="Keep SQLite additive.")
        deleted_event = self.store.set_calendar_event_status(event.id, "deleted")
        deleted_item = self.store.set_saved_item_status(item.id, "deleted")

        self.assertEqual(updated_event.source_quote, self.user_message.content)
        self.assertEqual(updated_event.start_at, date(2026, 9, 16))
        self.assertEqual(updated_item.source_message_id, self.user_message.id)
        self.assertEqual(deleted_event.status, "deleted")
        self.assertEqual(deleted_item.status, "deleted")
        self.assertEqual(self.store.set_calendar_event_status(event.id, "deleted"), deleted_event)
        self.assertEqual(self.store.set_saved_item_status(item.id, "deleted"), deleted_item)
        self.assertEqual(self.store.set_calendar_event_status(event.id, "active").status, "active")
        self.assertEqual(self.store.set_saved_item_status(item.id, "active").status, "active")

    def test_conversation_delete_cascades_v2_records(self) -> None:
        assistant = self.store.save_message(
            Message(
                id="message-2",
                conversation_id=self.conversation.id,
                role="assistant",
                content="Which day next week?",
                reply_to_message_id=self.user_message.id,
            )
        )
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="event-1",
                conversation_id=self.conversation.id,
                title="Submit report",
                source_message_id=self.user_message.id,
                source_quote=self.user_message.content,
                start_at=date(2026, 9, 15),
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        item = self.store.save_saved_item(
            SavedItem(
                id="item-1",
                conversation_id=self.conversation.id,
                kind="idea",
                content="Automate report checks.",
                source_message_id=self.user_message.id,
                source_quote=self.user_message.content,
            )
        )
        operation = self.store.save_tool_operation(
            ToolOperation(
                id="operation-1",
                source_message_id=self.user_message.id,
                action_index=0,
                tool_name="save_idea",
                request={"content": item.content},
                status="prepared",
            )
        )
        clarification = self.store.save_pending_clarification(
            PendingClarification(
                id="clarification-1",
                conversation_id=self.conversation.id,
                source_message_id=self.user_message.id,
                assistant_message_id=assistant.id,
                original_action={"title": "Submit report"},
            )
        )

        self.assertTrue(self.store.delete_conversation(self.conversation.id))

        for getter, item_id in (
            (self.store.get_message, self.user_message.id),
            (self.store.get_calendar_event, event.id),
            (self.store.get_saved_item, item.id),
            (self.store.get_tool_operation, operation.id),
            (self.store.get_clarification, clarification.id),
        ):
            with self.assertRaises(KeyError):
                getter(item_id)
