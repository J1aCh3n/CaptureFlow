import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from captureflow.models import (
    CalendarEvent,
    CalendarEventUpdateOperation,
    Conversation,
    EvidenceReference,
    Message,
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
        self.assertEqual(
            self.store.list_calendar_event_sources(event.id),
            [EvidenceReference(message_id=self.user_message.id, source_quote=self.user_message.content)],
        )
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

    def test_calendar_update_is_idempotent_auditable_and_reversible(self) -> None:
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="event-update",
                conversation_id=self.conversation.id,
                title="Algorithm review",
                description="Discuss current bottlenecks.",
                source_message_id=self.user_message.id,
                source_quote=self.user_message.content,
                start_at=datetime(2026, 9, 10, 16, tzinfo=timezone.utc),
                end_at=datetime(2026, 9, 10, 17, tzinfo=timezone.utc),
                all_day=False,
                timezone="America/Vancouver",
            )
        )
        correction = self.store.save_message(
            Message(
                id="correction-message",
                conversation_id=self.conversation.id,
                role="user",
                content="Move this meeting to Thursday from 14:00 to 15:00.",
            )
        )
        updated = event.model_copy(
            update={
                "title": "Algorithm and code review",
                "description": "Review prefix-tree and heap performance.",
                "start_at": datetime(2026, 9, 10, 14, tzinfo=timezone.utc),
                "end_at": datetime(2026, 9, 10, 15, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 9, 9, 12, tzinfo=timezone.utc),
            }
        )
        operation = CalendarEventUpdateOperation(
            id="calendar-update-1",
            source_message_id=correction.id,
            action_index=0,
            event_id=event.id,
            before=event,
            after=updated,
            status="prepared",
            created_at=datetime(2026, 9, 9, 12, tzinfo=timezone.utc),
            updated_at=datetime(2026, 9, 9, 12, tzinfo=timezone.utc),
        )
        evidence = [
            EvidenceReference(message_id=self.user_message.id, source_quote="Submit the report"),
            EvidenceReference(message_id=correction.id, source_quote="Move this meeting"),
        ]

        self.assertEqual(self.store.prepare_calendar_event_update(operation, evidence), operation)
        self.assertEqual(self.store.prepare_calendar_event_update(operation, evidence), operation)
        committed, committed_event = self.store.commit_prepared_calendar_event_update(operation.id)
        repeated, repeated_event = self.store.commit_prepared_calendar_event_update(operation.id)

        self.assertEqual(committed.status, "committed")
        self.assertEqual(committed, repeated)
        self.assertEqual(committed_event, updated)
        self.assertEqual(repeated_event, updated)
        self.assertEqual(self.store.list_calendar_event_update_sources(operation.id), evidence)

        undone = self.store.undo_calendar_event_update(operation.id)
        self.assertEqual(undone.status, "undone")
        self.assertEqual(self.store.undo_calendar_event_update(operation.id), undone)
        self.assertEqual(self.store.get_calendar_event(event.id), event)

    def test_calendar_update_rejects_foreign_workspace_and_stale_target(self) -> None:
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="event-stale",
                conversation_id=self.conversation.id,
                title="Architecture review",
                source_message_id=self.user_message.id,
                source_quote=self.user_message.content,
                start_at=date(2026, 9, 10),
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        operation = CalendarEventUpdateOperation(
            id="calendar-update-stale",
            source_message_id=self.user_message.id,
            action_index=0,
            event_id=event.id,
            before=event,
            after=event.model_copy(
                update={"title": "Architecture and driver review", "updated_at": datetime(2026, 9, 9, tzinfo=timezone.utc)}
            ),
            status="prepared",
        )
        evidence = [EvidenceReference(message_id=self.user_message.id, source_quote="Submit the report")]

        self.store.prepare_calendar_event_update(operation, evidence)
        self.store.update_calendar_event(event.id, title="Manual change", start_at=date(2026, 9, 10))
        with self.assertRaisesRegex(ValueError, "changed before commit"):
            self.store.commit_prepared_calendar_event_update(operation.id)

        foreign = self.store.save_conversation(
            Conversation(id="foreign-conversation", workspace_id="foreign", title="Foreign")
        )
        foreign_message = self.store.save_message(
            Message(
                id="foreign-message",
                conversation_id=foreign.id,
                role="user",
                content="Move the architecture review.",
            )
        )
        current = self.store.get_calendar_event(event.id)
        foreign_operation = operation.model_copy(
            update={
                "id": "calendar-update-foreign",
                "source_message_id": foreign_message.id,
                "before": current,
                "after": current.model_copy(
                    update={
                        "title": "Foreign rewrite",
                        "updated_at": datetime(2026, 9, 9, 1, tzinfo=timezone.utc),
                    }
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "outside the intended workspace"):
            self.store.prepare_calendar_event_update(
                foreign_operation,
                [EvidenceReference(message_id=foreign_message.id, source_quote="Move the architecture review")],
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
        self.assertTrue(self.store.delete_conversation(self.conversation.id))

        for getter, item_id in (
            (self.store.get_message, self.user_message.id),
            (self.store.get_calendar_event, event.id),
            (self.store.get_saved_item, item.id),
            (self.store.get_tool_operation, operation.id),
        ):
            with self.assertRaises(KeyError):
                getter(item_id)
