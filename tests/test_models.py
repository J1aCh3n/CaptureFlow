import unittest
from datetime import date, datetime, timezone

from pydantic import ValidationError

from captureflow.models import (
    CalendarEvent,
    CalendarEventUpdateOperation,
    ContextualAction,
    Conversation,
    Message,
    ProjectChange,
    SavedItem,
    ToolOperation,
)


class ModelTests(unittest.TestCase):
    def test_conversation_title_is_trimmed(self) -> None:
        conversation = Conversation(title="  Reading list  ")

        self.assertEqual(conversation.title, "Reading list")

    def test_blank_message_is_invalid(self) -> None:
        with self.assertRaises(ValidationError):
            Message(conversation_id="conversation-1", role="user", content="   ")

    def test_calendar_event_shape_matches_all_day_flag(self) -> None:
        all_day = CalendarEvent(
            conversation_id="conversation-1",
            title="Submit report",
            source_message_id="message-1",
            source_quote="Submit the report on September 15.",
            start_at=date(2026, 9, 15),
            all_day=True,
            timezone="America/Vancouver",
        )
        timed = CalendarEvent(
            conversation_id="conversation-1",
            title="Review results",
            source_message_id="message-1",
            source_quote="Review the results at 3 PM.",
            start_at=datetime(2026, 9, 15, 15, tzinfo=timezone.utc),
            all_day=False,
            timezone="America/Vancouver",
        )

        self.assertIs(type(all_day.start_at), date)
        self.assertIsInstance(timed.start_at, datetime)
        with self.assertRaises(ValidationError):
            CalendarEvent(
                conversation_id="conversation-1",
                title="Invalid",
                source_message_id="message-1",
                source_quote="Invalid",
                start_at=date(2026, 9, 15),
                all_day=False,
                timezone="America/Vancouver",
            )

    def test_saved_item_rejects_blank_content(self) -> None:
        with self.assertRaises(ValidationError):
            SavedItem(
                conversation_id="conversation-1",
                kind="note",
                content=" ",
                source_message_id="message-1",
                source_quote="Remember this.",
            )

    def test_failed_tool_operation_requires_error(self) -> None:
        with self.assertRaises(ValidationError):
            ToolOperation(
                id="operation-1",
                source_message_id="message-1",
                action_index=0,
                tool_name="save_note",
                request={"content": "Remember this."},
                status="failed",
            )

    def test_contextual_calendar_update_names_its_source_messages(self) -> None:
        action = ContextualAction.model_validate(
            {
                "action": "update_event",
                "date": "2026-09-10",
                "start_time": "14:00",
                "end_time": "15:00",
                "target_event_id": "algorithm-review-event",
                "source_message_ids": ["meeting-message", "correction-message", "meeting-message"],
            }
        )

        self.assertEqual(action.target_event_id, "algorithm-review-event")
        self.assertEqual(action.source_message_ids, ["meeting-message", "correction-message"])
        self.assertEqual(
            ContextualAction(action="save_note", content="纪要", source_message_ids=None).source_message_ids,
            [],
        )

    def test_a_retired_evidence_field_on_a_stored_action_is_ignored(self) -> None:
        action = ContextualAction.model_validate(
            {
                "action": "save_note",
                "content": "纪要",
                "evidence": [{"message_id": "message-1", "source_quote": "纪要"}],
            }
        )

        self.assertEqual(action.content, "纪要")
        self.assertEqual(action.source_message_ids, [])

    def test_normalized_calendar_fields_are_checked_structurally(self) -> None:
        created = ContextualAction(
            action="create_event", content="评审会", date="2026-09-13", start_time="15:00", end_time="16:00"
        )
        undated = ContextualAction(action="create_event", content="评审会")

        self.assertEqual((created.date, created.start_time, created.end_time), ("2026-09-13", "15:00", "16:00"))
        self.assertIsNone(undated.date)
        for invalid in (
            {"date": "2026-02-30"},
            {"date": "13 September"},
            {"date": "2026-09-13", "start_time": "3 PM"},
            {"date": "2026-09-13", "start_time": "24:00"},
            {"start_time": "15:00"},
            {"date": "2026-09-13", "end_time": "16:00"},
            {"date": "2026-09-13", "start_time": "15:00", "duration_minutes": 0},
            {"date": "2026-09-13", "start_time": "15:00", "duration_minutes": 721},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                ContextualAction(action="create_event", content="评审会", **invalid)
        repeating = ContextualAction(action="create_event", content="团队例会", start_time="15:00", recurring=True)
        self.assertIsNone(repeating.date)
        with self.assertRaises(ValidationError):
            ContextualAction(action="save_note", content="纪要", recurring=True)

    def test_contextual_action_may_carry_an_end_time_and_a_duration_together(self) -> None:
        # Whether they agree is the router's question, not a schema error.
        action = ContextualAction(
            action="create_event",
            content="代码审查",
            date="2026-09-10",
            start_time="16:00",
            end_time="17:00",
            duration_minutes=90,
        )

        self.assertEqual((action.end_time, action.duration_minutes), ("17:00", 90))

    def test_a_proposed_update_may_lack_its_new_value(self) -> None:
        # The write tool asks the user for it instead of refusing the change.
        undecided = ContextualAction(action="update_event", target_event_id="event-1")

        self.assertFalse(undecided.has_replacement())
        self.assertTrue(
            ContextualAction(action="update_event", target_event_id="event-1", start_time="14:00").has_replacement()
        )

    def test_a_contextual_deletion_carries_nothing_but_its_target(self) -> None:
        deletion = ContextualAction(action="delete_event", target_event_id="expired-meeting-event")

        self.assertEqual(deletion.target_event_id, "expired-meeting-event")
        with self.assertRaises(ValidationError):
            ContextualAction(action="delete_event", target_event_id="expired-meeting-event", start_time="16:00")

    def test_a_stored_pre_phase_11_deadline_change_still_validates(self) -> None:
        stored = {
            "action": "create_record",
            "kind": "task",
            "content": "提交修复",
            "due_date_expression": "下周三",
            "due_time_expression": "晚上八点",
            "due_at": "2026-09-16T20:00:00-07:00",
            "source_quote": "修复需要下周三晚上八点之前提交",
        }

        change = ProjectChange.model_validate(stored)

        self.assertEqual((change.due_date, change.due_time), ("2026-09-16", "20:00"))
        self.assertEqual(change.due_at, datetime.fromisoformat("2026-09-16T20:00:00-07:00"))
        self.assertIsNone(change.source_message_id)

    def test_a_tracker_change_may_omit_its_quote_and_name_its_source_message(self) -> None:
        change = ProjectChange.model_validate(
            {
                "action": "create_record",
                "kind": "task",
                "content": "整理部署文档",
                "source_quote": " ",
                "source_message_id": "pasted-minutes",
            }
        )

        self.assertIsNone(change.source_quote)
        self.assertEqual(change.source_message_id, "pasted-minutes")

    def test_an_unstated_recurrence_may_arrive_as_null(self) -> None:
        change = ProjectChange.model_validate(
            {
                "action": "create_record",
                "kind": "task",
                "content": "排查延迟问题",
                "source_quote": "我需要排查延迟问题。",
                "due_date": None,
                "recurring": None,
            }
        )
        action = ContextualAction.model_validate(
            {"action": "update_event", "target_event_id": "event-1", "recurring": None}
        )

        self.assertFalse(change.recurring)
        self.assertFalse(action.recurring)
        schema = ContextualAction.model_json_schema()["properties"]["recurring"]
        self.assertEqual(schema["type"], "boolean")

    def test_only_a_new_task_may_carry_a_deadline(self) -> None:
        change = ProjectChange(
            action="create_record",
            kind="task",
            content="提交修复",
            due_date="2026-09-16",
            due_time="20:00",
            source_quote="修复需要下周三晚上八点之前提交",
        )

        self.assertIsNone(change.due_at)
        with self.assertRaises(ValidationError):
            ProjectChange(
                action="create_record",
                kind="blocker",
                content="驱动没有到货",
                due_date="2026-09-16",
                source_quote="驱动没有到货",
            )
        with self.assertRaises(ValidationError):
            ProjectChange(
                action="create_record",
                kind="task",
                content="提交修复",
                due_time="20:00",
                source_quote="修复需要晚上八点之前提交",
            )
        with self.assertRaises(ValidationError):
            ProjectChange(
                action="transition_record",
                target_record_id="record-1",
                target_state="completed",
                due_date="2026-09-16",
                source_quote="这个任务做完了",
            )

    def test_calendar_delete_operation_only_changes_event_status(self) -> None:
        before = CalendarEvent(
            id="event-2",
            conversation_id="conversation-1",
            title="Review",
            source_message_id="message-1",
            source_quote="Review the current plan.",
            start_at=date(2026, 9, 15),
            all_day=True,
            timezone="America/Vancouver",
        )
        operation = CalendarEventUpdateOperation(
            id="delete-1",
            source_message_id="message-2",
            action_index=0,
            event_id=before.id,
            before=before,
            after=before.model_copy(update={"status": "deleted"}),
            status="review",
        )

        self.assertEqual(operation.after.status, "deleted")
        with self.assertRaises(ValidationError):
            CalendarEventUpdateOperation(
                id="delete-2",
                source_message_id="message-2",
                action_index=0,
                event_id=before.id,
                before=before,
                after=before.model_copy(update={"status": "deleted", "title": "Renamed"}),
                status="prepared",
            )

    def test_calendar_update_operation_cannot_rewrite_original_event_provenance(self) -> None:
        before = CalendarEvent(
            id="event-1",
            conversation_id="conversation-1",
            title="Review",
            source_message_id="message-1",
            source_quote="Review the current plan.",
            start_at=date(2026, 9, 15),
            all_day=True,
            timezone="America/Vancouver",
        )
        with self.assertRaises(ValidationError):
            CalendarEventUpdateOperation(
                id="update-1",
                source_message_id="message-2",
                action_index=0,
                event_id=before.id,
                before=before,
                after=before.model_copy(update={"source_quote": "A replacement quote."}),
                status="prepared",
            )
