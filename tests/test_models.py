import unittest
from datetime import date, datetime, timezone

from pydantic import ValidationError

from captureflow.models import (
    CalendarEvent,
    CalendarEventUpdateOperation,
    ContextualAction,
    ConversationDecision,
    Conversation,
    EvidenceReference,
    Message,
    PendingClarification,
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

    def test_resolved_clarification_requires_resolution_time(self) -> None:
        with self.assertRaises(ValidationError):
            PendingClarification(
                conversation_id="conversation-1",
                source_message_id="message-1",
                assistant_message_id="message-2",
                original_action={"title": "Submit report"},
                status="resolved",
            )

    def test_contextual_calendar_update_accepts_two_message_evidence(self) -> None:
        decision = ConversationDecision(
            intent="calendar_update",
            action=ContextualAction(
                action="update_event",
                date="2026-09-10",
                start_time="14:00",
                end_time="15:00",
                target_event_id="algorithm-review-event",
                evidence=[
                    EvidenceReference(
                        message_id="meeting-message",
                        source_quote="我这周四下午四点到五点有算法优化与代码审查会议。",
                    ),
                    EvidenceReference(
                        message_id="correction-message",
                        source_quote="这个会议在周四 14:00 - 15:00",
                    ),
                ],
            ),
        )

        self.assertEqual(decision.action.target_event_id, "algorithm-review-event")
        self.assertEqual(len(decision.action.evidence), 2)

    def test_contextual_calendar_update_requires_candidate_target(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationDecision(
                intent="calendar_update",
                action=ContextualAction(
                    action="update_event",
                    start_time="14:00",
                    evidence=[
                        EvidenceReference(
                            message_id="correction-message",
                            source_quote="这个会议改到下午两点。",
                        )
                    ],
                ),
            )

    def test_contextual_action_rejects_duplicate_message_evidence(self) -> None:
        with self.assertRaises(ValidationError):
            ContextualAction(
                action="create_event",
                content="代码审查",
                date="2026-09-10",
                evidence=[
                    EvidenceReference(message_id="message-1", source_quote="代码审查"),
                    EvidenceReference(message_id="message-1", source_quote="这周四"),
                ],
            )

    def test_normalized_calendar_fields_are_checked_structurally(self) -> None:
        evidence = [EvidenceReference(message_id="message-1", source_quote="评审会")]
        created = ContextualAction(
            action="create_event",
            content="评审会",
            date="2026-09-13",
            start_time="15:00",
            end_time="16:00",
            evidence=evidence,
        )
        undated = ContextualAction(action="create_event", content="评审会", evidence=evidence)

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
                ContextualAction(action="create_event", content="评审会", evidence=evidence, **invalid)
        repeating = ContextualAction(
            action="create_event", content="团队例会", start_time="15:00", recurring=True, evidence=evidence
        )
        self.assertIsNone(repeating.date)
        with self.assertRaises(ValidationError):
            ContextualAction(action="save_note", content="纪要", recurring=True, evidence=evidence)

    def test_contextual_action_rejects_an_end_time_and_duration_together(self) -> None:
        with self.assertRaises(ValidationError):
            ContextualAction(
                action="create_event",
                content="代码审查",
                date="2026-09-10",
                start_time="16:00",
                end_time="17:00",
                duration_minutes=60,
                evidence=[EvidenceReference(message_id="message-1", source_quote="代码审查")],
            )

    def test_a_pending_update_may_lack_the_field_it_is_asking_for(self) -> None:
        evidence = [EvidenceReference(message_id="correction", source_quote="这个会议改到周四。")]
        decision = ConversationDecision(
            intent="clarify",
            clarification_question="你是指 9月10日还是 9月17日的周四？",
            pending_action=ContextualAction(
                action="update_event", target_event_id="event-1", evidence=evidence
            ),
        )

        self.assertIsNone(decision.pending_action.date)
        with self.assertRaises(ValidationError):
            ConversationDecision(
                intent="calendar_update",
                action=ContextualAction(
                    action="update_event", target_event_id="event-1", evidence=evidence
                ),
            )

    def test_contextual_clarification_retains_an_unresolved_calendar_update(self) -> None:
        decision = ConversationDecision(
            intent="clarify",
            clarification_question="Which existing meeting should I update?",
            pending_action=ContextualAction(
                action="update_event",
                start_time="14:00",
                evidence=[
                    EvidenceReference(
                        message_id="correction-message",
                        source_quote="这个会议改到下午两点。",
                    )
                ],
            ),
        )

        self.assertEqual(decision.pending_action.action, "update_event")

    def test_contextual_calendar_delete_requires_a_bare_candidate_target(self) -> None:
        decision = ConversationDecision(
            intent="calendar_delete",
            action=ContextualAction(
                action="delete_event",
                target_event_id="expired-meeting-event",
                evidence=[
                    EvidenceReference(
                        message_id="delete-message",
                        source_quote="昨天的会议已结束，删除过期会议",
                    )
                ],
            ),
        )

        self.assertEqual(decision.action.target_event_id, "expired-meeting-event")
        with self.assertRaises(ValidationError):
            ConversationDecision(
                intent="calendar_delete",
                action=ContextualAction(
                    action="delete_event",
                    evidence=[
                        EvidenceReference(message_id="delete-message", source_quote="删除过期会议")
                    ],
                ),
            )
        with self.assertRaises(ValidationError):
            ContextualAction(
                action="delete_event",
                target_event_id="expired-meeting-event",
                start_time="16:00",
                evidence=[
                    EvidenceReference(message_id="delete-message", source_quote="删除过期会议")
                ],
            )

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

    def test_contextual_decision_bounds_a_deferred_second_item(self) -> None:
        note_evidence = [
            EvidenceReference(
                message_id="compound-message",
                source_quote="今天中午十二点的会议是关于系统架构与驱动集成，修复需要下周三晚上八点之前提交",
            )
        ]
        decision = ConversationDecision(
            intent="save_note",
            action=ContextualAction(
                action="save_note", content="系统架构与驱动集成会议纪要。", evidence=note_evidence
            ),
            deferred_action=ContextualAction(
                action="create_event",
                content="提交修复",
                date="2026-09-16",
                start_time="20:00",
                evidence=note_evidence,
            ),
            follow_up_question="Should I also add that deadline to the local calendar?",
        )

        self.assertEqual(decision.deferred_action.action, "create_event")
        with self.assertRaises(ValidationError):
            ConversationDecision(
                intent="save_note",
                action=ContextualAction(
                    action="save_note", content="会议纪要。", evidence=note_evidence
                ),
                deferred_action=ContextualAction(
                    action="create_event",
                    content="提交修复",
                    date="2026-09-16",
                    evidence=note_evidence,
                ),
            )
        with self.assertRaises(ValidationError):
            ConversationDecision(
                intent="save_note",
                action=ContextualAction(
                    action="save_note", content="会议纪要。", evidence=note_evidence
                ),
                deferred_action=ContextualAction(
                    action="delete_event", target_event_id="event-1", evidence=note_evidence
                ),
                follow_up_question="Should I also delete that meeting?",
            )

    def test_contextual_clarification_retains_an_unresolved_calendar_delete(self) -> None:
        decision = ConversationDecision(
            intent="clarify",
            clarification_question="Which local calendar event should I delete?",
            pending_action=ContextualAction(
                action="delete_event",
                evidence=[
                    EvidenceReference(message_id="delete-message", source_quote="删除过期会议")
                ],
            ),
        )

        self.assertIsNone(decision.pending_action.target_event_id)

    def test_contextual_answer_sources_are_unique_and_reply_only_has_none(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationDecision(
                intent="project_answer",
                reply="The current task is stored locally.",
                source_ids=["record-1", "record-1"],
            )
        with self.assertRaises(ValidationError):
            ConversationDecision(
                intent="reply_only",
                reply="I cannot help with that local action.",
                source_ids=["record-1"],
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
            status="prepared",
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
