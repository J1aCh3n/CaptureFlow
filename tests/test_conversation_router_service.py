import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from captureflow.conversation_router_service import ConversationRouterService
from captureflow.models import (
    CalendarEvent,
    CalendarEventUpdateOperation,
    Conversation,
    EvidenceReference,
    Message,
    Project,
    ProjectRecord,
    ToolOperation,
)
from captureflow.storage import CaptureStore


class ScriptedContextualModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def propose_conversation_decision(self, **kwargs):
        self.calls.append(kwargs)
        return self.decisions.pop(0)


class ConversationRouterServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.conversation = self.store.save_conversation(
            Conversation(id="contextual-router-chat", title="Contextual router")
        )
        self.project = self.store.save_project(Project(id="contextual-router-project", name="Workflow"))
        self.store.assign_conversation_project(self.conversation.id, self.project.id)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_event_statement_creates_calendar_event_not_project_task(self) -> None:
        content = "我这周四下午四点到五点有算法优化与代码审查会议。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "算法优化与代码审查",
                        "description": "讨论前缀树与最小堆性能瓶颈。",
                        "date_expression": "这周四",
                        "time_expression": "下午四点",
                        "end_time_expression": "下午五点",
                        "target_event_id": None,
                        "evidence": [{"message_id": "event-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(
            self.store, model, date(2026, 9, 9)
        ).submit(self.conversation.id, content, "event-turn")
        replay = ConversationRouterService(
            self.store, model, date(2026, 9, 9)
        ).submit(self.conversation.id, content, "event-turn")

        event = self.store.list_calendar_events()[0]
        self.assertEqual(turn.outcome, "calendar_create")
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(event.title, "算法优化与代码审查")
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        self.assertEqual(event.end_at, datetime.fromisoformat("2026-09-10T17:00:00-07:00"))
        self.assertEqual(
            self.store.list_calendar_event_sources(event.id),
            [EvidenceReference(message_id="event-turn", source_quote=content)],
        )
        self.assertEqual(self.store.list_project_records(self.project.id), [])
        self.assertEqual(model.calls[0]["candidate_events"], [])
        self.assertEqual(replay.model_calls, 0)
        self.assertEqual(len(model.calls), 1)

    def test_bare_future_weekday_creates_an_event_with_its_resolved_date(self) -> None:
        content = "周五下午两点有架构评审。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "架构评审",
                        "description": None,
                        "date_expression": "周五",
                        "time_expression": "下午两点",
                        "end_time_expression": None,
                        "target_event_id": None,
                        "evidence": [{"message_id": "bare-weekday-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 10)).submit(
            self.conversation.id, content, "bare-weekday-turn"
        )

        event = self.store.list_calendar_events()[0]
        self.assertEqual(turn.outcome, "calendar_create")
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-11T14:00:00-07:00"))
        self.assertIn("2026-09-11T14:00:00-07:00 (America/Vancouver)", turn.assistant_message.content)

    def test_event_correction_updates_supplied_event_without_second_model_call(self) -> None:
        source = self.store.save_message(
            Message(
                id="existing-event-source",
                conversation_id=self.conversation.id,
                role="user",
                content="本周四下午四点有工作流会议。",
            )
        )
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="workflow-event",
                conversation_id=self.conversation.id,
                title="工作流会议",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=datetime.fromisoformat("2026-09-10T16:00:00-07:00"),
                end_at=datetime.fromisoformat("2026-09-10T17:00:00-07:00"),
                all_day=False,
                timezone="America/Vancouver",
            )
        )
        correction = "帮我记录一下，这个会议在周四 14:00 - 15:00"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": {
                        "action": "update_event",
                        "content": None,
                        "description": None,
                        "date_expression": "这周四",
                        "time_expression": "14:00",
                        "end_time_expression": "15:00",
                        "target_event_id": event.id,
                        "evidence": [
                            {"message_id": "existing-event-source", "source_quote": source.content},
                            {"message_id": "correction-turn", "source_quote": correction},
                        ],
                    },
                }
            ]
        )

        turn = ConversationRouterService(
            self.store, model, date(2026, 9, 9)
        ).submit(self.conversation.id, correction, "correction-turn")

        updated = self.store.get_calendar_event(event.id)
        self.assertEqual(turn.outcome, "calendar_update")
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(updated.start_at, datetime.fromisoformat("2026-09-10T14:00:00-07:00"))
        self.assertEqual(updated.end_at, datetime.fromisoformat("2026-09-10T15:00:00-07:00"))
        self.assertEqual(len(self.store.list_calendar_event_update_operations("correction-turn")), 1)
        self.assertEqual(len(model.calls), 1)

    def test_calendar_update_resolves_chinese_time_and_explicit_duration(self) -> None:
        original = "今天开始进行A测，下午三点钟要准备好，四点有个相关的说明会"
        correction = "会议延期了，时间也被缩短到半个小时，现在晚上七点才开始"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "相关的说明会",
                        "description": "今天开始进行A测，下午三点钟要准备好",
                        "date_expression": "今天",
                        "time_expression": "下午四点",
                        "end_time_expression": None,
                        "duration_expression": None,
                        "target_event_id": None,
                        "evidence": [{"message_id": "phase-10-3-create", "source_quote": original}],
                    },
                },
                {
                    "intent": "calendar_update",
                    "action": {
                        "action": "update_event",
                        "content": None,
                        "description": None,
                        "date_expression": None,
                        "time_expression": "晚上七点",
                        "end_time_expression": None,
                        "duration_expression": "半个小时",
                        "target_event_id": "phase-10-3-event",
                        "evidence": [
                            {"message_id": "phase-10-3-create", "source_quote": original},
                            {"message_id": "phase-10-3-update", "source_quote": correction},
                        ],
                    },
                },
            ]
        )
        service = ConversationRouterService(self.store, model, date(2026, 9, 10))

        created = service.submit(self.conversation.id, original, "phase-10-3-create")
        event = self.store.get_calendar_event(created.operations[0].result["event_id"])
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        self.assertIsNone(event.end_at)
        model.decisions[0]["action"]["target_event_id"] = event.id

        updated = service.submit(self.conversation.id, correction, "phase-10-3-update")
        replay = service.submit(self.conversation.id, correction, "phase-10-3-update")
        operation = self.store.list_calendar_event_update_operations("phase-10-3-update")[0]
        event = self.store.get_calendar_event(event.id)

        self.assertEqual(updated.outcome, "calendar_update")
        self.assertEqual(updated.model_calls, 1)
        self.assertEqual(replay.model_calls, 0)
        self.assertEqual(len(self.store.list_calendar_events()), 1)
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-10T19:00:00-07:00"))
        self.assertEqual(event.end_at, datetime.fromisoformat("2026-09-10T19:30:00-07:00"))
        self.assertEqual(len(self.store.list_calendar_event_update_operations("phase-10-3-update")), 1)
        self.assertIsNone(self.store.get_pending_clarification(self.conversation.id))
        self.assertEqual(self.store.undo_calendar_event_update(operation.id).status, "undone")
        restored = self.store.get_calendar_event(event.id)
        self.assertEqual(restored.start_at, datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        self.assertIsNone(restored.end_at)

    def test_calendar_end_time_can_cross_midnight(self) -> None:
        content = "明天 23:30 到 00:30 有发布检查。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "发布检查",
                        "description": None,
                        "date_expression": "明天",
                        "time_expression": "23:30",
                        "end_time_expression": "00:30",
                        "duration_expression": None,
                        "target_event_id": None,
                        "evidence": [{"message_id": "cross-midnight", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 10)).submit(
            self.conversation.id, content, "cross-midnight"
        )

        event = self.store.get_calendar_event(turn.operations[0].result["event_id"])
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-11T23:30:00-07:00"))
        self.assertEqual(event.end_at, datetime.fromisoformat("2026-09-12T00:30:00-07:00"))

    def test_unsupported_calendar_time_or_duration_creates_a_clarification_without_a_write(self) -> None:
        source = self.store.save_message(
            Message(
                id="unsupported-event-source",
                conversation_id=self.conversation.id,
                role="user",
                content="今天下午四点有工作流会议。",
            )
        )
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="unsupported-event",
                conversation_id=self.conversation.id,
                title="工作流会议",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=datetime.fromisoformat("2026-09-10T16:00:00-07:00"),
                all_day=False,
                timezone="America/Vancouver",
            )
        )
        correction = "这个会议改到晚上七点才开始，持续一会儿。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": {
                        "action": "update_event",
                        "content": None,
                        "description": None,
                        "date_expression": None,
                        "time_expression": "晚上七点才开始",
                        "end_time_expression": None,
                        "duration_expression": "一会儿",
                        "target_event_id": event.id,
                        "evidence": [{"message_id": "unsupported-correction", "source_quote": correction}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 10)).submit(
            self.conversation.id, correction, "unsupported-correction"
        )

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(
            turn.assistant_message.content,
            "What exact start time should I use for this local event?",
        )
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_event_update_operations("unsupported-correction"), [])
        pending = self.store.get_pending_clarification(self.conversation.id)
        self.assertIsNotNone(pending)
        self.assertEqual(pending.missing_field, "time")

    def test_unsupported_calendar_duration_creates_a_duration_clarification_without_a_write(self) -> None:
        source = self.store.save_message(
            Message(
                id="duration-event-source",
                conversation_id=self.conversation.id,
                role="user",
                content="今天下午四点有工作流会议。",
            )
        )
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="duration-event",
                conversation_id=self.conversation.id,
                title="工作流会议",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=datetime.fromisoformat("2026-09-10T16:00:00-07:00"),
                all_day=False,
                timezone="America/Vancouver",
            )
        )
        correction = "这个会议改到晚上七点，持续一会儿。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": {
                        "action": "update_event",
                        "content": None,
                        "description": None,
                        "date_expression": None,
                        "time_expression": "晚上七点",
                        "end_time_expression": None,
                        "duration_expression": "一会儿",
                        "target_event_id": event.id,
                        "evidence": [{"message_id": "duration-correction", "source_quote": correction}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 10)).submit(
            self.conversation.id, correction, "duration-correction"
        )

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(
            turn.assistant_message.content,
            "How long should this local event last (for example, 30 minutes)?",
        )
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_event_update_operations("duration-correction"), [])
        self.assertEqual(
            self.store.get_pending_clarification(self.conversation.id).missing_field, "duration"
        )

    def test_prepared_duration_update_resumes_after_reopening_the_database(self) -> None:
        source = self.store.save_message(
            Message(
                id="prepared-update-source",
                conversation_id=self.conversation.id,
                role="user",
                content="今天下午四点有工作流会议。",
            )
        )
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="prepared-update-event",
                conversation_id=self.conversation.id,
                title="工作流会议",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=datetime.fromisoformat("2026-09-10T16:00:00-07:00"),
                all_day=False,
                timezone="America/Vancouver",
            )
        )
        correction = self.store.save_message(
            Message(
                id="prepared-update-correction",
                conversation_id=self.conversation.id,
                role="user",
                content="会议延期到晚上七点，缩短到半个小时。",
            )
        )
        updated = event.model_copy(
            update={
                "start_at": datetime.fromisoformat("2026-09-10T19:00:00-07:00"),
                "end_at": datetime.fromisoformat("2026-09-10T19:30:00-07:00"),
            }
        )
        operation = CalendarEventUpdateOperation(
            id="prepared-duration-update",
            source_message_id=correction.id,
            action_index=0,
            event_id=event.id,
            before=event,
            after=updated,
            status="prepared",
        )
        self.store.prepare_calendar_event_update(
            operation,
            [
                EvidenceReference(message_id=source.id, source_quote=source.content),
                EvidenceReference(message_id=correction.id, source_quote=correction.content),
            ],
        )
        restarted = CaptureStore(self.store.database_path)
        restarted.initialize()
        model = ScriptedContextualModel([])

        resumed = ConversationRouterService(restarted, model, date(2026, 9, 10)).submit(
            self.conversation.id, correction.content, correction.id
        )

        self.assertEqual(resumed.outcome, "resumed")
        self.assertEqual(resumed.model_calls, 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(restarted.get_calendar_event(event.id), updated)
        self.assertEqual(
            restarted.get_calendar_event_update_operation(operation.id).status, "committed"
        )

    def test_finished_meeting_is_kept_as_a_note_without_a_calendar_event(self) -> None:
        content = (
            "今天中午十二点的会议是关于系统架构与驱动集成，主要分析了ALOHA 遥操作系统的驱动调试。"
        )
        model = ScriptedContextualModel(
            [
                {
                    "intent": "save_note",
                    "action": {
                        "action": "save_note",
                        "content": "系统架构与驱动集成会议：主要分析了ALOHA 遥操作系统的驱动调试。",
                        "evidence": [{"message_id": "minutes-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "minutes-turn"
        )
        items = self.store.list_saved_items(kinds=("note",))

        self.assertEqual(turn.outcome, "save_note")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_message_id, "minutes-turn")
        self.assertEqual(items[0].source_quote, content)

    def test_compound_turn_saves_minutes_then_asks_about_the_stated_deadline(self) -> None:
        content = (
            "今天中午十二点的会议是关于系统架构与驱动集成，主要分析了ALOHA 遥操作系统的驱动调试。"
            "修复需要下周三晚上八点之前提交"
        )
        confirmation = "是的，请建下周三的日历。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "save_note",
                    "action": {
                        "action": "save_note",
                        "content": "系统架构与驱动集成会议：主要分析了ALOHA 遥操作系统的驱动调试。",
                        "evidence": [{"message_id": "compound-turn", "source_quote": content}],
                    },
                    "deferred_action": {
                        "action": "create_event",
                        "content": "提交修复",
                        "date_expression": "下周三",
                        "time_expression": "晚上八点",
                        "evidence": [{"message_id": "compound-turn", "source_quote": content}],
                    },
                    "follow_up_question": "Should I also add a local calendar event for 下周三晚上八点?",
                },
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "提交修复",
                        "date_expression": "下周三",
                        "time_expression": "晚上八点",
                        "evidence": [
                            {"message_id": "compound-turn", "source_quote": content},
                            {
                                "message_id": "compound-confirmation",
                                "source_quote": "请建下周三的日历",
                            },
                        ],
                    },
                },
            ]
        )
        service = ConversationRouterService(self.store, model, date(2026, 9, 11))

        first = service.submit(self.conversation.id, content, "compound-turn")
        pending = self.store.get_pending_clarification(self.conversation.id)

        self.assertEqual(first.outcome, "save_note")
        self.assertEqual(first.model_calls, 1)
        self.assertEqual(len(self.store.list_saved_items(kinds=("note",))), 1)
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertIsNotNone(pending)
        self.assertEqual(pending.missing_field, "follow_up")
        self.assertEqual(pending.original_action["date_expression"], "下周三")
        self.assertIn(
            "Should I also add a local calendar event for 下周三晚上八点?",
            first.assistant_message.content,
        )

        second = service.submit(self.conversation.id, confirmation, "compound-confirmation")
        event = self.store.list_calendar_events()[0]

        self.assertEqual(second.outcome, "calendar_create")
        self.assertEqual(model.calls[1]["pending_clarification"], pending.original_action)
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-16T20:00:00-07:00"))
        self.assertEqual(
            self.store.get_pending_clarification(self.conversation.id), None
        )

    def test_task_deadline_is_resolved_locally_and_listed_for_the_calendar(self) -> None:
        content = "修复需要下周三晚上八点之前提交"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "提交修复",
                            "due_date_expression": "下周三",
                            "due_time_expression": "晚上八点",
                            "source_quote": content,
                        }
                    ],
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "deadline-record-turn"
        )
        record = self.store.list_project_records(self.project.id)[0]

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(record.due_at, datetime.fromisoformat("2026-09-16T20:00:00-07:00"))
        self.assertEqual(self.store.list_task_deadlines(), [record])
        self.assertIn("2026-09-16T20:00:00-07:00", turn.assistant_message.content)

        self.store.transition_project_record(record.id, "completed")

        self.assertEqual(self.store.list_task_deadlines(), [])

    def test_a_task_deadline_without_a_time_stays_a_calendar_date(self) -> None:
        content = "文档下周三之前交。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "交文档",
                            "due_date_expression": "下周三",
                            "source_quote": content,
                        }
                    ],
                }
            ]
        )

        ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "date-only-deadline-turn"
        )
        record = self.store.list_project_records(self.project.id)[0]

        self.assertEqual(record.due_at, date(2026, 9, 16))
        self.assertEqual(self.store.list_task_deadlines(), [record])

    def test_an_unresolvable_deadline_keeps_the_task_without_one(self) -> None:
        content = "修复要尽快提交。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "提交修复",
                            "due_date_expression": "尽快",
                            "source_quote": content,
                        }
                    ],
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "vague-deadline-turn"
        )
        record = self.store.list_project_records(self.project.id)[0]

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertIsNone(record.due_at)
        self.assertEqual(self.store.list_task_deadlines(), [])
        self.assertIn("could not resolve the stated deadline", turn.assistant_message.content)

    def test_a_model_supplied_deadline_timestamp_is_refused(self) -> None:
        content = "修复需要下周三晚上八点之前提交"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "提交修复",
                            "due_date_expression": "下周三",
                            "due_at": "2026-09-16T20:00:00-07:00",
                            "source_quote": content,
                        }
                    ],
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "model-deadline-turn"
        )

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertEqual(self.store.list_project_records(self.project.id), [])

    def test_committed_tracker_deadline_can_defer_the_meeting_minutes(self) -> None:
        content = "今天中午十二点的会议分析了驱动调试。修复需要下周三晚上八点之前提交"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "修复需要下周三晚上八点之前提交",
                            "source_quote": "修复需要下周三晚上八点之前提交",
                        }
                    ],
                    "deferred_action": {
                        "action": "save_note",
                        "content": "今天中午十二点的会议分析了驱动调试。",
                        "evidence": [{"message_id": "deadline-turn", "source_quote": content}],
                    },
                    "follow_up_question": "要我把会议内容保存为会议纪要吗？",
                },
                {
                    "intent": "save_note",
                    "action": {
                        "action": "save_note",
                        "content": "今天中午十二点的会议分析了驱动调试。",
                        "evidence": [
                            {"message_id": "deadline-turn", "source_quote": content},
                            {"message_id": "deadline-confirmation", "source_quote": "是的，请保存。"},
                        ],
                    },
                },
            ]
        )
        service = ConversationRouterService(self.store, model, date(2026, 9, 11))

        turn = service.submit(self.conversation.id, content, "deadline-turn")
        pending = self.store.get_pending_clarification(self.conversation.id)

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(len(self.store.list_project_records(self.project.id)), 1)
        self.assertEqual(self.store.list_saved_items(kinds=("note",)), [])
        self.assertIsNotNone(pending)
        self.assertEqual(pending.missing_field, "follow_up")
        self.assertEqual(pending.original_action["action"], "save_note")
        self.assertIn("要我把会议内容保存为会议纪要吗？", turn.assistant_message.content)

        confirmed = service.submit(self.conversation.id, "是的，请保存。", "deadline-confirmation")

        self.assertEqual(confirmed.outcome, "save_note")
        self.assertEqual(len(self.store.list_saved_items(kinds=("note",))), 1)
        self.assertIsNone(self.store.get_pending_clarification(self.conversation.id))

    def test_a_deferred_item_without_a_committed_action_is_refused(self) -> None:
        content = "这条消息只是闲聊，顺便下周三晚上八点提交修复。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "reply_only",
                    "reply": "I cannot take a local action for that.",
                    "deferred_action": {
                        "action": "create_event",
                        "content": "提交修复",
                        "date_expression": "下周三",
                        "time_expression": "晚上八点",
                        "evidence": [{"message_id": "reply-only-turn", "source_quote": content}],
                    },
                    "follow_up_question": "Should I also add that deadline?",
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "reply-only-turn"
        )

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertIsNone(self.store.get_pending_clarification(self.conversation.id))
        self.assertEqual(self.store.list_calendar_events(), [])

    def _saved_event(self, suffix: str, title: str, start_at: datetime) -> CalendarEvent:
        source = self.store.save_message(
            Message(
                id=f"{suffix}-source",
                conversation_id=self.conversation.id,
                role="user",
                content=f"{title} {start_at.isoformat()}",
            )
        )
        return self.store.save_calendar_event(
            CalendarEvent(
                id=f"{suffix}-event",
                conversation_id=self.conversation.id,
                title=title,
                source_message_id=source.id,
                source_quote=source.content,
                start_at=start_at,
                all_day=False,
                timezone="America/Vancouver",
            )
        )

    def test_expired_meeting_is_deleted_from_chat_and_can_be_undone(self) -> None:
        event = self._saved_event(
            "expired-meeting", "相关的说明会", datetime.fromisoformat("2026-09-10T16:00:00-07:00")
        )
        content = "昨天的会议已结束，删除过期会议"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_delete",
                    "action": {
                        "action": "delete_event",
                        "target_event_id": event.id,
                        "evidence": [{"message_id": "delete-turn", "source_quote": content}],
                    },
                }
            ]
        )
        service = ConversationRouterService(self.store, model, date(2026, 9, 11))

        turn = service.submit(self.conversation.id, content, "delete-turn")
        replay = service.submit(self.conversation.id, content, "delete-turn")
        operation = self.store.list_calendar_event_update_operations("delete-turn")[0]

        self.assertEqual(turn.outcome, "calendar_delete")
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(replay.model_calls, 0)
        self.assertEqual(self.store.get_calendar_event(event.id).status, "deleted")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(len(self.store.list_calendar_event_update_operations("delete-turn")), 1)
        self.assertEqual(
            self.store.list_calendar_event_update_sources(operation.id),
            [EvidenceReference(message_id="delete-turn", source_quote=content)],
        )
        self.assertEqual(self.store.undo_calendar_event_update(operation.id).status, "undone")
        self.assertEqual(self.store.get_calendar_event(event.id), event)

    def test_a_chinese_credential_disclosure_never_reaches_the_model(self) -> None:
        model = ScriptedContextualModel([])

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, "我的密钥是 sk-test-1234567890abcdef", "credential-turn"
        )

        self.assertEqual(turn.outcome, "blocked_credential")
        self.assertEqual(turn.model_calls, 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(self.store.list_saved_items(kinds=("note", "idea")), [])

    def test_a_stated_foreign_timezone_clarifies_instead_of_storing_local_time(self) -> None:
        content = "纽约时间明天上午九点有电话会"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "电话会",
                        "date_expression": "明天",
                        "time_expression": "上午九点",
                        "evidence": [{"message_id": "foreign-timezone-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "foreign-timezone-turn"
        )
        pending = self.store.get_pending_clarification(self.conversation.id)

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertIsNotNone(pending)
        self.assertEqual(pending.missing_field, "timezone")
        self.assertIn("America/Vancouver", turn.assistant_message.content)

    def test_the_local_timezone_by_name_still_creates_the_event(self) -> None:
        content = "温哥华时间明天上午九点有电话会"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "电话会",
                        "date_expression": "明天",
                        "time_expression": "上午九点",
                        "evidence": [{"message_id": "local-timezone-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "local-timezone-turn"
        )

        self.assertEqual(turn.outcome, "calendar_create")
        self.assertEqual(
            self.store.list_calendar_events()[0].start_at,
            datetime.fromisoformat("2026-09-12T09:00:00-07:00"),
        )
        self.assertIsNone(self.store.get_pending_clarification(self.conversation.id))

    def test_stale_events_do_not_hide_the_event_a_turn_is_about(self) -> None:
        for suffix, title, start_at in (
            ("stale-1", "上月周会", "2026-08-03T10:00:00-07:00"),
            ("stale-2", "上月评审", "2026-08-10T10:00:00-07:00"),
            ("stale-3", "上月复盘", "2026-08-17T10:00:00-07:00"),
        ):
            self._saved_event(suffix, title, datetime.fromisoformat(start_at))
        target = self._saved_event(
            "call-in", "客户电话会", datetime.fromisoformat("2026-09-12T09:00:00-07:00")
        )
        content = "明天的客户电话会取消了，删掉它"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_delete",
                    "action": {
                        "action": "delete_event",
                        "target_event_id": target.id,
                        "evidence": [{"message_id": "stale-crowding-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "stale-crowding-turn"
        )

        self.assertIn(target.id, [event["id"] for event in model.calls[0]["candidate_events"]])
        self.assertEqual(turn.outcome, "calendar_delete")
        self.assertEqual(self.store.get_calendar_event(target.id).status, "deleted")

    def test_ambiguous_calendar_delete_creates_a_clarification_without_a_write(self) -> None:
        self._saved_event(
            "morning-standup", "站会", datetime.fromisoformat("2026-09-10T09:00:00-07:00")
        )
        self._saved_event(
            "afternoon-briefing", "说明会", datetime.fromisoformat("2026-09-10T16:00:00-07:00")
        )
        content = "删除过期会议"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "clarify",
                    "clarification_question": "Which local calendar event should I delete?",
                    "pending_action": {
                        "action": "delete_event",
                        "target_event_id": None,
                        "evidence": [{"message_id": "ambiguous-delete-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "ambiguous-delete-turn"
        )
        pending = self.store.get_pending_clarification(self.conversation.id)

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(len(self.store.list_calendar_events()), 2)
        self.assertEqual(self.store.list_calendar_event_update_operations("ambiguous-delete-turn"), [])
        self.assertIsNotNone(pending)
        self.assertEqual(pending.original_action["action"], "delete_event")
        self.assertEqual(
            turn.assistant_message.content, "Which local calendar event should I delete?"
        )

    def test_calendar_delete_outside_supplied_candidates_changes_nothing(self) -> None:
        event = self._saved_event(
            "kept-meeting", "架构评审", datetime.fromisoformat("2026-09-10T16:00:00-07:00")
        )
        content = "删除那个会议"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_delete",
                    "action": {
                        "action": "delete_event",
                        "target_event_id": "invented-event",
                        "evidence": [{"message_id": "invalid-delete-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "invalid-delete-turn"
        )

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertEqual(self.store.list_calendar_events(), [event])
        self.assertEqual(self.store.list_calendar_event_update_operations("invalid-delete-turn"), [])

    def test_tracker_change_uses_current_decision_without_another_model_call(self) -> None:
        content = "排查 ROS 2 发布者-订阅者节点之间的延迟问题。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": content,
                            "source_quote": content,
                            "target_record_id": None,
                            "target_state": None,
                            "rationale": None,
                        }
                    ],
                }
            ]
        )

        turn = ConversationRouterService(
            self.store, model, date(2026, 9, 9)
        ).submit(self.conversation.id, content, "tracker-turn")

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            [record.content for record in self.store.list_project_records(self.project.id)], [content]
        )

    def test_project_answer_keeps_only_supplied_candidate_source_cards(self) -> None:
        source = self.store.save_message(
            Message(
                id="project-answer-source",
                conversation_id=self.conversation.id,
                role="user",
                content="排查 ROS 2 节点延迟。",
            )
        )
        record = self.store.save_project_record(
            ProjectRecord(
                id="ros-delay-task",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content=source.content,
                state="active",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        query = "项目现在有什么任务？"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "project_answer",
                    "reply": "当前任务是排查 ROS 2 节点延迟。",
                    "source_ids": [record.id],
                }
            ]
        )

        turn = ConversationRouterService(
            self.store, model, date(2026, 9, 9)
        ).submit(self.conversation.id, query, "project-answer-turn")

        self.assertEqual(turn.outcome, "project_answer")
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(
            [source.id for source in self.store.list_project_answer_sources(turn.assistant_message.id)],
            [record.id],
        )

    def test_invalid_event_target_leaves_local_data_unchanged(self) -> None:
        content = "把会议改到下午两点。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": {
                        "action": "update_event",
                        "content": None,
                        "description": None,
                        "date_expression": None,
                        "time_expression": "下午两点",
                        "end_time_expression": None,
                        "target_event_id": "invented-event",
                        "evidence": [{"message_id": "invalid-target-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(
            self.store, model, date(2026, 9, 9)
        ).submit(self.conversation.id, content, "invalid-target-turn")

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertEqual(self.store.list_calendar_events(), [])

    def test_unresolved_update_date_creates_a_pending_clarification_without_a_write(self) -> None:
        source = self.store.save_message(
            Message(
                id="ambiguous-event-source",
                conversation_id=self.conversation.id,
                role="user",
                content="本周四下午四点有工作流会议。",
            )
        )
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="ambiguous-workflow-event",
                conversation_id=self.conversation.id,
                title="工作流会议",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=datetime.fromisoformat("2026-09-10T16:00:00-07:00"),
                end_at=datetime.fromisoformat("2026-09-10T17:00:00-07:00"),
                all_day=False,
                timezone="America/Vancouver",
            )
        )
        correction = "这个会议改到周三。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": {
                        "action": "update_event",
                        "content": None,
                        "description": None,
                        "date_expression": "周三",
                        "time_expression": None,
                        "end_time_expression": None,
                        "target_event_id": event.id,
                        "evidence": [
                            {
                                "message_id": "ambiguous-correction-turn",
                                "source_quote": correction,
                            }
                        ],
                    },
                }
            ]
        )

        turn = ConversationRouterService(
            self.store, model, date(2026, 9, 9)
        ).submit(self.conversation.id, correction, "ambiguous-correction-turn")

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(
            self.store.list_calendar_event_update_operations("ambiguous-correction-turn"), []
        )
        self.assertIsNotNone(self.store.get_pending_clarification(self.conversation.id))

    def test_unresolved_event_creation_clarifies_then_persists_all_message_sources(self) -> None:
        original = "与 Daohui 碰头，讨论 UR5 交接和 Vention 工装板尺寸。"
        answer = "会议时间是 2026-09-11 下午两点。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "与 Daohui 碰头",
                        "description": "讨论 UR5 交接和 Vention 工装板尺寸。",
                        "date_expression": "周四",
                        "time_expression": None,
                        "end_time_expression": None,
                        "target_event_id": None,
                        "evidence": [{"message_id": "initial-event-turn", "source_quote": original}],
                    },
                },
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "与 Daohui 碰头",
                        "description": "讨论 UR5 交接和 Vention 工装板尺寸。",
                        "date_expression": "2026-09-11",
                        "time_expression": "下午两点",
                        "end_time_expression": None,
                        "target_event_id": None,
                        "evidence": [
                            {"message_id": "initial-event-turn", "source_quote": original},
                            {"message_id": "date-answer-turn", "source_quote": answer},
                        ],
                    },
                },
            ]
        )
        service = ConversationRouterService(self.store, model, date(2026, 9, 10))

        clarification = service.submit(self.conversation.id, original, "initial-event-turn")

        self.assertEqual(clarification.outcome, "clarification_created")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(self.store.list_tool_operations("initial-event-turn"), [])
        pending = self.store.get_pending_clarification(self.conversation.id)
        self.assertIsNotNone(pending)
        self.assertEqual(pending.original_action["date_expression"], "周四")

        created = service.submit(self.conversation.id, answer, "date-answer-turn")

        event = self.store.list_calendar_events()[0]
        self.assertEqual(created.outcome, "calendar_create")
        self.assertEqual(event.title, "与 Daohui 碰头")
        self.assertEqual(event.description, "讨论 UR5 交接和 Vention 工装板尺寸。")
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-11T14:00:00-07:00"))
        self.assertIn("2026-09-11T14:00:00-07:00 (America/Vancouver)", created.assistant_message.content)
        expected_sources = [
            EvidenceReference(message_id="initial-event-turn", source_quote=original),
            EvidenceReference(message_id="date-answer-turn", source_quote=answer),
        ]
        self.assertEqual(self.store.list_calendar_event_sources(event.id), expected_sources)
        operation = self.store.list_tool_operations("date-answer-turn")[0]
        self.assertEqual(
            operation.request["evidence"],
            [source.model_dump(mode="json") for source in expected_sources],
        )
        self.assertIsNone(self.store.get_pending_clarification(self.conversation.id))

    def test_resumed_calendar_creation_restores_all_persisted_sources(self) -> None:
        original = self.store.save_message(
            Message(
                id="resume-original-turn",
                conversation_id=self.conversation.id,
                role="user",
                content="与 Daohui 碰头，讨论 UR5 交接。",
            )
        )
        current = self.store.save_message(
            Message(
                id="resume-date-turn",
                conversation_id=self.conversation.id,
                role="user",
                content="会议时间是 2026-09-11 下午两点。",
            )
        )
        sources = [
            EvidenceReference(message_id=original.id, source_quote=original.content),
            EvidenceReference(message_id=current.id, source_quote=current.content),
        ]
        self.store.save_tool_operation(
            ToolOperation(
                id="prepared-contextual-event",
                source_message_id=current.id,
                action_index=0,
                tool_name="create_local_event",
                request={
                    "title": "与 Daohui 碰头",
                    "description": "讨论 UR5 交接。",
                    "source_quote": current.content,
                    "evidence": [source.model_dump(mode="json") for source in sources],
                    "start_at": "2026-09-11T14:00:00-07:00",
                    "end_at": None,
                    "all_day": False,
                    "timezone": "America/Vancouver",
                },
                status="prepared",
            )
        )
        model = ScriptedContextualModel([])

        resumed = ConversationRouterService(self.store, model, date(2026, 9, 10)).submit(
            self.conversation.id, current.content, current.id
        )

        event = self.store.list_calendar_events()[0]
        self.assertEqual(resumed.outcome, "resumed")
        self.assertEqual(resumed.model_calls, 0)
        self.assertEqual(event.title, "与 Daohui 碰头")
        self.assertEqual(self.store.list_calendar_event_sources(event.id), sources)
        self.assertEqual(model.calls, [])
