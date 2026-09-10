import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from captureflow.conversation_router_service import ConversationRouterService
from captureflow.models import (
    CalendarEvent,
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
