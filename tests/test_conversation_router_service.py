import tempfile
import unittest
from datetime import date, datetime, time, timezone
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
                        "date": "2026-09-10",
                        "start_time": "16:00",
                        "end_time": "17:00",
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
        self.assertEqual(model.calls[0]["calendar"][7]["date"], "2026-09-09")
        self.assertIsNone(model.calls[0]["reference_time"])
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
                        "date": "2026-09-11",
                        "start_time": "14:00",
                        "end_time": None,
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
                        "date": "2026-09-10",
                        "start_time": "14:00",
                        "end_time": "15:00",
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

    def test_an_end_time_and_a_duration_that_agree_create_the_event(self) -> None:
        for label, start, end, minutes, expected_end in (
            ("same-day", "16:00", "17:00", 60, "2026-09-10T17:00:00-07:00"),
            ("past-midnight", "23:30", "00:30", 60, "2026-09-11T00:30:00-07:00"),
        ):
            with self.subTest(label=label):
                content = f"明天{start}到{end}有代码审查会议，开{minutes}分钟。"
                message_id = f"end-and-duration-{label}"
                model = ScriptedContextualModel(
                    [
                        {
                            "intent": "calendar_create",
                            "action": {
                                "action": "create_event",
                                "content": f"代码审查会议 {label}",
                                "date": "2026-09-10",
                                "start_time": start,
                                "end_time": end,
                                "duration_minutes": minutes,
                                "evidence": [{"message_id": message_id, "source_quote": content}],
                            },
                        }
                    ]
                )

                turn = ConversationRouterService(self.store, model, date(2026, 9, 9)).submit(
                    self.conversation.id, content, message_id
                )

                self.assertEqual(turn.outcome, "calendar_create")
                event = self.store.get_calendar_event(turn.operations[0].result["event_id"])
                self.assertEqual(event.end_at, datetime.fromisoformat(expected_end))

    def test_an_end_time_and_a_duration_that_disagree_ask_for_the_length(self) -> None:
        content = "明天下午四点到五点有代码审查会议，开一个半小时。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "代码审查会议",
                        "date": "2026-09-10",
                        "start_time": "16:00",
                        "end_time": "17:00",
                        "duration_minutes": 90,
                        "evidence": [{"message_id": "end-duration-conflict", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 9)).submit(
            self.conversation.id, content, "end-duration-conflict"
        )

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(self.store.get_pending_clarification(self.conversation.id).missing_field, "duration")

    def test_calendar_update_applies_a_new_start_and_duration(self) -> None:
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
                        "date": "2026-09-10",
                        "start_time": "16:00",
                        "end_time": None,
                        "duration_minutes": None,
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
                        "date": None,
                        "start_time": "19:00",
                        "end_time": None,
                        "duration_minutes": 30,
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

    def _fresh_conversation(self) -> Conversation:
        conversation = self.store.save_conversation(
            Conversation(id="fresh-conversation", title="Fresh")
        )
        self.store.assign_conversation_project(conversation.id, self.project.id)
        return conversation

    def _update_decision(self, event_id: str, message_id: str, content: str) -> dict:
        return {
            "intent": "calendar_update",
            "action": {
                "action": "update_event",
                "date": "2026-09-10",
                "target_event_id": event_id,
                "evidence": [{"message_id": message_id, "source_quote": content}],
            },
        }

    def test_the_model_asking_which_date_an_update_means_is_kept_as_a_question(self) -> None:
        event = self._saved_event(
            "which-thursday", "算法优化与代码审查会议", datetime.fromisoformat("2026-09-10T14:00:00-07:00")
        )
        content = "这个会议改到周四。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "clarify",
                    "clarification_question": "你是指 9月10日（明天）还是 9月17日的周四？",
                    "pending_action": {
                        "action": "update_event",
                        "date": None,
                        "target_event_id": event.id,
                        "evidence": [{"message_id": "which-thursday-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 9)).submit(
            self.conversation.id, content, "which-thursday-turn"
        )

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(
            self.store.get_pending_clarification(self.conversation.id).missing_field, "date"
        )

    def test_an_update_that_changes_nothing_writes_nothing(self) -> None:
        event = self._saved_event(
            "already-thursday", "架构评审", datetime.fromisoformat("2026-09-10T14:00:00-07:00")
        )
        content = "架构评审改到周四。"
        model = ScriptedContextualModel([self._update_decision(event.id, "no-op-update", content)])

        turn = ConversationRouterService(self.store, model, date(2026, 9, 9)).submit(
            self.conversation.id, content, "no-op-update"
        )

        self.assertEqual(turn.outcome, "calendar_unchanged")
        self.assertEqual(turn.operations, ())
        self.assertEqual(self.store.list_calendar_event_update_operations("no-op-update"), [])
        self.assertEqual(self.store.get_calendar_event(event.id).updated_at, event.updated_at)
        self.assertIn("nothing was changed", turn.assistant_message.content)

    def test_a_demonstrative_in_a_new_conversation_asks_which_event(self) -> None:
        event = self._saved_event(
            "review-meeting",
            "算法优化与代码审查会议",
            datetime.fromisoformat("2026-09-10T14:00:00-07:00"),
        )
        fresh = self._fresh_conversation()
        content = "这个会议改到周四。"
        model = ScriptedContextualModel([self._update_decision(event.id, "fresh-demonstrative", content)])

        turn = ConversationRouterService(self.store, model, date(2026, 9, 9)).submit(
            fresh.id, content, "fresh-demonstrative"
        )

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(turn.operations, ())
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.get_pending_clarification(fresh.id).missing_field, "target")
        self.assertIn("has not mentioned that event", turn.assistant_message.content)

    def test_naming_the_event_in_a_new_conversation_still_updates_it(self) -> None:
        event = self._saved_event(
            "named-meeting",
            "算法优化与代码审查会议",
            datetime.fromisoformat("2026-09-10T14:00:00-07:00"),
        )
        fresh = self._fresh_conversation()
        content = "把算法优化与代码审查会议改到周四下午三点。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": {
                        "action": "update_event",
                        "date": "2026-09-10",
                        "start_time": "15:00",
                        "target_event_id": event.id,
                        "evidence": [{"message_id": "fresh-named", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 9)).submit(
            fresh.id, content, "fresh-named"
        )

        self.assertEqual(turn.outcome, "calendar_update")
        self.assertEqual(
            self.store.get_calendar_event(event.id).start_at,
            datetime.fromisoformat("2026-09-10T15:00:00-07:00"),
        )

    def test_a_date_word_does_not_count_as_naming_the_event(self) -> None:
        event = self._saved_event(
            "weekly", "周例会", datetime.fromisoformat("2026-09-10T10:00:00-07:00")
        )
        fresh = self._fresh_conversation()
        content = "那个会删掉，周四不开了"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_delete",
                    "action": {
                        "action": "delete_event",
                        "target_event_id": event.id,
                        "evidence": [{"message_id": "fresh-delete", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 9)).submit(
            fresh.id, content, "fresh-delete"
        )

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.get_calendar_event(event.id).status, "active")

    def _pending_time_question(self) -> None:
        content = "明天有个评审会"
        ConversationRouterService(
            self.store,
            ScriptedContextualModel(
                [
                    {
                        "intent": "calendar_create",
                        "action": {
                            "action": "create_event",
                            "content": "评审会",
                            "evidence": [{"message_id": "pending-time-turn", "source_quote": content}],
                        },
                    }
                ]
            ),
            date(2026, 9, 11),
        ).submit(self.conversation.id, content, "pending-time-turn")

    def test_an_unrelated_task_does_not_discard_a_pending_calendar_question(self) -> None:
        self._pending_time_question()
        content = "我需要排查 ROS 2 节点延迟。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "排查 ROS 2 节点延迟",
                            "source_quote": content,
                        }
                    ],
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "unrelated-task-turn"
        )
        pending = self.store.get_pending_clarification(self.conversation.id)

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertIsNotNone(pending)
        self.assertEqual(pending.source_message_id, "pending-time-turn")

    def test_a_follow_up_question_replaces_an_unanswered_calendar_question(self) -> None:
        self._pending_time_question()
        content = "会议分析了驱动调试。修复需要下周三之前提交"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "提交修复",
                            "source_quote": "修复需要下周三之前提交",
                        }
                    ],
                    "deferred_action": {
                        "action": "save_note",
                        "content": "会议分析了驱动调试。",
                        "evidence": [{"message_id": "follow-up-over-stale", "source_quote": content}],
                    },
                    "follow_up_question": "要保存会议纪要吗？",
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "follow-up-over-stale"
        )
        pending = self.store.get_pending_clarification(self.conversation.id)

        self.assertIn("要保存会议纪要吗？", turn.assistant_message.content)
        self.assertEqual(pending.missing_field, "follow_up")
        self.assertEqual(pending.source_message_id, "follow-up-over-stale")

    def test_a_second_unanswered_calendar_question_replaces_the_first(self) -> None:
        first = "周末有个会"
        second = "下下周三下午三点有评审会"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "会",
                        "evidence": [{"message_id": "first-question-turn", "source_quote": first}],
                    },
                },
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "评审会",
                        "evidence": [{"message_id": "second-question-turn", "source_quote": second}],
                    },
                },
            ]
        )
        service = ConversationRouterService(self.store, model, date(2026, 9, 11))

        service.submit(self.conversation.id, first, "first-question-turn")
        replaced = self.store.get_pending_clarification(self.conversation.id)
        turn = service.submit(self.conversation.id, second, "second-question-turn")
        current = self.store.get_pending_clarification(self.conversation.id)

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(current.source_message_id, "second-question-turn")
        self.assertNotEqual(current.id, replaced.id)
        self.assertEqual(self.store.list_calendar_events(), [])

    def test_the_current_local_time_reaches_the_model(self) -> None:
        content = "半小时后开个短会。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "短会",
                        "date": "2026-09-12",
                        "start_time": "14:40",
                        "evidence": [{"message_id": "relative-now-turn", "source_quote": content}],
                    },
                }
            ]
        )

        ConversationRouterService(
            self.store, model, date(2026, 9, 12), reference_time=time(14, 10, 37)
        ).submit(self.conversation.id, content, "relative-now-turn")

        self.assertEqual(model.calls[0]["reference_time"], "14:10")
        self.assertEqual(model.calls[0]["reference_date"], "2026-09-12")
        self.assertEqual(
            self.store.list_calendar_events()[0].start_at,
            datetime.fromisoformat("2026-09-12T14:40:00-07:00"),
        )

    def test_python_attaches_the_offset_a_date_actually_has(self) -> None:
        content = "11月2日下午三点到四点有季度评审。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "季度评审",
                        "date": "2026-11-02",
                        "start_time": "15:00",
                        "end_time": "16:00",
                        "evidence": [{"message_id": "after-dst-turn", "source_quote": content}],
                    },
                }
            ]
        )

        ConversationRouterService(self.store, model, date(2026, 9, 12)).submit(
            self.conversation.id, content, "after-dst-turn"
        )
        event = self.store.list_calendar_events()[0]

        self.assertEqual(event.start_at, datetime.fromisoformat("2026-11-02T15:00:00-08:00"))
        self.assertEqual(event.end_at, datetime.fromisoformat("2026-11-02T16:00:00-08:00"))

    def test_an_update_reply_states_the_new_schedule(self) -> None:
        event = self._saved_event(
            "echo-meeting", "架构评审", datetime.fromisoformat("2026-09-10T14:00:00-07:00")
        )
        content = "架构评审改到下午四点到五点。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": {
                        "action": "update_event",
                        "start_time": "16:00",
                        "end_time": "17:00",
                        "target_event_id": event.id,
                        "evidence": [{"message_id": "echo-update", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 9)).submit(
            self.conversation.id, content, "echo-update"
        )

        self.assertEqual(turn.outcome, "calendar_update")
        self.assertIn("2026-09-10T16:00:00-07:00", turn.assistant_message.content)
        self.assertIn("2026-09-10T17:00:00-07:00", turn.assistant_message.content)

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
                        "date": "2026-09-11",
                        "start_time": "23:30",
                        "end_time": "00:30",
                        "duration_minutes": None,
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

    def test_a_malformed_normalized_time_is_refused_without_a_write(self) -> None:
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
                        "date": None,
                        "start_time": "7 PM",
                        "end_time": None,
                        "duration_minutes": None,
                        "target_event_id": event.id,
                        "evidence": [{"message_id": "unsupported-correction", "source_quote": correction}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 10)).submit(
            self.conversation.id, correction, "unsupported-correction"
        )

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_event_update_operations("unsupported-correction"), [])
        self.assertIsNone(self.store.get_pending_clarification(self.conversation.id))

    def test_a_new_length_for_an_all_day_event_asks_for_a_start_time(self) -> None:
        source = self.store.save_message(
            Message(
                id="duration-event-source",
                conversation_id=self.conversation.id,
                role="user",
                content="今天有工作流会议。",
            )
        )
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="duration-event",
                conversation_id=self.conversation.id,
                title="工作流会议",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=date(2026, 9, 10),
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        correction = "这个会议开半个小时。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": {
                        "action": "update_event",
                        "content": None,
                        "description": None,
                        "date": None,
                        "start_time": None,
                        "end_time": None,
                        "duration_minutes": 30,
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
            "What exact start time should I use for this local event?",
        )
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_event_update_operations("duration-correction"), [])
        self.assertEqual(
            self.store.get_pending_clarification(self.conversation.id).missing_field, "time"
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
                        "date": "2026-09-16",
                        "start_time": "20:00",
                        "evidence": [{"message_id": "compound-turn", "source_quote": content}],
                    },
                    "follow_up_question": "Should I also add a local calendar event for 下周三晚上八点?",
                },
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "提交修复",
                        "date": "2026-09-16",
                        "start_time": "20:00",
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
        self.assertEqual(pending.original_action["date"], "2026-09-16")
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

    def test_a_normalized_task_deadline_is_listed_for_the_calendar(self) -> None:
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
                            "due_date": "2026-09-16",
                            "due_time": "20:00",
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
                            "due_date": "2026-09-16",
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

    def test_an_implausible_deadline_keeps_the_task_without_one(self) -> None:
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
                            "due_date": "2031-01-01",
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
        self.assertIn("outside the planning horizon", turn.assistant_message.content)
        self.assertIn("no deadline yet", turn.assistant_message.content)

    def test_a_follow_up_sets_a_deadline_on_an_existing_task(self) -> None:
        content = "提醒我明天十点交材料。"
        answer = "上午十点"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "交材料",
                            "source_quote": content,
                        }
                    ],
                },
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "set_deadline",
                            "target_record_id": None,
                            "due_date": "2026-09-12",
                            "due_time": "10:00",
                            "source_quote": answer,
                        }
                    ],
                },
            ]
        )
        service = ConversationRouterService(self.store, model, date(2026, 9, 11))

        first = service.submit(self.conversation.id, content, "bare-hour-turn")
        record = self.store.list_project_records(self.project.id)[0]
        model.decisions[0]["project_changes"][0]["target_record_id"] = record.id

        self.assertEqual(first.outcome, "tracker_change")
        self.assertIsNone(record.due_at)

        second = service.submit(self.conversation.id, answer, "bare-hour-answer")
        updated = self.store.get_project_record(record.id)
        operation = self.store.list_project_change_operations("bare-hour-answer")[0]

        self.assertEqual(second.outcome, "tracker_change")
        self.assertEqual(updated.due_at, datetime.fromisoformat("2026-09-12T10:00:00-07:00"))
        self.assertEqual(self.store.list_task_deadlines(), [updated])
        self.assertEqual(
            self.store.undo_project_change_operation(operation.id).status, "undone"
        )
        self.assertIsNone(self.store.get_project_record(record.id).due_at)

    def test_a_description_the_user_did_not_write_is_dropped(self) -> None:
        content = "明天下午三点在 A301 和张伟开需求评审，讨论驱动排期"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "需求评审",
                        "description": "在 A301 和张伟开会，讨论驱动排期",
                        "date": "2026-09-12",
                        "start_time": "15:00",
                        "evidence": [{"message_id": "paraphrase-turn", "source_quote": content}],
                    },
                },
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "需求评审",
                        "description": "在 A301 和张伟开需求评审，讨论驱动排期",
                        "date": "2026-09-12",
                        "start_time": "15:00",
                        "evidence": [{"message_id": "quoted-turn", "source_quote": content}],
                    },
                },
            ]
        )
        service = ConversationRouterService(self.store, model, date(2026, 9, 11))

        service.submit(self.conversation.id, content, "paraphrase-turn")
        paraphrased = self.store.list_calendar_events()[0]

        self.assertIsNone(paraphrased.description)

        service.submit(self.conversation.id, content, "quoted-turn")
        quoted = [event for event in self.store.list_calendar_events() if event.id != paraphrased.id][0]

        self.assertEqual(quoted.description, "在 A301 和张伟开需求评审，讨论驱动排期")

    def test_an_ambiguous_target_is_recorded_as_a_target_clarification(self) -> None:
        self._saved_event(
            "standup", "站会", datetime.fromisoformat("2026-09-14T09:00:00-07:00")
        )
        self._saved_event(
            "review", "需求评审", datetime.fromisoformat("2026-09-15T15:00:00-07:00")
        )
        content = "把那个会删掉"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "clarify",
                    "clarification_question": "Which local calendar event should I delete?",
                    "pending_action": {
                        "action": "delete_event",
                        "evidence": [{"message_id": "target-clarify-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "target-clarify-turn"
        )

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(
            self.store.get_pending_clarification(self.conversation.id).missing_field, "target"
        )
        self.assertEqual(len(self.store.list_calendar_events()), 2)

    def test_a_repeating_event_says_repeats_are_not_stored(self) -> None:
        content = "每周一下午三点有团队例会"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": {
                        "action": "create_event",
                        "content": "团队例会",
                        "date": None,
                        "recurring": True,
                        "start_time": "15:00",
                        "evidence": [{"message_id": "recurring-turn", "source_quote": content}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "recurring-turn"
        )
        pending = self.store.get_pending_clarification(self.conversation.id)

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(pending.missing_field, "recurrence")
        self.assertIn("cannot store a repeating schedule", turn.assistant_message.content)

    def test_a_repeating_deadline_keeps_the_task_without_one(self) -> None:
        content = "每周三之前要交周报。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "交周报",
                            "due_date": "2026-09-16",
                            "recurring": True,
                            "source_quote": content,
                        }
                    ],
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 11)).submit(
            self.conversation.id, content, "recurring-deadline-turn"
        )
        record = self.store.list_project_records(self.project.id)[0]

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertIsNone(record.due_at)
        self.assertIn("repeating deadline", turn.assistant_message.content)

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
                            "due_date": "2026-09-16",
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
                        "date": "2026-09-16",
                        "start_time": "20:00",
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
                        "date": "2026-09-12",
                        "timezone": "America/New_York",
                        "start_time": "09:00",
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
                        "date": "2026-09-12",
                        "timezone": "America/Vancouver",
                        "start_time": "09:00",
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

    def test_an_older_task_is_reachable_once_a_project_has_many(self) -> None:
        for index, content in enumerate(
            ["完成驱动调试", "完成代码审查", "整理会议纪要", "更新部署文档", "修复登录页样式"]
        ):
            source = self.store.save_message(
                Message(
                    id=f"many-tasks-source-{index}",
                    conversation_id=self.conversation.id,
                    role="user",
                    content=content,
                )
            )
            self.store.save_project_record(
                ProjectRecord(
                    id=f"many-tasks-{index}",
                    project_id=self.project.id,
                    conversation_id=self.conversation.id,
                    kind="task",
                    content=content,
                    state="active",
                    source_message_id=source.id,
                    source_quote=content,
                    updated_at=datetime(2026, 9, 1 + index, tzinfo=timezone.utc),
                )
            )
        content = "驱动调试的任务已经做完了"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "transition_record",
                            "target_record_id": "many-tasks-0",
                            "target_state": "completed",
                            "source_quote": content,
                        }
                    ],
                }
            ]
        )

        turn = ConversationRouterService(self.store, model, date(2026, 9, 12)).submit(
            self.conversation.id, content, "complete-oldest-task"
        )

        self.assertIn(
            "many-tasks-0",
            [record["id"] for record in model.calls[0]["candidate_project_records"]],
        )
        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(self.store.get_project_record("many-tasks-0").state, "completed")

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

    def test_a_project_answer_about_a_calendar_event_is_rendered_from_sqlite(self) -> None:
        source = self.store.save_message(
            Message(
                id="review-meeting-source",
                conversation_id=self.conversation.id,
                role="user",
                content="我明天下午两点到三点有算法优化与代码审查会议。",
            )
        )
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="review-meeting",
                conversation_id=self.conversation.id,
                title="算法优化与代码审查会议",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=datetime.fromisoformat("2026-09-10T14:00:00-07:00"),
                end_at=datetime.fromisoformat("2026-09-10T15:00:00-07:00"),
                all_day=False,
                timezone="America/Vancouver",
            )
        )
        query = "我明天的算法优化与代码审查会议几点开始？"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "project_answer",
                    "reply": "会议明天上午十点开始。",
                    "source_ids": [event.id],
                }
            ]
        )

        turn = ConversationRouterService(
            self.store, model, date(2026, 9, 9)
        ).submit(self.conversation.id, query, "calendar-project-answer-turn")

        self.assertEqual(turn.decision.intent, "project_answer")
        self.assertEqual(turn.outcome, "grounded_local_context")
        self.assertEqual(turn.model_calls, 1)
        self.assertNotIn("上午十点", turn.assistant_message.content)
        self.assertIn(event.start_at.isoformat(), turn.assistant_message.content)
        self.assertEqual(
            [
                source.reference.source_id
                for source in self.store.list_local_context_answer_sources(turn.assistant_message.id)
            ],
            [event.id],
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
                        "date": None,
                        "start_time": "14:00",
                        "end_time": None,
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
        self.assertEqual(
            turn.failure_reason, "ValueError: calendar update target is outside supplied candidates"
        )
        self.assertEqual(self.store.list_calendar_events(), [])

    def test_an_update_without_a_new_value_asks_for_the_date(self) -> None:
        source = self.store.save_message(
            Message(
                id="thursday-event-source",
                conversation_id=self.conversation.id,
                role="user",
                content="本周四下午四点有工作流会议。",
            )
        )
        event = self.store.save_calendar_event(
            CalendarEvent(
                id="thursday-workflow-event",
                conversation_id=self.conversation.id,
                title="工作流会议",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=datetime.fromisoformat("2026-09-10T16:00:00-07:00"),
                all_day=False,
                timezone="America/Vancouver",
            )
        )
        correction = "这个会议改到周四。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": {
                        "action": "update_event",
                        "target_event_id": event.id,
                        "date": None,
                        "evidence": [{"message_id": "undecided-update-turn", "source_quote": correction}],
                    },
                }
            ]
        )

        turn = ConversationRouterService(
            self.store, model, date(2026, 9, 9)
        ).submit(self.conversation.id, correction, "undecided-update-turn")

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_event_update_operations("undecided-update-turn"), [])
        self.assertEqual(self.store.get_pending_clarification(self.conversation.id).missing_field, "date")

    def test_an_implausible_update_date_asks_for_the_date(self) -> None:
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
                        "date": "2030-09-10",
                        "start_time": None,
                        "end_time": None,
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
        self.assertEqual(
            self.store.get_pending_clarification(self.conversation.id).missing_field, "date"
        )

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
                        "start_time": None,
                        "end_time": None,
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
                        "date": "2026-09-11",
                        "start_time": "14:00",
                        "end_time": None,
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
        self.assertEqual(pending.missing_field, "date")

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
