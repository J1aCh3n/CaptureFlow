import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from captureflow.llm import InvalidStructuredOutput, ModelError
from captureflow.models import Conversation, Message, ProposedAction, ToolOperation, TurnDecision
from captureflow.router_service import BoundedChatService
from captureflow.storage import CaptureStore


class ScriptedTurnRouter:
    def __init__(self, decisions, repairs=(), fail=False):
        self.decisions = list(decisions)
        self.repairs = list(repairs)
        self.fail = fail
        self.calls = []

    def route_turn(self, user_message, reference_date, timezone, clarification_context=None):
        self.calls.append(("route", clarification_context))
        if self.fail:
            raise ModelError("router unavailable")
        decision = self.decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision

    def repair_turn(
        self, user_message, reference_date, timezone, invalid_decision, validation_error,
        clarification_context=None,
    ):
        self.calls.append(("repair", clarification_context))
        return self.repairs.pop(0)


def event(
    content,
    quote,
    date_expression,
    time_expression=None,
    description=None,
    end_time_expression=None,
):
    return ProposedAction(
        action="create_event",
        content=content,
        description=description,
        source_quote=quote,
        date_expression=date_expression,
        time_expression=time_expression,
        end_time_expression=end_time_expression,
    )


class BoundedChatServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.conversation = self.store.save_conversation(
            Conversation(id="router-conversation", title="Router")
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def service(self, router):
        return BoundedChatService(
            self.store, router, reference_date=date(2026, 9, 8), timezone="America/Vancouver"
        )

    def test_deadline_creates_one_source_grounded_event(self) -> None:
        text = "My report is due tomorrow."
        router = ScriptedTurnRouter(
            [TurnDecision(route="execute", actions=[event("Submit report", text, "tomorrow")])]
        )

        turn = self.service(router).submit(self.conversation.id, text, "deadline-message")
        saved = self.store.list_calendar_events()

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].start_at, date(2026, 9, 9))
        self.assertEqual(saved[0].source_quote, text)
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(len(router.calls), 1)
        self.assertEqual(turn.diagnostics.outcome, "committed")
        self.assertGreaterEqual(turn.diagnostics.model_seconds, 0)
        self.assertGreaterEqual(turn.diagnostics.local_seconds, 0)
        self.assertFalse(turn.diagnostics.repair_used)

    def test_explicit_time_creates_a_timed_event(self) -> None:
        text = "Remind me to review results tomorrow at 15:30."
        router = ScriptedTurnRouter(
            [TurnDecision(route="execute", actions=[event("Review results", text, "tomorrow", "15:30")])]
        )

        self.service(router).submit(self.conversation.id, text, "timed-message")
        saved = self.store.list_calendar_events()[0]

        self.assertFalse(saved.all_day)
        self.assertEqual(saved.start_at, datetime(2026, 9, 9, 15, 30, tzinfo=saved.start_at.tzinfo))

    def test_supported_explicit_chinese_times_create_timed_events(self) -> None:
        cases = [
            ("明天下午两点开会。", "下午两点", 14, 0),
            ("明天上午九点半开会。", "上午九点半", 9, 30),
            ("明天晚上八点开会。", "晚上八点", 20, 0),
        ]
        for index, (text, expression, hour, minute) in enumerate(cases):
            router = ScriptedTurnRouter(
                [TurnDecision(route="execute", actions=[event("Meeting", text, "明天", expression)])]
            )

            turn = self.service(router).submit(self.conversation.id, text, f"chinese-time-{index}")
            saved = self.store.get_calendar_event(turn.operations[0].result["event_id"])

            self.assertFalse(saved.all_day)
            self.assertEqual((saved.start_at.hour, saved.start_at.minute), (hour, minute))

    def test_description_and_explicit_time_range_are_preserved(self) -> None:
        text = (
            "仿真环境与基准测试同步\n"
            "会议时间：2026年9月11日 (周五) 10:30 - 11:30\n"
            "会议主题：NVIDIA Isaac Sim 仿真环境进度汇报，重点讨论柔性线缆操作的"
            "打结与理线基准测试数据收集情况。"
        )
        description = (
            "NVIDIA Isaac Sim 仿真环境进度汇报，重点讨论柔性线缆操作的"
            "打结与理线基准测试数据收集情况。"
        )
        router = ScriptedTurnRouter(
            [
                TurnDecision(
                    route="execute",
                    actions=[
                        event(
                            "仿真环境与基准测试同步",
                            text,
                            "2026年9月11日 (周五)",
                            "10:30",
                            description,
                            "11:30",
                        )
                    ],
                )
            ]
        )

        turn = self.service(router).submit(self.conversation.id, text, "meeting-range")
        saved = self.store.get_calendar_event(turn.operations[0].result["event_id"])

        self.assertEqual(saved.description, description)
        self.assertEqual((saved.start_at.hour, saved.start_at.minute), (10, 30))
        self.assertEqual((saved.end_at.hour, saved.end_at.minute), (11, 30))
        self.assertEqual(saved.source_quote, text)

    def test_day_after_tomorrow_meeting_keeps_attendance_details(self) -> None:
        text = "后天下午四点有组会，所有人都要按时到场，并带好工牌"
        description = "所有人都要按时到场，并带好工牌"
        router = ScriptedTurnRouter(
            [
                TurnDecision(
                    route="execute",
                    actions=[
                        event("组会", text, "后天", "下午四点", description)
                    ],
                )
            ]
        )

        turn = self.service(router).submit(self.conversation.id, text, "group-meeting")
        saved = self.store.get_calendar_event(turn.operations[0].result["event_id"])

        self.assertEqual(saved.start_at.date(), date(2026, 9, 10))
        self.assertEqual((saved.start_at.hour, saved.start_at.minute), (16, 0))
        self.assertEqual(saved.description, description)

    def test_event_description_must_come_from_the_source(self) -> None:
        text = "明天下午两点有组会。"
        invalid = TurnDecision(
            route="execute",
            actions=[event("组会", text, "明天", "下午两点", "携带工牌")],
        )
        router = ScriptedTurnRouter([invalid], [invalid])

        turn = self.service(router).submit(self.conversation.id, text, "invented-description")

        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(turn.diagnostics.outcome, "repair_invalid")

    def test_ambiguous_chinese_time_does_not_create_an_event(self) -> None:
        text = "明天两点开会。"
        router = ScriptedTurnRouter(
            [TurnDecision(route="execute", actions=[event("Meeting", text, "明天", "两点")])]
        )

        turn = self.service(router).submit(self.conversation.id, text, "ambiguous-chinese-time")

        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertIn("Which exact time", turn.assistant_message.content)

    def test_idea_note_and_two_independent_actions_commit(self) -> None:
        text = "Idea: use a migration smoke test. Note: keep SQLite additive."
        router = ScriptedTurnRouter(
            [
                TurnDecision(
                    route="execute",
                    actions=[
                        ProposedAction(action="save_idea", content="Use a migration smoke test.", source_quote="Idea: use a migration smoke test."),
                        ProposedAction(action="save_note", content="Keep SQLite additive.", source_quote="Note: keep SQLite additive."),
                    ],
                )
            ]
        )

        turn = self.service(router).submit(self.conversation.id, text, "items-message")

        self.assertEqual({item.kind for item in self.store.list_saved_items()}, {"idea", "note"})
        self.assertEqual(len(turn.operations), 2)
        self.assertTrue(all(operation.status == "committed" for operation in turn.operations))

    def test_ambiguous_date_creates_a_clarification_and_no_target(self) -> None:
        text = "My report is due Tuesday."
        router = ScriptedTurnRouter(
            [TurnDecision(route="execute", actions=[event("Submit report", text, "Tuesday")])]
        )

        self.service(router).submit(self.conversation.id, text, "ambiguous-message")

        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(self.store.list_tool_operations("ambiguous-message"), [])
        self.assertIsNotNone(self.store.get_pending_clarification(self.conversation.id))

    def test_mixed_turn_with_an_ambiguous_event_creates_nothing(self) -> None:
        text = "My report is due Tuesday. Note: keep SQLite additive."
        router = ScriptedTurnRouter(
            [
                TurnDecision(
                    route="execute",
                    actions=[
                        event("Submit report", "My report is due Tuesday.", "Tuesday"),
                        ProposedAction(
                            action="save_note",
                            content="Keep SQLite additive.",
                            source_quote="Note: keep SQLite additive.",
                        ),
                    ],
                )
            ]
        )

        self.service(router).submit(self.conversation.id, text, "mixed-message")

        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(self.store.list_saved_items(), [])
        self.assertEqual(self.store.list_tool_operations("mixed-message"), [])

    def test_follow_up_resolves_pending_once(self) -> None:
        original = "My report is due Tuesday."
        answer = "next Tuesday"
        router = ScriptedTurnRouter(
            [
                TurnDecision(route="execute", actions=[event("Submit report", original, "Tuesday")]),
                TurnDecision(route="execute", actions=[event("Submit report", answer, "next Tuesday")]),
            ]
        )
        service = self.service(router)
        service.submit(self.conversation.id, original, "original-message")

        resolved = service.submit(self.conversation.id, answer, "answer-message")
        repeated = service.submit(self.conversation.id, answer, "answer-message")

        self.assertEqual(len(self.store.list_calendar_events()), 1)
        self.assertEqual(self.store.get_pending_clarification(self.conversation.id), None)
        self.assertEqual(resolved.assistant_message, repeated.assistant_message)
        self.assertEqual(len(router.calls), 2)

    def test_invalid_decision_gets_one_repair_then_commits(self) -> None:
        text = "Save this idea: add tests."
        invalid = {
            "route": "execute",
            "actions": [{"action": "save_idea", "content": "Add tests.", "source_quote": "invented"}],
        }
        repaired = TurnDecision(
            route="execute",
            actions=[ProposedAction(action="save_idea", content="Add tests.", source_quote=text)],
        )
        router = ScriptedTurnRouter([invalid], [repaired])

        turn = self.service(router).submit(self.conversation.id, text, "repair-message")

        self.assertEqual(turn.model_calls, 2)
        self.assertEqual(len(self.store.list_saved_items()), 1)
        self.assertEqual([call[0] for call in router.calls], ["route", "repair"])
        self.assertTrue(turn.diagnostics.repair_used)

    def test_schema_invalid_router_output_gets_one_repair(self) -> None:
        text = "Save this idea: add tests."
        repaired = TurnDecision(
            route="execute",
            actions=[ProposedAction(action="save_idea", content="Add tests.", source_quote=text)],
        )
        router = ScriptedTurnRouter(
            [InvalidStructuredOutput("invalid router JSON", "not JSON")], [repaired]
        )

        turn = self.service(router).submit(self.conversation.id, text, "schema-repair-message")

        self.assertEqual(turn.model_calls, 2)
        self.assertEqual(len(self.store.list_saved_items()), 1)

    def test_repair_exhaustion_creates_no_target(self) -> None:
        text = "Save this idea: add tests."
        invalid = {
            "route": "execute",
            "actions": [{"action": "save_idea", "content": "Add tests.", "source_quote": "invented"}],
        }
        router = ScriptedTurnRouter([invalid], [invalid])

        turn = self.service(router).submit(self.conversation.id, text, "failed-repair-message")

        self.assertEqual(turn.model_calls, 2)
        self.assertEqual(self.store.list_saved_items(), [])
        self.assertIn("after one repair", turn.assistant_message.content)

    def test_reply_only_model_failure_and_credential_never_write(self) -> None:
        reply_router = ScriptedTurnRouter([TurnDecision(route="reply_only", reply="I only save local records.")])
        reply = self.service(reply_router).submit(self.conversation.id, "What is the weather?", "reply-message")
        failure = self.service(ScriptedTurnRouter([], fail=True)).submit(
            self.conversation.id, "Save an idea", "failure-message"
        )
        credential = self.service(ScriptedTurnRouter([], fail=True)).submit(
            self.conversation.id, "api_key=abc", "credential-message"
        )

        self.assertEqual(self.store.list_saved_items(), [])
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(reply.model_calls, 1)
        self.assertEqual(failure.model_calls, 1)
        self.assertEqual(credential.model_calls, 0)

    def test_prepared_operation_resumes_without_router_call(self) -> None:
        source = self.store.save_message(
            Message(
                id="prepared-message",
                conversation_id=self.conversation.id,
                role="user",
                content="Save this idea.",
            )
        )
        operation = self.store.save_tool_operation(
            ToolOperation(
                id="prepared-operation",
                source_message_id=source.id,
                action_index=0,
                tool_name="save_idea",
                request={"content": "Add tests.", "source_quote": source.content},
                status="prepared",
            )
        )
        router = ScriptedTurnRouter([], fail=True)

        restarted = CaptureStore(self.store.database_path)
        restarted.initialize()
        turn = BoundedChatService(
            restarted, router, reference_date=date(2026, 9, 8), timezone="America/Vancouver"
        ).submit(self.conversation.id, source.content, source.id)

        self.assertEqual(turn.operations[0].status, "committed")
        self.assertEqual(restarted.get_tool_operation(operation.id).status, "committed")
        self.assertEqual(len(router.calls), 0)
        self.assertEqual(turn.diagnostics.outcome, "resumed")
        self.assertEqual(turn.diagnostics.model_seconds, 0)

    def test_prepared_calendar_range_resumes_with_description(self) -> None:
        source = self.store.save_message(
            Message(
                id="prepared-calendar-message",
                conversation_id=self.conversation.id,
                role="user",
                content="2026-09-11 10:30 - 11:30 benchmark meeting.",
            )
        )
        self.store.save_tool_operation(
            ToolOperation(
                id="prepared-calendar-operation",
                source_message_id=source.id,
                action_index=0,
                tool_name="create_local_event",
                request={
                    "title": "Benchmark meeting",
                    "description": "benchmark meeting",
                    "source_quote": source.content,
                    "start_at": "2026-09-11T10:30:00-07:00",
                    "end_at": "2026-09-11T11:30:00-07:00",
                    "all_day": False,
                    "timezone": "America/Vancouver",
                },
                status="prepared",
            )
        )

        restarted = CaptureStore(self.store.database_path)
        restarted.initialize()
        turn = BoundedChatService(
            restarted,
            ScriptedTurnRouter([], fail=True),
            reference_date=date(2026, 9, 8),
            timezone="America/Vancouver",
        ).submit(self.conversation.id, source.content, source.id)
        saved = restarted.get_calendar_event(turn.operations[0].result["event_id"])

        self.assertEqual(saved.description, "benchmark meeting")
        self.assertEqual(saved.end_at.isoformat(), "2026-09-11T11:30:00-07:00")
        self.assertEqual(turn.diagnostics.outcome, "resumed")
