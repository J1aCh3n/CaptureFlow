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
from captureflow.read_agent import AgentFinalResponse, AgentModelTurn, AgentToolCall


class ScriptedContextualModel:
    def __init__(self, decisions, project_answers=(), agent_turns=()):
        self.decisions = list(decisions)
        self.project_answers = list(project_answers)
        self.agent_turns = list(agent_turns)
        self.calls = []
        self.project_answer_calls = []
        self.agent_calls = []

    def propose_conversation_decision(self, **kwargs):
        self.calls.append(kwargs)
        return self.decisions.pop(0)

    def answer_project_question(self, **kwargs):
        self.project_answer_calls.append(kwargs)
        return self.project_answers.pop(0)

    def next_read_agent_turn(self, **kwargs):
        self.agent_calls.append(kwargs)
        return self.agent_turns.pop(0)


def action(kind: str, *source_message_ids: str, **fields) -> dict:
    return {"action": kind, **fields, "source_message_ids": list(source_message_ids)}


class ConversationRouterServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.conversation = self.store.save_conversation(
            Conversation(id="contextual-router-chat", title="Contextual router")
        )
        self.project = self.store.save_project(Project(id="contextual-router-project", name="Workflow"))
        self.store.assign_conversation_project(self.conversation.id, self.project.id)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def submit(self, model, content, message_id, reference=date(2026, 9, 9), conversation_id=None, **options):
        return ConversationRouterService(self.store, model, reference, **options).submit(
            conversation_id or self.conversation.id, content, message_id
        )

    def _fresh_conversation(self, conversation_id: str = "fresh-conversation") -> Conversation:
        conversation = self.store.save_conversation(Conversation(id=conversation_id, title="Fresh"))
        self.store.assign_conversation_project(conversation.id, self.project.id)
        return conversation

    def _saved_event(self, suffix: str, title: str, start_at: datetime | date, **fields) -> CalendarEvent:
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
                all_day=not isinstance(start_at, datetime),
                timezone="America/Vancouver",
                **fields,
            )
        )

    def _project_record(self, record_id: str, kind: str, state: str, content: str, **fields) -> ProjectRecord:
        source = self.store.save_message(
            Message(id=f"{record_id}-source", conversation_id=self.conversation.id, role="user", content=content)
        )
        return self.store.save_project_record(
            ProjectRecord(
                id=record_id,
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind=kind,
                content=content,
                state=state,
                source_message_id=source.id,
                source_quote=content,
                **fields,
            )
        )

    # Optional Phase A read agent

    def test_agent_local_answer_uses_tools_but_python_renders_the_cited_rows(self) -> None:
        event = self._saved_event("agent-review", "Agent design review", date(2026, 9, 18))
        model = ScriptedContextualModel(
            [{"intent": "local_answer"}],
            agent_turns=[
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="calendar-search",
                            name="search_calendar",
                            arguments={
                                "start_date": "2026-09-18",
                                "end_date": "2026-09-18",
                            },
                        )
                    ]
                ),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        response_kind="grounded_answer",
                        reply="MODEL PROSE MUST NOT BE SHOWN",
                        source_ids=[event.id],
                    )
                ),
            ],
        )

        turn = self.submit(
            model,
            "What is scheduled on September 18?",
            "agent-local-turn",
            answer_mode="agent",
        )

        self.assertEqual(turn.outcome, "agent_local_answer")
        self.assertEqual(turn.model_calls, 3)
        self.assertEqual(turn.tool_calls, 1)
        self.assertEqual(turn.agent_stop_reason, "completed")
        self.assertIn(event.title, turn.assistant_message.content)
        self.assertIn(event.start_at.isoformat(), turn.assistant_message.content)
        self.assertNotIn("MODEL PROSE", turn.assistant_message.content)
        self.assertEqual(
            [source.reference.source_id for source in self.store.list_local_context_answer_sources(turn.assistant_message.id)],
            [event.id],
        )
        self.assertNotIn("candidate_events", model.agent_calls[0]["messages"][1]["content"])

    def test_agent_project_answer_shows_model_prose_and_sqlite_dates(self) -> None:
        task = self._project_record(
            "latency-task",
            "task",
            "active",
            "Measure agent latency",
            due_at=date(2026, 9, 25),
        )
        model = ScriptedContextualModel(
            [{"intent": "project_answer"}],
            agent_turns=[
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="record-search",
                            name="search_project_records",
                            arguments={"kinds": ["task"], "keywords": ["latency"]},
                        )
                    ]
                ),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        response_kind="grounded_answer",
                        reply="Latency measurement is still open.",
                        source_ids=[task.id],
                    )
                ),
            ],
        )

        turn = self.submit(
            model,
            "What remains for this project?",
            "agent-project-turn",
            answer_mode="agent",
        )

        self.assertEqual(turn.outcome, "agent_project_answer")
        self.assertIn("Latency measurement is still open.", turn.assistant_message.content)
        self.assertIn(task.due_at.isoformat(), turn.assistant_message.content)
        self.assertEqual(
            [record.id for record in self.store.list_project_answer_sources(turn.assistant_message.id)],
            [task.id],
        )

    # Calendar creation

    def test_event_statement_creates_calendar_event_not_project_task(self) -> None:
        content = "我这周四下午四点到五点有算法优化与代码审查会议。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action(
                        "create_event",
                        content="算法优化与代码审查",
                        description="讨论前缀树与最小堆性能瓶颈。",
                        date="2026-09-10",
                        start_time="16:00",
                        end_time="17:00",
                    ),
                }
            ]
        )

        turn = self.submit(model, content, "event-turn")
        replay = self.submit(model, content, "event-turn")

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

    def test_a_description_in_the_models_own_words_is_kept(self) -> None:
        content = "明天下午三点在 A301 和张伟开需求评审，讨论驱动排期"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action(
                        "create_event",
                        content="需求评审",
                        description="与张伟在 A301 讨论驱动排期",
                        date="2026-09-10",
                        start_time="15:00",
                    ),
                }
            ]
        )

        self.submit(model, content, "paraphrase-turn")

        event = self.store.list_calendar_events()[0]
        self.assertEqual(event.description, "与张伟在 A301 讨论驱动排期")
        self.assertEqual(event.source_quote, content)

    def test_bare_future_weekday_creates_an_event_with_its_resolved_date(self) -> None:
        content = "周五下午两点有架构评审。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action("create_event", content="架构评审", date="2026-09-11", start_time="14:00"),
                }
            ]
        )

        turn = self.submit(model, content, "bare-weekday-turn", date(2026, 9, 10))

        event = self.store.list_calendar_events()[0]
        self.assertEqual(turn.outcome, "calendar_create")
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-11T14:00:00-07:00"))
        self.assertIn("2026-09-11T14:00:00-07:00 (America/Vancouver)", turn.assistant_message.content)

    def test_an_end_time_and_a_duration_that_agree_create_the_event(self) -> None:
        for label, start, end, minutes, expected_end in (
            ("same-day", "16:00", "17:00", 60, "2026-09-10T17:00:00-07:00"),
            ("past-midnight", "23:30", "00:30", 60, "2026-09-11T00:30:00-07:00"),
        ):
            with self.subTest(label=label):
                content = f"明天{start}到{end}有代码审查会议，开{minutes}分钟。"
                model = ScriptedContextualModel(
                    [
                        {
                            "intent": "calendar_create",
                            "action": action(
                                "create_event",
                                content=f"代码审查会议 {label}",
                                date="2026-09-10",
                                start_time=start,
                                end_time=end,
                                duration_minutes=minutes,
                            ),
                        }
                    ]
                )

                turn = self.submit(model, content, f"end-and-duration-{label}")

                self.assertEqual(turn.outcome, "calendar_create")
                event = self.store.get_calendar_event(turn.operations[0].result["event_id"])
                self.assertEqual(event.end_at, datetime.fromisoformat(expected_end))

    def test_an_end_time_and_a_duration_that_disagree_ask_for_the_length(self) -> None:
        content = "明天下午四点到五点有代码审查会议，开一个半小时。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action(
                        "create_event",
                        content="代码审查会议",
                        date="2026-09-10",
                        start_time="16:00",
                        end_time="17:00",
                        duration_minutes=90,
                    ),
                }
            ]
        )

        turn = self.submit(model, content, "end-duration-conflict")

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertIn("How long", turn.assistant_message.content)

    def test_the_current_local_time_reaches_the_model(self) -> None:
        content = "半小时后开个短会。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action("create_event", content="短会", date="2026-09-12", start_time="14:40"),
                }
            ]
        )

        self.submit(model, content, "relative-now-turn", date(2026, 9, 12), reference_time=time(14, 10, 37))

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
                    "action": action(
                        "create_event", content="季度评审", date="2026-11-02", start_time="15:00", end_time="16:00"
                    ),
                }
            ]
        )

        self.submit(model, content, "after-dst-turn", date(2026, 9, 12))
        event = self.store.list_calendar_events()[0]

        self.assertEqual(event.start_at, datetime.fromisoformat("2026-11-02T15:00:00-08:00"))
        self.assertEqual(event.end_at, datetime.fromisoformat("2026-11-02T16:00:00-08:00"))

    def test_calendar_end_time_can_cross_midnight(self) -> None:
        content = "明天 23:30 到 00:30 有发布检查。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action(
                        "create_event", content="发布检查", date="2026-09-11", start_time="23:30", end_time="00:30"
                    ),
                }
            ]
        )

        turn = self.submit(model, content, "cross-midnight", date(2026, 9, 10))

        event = self.store.get_calendar_event(turn.operations[0].result["event_id"])
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-11T23:30:00-07:00"))
        self.assertEqual(event.end_at, datetime.fromisoformat("2026-09-12T00:30:00-07:00"))

    def test_a_repeating_event_says_repeats_are_not_stored(self) -> None:
        content = "每周一下午三点有团队例会"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action("create_event", content="团队例会", recurring=True, start_time="15:00"),
                }
            ]
        )

        turn = self.submit(model, content, "recurring-turn", date(2026, 9, 11))

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertIn("cannot store a repeating schedule", turn.assistant_message.content)

    def test_a_stated_foreign_timezone_asks_instead_of_storing_local_time(self) -> None:
        content = "纽约时间明天上午九点有电话会"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action(
                        "create_event",
                        content="电话会",
                        date="2026-09-12",
                        timezone="America/New_York",
                        start_time="09:00",
                    ),
                }
            ]
        )

        turn = self.submit(model, content, "foreign-timezone-turn", date(2026, 9, 11))

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertIn("America/Vancouver", turn.assistant_message.content)

    def test_the_local_timezone_by_name_still_creates_the_event(self) -> None:
        content = "温哥华时间明天上午九点有电话会"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action(
                        "create_event",
                        content="电话会",
                        date="2026-09-12",
                        timezone="America/Vancouver",
                        start_time="09:00",
                    ),
                }
            ]
        )

        turn = self.submit(model, content, "local-timezone-turn", date(2026, 9, 11))

        self.assertEqual(turn.outcome, "calendar_create")
        self.assertEqual(
            self.store.list_calendar_events()[0].start_at, datetime.fromisoformat("2026-09-12T09:00:00-07:00")
        )

    # A question is a plain reply, and the next turn continues from the transcript

    def test_a_question_then_its_answer_creates_the_event_with_both_messages_as_sources(self) -> None:
        original = "与 Daohui 碰头，讨论 UR5 交接和 Vention 工装板尺寸。"
        answer = "会议时间是 2026-09-11 下午两点。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action(
                        "create_event", "initial-event-turn", content="与 Daohui 碰头", description="讨论 UR5 交接"
                    ),
                },
                {
                    "intent": "calendar_create",
                    "action": action(
                        "create_event",
                        "initial-event-turn",
                        "date-answer-turn",
                        content="与 Daohui 碰头",
                        description="讨论 UR5 交接",
                        date="2026-09-11",
                        start_time="14:00",
                    ),
                },
            ]
        )

        question = self.submit(model, original, "initial-event-turn", date(2026, 9, 10))

        self.assertEqual(question.outcome, "clarification_created")
        self.assertEqual(question.assistant_message.content, "Which exact date should I use for this local event?")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(self.store.list_tool_operations("initial-event-turn"), [])

        created = self.submit(model, answer, "date-answer-turn", date(2026, 9, 10))

        event = self.store.list_calendar_events()[0]
        self.assertEqual(created.outcome, "calendar_create")
        self.assertIn(
            question.assistant_message.content,
            [message.get("content") for message in model.calls[1]["recent_messages"]],
        )
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-11T14:00:00-07:00"))
        expected_sources = [
            EvidenceReference(message_id="initial-event-turn", source_quote=original),
            EvidenceReference(message_id="date-answer-turn", source_quote=answer),
        ]
        self.assertEqual(self.store.list_calendar_event_sources(event.id), expected_sources)
        self.assertEqual(
            self.store.list_tool_operations("date-answer-turn")[0].request["evidence"],
            [source.model_dump(mode="json") for source in expected_sources],
        )

    def test_a_source_id_python_cannot_verify_is_dropped_not_rejected(self) -> None:
        earlier = self.store.save_message(
            Message(id="earlier-user", conversation_id=self.conversation.id, role="user", content="明天有评审会")
        )
        reply = self.store.save_message(
            Message(
                id="earlier-reply",
                conversation_id=self.conversation.id,
                role="assistant",
                content="几点开始？",
                reply_to_message_id=earlier.id,
            )
        )
        content = "下午三点"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action(
                        "create_event",
                        reply.id,
                        "invented-message",
                        earlier.id,
                        content="评审会",
                        date="2026-09-10",
                        start_time="15:00",
                    ),
                }
            ]
        )

        turn = self.submit(model, content, "time-answer")

        event = self.store.list_calendar_events()[0]
        self.assertEqual(turn.outcome, "calendar_create")
        self.assertEqual(
            [source.message_id for source in self.store.list_calendar_event_sources(event.id)],
            [earlier.id, "time-answer"],
        )

    def test_the_models_own_question_is_the_reply_and_nothing_is_stored(self) -> None:
        event = self._saved_event(
            "which-thursday", "算法优化与代码审查会议", datetime.fromisoformat("2026-09-10T14:00:00-07:00")
        )
        model = ScriptedContextualModel(
            [{"intent": "clarify", "reply": "你是指 9月10日（明天）还是 9月17日的周四？", "action": None}]
        )

        turn = self.submit(model, "这个会议改到周四。", "which-thursday-turn")

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(turn.assistant_message.content, "你是指 9月10日（明天）还是 9月17日的周四？")
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_event_update_operations("which-thursday-turn"), [])

    def test_a_clarification_without_a_question_is_refused(self) -> None:
        model = ScriptedContextualModel([{"intent": "clarify", "reply": None}])

        turn = self.submit(model, "把那个会删掉", "empty-question-turn")

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertIn("reply", turn.failure_reason)

    def test_retired_clarification_fields_are_ignored(self) -> None:
        model = ScriptedContextualModel(
            [
                {
                    "intent": "clarify",
                    "reply": "Which meeting do you mean?",
                    "clarification_question": "Which meeting do you mean?",
                    "pending_action": {"action": "delete_event", "evidence": []},
                    "evidence": [{"message_id": "retired-fields-turn", "source_quote": "删掉"}],
                }
            ]
        )

        turn = self.submit(model, "删掉", "retired-fields-turn")

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(turn.assistant_message.content, "Which meeting do you mean?")

    # Finding a target is the model's judgement

    def test_a_target_not_found_is_asked_about_then_completed_with_the_named_event(self) -> None:
        event = self._saved_event(
            "review-meeting", "算法优化与代码审查会议", datetime.fromisoformat("2026-09-10T14:00:00-07:00")
        )
        fresh = self._fresh_conversation()
        vague = "这个会议改到周四。"
        named = "算法优化与代码审查会议，改到 9 月 17 日。"
        model = ScriptedContextualModel(
            [
                {"intent": "clarify", "reply": "你指的是哪个会议？"},
                {
                    "intent": "calendar_update",
                    "action": action(
                        "update_event", "fresh-vague", "fresh-named", date="2026-09-17", target_event_id=event.id
                    ),
                },
            ]
        )

        question = self.submit(model, vague, "fresh-vague", conversation_id=fresh.id)

        self.assertEqual(question.outcome, "clarification_created")
        self.assertEqual(self.store.get_calendar_event(event.id), event)

        updated = self.submit(model, named, "fresh-named", conversation_id=fresh.id)

        self.assertEqual(updated.outcome, "calendar_update")
        self.assertEqual(
            self.store.get_calendar_event(event.id).start_at, datetime.fromisoformat("2026-09-17T14:00:00-07:00")
        )
        operation = self.store.list_calendar_event_update_operations("fresh-named")[0]
        self.assertEqual(
            [source.message_id for source in self.store.list_calendar_event_update_sources(operation.id)],
            ["fresh-vague", "fresh-named"],
        )

    def test_a_target_the_model_found_in_a_new_conversation_is_not_second_guessed(self) -> None:
        event = self._saved_event(
            "named-meeting", "算法优化与代码审查会议", datetime.fromisoformat("2026-09-10T14:00:00-07:00")
        )
        fresh = self._fresh_conversation()
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": action("update_event", date="2026-09-10", start_time="15:00", target_event_id=event.id),
                }
            ]
        )

        turn = self.submit(model, "把算法优化与代码审查会议改到周四下午三点。", "fresh-named", conversation_id=fresh.id)

        self.assertEqual(turn.outcome, "calendar_update")
        self.assertEqual(
            self.store.get_calendar_event(event.id).start_at, datetime.fromisoformat("2026-09-10T15:00:00-07:00")
        )

    def test_a_target_outside_the_candidates_changes_nothing(self) -> None:
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": action("update_event", start_time="14:00", target_event_id="invented-event"),
                }
            ]
        )

        turn = self.submit(model, "把会议改到下午两点。", "invalid-target-turn")

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertEqual(turn.failure_reason, "ValueError: calendar update target is outside supplied candidates")
        self.assertEqual(self.store.list_calendar_events(), [])

    # Calendar updates

    def test_event_correction_updates_the_event_and_records_both_sources(self) -> None:
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
                    "action": action(
                        "update_event",
                        source.id,
                        "correction-turn",
                        date="2026-09-10",
                        start_time="14:00",
                        end_time="15:00",
                        target_event_id=event.id,
                    ),
                }
            ]
        )

        turn = self.submit(model, correction, "correction-turn")

        updated = self.store.get_calendar_event(event.id)
        operation = self.store.list_calendar_event_update_operations("correction-turn")[0]
        self.assertEqual(turn.outcome, "calendar_update")
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(updated.start_at, datetime.fromisoformat("2026-09-10T14:00:00-07:00"))
        self.assertEqual(updated.end_at, datetime.fromisoformat("2026-09-10T15:00:00-07:00"))
        self.assertEqual(
            self.store.list_calendar_event_update_sources(operation.id),
            [
                EvidenceReference(message_id=source.id, source_quote=source.content),
                EvidenceReference(message_id="correction-turn", source_quote=correction),
            ],
        )
        self.assertIn(event.id, model.calls[0]["candidate_events"][0])

    def test_calendar_update_applies_a_new_start_and_duration(self) -> None:
        original = "今天开始进行A测，下午三点钟要准备好，四点有个相关的说明会"
        correction = "会议延期了，时间也被缩短到半个小时，现在晚上七点才开始"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_create",
                    "action": action("create_event", content="相关的说明会", date="2026-09-10", start_time="16:00"),
                },
                {
                    "intent": "calendar_update",
                    "action": action(
                        "update_event", start_time="19:00", duration_minutes=30, target_event_id="replaced"
                    ),
                },
            ]
        )
        service = ConversationRouterService(self.store, model, date(2026, 9, 10))

        created = service.submit(self.conversation.id, original, "phase-10-3-create")
        event = self.store.get_calendar_event(created.operations[0].result["event_id"])
        model.decisions[0]["action"]["target_event_id"] = event.id

        updated = service.submit(self.conversation.id, correction, "phase-10-3-update")
        replay = service.submit(self.conversation.id, correction, "phase-10-3-update")
        operation = self.store.list_calendar_event_update_operations("phase-10-3-update")[0]
        event = self.store.get_calendar_event(event.id)

        self.assertEqual(updated.outcome, "calendar_update")
        self.assertEqual(replay.model_calls, 0)
        self.assertEqual(len(self.store.list_calendar_events()), 1)
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-10T19:00:00-07:00"))
        self.assertEqual(event.end_at, datetime.fromisoformat("2026-09-10T19:30:00-07:00"))
        self.assertEqual(self.store.undo_calendar_event_update(operation.id).status, "undone")
        restored = self.store.get_calendar_event(event.id)
        self.assertEqual(restored.start_at, datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        self.assertIsNone(restored.end_at)

    def test_an_update_that_changes_nothing_writes_nothing(self) -> None:
        event = self._saved_event("already-thursday", "架构评审", datetime.fromisoformat("2026-09-10T14:00:00-07:00"))
        model = ScriptedContextualModel(
            [{"intent": "calendar_update", "action": action("update_event", date="2026-09-10", target_event_id=event.id)}]
        )

        turn = self.submit(model, "架构评审改到周四。", "no-op-update")

        self.assertEqual(turn.outcome, "calendar_unchanged")
        self.assertEqual(turn.operations, ())
        self.assertEqual(self.store.list_calendar_event_update_operations("no-op-update"), [])
        self.assertIn("nothing was changed", turn.assistant_message.content)

    def test_an_update_reply_states_the_new_schedule(self) -> None:
        event = self._saved_event("echo-meeting", "架构评审", datetime.fromisoformat("2026-09-10T14:00:00-07:00"))
        model = ScriptedContextualModel(
            [
                {
                    "intent": "calendar_update",
                    "action": action("update_event", start_time="16:00", end_time="17:00", target_event_id=event.id),
                }
            ]
        )

        turn = self.submit(model, "架构评审改到下午四点到五点。", "echo-update")

        self.assertEqual(turn.outcome, "calendar_update")
        self.assertIn("2026-09-10T16:00:00-07:00", turn.assistant_message.content)
        self.assertIn("2026-09-10T17:00:00-07:00", turn.assistant_message.content)

    def test_an_update_without_a_new_value_asks_for_the_date(self) -> None:
        event = self._saved_event("undecided", "工作流会议", datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        model = ScriptedContextualModel(
            [{"intent": "calendar_update", "action": action("update_event", target_event_id=event.id, date=None)}]
        )

        turn = self.submit(model, "这个会议改到周四。", "undecided-update-turn")

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_event_update_operations("undecided-update-turn"), [])

    def test_an_implausible_update_date_asks_for_the_date(self) -> None:
        event = self._saved_event("implausible", "工作流会议", datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        model = ScriptedContextualModel(
            [{"intent": "calendar_update", "action": action("update_event", date="2030-09-10", target_event_id=event.id)}]
        )

        turn = self.submit(model, "这个会议改到周三。", "implausible-update-turn")

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertIn("Which exact date", turn.assistant_message.content)
        self.assertEqual(self.store.get_calendar_event(event.id), event)

    def test_a_malformed_normalized_time_is_refused_without_a_write(self) -> None:
        event = self._saved_event("unsupported", "工作流会议", datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        model = ScriptedContextualModel(
            [{"intent": "calendar_update", "action": action("update_event", start_time="7 PM", target_event_id=event.id)}]
        )

        turn = self.submit(model, "这个会议改到晚上七点才开始，持续一会儿。", "unsupported-correction", date(2026, 9, 10))

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_event_update_operations("unsupported-correction"), [])

    def test_a_new_length_for_an_all_day_event_asks_for_a_start_time(self) -> None:
        event = self._saved_event("all-day", "工作流会议", date(2026, 9, 10))
        model = ScriptedContextualModel(
            [{"intent": "calendar_update", "action": action("update_event", duration_minutes=30, target_event_id=event.id)}]
        )

        turn = self.submit(model, "这个会议开半个小时。", "duration-correction", date(2026, 9, 10))

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(turn.assistant_message.content, "What exact start time should I use for this local event?")
        self.assertEqual(self.store.get_calendar_event(event.id), event)

    def test_prepared_duration_update_resumes_after_reopening_the_database(self) -> None:
        event = self._saved_event("prepared-update", "工作流会议", datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
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
            operation, [EvidenceReference(message_id=correction.id, source_quote=correction.content)]
        )
        restarted = CaptureStore(self.store.database_path)
        restarted.initialize()
        model = ScriptedContextualModel([])

        resumed = ConversationRouterService(restarted, model, date(2026, 9, 10)).submit(
            self.conversation.id, correction.content, correction.id
        )

        self.assertEqual(resumed.outcome, "resumed")
        self.assertEqual(model.calls, [])
        self.assertEqual(restarted.get_calendar_event(event.id), updated)
        self.assertEqual(restarted.get_calendar_event_update_operation(operation.id).status, "committed")

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

        resumed = self.submit(model, current.content, current.id, date(2026, 9, 10))

        event = self.store.list_calendar_events()[0]
        self.assertEqual(resumed.outcome, "resumed")
        self.assertEqual(self.store.list_calendar_event_sources(event.id), sources)
        self.assertEqual(model.calls, [])

    # Calendar deletion waits for the user

    def test_a_deletion_waits_for_confirmation_then_can_be_undone(self) -> None:
        event = self._saved_event(
            "expired-meeting", "相关的说明会", datetime.fromisoformat("2026-09-10T16:00:00-07:00")
        )
        content = "昨天的会议已结束，删除过期会议"
        model = ScriptedContextualModel(
            [{"intent": "calendar_delete", "action": action("delete_event", target_event_id=event.id)}]
        )

        turn = self.submit(model, content, "delete-turn", date(2026, 9, 11))
        replay = self.submit(model, content, "delete-turn", date(2026, 9, 11))
        operation = self.store.list_calendar_event_update_operations("delete-turn")[0]

        self.assertEqual(turn.outcome, "calendar_delete_review")
        self.assertEqual(turn.operations[0].status, "review")
        self.assertEqual(replay.model_calls, 0)
        self.assertIn("相关的说明会", turn.assistant_message.content)
        self.assertIn("2026-09-10T16:00:00-07:00", turn.assistant_message.content)
        self.assertEqual(self.store.get_calendar_event(event.id).status, "active")
        self.assertEqual(
            self.store.list_calendar_event_update_sources(operation.id),
            [EvidenceReference(message_id="delete-turn", source_quote=content)],
        )

        confirmed, deleted = self.store.approve_calendar_event_update(operation.id)

        self.assertEqual(confirmed.status, "committed")
        self.assertEqual(deleted.status, "deleted")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(self.store.undo_calendar_event_update(operation.id).status, "undone")
        self.assertEqual(self.store.get_calendar_event(event.id), event)

    def test_keeping_an_event_rejects_its_deletion(self) -> None:
        event = self._saved_event("kept", "架构评审", datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        model = ScriptedContextualModel(
            [{"intent": "calendar_delete", "action": action("delete_event", target_event_id=event.id)}]
        )

        turn = self.submit(model, "删掉架构评审", "keep-turn")
        rejected = self.store.reject_calendar_event_update(turn.operations[0].id)

        self.assertEqual(rejected.status, "failed")
        self.assertEqual(rejected.error, "rejected by user")
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        with self.assertRaises(ValueError):
            self.store.approve_calendar_event_update(rejected.id)

    def test_a_resubmitted_deletion_in_review_is_not_committed(self) -> None:
        event = self._saved_event("waiting", "架构评审", datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        user = self.store.save_message(
            Message(id="waiting-delete", conversation_id=self.conversation.id, role="user", content="删掉架构评审")
        )
        self.store.prepare_calendar_event_update(
            CalendarEventUpdateOperation(
                id="waiting-delete-operation",
                source_message_id=user.id,
                action_index=0,
                event_id=event.id,
                before=event,
                after=event.model_copy(update={"status": "deleted"}),
                status="review",
            ),
            [EvidenceReference(message_id=user.id, source_quote=user.content)],
        )
        model = ScriptedContextualModel([])

        turn = self.submit(model, user.content, user.id)

        self.assertEqual(model.calls, [])
        self.assertEqual(turn.operations[0].status, "review")
        self.assertEqual(self.store.get_calendar_event(event.id).status, "active")

    def test_the_model_asking_which_event_to_delete_writes_nothing(self) -> None:
        self._saved_event("morning-standup", "站会", datetime.fromisoformat("2026-09-10T09:00:00-07:00"))
        self._saved_event("afternoon-briefing", "说明会", datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        model = ScriptedContextualModel([{"intent": "clarify", "reply": "Which local calendar event should I delete?"}])

        turn = self.submit(model, "删除过期会议", "ambiguous-delete-turn", date(2026, 9, 11))

        self.assertEqual(turn.outcome, "clarification_created")
        self.assertEqual(len(self.store.list_calendar_events()), 2)
        self.assertEqual(self.store.list_calendar_event_update_operations("ambiguous-delete-turn"), [])
        self.assertEqual(turn.assistant_message.content, "Which local calendar event should I delete?")

    def test_calendar_delete_outside_supplied_candidates_changes_nothing(self) -> None:
        event = self._saved_event("kept-meeting", "架构评审", datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        model = ScriptedContextualModel(
            [{"intent": "calendar_delete", "action": action("delete_event", target_event_id="invented-event")}]
        )

        turn = self.submit(model, "删除那个会议", "invalid-delete-turn", date(2026, 9, 11))

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertEqual(self.store.list_calendar_events(), [event])
        self.assertEqual(self.store.list_calendar_event_update_operations("invalid-delete-turn"), [])

    # What the router sees

    def test_stale_events_do_not_hide_the_event_a_turn_is_about(self) -> None:
        for suffix, title, start_at in (
            ("stale-1", "上月周会", "2026-08-03T10:00:00-07:00"),
            ("stale-2", "上月评审", "2026-08-10T10:00:00-07:00"),
            ("stale-3", "上月复盘", "2026-08-17T10:00:00-07:00"),
        ):
            self._saved_event(suffix, title, datetime.fromisoformat(start_at))
        target = self._saved_event("call-in", "客户电话会", datetime.fromisoformat("2026-09-12T09:00:00-07:00"))
        model = ScriptedContextualModel(
            [{"intent": "calendar_delete", "action": action("delete_event", target_event_id=target.id)}]
        )

        turn = self.submit(model, "明天的客户电话会取消了，删掉它", "stale-crowding-turn", date(2026, 9, 11))

        lines = model.calls[0]["candidate_events"]
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(target.id))
        self.assertEqual(turn.outcome, "calendar_delete_review")

    def test_an_event_beyond_the_bound_is_found_by_its_name(self) -> None:
        for offset in range(17):
            self._saved_event(
                f"filler-{offset}", f"日常同步 {offset}", datetime(2026, 9, 1 + offset, 9, tzinfo=timezone.utc)
            )
        far = self._saved_event("roadmap", "季度路线图评审", datetime.fromisoformat("2026-11-20T10:00:00-08:00"))
        model = ScriptedContextualModel([{"intent": "clarify", "reply": "改到几点？"}])

        self.submit(model, "季度路线图评审要改时间", "named-far-event")

        self.assertIn("events", model.calls[0]["truncated_candidate_lists"])
        self.assertTrue(any(line.startswith(far.id) for line in model.calls[0]["candidate_events"]))

    def test_an_older_task_is_reachable_once_a_project_has_many(self) -> None:
        for index, content in enumerate(["完成驱动调试", "完成代码审查", "整理会议纪要", "更新部署文档", "修复登录页样式"]):
            self._project_record(
                f"many-tasks-{index}",
                "task",
                "active",
                content,
                updated_at=datetime(2026, 9, 1 + index, tzinfo=timezone.utc),
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

        turn = self.submit(model, content, "complete-oldest-task", date(2026, 9, 12))

        self.assertTrue(any(line.startswith("many-tasks-0 |") for line in model.calls[0]["candidate_project_records"]))
        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(self.store.get_project_record("many-tasks-0").state, "completed")

    def test_a_chinese_credential_disclosure_never_reaches_the_model(self) -> None:
        secret = "我的密钥是 sk-test-1234567890abcdef"
        blocked = self.submit(ScriptedContextualModel([]), secret, "credential-turn", date(2026, 9, 11))
        model = ScriptedContextualModel([{"intent": "reply_only", "reply": "好的。"}])

        self.submit(model, "你好", "after-credential-turn", date(2026, 9, 11))

        self.assertEqual(blocked.outcome, "blocked_credential")
        self.assertEqual(blocked.model_calls, 0)
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertNotIn(
            "credential-turn", [message["message_id"] for message in model.calls[0]["recent_messages"]]
        )
        self.assertNotIn("sk-test", str(model.calls[0]))

    # Ideas, notes, and a second item

    def test_finished_meeting_is_kept_as_a_note_without_a_calendar_event(self) -> None:
        content = "今天中午十二点的会议是关于系统架构与驱动集成，主要分析了ALOHA 遥操作系统的驱动调试。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "save_note",
                    "action": action("save_note", content="系统架构与驱动集成会议：主要分析了ALOHA 遥操作系统的驱动调试。"),
                }
            ]
        )

        turn = self.submit(model, content, "minutes-turn", date(2026, 9, 11))
        items = self.store.list_saved_items(kinds=("note",))

        self.assertEqual(turn.outcome, "save_note")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_message_id, "minutes-turn")
        self.assertEqual(items[0].source_quote, content)

    def test_a_second_item_is_asked_about_and_the_answer_completes_it(self) -> None:
        content = (
            "今天中午十二点的会议是关于系统架构与驱动集成，主要分析了ALOHA 遥操作系统的驱动调试。"
            "修复需要下周三晚上八点之前提交"
        )
        confirmation = "是的，请建下周三的日历。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "save_note",
                    "action": action("save_note", content="系统架构与驱动集成会议：主要分析了驱动调试。"),
                    "follow_up_question": "Should I also add a local calendar event for 下周三晚上八点?",
                },
                {
                    "intent": "calendar_create",
                    "action": action(
                        "create_event",
                        "compound-turn",
                        "compound-confirmation",
                        content="提交修复",
                        date="2026-09-16",
                        start_time="20:00",
                    ),
                },
            ]
        )

        first = self.submit(model, content, "compound-turn", date(2026, 9, 11))

        self.assertEqual(first.outcome, "save_note")
        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertTrue(
            first.assistant_message.content.endswith("Should I also add a local calendar event for 下周三晚上八点?")
        )

        second = self.submit(model, confirmation, "compound-confirmation", date(2026, 9, 11))
        event = self.store.list_calendar_events()[0]

        self.assertEqual(second.outcome, "calendar_create")
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-16T20:00:00-07:00"))
        self.assertEqual(
            [source.message_id for source in self.store.list_calendar_event_sources(event.id)],
            ["compound-turn", "compound-confirmation"],
        )

    def test_a_follow_up_question_without_a_write_is_ignored(self) -> None:
        model = ScriptedContextualModel(
            [
                {
                    "intent": "reply_only",
                    "reply": "I cannot take a local action for that.",
                    "follow_up_question": "Should I also add that deadline?",
                }
            ]
        )

        turn = self.submit(model, "这条消息只是闲聊。", "reply-only-turn", date(2026, 9, 11))

        self.assertEqual(turn.outcome, "reply_only")
        self.assertEqual(turn.assistant_message.content, "I cannot take a local action for that.")

    # Tracker changes

    def test_tracker_change_uses_current_decision_without_another_model_call(self) -> None:
        content = "排查 ROS 2 发布者-订阅者节点之间的延迟问题。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [{"action": "create_record", "kind": "task", "content": content, "source_quote": content}],
                }
            ]
        )

        turn = self.submit(model, content, "tracker-turn")
        record = self.store.list_project_records(self.project.id)[0]

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(record.content, content)
        self.assertEqual(record.source_message_id, "tracker-turn")

    def test_a_task_saved_a_turn_later_quotes_the_pasted_minutes(self) -> None:
        minutes = "会议纪要：一、完成驱动调试。二、下周前整理部署文档。三、确认 UR5 交接时间。"
        self.store.save_message(
            Message(id="pasted-minutes", conversation_id=self.conversation.id, role="user", content=minutes)
        )
        content = "继续记剩下的"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "整理部署文档",
                            "source_quote": "下周前整理部署文档",
                            "source_message_id": "pasted-minutes",
                        },
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "确认 UR5 交接时间",
                            "source_quote": "确认 UR5 交接的时间",
                            "source_message_id": "pasted-minutes",
                        },
                    ],
                }
            ]
        )

        turn = self.submit(model, content, "continue-turn")
        records = {record.content: record for record in self.store.list_project_records(self.project.id)}

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(records["整理部署文档"].source_message_id, "pasted-minutes")
        self.assertEqual(records["整理部署文档"].source_quote, "下周前整理部署文档")
        # A quote that is not exact falls back to the whole message instead of failing.
        self.assertEqual(records["确认 UR5 交接时间"].source_quote, minutes)
        operation = self.store.list_project_change_operations("continue-turn")[0]
        self.assertEqual(self.store.undo_project_change_operation(operation.id).status, "undone")

    def test_a_tracker_quote_from_a_message_outside_the_context_uses_the_current_message(self) -> None:
        content = "新增任务：整理部署文档"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {
                            "action": "create_record",
                            "kind": "task",
                            "content": "整理部署文档",
                            "source_quote": "整理部署文档",
                            "source_message_id": "some-other-conversation-message",
                        }
                    ],
                }
            ]
        )

        self.submit(model, content, "unknown-source-turn")
        record = self.store.list_project_records(self.project.id)[0]

        self.assertEqual(record.source_message_id, "unknown-source-turn")
        self.assertEqual(record.source_quote, "整理部署文档")

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

        turn = self.submit(model, content, "deadline-record-turn", date(2026, 9, 11))
        record = self.store.list_project_records(self.project.id)[0]

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(record.due_at, datetime.fromisoformat("2026-09-16T20:00:00-07:00"))
        self.assertEqual(self.store.list_task_deadlines(), [record])
        self.assertIn("2026-09-16T20:00:00-07:00", turn.assistant_message.content)

    def test_a_task_deadline_without_a_time_stays_a_calendar_date(self) -> None:
        content = "文档下周三之前交。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [
                        {"action": "create_record", "kind": "task", "content": "交文档", "due_date": "2026-09-16"}
                    ],
                }
            ]
        )

        self.submit(model, content, "date-only-deadline-turn", date(2026, 9, 11))
        record = self.store.list_project_records(self.project.id)[0]

        self.assertEqual(record.due_at, date(2026, 9, 16))
        self.assertEqual(record.source_quote, content)

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

        turn = self.submit(model, content, "vague-deadline-turn", date(2026, 9, 11))
        record = self.store.list_project_records(self.project.id)[0]

        self.assertIsNone(record.due_at)
        self.assertIn("outside the planning horizon", turn.assistant_message.content)
        self.assertIn("no deadline yet", turn.assistant_message.content)

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

        turn = self.submit(model, content, "recurring-deadline-turn", date(2026, 9, 11))

        self.assertIsNone(self.store.list_project_records(self.project.id)[0].due_at)
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

        turn = self.submit(model, content, "model-deadline-turn", date(2026, 9, 11))

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertEqual(self.store.list_project_records(self.project.id), [])

    def test_a_follow_up_sets_a_deadline_on_an_existing_task(self) -> None:
        content = "提醒我明天十点交材料。"
        answer = "上午十点"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [{"action": "create_record", "kind": "task", "content": "交材料", "source_quote": content}],
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

        service.submit(self.conversation.id, content, "bare-hour-turn")
        record = self.store.list_project_records(self.project.id)[0]
        model.decisions[0]["project_changes"][0]["target_record_id"] = record.id
        second = service.submit(self.conversation.id, answer, "bare-hour-answer")
        updated = self.store.get_project_record(record.id)
        operation = self.store.list_project_change_operations("bare-hour-answer")[0]

        self.assertEqual(second.outcome, "tracker_change")
        self.assertEqual(updated.due_at, datetime.fromisoformat("2026-09-12T10:00:00-07:00"))
        self.assertEqual(self.store.undo_project_change_operation(operation.id).status, "undone")
        self.assertIsNone(self.store.get_project_record(record.id).due_at)

    # Answers

    def test_local_answer_renders_the_items_the_model_named(self) -> None:
        task = self._project_record(
            "ros-delay-task",
            "task",
            "active",
            "排查 ROS 2 节点延迟。",
            due_at=datetime.fromisoformat("2026-09-12T18:00:00-07:00"),
        )
        model = ScriptedContextualModel([{"intent": "local_answer", "reply": "周五截止。", "source_ids": [task.id]}])

        turn = self.submit(model, "排查延迟那个任务什么时候截止？", "task-deadline-question")

        self.assertEqual(turn.outcome, "grounded_local_context")
        self.assertIn("2026-09-12T18:00:00-07:00", turn.assistant_message.content)
        self.assertNotIn("周五截止", turn.assistant_message.content)
        self.assertEqual(
            [record.id for record in self.store.list_project_answer_sources(turn.assistant_message.id)], [task.id]
        )

    def test_local_answer_without_named_items_falls_back_to_the_lookup(self) -> None:
        event = self._saved_event(
            "review-meeting",
            "算法优化与代码审查会议",
            datetime.fromisoformat("2026-09-10T14:00:00-07:00"),
            end_at=datetime.fromisoformat("2026-09-10T15:00:00-07:00"),
        )
        model = ScriptedContextualModel([{"intent": "local_answer", "source_ids": []}])

        turn = self.submit(model, "我明天有什么安排？", "fallback-lookup-question")

        self.assertEqual(turn.outcome, "grounded_local_context")
        self.assertIn(event.start_at.isoformat(), turn.assistant_message.content)
        self.assertEqual(
            [source.reference.source_id for source in self.store.list_local_context_answer_sources(turn.assistant_message.id)],
            [event.id],
        )

    def test_project_answer_is_written_from_the_project_snapshot(self) -> None:
        task = self._project_record(
            "ros-delay-task",
            "task",
            "active",
            "排查 ROS 2 节点延迟。",
            due_at=datetime.fromisoformat("2026-09-12T18:00:00-07:00"),
        )
        self._project_record("setup-task", "task", "completed", "搭建仿真环境。")
        old = self._project_record("storage-sqlite", "decision", "active", "使用 SQLite 存储。")
        new = self._project_record(
            "storage-postgres",
            "decision",
            "active",
            "改用 PostgreSQL。",
            rationale="需要多人协作。",
            supersedes_record_id=old.id,
        )
        self.store.transition_project_record(old.id, "superseded")
        model = ScriptedContextualModel(
            [{"intent": "project_answer"}],
            [
                {
                    "reply": "还剩排查节点延迟一项；存储已从 SQLite 换成 PostgreSQL，因为需要多人协作。",
                    "source_ids": [task.id, new.id, task.id],
                }
            ],
        )

        turn = self.submit(model, "项目现在进展如何，为什么换了存储？", "project-progress-question")

        self.assertEqual(turn.outcome, "project_answer")
        self.assertEqual(turn.model_calls, 2)
        snapshot = model.project_answer_calls[0]["project_snapshot"]
        self.assertEqual({record["id"] for record in snapshot["open_records"]}, {new.id, task.id})
        self.assertEqual([record["id"] for record in snapshot["closed_records"]], ["setup-task"])
        self.assertEqual(snapshot["superseded_records"][0]["superseded_by_record_id"], new.id)
        self.assertIn("Stored dates:", turn.assistant_message.content)
        self.assertIn("2026-09-12T18:00:00-07:00", turn.assistant_message.content)
        self.assertEqual(
            [record.id for record in self.store.list_project_answer_sources(turn.assistant_message.id)],
            [task.id, new.id],
        )

    def test_a_project_answer_citing_outside_the_snapshot_changes_nothing(self) -> None:
        self._project_record("ros-delay-task", "task", "active", "排查 ROS 2 节点延迟。")
        model = ScriptedContextualModel(
            [{"intent": "project_answer"}],
            [{"reply": "当前有三项任务。", "source_ids": ["invented-record"]}],
        )

        turn = self.submit(model, "项目还有什么没做？", "invented-source-question")

        self.assertEqual(turn.outcome, "model_or_validation_failure")
        self.assertEqual(turn.model_calls, 2)
        self.assertIn("outside the project snapshot", turn.failure_reason)

    def test_a_project_question_without_a_project_makes_no_second_call(self) -> None:
        unassigned = self.store.save_conversation(Conversation(id="unassigned-chat", title="Loose"))
        model = ScriptedContextualModel([{"intent": "project_answer"}])

        turn = self.submit(model, "项目进展如何？", "unassigned-project-question", conversation_id=unassigned.id)

        self.assertEqual(turn.outcome, "project_answer_unavailable")
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(model.project_answer_calls, [])

    # Failures around a decision

    def test_a_write_the_local_rules_refuse_is_a_dispatch_failure(self) -> None:
        unassigned = self.store.save_conversation(Conversation(id="unassigned-tracker-chat", title="Loose"))
        content = "排查 ROS 2 节点延迟。"
        model = ScriptedContextualModel(
            [
                {
                    "intent": "tracker_change",
                    "project_changes": [{"action": "create_record", "kind": "task", "content": content, "source_quote": content}],
                }
            ]
        )

        turn = self.submit(model, content, "unassigned-tracker-turn", conversation_id=unassigned.id)

        self.assertEqual(turn.outcome, "write_or_dispatch_failure")
        self.assertEqual(turn.model_calls, 1)
        self.assertEqual(turn.operations, ())
        self.assertEqual(turn.decision.intent, "tracker_change")
        self.assertIn("assign a project", turn.failure_reason)
        self.assertEqual(self.store.list_project_records(self.project.id), [])

    def test_a_reused_message_id_with_different_content_raises_before_the_model(self) -> None:
        model = ScriptedContextualModel([{"intent": "reply_only", "reply": "好的。"}])
        self.submit(model, "第一条消息。", "reused-message-id")

        with self.assertRaisesRegex(ValueError, "different content"):
            self.submit(ScriptedContextualModel([]), "另一条消息。", "reused-message-id")

    # The turn graph

    def test_a_turn_runs_through_the_router_graph_without_a_checkpointer(self) -> None:
        service = ConversationRouterService(self.store, ScriptedContextualModel([]), date(2026, 9, 9))
        graph = service.graph.get_graph()

        self.assertIsNone(service.graph.checkpointer)
        self.assertEqual(
            set(graph.nodes) - {"__start__", "__end__"},
            {
                "load_turn",
                "reload",
                "resume",
                "guard_credentials",
                "decide",
                "calendar_policy",
                "clarify",
                "local_answer",
                "project_answer",
                "execute",
            },
        )
        # docs/langgraph-router-migration.md draws these edges; keep the two in step.
        self.assertEqual(
            {(edge.source, edge.target) for edge in graph.edges},
            {
                ("__start__", "load_turn"),
                ("load_turn", "reload"),
                ("load_turn", "resume"),
                ("reload", "__end__"),
                ("resume", "__end__"),
                ("resume", "guard_credentials"),
                ("guard_credentials", "__end__"),
                ("guard_credentials", "decide"),
                ("decide", "__end__"),
                ("decide", "calendar_policy"),
                ("decide", "clarify"),
                ("decide", "local_answer"),
                ("decide", "project_answer"),
                ("decide", "execute"),
                ("calendar_policy", "__end__"),
                ("calendar_policy", "execute"),
                ("clarify", "__end__"),
                ("local_answer", "__end__"),
                ("project_answer", "__end__"),
                ("execute", "__end__"),
            },
        )


if __name__ == "__main__":
    unittest.main()
