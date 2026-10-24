import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, time, timezone
from pathlib import Path

from captureflow.conversation_router_service import ConversationRouterService, _stable_id
from captureflow.date_tool import calendar_reference
from captureflow.llm import ModelError
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
    """Answer the agent's calls from a script, keeping the request each one was sent.

    A turn makes one kind of call now. This used to script a second -- the router's
    one-shot classification, which chose an intent and the write to carry out -- and a
    write test scripted that instead of the agent. A write is a tool the agent calls, so
    a write test now scripts the tool call and the reply after it, and the rule's own
    wording comes back in the tool result rather than in the turn's reply.
    """

    def __init__(self, agent_turns=None):
        self.agent_turns = list(agent_turns or [])
        self.agent_calls = []

    def next_read_agent_turn(self, **kwargs):
        self.agent_calls.append(kwargs)
        turn = self.agent_turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def tool_turn(name: str, arguments: dict, call_id: str = "write") -> AgentModelTurn:
    return AgentModelTurn(tool_calls=[AgentToolCall(id=call_id, name=name, arguments=arguments)])


def reply_turn(reply: str = "好了。", *source_ids: str) -> AgentModelTurn:
    return AgentModelTurn(
        final_response=AgentFinalResponse(reply=reply, source_ids=list(source_ids))
    )



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

    # The agent in front of the turn

    def test_the_agents_prompt_is_pinned_so_a_change_to_it_is_deliberate(self) -> None:
        # This replaces the pin on the router's write prompt, which no longer exists. The
        # prompt is part of every recorded request, so changing it makes every recording
        # unreplayable: the pin is here to make that a decision rather than a surprise.
        pinned = "050c8cd8d03768448773d852290991b3030f2fb5224463d0b5bbae6160e09690"
        model = ScriptedContextualModel([reply_turn("好的。")])

        self.submit(model, "你好", "pinned-prompt-turn")

        system = model.agent_calls[0]["messages"][0]["content"]
        self.assertEqual(hashlib.sha256(system.encode("utf-8")).hexdigest(), pinned)

    def test_a_question_is_answered_without_the_router_classifying_it(self) -> None:
        # The router's one-shot classification is what kept read questions from reaching
        # the agent, so in this mode it is not consulted at all.
        event = self._saved_event("front-review", "Design review", date(2026, 9, 18))
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="search",
                            name="search_calendar",
                            arguments={"start_date": "2026-09-18", "end_date": "2026-09-18"},
                        )
                    ]
                ),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        reply="The design review is on the 18th.", source_ids=[event.id]
                    )
                ),
            ],
        )

        turn = self.submit(model, "What is scheduled on September 18?", "front-read")

        self.assertEqual(turn.outcome, "agent_answer")
        self.assertEqual(turn.agent_stop_reason, "completed")
        self.assertEqual((turn.model_calls, turn.tool_calls), (2, 1))
        self.assertEqual(turn.operations, ())
        self.assertIn(event.title, turn.assistant_message.content)

    def test_a_replayed_question_needs_no_model_call_in_agent_front_mode(self) -> None:
        event = self._saved_event("replay-review", "Design review", date(2026, 9, 18))
        turns = [
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(
                        id="search",
                        name="search_calendar",
                        arguments={"start_date": "2026-09-18", "end_date": "2026-09-18"},
                    )
                ]
            ),
            AgentModelTurn(
                final_response=AgentFinalResponse(reply="The review is on the 18th.", source_ids=[event.id])
            ),
        ]
        model = ScriptedContextualModel(turns)
        router = ConversationRouterService(self.store, model, date(2026, 9, 9))

        first = router.submit(self.conversation.id, "What is on September 18?", "replay-read")
        again = router.submit(self.conversation.id, "What is on September 18?", "replay-read")

        self.assertEqual(again.model_calls, 0)
        self.assertEqual(again.assistant_message.id, first.assistant_message.id)
        self.assertEqual(len(model.agent_calls), 2)

    def test_a_question_without_a_project_still_reaches_the_agent(self) -> None:
        # The router used to refuse a project question before any model call; with no intent
        # there is nothing to refuse, and the project tools simply are not offered.
        unassigned = self.store.save_conversation(Conversation(id="no-project-chat", title="Inbox"))
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    final_response=AgentFinalResponse(reply="No project data is stored for this chat.")
                )
            ],
        )

        turn = self.submit(
            model, "这个项目有什么待办？", "no-project-read", conversation_id=unassigned.id
        )

        self.assertEqual(turn.outcome, "agent_answer")
        offered = {tool["function"]["name"] for tool in model.agent_calls[0]["tools"]}
        self.assertNotIn("search_project_records", offered)
        self.assertNotIn("set_deadline", offered)
        self.assertIn("save_note", offered)

    # The agent's own first write

    def test_the_agent_saves_a_note_through_the_routers_write_path(self) -> None:
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="save",
                            name="save_note",
                            arguments={"content": "Latency work starts Monday."},
                        )
                    ]
                ),
                AgentModelTurn(final_response=AgentFinalResponse(reply="Saved your note.")),
            ],
        )

        turn = self.submit(model, "记一下：延迟排查从周一开始。", "agent-note")

        item = self.store.list_saved_items(workspace_id="default", kinds=("note",))[0]
        self.assertEqual(turn.outcome, "save_note")
        self.assertEqual([operation.tool_name for operation in turn.operations], ["save_note"])
        self.assertEqual(item.content, "Latency work starts Monday.")
        # The write keeps the router's grounding: the stored row quotes the message it came from.
        self.assertEqual(item.source_message_id, "agent-note")
        self.assertEqual(item.source_quote, "记一下：延迟排查从周一开始。")
        self.assertIn("Saved your note.", turn.assistant_message.content)

    def test_the_agent_creates_a_tracker_record_through_the_validated_path(self) -> None:
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="record",
                            name="create_record",
                            arguments={
                                "kind": "task",
                                "content": "排查发布者延迟",
                                "due_date": "2026-09-18",
                            },
                        )
                    ]
                ),
                AgentModelTurn(final_response=AgentFinalResponse(reply="记下了这个任务。")),
            ],
        )

        turn = self.submit(model, "加一个任务：排查发布者延迟，9 月 18 号前。", "agent-record")

        record = self.store.list_project_records(self.project.id, states=("active",))[0]
        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(record.content, "排查发布者延迟")
        self.assertEqual(record.kind, "task")
        # Python attaches the deadline; a date with no clock time stays a date, as it does
        # on the routed path.
        self.assertEqual(record.due_at.isoformat(), "2026-09-18")
        self.assertEqual(record.source_message_id, "agent-record")

    def test_a_repeating_deadline_is_refused_by_the_same_rule_as_the_router(self) -> None:
        # The agent cannot state one: the parameter does not exist, so the rule is the
        # schema rather than a sentence.
        offered = None
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="record",
                            name="create_record",
                            arguments={"kind": "task", "content": "每周复盘", "recurring": True},
                        )
                    ]
                ),
                AgentModelTurn(final_response=AgentFinalResponse(reply="我没有存重复的截止日期。")),
            ],
        )

        turn = self.submit(model, "每周一复盘。", "agent-recurring")

        offered = {tool["function"]["name"] for tool in model.agent_calls[0]["tools"]}
        self.assertIn("create_record", offered)
        self.assertEqual(self.store.list_project_records(self.project.id), [])
        self.assertEqual(turn.operations, ())
        feedback = model.agent_calls[1]["messages"][-1]["content"]
        self.assertIn("recurring", feedback)

    def test_a_second_decision_is_held_for_review_and_cannot_be_cited(self) -> None:
        self._project_record("first-decision", "decision", "active", "用 SQLite 存储")
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="record",
                            name="create_record",
                            arguments={"kind": "decision", "content": "改用 Postgres"},
                        )
                    ]
                ),
                AgentModelTurn(final_response=AgentFinalResponse(reply="这条决定等待复核。")),
            ],
        )

        turn = self.submit(model, "我们改用 Postgres。", "agent-decision")

        self.assertEqual([operation.status for operation in turn.operations], ["review"])
        self.assertEqual(
            [record.content for record in self.store.list_project_records(self.project.id, states=("active",))],
            ["用 SQLite 存储"],
        )
        result = json.loads(model.agent_calls[1]["messages"][-1]["content"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["items"], [])
        self.assertIn("review", result["detail"])

    def test_the_agent_creates_an_event_and_python_attaches_the_timezone(self) -> None:
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="event",
                            name="create_event",
                            arguments={
                                "title": "和导师开会",
                                "date": "2026-09-16",
                                "start_time": "15:00",
                                "duration_minutes": 60,
                            },
                        )
                    ]
                ),
                AgentModelTurn(final_response=AgentFinalResponse(reply="建好了。")),
            ],
        )

        turn = self.submit(model, "下周三下午三点到四点和导师开会。", "agent-event")

        event = self.store.list_calendar_events()[0]
        self.assertEqual(turn.outcome, "create_local_event")
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-09-16T15:00:00-07:00"))
        self.assertEqual(event.end_at, datetime.fromisoformat("2026-09-16T16:00:00-07:00"))
        self.assertEqual(event.source_message_id, "agent-event")

    def test_a_calendar_rule_that_asks_the_user_reaches_the_agent_as_a_question(self) -> None:
        # The router's own wording, so the rule lives in one place whoever asked for the write.
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="event",
                            name="create_event",
                            arguments={"title": "周会", "date": "2026-09-14", "recurring": True},
                        )
                    ]
                ),
                AgentModelTurn(
                    final_response=AgentFinalResponse(reply="我只能存单次日程，具体哪一天？")
                ),
            ],
        )

        turn = self.submit(model, "每周一上午十点开周会。", "agent-recurring-event")

        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(turn.operations, ())
        feedback = json.loads(model.agent_calls[1]["messages"][-1]["content"])
        self.assertEqual(feedback["error"]["category"], "clarification_required")
        self.assertIn("repeating schedule", feedback["error"]["message"])

    # The search a write that names a stored target has to make first, by what it changes.
    # The ledger only vouches for an id a tool returned in this same turn, so a script that
    # left the search out would be refused as a stale target whatever else it got right.
    SEARCH_FIRST = {
        "update_event": "search_calendar",
        "delete_event": "search_calendar",
        "transition_record": "search_project_records",
        "set_deadline": "search_project_records",
        "supersede_record": "search_project_records",
    }

    def _writes(self, tool: str, arguments: dict, search: dict | None = None) -> "ScriptedContextualModel":
        """A scripted turn that calls one write tool, after the search that write needs."""

        turns = []
        if tool in self.SEARCH_FIRST:
            turns.append(
                tool_turn(
                    self.SEARCH_FIRST[tool] if search is None else search["name"],
                    {} if search is None else search.get("arguments", {}),
                    "search",
                )
            )
        turns.extend([tool_turn(tool, arguments), reply_turn("改好了。")])
        return ScriptedContextualModel(turns)

    @staticmethod
    def _tool_result(model: "ScriptedContextualModel") -> dict:
        """What the last write returned to the model, which is where a rule's wording goes.

        A routed write put the rule's own sentence in the turn's reply. The agent writes
        through a tool, so the sentence comes back as that tool's result and the reply is
        the model's; asserting on the result is asserting on the rule.
        """

        messages = model.agent_calls[-1]["messages"]
        return json.loads([message for message in messages if message["role"] == "tool"][-1]["content"])

    def test_the_agent_updates_an_event_it_searched_for(self) -> None:
        event = self._saved_event("target-review", "架构评审", datetime.fromisoformat("2026-09-15T10:00:00-07:00"))
        model = self._writes("update_event", {"target_id": event.id, "date": "2026-09-18", "start_time": "14:00"})

        turn = self.submit(model, "把架构评审改到下周五下午两点。", "agent-update")

        updated = self.store.get_calendar_event(event.id)
        self.assertEqual(turn.outcome, "calendar_update")
        self.assertEqual(updated.start_at, datetime.fromisoformat("2026-09-18T14:00:00-07:00"))
        self.assertEqual(
            [operation.status for operation in turn.operations], ["committed"]
        )

    def test_a_deletion_the_agent_asks_for_still_waits_for_the_user(self) -> None:
        # There is no checkpointer, so a deletion is a stored operation the next turn
        # confirms, exactly as on the routed path.
        event = self._saved_event("doomed", "发布演示", date(2026, 9, 20))
        model = self._writes("delete_event", {"target_id": event.id})

        turn = self.submit(model, "把发布演示删掉。", "agent-delete")

        self.assertEqual([operation.status for operation in turn.operations], ["review"])
        self.assertEqual(self.store.get_calendar_event(event.id).status, "active")
        tool_messages = [
            message
            for message in model.agent_calls[2]["messages"]
            if message["role"] == "tool"
        ]
        result = json.loads(tool_messages[-1]["content"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["items"], [])
        self.assertIn("Confirm", result["detail"])

    def test_the_agent_finishes_a_task_it_searched_for(self) -> None:
        task = self._project_record("done-task", "task", "active", "排查延迟")
        model = self._writes(
            "transition_record",
            {"target_id": task.id, "target_state": "completed"},
        )

        turn = self.submit(model, "延迟排查做完了。", "agent-transition")

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(self.store.get_project_record(task.id).state, "completed")

    def test_a_deadline_the_agent_sets_is_attached_by_python(self) -> None:
        task = self._project_record("dated-task", "task", "active", "写报告")
        model = self._writes(
            "set_deadline",
            {"target_id": task.id, "due_date": "2026-09-18", "due_time": "17:00"},
        )

        self.submit(model, "报告周五下午五点前交。", "agent-deadline")

        self.assertEqual(
            self.store.get_project_record(task.id).due_at,
            datetime.fromisoformat("2026-09-18T17:00:00-07:00"),
        )

    def test_a_target_the_agent_did_not_search_for_is_refused_before_any_write(self) -> None:
        event = self._saved_event("unsearched", "代码冻结", date(2026, 9, 11))
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="change",
                            name="delete_event",
                            arguments={"target_id": event.id},
                        )
                    ]
                ),
                AgentModelTurn(final_response=AgentFinalResponse(reply="我需要先找到它。")),
            ],
        )

        turn = self.submit(model, "把代码冻结删掉。", "agent-unsearched")

        self.assertEqual(turn.operations, ())
        self.assertEqual(self.store.get_calendar_event(event.id).status, "active")
        feedback = json.loads(model.agent_calls[1]["messages"][-1]["content"])
        self.assertEqual(feedback["error"]["category"], "stale_target")

    def test_a_write_is_reported_even_when_the_turn_runs_out_of_budget_after_it(self) -> None:
        # The note is in SQLite by then, so a reply that did not mention it would be false.
        save = AgentModelTurn(
            tool_calls=[
                AgentToolCall(id="save", name="save_note", arguments={"content": "Latency work."})
            ]
        )
        model = ScriptedContextualModel([save, save, save])

        turn = self.submit(model, "记一下：延迟排查。", "agent-note-budget")

        self.assertEqual([operation.tool_name for operation in turn.operations], ["save_note"])
        self.assertEqual(turn.agent_stop_reason, "model_budget_exhausted")
        self.assertIn("Saved note", turn.assistant_message.content)
        self.assertEqual(len(self.store.list_saved_items(workspace_id="default", kinds=("note",))), 1)

    def test_the_same_message_submitted_twice_saves_one_note(self) -> None:
        turns = [
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(id="save", name="save_note", arguments={"content": "Once only."})
                ]
            ),
            AgentModelTurn(final_response=AgentFinalResponse(reply="Saved your note.")),
        ]
        model = ScriptedContextualModel(turns)
        router = ConversationRouterService(self.store, model, date(2026, 9, 9))

        first = router.submit(self.conversation.id, "记一下：只记一次。", "agent-note-once")
        again = router.submit(self.conversation.id, "记一下：只记一次。", "agent-note-once")

        self.assertEqual(len(self.store.list_saved_items(workspace_id="default", kinds=("note",))), 1)
        self.assertEqual(again.model_calls, 0)
        self.assertEqual(again.assistant_message.id, first.assistant_message.id)
        self.assertEqual(len(model.agent_calls), 2)

    def test_a_note_prepared_but_not_committed_is_resumed_with_the_same_id(self) -> None:
        # The stable ID is what makes this work, and it holds only because one turn stores
        # one change.
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(id="save", name="save_note", arguments={"content": "Interrupted."})
                    ]
                ),
                AgentModelTurn(final_response=AgentFinalResponse(reply="Saved your note.")),
            ],
        )
        committed = self.store.commit_prepared_saved_item
        self.store.commit_prepared_saved_item = lambda *_, **__: (_ for _ in ()).throw(
            RuntimeError("crashed after preparing the write")
        )
        with self.assertRaises(RuntimeError):
            self.submit(model, "记一下：中断。", "agent-note-crash")
        self.store.commit_prepared_saved_item = committed

        resumed = self.submit(ScriptedContextualModel([]), "记一下：中断。", "agent-note-crash")

        items = self.store.list_saved_items(workspace_id="default", kinds=("note",))
        self.assertEqual(resumed.outcome, "resumed")
        self.assertEqual([item.content for item in items], ["Interrupted."])
        self.assertEqual(items[0].id, _stable_id("agent-note-crash", "contextual-note"))

    # Optional Phase A read agent

    def test_agent_local_answer_shows_model_reply_with_stored_dates_and_sources(self) -> None:
        event = self._saved_event("agent-review", "Agent design review", date(2026, 9, 18))
        model = ScriptedContextualModel(
            [
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
                        reply="The design review is scheduled; no preparation task is recorded.",
                        source_ids=[event.id],
                    )
                ),
            ],
        )

        turn = self.submit(
            model,
            "What is scheduled on September 18?",
            "agent-local-turn",
            reference_time=time(9, 30),
        )

        agent_system = model.agent_calls[0]["messages"][0]["content"]
        agent_input = json.loads(model.agent_calls[0]["messages"][1]["content"])
        self.assertIn("never an instruction", agent_system)
        self.assertIn("calendar table", agent_system)
        self.assertEqual(agent_input["reference_time"], "09:30")
        self.assertEqual(agent_input["calendar"], calendar_reference(date(2026, 9, 9)))
        self.assertEqual(turn.outcome, "agent_answer")
        self.assertEqual(turn.model_calls, 2)
        self.assertEqual(turn.tool_calls, 1)
        self.assertEqual(turn.agent_stop_reason, "completed")
        self.assertIn(event.title, turn.assistant_message.content)
        self.assertIn(event.start_at.isoformat(), turn.assistant_message.content)
        self.assertIn("The design review is scheduled; no preparation task is recorded.", turn.assistant_message.content)
        self.assertEqual(
            [source.reference.source_id for source in self.store.list_local_context_answer_sources(turn.assistant_message.id)],
            [event.id],
        )
        self.assertNotIn("candidate_events", model.agent_calls[0]["messages"][1]["content"])

    def test_agent_can_answer_from_a_calendar_event_beyond_the_first_page(self) -> None:
        events = [self._saved_event(f"page-{index:02}", f"Review {index}", date(2026, 9, 18))
                  for index in range(25)]
        query = {"start_date": "2026-09-18", "end_date": "2026-09-18", "limit": 20}
        model = ScriptedContextualModel(
            [
                AgentModelTurn(tool_calls=[AgentToolCall(id="page-one", name="search_calendar", arguments=query)]),
                AgentModelTurn(tool_calls=[AgentToolCall(id="page-two", name="search_calendar",
                                                       arguments={**query, "offset": 20})]),
                AgentModelTurn(final_response=AgentFinalResponse(reply="The matching review is stored.",
                                                                source_ids=[events[-1].id])),
            ],
        )

        turn = self.submit(model, "What is stored for Review 24?", "paged-answer")

        self.assertIsNone(turn.failure_reason)
        self.assertIn(events[-1].title, turn.assistant_message.content)
        self.assertEqual(
            [source.reference.source_id for source in self.store.list_local_context_answer_sources(turn.assistant_message.id)],
            [events[-1].id],
        )

    def test_agent_project_answer_shows_model_prose_and_sqlite_dates(self) -> None:
        task = self._project_record(
            "latency-task",
            "task",
            "active",
            "Measure agent latency",
            due_at=date(2026, 9, 25),
        )
        model = ScriptedContextualModel(
            [
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
        )

        self.assertEqual(turn.outcome, "agent_answer")
        self.assertIn("Latency measurement is still open.", turn.assistant_message.content)
        self.assertIn(task.due_at.isoformat(), turn.assistant_message.content)
        self.assertEqual(
            [record.id for record in self.store.list_project_answer_sources(turn.assistant_message.id)],
            [task.id],
        )

    def test_agent_can_save_more_than_twelve_sources_of_each_type(self) -> None:
        events = [self._saved_event(f"review-{index:02}", f"Review {index}", date(2026, 9, 18))
                  for index in range(13)]
        tasks = [self._project_record(f"task-{index}", "task", "active", f"Prepare review {index}")
                 for index in range(13)]
        ids = [row.id for row in [*events, *tasks]]
        model = ScriptedContextualModel(
            [
                AgentModelTurn(tool_calls=[
                    AgentToolCall(id="events", name="search_calendar", arguments={"limit": 20}),
                    AgentToolCall(id="tasks", name="search_project_records", arguments={"kinds": ["task"], "limit": 20}),
                ]),
                AgentModelTurn(final_response=AgentFinalResponse(
                    reply="Preparation tasks remain open alongside the scheduled reviews.", source_ids=[*ids, ids[0]]
                )),
            ],
        )

        turn = self.submit(model, "What remains for the reviews?", "many-sources")

        self.assertIsNone(turn.failure_reason)
        self.assertEqual({row.id for row in self.store.list_project_answer_sources(turn.assistant_message.id)},
                         {task.id for task in tasks})
        self.assertEqual([row.reference.source_id for row in self.store.list_local_context_answer_sources(turn.assistant_message.id)],
                         [event.id for event in events])
        self.assertEqual(turn.operations, ())

    def test_agent_can_ask_a_question_or_explain_uncertainty_without_sources(self) -> None:
        for index, reply in enumerate(("Which meeting do you mean?", "I have not searched the records yet, so I cannot confirm that.")):
            with self.subTest(reply=reply):
                model = ScriptedContextualModel(
                    [AgentModelTurn(final_response=AgentFinalResponse(reply=reply))],
                )

                turn = self.submit(model, "What about that meeting?", f"unsourced-{index}")

                self.assertIsNone(turn.failure_reason)
                self.assertEqual(turn.assistant_message.content, reply)
                self.assertEqual(self.store.list_local_context_answer_sources(turn.assistant_message.id), [])
                self.assertEqual(turn.operations, ())

    def test_agent_local_answer_lists_cited_events_in_time_order(self) -> None:
        later = self._saved_event("later-review", "Later review", datetime.fromisoformat("2026-09-17T09:00:00-07:00"))
        all_day = self._saved_event("all-day-offsite", "Offsite", date(2026, 9, 15))
        earlier = self._saved_event("earlier-review", "Earlier review", datetime.fromisoformat("2026-09-15T14:00:00-07:00"))
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="week-search",
                            name="search_calendar",
                            arguments={"start_date": "2026-09-14", "end_date": "2026-09-20"},
                        )
                    ]
                ),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        reply="The stored records answer the question.", source_ids=[later.id, earlier.id, all_day.id]
                    )
                ),
            ],
        )

        turn = self.submit(model, "我下周有什么安排？", "week-question")

        content = turn.assistant_message.content
        positions = [content.index(event.start_at.isoformat()) for event in (all_day, earlier, later)]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(
            [source.reference.source_id for source in self.store.list_local_context_answer_sources(turn.assistant_message.id)],
            [all_day.id, earlier.id, later.id],
        )

    def test_an_agent_failure_counts_every_call_the_agent_made(self) -> None:
        self._saved_event("agent-failure", "Agent design review", date(2026, 9, 18))
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="calendar-search",
                            name="search_calendar",
                            arguments={"start_date": "2026-09-18", "end_date": "2026-09-18"},
                        )
                    ]
                ),
                ModelError("Model request failed: timeout"),
            ],
        )

        turn = self.submit(model, "What is on September 18?", "agent-failure-turn")

        self.assertEqual(turn.outcome, "agent_model_failure")
        self.assertEqual(turn.model_calls, 2)
        self.assertEqual(turn.tool_calls, 1)
        self.assertEqual(turn.operations, ())
        self.assertIn("timeout", turn.failure_reason)

    def test_an_answer_whose_sources_fail_to_save_leaves_only_the_failure_reply(self) -> None:
        task = self._project_record("unsaved-source-task", "task", "active", "Measure agent latency")
        model = ScriptedContextualModel(
            [
                AgentModelTurn(
                    tool_calls=[AgentToolCall(id="record-search", name="search_project_records", arguments={})]
                ),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        reply="Latency is still open.", source_ids=[task.id]
                    )
                ),
            ],
        )

        def fail_inside_the_transaction(connection, assistant_message_id, record_ids):
            raise ValueError("simulated source write failure")

        self.store._insert_project_answer_sources = fail_inside_the_transaction
        turn = self.submit(model, "What remains?", "unsaved-sources-turn")
        del self.store._insert_project_answer_sources

        replies = [
            message
            for message in self.store.list_messages(self.conversation.id)
            if message.reply_to_message_id == "unsaved-sources-turn"
        ]
        self.assertEqual(turn.outcome, "write_or_dispatch_failure")
        self.assertEqual(turn.model_calls, 2)
        self.assertEqual(turn.tool_calls, 1)
        self.assertEqual(replies, [turn.assistant_message])
        self.assertNotIn("Latency is still open.", turn.assistant_message.content)
        self.assertEqual(self.store.list_project_answer_sources(turn.assistant_message.id), [])

    # Calendar creation

    def test_a_description_in_the_models_own_words_is_kept(self) -> None:
        content = "明天下午三点在 A301 和张伟开需求评审，讨论驱动排期"
        model = self._writes(
            "create_event",
            {
                "title": "需求评审",
                "description": "与张伟在 A301 讨论驱动排期",
                "date": "2026-09-10",
                "start_time": "15:00",
            },
        )

        self.submit(model, content, "paraphrase-turn")

        event = self.store.list_calendar_events()[0]
        self.assertEqual(event.description, "与张伟在 A301 讨论驱动排期")
        self.assertEqual(event.source_quote, content)

    def test_an_end_time_and_a_duration_that_agree_create_the_event(self) -> None:
        # Both are stated for one meeting often enough -- 四点到五点，开一个小时 -- and the
        # second case is the one that needs the day rolled: 23:30 to 00:30 is tomorrow.
        for label, start, end, minutes, expected_end in (
            ("same-day", "16:00", "17:00", 60, "2026-09-10T17:00:00-07:00"),
            ("past-midnight", "23:30", "00:30", 60, "2026-09-11T00:30:00-07:00"),
        ):
            with self.subTest(label=label):
                model = self._writes(
                    "create_event",
                    {
                        "title": f"代码审查会议 {label}",
                        "date": "2026-09-10",
                        "start_time": start,
                        "end_time": end,
                        "duration_minutes": minutes,
                    },
                )

                turn = self.submit(
                    model,
                    f"明天{start}到{end}有代码审查会议，开{minutes}分钟。",
                    f"end-and-duration-{label}",
                )

                self.assertEqual(turn.outcome, "create_local_event")
                event = self.store.get_calendar_event(turn.operations[0].result["event_id"])
                self.assertEqual(event.end_at, datetime.fromisoformat(expected_end))

    def test_an_end_time_and_a_duration_that_disagree_ask_for_the_length(self) -> None:
        model = self._writes(
            "create_event",
            {
                "title": "代码审查会议",
                "date": "2026-09-10",
                "start_time": "16:00",
                "end_time": "17:00",
                "duration_minutes": 90,
            },
        )

        turn = self.submit(model, "明天下午四点到五点有代码审查会议，开一个半小时。", "end-duration-conflict")

        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(turn.operations, ())
        refused = self._tool_result(model)["error"]
        self.assertEqual(refused["category"], "clarification_required")
        self.assertIn("How long", refused["message"])

    def test_the_current_local_time_reaches_the_model(self) -> None:
        model = self._writes("create_event", {"title": "短会", "date": "2026-09-12", "start_time": "14:40"})

        self.submit(
            model,
            "半小时后开个短会。",
            "relative-now-turn",
            date(2026, 9, 12),
            reference_time=time(14, 10, 37),
        )

        sent = json.loads(model.agent_calls[0]["messages"][1]["content"])
        self.assertEqual(sent["reference_time"], "14:10")
        self.assertEqual(sent["reference_date"], "2026-09-12")
        self.assertEqual(sent["calendar"], calendar_reference(date(2026, 9, 12)))
        self.assertEqual(
            self.store.list_calendar_events()[0].start_at,
            datetime.fromisoformat("2026-09-12T14:40:00-07:00"),
        )

    def test_python_attaches_the_offset_a_date_actually_has(self) -> None:
        model = self._writes(
            "create_event",
            {"title": "季度评审", "date": "2026-11-02", "start_time": "15:00", "end_time": "16:00"},
        )

        self.submit(model, "11月2日下午三点到四点有季度评审。", "after-dst-turn", date(2026, 9, 12))

        event = self.store.list_calendar_events()[0]
        self.assertEqual(event.start_at, datetime.fromisoformat("2026-11-02T15:00:00-08:00"))
        self.assertEqual(event.end_at, datetime.fromisoformat("2026-11-02T16:00:00-08:00"))

    def test_a_stated_foreign_timezone_asks_instead_of_storing_local_time(self) -> None:
        model = self._writes(
            "create_event",
            {"title": "电话会", "date": "2026-09-12", "timezone": "America/New_York", "start_time": "09:00"},
        )

        turn = self.submit(model, "纽约时间明天上午九点有电话会", "foreign-timezone-turn", date(2026, 9, 11))

        self.assertEqual(self.store.list_calendar_events(), [])
        self.assertEqual(turn.operations, ())
        refused = self._tool_result(model)["error"]
        self.assertEqual(refused["category"], "clarification_required")
        self.assertIn("America/Vancouver", refused["message"])

    def test_the_local_timezone_by_name_still_creates_the_event(self) -> None:
        model = self._writes(
            "create_event",
            {"title": "电话会", "date": "2026-09-12", "timezone": "America/Vancouver", "start_time": "09:00"},
        )

        turn = self.submit(model, "温哥华时间明天上午九点有电话会", "local-timezone-turn", date(2026, 9, 11))

        self.assertEqual(turn.outcome, "create_local_event")
        self.assertEqual(
            self.store.list_calendar_events()[0].start_at,
            datetime.fromisoformat("2026-09-12T09:00:00-07:00"),
        )

    # Changing an event the agent searched for

    def test_an_update_applies_a_new_start_and_duration_and_can_be_undone(self) -> None:
        event = self._saved_event(
            "shortened", "相关的说明会", datetime.fromisoformat("2026-09-10T16:00:00-07:00")
        )
        correction = "会议延期了，时间也被缩短到半个小时，现在晚上七点才开始"
        model = self._writes(
            "update_event",
            {"target_id": event.id, "start_time": "19:00", "duration_minutes": 30},
        )

        updated = self.submit(model, correction, "shortened-update", date(2026, 9, 10))
        # The same message again: the stored operation answers it, and the script has no
        # turn left, so a second model call would raise rather than pass unnoticed.
        replay = self.submit(model, correction, "shortened-update", date(2026, 9, 10))
        operation = self.store.list_calendar_event_update_operations("shortened-update")[0]
        stored = self.store.get_calendar_event(event.id)

        self.assertEqual(updated.outcome, "calendar_update")
        self.assertEqual(replay.model_calls, 0)
        self.assertEqual(stored.start_at, datetime.fromisoformat("2026-09-10T19:00:00-07:00"))
        self.assertEqual(stored.end_at, datetime.fromisoformat("2026-09-10T19:30:00-07:00"))
        self.assertEqual(self.store.undo_calendar_event_update(operation.id).status, "undone")
        restored = self.store.get_calendar_event(event.id)
        self.assertEqual(restored.start_at, datetime.fromisoformat("2026-09-10T16:00:00-07:00"))
        self.assertIsNone(restored.end_at)

    def test_an_update_that_changes_nothing_writes_nothing(self) -> None:
        event = self._saved_event(
            "already-thursday", "架构评审", datetime.fromisoformat("2026-09-10T14:00:00-07:00")
        )
        model = self._writes(
            "update_event",
            {"target_id": event.id, "date": "2026-09-10"},
        )

        turn = self.submit(model, "架构评审改到周四。", "no-op-update")

        self.assertEqual(turn.outcome, "calendar_unchanged")
        self.assertEqual(turn.operations, ())
        self.assertEqual(self.store.list_calendar_event_update_operations("no-op-update"), [])
        self.assertIn("nothing was changed", self._tool_result(model)["detail"])

    def test_an_update_without_a_new_value_asks_for_the_date(self) -> None:
        # 改到周四 with two plausible Thursdays: the event was found, its new value was not.
        event = self._saved_event(
            "undecided", "工作流会议", datetime.fromisoformat("2026-09-10T16:00:00-07:00")
        )
        model = self._writes("update_event", {"target_id": event.id})

        turn = self.submit(model, "这个会议改到周四。", "undecided-update-turn")

        self.assertEqual(turn.operations, ())
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_event_update_operations("undecided-update-turn"), [])
        refused = self._tool_result(model)["error"]
        self.assertEqual(refused["category"], "clarification_required")
        self.assertIn("Which exact date", refused["message"])

    def test_an_implausible_update_date_asks_for_the_date(self) -> None:
        event = self._saved_event(
            "implausible", "工作流会议", datetime.fromisoformat("2026-09-10T16:00:00-07:00")
        )
        model = self._writes(
            "update_event",
            {"target_id": event.id, "date": "2030-09-10"},
        )

        turn = self.submit(model, "这个会议改到周三。", "implausible-update-turn")

        self.assertEqual(turn.operations, ())
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertIn("Which exact date", self._tool_result(model)["error"]["message"])

    def test_a_malformed_normalized_time_is_refused_before_the_write_path(self) -> None:
        # The router validated the shape after choosing the write. The parameter's own
        # pattern refuses it first now, so the rule is the schema instead of a check.
        event = self._saved_event(
            "unsupported", "工作流会议", datetime.fromisoformat("2026-09-10T16:00:00-07:00")
        )
        model = self._writes(
            "update_event",
            {"target_id": event.id, "start_time": "7 PM"},
        )

        turn = self.submit(model, "这个会议改到晚上七点才开始。", "unsupported-correction", date(2026, 9, 10))

        self.assertEqual(turn.operations, ())
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(self.store.list_calendar_event_update_operations("unsupported-correction"), [])
        refused = self._tool_result(model)["error"]
        self.assertEqual(refused["category"], "retryable")
        self.assertIn("start_time", refused["message"])

    def test_a_new_length_for_an_all_day_event_asks_for_a_start_time(self) -> None:
        event = self._saved_event("all-day", "工作流会议", date(2026, 9, 10))
        model = self._writes(
            "update_event",
            {"target_id": event.id, "duration_minutes": 30},
        )

        turn = self.submit(model, "这个会议开半个小时。", "duration-correction", date(2026, 9, 10))

        self.assertEqual(turn.operations, ())
        self.assertEqual(self.store.get_calendar_event(event.id), event)
        self.assertEqual(
            self._tool_result(model)["error"]["message"],
            "What exact start time should I use for this local event?",
        )

    # A write interrupted mid-turn is recovered from SQLite, not from a checkpointer

    def test_prepared_duration_update_resumes_after_reopening_the_database(self) -> None:
        event = self._saved_event(
            "prepared-update", "工作流会议", datetime.fromisoformat("2026-09-10T16:00:00-07:00")
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
            operation, [EvidenceReference(message_id=correction.id, source_quote=correction.content)]
        )
        restarted = CaptureStore(self.store.database_path)
        restarted.initialize()
        model = ScriptedContextualModel()

        resumed = ConversationRouterService(restarted, model, date(2026, 9, 10)).submit(
            self.conversation.id, correction.content, correction.id
        )

        self.assertEqual(resumed.outcome, "resumed")
        self.assertEqual(model.agent_calls, [])
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
        model = ScriptedContextualModel()

        resumed = self.submit(model, current.content, current.id, date(2026, 9, 10))

        event = self.store.list_calendar_events()[0]
        self.assertEqual(resumed.outcome, "resumed")
        self.assertEqual(self.store.list_calendar_event_sources(event.id), sources)
        self.assertEqual(model.agent_calls, [])

    # Calendar deletion waits for the user

    def test_a_deletion_waits_for_confirmation_then_can_be_undone(self) -> None:
        event = self._saved_event(
            "expired-meeting", "相关的说明会", datetime.fromisoformat("2026-09-10T16:00:00-07:00")
        )
        content = "昨天的会议已结束，删除过期会议"
        model = self._writes("delete_event", {"target_id": event.id})

        turn = self.submit(model, content, "delete-turn", date(2026, 9, 11))
        replay = self.submit(model, content, "delete-turn", date(2026, 9, 11))
        operation = self.store.list_calendar_event_update_operations("delete-turn")[0]

        self.assertEqual(turn.outcome, "calendar_delete_review")
        self.assertEqual(turn.operations[0].status, "review")
        self.assertEqual(replay.model_calls, 0)
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
        model = self._writes("delete_event", {"target_id": event.id})

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
        model = ScriptedContextualModel()

        turn = self.submit(model, user.content, user.id)

        self.assertEqual(model.agent_calls, [])
        self.assertEqual(turn.operations[0].status, "review")
        self.assertEqual(self.store.get_calendar_event(event.id).status, "active")

    # Credentials never reach the model

    def test_a_chinese_credential_disclosure_never_reaches_the_model(self) -> None:
        secret = "我的密钥是 sk-test-1234567890abcdef"
        blocked = self.submit(ScriptedContextualModel(), secret, "credential-turn", date(2026, 9, 11))
        model = ScriptedContextualModel([reply_turn("好的。")])

        self.submit(model, "你好", "after-credential-turn", date(2026, 9, 11))

        self.assertEqual(blocked.outcome, "blocked_credential")
        self.assertEqual(blocked.model_calls, 0)
        self.assertEqual(self.store.list_calendar_events(), [])
        sent = json.loads(model.agent_calls[0]["messages"][1]["content"])
        self.assertNotIn(
            "credential-turn", [message["message_id"] for message in sent["recent_messages"]]
        )
        self.assertNotIn("sk-test", str(model.agent_calls[0]))

    # Tracker deadlines

    def test_a_normalized_task_deadline_is_listed_for_the_calendar(self) -> None:
        model = self._writes(
            "create_record",
            {"kind": "task", "content": "提交修复", "due_date": "2026-09-16", "due_time": "20:00"},
        )

        turn = self.submit(model, "修复需要下周三晚上八点之前提交", "deadline-record-turn", date(2026, 9, 11))
        record = self.store.list_project_records(self.project.id)[0]

        self.assertEqual(turn.outcome, "tracker_change")
        self.assertEqual(record.due_at, datetime.fromisoformat("2026-09-16T20:00:00-07:00"))
        self.assertEqual(self.store.list_task_deadlines(), [record])
        self.assertIn("Deadline on the calendar", self._tool_result(model)["detail"])

    def test_an_implausible_deadline_keeps_the_task_without_one(self) -> None:
        model = self._writes("create_record", {"kind": "task", "content": "提交修复", "due_date": "2031-01-01"})

        self.submit(model, "修复要尽快提交。", "vague-deadline-turn", date(2026, 9, 11))
        record = self.store.list_project_records(self.project.id)[0]

        self.assertIsNone(record.due_at)
        detail = self._tool_result(model)["detail"]
        self.assertIn("outside the planning horizon", detail)
        self.assertIn("no deadline yet", detail)

    def test_a_deadline_timestamp_the_agent_tries_to_supply_is_refused(self) -> None:
        # Python attaches the timezone; the parameter for a ready-made timestamp does not
        # exist, so supplying one is refused by the schema rather than by a check after it.
        model = self._writes(
            "create_record",
            {
                "kind": "task",
                "content": "提交修复",
                "due_date": "2026-09-16",
                "due_at": "2026-09-16T20:00:00-07:00",
            },
        )

        turn = self.submit(model, "修复需要下周三晚上八点之前提交", "model-deadline-turn", date(2026, 9, 11))

        self.assertEqual(turn.operations, ())
        self.assertEqual(self.store.list_project_records(self.project.id), [])
        refused = self._tool_result(model)["error"]
        self.assertEqual(refused["category"], "retryable")
        self.assertIn("due_at", refused["message"])

    # The turn itself

    def test_a_reused_message_id_with_different_content_raises_before_the_model(self) -> None:
        model = ScriptedContextualModel([reply_turn("好的。")])
        self.submit(model, "第一条消息。", "reused-message-id")

        with self.assertRaisesRegex(ValueError, "different content"):
            self.submit(ScriptedContextualModel(), "另一条消息。", "reused-message-id")

    def test_a_turn_runs_through_the_router_graph_without_a_checkpointer(self) -> None:
        service = ConversationRouterService(self.store, ScriptedContextualModel(), date(2026, 9, 9))
        graph = service.graph.get_graph()

        self.assertIsNone(service.graph.checkpointer)
        self.assertEqual(
            set(graph.nodes) - {"__start__", "__end__"},
            {"load_turn", "reload", "resume", "guard_credentials", "agent"},
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
                ("guard_credentials", "agent"),
                ("agent", "__end__"),
            },
        )


if __name__ == "__main__":
    unittest.main()