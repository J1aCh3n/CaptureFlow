"""Does our own code send the same requests twice? Asked on every run, before any are paid for.

A recording is only replayable if a turn is reproducible, and a turn stops being
reproducible the moment a value made inside it reaches the model. That has three times
invalidated a recording after it was paid for: the acceptance seed's clock, a created
tracker row's commit timestamp in its own write result, and then the same timestamp
arriving from a later search that found the row the run had just created.

Everything a tool returns is the user's own data except values a write creates, so the
writes are what this covers. Each one is driven twice from a script -- no recording, so the
question can be asked of a turn that has never been recorded, which is exactly when it
needs asking.
"""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from captureflow.conversation_router_service import ConversationRouterService
from captureflow.read_agent import AgentFinalResponse, AgentModelTurn, AgentToolCall
from captureflow.storage import CaptureStore
from scripts.model_replay import first_unreproducible_request, request_digest
from scripts.run_read_agent_acceptance import (
    CASE_SEQUENCE,
    FIXTURE,
    WRITE_CASES,
    ToolResultRecorder,
    seed_database,
)


def _tools(*calls) -> AgentModelTurn:
    return AgentModelTurn(
        tool_calls=[
            AgentToolCall(id=f"call-{index}", name=name, arguments=arguments)
            for index, (name, arguments) in enumerate(calls)
        ]
    )


def _answer(reply: str) -> AgentModelTurn:
    return AgentModelTurn(final_response=AgentFinalResponse(reply=reply))


class _ScriptedTurns:
    """Answer from a script, recording the digest of each request it was sent."""

    def __init__(self, turns: list[AgentModelTurn]) -> None:
        self.turns = list(turns)
        self.sent: list[tuple[str, str, dict]] = []

    def next_read_agent_turn(self, **kwargs) -> AgentModelTurn:
        self.sent.append(
            ("next_read_agent_turn", request_digest("next_read_agent_turn", kwargs), kwargs)
        )
        return self.turns.pop(0)


class WriteDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.temporary_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary_directory.cleanup)

    def _script(self, seed) -> list[AgentModelTurn]:
        """Every write, in the order the live run submits them.

        A target write takes three turns, because the ledger makes the search a
        requirement: search, change what was found, then answer.
        """

        return [
            _tools(("save_note", {"content": "延迟排查从下周一开始。"})),
            _answer("已记下。"),
            _tools(("save_idea", {"content": "用流式读取来降低延迟。"})),
            _answer("记下这个主意。"),
            _tools(("create_record", {"kind": "task", "content": "排查发布者节点的延迟", "due_date": "2026-09-18"})),
            _answer("已添加任务。"),
            _tools(("create_event", {"title": "和导师开会", "date": "2026-09-16", "start_time": "15:00", "duration_minutes": 60})),
            _answer("已创建日程。"),
            _tools(("search_calendar", {})),
            _tools(("update_event", {"target_id": "read-agent-architecture-review", "date": "2026-09-18", "start_time": "14:00"})),
            _answer("已改时间。"),
            _tools(("search_project_records", {})),
            _tools(("set_deadline", {"target_id": seed["task"].id, "due_date": "2026-09-18", "due_time": "17:00"})),
            _answer("已记下截止时间。"),
            _tools(("search_project_records", {})),
            _tools(("transition_record", {"target_id": seed["task"].id, "target_state": "completed"})),
            _answer("已标记完成。"),
            _tools(("search_calendar", {})),
            _tools(("delete_event", {"target_id": "read-agent-release-demo"})),
            _answer("删除要你确认一下。"),
        ]

    def _one_pass(self, name: str) -> list:
        store = CaptureStore(Path(self.temporary_directory.name) / f"{name}.sqlite3")
        store.initialize()
        seed = seed_database(store)
        model = _ScriptedTurns(self._script(seed))
        tools = ToolResultRecorder()
        self.addCleanup(tools.restore)
        router = ConversationRouterService(
            store,
            model,
            date.fromisoformat(self.fixture["protocol"]["reference_date"]),
            self.fixture["protocol"]["timezone"],
        )
        inputs = {case["name"]: case["input"] for case in self.fixture["cases"]}
        conversations = {
            "main": seed["main_conversation"].id,
            "later": seed["later_conversation"].id,
        }
        for case, conversation, message_id in CASE_SEQUENCE:
            if case not in WRITE_CASES:
                continue
            router.submit(conversations[conversation], inputs[case], message_id)
        return model.sent

    def test_two_identical_runs_of_every_write_send_identical_requests(self) -> None:
        difference = first_unreproducible_request(self._one_pass("first"), self._one_pass("second"))

        self.assertIsNone(difference, difference)

    def test_the_script_covers_every_write_the_live_run_submits(self) -> None:
        # A write case added to the fixture without a script here would go unchecked, and
        # the next recording would be found unreplayable only after it was paid for.
        live = [case for case, _, _ in CASE_SEQUENCE if case in WRITE_CASES]
        answers = sum(1 for turn in self._script({"task": _Stub()}) if turn.final_response is not None)

        self.assertEqual(len(live), answers, f"{live} are submitted, {answers} are scripted")


class _Stub:
    id = "read-agent-latency-task"


if __name__ == "__main__":
    unittest.main()
