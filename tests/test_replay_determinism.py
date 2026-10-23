"""Does our own code send the same requests twice? Asked on every run, before any are paid for.

A recording is only replayable if a turn is reproducible, and a turn stops being
reproducible the moment a value made inside it reaches the model: a row's commit
timestamp, a clock, an unordered set. That has twice invalidated a recording that had
already been paid for -- once from the acceptance seed, once from a created tracker row --
and both times the recording was bought first and found unusable afterwards.

ReplayModel cannot answer this. It compares each request against the recorded digest, so
once a tool or a prompt changes it refuses every request, and that refusal does not say
whether the change was ours or the clock's. This drives the recorded answers twice while
ignoring the digests, and compares what we sent. It stays valid against a stale recording,
which is exactly when it is needed: after the code changed and before paying to record it
again.
"""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from captureflow.conversation_router_service import ConversationRouterService
from captureflow.storage import CaptureStore
from scripts.model_replay import LooseReplayModel, first_unreproducible_request
from scripts.run_read_agent_acceptance import (
    CASE_SEQUENCE,
    FIXTURE,
    ToolResultRecorder,
    seed_database,
)

from captureflow.read_agent import AgentFinalResponse, AgentModelTurn, AgentToolCall
from scripts.model_replay import request_digest

from tests.test_replay_acceptance import RECORDING, SEQUENCE


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
        self.sent.append(("next_read_agent_turn", request_digest("next_read_agent_turn", kwargs), kwargs))
        return self.turns.pop(0)


class ReplayDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.recording = json.loads(RECORDING.read_text(encoding="utf-8"))
        self.temporary_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary_directory.cleanup)

    def _one_pass(self, name: str) -> list:
        store = CaptureStore(Path(self.temporary_directory.name) / f"{name}.sqlite3")
        store.initialize()
        seed = seed_database(store)
        model = LooseReplayModel(self.recording)
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
        for case, conversation, message_id in SEQUENCE:
            router.submit(conversations[conversation], inputs[case], message_id)
        return model.sent

    def test_two_identical_runs_send_identical_requests(self) -> None:
        difference = first_unreproducible_request(self._one_pass("first"), self._one_pass("second"))

        self.assertIsNone(difference, difference)

    def test_a_target_write_turn_is_reproducible_before_it_is_ever_recorded(self) -> None:
        # Driven by a script rather than a recording, so the question can be asked of a case
        # that has never been recorded, which is exactly when it needs asking. A target
        # write searches, changes what it found, and answers: three turns' worth of values
        # that could each carry a clock.
        seeded_event = "read-agent-architecture-review"

        def script(seed) -> list:
            return [
                _tools(("search_calendar", {})),
                _tools(("update_event", {"target_id": seeded_event, "date": "2026-09-18", "start_time": "14:00"})),
                _answer("改好了。"),
                _tools(("search_project_records", {})),
                _tools(("transition_record", {"target_id": seed["task"].id, "target_state": "completed"})),
                _answer("标记完成。"),
            ]

        sent = [
            self._scripted_pass(f"target-{attempt}", script, ("event_update", "record_done"))
            for attempt in ("first", "second")
        ]
        difference = first_unreproducible_request(*sent)

        self.assertIsNone(difference, difference)

    def _scripted_pass(self, name: str, script, cases: tuple[str, ...]) -> list:
        store = CaptureStore(Path(self.temporary_directory.name) / f"{name}.sqlite3")
        store.initialize()
        seed = seed_database(store)
        model = _ScriptedTurns(script(seed))
        tools = ToolResultRecorder()
        self.addCleanup(tools.restore)
        router = ConversationRouterService(
            store,
            model,
            date.fromisoformat(self.fixture["protocol"]["reference_date"]),
            self.fixture["protocol"]["timezone"],
        )
        inputs = {case["name"]: case["input"] for case in self.fixture["cases"]}
        wanted = {case: (conversation, message_id) for case, conversation, message_id in CASE_SEQUENCE}
        conversations = {
            "main": seed["main_conversation"].id,
            "later": seed["later_conversation"].id,
        }
        for case in cases:
            conversation, message_id = wanted[case]
            router.submit(conversations[conversation], inputs[case], message_id)
        return model.sent

    def test_the_sequence_the_replay_uses_is_the_one_the_live_run_submits(self) -> None:
        # Two lists of the same turns would drift, and the digests would then refuse a
        # recording for a reason that looks like nondeterminism.
        canonical = {case: (conversation, message_id) for case, conversation, message_id in CASE_SEQUENCE}

        for case, conversation, message_id in SEQUENCE:
            with self.subTest(case=case):
                self.assertEqual(canonical[case], (conversation, message_id))
        self.assertEqual(
            [case for case, _, _ in SEQUENCE],
            [case for case, _, _ in CASE_SEQUENCE if case in {name for name, _, _ in SEQUENCE}],
            "the replay order must follow the live run's order",
        )


if __name__ == "__main__":
    unittest.main()
